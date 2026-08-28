from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from ppi_ddg.data.paired_dataset import (
    PairedDDGDataset,
    collate_paired_ddg_samples,
)
from ppi_ddg.data.splits import (
    _dataset_item,
    audit_fold_group_leakage,
    build_reference_exact_complex_folds,
    make_complex_disjoint_validation_folds,
)
from ppi_ddg.local_geom.cache_audit import audit_paired_ddg_cache
from ppi_ddg.local_geom.interface_context import prepare_collated_graph_cpu
from ppi_ddg.metrics import regression_metrics
from ppi_ddg.train.downstream_protocol import (
    MAX_TRAINABLE_PARAMETERS,
    ScaleOnlyTargetScaler,
    TrainingWeights,
    _autocast_context,
    _move_to_device,
    _resolve_accumulation_steps,
    build_model,
    build_optimizer,
    build_training_weights,
    fit_scale_only_target_scaler,
    load_trainable_state_dict,
    train_one_epoch,
    trainable_state_dict,
    validate_config,
)


METRIC_FIELDS = ("rmse", "mae", "pearson", "spearman")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the paired-state DDG model"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--pretrain-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    return parser


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"config {path} must contain a YAML mapping")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _assert_fresh_output(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    if not run_dir.is_dir():
        raise FileExistsError(f"output path {run_dir} is not a directory")
    unexpected = [
        child.name
        for child in run_dir.iterdir()
        if not (child.name == "train.log" and child.is_file())
    ]
    if unexpected:
        raise FileExistsError(
            "output directory must be fresh; existing artifacts: "
            + ", ".join(sorted(unexpected)[:8])
        )


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def _load_checkpoint(path: Path, device: torch.device) -> Mapping[str, Any]:
    try:
        value = torch.load(
            path, map_location=device, weights_only=False
        )
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, Mapping):
        raise TypeError(f"checkpoint {path} must contain a mapping")
    return value


def _loader_kwargs(training_cfg: Mapping[str, Any]) -> dict[str, Any]:
    workers = int(training_cfg.get("num_workers", 0))
    if workers < 0:
        raise ValueError("training.num_workers must be non-negative")
    kwargs: dict[str, Any] = {
        "num_workers": workers,
        "pin_memory": bool(training_cfg.get("pin_memory", False)),
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(
            training_cfg.get("persistent_workers", False)
        )
        if "prefetch_factor" in training_cfg:
            factor = int(training_cfg["prefetch_factor"])
            if factor <= 0:
                raise ValueError("training.prefetch_factor must be positive")
            kwargs["prefetch_factor"] = factor
    return kwargs


class WeightedSubset:
    """Fold view that attaches immutable per-record sample weights."""

    def __init__(
        self,
        dataset: Any,
        indices: Sequence[int],
        weights: Mapping[int, float],
    ) -> None:
        self.dataset = dataset
        self.dataset_indices = tuple(int(index) for index in indices)
        if len(set(self.dataset_indices)) != len(self.dataset_indices):
            raise ValueError("subset indices must be unique")
        self.weights: dict[int, float] = {}
        for index in self.dataset_indices:
            if index < 0 or index >= len(dataset):
                raise IndexError("subset index lies outside the dataset")
            if index not in weights:
                raise KeyError(f"missing sample weight for index {index}")
            weight = float(weights[index])
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("sample weights must be finite and positive")
            self.weights[index] = weight

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def __getitem__(self, subset_index: int) -> dict[str, Any]:
        dataset_index = self.dataset_indices[subset_index]
        item = self.dataset[dataset_index]
        if not isinstance(item, Mapping):
            raise TypeError("dataset items must be mappings")
        result = dict(item)
        result["sample_weight"] = self.weights[dataset_index]
        return result


def _metadata_column(
    batch: Mapping[str, Any], key: str, size: int
) -> list[Any]:
    value = batch.get(key)
    if value is None:
        return [""] * size
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        result = list(value)
    elif size == 1:
        result = [value]
    else:
        raise ValueError(f"metadata {key} must contain one value per sample")
    if len(result) != size:
        raise ValueError(
            f"metadata {key} has {len(result)} values for {size} samples"
        )
    return result


def collate_training_samples(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch = collate_paired_ddg_samples(items)
    batch_size = int(batch["ddg"].reshape(-1).shape[0])
    weights = torch.as_tensor(
        _metadata_column(batch, "sample_weight", batch_size),
        dtype=torch.float32,
    ).reshape(-1)
    if not bool(torch.isfinite(weights).all()) or not bool((weights > 0).all()):
        raise ValueError("sample weights must be finite and positive")
    batch["sample_weight"] = weights

    mode = _metadata_column(batch, "mode", batch_size)
    direction = _metadata_column(batch, "direction", batch_size)
    normalized_direction: list[str] = []
    for mode_value, direction_value in zip(mode, direction):
        mode_text = str(mode_value or "").strip()
        direction_text = str(direction_value or "").strip()
        if mode_text and direction_text and mode_text != direction_text:
            raise ValueError("mode and direction metadata disagree")
        normalized_direction.append(direction_text or mode_text)
    batch["direction"] = normalized_direction
    batch["wt_graph"] = prepare_collated_graph_cpu(batch["wt_graph"])
    batch["mt_graph"] = prepare_collated_graph_cpu(batch["mt_graph"])
    return batch


def _unit_weights(indices: Iterable[int]) -> dict[int, float]:
    return {int(index): 1.0 for index in indices}


def _build_folds(
    dataset: Any, split_cfg: Mapping[str, Any]
) -> list[dict[str, list[int]]]:
    source_folds = build_reference_exact_complex_folds(
        dataset,
        num_folds=int(split_cfg["num_folds"]),
        seed=int(split_cfg["seed"]),
        reference_records_csv=split_cfg.get("reference_records_csv"),
        reference_group_column=str(
            split_cfg.get("reference_group_column", "pdb_id")
        ),
        reference_source_row_field=str(
            split_cfg.get("reference_source_row_field", "source_row")
        ),
        reference_source_row_base=int(
            split_cfg.get("reference_source_row_base", 1)
        ),
    )
    return make_complex_disjoint_validation_folds(
        dataset,
        source_folds,
        seed=int(split_cfg["seed"]),
        val_fraction=float(split_cfg.get("val_fraction", 0.1)),
    )


def _record_metadata(dataset: Any, index: int) -> dict[str, Any]:
    item = _dataset_item(dataset, index)
    return {
        "dataset_index": index,
        "row_id": str(item.get("row_id", "")),
        "sample_id": str(item.get("sample_id", item.get("row_id", ""))),
        "pdb_id": str(item.get("pdb_id", item.get("complex_id", ""))),
        "mutation": str(item.get("mutation", "")),
    }


def _write_split_artifacts(
    run_dir: Path,
    dataset: Any,
    folds: list[dict[str, list[int]]],
    split_audit: Mapping[str, Any],
) -> None:
    rows: list[dict[str, Any]] = []
    for fold_index, fold in enumerate(folds):
        for split in ("train", "val", "test"):
            for index in fold[split]:
                rows.append(
                    {
                        "fold": fold_index,
                        "split": split,
                        **_record_metadata(dataset, index),
                    }
                )
    _write_csv(
        run_dir / "split_manifest.csv",
        rows,
        (
            "fold",
            "split",
            "dataset_index",
            "row_id",
            "sample_id",
            "pdb_id",
            "mutation",
        ),
    )
    _save_json(
        run_dir / "split_summary.json",
        {
            "dataset_rows": len(dataset),
            "num_folds": len(folds),
            "num_unassigned_test_groups": split_audit[
                "num_unassigned_test_groups"
            ],
            "unassigned_test_groups": split_audit[
                "unassigned_test_groups"
            ],
        },
    )


def _training_weight_payload(weights: TrainingWeights) -> dict[str, Any]:
    return {
        "enabled": weights.enabled,
        "rows": len(weights.by_index),
        "mean": weights.mean,
        "minimum": weights.minimum,
        "maximum": weights.maximum,
        "complex_counts": weights.complex_counts,
        "mutation_count_counts": {
            str(key): value
            for key, value in weights.mutation_count_counts.items()
        },
    }


def _protocol_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    training_cfg = config["training"]
    return {
        "architecture": "paired_state_ddg",
        "split_unit": "complex",
        "split_style": "reference_exact",
        "split_seed": int(config["splitting"]["seed"]),
        "fold_training_seed": "training.seed + fold_index",
        "validation_mode": "complex_disjoint",
        "target_scaling": "std_only",
        "physical_batch_size": int(training_cfg["batch_size"]),
        "effective_batch_size": int(training_cfg["effective_batch_size"]),
        "gradient_accumulation_steps": _resolve_accumulation_steps(
            training_cfg
        ),
        "amp": bool(training_cfg.get("amp", False)),
        "pretrain_checkpoint": str(config["pretrain"]["checkpoint"]),
        "sample_weighting": dict(training_cfg["sample_weighting"]),
        "energy_components": ["additive", "set", "global"],
        "trainable_parameter_budget": MAX_TRAINABLE_PARAMETERS,
    }


def _evaluate(
    model: torch.nn.Module,
    loader: Iterable[dict[str, Any]],
    device: torch.device,
    *,
    target_scaler: ScaleOnlyTargetScaler,
    amp: bool,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    amp_enabled = bool(amp) and device.type == "cuda"
    true_values: list[float] = []
    predicted_values: list[float] = []
    rows: list[dict[str, Any]] = []
    metadata_keys = (
        "sample_id",
        "row_id",
        "pdb_id",
        "mutation",
        "direction",
        "construction_method",
        "source_row",
    )
    with torch.no_grad():
        for raw_batch in loader:
            batch = _move_to_device(raw_batch, device)
            with _autocast_context(amp_enabled):
                scaled = model(batch["wt_graph"], batch["mt_graph"])
            prediction = target_scaler.inverse_transform(scaled.float())
            target = batch["ddg"].float()
            batch_true = target.detach().cpu().tolist()
            batch_pred = prediction.detach().cpu().tolist()
            true_values.extend(batch_true)
            predicted_values.extend(batch_pred)
            metadata = {
                key: _metadata_column(raw_batch, key, len(batch_true))
                for key in metadata_keys
            }
            for row_index, (truth, estimate) in enumerate(
                zip(batch_true, batch_pred)
            ):
                rows.append(
                    {
                        **{
                            key: metadata[key][row_index]
                            for key in metadata_keys
                        },
                        "ddg_true": truth,
                        "ddg_pred": estimate,
                    }
                )
    return (
        regression_metrics(
            np.asarray(true_values), np.asarray(predicted_values)
        ),
        rows,
    )


def _mutation_count(value: Any) -> int:
    return sum(bool(part.strip()) for part in str(value or "").split(","))


def _metric_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = regression_metrics(
        np.asarray([float(row["ddg_true"]) for row in rows]),
        np.asarray([float(row["ddg_pred"]) for row in rows]),
    )
    return {"rows": len(rows), **metrics}


def _subgroup_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    output = [
        {"group_type": "all", "group_value": "all", **_metric_payload(rows)}
    ]
    for key in ("direction", "construction_method"):
        values = sorted(
            {
                str(row.get(key, "")).strip()
                for row in rows
                if str(row.get(key, "")).strip()
            }
        )
        for value in values:
            selected = [
                row for row in rows if str(row.get(key, "")).strip() == value
            ]
            output.append(
                {
                    "group_type": key,
                    "group_value": value,
                    **_metric_payload(selected),
                }
            )
    for count in sorted({_mutation_count(row.get("mutation")) for row in rows}):
        selected = [
            row
            for row in rows
            if _mutation_count(row.get("mutation")) == count
        ]
        output.append(
            {
                "group_type": "mutation_count",
                "group_value": count,
                **_metric_payload(selected),
            }
        )
    return output


def _is_better(
    metrics: Mapping[str, float],
    best: Mapping[str, float] | None,
    min_delta: float,
) -> bool:
    if best is None:
        return True
    pearson = float(metrics.get("pearson", float("nan")))
    best_pearson = float(best.get("pearson", float("nan")))
    pearson = float("-inf") if math.isnan(pearson) else pearson
    best_pearson = float("-inf") if math.isnan(best_pearson) else best_pearson
    if pearson > best_pearson + min_delta:
        return True
    if abs(pearson - best_pearson) <= min_delta:
        rmse = float(metrics.get("rmse", float("inf")))
        best_rmse = float(best.get("rmse", float("inf")))
        return rmse < best_rmse
    return False


def _prefixed(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _prediction_fields(rows: list[dict[str, Any]]) -> list[str]:
    fields = [
        "sample_id",
        "row_id",
        "pdb_id",
        "mutation",
        "ddg_true",
        "ddg_pred",
    ]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    return fields


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _load_config(args.config)
    if args.dataset:
        config.setdefault("data", {})["dataset"] = args.dataset
    if args.pretrain_checkpoint:
        config.setdefault("pretrain", {})["checkpoint"] = (
            args.pretrain_checkpoint
        )
    validate_config(config)
    data_cfg = config["data"]
    split_cfg = config["splitting"]
    training_cfg = config["training"]
    seed = int(training_cfg.get("seed", split_cfg["seed"]))
    _set_seed(seed)
    device = _resolve_device(args.device)
    run_dir = Path(config["output"]["run_dir"])
    _assert_fresh_output(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    max_residues_value = data_cfg.get("max_residues")
    max_residues = (
        None if max_residues_value is None else int(max_residues_value)
    )
    cache_dir = Path(data_cfg["cache_dir"]) / str(data_cfg["dataset"])
    dataset = PairedDDGDataset(
        cache_dir,
        max_residues=max_residues,
        local_radius=float(data_cfg.get("local_radius", 12.0)),
    )
    if len(dataset) == 0:
        raise FileNotFoundError(f"no paired graphs found in {cache_dir}")
    print(
        f"Preparing dataset={data_cfg['dataset']} rows={len(dataset)} "
        f"cache={cache_dir}",
        flush=True,
    )
    if bool(data_cfg.get("audit_cache", False)):
        print("Running paired-cache audit...", flush=True)
        audit = audit_paired_ddg_cache(
            dataset,
            max_examples=int(data_cfg.get("audit_max_examples", 8)),
            strict=False,
        )
        _save_json(run_dir / "cache_audit.json", audit)
        if bool(data_cfg.get("strict_cache_audit", True)) and int(
            audit.get("num_failures", 0)
        ):
            raise ValueError(
                f"cache audit failed for {audit['num_failures']} samples"
            )
        print("Paired-cache audit passed.", flush=True)

    print("Building reference folds and disjoint validation...", flush=True)
    folds = _build_folds(dataset, split_cfg)
    split_audit = audit_fold_group_leakage(dataset, folds)
    _save_json(run_dir / "split_audit.json", split_audit)
    if split_audit.get("passed") is not True:
        raise ValueError("complex split leakage or repeated test assignment detected")
    if bool(split_cfg.get("write_manifest", False)):
        _write_split_artifacts(run_dir, dataset, folds, split_audit)
    _save_json(run_dir / "protocol.json", _protocol_payload(config))
    print(
        f"Split audit passed; unassigned_test_groups="
        f"{split_audit['num_unassigned_test_groups']}",
        flush=True,
    )

    physical_batch = int(training_cfg["batch_size"])
    loader_kwargs = _loader_kwargs(training_cfg)
    grad_clip_value = training_cfg.get("grad_clip_norm")
    grad_clip = None if grad_clip_value is None else float(grad_clip_value)
    epochs = int(training_cfg.get("epochs", 100))
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    patience = int(early_cfg.get("patience", 15))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    amp = bool(training_cfg.get("amp", False))
    print(
        f"Training {len(folds)} folds on {device}; "
        f"physical_batch={physical_batch} "
        f"effective_batch={training_cfg['effective_batch_size']} "
        f"run_dir={run_dir}",
        flush=True,
    )

    summary_rows: list[dict[str, Any]] = []
    all_test_rows: list[dict[str, Any]] = []
    subgroup_rows: list[dict[str, Any]] = []
    summary_fields = [
        "fold",
        "num_train",
        "num_val",
        "num_test",
        "best_epoch",
        *[f"val_{metric}" for metric in METRIC_FIELDS],
        *[f"test_{metric}" for metric in METRIC_FIELDS],
    ]
    for fold_index, fold in enumerate(folds):
        _set_seed(seed + fold_index)
        print(
            f"fold={fold_index} status=starting "
            f"train={len(fold['train'])} val={len(fold['val'])} "
            f"test={len(fold['test'])}",
            flush=True,
        )
        fold_dir = run_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        training_weights = build_training_weights(
            dataset, fold["train"], training_cfg["sample_weighting"]
        )
        weight_payload = _training_weight_payload(training_weights)
        _save_json(
            fold_dir / "training_weight_statistics.json", weight_payload
        )
        target_scaler = fit_scale_only_target_scaler(
            dataset, fold["train"], enabled=True
        )
        model = build_model(config, device)
        optimizer = build_optimizer(model, training_cfg)
        train_loader = DataLoader(
            WeightedSubset(dataset, fold["train"], training_weights.by_index),
            batch_size=physical_batch,
            shuffle=True,
            collate_fn=collate_training_samples,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            WeightedSubset(dataset, fold["val"], _unit_weights(fold["val"])),
            batch_size=physical_batch,
            shuffle=False,
            collate_fn=collate_training_samples,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            WeightedSubset(dataset, fold["test"], _unit_weights(fold["test"])),
            batch_size=physical_batch,
            shuffle=False,
            collate_fn=collate_training_samples,
            **loader_kwargs,
        )
        best_epoch = 0
        best_metrics: dict[str, float] | None = None
        without_improvement = 0
        curve_rows: list[dict[str, Any]] = []
        best_path = fold_dir / "best_trainable_state.pt"
        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                target_scaler=target_scaler,
                training_cfg=training_cfg,
                grad_clip_norm=grad_clip,
                progress_callback=lambda done, total, running, fi=fold_index, ep=epoch: print(
                    f"fold={fi} epoch={ep} batch={done}/{total} "
                    f"running_loss={running:.4f}",
                    flush=True,
                ),
            )
            val_metrics, _ = _evaluate(
                model,
                val_loader,
                device,
                target_scaler=target_scaler,
                amp=amp,
            )
            curve_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    **_prefixed("val", val_metrics),
                }
            )
            if _is_better(val_metrics, best_metrics, min_delta):
                best_epoch = epoch
                best_metrics = dict(val_metrics)
                without_improvement = 0
                torch.save(
                    {
                        "trainable_state_dict": trainable_state_dict(model),
                        "config": config,
                        "fold": fold_index,
                        "epoch": epoch,
                        "val_metrics": val_metrics,
                        "train_loss": train_loss,
                        "target_scaler": {
                            "enabled": True,
                            "mean": 0.0,
                            "std": target_scaler.std,
                        },
                        "training_weight_statistics": weight_payload,
                    },
                    best_path,
                )
            else:
                without_improvement += 1
            if epoch == epochs or epoch % int(
                training_cfg.get("log_every", 10)
            ) == 0:
                print(
                    f"fold={fold_index} epoch={epoch} "
                    f"train_loss={train_loss:.4f} "
                    f"val_pearson={val_metrics['pearson']:.4f} "
                    f"val_rmse={val_metrics['rmse']:.4f}",
                    flush=True,
                )
            if early_enabled and without_improvement >= patience:
                print(
                    f"fold={fold_index} early stopping at epoch={epoch}; "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )
                break
        _write_csv(
            fold_dir / "train_curve.csv",
            curve_rows,
            ["epoch", "train_loss", *[f"val_{m}" for m in METRIC_FIELDS]],
        )
        if best_metrics is None or not best_path.is_file():
            raise RuntimeError(f"fold {fold_index} did not produce a checkpoint")
        checkpoint = _load_checkpoint(best_path, device)
        state = checkpoint.get("trainable_state_dict")
        if not isinstance(state, Mapping):
            raise KeyError("fold checkpoint is missing trainable_state_dict")
        load_trainable_state_dict(model, state)
        test_metrics, test_rows = _evaluate(
            model,
            test_loader,
            device,
            target_scaler=target_scaler,
            amp=amp,
        )
        for row in test_rows:
            row["fold"] = fold_index
        _write_csv(
            fold_dir / "test_predictions.csv",
            test_rows,
            _prediction_fields(test_rows),
        )
        for row in _subgroup_rows(test_rows):
            row["fold"] = fold_index
            subgroup_rows.append(row)
        all_test_rows.extend(test_rows)
        summary_rows.append(
            {
                "fold": fold_index,
                "num_train": len(fold["train"]),
                "num_val": len(fold["val"]),
                "num_test": len(fold["test"]),
                "best_epoch": best_epoch,
                **_prefixed("val", best_metrics),
                **_prefixed("test", test_metrics),
            }
        )
        print(
            f"fold={fold_index} best_epoch={best_epoch} "
            f"test_pearson={test_metrics['pearson']:.4f} "
            f"test_rmse={test_metrics['rmse']:.4f}",
            flush=True,
        )
        del model, optimizer, train_loader, val_loader, test_loader
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_row: dict[str, Any] = {
        "fold": "mean",
        "num_train": "",
        "num_val": "",
        "num_test": "",
        "best_epoch": "",
    }
    for prefix in ("val", "test"):
        for metric in METRIC_FIELDS:
            key = f"{prefix}_{metric}"
            values = [
                float(row[key])
                for row in summary_rows
                if not math.isnan(float(row[key]))
            ]
            mean_row[key] = float(np.mean(values)) if values else float("nan")
    summary_rows.append(mean_row)
    _write_csv(run_dir / "summary.csv", summary_rows, summary_fields)
    pooled = regression_metrics(
        np.asarray([float(row["ddg_true"]) for row in all_test_rows]),
        np.asarray([float(row["ddg_pred"]) for row in all_test_rows]),
    )
    _save_json(
        run_dir / "pooled_metrics.json",
        {"folds": list(range(len(folds))), "rows": len(all_test_rows), **pooled},
    )
    _write_csv(
        run_dir / "subgroup_metrics.csv",
        subgroup_rows,
        (
            "fold",
            "group_type",
            "group_value",
            "rows",
            "pearson",
            "spearman",
            "rmse",
            "mae",
        ),
    )
    print(
        f"mean test_pearson={mean_row['test_pearson']:.4f} "
        f"test_spearman={mean_row['test_spearman']:.4f} "
        f"test_rmse={mean_row['test_rmse']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
