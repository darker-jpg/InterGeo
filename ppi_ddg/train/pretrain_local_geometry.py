from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
import yaml

from ppi_ddg.data.datasets import CachedPretrainDataset, collate_pretrain_graphs
from ppi_ddg.local_geom.monitoring import (
    AssignmentAccumulator,
    codebook_health,
    deterministic_branch_masks,
    shuffle_quantized_codes,
    split_pretrain_indices,
)
from ppi_ddg.local_geom.pretrainer import LocalGeometricVQPretrainer

BRANCHES = ("type", "angle", "geometry", "rsa")
CODEBOOK_BRANCHES = ("type", "angle", "geometry")
CA_INDEX = 1
VALIDATION_MODES = {"holdout", "train_monitor"}
GEOMETRY_COMPONENT_COUNT_KEYS = {
    "loss_local": "local_count",
    "loss_structure_distance": "structure_distance_count",
    "loss_interface_distance": "interface_distance_count",
    "loss_structure_direction": "structure_direction_count",
    "loss_interface_direction": "interface_direction_count",
}


@dataclass(frozen=True)
class ValidationEpochResult:
    metrics: dict[str, float]
    codebook_statistics: dict[str, Any]
    codebook_health_passed: bool
    true_geometry_loss: float
    shuffled_geometry_loss: float


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain local-geometry vector-quantized representations"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="Training device, e.g. cuda:0 or cpu")
    return parser


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise TypeError(f"config {path} must contain a YAML mapping")
    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _weighted_metric(logs: list[dict[str, float]], value_key: str, count_key: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for log in logs:
        count = float(log.get(count_key, 0.0))
        numerator += float(log.get(value_key, 0.0)) * count
        denominator += count
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _aggregate_geometry_breakdowns(logs: list[dict[str, float]]) -> dict[str, float]:
    metrics = {
        count_key: sum(float(log.get(count_key, 0.0)) for log in logs)
        for count_key in GEOMETRY_COMPONENT_COUNT_KEYS.values()
    }
    for loss_key, count_key in GEOMETRY_COMPONENT_COUNT_KEYS.items():
        metrics[loss_key] = _weighted_metric(logs, loss_key, count_key)
    metrics["loss_distance"] = metrics["loss_structure_distance"] + metrics["loss_interface_distance"]
    metrics["loss_direction"] = metrics["loss_structure_direction"] + metrics["loss_interface_direction"]
    metrics["loss_geometry"] = metrics["loss_local"] + 0.5 * metrics["loss_distance"] + 0.5 * metrics["loss_direction"]
    return metrics


def aggregate_epoch_metrics(model: LocalGeometricVQPretrainer, logs: list[dict[str, float]]) -> dict[str, float]:
    if not logs:
        return {}

    metrics = _aggregate_geometry_breakdowns(logs)
    count_keys = {
        "type_count",
        "angle_count",
        "rsa_count",
        "vq_type_candidate_count",
        "vq_angle_candidate_count",
        "vq_geometry_candidate_count",
        "mask_type_count",
        "mask_angle_count",
        "mask_geometry_count",
        "mask_rsa_count",
        "angle_supervised_count",
        "local_supervised_count",
        "distance_supervised_count",
        "direction_supervised_count",
        "rsa_supervised_count",
        "vq_candidate_count",
    }
    for key in count_keys:
        metrics[key] = sum(float(log.get(key, 0.0)) for log in logs)

    metrics["loss_type"] = _weighted_metric(logs, "loss_type", "type_count")
    metrics["loss_angle"] = _weighted_metric(logs, "loss_angle", "angle_count")
    metrics["loss_rsa"] = _weighted_metric(logs, "loss_rsa", "rsa_count")
    metrics["loss_vq_type"] = _weighted_metric(logs, "loss_vq_type", "vq_type_candidate_count")
    metrics["loss_vq_angle"] = _weighted_metric(logs, "loss_vq_angle", "vq_angle_candidate_count")
    metrics["loss_vq_geometry"] = _weighted_metric(logs, "loss_vq_geometry", "vq_geometry_candidate_count")
    metrics["loss_vq_total"] = metrics["loss_vq_type"] + metrics["loss_vq_angle"] + metrics["loss_vq_geometry"]
    metrics["total_loss"] = (
        model.lambda_type * metrics["loss_type"]
        + model.lambda_angle * metrics["loss_angle"]
        + model.lambda_geometry * metrics["loss_geometry"]
        + model.lambda_rsa * metrics["loss_rsa"]
        + model.lambda_vq * metrics["loss_vq_total"]
    )
    return metrics


def _candidate_mask(batch: dict[str, Any]) -> torch.Tensor:
    pos_atoms = batch["pos_atoms"]
    device = pos_atoms.device
    shape = pos_atoms.shape[:2]

    node_mask = batch.get("node_mask")
    if node_mask is None:
        node_mask = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        node_mask = node_mask.to(device=device, dtype=torch.bool)

    interface_mask = batch.get("interface_mask")
    if interface_mask is None:
        interface_mask = node_mask
    else:
        interface_mask = interface_mask.to(device=device, dtype=torch.bool)

    mask_atoms = batch.get("mask_atoms")
    if mask_atoms is None:
        ca_mask = torch.ones(shape, dtype=torch.bool, device=device)
    else:
        ca_mask = mask_atoms.to(device=device, dtype=torch.bool)[:, :, CA_INDEX]
    return node_mask & interface_mask & ca_mask


def random_branch_masks(batch: dict[str, Any], mask_ratio: float) -> dict[str, torch.Tensor]:
    candidate = _candidate_mask(batch)
    return {
        branch: candidate & (torch.rand(candidate.shape, device=candidate.device) < mask_ratio)
        for branch in BRANCHES
    }


def validation_branch_masks(batch: dict[str, Any], mask_ratio: float, seed: int) -> dict[str, torch.Tensor]:
    return deterministic_branch_masks(
        sample_ids=batch["sample_id"],
        candidate_mask=_candidate_mask(batch),
        mask_ratio=mask_ratio,
        seed=seed,
        branches=BRANCHES,
    )


def _read_dataset_sample_ids(dataset: CachedPretrainDataset) -> list[str]:
    sample_ids: list[str] = []
    for path in dataset.paths:
        payload = torch.load(path, map_location="cpu")
        sample_id = payload.get("sample_id")
        sample_ids.append(str(path.stem if sample_id is None else sample_id))
    return sample_ids


def _normalise_validation_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("mode", "holdout")).lower()
    if mode not in VALIDATION_MODES:
        raise ValueError(f"Unsupported validation mode: {config.get('mode')}")
    return mode


def _build_pretrain_split(
    *,
    sample_ids: list[str],
    val_fraction: float,
    seed: int,
    validation_mode: str,
) -> dict[str, Any]:
    if validation_mode == "train_monitor":
        indices = list(range(len(sample_ids)))
        return {
            "validation_mode": validation_mode,
            "heldout_validation": False,
            "train_indices": indices,
            "val_indices": indices,
            "train_sample_ids": list(sample_ids),
            "val_sample_ids": list(sample_ids),
        }

    train_indices, val_indices = split_pretrain_indices(
        sample_ids,
        val_fraction=val_fraction,
        seed=seed,
    )
    return {
        "validation_mode": validation_mode,
        "heldout_validation": True,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_sample_ids": [sample_ids[index] for index in train_indices],
        "val_sample_ids": [sample_ids[index] for index in val_indices],
    }


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    name = str(config.get("name", "none")).lower()
    if name in {"none", "false", "off", ""}:
        return None
    if name != "reduce_on_plateau":
        raise ValueError(f"Unsupported scheduler: {config.get('name')}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config.get("factor", 0.5)),
        patience=int(config.get("patience", 10)),
        threshold=float(config.get("threshold", 1e-3)),
        threshold_mode=str(config.get("threshold_mode", "rel")),
        cooldown=int(config.get("cooldown", 0)),
        min_lr=float(config.get("min_lr", 1e-5)),
    )


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _checkpoint_payload(
    *,
    model: LocalGeometricVQPretrainer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    split: dict[str, Any],
    train_metrics: dict[str, float],
    validation: ValidationEpochResult,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "split": split,
        "train_metrics": train_metrics,
        "val_metrics": validation.metrics,
        "codebook_statistics": validation.codebook_statistics,
        "codebook_health_passed": validation.codebook_health_passed,
        "true_geometry_loss": validation.true_geometry_loss,
        "shuffled_geometry_loss": validation.shuffled_geometry_loss,
    }


def _metric_value(metrics: dict[str, float], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    return float(value)


def _format_epoch_progress(
    *,
    epoch: int,
    epochs: int,
    global_step: int,
    lr: float,
    train_metrics: dict[str, float],
    validation: ValidationEpochResult,
    saved_best_this_epoch: bool,
) -> str:
    val_metrics = validation.metrics
    return (
        f"epoch={epoch}/{epochs} "
        f"step={global_step} "
        f"lr={lr:.6g} "
        f"train_loss={_metric_value(train_metrics, 'total_loss'):.4f} "
        f"val_loss={_metric_value(val_metrics, 'total_loss'):.4f} "
        f"type_active={_metric_value(val_metrics, 'type_active_codes'):.0f} "
        f"type_ppl={_metric_value(val_metrics, 'type_perplexity'):.2f} "
        f"angle_active={_metric_value(val_metrics, 'angle_active_codes'):.0f} "
        f"angle_ppl={_metric_value(val_metrics, 'angle_perplexity'):.2f} "
        f"geometry_active={_metric_value(val_metrics, 'geometry_active_codes'):.0f} "
        f"geometry_ppl={_metric_value(val_metrics, 'geometry_perplexity'):.2f} "
        f"geom_bypass={int(bool(_metric_value(val_metrics, 'geometry_bypass_passed')))} "
        f"healthy={int(validation.codebook_health_passed)} "
        f"best={int(saved_best_this_epoch)}"
    )


def _stats_to_dict(stats: Any) -> dict[str, Any]:
    if hasattr(stats, "to_dict"):
        return stats.to_dict()
    return dict(stats)


def run_training_epoch(
    *,
    model: LocalGeometricVQPretrainer,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_ratio: float,
    grad_clip_norm: float | None,
    grad_accum_steps: int,
    max_steps: int | None,
    global_step: int,
    epoch: int,
) -> tuple[dict[str, float], int, bool]:
    model.train()
    epoch_logs: list[dict[str, float]] = []
    reached_max_steps = False
    grad_accum_steps = max(1, int(grad_accum_steps))
    accumulated_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch in loader:
        if max_steps is not None and global_step >= max_steps:
            reached_max_steps = True
            break

        batch = move_batch_to_device(batch, device)
        masks = random_branch_masks(batch, mask_ratio=mask_ratio)

        loss, logs, _assignments = model(batch, masks=masks)
        (loss / float(grad_accum_steps)).backward()
        accumulated_batches += 1
        epoch_logs.append(logs)

        if accumulated_batches % grad_accum_steps == 0:
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

    if accumulated_batches > 0 and accumulated_batches % grad_accum_steps != 0 and not reached_max_steps:
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    if max_steps is not None and global_step >= max_steps:
        reached_max_steps = True
    return aggregate_epoch_metrics(model, epoch_logs), global_step, reached_max_steps


def run_validation_epoch(
    *,
    model: LocalGeometricVQPretrainer,
    loader: DataLoader,
    device: torch.device,
    mask_ratio: float,
    seed: int,
    epoch: int,
) -> ValidationEpochResult:
    model.eval()
    metric_logs: list[dict[str, float]] = []
    true_geometry_breakdowns: list[dict[str, float]] = []
    shuffled_geometry_breakdowns: list[dict[str, float]] = []
    accumulators = {
        "type": AssignmentAccumulator(model.vq_type.num_embeddings),
        "angle": AssignmentAccumulator(model.vq_angle.num_embeddings),
        "geometry": AssignmentAccumulator(model.vq_geometry.num_embeddings),
    }

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            batch = move_batch_to_device(batch, device)
            masks = validation_branch_masks(batch, mask_ratio=mask_ratio, seed=seed)
            generator_seed = seed + epoch * 1000 + batch_index
            loss, logs, assignments = model(
                batch,
                masks=masks,
                generator=torch.Generator(device="cpu").manual_seed(generator_seed),
            )
            del loss
            metric_logs.append(logs)

            for branch, accumulator in accumulators.items():
                indices, valid_mask = assignments[branch]
                accumulator.update(indices, valid_mask)

            true_geometry_breakdown = model.geometry_loss_breakdown_with_quantized(
                batch,
                q_geometry=model.last_quantized["geometry"],
                geometry_mask=masks["geometry"],
                encoder_h=model.last_encoder_h,
            )
            shuffled_q = shuffle_quantized_codes(
                model.last_quantized["geometry"],
                assignments["geometry"][1],
                torch.Generator(device="cpu").manual_seed(generator_seed),
            )
            shuffled_geometry_breakdown = model.geometry_loss_breakdown_with_quantized(
                batch,
                q_geometry=shuffled_q,
                geometry_mask=masks["geometry"],
                encoder_h=model.last_encoder_h,
            )
            true_geometry_breakdowns.append(true_geometry_breakdown)
            shuffled_geometry_breakdowns.append(shuffled_geometry_breakdown)

    metrics = aggregate_epoch_metrics(model, metric_logs)
    true_geometry_metrics = _aggregate_geometry_breakdowns(true_geometry_breakdowns)
    shuffled_geometry_metrics = _aggregate_geometry_breakdowns(shuffled_geometry_breakdowns)
    true_geometry_loss = true_geometry_metrics["loss_geometry"]
    shuffled_geometry_loss = shuffled_geometry_metrics["loss_geometry"]

    codebook_statistics = {branch: _stats_to_dict(accumulator.stats()) for branch, accumulator in accumulators.items()}
    branch_health = {branch: codebook_health(stats) for branch, stats in codebook_statistics.items()}
    geometry_bypass_passed = shuffled_geometry_loss > true_geometry_loss
    combined_health = all(branch_health.values()) and geometry_bypass_passed

    for branch, stats in codebook_statistics.items():
        metrics[f"{branch}_active_codes"] = float(stats["active_codes"])
        metrics[f"{branch}_perplexity"] = float(stats["perplexity"])
        metrics[f"{branch}_max_share"] = float(stats["max_share"])
        metrics[f"{branch}_codebook_healthy"] = float(branch_health[branch])
    metrics["true_geometry_loss"] = true_geometry_loss
    metrics["shuffled_geometry_loss"] = shuffled_geometry_loss
    metrics["geometry_bypass_passed"] = float(geometry_bypass_passed)
    metrics["combined_codebook_healthy"] = float(combined_health)

    codebook_statistics["branch_health"] = branch_health
    codebook_statistics["geometry_bypass_passed"] = geometry_bypass_passed
    return ValidationEpochResult(
        metrics=metrics,
        codebook_statistics=codebook_statistics,
        codebook_health_passed=combined_health,
        true_geometry_loss=true_geometry_loss,
        shuffled_geometry_loss=shuffled_geometry_loss,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    loss_cfg = config.get("loss", {})
    validation_cfg = config.get("validation", {})
    logging_cfg = config.get("logging", {})
    output_cfg = config.get("output", {})

    seed = int(training_cfg.get("seed", 2026))
    _set_seed(seed)

    dataset = CachedPretrainDataset(
        data_cfg["cache_dir"],
        interface_only=bool(data_cfg.get("interface_only", True)),
        max_residues=None if data_cfg.get("max_residues") is None else int(data_cfg["max_residues"]),
    )
    if len(dataset) == 0:
        print(f"No cached pretraining graphs found in {data_cfg['cache_dir']}")
        return 0

    sample_ids = _read_dataset_sample_ids(dataset)
    try:
        validation_mode = _normalise_validation_mode(validation_cfg)
    except ValueError as exc:
        print(str(exc))
        return 1

    split = _build_pretrain_split(
        sample_ids=sample_ids,
        val_fraction=float(data_cfg.get("val_fraction", 0.1)),
        seed=seed,
        validation_mode=validation_mode,
    )
    train_indices = split["train_indices"]
    val_indices = split["val_indices"]
    if not train_indices or not val_indices:
        print("Local geometry pretraining requires at least one train and one validation sample after splitting.")
        return 1

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    num_workers = int(training_cfg.get("num_workers", 0))
    batch_size = int(training_cfg.get("batch_size", 16))
    eval_batch_size = int(training_cfg.get("eval_batch_size", batch_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_pretrain_graphs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pretrain_graphs,
    )

    device = _resolve_device(args.device)
    model = LocalGeometricVQPretrainer(
        node_dim=int(model_cfg.get("node_dim", 256)),
        num_blocks=int(model_cfg.get("num_blocks", 4)),
        num_heads=int(model_cfg.get("num_heads", 4)),
        structure_cutoff=float(model_cfg.get("structure_cutoff", 12.0)),
        interface_cutoff=float(model_cfg.get("interface_cutoff", 10.0)),
        structure_k=int(model_cfg.get("structure_k", 16)),
        interface_k=int(model_cfg.get("interface_k", 16)),
        codebook_size=int(model_cfg.get("codebook_size", 256)),
        decoder_layers=int(model_cfg.get("decoder_layers", 2)),
        lambda_type=float(loss_cfg.get("lambda_type", 0.3)),
        lambda_angle=float(loss_cfg.get("lambda_angle", 1.0)),
        lambda_geometry=float(loss_cfg.get("lambda_geometry", 1.0)),
        lambda_rsa=float(loss_cfg.get("lambda_rsa", 0.2)),
        lambda_vq=float(loss_cfg.get("lambda_vq", 0.005)),
        commitment_cost=float(loss_cfg.get("beta", loss_cfg.get("commitment_cost", 0.25))),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    scheduler = _build_scheduler(optimizer, config.get("scheduler", {}))
    grad_clip_norm = training_cfg.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    epochs = int(training_cfg.get("epochs", 1000))
    max_steps = training_cfg.get("max_steps")
    max_steps = None if max_steps is None else int(max_steps)
    mask_ratio = float(loss_cfg.get("mask_ratio", 0.1))
    early_cfg = config.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", True))
    early_patience = int(early_cfg.get("patience", 20))
    early_min_delta = float(early_cfg.get("min_delta", 1e-3))
    print_every = max(0, int(logging_cfg.get("print_every", 0)))

    run_dir = Path(
        output_cfg.get("run_dir", "runs/pretrain_local_geometry_vq")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_json(run_dir / "split.json", split)

    history_rows: list[dict[str, Any]] = []
    fieldnames = {"epoch", "global_step", "lr"}
    best_val_loss = float("inf")
    best_healthy_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0
    last_validation: ValidationEpochResult | None = None
    train_metrics: dict[str, float] = {}
    saved_healthy_best = False

    for epoch in range(1, epochs + 1):
        train_metrics, global_step, reached_max_steps = run_training_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_ratio=mask_ratio,
            grad_clip_norm=grad_clip_norm,
            grad_accum_steps=grad_accum_steps,
            max_steps=max_steps,
            global_step=global_step,
            epoch=epoch,
        )
        last_validation = run_validation_epoch(
            model=model,
            loader=val_loader,
            device=device,
            mask_ratio=mask_ratio,
            seed=seed,
            epoch=epoch,
        )

        val_total_loss = float(last_validation.metrics["total_loss"])
        if scheduler is not None:
            scheduler.step(val_total_loss)

        improved_for_early_stop = val_total_loss < (best_val_loss - early_min_delta)
        if improved_for_early_stop:
            best_val_loss = val_total_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        saved_best_this_epoch = False
        if last_validation.codebook_health_passed and val_total_loss < best_healthy_val_loss:
            best_healthy_val_loss = val_total_loss
            torch.save(
                _checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    global_step=global_step,
                    config=config,
                    split=split,
                    train_metrics=train_metrics,
                    validation=last_validation,
                ),
                run_dir / "best.pt",
            )
            saved_healthy_best = True
            saved_best_this_epoch = True

        torch.save(
            _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config,
                split=split,
                train_metrics=train_metrics,
                validation=last_validation,
            ),
            run_dir / "last.pt",
        )

        row: dict[str, Any] = {"epoch": epoch, "global_step": global_step, "lr": _current_lr(optimizer)}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in last_validation.metrics.items()})
        history_rows.append(row)
        fieldnames.update(row.keys())

        if print_every and (epoch % print_every == 0 or saved_best_this_epoch or reached_max_steps):
            print(
                _format_epoch_progress(
                    epoch=epoch,
                    epochs=epochs,
                    global_step=global_step,
                    lr=_current_lr(optimizer),
                    train_metrics=train_metrics,
                    validation=last_validation,
                    saved_best_this_epoch=saved_best_this_epoch,
                ),
                flush=True,
            )

        if reached_max_steps:
            break
        if early_enabled and epochs_without_improvement > early_patience:
            break

    metrics_fields = ["epoch", "global_step", "lr"] + sorted(
        field for field in fieldnames if field not in {"epoch", "global_step", "lr"}
    )
    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metrics_fields)
        writer.writeheader()
        writer.writerows(history_rows)

    if last_validation is not None:
        _save_json(
            run_dir / "codebook_histograms.json",
            {
                branch: last_validation.codebook_statistics[branch]["histogram"]
                for branch in CODEBOOK_BRANCHES
            },
        )

    if not saved_healthy_best:
        print("No healthy best checkpoint was saved; validation never met the codebook and geometry bypass criteria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
