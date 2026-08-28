from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppi_ddg.data.splits import _dataset_item
from ppi_ddg.local_geom.paired_model import PairedStateDDGModel
from ppi_ddg.local_geom.pretrained_encoder import FrozenLocalGeometryEncoder


MAX_TRAINABLE_PARAMETERS = 12_000_000
PUBLIC_PROTOCOL_API = (
    "TrainingWeights",
    "ScaleOnlyTargetScaler",
    "validate_config",
    "build_model",
    "build_optimizer",
    "build_training_weights",
    "train_one_epoch",
    "trainable_state_dict",
    "load_trainable_state_dict",
)

__all__ = [
    "MAX_TRAINABLE_PARAMETERS",
    "PUBLIC_PROTOCOL_API",
    "ScaleOnlyTargetScaler",
    "TrainingWeights",
    "build_model",
    "build_optimizer",
    "build_training_weights",
    "fit_scale_only_target_scaler",
    "load_trainable_state_dict",
    "train_one_epoch",
    "trainable_state_dict",
    "validate_config",
]


@dataclass(frozen=True)
class ScaleOnlyTargetScaler:
    enabled: bool
    mean: float = 0.0
    std: float = 1.0

    @classmethod
    def fit(
        cls,
        values: torch.Tensor | Iterable[float],
        *,
        enabled: bool = True,
    ) -> "ScaleOnlyTargetScaler":
        if not enabled:
            return cls(enabled=False)
        tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
        if tensor.numel() == 0:
            raise ValueError("cannot fit target scaling without training values")
        std = float(tensor.std(unbiased=False).item())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        return cls(enabled=True, mean=0.0, std=std)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return values if not self.enabled else values / float(self.std)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        return values if not self.enabled else values * float(self.std)


@dataclass(frozen=True)
class TrainingWeights:
    by_index: dict[int, float]
    complex_counts: dict[str, int]
    mutation_count_counts: dict[int, int]
    enabled: bool
    mean: float
    minimum: float
    maximum: float


def fit_scale_only_target_scaler(
    dataset: Any,
    train_indices: Iterable[int],
    *,
    enabled: bool = True,
) -> ScaleOnlyTargetScaler:
    values = [
        float(
            torch.as_tensor(
                _dataset_item(dataset, int(index))["ddg"],
                dtype=torch.float32,
            ).reshape(()).item()
        )
        for index in train_indices
    ]
    return ScaleOnlyTargetScaler.fit(values, enabled=enabled)


def _weighting_parameters(
    config: Mapping[str, Any],
) -> tuple[bool, float, float, float, float]:
    enabled = bool(config.get("enabled", True))
    complex_alpha = float(config.get("complex_alpha", 0.5))
    mutation_alpha = float(config.get("mutation_count_alpha", 0.5))
    minimum = float(config.get("min_weight", 0.25))
    maximum = float(config.get("max_weight", 4.0))
    if not np.isfinite(complex_alpha) or complex_alpha < 0.0:
        raise ValueError("complex_alpha must be finite and non-negative")
    if not np.isfinite(mutation_alpha) or mutation_alpha < 0.0:
        raise ValueError(
            "mutation_count_alpha must be finite and non-negative"
        )
    if (
        not np.isfinite(minimum)
        or not np.isfinite(maximum)
        or minimum <= 0.0
        or maximum < minimum
    ):
        raise ValueError("sample weight bounds must be finite and positive")
    if minimum > 1.0 or maximum < 1.0:
        raise ValueError("sample weight bounds must straddle one")
    return enabled, complex_alpha, mutation_alpha, minimum, maximum


def _normalize_with_bounds(
    raw: np.ndarray,
    minimum: float,
    maximum: float,
) -> np.ndarray:
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("raw sample weights must be a non-empty vector")
    if not np.isfinite(raw).all() or np.any(raw <= 0.0):
        raise ValueError("raw sample weights must be finite and positive")
    fixed = np.zeros(raw.shape, dtype=bool)
    result = np.zeros(raw.shape, dtype=np.float64)
    target_sum = float(raw.size)
    for _ in range(raw.size + 1):
        free = ~fixed
        if not free.any():
            break
        remaining = target_sum - float(result[fixed].sum())
        denominator = float(raw[free].sum())
        if denominator <= 0.0:
            raise ValueError("free sample weights must have positive sum")
        proposal = raw[free] * (remaining / denominator)
        low = proposal < minimum
        high = proposal > maximum
        if not low.any() and not high.any():
            result[free] = proposal
            break
        free_indices = np.flatnonzero(free)
        result[free_indices[low]] = minimum
        fixed[free_indices[low]] = True
        result[free_indices[high]] = maximum
        fixed[free_indices[high]] = True
    else:
        raise RuntimeError("sample-weight normalization did not converge")
    if not np.isclose(float(result.mean()), 1.0, atol=1e-10):
        raise RuntimeError("normalized sample weights must have mean one")
    if float(result.min()) < minimum - 1e-10:
        raise RuntimeError("normalized sample weights violated minimum")
    if float(result.max()) > maximum + 1e-10:
        raise RuntimeError("normalized sample weights violated maximum")
    return result


def _record_complex_id(item: Mapping[str, Any]) -> str:
    value = item.get("pdb_id", item.get("complex_id"))
    if value is None or not str(value).strip():
        raise ValueError("training weights require pdb_id or complex_id")
    return str(value).strip()


def _record_mutation_count(item: Mapping[str, Any]) -> int:
    graph = item.get("wt_graph")
    if not isinstance(graph, Mapping):
        raise ValueError("training weights require wt_graph metadata")
    mask = graph.get("mutation_mask")
    if not isinstance(mask, torch.Tensor):
        raise ValueError("training weights require a mutation_mask tensor")
    count = int(mask.to(dtype=torch.bool).sum().item())
    if count <= 0:
        raise ValueError("each training record requires at least one mutation")
    return count


def build_training_weights(
    dataset: Any,
    train_indices: Iterable[int],
    config: Mapping[str, Any],
) -> TrainingWeights:
    indices = [int(index) for index in train_indices]
    if not indices:
        raise ValueError("at least one training index is required")
    if len(set(indices)) != len(indices):
        raise ValueError("training indices must be unique")
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise IndexError("training index lies outside the dataset")
    enabled, complex_alpha, mutation_alpha, minimum, maximum = (
        _weighting_parameters(config)
    )
    complex_ids: list[str] = []
    mutation_counts: list[int] = []
    for index in indices:
        item = _dataset_item(dataset, index)
        complex_ids.append(_record_complex_id(item))
        mutation_counts.append(_record_mutation_count(item))
    complex_frequency = Counter(complex_ids)
    mutation_frequency = Counter(mutation_counts)
    if enabled:
        raw = np.asarray(
            [
                complex_frequency[complex_id] ** (-complex_alpha)
                * mutation_frequency[mutation_count] ** (-mutation_alpha)
                for complex_id, mutation_count in zip(
                    complex_ids, mutation_counts
                )
            ],
            dtype=np.float64,
        )
        normalized = _normalize_with_bounds(raw, minimum, maximum)
    else:
        normalized = np.ones(len(indices), dtype=np.float64)
    by_index = {
        index: float(weight)
        for index, weight in zip(indices, normalized.tolist())
    }
    values = np.asarray(list(by_index.values()), dtype=np.float64)
    return TrainingWeights(
        by_index=by_index,
        complex_counts=dict(sorted(complex_frequency.items())),
        mutation_count_counts=dict(sorted(mutation_frequency.items())),
        enabled=enabled,
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
    )


def _resolve_accumulation_steps(training_cfg: Mapping[str, Any]) -> int:
    physical = int(training_cfg.get("batch_size", 0))
    effective = int(training_cfg.get("effective_batch_size", physical))
    if physical <= 0 or effective <= 0:
        raise ValueError("batch sizes must be positive")
    if effective < physical or effective % physical != 0:
        raise ValueError(
            "effective_batch_size must be divisible by batch_size and no smaller"
        )
    return effective // physical


def validate_config(config: Mapping[str, Any]) -> None:
    model_cfg = config.get("model", {})
    pretrain_cfg = config.get("pretrain", {})
    split_cfg = config.get("splitting", {})
    training_cfg = config.get("training", {})
    loss_cfg = training_cfg.get("loss", {})
    if str(model_cfg.get("head", "")) != "paired_state_ddg":
        raise ValueError("model.head must be paired_state_ddg")
    if pretrain_cfg.get("freeze_encoder") is not True:
        raise ValueError("pretrain.freeze_encoder must be true")
    if bool(training_cfg.get("target_standardization", False)):
        raise ValueError("exact antisymmetry forbids target mean-centering")
    if str(training_cfg.get("target_scaling", "std_only")) != "std_only":
        raise ValueError("training.target_scaling must be std_only")
    if float(loss_cfg.get("rank_weight", 0.0)) != 0.0:
        raise ValueError("rank_weight must be zero")
    if float(loss_cfg.get("reverse_weight", 0.0)) != 0.0:
        raise ValueError("reverse_weight must be zero")
    if str(split_cfg.get("mode", "")) != "complex":
        raise ValueError("splitting.mode must be complex")
    if int(split_cfg.get("num_folds", 0)) != 10:
        raise ValueError("splitting.num_folds must be 10")
    if int(split_cfg.get("seed", -1)) != 2023:
        raise ValueError("splitting.seed must be 2023")
    if str(split_cfg.get("style", "")) != "reference_exact":
        raise ValueError("splitting.style must be reference_exact")
    if str(split_cfg.get("validation_mode", "")) != "complex":
        raise ValueError("validation must be complex-disjoint")
    if int(model_cfg.get("hidden_dim", 0)) != 256:
        raise ValueError("model.hidden_dim must be 256")
    if int(model_cfg.get("downstream_blocks", 0)) != 3:
        raise ValueError("model.downstream_blocks must be 3")
    if int(model_cfg.get("downstream_heads", 0)) != 8:
        raise ValueError("model.downstream_heads must be 8")
    if int(model_cfg.get("set_blocks", 0)) != 2:
        raise ValueError("model.set_blocks must be 2")
    if not np.isclose(float(model_cfg.get("dropout", -1.0)), 0.08):
        raise ValueError("model.dropout must be 0.08")
    enabled, complex_alpha, mutation_alpha, minimum, maximum = (
        _weighting_parameters(training_cfg.get("sample_weighting", {}))
    )
    expected = (True, 0.5, 0.5, 0.25, 4.0)
    actual = (enabled, complex_alpha, mutation_alpha, minimum, maximum)
    if actual != expected:
        raise ValueError("sample weighting parameters do not match the protocol")
    _resolve_accumulation_steps(training_cfg)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def build_model(
    config: Mapping[str, Any],
    device: torch.device,
) -> PairedStateDDGModel:
    validate_config(config)
    model_cfg = config.get("model", {})
    pretrain_cfg = config.get("pretrain", {})
    checkpoint = pretrain_cfg.get("checkpoint")
    if not checkpoint:
        raise ValueError("pretrain.checkpoint is required")
    pretrained_encoder = FrozenLocalGeometryEncoder(
        checkpoint,
        require_healthy_codebooks=bool(
            pretrain_cfg.get("require_healthy_codebooks", True)
        ),
        node_dim=_optional_int(model_cfg.get("node_dim")),
        num_blocks=_optional_int(model_cfg.get("num_blocks")),
        num_heads=_optional_int(model_cfg.get("num_heads")),
        structure_cutoff=_optional_float(model_cfg.get("structure_cutoff")),
        interface_cutoff=_optional_float(model_cfg.get("interface_cutoff")),
        structure_k=_optional_int(model_cfg.get("structure_k")),
        interface_k=_optional_int(model_cfg.get("interface_k")),
        codebook_size=_optional_int(model_cfg.get("codebook_size")),
    )
    model = PairedStateDDGModel(
        pretrained_encoder=pretrained_encoder,
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        downstream_blocks=int(model_cfg["downstream_blocks"]),
        downstream_heads=int(model_cfg["downstream_heads"]),
        set_blocks=int(model_cfg["set_blocks"]),
        structure_cutoff=_optional_float(model_cfg.get("structure_cutoff")),
        interface_cutoff=_optional_float(model_cfg.get("interface_cutoff")),
        structure_k=_optional_int(model_cfg.get("structure_k")),
        interface_k=_optional_int(model_cfg.get("interface_k")),
        pair_rbf_bins=int(model_cfg.get("pair_rbf_bins", 8)),
        partner_rbf_bins=int(model_cfg.get("partner_rbf_bins", 8)),
        relay_rbf_bins=_optional_int(model_cfg.get("relay_rbf_bins")),
        max_partner_tokens=int(model_cfg.get("max_partner_tokens", 128)),
        max_interface_tokens=int(model_cfg.get("max_interface_tokens", 128)),
    ).to(device)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable > MAX_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"trainable model exceeds parameter budget: "
            f"{trainable} > {MAX_TRAINABLE_PARAMETERS}"
        )
    return model


def build_optimizer(
    model: PairedStateDDGModel,
    training_cfg: Mapping[str, Any],
) -> torch.optim.Optimizer:
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    encoder_ids = {
        id(parameter) for parameter in model.pretrained_encoder.parameters()
    }
    if any(id(parameter) in encoder_ids for parameter in trainable):
        raise ValueError("optimizer includes frozen encoder parameters")
    if not trainable:
        raise ValueError("model has no trainable parameters")
    if sum(parameter.numel() for parameter in trainable) > MAX_TRAINABLE_PARAMETERS:
        raise ValueError("optimizer exceeds trainable parameter budget")
    return torch.optim.AdamW(
        trainable,
        lr=float(training_cfg.get("lr", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )


def trainable_state_dict(model: PairedStateDDGModel) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("pretrained_encoder.")
    }


def load_trainable_state_dict(
    model: PairedStateDDGModel,
    state: Mapping[str, torch.Tensor],
) -> None:
    if any(key.startswith("pretrained_encoder.") for key in state):
        raise ValueError("trainable checkpoint contains frozen encoder parameters")
    current = model.state_dict()
    expected = {
        key for key in current if not key.startswith("pretrained_encoder.")
    }
    missing = sorted(expected - set(state))
    extra = sorted(set(state) - expected)
    if missing or extra:
        raise KeyError(
            f"trainable state mismatch; missing={missing[:5]} extra={extra[:5]}"
        )
    device = next(model.parameters()).device
    current.update({key: value.to(device) for key, value in state.items()})
    model.load_state_dict(current, strict=True)
    model.pretrained_encoder.eval()


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(
            device=device,
            non_blocking=device.type == "cuda",
        )
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def _make_grad_scaler(enabled: bool) -> Any:
    namespace = getattr(torch, "amp", None)
    scaler_type = getattr(namespace, "GradScaler", None)
    if scaler_type is not None:
        try:
            return scaler_type("cuda", enabled=enabled)
        except TypeError:
            return scaler_type(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(enabled: bool) -> Any:
    if not enabled:
        return nullcontext()
    namespace = getattr(torch, "amp", None)
    autocast = getattr(namespace, "autocast", None)
    if autocast is not None:
        try:
            return autocast("cuda", enabled=True)
        except TypeError:
            return autocast(enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def _weighted_loss(
    prediction: torch.Tensor,
    raw_target: torch.Tensor,
    sample_weight: torch.Tensor,
    target_scaler: ScaleOnlyTargetScaler,
    training_cfg: Mapping[str, Any],
) -> torch.Tensor:
    prediction = prediction.reshape(-1)
    raw_target = raw_target.reshape(-1).to(
        device=prediction.device, dtype=prediction.dtype
    )
    sample_weight = sample_weight.reshape(-1).to(
        device=prediction.device, dtype=prediction.dtype
    )
    if raw_target.shape != prediction.shape:
        raise ValueError("target shape must match predictions")
    if sample_weight.shape != prediction.shape:
        raise ValueError("sample weight shape must match predictions")
    if not bool(torch.isfinite(sample_weight).all()) or not bool(
        (sample_weight > 0).all()
    ):
        raise ValueError("sample weights must be finite and positive")
    target = target_scaler.transform(raw_target)
    loss_cfg = training_cfg.get("loss", {})
    mse_weight = float(loss_cfg.get("mse_weight", 1.0))
    huber_weight = float(loss_cfg.get("huber_weight", 0.0))
    if mse_weight < 0.0 or huber_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if mse_weight == 0.0 and huber_weight == 0.0:
        raise ValueError("at least one loss weight must be positive")
    per_record = prediction * 0.0
    if mse_weight:
        per_record = per_record + mse_weight * F.mse_loss(
            prediction, target, reduction="none"
        )
    if huber_weight:
        per_record = per_record + huber_weight * F.smooth_l1_loss(
            prediction, target, reduction="none"
        )
    return (per_record * sample_weight).mean()


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    target_scaler: ScaleOnlyTargetScaler,
    training_cfg: Mapping[str, Any],
    grad_clip_norm: float | None,
    progress_callback: Callable[[int, int, float], None] | None = None,
) -> float:
    model.train()
    accumulation_steps = _resolve_accumulation_steps(training_cfg)
    total_batches = len(loader)
    amp_enabled = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    total_rows = 0
    window_rows = 0
    log_every = int(training_cfg.get("batch_log_every", 0))
    for batch_index, raw_batch in enumerate(loader):
        batch = _move_to_device(raw_batch, device)
        sample_weight = batch.get("sample_weight")
        if not isinstance(sample_weight, torch.Tensor):
            sample_weight = torch.as_tensor(
                sample_weight, dtype=torch.float32, device=device
            )
        with _autocast_context(amp_enabled):
            prediction = model(batch["wt_graph"], batch["mt_graph"])
            loss = _weighted_loss(
                prediction,
                batch["ddg"],
                sample_weight,
                target_scaler,
                training_cfg,
            )
        batch_rows = int(batch["ddg"].reshape(-1).shape[0])
        if batch_rows <= 0:
            raise ValueError("training batch must contain records")
        scaler.scale(loss * float(batch_rows)).backward()
        window_rows += batch_rows
        total_rows += batch_rows
        total_loss_sum += float(loss.detach().cpu()) * batch_rows
        completed = batch_index + 1
        if (
            progress_callback is not None
            and log_every > 0
            and (completed % log_every == 0 or completed == total_batches)
        ):
            progress_callback(completed, total_batches, total_loss_sum / total_rows)
        end_of_window = (
            completed % accumulation_steps == 0 or completed == total_batches
        )
        if end_of_window:
            scaler.unscale_(optimizer)
            parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(float(window_rows))
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            window_rows = 0
    return total_loss_sum / max(1, total_rows)
