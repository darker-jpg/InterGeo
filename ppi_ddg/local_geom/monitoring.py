from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

__all__ = [
    "AssignmentAccumulator",
    "CodebookStatistics",
    "codebook_health",
    "deterministic_branch_masks",
    "shuffle_quantized_codes",
    "split_pretrain_indices",
]


def _sha256_int(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def _lookup_stat(stats: Any, name: str) -> Any:
    if isinstance(stats, Mapping):
        return stats[name]
    return getattr(stats, name)


def _as_scalar_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _as_scalar_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().item())
    return int(value)


def _as_tensor_histogram(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(value, dtype=torch.long)


@dataclass(frozen=True)
class CodebookStatistics:
    histogram: torch.Tensor
    active_codes: int
    perplexity: torch.Tensor
    max_share: torch.Tensor
    total: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "histogram": self.histogram.detach().cpu().tolist(),
            "active_codes": int(self.active_codes),
            "perplexity": float(self.perplexity.detach().cpu().item()),
            "max_share": float(self.max_share.detach().cpu().item()),
            "total": int(self.total),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class AssignmentAccumulator:
    def __init__(self, num_embeddings: int):
        if num_embeddings <= 0:
            raise ValueError("num_embeddings must be positive")
        self.num_embeddings = int(num_embeddings)
        self._histogram = torch.zeros(self.num_embeddings, dtype=torch.long)

    def reset(self) -> None:
        self._histogram.zero_()

    def update(self, indices: torch.Tensor, mask: torch.Tensor) -> None:
        indices = indices.to(dtype=torch.long, device=self._histogram.device)
        mask = mask.to(dtype=torch.bool, device=self._histogram.device)
        if indices.shape != mask.shape:
            raise ValueError("indices and mask must have the same shape")
        if not mask.any():
            return
        selected = indices[mask].reshape(-1)
        if selected.numel() == 0:
            return
        self._histogram += torch.bincount(selected, minlength=self.num_embeddings)

    def stats(self) -> CodebookStatistics:
        histogram = self._histogram.clone()
        total = int(histogram.sum().item())
        if total > 0:
            probs = histogram.to(dtype=torch.float32) / float(total)
            active = histogram > 0
            active_codes = int(active.sum().item())
            entropy = -(probs[active] * probs[active].log()).sum()
            perplexity = entropy.exp()
            max_share = probs.max()
        else:
            active_codes = 0
            perplexity = torch.zeros((), dtype=torch.float32, device=histogram.device)
            max_share = torch.zeros((), dtype=torch.float32, device=histogram.device)
        return CodebookStatistics(
            histogram=histogram,
            active_codes=active_codes,
            perplexity=perplexity,
            max_share=max_share,
            total=total,
        )


def codebook_health(
    stats: Any,
    min_active: int = 16,
    min_perplexity: float = 8.0,
    max_share: float = 0.5,
) -> bool:
    active_codes = _as_scalar_int(_lookup_stat(stats, "active_codes"))
    perplexity = _as_scalar_float(_lookup_stat(stats, "perplexity"))
    max_share_value = _as_scalar_float(_lookup_stat(stats, "max_share"))
    return active_codes >= min_active and perplexity >= min_perplexity and max_share_value < max_share


def split_pretrain_indices(
    sample_ids: Sequence[object],
    val_fraction: float = 0.1,
    seed: int = 2026,
) -> tuple[list[int], list[int]]:
    sample_ids = list(sample_ids)
    if not sample_ids:
        return [], []

    grouped_indices: dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        grouped_indices.setdefault(str(sample_id), []).append(index)

    ordered_ids = sorted(grouped_indices, key=lambda sample_id: (_sha256_int(seed, sample_id), sample_id))
    total = len(sample_ids)

    if len(ordered_ids) == 1:
        if 0.0 < val_fraction < 1.0 and total > 1:
            return [], list(grouped_indices[ordered_ids[0]])
        return ([], list(grouped_indices[ordered_ids[0]])) if val_fraction >= 0.5 else (list(grouped_indices[ordered_ids[0]]), [])

    if val_fraction <= 0.0:
        return list(range(total)), []
    if val_fraction >= 1.0:
        return [], list(range(total))

    target_val = int(round(total * val_fraction))
    target_val = min(max(target_val, 1), total - 1)

    val_ids: list[str] = []
    val_count = 0
    for sample_id in ordered_ids:
        if val_count >= target_val and val_ids:
            break
        val_ids.append(sample_id)
        val_count += len(grouped_indices[sample_id])
        if val_count >= target_val and val_count < total:
            break

    if not val_ids:
        val_ids = [ordered_ids[0]]
    if len(val_ids) == len(ordered_ids):
        val_ids = ordered_ids[:-1]

    val_index_set: set[int] = set()
    for sample_id in val_ids:
        val_index_set.update(grouped_indices[sample_id])

    val_idx = sorted(val_index_set)
    train_idx = [index for index in range(total) if index not in val_index_set]
    return train_idx, val_idx


def deterministic_branch_masks(
    sample_ids: Sequence[object],
    candidate_mask: torch.Tensor,
    mask_ratio: float,
    seed: int,
    branches: Sequence[str] = ("type", "angle", "geometry", "rsa"),
) -> dict[str, torch.Tensor]:
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    candidate_mask = candidate_mask.to(dtype=torch.bool)
    if candidate_mask.shape[0] != len(sample_ids):
        raise ValueError("sample_ids length must match candidate_mask batch dimension")

    flat_candidates = candidate_mask.reshape(candidate_mask.shape[0], -1)
    device = candidate_mask.device
    outputs: dict[str, torch.Tensor] = {}

    for branch in branches:
        branch_mask = torch.zeros_like(flat_candidates)
        for row, sample_id in enumerate(sample_ids):
            positions = torch.nonzero(flat_candidates[row], as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            branch_seed = _sha256_int(seed, sample_id, branch)
            generator = torch.Generator(device="cpu").manual_seed(branch_seed)
            draws = torch.rand(positions.numel(), generator=generator)
            keep = draws < mask_ratio
            if keep.any():
                kept_positions = positions[keep].to(device=device)
                branch_mask[row, kept_positions] = True
        outputs[branch] = branch_mask.reshape(candidate_mask.shape)
    return outputs


def shuffle_quantized_codes(
    q: torch.Tensor,
    mask: torch.Tensor,
    generator: torch.Generator | None,
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=q.device)
    if mask.shape != q.shape[:-1]:
        raise ValueError("mask must match q.shape[:-1]")

    flat_q = q.reshape(-1, q.shape[-1])
    flat_mask = mask.reshape(-1)
    valid_positions = torch.nonzero(flat_mask, as_tuple=False).flatten()
    if valid_positions.numel() <= 1:
        return q.clone()

    perm_device = q.device if generator is None else torch.device(generator.device)
    permutation = torch.randperm(valid_positions.numel(), device=perm_device, generator=generator)
    if permutation.device != q.device:
        permutation = permutation.to(q.device)

    shuffled = q.clone().reshape(-1, q.shape[-1])
    shuffled[valid_positions] = flat_q[valid_positions[permutation]]
    return shuffled.reshape_as(q)
