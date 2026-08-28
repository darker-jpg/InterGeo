from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from ppi_ddg.local_geom.encoder import masked_softmax
from ppi_ddg.local_geom.interface_context import (
    MutationSiteBatch,
    attention_pool,
    float32_cdist,
    float32_distance_rbf,
    require_finite,
    require_true,
)


__all__ = [
    "BranchFeatureAdapter",
    "BranchFeatures",
    "MutationEnergyReadout",
    "MutationSetEncoder",
    "MutationSetOutput",
    "gather_site_features",
    "masked_sum",
    "optional_mask",
]


@dataclass(frozen=True)
class BranchFeatures:
    h: torch.Tensor
    type: torch.Tensor
    angle: torch.Tensor
    geometry: torch.Tensor
    mixed: torch.Tensor


def _branch_adapter(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
    )


class BranchFeatureAdapter(nn.Module):
    """Keep continuous/type/angle/geometry paths separate until learned fusion."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.h_adapter = _branch_adapter(input_dim, hidden_dim)
        self.type_adapter = _branch_adapter(input_dim, hidden_dim)
        self.angle_adapter = _branch_adapter(input_dim, hidden_dim)
        self.geometry_adapter = _branch_adapter(input_dim, hidden_dim)
        self.mixer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> BranchFeatures:
        required = ("h", "q_type", "q_angle", "q_geometry")
        missing = [key for key in required if key not in features]
        if missing:
            raise KeyError(f"missing pretrained encoder branch features: {', '.join(missing)}")
        mask = valid_mask.to(device=features["h"].device, dtype=torch.bool).unsqueeze(-1)
        h = self.h_adapter(features["h"])
        type_feature = self.type_adapter(features["q_type"])
        angle = self.angle_adapter(features["q_angle"])
        geometry = self.geometry_adapter(features["q_geometry"])
        mixed = self.mixer(torch.cat([h, type_feature, angle, geometry], dim=-1))

        def apply_mask(value: torch.Tensor) -> torch.Tensor:
            return torch.where(mask, value, torch.zeros_like(value))

        return BranchFeatures(
            h=apply_mask(h),
            type=apply_mask(type_feature),
            angle=apply_mask(angle),
            geometry=apply_mask(geometry),
            mixed=apply_mask(mixed),
        )


def optional_mask(graph: dict[str, Any], key: str, valid_mask: torch.Tensor) -> torch.Tensor:
    value = graph.get(key)
    if not isinstance(value, torch.Tensor):
        return torch.zeros_like(valid_mask, dtype=torch.bool)
    output = value.to(device=valid_mask.device, dtype=torch.bool)
    if output.shape[1] < valid_mask.shape[1]:
        pad = torch.zeros(
            (output.shape[0], valid_mask.shape[1] - output.shape[1]),
            dtype=torch.bool,
            device=valid_mask.device,
        )
        output = torch.cat([output, pad], dim=1)
    elif output.shape[1] > valid_mask.shape[1]:
        output = output[:, : valid_mask.shape[1]]
    return output & valid_mask
def masked_sum(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask.to(values.dtype)).sum(dim=1)
class MutationEnergyReadout(nn.Module):
    """Score each mutation independently and preserve magnitude by summation."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(
        self,
        mutation_features: torch.Tensor,
        mutation_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(mutation_features)
        if scores.shape[-1] != 1:
            raise ValueError("mutation score network must return one scalar per mutation")
        scores = scores.squeeze(-1)
        mask = mutation_mask.to(device=scores.device, dtype=torch.bool)
        energy = (scores * mask.to(scores.dtype)).sum(dim=1)
        return energy, scores
def gather_site_features(
    values: torch.Tensor,
    mutation_sites: MutationSiteBatch,
) -> torch.Tensor:
    if not isinstance(values, torch.Tensor) or values.ndim != 3:
        raise ValueError("node features must be [B, L, H]")
    if values.shape[0] != mutation_sites.tokens.shape[0]:
        raise ValueError("node feature batch must match mutation sites")
    if values.device != mutation_sites.tokens.device:
        raise ValueError("node features and mutation sites must share a device")

    batch_size, length, hidden_dim = values.shape
    if length <= 0:
        raise ValueError("node features require a non-empty residue axis")
    safe_indices = mutation_sites.indices.clamp_min(0)
    batch_index = torch.arange(
        batch_size,
        device=values.device,
    ).unsqueeze(1).expand_as(safe_indices)
    gathered = values[batch_index, safe_indices]
    require_finite(
        gathered,
        "gathered mutation features must be finite",
        mutation_sites.mask,
    )
    return torch.where(
        mutation_sites.mask.unsqueeze(-1),
        gathered,
        torch.zeros(
            (
                batch_size,
                mutation_sites.mask.shape[1],
                hidden_dim,
            ),
            dtype=values.dtype,
            device=values.device,
        ),
    )


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dropout_probability(value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return probability


@dataclass(frozen=True)
class MutationSetOutput:
    tokens: torch.Tensor
    mask: torch.Tensor
    pooled: torch.Tensor
    multi_mutation_gate: torch.Tensor


class _MutationSetBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pair_feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pair_bias = nn.Sequential(
            nn.Linear(pair_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )
        self.norm_attn = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        pair_features: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, mutation_count, _hidden_dim = tokens.shape
        query = self.q_proj(tokens).view(
            batch_size,
            mutation_count,
            self.num_heads,
            self.head_dim,
        )
        key = self.k_proj(tokens).view(
            batch_size,
            mutation_count,
            self.num_heads,
            self.head_dim,
        )
        value = self.v_proj(tokens).view(
            batch_size,
            mutation_count,
            self.num_heads,
            self.head_dim,
        )
        scores = torch.einsum("bihd,bjhd->bhij", query, key) * self.scale
        scores = scores + self.pair_bias(pair_features).permute(0, 3, 1, 2)
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        weights = masked_softmax(scores, pair_mask, dim=-1)
        attended = torch.einsum(
            "bhij,bjhd->bihd",
            weights,
            value,
        ).reshape(batch_size, mutation_count, self.hidden_dim)
        updated = self.norm_attn(tokens + self.out_proj(attended))
        updated = self.norm_ff(updated + self.ff(updated))
        return torch.where(
            mask.unsqueeze(-1),
            updated,
            torch.zeros_like(updated),
        )


class MutationSetEncoder(nn.Module):
    """Permutation-equivariant pair-biased attention over mutation sites."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        pair_rbf_bins: int = 8,
        num_layers: int = 2,
        max_pair_distance: float = 30.0,
        dropout: float = 0.08,
    ) -> None:
        super().__init__()
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.num_heads = _positive_int("num_heads", num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        pair_rbf_bins = _positive_int("pair_rbf_bins", pair_rbf_bins)
        self.num_layers = _positive_int("num_layers", num_layers)
        if self.num_layers != 2:
            raise ValueError("mutation-set encoder requires exactly two layers")
        dropout = _dropout_probability(dropout)
        self.blocks = nn.ModuleList(
            [
                _MutationSetBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=self.num_heads,
                    pair_feature_dim=pair_rbf_bins + 3,
                    dropout=dropout,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.pool_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer(
            "pair_distance_centers",
            torch.linspace(
                0.0,
                float(max_pair_distance),
                steps=pair_rbf_bins,
            ),
            persistent=False,
        )

    def _validate_sites(self, sites: MutationSiteBatch) -> None:
        if not isinstance(sites, MutationSiteBatch):
            raise TypeError("sites must be MutationSiteBatch")
        if sites.tokens.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"mutation token dimension must be {self.hidden_dim}"
            )
        require_true(
            sites.mask.any(dim=1).all(),
            "each sample requires at least one mutation",
        )
        require_finite(
            sites.tokens,
            "mutation tokens must be finite",
            sites.mask,
        )
        require_finite(
            sites.metadata.ca,
            "mutation CA coordinates must be finite",
            sites.mask,
        )

    def _pair_features(self, sites: MutationSiteBatch) -> torch.Tensor:
        distance = float32_cdist(
            sites.metadata.ca,
            sites.metadata.ca,
        )
        distance_rbf = float32_distance_rbf(
            distance,
            self.pair_distance_centers,
        ).to(dtype=sites.tokens.dtype)
        sequence_separation = torch.log1p(
            (
                sites.metadata.resseq.unsqueeze(2)
                - sites.metadata.resseq.unsqueeze(1)
            )
            .abs()
            .to(dtype=sites.tokens.dtype)
        ).unsqueeze(-1)
        chain_left = sites.metadata.chain.unsqueeze(2)
        chain_right = sites.metadata.chain.unsqueeze(1)
        same_chain = (
            (chain_left == chain_right)
            & (chain_left >= 0)
            & (chain_right >= 0)
        ).to(dtype=sites.tokens.dtype).unsqueeze(-1)
        molecule_left = sites.metadata.molecule.unsqueeze(2)
        molecule_right = sites.metadata.molecule.unsqueeze(1)
        same_molecule = (
            (molecule_left == molecule_right)
            & (molecule_left >= 0)
            & (molecule_right >= 0)
        ).to(dtype=sites.tokens.dtype).unsqueeze(-1)
        return torch.cat(
            [
                distance_rbf,
                sequence_separation,
                same_chain,
                same_molecule,
            ],
            dim=-1,
        )

    def forward(self, sites: MutationSiteBatch) -> MutationSetOutput:
        self._validate_sites(sites)
        pair_features = self._pair_features(sites)
        tokens = torch.where(
            sites.mask.unsqueeze(-1),
            sites.tokens,
            torch.zeros_like(sites.tokens),
        )
        for block in self.blocks:
            tokens = block(tokens, sites.mask, pair_features)
        pooled = attention_pool(tokens, sites.mask, self.pool_gate)
        mutation_counts = sites.mask.sum(dim=1, keepdim=True)
        multi_mutation_gate = (mutation_counts > 1).to(tokens.dtype)
        return MutationSetOutput(
            tokens=tokens,
            mask=sites.mask,
            pooled=pooled,
            multi_mutation_gate=multi_mutation_gate,
        )
