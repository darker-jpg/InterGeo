from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ppi_ddg.constants import ATOM_TO_INDEX
from ppi_ddg.local_geom.encoder import LocalGeometricEncoder
from ppi_ddg.local_geom.interface_context import (
    BidirectionalInterfaceRelay,
    InterfaceRelayOutput,
    MutationSiteBatch,
    build_geometry_cache,
    gather_mutation_sites,
    prepare_graph_for_forward,
    require_finite,
    require_prepared_graph,
    require_true,
    run_encoder,
    run_pretrained_encoder,
    validate_host_axis_identity,
)
from ppi_ddg.local_geom.mutation_context import (
    BranchFeatureAdapter,
    BranchFeatures,
    MutationEnergyReadout,
    MutationSetEncoder,
    MutationSetOutput,
    gather_site_features,
    masked_sum,
    optional_mask,
)
from ppi_ddg.local_geom.pretrained_encoder import (
    FrozenLocalGeometryEncoder,
    valid_residue_mask,
)


__all__ = [
    "PairedStateDDGModel",
    "PairedStateEncoding",
    "PairedStateEnergies",
]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    scale = mask.unsqueeze(-1).to(values.dtype)
    if weights is not None:
        scale = scale * weights.to(device=values.device, dtype=values.dtype)
    numerator = (values * scale).sum(dim=1)
    denominator = scale.sum(dim=1).clamp_min(1.0)
    return numerator / denominator


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    neg = torch.full((), -1e4, dtype=values.dtype, device=values.device)
    pooled = torch.where(mask.unsqueeze(-1), values, neg).max(dim=1).values
    empty = ~mask.any(dim=1)
    return torch.where(empty.unsqueeze(-1), torch.zeros_like(pooled), pooled)
def _aa_physchem_table() -> torch.Tensor:
    # Features: hydrophobicity/5, charge, volume/200, polar, aromatic, glycine, proline.
    values = {
        "A": (1.8, 0.0, 88.6, 0.0, 0.0, 0.0, 0.0),
        "C": (2.5, 0.0, 108.5, 0.0, 0.0, 0.0, 0.0),
        "D": (-3.5, -1.0, 111.1, 1.0, 0.0, 0.0, 0.0),
        "E": (-3.5, -1.0, 138.4, 1.0, 0.0, 0.0, 0.0),
        "F": (2.8, 0.0, 189.9, 0.0, 1.0, 0.0, 0.0),
        "G": (-0.4, 0.0, 60.1, 0.0, 0.0, 1.0, 0.0),
        "H": (-3.2, 0.5, 153.2, 1.0, 1.0, 0.0, 0.0),
        "I": (4.5, 0.0, 166.7, 0.0, 0.0, 0.0, 0.0),
        "K": (-3.9, 1.0, 168.6, 1.0, 0.0, 0.0, 0.0),
        "L": (3.8, 0.0, 166.7, 0.0, 0.0, 0.0, 0.0),
        "M": (1.9, 0.0, 162.9, 0.0, 0.0, 0.0, 0.0),
        "N": (-3.5, 0.0, 114.1, 1.0, 0.0, 0.0, 0.0),
        "P": (-1.6, 0.0, 112.7, 0.0, 0.0, 0.0, 1.0),
        "Q": (-3.5, 0.0, 143.8, 1.0, 0.0, 0.0, 0.0),
        "R": (-4.5, 1.0, 173.4, 1.0, 0.0, 0.0, 0.0),
        "S": (-0.8, 0.0, 89.0, 1.0, 0.0, 0.0, 0.0),
        "T": (-0.7, 0.0, 116.1, 1.0, 0.0, 0.0, 0.0),
        "V": (4.2, 0.0, 140.0, 0.0, 0.0, 0.0, 0.0),
        "W": (-0.9, 0.0, 227.8, 0.0, 1.0, 0.0, 0.0),
        "Y": (-1.3, 0.0, 193.6, 1.0, 1.0, 0.0, 0.0),
        "X": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    }
    rows = []
    for aa in "ACDEFGHIKLMNPQRSTVWYX":
        hydrophobicity, charge, volume, polar, aromatic, glycine, proline = values[aa]
        rows.append([hydrophobicity / 5.0, charge, volume / 200.0, polar, aromatic, glycine, proline])
    return torch.tensor(rows, dtype=torch.float32)


def _positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _dropout_probability(value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    return probability


class _SetEnergyReadout(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim * 3),
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        contextual: torch.Tensor,
        raw: torch.Tensor,
        mask: torch.Tensor,
        multi_mutation_gate: torch.Tensor,
    ) -> torch.Tensor:
        if contextual.shape != raw.shape:
            raise ValueError(
                "contextual and raw mutation tokens must align"
            )
        features = torch.cat(
            [contextual, raw, contextual - raw],
            dim=-1,
        )
        scores = self.score(features).squeeze(-1)
        energy = (
            scores * mask.to(dtype=scores.dtype)
        ).sum(dim=1)
        return energy * multi_mutation_gate.squeeze(-1)


@dataclass(frozen=True)
class PairedStateEncoding:
    branches: BranchFeatures
    state_tokens: torch.Tensor
    fused_sites: torch.Tensor
    site_features: torch.Tensor
    mutation_sites: MutationSiteBatch
    mutation_set: MutationSetOutput
    relay: InterfaceRelayOutput
    valid_mask: torch.Tensor
    scope_mask: torch.Tensor
    interface_mask: torch.Tensor
    residue_mutation_mask: torch.Tensor

    @property
    def mutation_mask(self) -> torch.Tensor:
        return self.mutation_sites.mask


@dataclass(frozen=True)
class PairedStateEnergies:
    additive: torch.Tensor
    set: torch.Tensor
    global_: torch.Tensor


def validate_paired_graphs(
    wt_graph: dict[str, Any],
    mt_graph: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(wt_graph, dict) or not isinstance(mt_graph, dict):
        raise TypeError("WT and MT graphs must be dictionaries")
    wt_pos = wt_graph.get("pos_atoms")
    mt_pos = mt_graph.get("pos_atoms")
    if (
        not isinstance(wt_pos, torch.Tensor)
        or not isinstance(mt_pos, torch.Tensor)
        or wt_pos.ndim != 4
        or mt_pos.ndim != 4
        or wt_pos.shape[:2] != mt_pos.shape[:2]
        or wt_pos.device != mt_pos.device
    ):
        raise ValueError("WT and MT graphs require aligned residue axes")
    axis_shape = (wt_pos.shape[0], wt_pos.shape[1])

    wt_graph = prepare_graph_for_forward(wt_graph)
    mt_graph = prepare_graph_for_forward(mt_graph)
    wt_resseq, wt_chain, wt_molecule, _wt_counts, _wt_max = (
        require_prepared_graph(wt_graph)
    )
    mt_resseq, mt_chain, mt_molecule, _mt_counts, _mt_max = (
        require_prepared_graph(mt_graph)
    )
    wt_valid = valid_residue_mask(wt_graph)
    mt_valid = valid_residue_mask(mt_graph)
    if (
        wt_valid.shape != axis_shape
        or mt_valid.shape != axis_shape
        or wt_valid.device != mt_valid.device
    ):
        raise ValueError("WT and MT valid masks must match the residue axis")
    require_true((wt_valid == mt_valid).all(), "WT and MT valid masks do not match")
    require_true((wt_resseq == mt_resseq).all(), "WT and MT residue numbers do not match")
    require_true((wt_chain == mt_chain).all(), "WT and MT chain identities do not match")
    validate_host_axis_identity(
        wt_graph,
        mt_graph,
        key="chain_ids",
        batch_size=axis_shape[0],
        message="WT and MT chain identities do not match",
    )
    require_true(
        (wt_molecule == mt_molecule).all(),
        "WT and MT molecule identities do not match",
    )
    validate_host_axis_identity(
        wt_graph,
        mt_graph,
        key="mol_ids",
        batch_size=axis_shape[0],
        message="WT and MT molecule identities do not match",
    )
    validate_host_axis_identity(
        wt_graph,
        mt_graph,
        key="residue_ids",
        batch_size=axis_shape[0],
        message="WT and MT stable residue identifiers do not match",
    )

    wt_mutation = wt_graph.get("mutation_mask")
    mt_mutation = mt_graph.get("mutation_mask")
    if (
        not isinstance(wt_mutation, torch.Tensor)
        or not isinstance(mt_mutation, torch.Tensor)
        or wt_mutation.shape != axis_shape
        or mt_mutation.shape != axis_shape
        or wt_mutation.device != wt_pos.device
        or mt_mutation.device != mt_pos.device
        or wt_mutation.device != mt_mutation.device
    ):
        raise ValueError("WT and MT graphs require identical mutation masks")
    require_true(
        (
            wt_mutation.to(dtype=torch.bool)
            == mt_mutation.to(dtype=torch.bool)
        ).all(),
        "WT and MT mutation masks do not match",
    )
    return wt_graph, mt_graph


def validate_encoded_alignment(
    wt_state: PairedStateEncoding,
    mt_state: PairedStateEncoding,
) -> None:
    wt_sites = wt_state.mutation_sites
    mt_sites = mt_state.mutation_sites
    if (
        wt_sites.mask.shape != mt_sites.mask.shape
        or wt_sites.indices.shape != mt_sites.indices.shape
        or wt_sites.mask.device != mt_sites.mask.device
        or wt_sites.indices.device != mt_sites.indices.device
    ):
        raise ValueError("encoded mutation sites are not aligned")
    require_true(
        (wt_sites.mask == mt_sites.mask).all(),
        "encoded mutation padding masks are not aligned",
    )
    require_true(
        (wt_sites.indices == mt_sites.indices).all(),
        "encoded mutation indices are not aligned",
    )
    for metadata_name in ("resseq", "chain", "molecule"):
        wt_values = getattr(wt_sites.metadata, metadata_name)
        mt_values = getattr(mt_sites.metadata, metadata_name)
        message = f"encoded mutation {metadata_name} metadata is not aligned"
        if wt_values.shape != mt_values.shape or wt_values.device != mt_values.device:
            raise ValueError(message)
        require_true((wt_values == mt_values).all(), message)


class PairedStateDDGModel(nn.Module):
    """Shared-state energy model with exact inference antisymmetry."""

    def __init__(
        self,
        pretrained_encoder: FrozenLocalGeometryEncoder,
        *,
        hidden_dim: int | None = 256,
        dropout: float = 0.08,
        downstream_blocks: int = 3,
        downstream_heads: int = 8,
        set_blocks: int = 2,
        structure_cutoff: float | None = None,
        interface_cutoff: float | None = None,
        structure_k: int | None = None,
        interface_k: int | None = None,
        pair_rbf_bins: int = 8,
        partner_rbf_bins: int = 8,
        relay_rbf_bins: int | None = None,
        max_partner_tokens: int = 128,
        max_interface_tokens: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(pretrained_encoder, FrozenLocalGeometryEncoder):
            raise TypeError("pretrained_encoder must be FrozenLocalGeometryEncoder")
        self.pretrained_encoder = pretrained_encoder
        self.pretrained_encoder.eval()
        for parameter in self.pretrained_encoder.parameters():
            parameter.requires_grad_(False)

        self.hidden_dim = _positive_int(
            "hidden_dim",
            int(
                hidden_dim
                if hidden_dim is not None
                else pretrained_encoder.node_dim
            ),
        )
        self.downstream_blocks = _positive_int(
            "downstream_blocks",
            downstream_blocks,
        )
        self.downstream_heads = _positive_int(
            "downstream_heads",
            downstream_heads,
        )
        if self.hidden_dim % self.downstream_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by downstream_heads"
            )
        dropout = _dropout_probability(dropout)
        set_layer_count = _positive_int(
            "set_blocks",
            int(set_blocks),
        )
        if set_layer_count != 2:
            raise ValueError(
                "paired-state mutation-set encoder requires exactly 2 layers"
            )

        structure_cutoff_value = float(
            structure_cutoff
            if structure_cutoff is not None
            else pretrained_encoder.encoder.structure_cutoff
        )
        interface_cutoff_value = float(
            interface_cutoff
            if interface_cutoff is not None
            else pretrained_encoder.encoder.interface_cutoff
        )
        structure_k_value = int(
            structure_k
            if structure_k is not None
            else pretrained_encoder.encoder.structure_k
        )
        interface_k_value = int(
            interface_k
            if interface_k is not None
            else pretrained_encoder.encoder.interface_k
        )
        pretrained_encoder_neighbors = (
            float(pretrained_encoder.encoder.structure_cutoff),
            float(pretrained_encoder.encoder.interface_cutoff),
            int(pretrained_encoder.encoder.structure_k),
            int(pretrained_encoder.encoder.interface_k),
        )
        downstream_neighbors = (
            structure_cutoff_value,
            interface_cutoff_value,
            structure_k_value,
            interface_k_value,
        )
        if downstream_neighbors != pretrained_encoder_neighbors:
            raise ValueError(
                "pretrained encoder and downstream neighbor hyperparameters must match"
            )

        self.state_adapter = BranchFeatureAdapter(
            pretrained_encoder.node_dim,
            self.hidden_dim,
        )
        self.structural_encoder = LocalGeometricEncoder(
            node_dim=self.hidden_dim,
            num_blocks=self.downstream_blocks,
            num_heads=self.downstream_heads,
            structure_cutoff=structure_cutoff_value,
            interface_cutoff=interface_cutoff_value,
            structure_k=structure_k_value,
            interface_k=interface_k_value,
        )
        self.physchem_proj = nn.Sequential(
            nn.Linear(7, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.site_fusion = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 6),
            nn.Linear(self.hidden_dim * 6, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.mutation_set_encoder = MutationSetEncoder(
            hidden_dim=self.hidden_dim,
            num_heads=self.downstream_heads,
            pair_rbf_bins=pair_rbf_bins,
            num_layers=set_layer_count,
            dropout=dropout,
        )
        relay_bins = int(
            relay_rbf_bins
            if relay_rbf_bins is not None
            else partner_rbf_bins
        )
        self.interface_relay = BidirectionalInterfaceRelay(
            hidden_dim=self.hidden_dim,
            num_heads=self.downstream_heads,
            distance_rbf_bins=relay_bins,
            max_partner_tokens=max_partner_tokens,
            max_interface_tokens=max_interface_tokens,
        )
        self.additive_energy_head = MutationEnergyReadout(
            input_dim=self.hidden_dim * 3,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.set_energy_head = _SetEnergyReadout(
            hidden_dim=self.hidden_dim,
            dropout=dropout,
        )
        self.mutation_count_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.global_energy_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 15),
            nn.Linear(self.hidden_dim * 15, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, max(1, self.hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(1, self.hidden_dim // 2), 1),
        )
        self.additive_scale = nn.Parameter(torch.tensor(1.0))
        self.set_scale = nn.Parameter(torch.tensor(0.25))
        self.global_scale = nn.Parameter(torch.tensor(0.25))
        self.register_buffer(
            "aa_physchem",
            _aa_physchem_table(),
            persistent=False,
        )

    def train(self, mode: bool = True) -> "PairedStateDDGModel":
        super().train(mode)
        self.pretrained_encoder.eval()
        return self

    def _scoped_graph(
        self,
        graph: dict[str, Any],
        valid_mask: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        mutation = optional_mask(graph, "mutation_mask", valid_mask)
        local = optional_mask(graph, "local_mask", valid_mask)
        interface = optional_mask(graph, "interface_mask", valid_mask)
        scope = valid_mask & (mutation | local | interface)
        empty_scope = (
            valid_mask.any(dim=1, keepdim=True)
            & ~scope.any(dim=1, keepdim=True)
        )
        scope = torch.where(empty_scope, valid_mask, scope)
        scoped_graph = dict(graph)
        scoped_graph["node_mask"] = scope
        return scoped_graph, scope, mutation, interface

    def _node_physchem(
        self,
        graph: dict[str, Any],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        aa_value = graph.get("aa")
        if not isinstance(aa_value, torch.Tensor):
            raise ValueError("graph aa must be a tensor")
        aa = aa_value.to(device=valid_mask.device, dtype=torch.long)
        if aa.shape != valid_mask.shape:
            raise ValueError("graph aa must align with the residue axis")
        valid_type = (aa >= 0) & (aa < self.aa_physchem.shape[0])
        require_true(
            (valid_type | ~valid_mask).all(),
            "graph has invalid residue types at valid nodes",
        )
        unknown_index = self.aa_physchem.shape[0] - 1
        aa = torch.where(
            valid_mask,
            aa,
            torch.full_like(aa, unknown_index),
        ).clamp(min=0, max=unknown_index)
        encoded = self.physchem_proj(
            self.aa_physchem.to(
                device=aa.device,
                dtype=torch.float32,
            )[aa]
        )
        return torch.where(
            valid_mask.unsqueeze(-1),
            encoded,
            torch.zeros_like(encoded),
        )

    def _encode_state(self, graph: dict[str, Any]) -> PairedStateEncoding:
        graph = prepare_graph_for_forward(graph)
        valid_mask = valid_residue_mask(graph)
        scoped_graph, scope, mutation, interface = self._scoped_graph(
            graph,
            valid_mask,
        )
        geometry_cache = build_geometry_cache(
            scoped_graph,
            self.pretrained_encoder.encoder,
        )
        pretrained_encoder_features = run_pretrained_encoder(
            self.pretrained_encoder,
            scoped_graph,
            geometry_cache,
        )
        branches = self.state_adapter(
            pretrained_encoder_features,
            pretrained_encoder_features["valid_mask"],
        )
        state_tokens, _state_interface = run_encoder(
            self.structural_encoder,
            scoped_graph,
            geometry_cache,
            node_bias=branches.mixed,
            node_bias_mask=scope,
        )
        state_tokens = torch.where(
            scope.unsqueeze(-1),
            state_tokens,
            torch.zeros_like(state_tokens),
        )
        raw_sites = gather_mutation_sites(
            state_tokens,
            graph,
            mutation,
        )
        site_input = torch.cat(
            [
                raw_sites.tokens,
                gather_site_features(branches.h, raw_sites),
                gather_site_features(branches.type, raw_sites),
                gather_site_features(branches.angle, raw_sites),
                gather_site_features(branches.geometry, raw_sites),
                gather_site_features(
                    self._node_physchem(graph, valid_mask),
                    raw_sites,
                ),
            ],
            dim=-1,
        )
        fused_sites = self.site_fusion(site_input)
        fused_sites = torch.where(
            raw_sites.mask.unsqueeze(-1),
            fused_sites,
            torch.zeros_like(fused_sites),
        )
        fused_batch = MutationSiteBatch(
            tokens=fused_sites,
            mask=raw_sites.mask,
            indices=raw_sites.indices,
            metadata=raw_sites.metadata,
        )
        mutation_set = self.mutation_set_encoder(fused_batch)
        relay = self.interface_relay(
            state_tokens,
            graph,
            fused_batch,
            mutation_set,
            valid_mask=valid_mask,
            interface_mask=interface,
            mutation_mask=mutation,
            ca_distance=geometry_cache.ca_distance,
        )
        site_features = torch.cat(
            [
                fused_sites,
                mutation_set.tokens,
                relay.partner_context,
            ],
            dim=-1,
        )
        site_features = torch.where(
            raw_sites.mask.unsqueeze(-1),
            site_features,
            torch.zeros_like(site_features),
        )
        require_finite(
            site_features,
            "paired-state mutation site features must be finite",
            raw_sites.mask,
        )
        return PairedStateEncoding(
            branches=branches,
            state_tokens=state_tokens,
            fused_sites=fused_sites,
            site_features=site_features,
            mutation_sites=fused_batch,
            mutation_set=mutation_set,
            relay=relay,
            valid_mask=valid_mask,
            scope_mask=scope,
            interface_mask=interface,
            residue_mutation_mask=mutation,
        )

    def _global_energy(self, state: PairedStateEncoding) -> torch.Tensor:
        mutation_count = state.mutation_mask.sum(
            dim=1,
            keepdim=True,
        ).to(dtype=state.state_tokens.dtype)
        count_feature = self.mutation_count_proj(
            torch.log1p(mutation_count)
        )
        descriptor = torch.cat(
            [
                masked_sum(state.state_tokens, state.scope_mask),
                _masked_mean(state.state_tokens, state.scope_mask),
                _masked_max(state.state_tokens, state.scope_mask),
                masked_sum(state.state_tokens, state.interface_mask),
                _masked_mean(state.state_tokens, state.interface_mask),
                _masked_max(state.state_tokens, state.interface_mask),
                masked_sum(state.fused_sites, state.mutation_mask),
                masked_sum(
                    gather_site_features(
                        state.branches.h,
                        state.mutation_sites,
                    ),
                    state.mutation_mask,
                ),
                masked_sum(
                    gather_site_features(
                        state.branches.type,
                        state.mutation_sites,
                    ),
                    state.mutation_mask,
                ),
                masked_sum(
                    gather_site_features(
                        state.branches.angle,
                        state.mutation_sites,
                    ),
                    state.mutation_mask,
                ),
                masked_sum(
                    gather_site_features(
                        state.branches.geometry,
                        state.mutation_sites,
                    ),
                    state.mutation_mask,
                ),
                state.mutation_set.pooled,
                state.relay.partner_pooled,
                state.relay.interface_pooled,
                count_feature,
            ],
            dim=-1,
        )
        if descriptor.shape[-1] != self.hidden_dim * 15:
            raise RuntimeError(
                "paired-state global descriptor must contain 15 hidden branches"
            )
        return self.global_energy_head(descriptor).squeeze(-1)

    def _state_energies(
        self,
        state: PairedStateEncoding,
    ) -> PairedStateEnergies:
        additive, _scores = self.additive_energy_head(
            state.site_features,
            state.mutation_mask,
        )
        set_energy = self.set_energy_head(
            state.mutation_set.tokens,
            state.fused_sites,
            state.mutation_mask,
            state.mutation_set.multi_mutation_gate,
        )
        return PairedStateEnergies(
            additive=additive,
            set=set_energy,
            global_=self._global_energy(state),
        )

    def forward_components(
        self,
        wt_graph: dict[str, Any],
        mt_graph: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        wt_graph, mt_graph = validate_paired_graphs(wt_graph, mt_graph)
        wt_state = self._encode_state(wt_graph)
        mt_state = self._encode_state(mt_graph)
        validate_encoded_alignment(wt_state, mt_state)
        wt_energy = self._state_energies(wt_state)
        mt_energy = self._state_energies(mt_state)
        delta_additive = mt_energy.additive - wt_energy.additive
        delta_set = mt_energy.set - wt_energy.set
        delta_global = mt_energy.global_ - wt_energy.global_
        prediction = (
            self.additive_scale * delta_additive
            + self.set_scale * delta_set
            + self.global_scale * delta_global
        )
        return {
            "delta_additive": delta_additive,
            "delta_set": delta_set,
            "delta_global": delta_global,
            "prediction": prediction,
        }

    def forward(
        self,
        wt_graph: dict[str, Any],
        mt_graph: dict[str, Any],
    ) -> torch.Tensor:
        return self.forward_components(wt_graph, mt_graph)["prediction"]
