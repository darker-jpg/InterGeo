from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ppi_ddg.constants import ATOM_TO_INDEX
from ppi_ddg.local_geom.encoder import directed_edge_geometry
from ppi_ddg.local_geom.geometry import construct_local_frames, local_atom_targets, safe_normalize, to_local
from ppi_ddg.local_geom.graph import ComplexNeighborLists, NeighborList, batched_neighbor_gather, build_neighbor_lists

__all__ = [
    "GeometryDecoderState",
    "GeometryDecoderSupervision",
    "IterativeGeometricDecoder",
    "IterativeGeometryBlock",
    "RelationEdgeState",
    "SparseFeatureDecoder",
    "initialize_geometry_state",
]


@dataclass(frozen=True)
class RelationEdgeState:
    indices: torch.Tensor
    mask: torch.Tensor
    distance: torch.Tensor
    direction: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class GeometryDecoderState:
    local_atoms: torch.Tensor
    local_mask: torch.Tensor
    node_mask: torch.Tensor
    geometry_mask: torch.Tensor
    structure: RelationEdgeState
    interface: RelationEdgeState


@dataclass(frozen=True)
class GeometryDecoderSupervision:
    local_atoms: torch.Tensor
    structure: torch.Tensor
    interface: torch.Tensor


def _prepare_mask(
    value: torch.Tensor | None,
    shape: tuple[int, int],
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if value is None:
        return torch.full(shape, default, dtype=torch.bool, device=device)

    tensor = value.to(device=device, dtype=torch.bool)
    if tensor.shape[1] < shape[1]:
        pad = torch.full((tensor.shape[0], shape[1] - tensor.shape[1]), default, dtype=torch.bool, device=device)
        tensor = torch.cat([tensor, pad], dim=1)
    elif tensor.shape[1] > shape[1]:
        tensor = tensor[:, : shape[1]]
    return tensor


def _normalize_direction(direction: torch.Tensor, valid_mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    normalized = safe_normalize(direction, eps=eps)
    zero_norm = direction.norm(dim=-1, keepdim=True) <= eps
    fallback = torch.zeros_like(direction)
    fallback[..., 0] = 1.0
    normalized = torch.where(valid_mask.unsqueeze(-1) & zero_norm, fallback, normalized)
    return torch.where(valid_mask.unsqueeze(-1), normalized, torch.zeros_like(direction))


def _edge_targets(
    frame: torch.Tensor,
    frame_valid: torch.Tensor,
    ca_positions: torch.Tensor,
    neighbors: NeighborList,
    node_mask: torch.Tensor,
) -> RelationEdgeState:
    source_valid = frame_valid.to(dtype=torch.bool, device=ca_positions.device) & node_mask
    target_valid = batched_neighbor_gather(source_valid.unsqueeze(-1), neighbors.indices).squeeze(-1)
    valid_mask = neighbors.mask.to(dtype=torch.bool, device=ca_positions.device) & source_valid.unsqueeze(-1) & target_valid

    target_ca = batched_neighbor_gather(ca_positions, neighbors.indices)
    displacement = target_ca - ca_positions.unsqueeze(-2)
    distance = displacement.norm(dim=-1)
    direction = to_local(frame, displacement)
    direction = _normalize_direction(direction, valid_mask)

    distance = torch.where(valid_mask, distance, torch.zeros_like(distance))
    return RelationEdgeState(
        indices=neighbors.indices,
        mask=neighbors.mask.to(dtype=torch.bool, device=ca_positions.device),
        distance=distance,
        direction=direction,
        valid_mask=valid_mask,
    )


def _incident_mask(geometry_mask: torch.Tensor, relation: RelationEdgeState) -> torch.Tensor:
    target_mask = batched_neighbor_gather(geometry_mask.unsqueeze(-1), relation.indices).squeeze(-1)
    return relation.mask & (geometry_mask.unsqueeze(-1) | target_mask)


def _masked_relation_input(relation: RelationEdgeState, geometry_mask: torch.Tensor) -> RelationEdgeState:
    incident = _incident_mask(geometry_mask, relation)
    return RelationEdgeState(
        indices=relation.indices,
        mask=relation.mask,
        distance=torch.where(incident, torch.zeros_like(relation.distance), relation.distance),
        direction=torch.where(incident.unsqueeze(-1), torch.zeros_like(relation.direction), relation.direction),
        valid_mask=relation.valid_mask,
    )


def initialize_geometry_state(
    batch: dict,
    geometry_mask: torch.Tensor,
    neighbors: ComplexNeighborLists | None = None,
) -> tuple[GeometryDecoderState, GeometryDecoderState, GeometryDecoderSupervision]:
    pos_atoms = batch["pos_atoms"]
    device = pos_atoms.device
    batch_size, length, atom_count = pos_atoms.shape[:3]

    node_mask = _prepare_mask(batch.get("node_mask"), (batch_size, length), device, default=True)
    mask_atoms_value = batch.get("mask_atoms")
    if mask_atoms_value is None:
        mask_atoms = torch.ones((batch_size, length, atom_count), dtype=torch.bool, device=device)
    else:
        mask_atoms = mask_atoms_value.to(device=device, dtype=torch.bool)

    geometry_mask = _prepare_mask(geometry_mask, (batch_size, length), device, default=False) & node_mask

    frame, frame_valid = construct_local_frames(pos_atoms, mask_atoms)
    local_atoms, local_mask = local_atom_targets(pos_atoms, mask_atoms, frame, frame_valid)
    local_mask = local_mask & node_mask.unsqueeze(-1)
    local_atoms = torch.where(local_mask.unsqueeze(-1), local_atoms, torch.zeros_like(local_atoms))

    if neighbors is None:
        neighbors = build_neighbor_lists(batch)

    ca_positions = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]
    structure_target = _edge_targets(frame, frame_valid, ca_positions, neighbors.structure, node_mask)
    interface_target = _edge_targets(frame, frame_valid, ca_positions, neighbors.interface, node_mask)

    decoder_input = GeometryDecoderState(
        local_atoms=torch.where(geometry_mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(local_atoms), local_atoms),
        local_mask=local_mask,
        node_mask=node_mask,
        geometry_mask=geometry_mask,
        structure=_masked_relation_input(structure_target, geometry_mask),
        interface=_masked_relation_input(interface_target, geometry_mask),
    )
    clean_target = GeometryDecoderState(
        local_atoms=local_atoms,
        local_mask=local_mask,
        node_mask=node_mask,
        geometry_mask=geometry_mask,
        structure=structure_target,
        interface=interface_target,
    )
    supervision = GeometryDecoderSupervision(
        local_atoms=local_mask & geometry_mask.unsqueeze(-1),
        structure=structure_target.valid_mask & _incident_mask(geometry_mask, structure_target),
        interface=interface_target.valid_mask & _incident_mask(geometry_mask, interface_target),
    )
    return decoder_input, clean_target, supervision


class _RelationUpdate(nn.Module):
    def __init__(self, node_dim: int, relation_flag: float):
        super().__init__()
        self.relation_flag = relation_flag
        input_dim = node_dim * 2 + 1 + 3 + 3
        self.message_mlp = nn.Sequential(
            nn.Linear(input_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(node_dim + 1 + 3, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, 4),
        )

    def forward(
        self,
        h: torch.Tensor,
        relation: RelationEdgeState,
        geometry_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, RelationEdgeState]:
        if relation.indices.shape[-1] == 0:
            return torch.zeros_like(h), relation

        target_h = batched_neighbor_gather(h, relation.indices)
        source_h = h.unsqueeze(-2).expand_as(target_h)
        source_mask_flag = geometry_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=h.dtype).expand(*relation.indices.shape, 1)
        target_mask = batched_neighbor_gather(geometry_mask.unsqueeze(-1), relation.indices).to(dtype=h.dtype)
        relation_flag = torch.full(
            (*relation.indices.shape, 1),
            self.relation_flag,
            dtype=h.dtype,
            device=h.device,
        )

        message_input = torch.cat(
            [
                source_h,
                target_h,
                relation.distance.unsqueeze(-1),
                relation.direction,
                relation_flag,
                source_mask_flag,
                target_mask,
            ],
            dim=-1,
        )
        message = self.message_mlp(message_input)
        message = torch.where(relation.valid_mask.unsqueeze(-1), message, torch.zeros_like(message))

        denom = relation.valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
        aggregated = message.sum(dim=-2) / denom

        edge_delta = self.edge_update(torch.cat([message, relation.distance.unsqueeze(-1), relation.direction], dim=-1))
        delta_distance = edge_delta[..., 0]
        delta_direction = edge_delta[..., 1:]

        updated_distance = (relation.distance + delta_distance).clamp_min(0.0)
        updated_distance = torch.where(relation.valid_mask, updated_distance, torch.zeros_like(updated_distance))
        updated_direction = _normalize_direction(relation.direction + delta_direction, relation.valid_mask)
        updated_relation = RelationEdgeState(
            indices=relation.indices,
            mask=relation.mask,
            distance=updated_distance,
            direction=updated_direction,
            valid_mask=relation.valid_mask,
        )
        return aggregated, updated_relation


class IterativeGeometryBlock(nn.Module):
    def __init__(self, node_dim: int):
        super().__init__()
        self.structure_update = _RelationUpdate(node_dim=node_dim, relation_flag=0.0)
        self.interface_update = _RelationUpdate(node_dim=node_dim, relation_flag=1.0)
        self.node_norm = nn.LayerNorm(node_dim)
        self.node_update = nn.Sequential(
            nn.Linear(node_dim * 3 + 2, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.local_update = nn.Sequential(
            nn.Linear(node_dim * 3 + 14, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, 12),
        )

    def forward(
        self,
        h: torch.Tensor,
        state: GeometryDecoderState,
    ) -> tuple[torch.Tensor, GeometryDecoderState]:
        structure_agg, structure_relation = self.structure_update(h, state.structure, state.geometry_mask)
        interface_agg, interface_relation = self.interface_update(h, state.interface, state.geometry_mask)

        mask_features = torch.stack([state.node_mask, state.geometry_mask], dim=-1).to(dtype=h.dtype)
        node_delta = self.node_update(torch.cat([h, structure_agg, interface_agg, mask_features], dim=-1))
        updated_h = self.node_norm(h + node_delta)
        updated_h = updated_h * state.node_mask.unsqueeze(-1).to(dtype=updated_h.dtype)

        local_input = torch.cat(
            [
                updated_h,
                structure_agg,
                interface_agg,
                state.local_atoms.reshape(state.local_atoms.shape[0], state.local_atoms.shape[1], 12),
                mask_features,
            ],
            dim=-1,
        )
        local_delta = self.local_update(local_input).reshape(*state.local_atoms.shape)
        local_atoms = state.local_atoms + local_delta
        local_atoms = torch.where(state.local_mask.unsqueeze(-1), local_atoms, torch.zeros_like(local_atoms))

        updated_state = GeometryDecoderState(
            local_atoms=local_atoms,
            local_mask=state.local_mask,
            node_mask=state.node_mask,
            geometry_mask=state.geometry_mask,
            structure=structure_relation,
            interface=interface_relation,
        )
        return updated_h, updated_state


class IterativeGeometricDecoder(nn.Module):
    def __init__(self, node_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([IterativeGeometryBlock(node_dim=node_dim) for _ in range(num_layers)])

    def forward(
        self,
        q: torch.Tensor,
        state: GeometryDecoderState,
    ) -> GeometryDecoderState:
        h = q * state.node_mask.unsqueeze(-1).to(dtype=q.dtype)
        current = state
        for layer in self.layers:
            h, current = layer(h, current)
        return current


class _SparseFeatureBlock(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int = 28):
        super().__init__()
        input_dim = node_dim * 2 + edge_dim + 3
        self.structure_mlp = nn.Sequential(
            nn.Linear(input_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )
        self.interface_mlp = nn.Sequential(
            nn.Linear(input_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.node_update = nn.Sequential(
            nn.Linear(node_dim * 3 + 1, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )

    def _aggregate(
        self,
        h: torch.Tensor,
        relation: NeighborList,
        edge_feature: torch.Tensor,
        node_mask: torch.Tensor,
        relation_value: float,
        mlp: nn.Module,
    ) -> torch.Tensor:
        if relation.indices.shape[-1] == 0:
            return torch.zeros_like(h)

        target_h = batched_neighbor_gather(h, relation.indices)
        source_h = h.unsqueeze(-2).expand_as(target_h)
        target_mask = batched_neighbor_gather(node_mask.unsqueeze(-1), relation.indices).to(dtype=h.dtype)
        relation_flag = torch.full(
            (*relation.indices.shape, 1),
            relation_value,
            dtype=h.dtype,
            device=h.device,
        )
        message_input = torch.cat(
            [
                source_h,
                target_h,
                edge_feature,
                relation_flag,
                node_mask.unsqueeze(-1).unsqueeze(-1).to(dtype=h.dtype).expand(*relation.indices.shape, 1),
                target_mask,
            ],
            dim=-1,
        )
        valid_mask = relation.mask & node_mask.unsqueeze(-1) & batched_neighbor_gather(node_mask.unsqueeze(-1), relation.indices).squeeze(-1)
        message = mlp(message_input)
        message = torch.where(valid_mask.unsqueeze(-1), message, torch.zeros_like(message))
        denom = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype=h.dtype)
        return message.sum(dim=-2) / denom

    def forward(
        self,
        h: torch.Tensor,
        structure_neighbors: NeighborList,
        structure_edge: torch.Tensor,
        interface_neighbors: NeighborList,
        interface_edge: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        structure_agg = self._aggregate(h, structure_neighbors, structure_edge, node_mask, 0.0, self.structure_mlp)
        interface_agg = self._aggregate(h, interface_neighbors, interface_edge, node_mask, 1.0, self.interface_mlp)
        delta = self.node_update(
            torch.cat([h, structure_agg, interface_agg, node_mask.unsqueeze(-1).to(dtype=h.dtype)], dim=-1)
        )
        updated = self.norm(h + delta)
        return updated * node_mask.unsqueeze(-1).to(dtype=updated.dtype)


class SparseFeatureDecoder(nn.Module):
    def __init__(self, node_dim: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([_SparseFeatureBlock(node_dim=node_dim) for _ in range(num_layers)])

    def forward(
        self,
        q: torch.Tensor,
        batch: dict,
        neighbors: ComplexNeighborLists | None = None,
    ) -> torch.Tensor:
        pos_atoms = batch["pos_atoms"]
        device = pos_atoms.device
        batch_size, length, atom_count = pos_atoms.shape[:3]

        node_mask = _prepare_mask(batch.get("node_mask"), (batch_size, length), device, default=True)
        mask_atoms_value = batch.get("mask_atoms")
        if mask_atoms_value is None:
            mask_atoms = torch.ones((batch_size, length, atom_count), dtype=torch.bool, device=device)
        else:
            mask_atoms = mask_atoms_value.to(device=device, dtype=torch.bool)

        if neighbors is None:
            neighbors = build_neighbor_lists(batch)

        frame, frame_valid = construct_local_frames(pos_atoms, mask_atoms)
        ca_positions = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]
        structure_edge = directed_edge_geometry(frame, frame_valid, ca_positions, neighbors.structure, node_mask=node_mask)
        interface_edge = directed_edge_geometry(frame, frame_valid, ca_positions, neighbors.interface, node_mask=node_mask)

        h = q * node_mask.unsqueeze(-1).to(dtype=q.dtype)
        for layer in self.layers:
            h = layer(
                h,
                structure_neighbors=neighbors.structure,
                structure_edge=structure_edge,
                interface_neighbors=neighbors.interface,
                interface_edge=interface_edge,
                node_mask=node_mask,
            )
        return h
