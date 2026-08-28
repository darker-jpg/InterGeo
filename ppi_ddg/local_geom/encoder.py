from __future__ import annotations

import math

import torch
import torch.nn as nn

from ppi_ddg.constants import ATOM_TO_INDEX
from ppi_ddg.local_geom.geometry import chain_aware_angle_targets, construct_local_frames, safe_normalize, to_local
from ppi_ddg.local_geom.graph import ComplexNeighborLists, NeighborList, batched_neighbor_gather, build_neighbor_lists

__all__ = [
    "LocalGeometricBlock",
    "LocalGeometricEncoder",
    "SparseGeometricAttention",
    "directed_edge_geometry",
    "masked_softmax",
]


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    """Softmax that returns all zeros when every entry is masked."""
    if logits.shape[dim] == 0:
        return torch.zeros_like(logits)

    mask = mask.to(dtype=torch.bool, device=logits.device)
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    max_logits = masked_logits.max(dim=dim, keepdim=True).values
    max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
    stabilized = torch.exp(masked_logits - max_logits)
    stabilized = torch.where(mask, stabilized, torch.zeros_like(stabilized))
    denom = stabilized.sum(dim=dim, keepdim=True)
    return torch.where(denom > 0, stabilized / denom.clamp_min(eps), torch.zeros_like(stabilized))


def _rbf(distance: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    delta = distance.unsqueeze(-1) - centers
    return torch.exp(-0.5 * (delta / width) ** 2)


def directed_edge_geometry(
    frame: torch.Tensor,
    frame_valid: torch.Tensor,
    ca_positions: torch.Tensor,
    neighbors: NeighborList,
    node_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build directed edge geometry for a neighbor list.

    The returned tensor concatenates:
    - 16 RBF features for distance in [0, 20] Angstrom
    - 3 source-frame direction components
    - 9 relative-frame components
    All invalid edges are zeroed.
    """
    if neighbors.indices.shape[-1] == 0:
        edge_dim = 16 + 3 + 9
        return ca_positions.new_zeros((*neighbors.indices.shape, edge_dim))

    source_mask = frame_valid.to(dtype=torch.bool, device=ca_positions.device)
    if node_mask is not None:
        source_mask = source_mask & node_mask.to(dtype=torch.bool, device=ca_positions.device)

    target_mask = batched_neighbor_gather(source_mask, neighbors.indices)
    edge_mask = neighbors.mask.to(dtype=torch.bool, device=ca_positions.device) & source_mask.unsqueeze(-1) & target_mask

    source_ca = ca_positions
    target_ca = batched_neighbor_gather(ca_positions, neighbors.indices)
    displacement = target_ca - source_ca.unsqueeze(-2)
    distance = displacement.norm(dim=-1)

    local_direction = to_local(frame, displacement)
    local_direction = safe_normalize(local_direction)

    target_frame = batched_neighbor_gather(frame, neighbors.indices)
    relative_frame = torch.matmul(frame.unsqueeze(-3), target_frame.transpose(-1, -2))
    relative_frame = relative_frame.reshape(*relative_frame.shape[:-2], 9)

    centers = torch.linspace(0.0, 20.0, 16, device=ca_positions.device, dtype=ca_positions.dtype)
    if centers.numel() > 1:
        width = float((centers[1] - centers[0]).item())
    else:
        width = 1.0
    rbf = _rbf(distance, centers, width)
    edge = torch.cat([rbf, local_direction, relative_frame], dim=-1)
    return torch.where(edge_mask.unsqueeze(-1), edge, torch.zeros_like(edge))


class SparseGeometricAttention(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, num_heads: int):
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError("node_dim must be divisible by num_heads")
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim + node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim + node_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        h: torch.Tensor,
        neighbors: NeighborList,
        edge_geometry: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        if neighbors.indices.shape[-1] == 0:
            return torch.zeros_like(h)

        target_h = batched_neighbor_gather(h, neighbors.indices)
        edge_h = self.edge_mlp(edge_geometry)
        kv_input = torch.cat([target_h, edge_h], dim=-1)

        query = self.q_proj(h).view(*h.shape[:2], self.num_heads, self.head_dim)
        key = self.k_proj(kv_input).view(*kv_input.shape[:3], self.num_heads, self.head_dim)
        value = self.v_proj(kv_input).view(*kv_input.shape[:3], self.num_heads, self.head_dim)

        scores = (query.unsqueeze(2) * key).sum(dim=-1) * self.scale
        mask = neighbors.mask.unsqueeze(-1) & source_mask.unsqueeze(-1).unsqueeze(-1)
        any_neighbor = mask.any(dim=2)
        weights = masked_softmax(scores, mask, dim=2)
        attended = (weights.unsqueeze(-1) * value).sum(dim=2)
        attended = attended.reshape(h.shape[0], h.shape[1], self.node_dim)
        output = self.out_proj(attended)
        return output * any_neighbor.to(output.dtype)


class LocalGeometricBlock(nn.Module):
    def __init__(self, node_dim: int, num_heads: int):
        super().__init__()
        edge_dim = 16 + 3 + 9
        self.structure_attn = SparseGeometricAttention(node_dim=node_dim, edge_dim=edge_dim, num_heads=num_heads)
        self.interface_attn = SparseGeometricAttention(node_dim=node_dim, edge_dim=edge_dim, num_heads=num_heads)
        self.norm_structure = nn.LayerNorm(node_dim)
        self.norm_interface = nn.LayerNorm(node_dim)
        self.ff = nn.Sequential(
            nn.Linear(node_dim, node_dim * 2),
            nn.ReLU(),
            nn.Linear(node_dim * 2, node_dim),
        )
        self.norm_ff = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,
        frame: torch.Tensor,
        frame_valid: torch.Tensor,
        ca_positions: torch.Tensor,
        structure_neighbors: NeighborList,
        interface_neighbors: NeighborList,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        structure_edge = directed_edge_geometry(frame, frame_valid, ca_positions, structure_neighbors, node_mask=node_mask)
        h = self.norm_structure(h + self.structure_attn(h, structure_neighbors, structure_edge, node_mask))

        interface_edge = directed_edge_geometry(frame, frame_valid, ca_positions, interface_neighbors, node_mask=node_mask)
        h = self.norm_interface(h + self.interface_attn(h, interface_neighbors, interface_edge, node_mask))

        h = self.norm_ff(h + self.ff(h))
        return h * node_mask.unsqueeze(-1).to(h.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    summed = (values * weights).sum(dim=1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return summed / denom


class LocalGeometricEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        structure_cutoff: float = 12.0,
        interface_cutoff: float = 10.0,
        structure_k: int = 16,
        interface_k: int = 16,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.structure_cutoff = structure_cutoff
        self.interface_cutoff = interface_cutoff
        self.structure_k = structure_k
        self.interface_k = interface_k

        self.aa_embed = nn.Embedding(21, node_dim)
        self.angle_mlp = nn.Sequential(
            nn.Linear(7, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim),
        )
        self.blocks = nn.ModuleList([LocalGeometricBlock(node_dim=node_dim, num_heads=num_heads) for _ in range(num_blocks)])

    def _prepare_mask(self, batch: dict, key: str, shape: tuple[int, int], device: torch.device, default: bool) -> torch.Tensor:
        value = batch.get(key)
        if value is None:
            return torch.full(shape, default, dtype=torch.bool, device=device)
        tensor = value.to(device=device, dtype=torch.bool)
        if tensor.shape[1] < shape[1]:
            pad = torch.full((tensor.shape[0], shape[1] - tensor.shape[1]), default, dtype=torch.bool, device=device)
            tensor = torch.cat([tensor, pad], dim=1)
        elif tensor.shape[1] > shape[1]:
            tensor = tensor[:, : shape[1]]
        return tensor

    def forward(
        self,
        batch: dict,
        angle_features: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | None = None,
        neighbors: ComplexNeighborLists | None = None,
        node_bias: torch.Tensor | None = None,
        node_bias_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos_atoms = batch["pos_atoms"]
        device = pos_atoms.device
        batch_size, length = pos_atoms.shape[:2]
        atom_count = pos_atoms.shape[2]

        aa_value = batch.get("aa")
        if aa_value is None:
            aa = torch.zeros((batch_size, length), dtype=torch.long, device=device)
        else:
            aa = aa_value.to(device=device, dtype=torch.long)
        if aa.shape[1] < length:
            pad = torch.zeros((aa.shape[0], length - aa.shape[1]), dtype=torch.long, device=device)
            aa = torch.cat([aa, pad], dim=1)
        elif aa.shape[1] > length:
            aa = aa[:, :length]

        node_mask = self._prepare_mask(batch, "node_mask", (batch_size, length), device, default=True)
        interface_value = batch.get("interface_mask")
        interface_mask = self._prepare_mask(batch, "interface_mask", (batch_size, length), device, default=False)
        if interface_value is None:
            interface_mask = node_mask

        mask_atoms_value = batch.get("mask_atoms")
        if mask_atoms_value is None:
            mask_atoms = torch.ones((batch_size, length, atom_count), dtype=torch.bool, device=device)
        else:
            mask_atoms = mask_atoms_value.to(dtype=torch.bool, device=device)
        frame, frame_valid = construct_local_frames(pos_atoms, mask_atoms)
        ca_positions = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]

        if angle_features is None:
            angle_batch = dict(batch)
            angle_batch["mask_atoms"] = mask_atoms
            angle_features = chain_aware_angle_targets(angle_batch)
        if isinstance(angle_features, tuple):
            angle_values, angle_valid = angle_features
        else:
            angle_values = angle_features
            angle_valid = torch.ones(angle_values.shape[:2] + (angle_values.shape[-1],), dtype=torch.bool, device=device)
        angle_values = angle_values.to(device=device, dtype=pos_atoms.dtype)
        angle_valid = angle_valid.to(device=device, dtype=torch.bool)
        angle_values = torch.where(angle_valid, angle_values, torch.zeros_like(angle_values))

        if neighbors is None:
            neighbors = build_neighbor_lists(
                batch,
                structure_cutoff=self.structure_cutoff,
                interface_cutoff=self.interface_cutoff,
                structure_k=self.structure_k,
                interface_k=self.interface_k,
            )

        h = self.aa_embed(aa) + self.angle_mlp(angle_values)
        if node_bias is not None:
            bias = node_bias.to(device=device, dtype=h.dtype)
            if bias.shape[1] < length:
                pad = torch.zeros((bias.shape[0], length - bias.shape[1], bias.shape[2]), dtype=h.dtype, device=device)
                bias = torch.cat([bias, pad], dim=1)
            elif bias.shape[1] > length:
                bias = bias[:, :length]
            if node_bias_mask is None:
                bias_mask = node_mask
            else:
                bias_mask = node_bias_mask.to(device=device, dtype=torch.bool)
                if bias_mask.shape[1] < length:
                    pad = torch.zeros((bias_mask.shape[0], length - bias_mask.shape[1]), dtype=torch.bool, device=device)
                    bias_mask = torch.cat([bias_mask, pad], dim=1)
                elif bias_mask.shape[1] > length:
                    bias_mask = bias_mask[:, :length]
                bias_mask = bias_mask & node_mask
            h = h + bias * bias_mask.unsqueeze(-1).to(h.dtype)
        h = h * node_mask.unsqueeze(-1).to(h.dtype)

        for block in self.blocks:
            h = block(
                h,
                frame=frame,
                frame_valid=frame_valid,
                ca_positions=ca_positions,
                structure_neighbors=neighbors.structure,
                interface_neighbors=neighbors.interface,
                node_mask=node_mask & frame_valid,
            )

        interface = _masked_mean(h, interface_mask & node_mask)
        return h, interface
