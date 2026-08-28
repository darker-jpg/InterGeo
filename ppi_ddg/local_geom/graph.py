from __future__ import annotations

from dataclasses import dataclass

import torch

from ppi_ddg.constants import ATOM_TO_INDEX


@dataclass(frozen=True)
class NeighborList:
    indices: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class ComplexNeighborLists:
    structure: NeighborList
    interface: NeighborList


def _pad_string_ids(rows, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.full((len(rows), length), -1, dtype=torch.long)
    present = torch.zeros((len(rows), length), dtype=torch.bool)
    for batch_idx, row in enumerate(rows):
        mapping: dict[str, int] = {}
        limit = min(len(row), length)
        for item_idx in range(limit):
            raw = row[item_idx]
            key = "" if raw is None else str(raw)
            if not key:
                continue
            if key not in mapping:
                mapping[key] = len(mapping)
            values[batch_idx, item_idx] = mapping[key]
            present[batch_idx, item_idx] = True
    return values.to(device=device), present.to(device=device)


def _pad_tensor_rows(rows, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.zeros((len(rows), length), dtype=torch.long)
    present = torch.zeros((len(rows), length), dtype=torch.bool)
    for batch_idx, row in enumerate(rows):
        row_tensor = torch.as_tensor(row, dtype=torch.long, device="cpu")
        limit = min(int(row_tensor.shape[0]), length)
        values[batch_idx, :limit] = row_tensor[:limit]
        present[batch_idx, :limit] = True
    return values.to(device=device), present.to(device=device)


def _prepare_ids(ids, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(ids, torch.Tensor):
        values = ids.to(device=device, dtype=torch.long)
        present = values >= 0
        if values.shape[1] < length:
            pad = torch.full((values.shape[0], length - values.shape[1]), -1, dtype=torch.long, device=device)
            values = torch.cat([values, pad], dim=1)
            present = torch.cat([present, torch.zeros((present.shape[0], length - present.shape[1]), dtype=torch.bool, device=device)], dim=1)
        elif values.shape[1] > length:
            values = values[:, :length]
            present = present[:, :length]
        return values, present
    return _pad_string_ids(ids, length, device)


def _prepare_resseq(resseq, length: int, device: torch.device) -> torch.Tensor:
    if isinstance(resseq, torch.Tensor):
        values = resseq.to(device=device, dtype=torch.long)
        if values.shape[1] < length:
            pad = torch.zeros((values.shape[0], length - values.shape[1]), dtype=torch.long, device=device)
            values = torch.cat([values, pad], dim=1)
        elif values.shape[1] > length:
            values = values[:, :length]
        return values
    values, _ = _pad_tensor_rows(resseq, length, device)
    return values


def _prepare_mask(batch: dict, key: str, shape: tuple[int, int], device: torch.device, default: bool) -> torch.Tensor:
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


def _neighbor_list_from_scores(scores: torch.Tensor, valid_mask: torch.Tensor, k: int) -> NeighborList:
    batch, length, candidate_count = scores.shape
    device = scores.device
    if k <= 0:
        empty_indices = torch.zeros((batch, length, 0), dtype=torch.long, device=device)
        empty_mask = torch.zeros((batch, length, 0), dtype=torch.bool, device=device)
        return NeighborList(indices=empty_indices, mask=empty_mask)
    if candidate_count == 0:
        indices = torch.zeros((batch, length, k), dtype=torch.long, device=device)
        mask = torch.zeros((batch, length, k), dtype=torch.bool, device=device)
        return NeighborList(indices=indices, mask=mask)

    select_k = min(k, candidate_count)
    top_scores, top_indices = torch.topk(scores, k=select_k, dim=-1, largest=False)
    selected_mask = torch.gather(valid_mask, -1, top_indices) & torch.isfinite(top_scores)
    selected_indices = torch.where(selected_mask, top_indices, torch.zeros_like(top_indices))
    if select_k < k:
        pad_shape = (batch, length, k - select_k)
        selected_indices = torch.cat(
            [selected_indices, torch.zeros(pad_shape, dtype=torch.long, device=device)],
            dim=-1,
        )
        selected_mask = torch.cat(
            [selected_mask, torch.zeros(pad_shape, dtype=torch.bool, device=device)],
            dim=-1,
        )
    return NeighborList(indices=selected_indices, mask=selected_mask)


def build_neighbor_lists(
    batch: dict,
    structure_cutoff: float = 12.0,
    interface_cutoff: float = 10.0,
    structure_k: int = 16,
    interface_k: int = 16,
) -> ComplexNeighborLists:
    pos_atoms = batch["pos_atoms"]
    device = pos_atoms.device
    batch_size, length = pos_atoms.shape[:2]

    node_mask = _prepare_mask(batch, "node_mask", (batch_size, length), device, default=True)
    interface_mask = _prepare_mask(batch, "interface_mask", (batch_size, length), device, default=False)
    ca_mask = batch.get("mask_atoms")
    if ca_mask is None:
        ca_present = torch.ones((batch_size, length), dtype=torch.bool, device=device)
    else:
        ca_present = ca_mask.to(device=device, dtype=torch.bool)[:, :, ATOM_TO_INDEX["CA"]]

    valid_residue = node_mask & ca_present

    mol_ids = batch.get("mol_index")
    if mol_ids is None:
        mol_ids = batch.get("mol_ids")
    if mol_ids is None:
        mol_ids = [[""] * length for _ in range(batch_size)]
    mol_tensor, mol_present = _prepare_ids(mol_ids, length, device)
    valid_residue = valid_residue & mol_present

    ca = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]
    distance = torch.cdist(ca, ca)
    self_mask = torch.eye(length, dtype=torch.bool, device=device).unsqueeze(0)

    same_mol = mol_present[:, :, None] & mol_present[:, None, :] & (mol_tensor[:, :, None] == mol_tensor[:, None, :])
    same_mol = same_mol & valid_residue[:, :, None] & valid_residue[:, None, :] & ~self_mask

    chain_ids = batch.get("chain_index")
    if chain_ids is None:
        chain_ids = batch.get("chain_ids")
    seq_adj = torch.zeros_like(distance, dtype=torch.bool)
    if chain_ids is not None:
        chain_tensor, chain_present = _prepare_ids(chain_ids, length, device)
        resseq = batch.get("resseq")
        if resseq is None:
            resseq_tensor = torch.zeros((batch_size, length), dtype=torch.long, device=device)
        else:
            resseq_tensor = _prepare_resseq(resseq, length, device)
        same_chain = chain_present[:, :, None] & chain_present[:, None, :] & (chain_tensor[:, :, None] == chain_tensor[:, None, :])
        offset = (resseq_tensor[:, :, None] - resseq_tensor[:, None, :]).abs()
        seq_adj = same_chain & (offset == 1)
        seq_adj = seq_adj | (same_chain & (offset == 2))
        seq_adj = seq_adj & same_mol

    structure_valid = same_mol & ((distance <= structure_cutoff) | seq_adj)
    if structure_valid.any():
        structure_max = distance[structure_valid].max()
        structure_bonus = structure_max + 1.0
    else:
        structure_bonus = torch.tensor(1.0, device=device, dtype=distance.dtype)
    structure_scores = distance - seq_adj.to(dtype=distance.dtype) * structure_bonus
    structure_scores = torch.where(structure_valid, structure_scores, torch.full_like(structure_scores, float("inf")))

    different_mol = mol_present[:, :, None] & mol_present[:, None, :] & (mol_tensor[:, :, None] != mol_tensor[:, None, :])
    interface_valid = (
        valid_residue[:, :, None]
        & valid_residue[:, None, :]
        & interface_mask[:, :, None]
        & interface_mask[:, None, :]
        & different_mol
        & (distance < interface_cutoff)
        & ~self_mask
    )
    interface_scores = torch.where(interface_valid, distance, torch.full_like(distance, float("inf")))

    structure = _neighbor_list_from_scores(structure_scores, structure_valid, structure_k)
    interface = _neighbor_list_from_scores(interface_scores, interface_valid, interface_k)
    return ComplexNeighborLists(structure=structure, interface=interface)


def batched_neighbor_gather(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("value must have shape [B, L, ...]")
    if indices.ndim != 3:
        raise ValueError("indices must have shape [B, L, K]")
    if value.shape[0] != indices.shape[0] or value.shape[1] != indices.shape[1]:
        raise ValueError("value and indices must agree on batch and residue dimensions")

    batch = value.shape[0]
    gather_batch = torch.arange(batch, device=value.device).view(batch, 1, 1).expand_as(indices)
    gather_indices = indices.to(device=value.device, dtype=torch.long)
    return value[gather_batch, gather_indices]
