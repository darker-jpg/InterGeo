from __future__ import annotations

from collections.abc import Sequence

import torch

from ppi_ddg.constants import ATOM_TO_INDEX


def safe_normalize(value: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize the last dimension while preserving exact zeros.

    Args:
        value: Tensor with a final vector dimension of size 3.
        eps: Minimum norm treated as numerically safe.

    Returns:
        Tensor with the same shape and dtype as `value`.
    """
    if value.dtype in (torch.float16, torch.bfloat16):
        work = value.to(dtype=torch.float32)
        norm = work.norm(dim=-1, keepdim=True)
        zero = torch.zeros_like(work)
        normalized = torch.where(norm > eps, work / norm.clamp_min(eps), zero)
        return normalized.to(dtype=value.dtype)

    norm = value.norm(dim=-1, keepdim=True)
    return torch.where(norm > eps, value / norm.clamp_min(eps), torch.zeros_like(value))


def rigid_transform(
    pos_atoms: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply a rigid transform to atom coordinates.

    Shapes:
        pos_atoms: [..., 3]
        rotation: [3, 3] or [B, 3, 3]
        translation: [3] or [B, 3]

    Broadcasting follows the batch dimensions of `pos_atoms`.
    """
    if rotation.ndim == 2:
        transformed = torch.einsum("...j,ij->...i", pos_atoms, rotation)
    elif rotation.ndim == 3:
        if pos_atoms.shape[0] != rotation.shape[0]:
            raise ValueError("batched rotations must match the coordinate batch size")
        transformed = torch.einsum("b...j,bij->b...i", pos_atoms, rotation)
    else:
        raise ValueError("rotation must have shape [3, 3] or [B, 3, 3]")

    if translation.ndim == 1:
        return transformed + translation
    if translation.ndim == 2:
        if transformed.shape[0] != translation.shape[0]:
            raise ValueError("batched translations must match the coordinate batch size")
        shape = (translation.shape[0],) + (1,) * (transformed.ndim - 2) + (3,)
        return transformed + translation.reshape(shape)
    raise ValueError("translation must have shape [3] or [B, 3]")


def construct_local_frames(
    pos_atoms: torch.Tensor,
    mask_atoms: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build residue-local frames from N, CA, and C atoms.

    Shapes:
        pos_atoms: [B, L, A, 3]
        mask_atoms: [B, L, A]

    Returns:
        frame: [B, L, 3, 3] with rows [u, n, v]
        frame_valid: [B, L] boolean mask
    """
    ca = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]
    n = pos_atoms[:, :, ATOM_TO_INDEX["N"]]
    c = pos_atoms[:, :, ATOM_TO_INDEX["C"]]
    valid = mask_atoms.to(dtype=torch.bool)
    n_vec = n - ca
    c_vec = c - ca
    n_norm = n_vec.norm(dim=-1)
    c_norm = c_vec.norm(dim=-1)
    u = safe_normalize(n_vec, eps=eps)
    t = safe_normalize(c_vec, eps=eps)
    cross_ut = torch.cross(u, t, dim=-1)
    frame_valid = valid[:, :, ATOM_TO_INDEX["N"]]
    frame_valid = frame_valid & valid[:, :, ATOM_TO_INDEX["CA"]]
    frame_valid = frame_valid & valid[:, :, ATOM_TO_INDEX["C"]]
    frame_valid = frame_valid & (n_norm > eps)
    frame_valid = frame_valid & (c_norm > eps)
    frame_valid = frame_valid & (cross_ut.norm(dim=-1) > eps)

    normal = safe_normalize(cross_ut, eps=eps)
    v = torch.cross(normal, u, dim=-1)
    frame = torch.stack([u, normal, v], dim=-2)
    frame = torch.where(frame_valid.unsqueeze(-1).unsqueeze(-1), frame, torch.zeros_like(frame))
    return frame, frame_valid


def to_local(frame: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Project global vectors into a local row-frame."""
    return torch.einsum("blij,bl...j->bl...i", frame, vector)


def local_atom_targets(
    pos_atoms: torch.Tensor,
    mask_atoms: torch.Tensor,
    frame: torch.Tensor,
    frame_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local N/C/O/CB targets and a matching validity mask."""
    ca = pos_atoms[:, :, ATOM_TO_INDEX["CA"]].unsqueeze(2)
    local_atoms = to_local(frame, pos_atoms[:, :, [ATOM_TO_INDEX[a] for a in ("N", "C", "O", "CB")]] - ca)
    atom_valid = mask_atoms.to(dtype=torch.bool)
    local_mask = torch.stack(
        [
            atom_valid[:, :, ATOM_TO_INDEX["N"]],
            atom_valid[:, :, ATOM_TO_INDEX["C"]],
            atom_valid[:, :, ATOM_TO_INDEX["O"]],
            atom_valid[:, :, ATOM_TO_INDEX["CB"]],
        ],
        dim=-1,
    )
    local_mask = local_mask & frame_valid.unsqueeze(-1)
    local_atoms = torch.where(local_mask.unsqueeze(-1), local_atoms, torch.zeros_like(local_atoms))
    return local_atoms, local_mask


def _angle_sin_cos(
    first: torch.Tensor,
    center: torch.Tensor,
    last: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sin, cos, and geometric validity for a bond angle."""
    left = first - center
    right = last - center
    left_norm = left.norm(dim=-1)
    right_norm = right.norm(dim=-1)
    left_unit = safe_normalize(left, eps=eps)
    right_unit = safe_normalize(right, eps=eps)
    cross_lr = torch.cross(left_unit, right_unit, dim=-1)
    cosine = (left_unit * right_unit).sum(dim=-1).clamp(-1.0, 1.0)
    sine = cross_lr.norm(dim=-1)
    valid = (left_norm > eps) & (right_norm > eps) & (cross_lr.norm(dim=-1) > eps)
    return sine, cosine, valid


def _dihedral(
    first: torch.Tensor,
    second: torch.Tensor,
    third: torch.Tensor,
    fourth: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sin, cos, and geometric validity for a dihedral angle."""
    b0 = second - first
    b1 = third - second
    b2 = fourth - third
    b0_norm = b0.norm(dim=-1)
    b1_norm = b1.norm(dim=-1)
    b2_norm = b2.norm(dim=-1)
    b1_unit = safe_normalize(b1, eps=eps)
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    n1_norm = n1.norm(dim=-1)
    n2_norm = n2.norm(dim=-1)
    n1_unit = safe_normalize(n1, eps=eps)
    n2_unit = safe_normalize(n2, eps=eps)
    cos = (n1_unit * n2_unit).sum(dim=-1).clamp(-1.0, 1.0)
    sin_mag = torch.cross(n1_unit, n2_unit, dim=-1).norm(dim=-1)
    sign = torch.sign((torch.cross(n1_unit, n2_unit, dim=-1) * b1_unit).sum(dim=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    sin = sin_mag * sign
    valid = (
        (b0_norm > eps)
        & (b1_norm > eps)
        & (b2_norm > eps)
        & (n1_norm > eps)
        & (n2_norm > eps)
    )
    return sin, cos, valid


def _pad_chain_ids(chain_ids: Sequence[Sequence[str]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad chain metadata on CPU and move once to `device`."""
    batch = len(chain_ids)
    max_len = max(len(row) for row in chain_ids)
    ids_cpu = torch.full((batch, max_len), -1, dtype=torch.long)
    counts_cpu = torch.zeros((batch, max_len), dtype=torch.bool)
    for batch_idx, row in enumerate(chain_ids):
        chain_map: dict[str, int] = {}
        for res_idx, chain in enumerate(row):
            if chain not in chain_map:
                chain_map[chain] = len(chain_map)
            ids_cpu[batch_idx, res_idx] = chain_map[chain]
            counts_cpu[batch_idx, res_idx] = True
    return ids_cpu.to(device=device), counts_cpu.to(device=device)


def _pad_resseq(resseq: Sequence[torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad residue numbers on CPU and move once to `device`."""
    batch = len(resseq)
    max_len = max(int(row.shape[0]) for row in resseq)
    values_cpu = torch.zeros((batch, max_len), dtype=torch.long)
    counts_cpu = torch.zeros((batch, max_len), dtype=torch.bool)
    for batch_idx, row in enumerate(resseq):
        row_cpu = torch.as_tensor(row, dtype=torch.long, device="cpu")
        values_cpu[batch_idx, : row_cpu.shape[0]] = row_cpu
        counts_cpu[batch_idx, : row_cpu.shape[0]] = True
    return values_cpu.to(device=device), counts_cpu.to(device=device)

def chain_aware_angle_targets(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Build seven chain-aware local geometry targets.

    Returns:
        target: [B, L, 7]
        valid: [B, L, 7]
    """
    pos_atoms = batch["pos_atoms"]
    mask_atoms = batch["mask_atoms"].to(dtype=torch.bool, device=pos_atoms.device)
    node_mask = batch.get("node_mask")
    if node_mask is None:
        node_mask = torch.ones(pos_atoms.shape[:2], dtype=torch.bool, device=pos_atoms.device)
    else:
        node_mask = node_mask.to(dtype=torch.bool, device=pos_atoms.device)

    chain_ids = batch["chain_ids"]
    resseq = batch["resseq"]
    if isinstance(chain_ids, torch.Tensor):
        chain_tensor = chain_ids.to(device=pos_atoms.device, dtype=torch.long)
        chain_present = chain_tensor >= 0
    else:
        chain_tensor, chain_present = _pad_chain_ids(chain_ids, pos_atoms.device)
    if isinstance(resseq, torch.Tensor):
        resseq_tensor = resseq.to(device=pos_atoms.device, dtype=torch.long)
        resseq_present = torch.ones_like(resseq_tensor, dtype=torch.bool)
    else:
        resseq_tensor, resseq_present = _pad_resseq(resseq, pos_atoms.device)

    n_batch, n_res = pos_atoms.shape[:2]
    target = pos_atoms.new_zeros(n_batch, n_res, 7)
    valid = torch.zeros((n_batch, n_res, 7), dtype=torch.bool, device=pos_atoms.device)

    n_atom = pos_atoms[:, :, ATOM_TO_INDEX["N"]]
    ca_atom = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]
    c_atom = pos_atoms[:, :, ATOM_TO_INDEX["C"]]

    central_valid = (
        node_mask
        & mask_atoms[:, :, ATOM_TO_INDEX["N"]]
        & mask_atoms[:, :, ATOM_TO_INDEX["CA"]]
        & mask_atoms[:, :, ATOM_TO_INDEX["C"]]
    )
    angle_sin, angle_cos, angle_valid = _angle_sin_cos(n_atom, ca_atom, c_atom)
    central_valid = central_valid & angle_valid
    target[:, :, 0] = torch.where(central_valid, angle_sin, target[:, :, 0])
    target[:, :, 1] = torch.where(central_valid, angle_cos, target[:, :, 1])
    valid[:, :, 0] = central_valid
    valid[:, :, 1] = central_valid

    if n_res > 1:
        same_chain_prev = (
            chain_present[:, 1:]
            & chain_present[:, :-1]
            & (chain_tensor[:, 1:] == chain_tensor[:, :-1])
            & resseq_present[:, 1:]
            & resseq_present[:, :-1]
            & (resseq_tensor[:, 1:] == resseq_tensor[:, :-1] + 1)
            & node_mask[:, 1:]
            & node_mask[:, :-1]
        )
        phi_sin, phi_cos, phi_geom_valid = _dihedral(c_atom[:, :-1], n_atom[:, 1:], ca_atom[:, 1:], c_atom[:, 1:])
        phi_valid = same_chain_prev & central_valid[:, 1:] & mask_atoms[:, :-1, ATOM_TO_INDEX["C"]] & phi_geom_valid
        target[:, 1:, 2] = torch.where(phi_valid, phi_sin, target[:, 1:, 2])
        target[:, 1:, 3] = torch.where(phi_valid, phi_cos, target[:, 1:, 3])
        valid[:, 1:, 2] = phi_valid
        valid[:, 1:, 3] = phi_valid

        same_chain_next = (
            chain_present[:, :-1]
            & chain_present[:, 1:]
            & (chain_tensor[:, :-1] == chain_tensor[:, 1:])
            & resseq_present[:, :-1]
            & resseq_present[:, 1:]
            & (resseq_tensor[:, 1:] == resseq_tensor[:, :-1] + 1)
            & node_mask[:, :-1]
            & node_mask[:, 1:]
        )
        psi_sin, psi_cos, psi_geom_valid = _dihedral(n_atom[:, :-1], ca_atom[:, :-1], c_atom[:, :-1], n_atom[:, 1:])
        psi_valid = same_chain_next & central_valid[:, :-1] & mask_atoms[:, 1:, ATOM_TO_INDEX["N"]] & psi_geom_valid
        target[:, :-1, 4] = torch.where(psi_valid, psi_sin, target[:, :-1, 4])
        target[:, :-1, 5] = torch.where(psi_valid, psi_cos, target[:, :-1, 5])
        valid[:, :-1, 4] = psi_valid
        valid[:, :-1, 5] = psi_valid

    if n_res > 2:
        left_chain = (
            chain_present[:, 1:-1]
            & chain_present[:, :-2]
            & (chain_tensor[:, 1:-1] == chain_tensor[:, :-2])
            & resseq_present[:, 1:-1]
            & resseq_present[:, :-2]
            & (resseq_tensor[:, 1:-1] == resseq_tensor[:, :-2] + 1)
            & node_mask[:, 1:-1]
            & node_mask[:, :-2]
        )
        right_chain = (
            chain_present[:, 1:-1]
            & chain_present[:, 2:]
            & (chain_tensor[:, 1:-1] == chain_tensor[:, 2:])
            & resseq_present[:, 1:-1]
            & resseq_present[:, 2:]
            & (resseq_tensor[:, 2:] == resseq_tensor[:, 1:-1] + 1)
            & node_mask[:, 1:-1]
            & node_mask[:, 2:]
        )
        _, ca_cos, ca_geom_valid = _angle_sin_cos(ca_atom[:, :-2], ca_atom[:, 1:-1], ca_atom[:, 2:])
        ca_valid = left_chain & right_chain & mask_atoms[:, :-2, ATOM_TO_INDEX["CA"]] & mask_atoms[:, 1:-1, ATOM_TO_INDEX["CA"]] & mask_atoms[:, 2:, ATOM_TO_INDEX["CA"]] & ca_geom_valid
        target[:, 1:-1, 6] = torch.where(ca_valid, ca_cos, target[:, 1:-1, 6])
        valid[:, 1:-1, 6] = ca_valid

    return target, valid
