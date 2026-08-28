from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppi_ddg.constants import ATOM_TO_INDEX
from ppi_ddg.local_geom.encoder import LocalGeometricEncoder, masked_softmax
from ppi_ddg.local_geom.geometry import (
    chain_aware_angle_targets,
    construct_local_frames,
    safe_normalize,
    to_local,
)
from ppi_ddg.local_geom.graph import (
    ComplexNeighborLists,
    NeighborList,
    _neighbor_list_from_scores,
    _prepare_mask as _graph_prepare_mask,
    batched_neighbor_gather,
)
from ppi_ddg.local_geom.pretrained_encoder import (
    FrozenLocalGeometryEncoder,
    valid_residue_mask,
)

if TYPE_CHECKING:
    from ppi_ddg.local_geom.mutation_context import MutationSetOutput


__all__ = [
    "BidirectionalInterfaceRelay",
    "GeometryCache",
    "InterfaceRelayOutput",
    "MutationSiteBatch",
    "MutationSiteMetadata",
    "MutationSpecificInterfaceContext",
    "attention_pool",
    "build_geometry_cache",
    "float32_cdist",
    "float32_distance_rbf",
    "gather_mutation_sites",
    "opposite_partner_mask",
    "prepare_collated_graph_cpu",
    "require_finite",
    "require_prepared_graph",
    "require_true",
    "run_encoder",
    "run_pretrained_encoder",
    "select_nearest_partner_mask",
    "validate_host_axis_identity",
]


_RESSEQ_KEY = "prepared_resseq"
_MUTATION_COUNTS_KEY = "mutation_counts"


def attention_pool(
    values: torch.Tensor,
    mask: torch.Tensor,
    gate: nn.Module,
) -> torch.Tensor:
    mask = mask.to(device=values.device, dtype=torch.bool)
    logits = gate(values).squeeze(-1)
    weights = masked_softmax(logits, mask, dim=1)
    pooled = (values * weights.unsqueeze(-1)).sum(dim=1)
    return torch.where(
        mask.any(dim=1, keepdim=True),
        pooled,
        torch.zeros_like(pooled),
    )


def _validate_positive(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_dropout(dropout: float) -> float:
    value = float(dropout)
    if value != 0.0:
        raise ValueError("dropout must be 0.0 for deterministic paired-state mappings")
    return 0.0


def require_finite(
    values: torch.Tensor,
    message: str,
    mask: torch.Tensor | None = None,
) -> None:
    if mask is not None:
        while mask.ndim < values.ndim:
            mask = mask.unsqueeze(-1)
        values = torch.where(mask, values, torch.zeros_like(values))
    condition = torch.isfinite(values).all()
    if values.device.type == "cuda":
        torch._assert_async(condition)
    elif not bool(condition):
        raise ValueError(message)


def require_true(condition: torch.Tensor, message: str) -> None:
    if condition.device.type == "cuda":
        torch._assert_async(condition)
    elif not bool(condition):
        raise ValueError(message)


def _host_axis_rows(
    graph: dict[str, Any],
    key: str,
    batch_size: int,
) -> tuple[tuple[Any, ...] | None, ...] | None:
    value = graph.get(key)
    if not isinstance(value, (list, tuple)):
        return None
    if (
        len(value) == batch_size
        and all(
            row is None or isinstance(row, (list, tuple))
            for row in value
        )
    ):
        rows = value
    elif batch_size == 1:
        rows = (value,)
    else:
        raise ValueError(f"{key} metadata does not match the graph batch axis")

    normalized: list[tuple[Any, ...] | None] = []
    for row in rows:
        if row is None:
            normalized.append(None)
        elif isinstance(row, (list, tuple)):
            normalized.append(tuple(row))
        else:
            raise ValueError(f"{key} metadata does not match the residue axis")
    return tuple(normalized)


def validate_host_axis_identity(
    wt_graph: dict[str, Any],
    mt_graph: dict[str, Any],
    *,
    key: str,
    batch_size: int,
    message: str,
) -> None:
    wt_rows = _host_axis_rows(wt_graph, key, batch_size)
    mt_rows = _host_axis_rows(mt_graph, key, batch_size)
    if (wt_rows is None) != (mt_rows is None):
        raise ValueError(f"{message}: {key} presence differs")
    if wt_rows is None:
        return
    if mt_rows is None:
        raise RuntimeError("unreachable stable identity state")
    for wt_row, mt_row in zip(wt_rows, mt_rows):
        if (wt_row is None) != (mt_row is None):
            raise ValueError(f"{message}: {key} presence differs")
        if wt_row is None:
            continue
        if mt_row is None:
            raise RuntimeError("unreachable stable identity row")
        if wt_row != mt_row:
            raise ValueError(message)


def float32_cdist(
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    left = left.to(dtype=torch.float32)
    right = right.to(dtype=torch.float32)
    with torch.autocast(device_type=left.device.type, enabled=False):
        return torch.cdist(left, right)


def float32_distance_rbf(
    distance: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    distance = distance.to(dtype=torch.float32)
    centers = centers.to(device=distance.device, dtype=torch.float32)
    with torch.autocast(device_type=distance.device.type, enabled=False):
        if centers.numel() <= 1:
            width = 1.0
        else:
            width = (centers[1] - centers[0]).clamp_min(1e-6)
        delta = (distance.unsqueeze(-1) - centers) / width
        return torch.exp(-0.5 * delta.square())


@dataclass(frozen=True)
class GeometryCache:
    neighbors: ComplexNeighborLists
    frame: torch.Tensor
    frame_valid: torch.Tensor
    ca: torch.Tensor
    ca_distance: torch.Tensor
    angle_values: torch.Tensor
    angle_valid: torch.Tensor
    node_mask: torch.Tensor
    interface_mask: torch.Tensor
    structure_edge: torch.Tensor
    interface_edge: torch.Tensor
    neighbor_hyperparameters: tuple[float, float, int, int]


def _fast_prepare_ids(
    ids: Any,
    *,
    batch_size: int,
    length: int,
    device: torch.device,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(ids, torch.Tensor):
        values = ids.to(device=device, dtype=torch.long)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[0] != batch_size:
            raise ValueError(f"{name} must match the graph batch axis")
        present = values >= 0
        if values.shape[1] < length:
            missing = length - values.shape[1]
            values = torch.cat(
                [
                    values,
                    torch.full(
                        (batch_size, missing),
                        -1,
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
            present = torch.cat(
                [
                    present,
                    torch.zeros(
                        (batch_size, missing),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )
        elif values.shape[1] > length:
            values = values[:, :length]
            present = present[:, :length]
        return values, present

    if not isinstance(ids, (list, tuple)) or len(ids) != batch_size:
        raise ValueError(f"{name} must match the graph batch axis")
    encoded_rows: list[list[int]] = []
    present_rows: list[list[bool]] = []
    for row in ids:
        if isinstance(row, torch.Tensor):
            if row.device.type != "cpu":
                raise ValueError(
                    f"{name} list rows must remain on CPU or be padded tensors"
                )
            raw_row = row.detach().tolist()
        elif isinstance(row, (list, tuple)):
            raw_row = list(row)
        else:
            raise ValueError(f"{name} must match the residue axis")

        mapping: dict[str, int] = {}
        encoded = [-1] * length
        present = [False] * length
        for residue_index, raw in enumerate(raw_row[:length]):
            key = "" if raw is None else str(raw)
            if not key:
                continue
            if key not in mapping:
                mapping[key] = len(mapping)
            encoded[residue_index] = mapping[key]
            present[residue_index] = True
        encoded_rows.append(encoded)
        present_rows.append(present)
    return (
        torch.tensor(
            encoded_rows,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            present_rows,
            dtype=torch.bool,
            device=device,
        ),
    )


def _fast_prepare_resseq(
    resseq: Any,
    *,
    batch_size: int,
    length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(resseq, torch.Tensor):
        values = resseq.to(device=device, dtype=torch.long)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[0] != batch_size:
            raise ValueError("resseq must match the graph batch axis")
        present = torch.ones_like(values, dtype=torch.bool)
        if values.shape[1] < length:
            missing = length - values.shape[1]
            values = torch.cat(
                [
                    values,
                    torch.zeros(
                        (batch_size, missing),
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )
            present = torch.cat(
                [
                    present,
                    torch.zeros(
                        (batch_size, missing),
                        dtype=torch.bool,
                        device=device,
                    ),
                ],
                dim=1,
            )
        elif values.shape[1] > length:
            values = values[:, :length]
            present = present[:, :length]
        return values, present

    if not isinstance(resseq, (list, tuple)) or len(resseq) != batch_size:
        raise ValueError("resseq must match the graph batch axis")
    padded_rows: list[list[int]] = []
    present_rows: list[list[bool]] = []
    for row in resseq:
        if isinstance(row, torch.Tensor):
            if row.device.type != "cpu":
                raise ValueError(
                    "resseq list rows must remain on CPU or be padded tensors"
                )
            raw_row = row.detach().tolist()
        elif isinstance(row, (list, tuple)):
            raw_row = list(row)
        else:
            raise ValueError("resseq must match the residue axis")
        padded = [0] * length
        present = [False] * length
        for residue_index, raw in enumerate(raw_row[:length]):
            padded[residue_index] = int(raw)
            present[residue_index] = True
        padded_rows.append(padded)
        present_rows.append(present)
    return (
        torch.tensor(
            padded_rows,
            dtype=torch.long,
            device=device,
        ),
        torch.tensor(
            present_rows,
            dtype=torch.bool,
            device=device,
        ),
    )


def prepare_collated_graph_cpu(
    graph: dict[str, Any],
) -> dict[str, Any]:
    """Prepare padded paired-state metadata before a collated graph leaves CPU."""
    if not isinstance(graph, dict):
        raise TypeError("paired-state collated graph must be a dictionary")
    pos_atoms = graph.get("pos_atoms")
    if (
        not isinstance(pos_atoms, torch.Tensor)
        or pos_atoms.device.type != "cpu"
        or pos_atoms.ndim != 4
        or pos_atoms.shape[-1] != 3
        or pos_atoms.shape[2] <= ATOM_TO_INDEX["CA"]
    ):
        raise ValueError(
            "prepare_collated_graph_cpu requires CPU pos_atoms "
            "with shape [B, L, A, 3]"
        )
    batch_size, length = pos_atoms.shape[:2]
    if batch_size <= 0 or length <= 0:
        raise ValueError("paired-state collated graph requires non-empty batch/residue axes")

    for key, value in graph.items():
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError(
                "prepare_collated_graph_cpu must run before device "
                f"transfer; graph[{key!r}] is not on CPU"
            )

    def require_axis_tensor(
        key: str,
        *,
        required: bool,
    ) -> torch.Tensor | None:
        value = graph.get(key)
        if value is None and not required:
            return None
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"graph {key} must be a CPU tensor")
        if value.ndim < 2 or value.shape[:2] != (batch_size, length):
            raise ValueError(
                f"graph {key} must match the padded residue axis [B, L]"
            )
        return value

    node_mask = require_axis_tensor("node_mask", required=True)
    mutation_mask = require_axis_tensor("mutation_mask", required=True)
    if node_mask is None or mutation_mask is None:
        raise RuntimeError("unreachable required paired-state mask state")
    for key in ("aa", "interface_mask", "local_mask"):
        require_axis_tensor(key, required=False)
    mask_atoms = require_axis_tensor("mask_atoms", required=False)
    if (
        mask_atoms is not None
        and (
            mask_atoms.ndim != 3
            or mask_atoms.shape[2] != pos_atoms.shape[2]
        )
    ):
        raise ValueError("graph mask_atoms must match pos_atoms [B, L, A]")

    valid_axis = node_mask.to(dtype=torch.bool)
    resseq_source = graph.get(_RESSEQ_KEY, graph.get("resseq"))
    if resseq_source is None:
        raise ValueError("graph resseq metadata is required for paired-state")
    resseq, resseq_present = _fast_prepare_resseq(
        resseq_source,
        batch_size=batch_size,
        length=length,
        device=pos_atoms.device,
    )

    chain_index = graph.get("chain_index")
    if chain_index is not None and not isinstance(chain_index, torch.Tensor):
        raise ValueError("graph chain_index must be a CPU tensor when present")
    chain_source = (
        chain_index
        if isinstance(chain_index, torch.Tensor)
        else graph.get("chain_ids")
    )
    if chain_source is None:
        raise ValueError(
            "graph requires chain_index or chain_ids metadata for paired-state"
        )
    chain, chain_present = _fast_prepare_ids(
        chain_source,
        batch_size=batch_size,
        length=length,
        device=pos_atoms.device,
        name="chain identity",
    )

    mol_index = graph.get("mol_index")
    if mol_index is not None and not isinstance(mol_index, torch.Tensor):
        raise ValueError("graph mol_index must be a CPU tensor when present")
    molecule_source = (
        mol_index
        if isinstance(mol_index, torch.Tensor)
        else graph.get("mol_ids")
    )
    if molecule_source is None:
        raise ValueError(
            "graph requires mol_index or mol_ids metadata for paired-state"
        )
    molecule, _molecule_present = _fast_prepare_ids(
        molecule_source,
        batch_size=batch_size,
        length=length,
        device=pos_atoms.device,
        name="molecule identity",
    )

    if not bool((resseq_present | ~valid_axis).all()):
        raise ValueError("graph resseq metadata is incomplete at valid residues")
    if not bool((chain_present | ~valid_axis).all()):
        raise ValueError("graph chain identity is incomplete at valid residues")
    # Non-partner chains in real complexes can intentionally carry molecule
    # id -1. The geometric graph excludes them from molecule-specific edges,
    # while mutation partner attention uses its documented unknown-id fallback.

    mutation_mask = mutation_mask.to(dtype=torch.bool)
    mutation_counts = mutation_mask.sum(dim=1, dtype=torch.long)
    if not bool((mutation_counts > 0).all()):
        raise ValueError("at least one mutation per sample is required")
    max_mutations = int(mutation_counts.max().item())

    explicit_max = graph.get("max_mutations")
    if explicit_max is not None:
        if (
            not isinstance(explicit_max, Integral)
            or isinstance(explicit_max, bool)
            or int(explicit_max) != max_mutations
        ):
            raise ValueError(
                "host max_mutations does not match tensor mutation mask"
            )
    existing_counts = graph.get(_MUTATION_COUNTS_KEY)
    if existing_counts is not None:
        if (
            not isinstance(existing_counts, torch.Tensor)
            or existing_counts.device.type != "cpu"
            or existing_counts.dtype != torch.long
            or existing_counts.shape != (batch_size,)
            or not bool((existing_counts == mutation_counts).all())
        ):
            raise ValueError(
                "prepared mutation counts do not match tensor mutation mask"
            )

    metadata = graph.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, (list, tuple)) or len(metadata) != batch_size:
            raise ValueError(
                "graph metadata must contain one host record per sample"
            )
        host_counts: list[int] = []
        for batch_index, record in enumerate(metadata):
            if not isinstance(record, dict):
                raise ValueError(
                    f"graph metadata[{batch_index}] must be a dictionary"
                )
            value = record.get("num_mutation_residues")
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise ValueError(
                    f"graph metadata[{batch_index}] mutation count "
                    "must be a positive host integer"
                )
            host_counts.append(int(value))
        if host_counts != mutation_counts.tolist():
            raise ValueError(
                "host mutation counts do not match tensor mutation mask"
            )

    prepared = dict(graph)
    prepared[_RESSEQ_KEY] = resseq
    prepared["chain_index"] = chain
    prepared["mol_index"] = molecule
    prepared[_MUTATION_COUNTS_KEY] = mutation_counts
    prepared["max_mutations"] = max_mutations
    return prepared


def require_prepared_graph(
    graph: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    pos_atoms = graph.get("pos_atoms")
    if not isinstance(pos_atoms, torch.Tensor) or pos_atoms.ndim != 4:
        raise ValueError("graph pos_atoms must be [B, L, A, 3]")
    batch_size, length = pos_atoms.shape[:2]
    device = pos_atoms.device

    prepared_values: list[torch.Tensor] = []
    for key, expected_shape in (
        (_RESSEQ_KEY, (batch_size, length)),
        ("chain_index", (batch_size, length)),
        ("mol_index", (batch_size, length)),
        (_MUTATION_COUNTS_KEY, (batch_size,)),
    ):
        value = graph.get(key)
        if (
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.long
            or value.shape != expected_shape
            or value.device != device
        ):
            raise ValueError(
                "paired-state fast path requires prepared tensor metadata; call "
                "prepare_collated_graph_cpu before device transfer "
                f"(invalid {key})"
            )
        prepared_values.append(value)

    max_mutations = graph.get("max_mutations")
    if (
        not isinstance(max_mutations, Integral)
        or isinstance(max_mutations, bool)
        or int(max_mutations) <= 0
    ):
        raise ValueError(
            "paired-state fast path requires host max_mutations from "
            "prepare_collated_graph_cpu"
        )
    mutation_mask = graph.get("mutation_mask")
    if (
        not isinstance(mutation_mask, torch.Tensor)
        or mutation_mask.shape != (batch_size, length)
        or mutation_mask.device != device
    ):
        raise ValueError("graph mutation_mask must match the residue axis")

    resseq, chain, molecule, mutation_counts = prepared_values
    mutation_mask = mutation_mask.to(dtype=torch.bool)
    require_true(
        (mutation_counts == mutation_mask.sum(dim=1)).all(),
        "prepared mutation counts do not match tensor mutation mask",
    )
    require_true(
        (mutation_counts > 0).all()
        & (mutation_counts <= int(max_mutations)).all()
        & (mutation_counts == int(max_mutations)).any(),
        "prepared max_mutations does not match mutation counts",
    )
    return (
        resseq,
        chain,
        molecule,
        mutation_counts,
        int(max_mutations),
    )


def prepare_graph_for_forward(
    graph: dict[str, Any],
) -> dict[str, Any]:
    pos_atoms = graph.get("pos_atoms")
    if not isinstance(pos_atoms, torch.Tensor):
        raise ValueError("graph pos_atoms must be a tensor")
    if pos_atoms.device.type == "cpu":
        if all(
            key in graph
            for key in (
                _RESSEQ_KEY,
                "chain_index",
                "mol_index",
                _MUTATION_COUNTS_KEY,
                "max_mutations",
            )
        ):
            require_prepared_graph(graph)
            return graph
        return prepare_collated_graph_cpu(graph)
    require_prepared_graph(graph)
    return graph


def _build_fast_neighbor_lists(
    batch: dict[str, Any],
    *,
    structure_cutoff: float,
    interface_cutoff: float,
    structure_k: int,
    interface_k: int,
) -> tuple[ComplexNeighborLists, torch.Tensor]:
    """Build generic-equivalent neighbors without a device scalar branch."""
    pos_atoms = batch["pos_atoms"]
    device = pos_atoms.device
    batch_size, length = pos_atoms.shape[:2]
    resseq, chain_tensor, mol_tensor, _counts, _max_mutations = (
        require_prepared_graph(batch)
    )

    node_mask = _graph_prepare_mask(
        batch,
        "node_mask",
        (batch_size, length),
        device,
        default=True,
    )
    interface_mask = _graph_prepare_mask(
        batch,
        "interface_mask",
        (batch_size, length),
        device,
        default=False,
    )
    ca_mask = batch.get("mask_atoms")
    if ca_mask is None:
        ca_present = torch.ones(
            (batch_size, length),
            dtype=torch.bool,
            device=device,
        )
    else:
        ca_present = ca_mask.to(
            device=device,
            dtype=torch.bool,
        )[:, :, ATOM_TO_INDEX["CA"]]
    valid_residue = node_mask & ca_present

    mol_present = mol_tensor >= 0
    valid_residue = valid_residue & mol_present

    ca = pos_atoms[:, :, ATOM_TO_INDEX["CA"]].to(dtype=torch.float32)
    with torch.autocast(device_type=device.type, enabled=False):
        distance = torch.cdist(ca, ca)
    self_mask = torch.eye(
        length,
        dtype=torch.bool,
        device=device,
    ).unsqueeze(0)

    same_mol = (
        mol_present[:, :, None]
        & mol_present[:, None, :]
        & (mol_tensor[:, :, None] == mol_tensor[:, None, :])
    )
    same_mol = (
        same_mol
        & valid_residue[:, :, None]
        & valid_residue[:, None, :]
        & ~self_mask
    )

    chain_present = chain_tensor >= 0
    same_chain = (
        chain_present[:, :, None]
        & chain_present[:, None, :]
        & (chain_tensor[:, :, None] == chain_tensor[:, None, :])
    )
    offset = (resseq[:, :, None] - resseq[:, None, :]).abs()
    seq_adj = same_chain & ((offset == 1) | (offset == 2))
    seq_adj = seq_adj & same_mol

    structure_valid = same_mol & (
        (distance <= structure_cutoff) | seq_adj
    )
    structure_max = torch.where(
        structure_valid,
        distance,
        torch.full_like(distance, float("-inf")),
    ).amax()
    structure_max = torch.where(
        torch.isfinite(structure_max),
        structure_max,
        torch.zeros_like(structure_max),
    )
    structure_bonus = structure_max + 1.0
    structure_scores = (
        distance
        - seq_adj.to(dtype=distance.dtype) * structure_bonus
    )
    structure_scores = torch.where(
        structure_valid,
        structure_scores,
        torch.full_like(structure_scores, float("inf")),
    )

    different_mol = (
        mol_present[:, :, None]
        & mol_present[:, None, :]
        & (mol_tensor[:, :, None] != mol_tensor[:, None, :])
    )
    interface_valid = (
        valid_residue[:, :, None]
        & valid_residue[:, None, :]
        & interface_mask[:, :, None]
        & interface_mask[:, None, :]
        & different_mol
        & (distance < interface_cutoff)
        & ~self_mask
    )
    interface_scores = torch.where(
        interface_valid,
        distance,
        torch.full_like(distance, float("inf")),
    )

    neighbors = ComplexNeighborLists(
        structure=_neighbor_list_from_scores(
            structure_scores,
            structure_valid,
            structure_k,
        ),
        interface=_neighbor_list_from_scores(
            interface_scores,
            interface_valid,
            interface_k,
        ),
    )
    return neighbors, distance


def _fast_directed_edge_geometry(
    frame: torch.Tensor,
    frame_valid: torch.Tensor,
    ca_positions: torch.Tensor,
    neighbors: NeighborList,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """Build one reusable FP32 directed-edge representation."""
    if neighbors.indices.shape[-1] == 0:
        return ca_positions.to(dtype=torch.float32).new_zeros(
            (*neighbors.indices.shape, 28)
        )
    device = ca_positions.device
    with torch.autocast(device_type=device.type, enabled=False):
        frame = frame.to(dtype=torch.float32)
        ca_positions = ca_positions.to(dtype=torch.float32)
        source_mask = (
            frame_valid.to(dtype=torch.bool, device=device)
            & node_mask.to(dtype=torch.bool, device=device)
        )
        target_mask = batched_neighbor_gather(
            source_mask,
            neighbors.indices,
        )
        edge_mask = (
            neighbors.mask.to(dtype=torch.bool, device=device)
            & source_mask.unsqueeze(-1)
            & target_mask
        )

        target_ca = batched_neighbor_gather(
            ca_positions,
            neighbors.indices,
        )
        displacement = target_ca - ca_positions.unsqueeze(-2)
        distance = displacement.norm(dim=-1)
        local_direction = safe_normalize(to_local(frame, displacement))

        target_frame = batched_neighbor_gather(frame, neighbors.indices)
        relative_frame = torch.matmul(
            frame.unsqueeze(-3),
            target_frame.transpose(-1, -2),
        )
        relative_frame = relative_frame.reshape(
            *relative_frame.shape[:-2],
            9,
        )

        centers = torch.linspace(
            0.0,
            20.0,
            16,
            device=device,
            dtype=torch.float32,
        )
        delta = distance.unsqueeze(-1) - centers
        rbf = torch.exp(
            -0.5 * (delta / (20.0 / (16 - 1))).square()
        )
        edge = torch.cat(
            [rbf, local_direction, relative_frame],
            dim=-1,
        )
        return torch.where(
            edge_mask.unsqueeze(-1),
            edge,
            torch.zeros_like(edge),
        )


def build_geometry_cache(
    batch: dict[str, Any],
    encoder: LocalGeometricEncoder,
) -> GeometryCache:
    """Build all state geometry once for both frozen and trainable encoders."""
    batch = prepare_graph_for_forward(batch)
    pos_atoms = batch.get("pos_atoms")
    if not isinstance(pos_atoms, torch.Tensor) or pos_atoms.ndim != 4:
        raise ValueError("graph pos_atoms must be [B, L, A, 3]")
    pos_atoms = pos_atoms.to(dtype=torch.float32)
    device = pos_atoms.device
    batch_size, length, atom_count = pos_atoms.shape[:3]

    mask_atoms_value = batch.get("mask_atoms")
    if mask_atoms_value is None:
        mask_atoms = torch.ones(
            (batch_size, length, atom_count),
            dtype=torch.bool,
            device=device,
        )
    else:
        mask_atoms = mask_atoms_value.to(
            dtype=torch.bool,
            device=device,
        )

    resseq, chain_index, _molecule, _counts, _max_mutations = (
        require_prepared_graph(batch)
    )
    geometry_batch = dict(batch)
    geometry_batch["pos_atoms"] = pos_atoms
    geometry_batch["mask_atoms"] = mask_atoms
    geometry_batch["chain_ids"] = chain_index
    geometry_batch["resseq"] = resseq
    with torch.autocast(device_type=device.type, enabled=False):
        frame, frame_valid = construct_local_frames(pos_atoms, mask_atoms)
        ca = pos_atoms[:, :, ATOM_TO_INDEX["CA"]]
        angle_values, angle_valid = chain_aware_angle_targets(
            geometry_batch
        )
        neighbors, ca_distance = _build_fast_neighbor_lists(
            geometry_batch,
            structure_cutoff=encoder.structure_cutoff,
            interface_cutoff=encoder.interface_cutoff,
            structure_k=encoder.structure_k,
            interface_k=encoder.interface_k,
        )

    node_mask = _graph_prepare_mask(
        geometry_batch,
        "node_mask",
        (batch_size, length),
        device,
        default=True,
    )
    interface_value = geometry_batch.get("interface_mask")
    interface_mask = _graph_prepare_mask(
        geometry_batch,
        "interface_mask",
        (batch_size, length),
        device,
        default=False,
    )
    if interface_value is None:
        interface_mask = node_mask

    block_mask = node_mask & frame_valid
    structure_edge = _fast_directed_edge_geometry(
        frame,
        frame_valid,
        ca,
        neighbors.structure,
        block_mask,
    )
    interface_edge = _fast_directed_edge_geometry(
        frame,
        frame_valid,
        ca,
        neighbors.interface,
        block_mask,
    )
    return GeometryCache(
        neighbors=neighbors,
        frame=frame,
        frame_valid=frame_valid,
        ca=ca,
        ca_distance=ca_distance,
        angle_values=angle_values,
        angle_valid=angle_valid,
        node_mask=node_mask,
        interface_mask=interface_mask,
        structure_edge=structure_edge,
        interface_edge=interface_edge,
        neighbor_hyperparameters=(
            float(encoder.structure_cutoff),
            float(encoder.interface_cutoff),
            int(encoder.structure_k),
            int(encoder.interface_k),
        ),
    )


def run_encoder(
    encoder: LocalGeometricEncoder,
    batch: dict[str, Any],
    cache: GeometryCache,
    *,
    node_bias: torch.Tensor | None = None,
    node_bias_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run existing encoder weights against a precomputed state cache."""
    expected_hyperparameters = (
        float(encoder.structure_cutoff),
        float(encoder.interface_cutoff),
        int(encoder.structure_k),
        int(encoder.interface_k),
    )
    if expected_hyperparameters != cache.neighbor_hyperparameters:
        raise ValueError(
            "cached and encoder neighbor hyperparameters do not match"
        )

    batch_size, length = cache.node_mask.shape
    device = cache.node_mask.device
    aa_value = batch.get("aa")
    if (
        not isinstance(aa_value, torch.Tensor)
        or aa_value.shape != (batch_size, length)
        or aa_value.dtype != torch.long
        or aa_value.device != device
    ):
        raise ValueError(
            "paired-state fast encoder requires aligned device-resident long aa"
        )
    aa = aa_value

    angle_values = cache.angle_values.to(
        device=device,
        dtype=torch.float32,
    )
    angle_valid = cache.angle_valid.to(
        device=device,
        dtype=torch.bool,
    )
    angle_values = torch.where(
        angle_valid,
        angle_values,
        torch.zeros_like(angle_values),
    )
    h = encoder.aa_embed(aa) + encoder.angle_mlp(angle_values)

    if node_bias is not None:
        if (
            node_bias.ndim != 3
            or node_bias.shape[:2] != (batch_size, length)
            or node_bias.shape[-1] != h.shape[-1]
            or node_bias.device != device
        ):
            raise ValueError("node_bias must align with paired-state encoder nodes")
        bias = node_bias.to(dtype=h.dtype)
        if node_bias_mask is None:
            bias_mask = cache.node_mask
        else:
            if (
                node_bias_mask.shape != (batch_size, length)
                or node_bias_mask.device != device
            ):
                raise ValueError(
                    "node_bias_mask must align with paired-state encoder nodes"
                )
            bias_mask = node_bias_mask.to(dtype=torch.bool) & cache.node_mask
        h = h + bias * bias_mask.unsqueeze(-1).to(h.dtype)
    h = h * cache.node_mask.unsqueeze(-1).to(h.dtype)

    block_mask = cache.node_mask & cache.frame_valid
    for block in encoder.blocks:
        h = block.norm_structure(
            h
            + block.structure_attn(
                h,
                cache.neighbors.structure,
                cache.structure_edge,
                block_mask,
            )
        )
        h = block.norm_interface(
            h
            + block.interface_attn(
                h,
                cache.neighbors.interface,
                cache.interface_edge,
                block_mask,
            )
        )
        h = block.norm_ff(h + block.ff(h))
        h = h * block_mask.unsqueeze(-1).to(h.dtype)

    interface_mask = cache.interface_mask & cache.node_mask
    weights = interface_mask.unsqueeze(-1).to(h.dtype)
    interface = (h * weights).sum(dim=1) / weights.sum(
        dim=1
    ).clamp_min(1.0)
    return h, interface


def _fast_quantize_inference(
    projector: nn.Module,
    quantizer: nn.Module,
    h: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    projected = projector(h)
    z_norm = F.normalize(projected, p=2, dim=-1)
    weight_norm = F.normalize(
        quantizer.embedding.weight,
        p=2,
        dim=-1,
    )
    distances = (
        z_norm.square().sum(dim=-1, keepdim=True)
        + weight_norm.square().sum(dim=-1).view(
            *((1,) * (z_norm.ndim - 1)),
            -1,
        )
        - 2.0 * torch.matmul(z_norm, weight_norm.t())
    )
    indices = torch.argmin(distances, dim=-1)
    quantized = F.normalize(
        quantizer.embedding(indices),
        p=2,
        dim=-1,
    )
    quantized = z_norm + (quantized - z_norm).detach()
    quantized = torch.where(
        valid_mask.unsqueeze(-1),
        quantized,
        torch.zeros_like(quantized),
    )
    return quantized, indices


def run_pretrained_encoder(
    pretrained_encoder: FrozenLocalGeometryEncoder,
    graph: dict[str, Any],
    cache: GeometryCache,
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        valid_mask = valid_residue_mask(graph)
        h, _interface = run_encoder(
            pretrained_encoder.encoder,
            graph,
            cache,
        )
        h = torch.where(
            valid_mask.unsqueeze(-1),
            h,
            torch.zeros_like(h),
        )
        q_type, type_indices = _fast_quantize_inference(
            pretrained_encoder.type_projector,
            pretrained_encoder.vq_type,
            h,
            valid_mask,
        )
        q_angle, angle_indices = _fast_quantize_inference(
            pretrained_encoder.angle_projector,
            pretrained_encoder.vq_angle,
            h,
            valid_mask,
        )
        q_geometry, geometry_indices = _fast_quantize_inference(
            pretrained_encoder.geometry_projector,
            pretrained_encoder.vq_geometry,
            h,
            valid_mask,
        )
    return {
        "h": h,
        "q_type": q_type,
        "q_angle": q_angle,
        "q_geometry": q_geometry,
        "type_indices": type_indices,
        "angle_indices": angle_indices,
        "geometry_indices": geometry_indices,
        "valid_mask": valid_mask,
    }


@dataclass(frozen=True)
class MutationSiteMetadata:
    ca: torch.Tensor
    resseq: torch.Tensor
    chain: torch.Tensor
    molecule: torch.Tensor

    def __post_init__(self) -> None:
        fields = {
            "ca": self.ca,
            "resseq": self.resseq,
            "chain": self.chain,
            "molecule": self.molecule,
        }
        if any(not isinstance(value, torch.Tensor) for value in fields.values()):
            raise TypeError("metadata fields must be tensors")
        if self.ca.ndim != 3 or self.ca.shape[-1] != 3:
            raise ValueError("ca must be [B, M, 3]")
        expected = self.ca.shape[:2]
        for name, value in fields.items():
            if name != "ca" and (value.ndim != 2 or value.shape != expected):
                raise ValueError(f"{name} must be [B, M]")
            if value.device != self.ca.device:
                raise ValueError("metadata fields must share a device")
        if not self.ca.is_floating_point():
            raise ValueError("ca must be floating point")
        for name in ("resseq", "chain", "molecule"):
            if fields[name].dtype != torch.long:
                raise ValueError(f"{name} must use torch.long")

    def permute(self, order: torch.Tensor) -> "MutationSiteMetadata":
        if not isinstance(order, torch.Tensor):
            raise TypeError("order must be a tensor")
        if order.ndim != 1 or order.numel() != self.ca.shape[1]:
            raise ValueError("order must contain M entries")
        if order.dtype == torch.bool or order.is_floating_point() or order.is_complex():
            raise ValueError("order must contain integer indices")
        index = order.to(device=self.ca.device, dtype=torch.long)
        expected = torch.arange(index.numel(), device=index.device)
        if not torch.equal(index.sort().values, expected):
            raise ValueError("order must be a permutation")
        return MutationSiteMetadata(
            ca=self.ca.index_select(1, index),
            resseq=self.resseq.index_select(1, index),
            chain=self.chain.index_select(1, index),
            molecule=self.molecule.index_select(1, index),
        )


@dataclass(frozen=True)
class MutationSiteBatch:
    tokens: torch.Tensor
    mask: torch.Tensor
    indices: torch.Tensor
    metadata: MutationSiteMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor):
            raise TypeError("tokens must be a tensor")
        if self.tokens.ndim != 3:
            raise ValueError("tokens must be [B, M, H]")
        if not self.tokens.is_floating_point():
            raise ValueError("tokens must be floating point")
        if not isinstance(self.mask, torch.Tensor):
            raise TypeError("mask must be a tensor")
        if self.mask.ndim != 2 or self.mask.dtype != torch.bool:
            raise ValueError("mask must be bool [B, M]")
        if self.mask.shape != self.tokens.shape[:2]:
            raise ValueError("mask shape must match [B, M]")
        if not isinstance(self.indices, torch.Tensor):
            raise TypeError("indices must be a tensor")
        if self.indices.ndim != 2 or self.indices.shape != self.tokens.shape[:2]:
            raise ValueError("indices must be [B, M]")
        if self.indices.dtype != torch.long:
            raise ValueError("indices must use torch.long")
        if not isinstance(self.metadata, MutationSiteMetadata):
            raise TypeError("metadata must be MutationSiteMetadata")
        if self.metadata.ca.shape[:2] != self.tokens.shape[:2]:
            raise ValueError("metadata shape must match [B, M]")

        device = self.tokens.device
        if self.mask.device != device or self.indices.device != device:
            raise ValueError("tokens, mask, and indices must share a device")
        if self.metadata.ca.device != device:
            raise ValueError("metadata device must match tokens")
        if self.metadata.ca.dtype != torch.float32:
            raise ValueError("metadata CA must use torch.float32")


def gather_mutation_sites(
    node_tokens: torch.Tensor,
    graph: dict[str, Any],
    mutation_mask: torch.Tensor,
) -> MutationSiteBatch:
    if not isinstance(node_tokens, torch.Tensor):
        raise TypeError("node_tokens must be a tensor")
    if node_tokens.ndim != 3:
        raise ValueError("node_tokens must be [B, L, H]")
    if not node_tokens.is_floating_point():
        raise ValueError("node_tokens must be floating point")
    if not isinstance(mutation_mask, torch.Tensor):
        raise TypeError("mutation_mask must be a tensor")
    if mutation_mask.ndim != 2 or mutation_mask.dtype != torch.bool:
        raise ValueError("mutation_mask must be bool [B, L]")
    if mutation_mask.shape != node_tokens.shape[:2]:
        raise ValueError("mutation_mask shape must match [B, L]")
    if mutation_mask.device != node_tokens.device:
        raise ValueError("mutation_mask device must match node_tokens")
    if not isinstance(graph, dict):
        raise TypeError("graph must be a dictionary")

    pos_atoms = graph.get("pos_atoms")
    if not isinstance(pos_atoms, torch.Tensor):
        raise ValueError("graph pos_atoms must be a tensor")
    if (
        pos_atoms.ndim != 4
        or pos_atoms.shape[:2] != node_tokens.shape[:2]
        or pos_atoms.shape[2] <= ATOM_TO_INDEX["CA"]
        or pos_atoms.shape[-1] != 3
    ):
        raise ValueError("graph pos_atoms must be [B, L, A, 3]")
    if not pos_atoms.is_floating_point():
        raise ValueError("graph pos_atoms must be floating point")

    ca_full = pos_atoms[:, :, ATOM_TO_INDEX["CA"]].to(
        device=node_tokens.device,
        dtype=torch.float32,
    )
    require_finite(
        ca_full,
        "graph has non-finite valid mutation CA coordinates",
        mutation_mask,
    )

    batch_size, length, _hidden_dim = node_tokens.shape
    if pos_atoms.device.type == "cpu":
        graph_for_prepare = dict(graph)
        graph_for_prepare["mutation_mask"] = mutation_mask
        graph = prepare_graph_for_forward(graph_for_prepare)
    resseq_full, chain_full, molecule_full, counts, max_mutations = (
        require_prepared_graph(graph)
    )
    require_true(
        (
            graph["mutation_mask"].to(dtype=torch.bool)
            == mutation_mask
        ).all(),
        "prepared mutation mask does not match requested mutation sites",
    )
    require_true(
        (counts > 0).all(),
        "at least one mutation per sample is required",
    )
    mutation_slots = torch.arange(
        max_mutations,
        device=node_tokens.device,
    )
    residue_indices = torch.arange(
        length,
        device=node_tokens.device,
    ).unsqueeze(0).expand(batch_size, -1)
    ordered_indices = torch.where(
        mutation_mask,
        residue_indices,
        torch.full_like(residue_indices, length),
    ).sort(dim=1).values
    selected_indices = ordered_indices.index_select(
        1,
        mutation_slots,
    )
    mask = selected_indices < length
    indices = torch.where(
        mask,
        selected_indices,
        torch.full_like(selected_indices, -1),
    )
    safe_indices = selected_indices.clamp_max(length - 1)
    batch_index = torch.arange(
        batch_size,
        device=node_tokens.device,
    ).unsqueeze(1).expand_as(safe_indices)
    site_mask = mask.unsqueeze(-1)
    gathered_tokens = node_tokens[batch_index, safe_indices]
    tokens = torch.where(
        site_mask,
        gathered_tokens,
        torch.zeros_like(gathered_tokens),
    )
    gathered_ca = ca_full[batch_index, safe_indices]
    ca = torch.where(
        site_mask,
        gathered_ca,
        torch.zeros_like(gathered_ca),
    )

    def gather_ids(values: torch.Tensor) -> torch.Tensor:
        gathered = values[batch_index, safe_indices]
        return torch.where(mask, gathered, torch.full_like(gathered, -1))

    metadata = MutationSiteMetadata(
        ca=ca,
        resseq=gather_ids(resseq_full),
        chain=gather_ids(chain_full),
        molecule=gather_ids(molecule_full),
    )
    return MutationSiteBatch(
        tokens=tokens,
        mask=mask,
        indices=indices,
        metadata=metadata,
    )


def opposite_partner_mask(
    mutation_molecule: torch.Tensor,
    candidate_molecule: torch.Tensor,
) -> torch.Tensor:
    for name, value in (
        ("mutation_molecule", mutation_molecule),
        ("candidate_molecule", candidate_molecule),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim != 1:
            raise ValueError(f"{name} must be rank 1")
        if value.dtype != torch.long:
            raise ValueError(f"{name} must use torch.long")
    if mutation_molecule.device != candidate_molecule.device:
        raise ValueError("molecule tensors must share a device")

    mutation = mutation_molecule.unsqueeze(1)
    candidate = candidate_molecule.unsqueeze(0)
    both_known = (mutation >= 0) & (candidate >= 0)
    return ~both_known | (mutation != candidate)


@dataclass(frozen=True)
class _NearestPartnerSelection:
    mask: torch.Tensor
    indices: torch.Tensor
    slot_mask: torch.Tensor
    distances: torch.Tensor


def select_nearest_partner_mask(
    mutation_ca: torch.Tensor,
    candidate_ca: torch.Tensor,
    mutation_molecule: torch.Tensor,
    candidate_molecule: torch.Tensor,
    max_partner_tokens: int,
    *,
    mutation_mask: torch.Tensor | None = None,
    candidate_mask: torch.Tensor | None = None,
    precomputed_distance: torch.Tensor | None = None,
    return_details: bool = False,
) -> torch.Tensor | _NearestPartnerSelection:
    max_partner_tokens = _validate_positive(
        "max_partner_tokens",
        max_partner_tokens,
    )
    for name, value in (
        ("mutation_ca", mutation_ca),
        ("candidate_ca", candidate_ca),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be floating point")
    for name, value in (
        ("mutation_molecule", mutation_molecule),
        ("candidate_molecule", candidate_molecule),
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.dtype != torch.long:
            raise ValueError(f"{name} must use torch.long")

    unbatched = mutation_ca.ndim == 2
    if unbatched:
        if (
            mutation_ca.shape[-1] != 3
            or candidate_ca.ndim != 2
            or candidate_ca.shape[-1] != 3
            or mutation_molecule.ndim != 1
            or candidate_molecule.ndim != 1
        ):
            raise ValueError("unbatched partner inputs must be [M, 3], [L, 3], [M], [L]")
        mutation_ca = mutation_ca.unsqueeze(0)
        candidate_ca = candidate_ca.unsqueeze(0)
        mutation_molecule = mutation_molecule.unsqueeze(0)
        candidate_molecule = candidate_molecule.unsqueeze(0)
        if mutation_mask is not None:
            mutation_mask = mutation_mask.unsqueeze(0)
        if candidate_mask is not None:
            candidate_mask = candidate_mask.unsqueeze(0)
        if precomputed_distance is not None:
            if precomputed_distance.ndim != 2:
                raise ValueError(
                    "unbatched precomputed_distance must be [M, L]"
                )
            precomputed_distance = precomputed_distance.unsqueeze(0)
    elif (
        mutation_ca.ndim != 3
        or mutation_ca.shape[-1] != 3
        or candidate_ca.ndim != 3
        or candidate_ca.shape[-1] != 3
        or mutation_molecule.ndim != 2
        or candidate_molecule.ndim != 2
    ):
        raise ValueError(
            "batched partner inputs must be [B, M, 3], [B, L, 3], [B, M], [B, L]"
        )

    if mutation_ca.device != candidate_ca.device:
        raise ValueError("CA tensors must share a device")
    if (
        mutation_molecule.device != mutation_ca.device
        or candidate_molecule.device != mutation_ca.device
    ):
        raise ValueError("molecule tensors must share the CA device")
    if mutation_ca.shape[0] != candidate_ca.shape[0]:
        raise ValueError("mutation and candidate batches must match")
    if mutation_molecule.shape != mutation_ca.shape[:2]:
        raise ValueError("mutation_molecule shape must match mutation_ca")
    if candidate_molecule.shape != candidate_ca.shape[:2]:
        raise ValueError("candidate_molecule shape must match candidate_ca")

    def prepare_mask(
        name: str,
        value: torch.Tensor | None,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        if value is None:
            return torch.ones(shape, dtype=torch.bool, device=mutation_ca.device)
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape != shape
            or value.dtype != torch.bool
        ):
            raise ValueError(f"{name} must be bool with shape {shape}")
        if value.device != mutation_ca.device:
            raise ValueError(f"{name} device must match CA tensors")
        return value

    mutation_mask = prepare_mask(
        "mutation_mask",
        mutation_mask,
        tuple(mutation_ca.shape[:2]),
    )
    candidate_mask = prepare_mask(
        "candidate_mask",
        candidate_mask,
        tuple(candidate_ca.shape[:2]),
    )
    require_finite(mutation_ca, "mutation_ca must be finite", mutation_mask)
    require_finite(candidate_ca, "candidate_ca must be finite", candidate_mask)

    mutation = mutation_molecule.unsqueeze(-1)
    candidate = candidate_molecule.unsqueeze(1)
    both_known = (mutation >= 0) & (candidate >= 0)
    allowed = (
        mutation_mask.unsqueeze(-1)
        & candidate_mask.unsqueeze(1)
        & (~both_known | (mutation != candidate))
    )
    if precomputed_distance is None:
        distance = float32_cdist(mutation_ca, candidate_ca)
    else:
        if (
            not isinstance(precomputed_distance, torch.Tensor)
            or precomputed_distance.ndim != 3
            or precomputed_distance.shape
            != (
                mutation_ca.shape[0],
                mutation_ca.shape[1],
                candidate_ca.shape[1],
            )
            or not precomputed_distance.is_floating_point()
        ):
            raise ValueError(
                "precomputed_distance must be floating point [B, M, L]"
            )
        if precomputed_distance.device != mutation_ca.device:
            raise ValueError(
                "precomputed_distance device must match CA tensors"
            )
        distance = precomputed_distance.to(dtype=torch.float32)
        require_finite(
            distance,
            "precomputed_distance must be finite at allowed pairs",
            allowed,
        )
    masked_distance = distance.masked_fill(~allowed, float("inf"))
    keep = min(max_partner_tokens, candidate_ca.shape[1])
    nearest_distance, nearest_index = torch.topk(
        masked_distance,
        k=keep,
        dim=-1,
        largest=False,
        sorted=True,
    )
    slot_mask = torch.isfinite(nearest_distance)
    selected = torch.zeros_like(allowed)
    selected.scatter_(
        -1,
        nearest_index,
        slot_mask,
    )
    details = _NearestPartnerSelection(
        mask=selected,
        indices=nearest_index,
        slot_mask=slot_mask,
        distances=torch.where(
            slot_mask,
            nearest_distance,
            torch.zeros_like(nearest_distance),
        ),
    )
    if return_details:
        return details
    return selected[0] if unbatched else selected


class MutationSpecificInterfaceContext(nn.Module):
    """Attend from each mutation to its nearest opposite-partner interface residues."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        distance_rbf_bins: int,
        max_partner_tokens: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive("hidden_dim", hidden_dim)
        self.num_heads = _validate_positive("num_heads", num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        distance_rbf_bins = _validate_positive(
            "distance_rbf_bins",
            distance_rbf_bins,
        )
        self.max_partner_tokens = _validate_positive(
            "max_partner_tokens",
            max_partner_tokens,
        )
        self.dropout_p = _validate_dropout(dropout)
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.distance_bias = nn.Sequential(
            nn.Linear(distance_rbf_bins, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_heads),
        )
        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        self.register_buffer(
            "distance_centers",
            torch.linspace(0.0, 30.0, steps=distance_rbf_bins),
            persistent=False,
        )

    @staticmethod
    def _validate_mask(
        name: str,
        value: torch.Tensor,
        shape: tuple[int, int],
        device: torch.device,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if value.ndim != 2 or value.shape != shape or value.dtype != torch.bool:
            raise ValueError(f"{name} must be bool [B, L]")
        if value.device != device:
            raise ValueError(f"{name} device must match node_tokens")

    def _validate_inputs(
        self,
        node_tokens: torch.Tensor,
        graph: dict[str, Any],
        mutation_sites: MutationSiteBatch,
        valid_mask: torch.Tensor,
        interface_mask: torch.Tensor,
        mutation_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(node_tokens, torch.Tensor):
            raise TypeError("node_tokens must be a tensor")
        if node_tokens.ndim != 3:
            raise ValueError("node_tokens must be [B, L, H]")
        if not node_tokens.is_floating_point():
            raise ValueError("node_tokens must be floating point")
        if node_tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"node token dimension must be {self.hidden_dim}")
        if not isinstance(graph, dict):
            raise TypeError("graph must be a dictionary")
        if not isinstance(mutation_sites, MutationSiteBatch):
            raise TypeError("mutation_sites must be MutationSiteBatch")
        if mutation_sites.tokens.shape[0] != node_tokens.shape[0]:
            raise ValueError("mutation site batch size must match node_tokens")
        if mutation_sites.tokens.shape[-1] != self.hidden_dim:
            raise ValueError(f"mutation site dimension must be {self.hidden_dim}")
        if (
            mutation_sites.tokens.device != node_tokens.device
            or mutation_sites.tokens.dtype != node_tokens.dtype
        ):
            raise ValueError("mutation sites must match node_tokens dtype/device")

        shape = tuple(node_tokens.shape[:2])
        self._validate_mask("valid_mask", valid_mask, shape, node_tokens.device)
        self._validate_mask(
            "interface_mask",
            interface_mask,
            shape,
            node_tokens.device,
        )
        self._validate_mask(
            "mutation_mask",
            mutation_mask,
            shape,
            node_tokens.device,
        )

        pos_atoms = graph.get("pos_atoms")
        if not isinstance(pos_atoms, torch.Tensor):
            raise ValueError("graph pos_atoms must be a tensor")
        if (
            pos_atoms.ndim != 4
            or pos_atoms.shape[:2] != node_tokens.shape[:2]
            or pos_atoms.shape[2] <= ATOM_TO_INDEX["CA"]
            or pos_atoms.shape[-1] != 3
        ):
            raise ValueError("graph pos_atoms must be [B, L, A, 3]")
        if not pos_atoms.is_floating_point():
            raise ValueError("graph pos_atoms must be floating point")
        return pos_atoms[:, :, ATOM_TO_INDEX["CA"]].to(
            device=node_tokens.device,
            dtype=torch.float32,
        )

    def forward(
        self,
        node_tokens: torch.Tensor,
        graph: dict[str, Any],
        mutation_sites: MutationSiteBatch,
        *,
        valid_mask: torch.Tensor,
        interface_mask: torch.Tensor,
        mutation_mask: torch.Tensor,
        ca_distance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        graph = prepare_graph_for_forward(graph)
        ca_full = self._validate_inputs(
            node_tokens,
            graph,
            mutation_sites,
            valid_mask,
            interface_mask,
            mutation_mask,
        )
        batch_size, length, _hidden_dim = node_tokens.shape
        mutation_count = mutation_sites.tokens.shape[1]
        site_mask = mutation_sites.mask
        require_finite(
            mutation_sites.metadata.ca,
            "mutation sites have non-finite valid mutation CA coordinates",
            site_mask,
        )
        mutation_tokens = torch.where(
            site_mask.unsqueeze(-1),
            mutation_sites.tokens,
            torch.zeros_like(mutation_sites.tokens),
        )
        mutation_ca = torch.where(
            site_mask.unsqueeze(-1),
            mutation_sites.metadata.ca,
            torch.zeros_like(mutation_sites.metadata.ca),
        )
        candidate_mask = valid_mask & interface_mask & ~mutation_mask
        require_finite(
            ca_full,
            "graph has non-finite candidate CA coordinates",
            candidate_mask,
        )

        _resseq, _chain, molecule_full, _counts, _max_mutations = (
            require_prepared_graph(graph)
        )
        mutation_candidate_distance = None
        if ca_distance is not None:
            if (
                not isinstance(ca_distance, torch.Tensor)
                or ca_distance.ndim != 3
                or ca_distance.shape != (batch_size, length, length)
                or not ca_distance.is_floating_point()
            ):
                raise ValueError(
                    "ca_distance must be floating point [B, L, L]"
                )
            if ca_distance.device != node_tokens.device:
                raise ValueError(
                    "ca_distance device must match node_tokens"
                )
            safe_site_indices = mutation_sites.indices.clamp_min(0)
            distance_batch_index = torch.arange(
                batch_size,
                device=node_tokens.device,
            ).unsqueeze(1).expand_as(safe_site_indices)
            mutation_candidate_distance = ca_distance[
                distance_batch_index,
                safe_site_indices,
            ]
        partner_selection = select_nearest_partner_mask(
            mutation_ca,
            ca_full,
            mutation_sites.metadata.molecule,
            molecule_full,
            self.max_partner_tokens,
            mutation_mask=site_mask,
            candidate_mask=candidate_mask,
            precomputed_distance=mutation_candidate_distance,
            return_details=True,
        )
        if not isinstance(partner_selection, _NearestPartnerSelection):
            raise RuntimeError("batched partner selection did not return details")
        nearest_index = partner_selection.indices
        selected = partner_selection.slot_mask
        selected_distance = partner_selection.distances
        keep = nearest_index.shape[-1]

        gather_index = nearest_index.unsqueeze(-1).expand(
            batch_size,
            mutation_count,
            keep,
            self.hidden_dim,
        )
        expanded_node_tokens = node_tokens.unsqueeze(1).expand(
            batch_size,
            mutation_count,
            length,
            self.hidden_dim,
        )
        gathered_tokens = torch.gather(
            expanded_node_tokens,
            dim=2,
            index=gather_index,
        )
        require_finite(
            gathered_tokens,
            "graph has non-finite selected candidate tokens",
            selected,
        )
        candidate_tokens = torch.where(
            selected.unsqueeze(-1),
            gathered_tokens,
            torch.zeros_like(gathered_tokens),
        )

        query = self.q_proj(mutation_tokens).view(
            batch_size,
            mutation_count,
            self.num_heads,
            self.head_dim,
        )
        key = self.k_proj(candidate_tokens).view(
            batch_size,
            mutation_count,
            keep,
            self.num_heads,
            self.head_dim,
        )
        value = self.v_proj(candidate_tokens).view(
            batch_size,
            mutation_count,
            keep,
            self.num_heads,
            self.head_dim,
        )
        scores = torch.einsum(
            "bmhd,bmkhd->bhmk",
            query,
            key,
        ) * self.scale
        distance_rbf = float32_distance_rbf(
            selected_distance,
            self.distance_centers,
        ).to(dtype=query.dtype)
        distance_bias = self.distance_bias(distance_rbf).permute(
            0,
            3,
            1,
            2,
        )
        distance_bias = distance_bias.to(dtype=scores.dtype)
        pair_mask = selected.unsqueeze(1).expand(
            batch_size,
            self.num_heads,
            mutation_count,
            keep,
        )
        weights = masked_softmax(
            scores + distance_bias,
            pair_mask,
            dim=-1,
        )
        attended = torch.einsum(
            "bhmk,bmkhd->bmhd",
            weights,
            value,
        ).reshape(batch_size, mutation_count, self.hidden_dim)
        updated = self.norm_attn(
            mutation_tokens + self.out_proj(attended)
        )
        updated = self.norm_ff(updated + self.ff(updated))
        has_partner = selected.any(dim=-1) & site_mask
        return torch.where(
            has_partner.unsqueeze(-1),
            updated,
            torch.zeros_like(updated),
        )
@dataclass(frozen=True)
class InterfaceRelayOutput:
    partner_context: torch.Tensor
    partner_pooled: torch.Tensor
    interface_pooled: torch.Tensor


class BidirectionalInterfaceRelay(nn.Module):
    """Vectorized mutation-to-partner and set-to-interface communication."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        distance_rbf_bins: int = 8,
        max_partner_tokens: int = 128,
        max_interface_tokens: int = 128,
        max_distance: float = 60.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = _validate_positive("hidden_dim", hidden_dim)
        self.num_heads = _validate_positive("num_heads", num_heads)
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        distance_rbf_bins = _validate_positive(
            "distance_rbf_bins",
            distance_rbf_bins,
        )
        self.max_partner_tokens = _validate_positive(
            "max_partner_tokens",
            max_partner_tokens,
        )
        self.max_interface_tokens = _validate_positive(
            "max_interface_tokens",
            max_interface_tokens,
        )
        self.head_dim = self.hidden_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.partner_attention = MutationSpecificInterfaceContext(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            distance_rbf_bins=distance_rbf_bins,
            max_partner_tokens=self.max_partner_tokens,
            dropout=0.0,
        )
        self.partner_pool_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.distance_bias = nn.Sequential(
            nn.Linear(distance_rbf_bins, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.num_heads),
        )
        self.norm_attn = nn.LayerNorm(self.hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(self.hidden_dim)
        self.interface_pool_gate = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer(
            "distance_centers",
            torch.linspace(
                0.0,
                float(max_distance),
                steps=distance_rbf_bins,
            ),
            persistent=False,
        )

    @staticmethod
    def _validate_mask(
        name: str,
        value: torch.Tensor,
        shape: tuple[int, int],
        device: torch.device,
    ) -> None:
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.shape != shape
            or value.dtype != torch.bool
        ):
            raise ValueError(f"{name} must be bool [B, L]")
        if value.device != device:
            raise ValueError(f"{name} device must match node_tokens")

    def forward(
        self,
        node_tokens: torch.Tensor,
        graph: dict[str, Any],
        mutation_sites: MutationSiteBatch,
        mutation_set: MutationSetOutput,
        *,
        valid_mask: torch.Tensor,
        interface_mask: torch.Tensor,
        mutation_mask: torch.Tensor,
        ca_distance: torch.Tensor | None = None,
    ) -> InterfaceRelayOutput:
        if (
            not isinstance(node_tokens, torch.Tensor)
            or node_tokens.ndim != 3
            or node_tokens.shape[-1] != self.hidden_dim
        ):
            raise ValueError(
                f"node_tokens must be [B, L, {self.hidden_dim}]"
            )
        shape = tuple(node_tokens.shape[:2])
        self._validate_mask(
            "valid_mask",
            valid_mask,
            shape,
            node_tokens.device,
        )
        self._validate_mask(
            "interface_mask",
            interface_mask,
            shape,
            node_tokens.device,
        )
        self._validate_mask(
            "mutation_mask",
            mutation_mask,
            shape,
            node_tokens.device,
        )
        if mutation_set.tokens.shape != mutation_sites.tokens.shape:
            raise ValueError(
                "mutation set tokens must align with mutation sites"
            )
        if (
            mutation_set.mask.shape != mutation_sites.mask.shape
            or mutation_set.mask.device != mutation_sites.mask.device
        ):
            raise ValueError(
                "mutation set mask must align with mutation sites"
            )
        require_true(
            (mutation_set.mask == mutation_sites.mask).all(),
            "mutation set mask must align with mutation sites",
        )

        graph = prepare_graph_for_forward(graph)
        partner_context = self.partner_attention(
            node_tokens,
            graph,
            mutation_sites,
            valid_mask=valid_mask,
            interface_mask=interface_mask,
            mutation_mask=mutation_mask,
            ca_distance=ca_distance,
        )
        partner_mask = (
            mutation_sites.mask
            & (partner_context.abs().sum(dim=-1) > 0)
        )
        partner_pooled = attention_pool(
            partner_context,
            partner_mask,
            self.partner_pool_gate,
        )

        batch_size, length, _hidden_dim = node_tokens.shape
        ca_full = graph["pos_atoms"][:, :, ATOM_TO_INDEX["CA"]].to(
            device=node_tokens.device,
            dtype=torch.float32,
        )
        _resseq, _chain, molecule_full, _counts, _max_mutations = (
            require_prepared_graph(graph)
        )
        candidate_mask = valid_mask & interface_mask & ~mutation_mask
        if ca_distance is None:
            mutation_candidate_distance = float32_cdist(
                mutation_sites.metadata.ca,
                ca_full,
            )
        else:
            if (
                not isinstance(ca_distance, torch.Tensor)
                or ca_distance.shape != (batch_size, length, length)
                or ca_distance.device != node_tokens.device
            ):
                raise ValueError(
                    "ca_distance must align with [B, L, L]"
                )
            safe_indices = mutation_sites.indices.clamp_min(0)
            batch_index = torch.arange(
                batch_size,
                device=node_tokens.device,
            ).unsqueeze(1).expand_as(safe_indices)
            mutation_candidate_distance = ca_distance[
                batch_index,
                safe_indices,
            ].to(dtype=torch.float32)

        mutation_molecule = mutation_sites.metadata.molecule.unsqueeze(-1)
        candidate_molecule = molecule_full.unsqueeze(1)
        both_known = (
            (mutation_molecule >= 0)
            & (candidate_molecule >= 0)
        )
        allowed = (
            mutation_sites.mask.unsqueeze(-1)
            & candidate_mask.unsqueeze(1)
            & (~both_known | (mutation_molecule != candidate_molecule))
        )
        masked_distance = mutation_candidate_distance.masked_fill(
            ~allowed,
            float("inf"),
        )
        nearest_to_set = masked_distance.min(dim=1).values
        keep = min(self.max_interface_tokens, length)
        selected_distance, selected_indices = torch.topk(
            nearest_to_set,
            k=keep,
            dim=-1,
            largest=False,
            sorted=True,
        )
        selected_mask = torch.isfinite(selected_distance)
        gather_index = selected_indices.unsqueeze(-1).expand(
            batch_size,
            keep,
            self.hidden_dim,
        )
        interface_tokens = torch.gather(
            node_tokens,
            dim=1,
            index=gather_index,
        )
        interface_tokens = torch.where(
            selected_mask.unsqueeze(-1),
            interface_tokens,
            torch.zeros_like(interface_tokens),
        )
        selected_molecule = torch.gather(
            molecule_full,
            dim=1,
            index=selected_indices,
        )
        selected_pair_distance = torch.gather(
            mutation_candidate_distance,
            dim=2,
            index=selected_indices.unsqueeze(1).expand(
                batch_size,
                mutation_sites.mask.shape[1],
                keep,
            ),
        ).transpose(1, 2)

        interface_molecule = selected_molecule.unsqueeze(-1)
        set_molecule = mutation_sites.metadata.molecule.unsqueeze(1)
        selected_both_known = (
            (interface_molecule >= 0)
            & (set_molecule >= 0)
        )
        pair_mask = (
            selected_mask.unsqueeze(-1)
            & mutation_set.mask.unsqueeze(1)
            & (
                ~selected_both_known
                | (interface_molecule != set_molecule)
            )
        )
        query = self.q_proj(interface_tokens).view(
            batch_size,
            keep,
            self.num_heads,
            self.head_dim,
        )
        key = self.k_proj(mutation_set.tokens).view(
            batch_size,
            mutation_set.tokens.shape[1],
            self.num_heads,
            self.head_dim,
        )
        value = self.v_proj(mutation_set.tokens).view(
            batch_size,
            mutation_set.tokens.shape[1],
            self.num_heads,
            self.head_dim,
        )
        scores = torch.einsum(
            "bqhd,bmhd->bhqm",
            query,
            key,
        ) * self.scale
        distance_rbf = float32_distance_rbf(
            selected_pair_distance,
            self.distance_centers,
        ).to(dtype=scores.dtype)
        distance_bias = self.distance_bias(distance_rbf).permute(
            0,
            3,
            1,
            2,
        )
        attention_mask = pair_mask.unsqueeze(1).expand(
            batch_size,
            self.num_heads,
            keep,
            mutation_set.tokens.shape[1],
        )
        weights = masked_softmax(
            scores + distance_bias,
            attention_mask,
            dim=-1,
        )
        attended = torch.einsum(
            "bhqm,bmhd->bqhd",
            weights,
            value,
        ).reshape(batch_size, keep, self.hidden_dim)
        updated = self.norm_attn(
            interface_tokens + self.out_proj(attended)
        )
        updated = self.norm_ff(updated + self.ff(updated))
        has_set_partner = pair_mask.any(dim=-1)
        updated = torch.where(
            has_set_partner.unsqueeze(-1),
            updated,
            torch.zeros_like(updated),
        )
        interface_pooled = attention_pool(
            updated,
            has_set_partner,
            self.interface_pool_gate,
        )
        return InterfaceRelayOutput(
            partner_context=partner_context,
            partner_pooled=partner_pooled,
            interface_pooled=interface_pooled,
        )
