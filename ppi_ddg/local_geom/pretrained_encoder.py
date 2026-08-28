from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from ppi_ddg.constants import ATOM_TO_INDEX
from ppi_ddg.local_geom.encoder import LocalGeometricEncoder
from ppi_ddg.local_geom.vq import StandardVectorQuantizer


__all__ = ["FrozenLocalGeometryEncoder", "valid_residue_mask"]


_COMPONENT_PREFIXES = (
    "encoder.",
    "type_projector.",
    "angle_projector.",
    "geometry_projector.",
    "vq_type.",
    "vq_angle.",
    "vq_geometry.",
)
_CONTAINER_PREFIXES = ("module.", "model.", "pretrainer.")


def _load_payload(
    checkpoint: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise TypeError("checkpoint payload must be a mapping")
        return dict(payload)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("checkpoint must be a path or mapping")
    return dict(checkpoint)


def _state_mapping(payload: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            if not all(isinstance(item, torch.Tensor) for item in value.values()):
                raise TypeError(f"checkpoint {key} must contain only tensors")
            return value
    tensor_values = {
        str(key): value
        for key, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    if tensor_values:
        return tensor_values
    raise KeyError(
        "checkpoint must contain model, model_state_dict, state_dict, "
        "or direct tensor entries"
    )


def _strip_container_prefixes(key: str) -> str:
    stripped = str(key)
    changed = True
    while changed:
        changed = False
        for prefix in _CONTAINER_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix) :]
                changed = True
                break
    return stripped


def _normalized_component_state(
    payload: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, value in _state_mapping(payload).items():
        key = _strip_container_prefixes(str(raw_key))
        if not any(key.startswith(prefix) for prefix in _COMPONENT_PREFIXES):
            continue
        if key in normalized:
            raise KeyError(f"checkpoint contains duplicate normalized key: {key}")
        normalized[key] = value
    return normalized


def _config_section(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return {}
    section = config.get(key)
    return dict(section) if isinstance(section, Mapping) else {}


def valid_residue_mask(graph: Mapping[str, Any]) -> torch.Tensor:
    pos_atoms = graph.get("pos_atoms")
    if not isinstance(pos_atoms, torch.Tensor) or pos_atoms.ndim != 4:
        raise ValueError("graph pos_atoms must have shape [B, L, A, 3]")
    shape = pos_atoms.shape[:2]
    node_mask = graph.get("node_mask")
    if node_mask is None:
        valid = torch.ones(shape, dtype=torch.bool, device=pos_atoms.device)
    elif isinstance(node_mask, torch.Tensor):
        valid = node_mask.to(device=pos_atoms.device, dtype=torch.bool)
    else:
        raise TypeError("graph node_mask must be a tensor")
    if valid.shape != shape:
        raise ValueError("graph node_mask must align with the residue axis")

    mask_atoms = graph.get("mask_atoms")
    if mask_atoms is None:
        ca_valid = torch.ones(shape, dtype=torch.bool, device=pos_atoms.device)
    elif isinstance(mask_atoms, torch.Tensor):
        if mask_atoms.shape[:2] != shape:
            raise ValueError("graph mask_atoms must align with the residue axis")
        ca_valid = mask_atoms.to(
            device=pos_atoms.device,
            dtype=torch.bool,
        )[:, :, ATOM_TO_INDEX["CA"]]
    else:
        raise TypeError("graph mask_atoms must be a tensor")
    return valid & ca_valid


class FrozenLocalGeometryEncoder(nn.Module):
    """Frozen residue encoder with three local-geometry codebook branches."""

    def __init__(
        self,
        checkpoint: str | Path | Mapping[str, Any],
        *,
        require_healthy_codebooks: bool = True,
        node_dim: int | None = None,
        num_blocks: int | None = None,
        num_heads: int | None = None,
        structure_cutoff: float | None = None,
        interface_cutoff: float | None = None,
        structure_k: int | None = None,
        interface_k: int | None = None,
        codebook_size: int | None = None,
        commitment_cost: float | None = None,
    ) -> None:
        super().__init__()
        payload = _load_payload(checkpoint)
        if (
            require_healthy_codebooks
            and payload.get("codebook_health_passed") is not True
        ):
            raise ValueError(
                "checkpoint must report healthy type, angle, and geometry codebooks"
            )

        model_cfg = _config_section(payload, "model")
        loss_cfg = _config_section(payload, "loss")
        self.node_dim = int(
            node_dim if node_dim is not None else model_cfg.get("node_dim", 256)
        )
        num_blocks = int(
            num_blocks if num_blocks is not None else model_cfg.get("num_blocks", 4)
        )
        num_heads = int(
            num_heads if num_heads is not None else model_cfg.get("num_heads", 4)
        )
        structure_cutoff = float(
            structure_cutoff
            if structure_cutoff is not None
            else model_cfg.get("structure_cutoff", 12.0)
        )
        interface_cutoff = float(
            interface_cutoff
            if interface_cutoff is not None
            else model_cfg.get("interface_cutoff", 10.0)
        )
        structure_k = int(
            structure_k
            if structure_k is not None
            else model_cfg.get("structure_k", 16)
        )
        interface_k = int(
            interface_k
            if interface_k is not None
            else model_cfg.get("interface_k", 16)
        )
        self.codebook_size = int(
            codebook_size
            if codebook_size is not None
            else model_cfg.get("codebook_size", 256)
        )
        commitment_cost = float(
            commitment_cost
            if commitment_cost is not None
            else loss_cfg.get(
                "commitment_cost",
                loss_cfg.get("beta", 0.25),
            )
        )

        self.encoder = LocalGeometricEncoder(
            node_dim=self.node_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            structure_cutoff=structure_cutoff,
            interface_cutoff=interface_cutoff,
            structure_k=structure_k,
            interface_k=interface_k,
        )
        self.type_projector = nn.Linear(self.node_dim, self.node_dim)
        self.angle_projector = nn.Linear(self.node_dim, self.node_dim)
        self.geometry_projector = nn.Linear(self.node_dim, self.node_dim)
        self.vq_type = StandardVectorQuantizer(
            self.node_dim,
            self.codebook_size,
            commitment_cost=commitment_cost,
        )
        self.vq_angle = StandardVectorQuantizer(
            self.node_dim,
            self.codebook_size,
            commitment_cost=commitment_cost,
        )
        self.vq_geometry = StandardVectorQuantizer(
            self.node_dim,
            self.codebook_size,
            commitment_cost=commitment_cost,
        )

        state = _normalized_component_state(payload)
        expected = set(self.state_dict())
        actual = set(state)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise KeyError(
                "pretrained encoder state mismatch; "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )
        self.load_state_dict(state, strict=True)

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> "FrozenLocalGeometryEncoder":
        super().train(False)
        return self

    @staticmethod
    def _quantize(
        projector: nn.Module,
        quantizer: StandardVectorQuantizer,
        h: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = projector(h)
        quantized, _loss, indices = quantizer(projected, valid_mask)
        quantized = torch.where(
            valid_mask.unsqueeze(-1),
            quantized,
            torch.zeros_like(quantized),
        )
        return quantized, indices

    def forward(self, graph: dict[str, Any]) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            valid_mask = valid_residue_mask(graph)
            h, _interface = self.encoder(graph)
            h = torch.where(
                valid_mask.unsqueeze(-1),
                h,
                torch.zeros_like(h),
            )
            q_type, type_indices = self._quantize(
                self.type_projector,
                self.vq_type,
                h,
                valid_mask,
            )
            q_angle, angle_indices = self._quantize(
                self.angle_projector,
                self.vq_angle,
                h,
                valid_mask,
            )
            q_geometry, geometry_indices = self._quantize(
                self.geometry_projector,
                self.vq_geometry,
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
