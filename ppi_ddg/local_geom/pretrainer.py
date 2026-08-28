from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ppi_ddg.local_geom.decoder import IterativeGeometricDecoder, SparseFeatureDecoder, initialize_geometry_state
from ppi_ddg.local_geom.encoder import LocalGeometricEncoder
from ppi_ddg.local_geom.geometry import chain_aware_angle_targets
from ppi_ddg.local_geom.graph import ComplexNeighborLists, build_neighbor_lists
from ppi_ddg.local_geom.vq import StandardVectorQuantizer

__all__ = ["LocalGeometricVQPretrainer"]


def _float_log(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _sample_aa_noise(
    shape: tuple[int, ...],
    target_device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randint(0, 21, shape, device=target_device)

    generator_device = torch.device(generator.device)
    sample_device = target_device if generator_device == target_device else generator_device
    noise = torch.randint(0, 21, shape, device=sample_device, generator=generator)
    if noise.device != target_device:
        noise = noise.to(target_device)
    return noise


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


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=pred.device)
    if mask.any():
        return F.smooth_l1_loss(pred[mask], target[mask])
    return pred.sum() * 0.0


def _masked_cosine_distance(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=pred.device)
    if mask.any():
        cosine = F.cosine_similarity(pred[mask], target[mask], dim=-1)
        return (1.0 - cosine).mean()
    return pred.sum() * 0.0


def _masked_smooth_l1_with_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    mask = mask.to(dtype=torch.bool, device=pred.device)
    count = float(mask.sum().item())
    if mask.any():
        return F.smooth_l1_loss(pred[mask], target[mask]), count
    return pred.sum() * 0.0, count


def _masked_cosine_distance_with_count(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    mask = mask.to(dtype=torch.bool, device=pred.device)
    count = float(mask.sum().item())
    if mask.any():
        cosine = F.cosine_similarity(pred[mask], target[mask], dim=-1)
        return (1.0 - cosine).mean(), count
    return pred.sum() * 0.0, count


def _cross_entropy_with_count(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    values = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    mask = mask.to(dtype=torch.bool, device=target.device)
    count = float(mask.sum().item())
    if mask.any():
        return values[mask].mean(), count
    return logits.sum() * 0.0, count


class LocalGeometricVQPretrainer(nn.Module):
    def __init__(
        self,
        node_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        structure_cutoff: float = 12.0,
        interface_cutoff: float = 10.0,
        structure_k: int = 16,
        interface_k: int = 16,
        codebook_size: int = 256,
        decoder_layers: int = 2,
        lambda_type: float = 0.3,
        lambda_angle: float = 1.0,
        lambda_geometry: float = 1.0,
        lambda_rsa: float = 0.2,
        lambda_vq: float = 0.005,
        commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.lambda_type = lambda_type
        self.lambda_angle = lambda_angle
        self.lambda_geometry = lambda_geometry
        self.lambda_rsa = lambda_rsa
        self.lambda_vq = lambda_vq

        self.encoder = LocalGeometricEncoder(
            node_dim=node_dim,
            num_blocks=num_blocks,
            num_heads=num_heads,
            structure_cutoff=structure_cutoff,
            interface_cutoff=interface_cutoff,
            structure_k=structure_k,
            interface_k=interface_k,
        )
        self.type_projector = nn.Linear(node_dim, node_dim)
        self.angle_projector = nn.Linear(node_dim, node_dim)
        self.geometry_projector = nn.Linear(node_dim, node_dim)

        self.vq_type = StandardVectorQuantizer(node_dim, codebook_size, commitment_cost=commitment_cost)
        self.vq_angle = StandardVectorQuantizer(node_dim, codebook_size, commitment_cost=commitment_cost)
        self.vq_geometry = StandardVectorQuantizer(node_dim, codebook_size, commitment_cost=commitment_cost)

        self.type_decoder = SparseFeatureDecoder(node_dim=node_dim, num_layers=decoder_layers)
        self.angle_decoder = SparseFeatureDecoder(node_dim=node_dim, num_layers=decoder_layers)
        self.geometry_decoder = IterativeGeometricDecoder(node_dim=node_dim, num_layers=decoder_layers)

        self.type_head = nn.Linear(node_dim, 21)
        self.angle_head = nn.Linear(node_dim, 7)
        self.rsa_head = nn.Linear(node_dim, 1)

        self.last_assignments: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.last_quantized: dict[str, torch.Tensor] = {}
        self.last_encoder_h: torch.Tensor | None = None

    def _candidate_mask(self, batch: dict) -> torch.Tensor:
        pos_atoms = batch["pos_atoms"]
        device = pos_atoms.device
        shape = pos_atoms.shape[:2]
        node_mask = _prepare_mask(batch.get("node_mask"), shape, device, default=True)
        interface_value = batch.get("interface_mask")
        interface_mask = _prepare_mask(interface_value, shape, device, default=False)
        if interface_value is None:
            interface_mask = node_mask

        mask_atoms_value = batch.get("mask_atoms")
        if mask_atoms_value is None:
            ca_mask = torch.ones(shape, dtype=torch.bool, device=device)
        else:
            ca_mask = mask_atoms_value.to(device=device, dtype=torch.bool)[:, :, 1]
        return node_mask & interface_mask & ca_mask

    def _branch_masks(self, batch: dict, masks: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        candidate = self._candidate_mask(batch)
        branch_masks = {}
        for name in ("type", "angle", "geometry", "rsa"):
            if name not in masks:
                raise KeyError(f"masks must contain '{name}'")
            branch_masks[name] = masks[name].to(device=candidate.device, dtype=torch.bool) & candidate
        return candidate, branch_masks

    def _build_neighbors(self, batch: dict) -> ComplexNeighborLists:
        return build_neighbor_lists(
            batch,
            structure_cutoff=self.encoder.structure_cutoff,
            interface_cutoff=self.encoder.interface_cutoff,
            structure_k=self.encoder.structure_k,
            interface_k=self.encoder.interface_k,
        )

    def _corrupt_encoder_inputs(
        self,
        batch: dict,
        masks: dict[str, torch.Tensor],
        angle_target: torch.Tensor,
        generator: torch.Generator | None,
    ) -> tuple[dict, torch.Tensor]:
        corrupted = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        aa = batch["aa"].to(device=angle_target.device, dtype=torch.long)
        aa_noise = _sample_aa_noise(aa.shape, aa.device, generator)
        corrupted["aa"] = aa.clone()
        corrupted["aa"][masks["type"]] = aa_noise[masks["type"]]

        angle_features = angle_target.clone()
        angle_features[masks["angle"]] = 0.0
        return corrupted, angle_features

    def _quantize_branch(
        self,
        projector: nn.Module,
        quantizer: StandardVectorQuantizer,
        h: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = projector(h)
        quantized_raw, loss_vq, indices = quantizer(projected, candidate_mask)
        quantized = torch.where(candidate_mask.unsqueeze(-1), quantized_raw, projected)
        return quantized, loss_vq, indices

    def decode_with_quantized(
        self,
        batch: dict,
        q_type: torch.Tensor,
        q_angle: torch.Tensor,
        q_geometry: torch.Tensor,
        geometry_mask: torch.Tensor,
        encoder_h: torch.Tensor | None = None,
        neighbors: ComplexNeighborLists | None = None,
    ) -> dict[str, torch.Tensor]:
        if encoder_h is None:
            if self.last_encoder_h is None:
                raise ValueError("encoder_h is required before the first forward pass")
            encoder_h = self.last_encoder_h
        if neighbors is None:
            neighbors = self._build_neighbors(batch)

        type_hidden = self.type_decoder(q_type, batch, neighbors=neighbors)
        angle_hidden = self.angle_decoder(q_angle, batch, neighbors=neighbors)
        geometry_input, geometry_target, geometry_supervision = initialize_geometry_state(
            batch,
            geometry_mask,
            neighbors=neighbors,
        )
        geometry_state = self.geometry_decoder(q_geometry, geometry_input)
        return {
            "type_logits": self.type_head(type_hidden),
            "angle_logits": self.angle_head(angle_hidden),
            "rsa_pred": self.rsa_head(encoder_h).squeeze(-1),
            "geometry": geometry_state,
            "geometry_target": geometry_target,
            "geometry_supervision": geometry_supervision,
        }

    def _geometry_loss_breakdown_from_decoded(self, decoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | float]:
        geometry_state = decoded["geometry"]
        geometry_target = decoded["geometry_target"]
        geometry_supervision = decoded["geometry_supervision"]

        loss_local, local_count = _masked_smooth_l1_with_count(
            geometry_state.local_atoms,
            geometry_target.local_atoms.to(geometry_state.local_atoms.device),
            geometry_supervision.local_atoms,
        )
        loss_structure_distance, structure_distance_count = _masked_smooth_l1_with_count(
            geometry_state.structure.distance,
            geometry_target.structure.distance.to(geometry_state.structure.distance.device),
            geometry_supervision.structure,
        )
        loss_interface_distance, interface_distance_count = _masked_smooth_l1_with_count(
            geometry_state.interface.distance,
            geometry_target.interface.distance.to(geometry_state.interface.distance.device),
            geometry_supervision.interface,
        )
        loss_distance = loss_structure_distance + loss_interface_distance

        loss_structure_direction, structure_direction_count = _masked_cosine_distance_with_count(
            geometry_state.structure.direction,
            geometry_target.structure.direction.to(geometry_state.structure.direction.device),
            geometry_supervision.structure,
        )
        loss_interface_direction, interface_direction_count = _masked_cosine_distance_with_count(
            geometry_state.interface.direction,
            geometry_target.interface.direction.to(geometry_state.interface.direction.device),
            geometry_supervision.interface,
        )
        loss_direction = loss_structure_direction + loss_interface_direction
        loss_geometry = loss_local + 0.5 * loss_distance + 0.5 * loss_direction
        return {
            "loss_local": loss_local,
            "local_count": local_count,
            "loss_structure_distance": loss_structure_distance,
            "structure_distance_count": structure_distance_count,
            "loss_interface_distance": loss_interface_distance,
            "interface_distance_count": interface_distance_count,
            "loss_distance": loss_distance,
            "loss_structure_direction": loss_structure_direction,
            "structure_direction_count": structure_direction_count,
            "loss_interface_direction": loss_interface_direction,
            "interface_direction_count": interface_direction_count,
            "loss_direction": loss_direction,
            "loss_geometry": loss_geometry,
        }

    def geometry_loss_breakdown_with_quantized(
        self,
        batch: dict,
        q_geometry: torch.Tensor,
        geometry_mask: torch.Tensor,
        q_type: torch.Tensor | None = None,
        q_angle: torch.Tensor | None = None,
        encoder_h: torch.Tensor | None = None,
        neighbors: ComplexNeighborLists | None = None,
    ) -> dict[str, float]:
        if q_type is None:
            q_type = self.last_quantized["type"]
        if q_angle is None:
            q_angle = self.last_quantized["angle"]

        decoded = self.decode_with_quantized(
            batch,
            q_type=q_type,
            q_angle=q_angle,
            q_geometry=q_geometry,
            geometry_mask=geometry_mask,
            encoder_h=encoder_h,
            neighbors=neighbors,
        )
        breakdown = self._geometry_loss_breakdown_from_decoded(decoded)
        out: dict[str, float] = {}
        for key, value in breakdown.items():
            out[key] = _float_log(value) if torch.is_tensor(value) else float(value)
        return out

    def geometry_loss_with_quantized(
        self,
        batch: dict,
        q_geometry: torch.Tensor,
        geometry_mask: torch.Tensor,
        q_type: torch.Tensor | None = None,
        q_angle: torch.Tensor | None = None,
        encoder_h: torch.Tensor | None = None,
        neighbors: ComplexNeighborLists | None = None,
    ) -> torch.Tensor:
        if q_type is None:
            q_type = self.last_quantized["type"]
        if q_angle is None:
            q_angle = self.last_quantized["angle"]

        decoded = self.decode_with_quantized(
            batch,
            q_geometry=q_geometry,
            geometry_mask=geometry_mask,
            q_type=q_type,
            q_angle=q_angle,
            encoder_h=encoder_h,
            neighbors=neighbors,
        )
        breakdown = self._geometry_loss_breakdown_from_decoded(decoded)
        return breakdown["loss_geometry"]

    def forward(
        self,
        batch: dict,
        masks: dict[str, torch.Tensor],
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        candidate_mask, branch_masks = self._branch_masks(batch, masks)
        angle_target, angle_valid = chain_aware_angle_targets(batch)
        corrupted_batch, input_angle_features = self._corrupt_encoder_inputs(batch, branch_masks, angle_target, generator)
        neighbors = self._build_neighbors(batch)
        h, _interface = self.encoder(corrupted_batch, angle_features=(input_angle_features, angle_valid), neighbors=neighbors)

        q_type, loss_vq_type, type_indices = self._quantize_branch(self.type_projector, self.vq_type, h, candidate_mask)
        q_angle, loss_vq_angle, angle_indices = self._quantize_branch(
            self.angle_projector,
            self.vq_angle,
            h,
            candidate_mask,
        )
        q_geometry, loss_vq_geometry, geometry_indices = self._quantize_branch(
            self.geometry_projector,
            self.vq_geometry,
            h,
            candidate_mask,
        )

        decoded = self.decode_with_quantized(
            batch,
            q_type=q_type,
            q_angle=q_angle,
            q_geometry=q_geometry,
            geometry_mask=branch_masks["geometry"],
            encoder_h=h,
            neighbors=neighbors,
        )

        type_logits = decoded["type_logits"]
        angle_logits = decoded["angle_logits"]
        rsa_pred = decoded["rsa_pred"]
        geometry_state = decoded["geometry"]
        geometry_target = decoded["geometry_target"]
        geometry_supervision = decoded["geometry_supervision"]

        aa_target = batch["aa"].to(device=type_logits.device, dtype=torch.long)
        loss_type, type_count = _cross_entropy_with_count(type_logits, aa_target, branch_masks["type"])

        angle_supervision = angle_valid & branch_masks["angle"].unsqueeze(-1)
        loss_angle, angle_count = _masked_smooth_l1_with_count(
            angle_logits,
            angle_target.to(angle_logits.device),
            angle_supervision,
        )

        geometry_breakdown = self._geometry_loss_breakdown_from_decoded(decoded)
        loss_local = geometry_breakdown["loss_local"]
        loss_structure_distance = geometry_breakdown["loss_structure_distance"]
        loss_interface_distance = geometry_breakdown["loss_interface_distance"]
        loss_distance = geometry_breakdown["loss_distance"]
        loss_structure_direction = geometry_breakdown["loss_structure_direction"]
        loss_interface_direction = geometry_breakdown["loss_interface_direction"]
        loss_direction = geometry_breakdown["loss_direction"]
        loss_geometry = geometry_breakdown["loss_geometry"]
        local_count = float(geometry_breakdown["local_count"])
        structure_distance_count = float(geometry_breakdown["structure_distance_count"])
        interface_distance_count = float(geometry_breakdown["interface_distance_count"])
        structure_direction_count = float(geometry_breakdown["structure_direction_count"])
        interface_direction_count = float(geometry_breakdown["interface_direction_count"])

        rsa_valid = batch.get("rsa_valid")
        if rsa_valid is None:
            rsa_valid = torch.zeros_like(branch_masks["rsa"])
        rsa_valid = rsa_valid.to(device=rsa_pred.device, dtype=torch.bool)
        rsa_target = batch.get("rsa")
        if rsa_target is None:
            rsa_target = torch.zeros_like(rsa_pred)
        else:
            rsa_target = rsa_target.to(device=rsa_pred.device, dtype=rsa_pred.dtype)
        rsa_supervision = branch_masks["rsa"] & rsa_valid
        loss_rsa, rsa_count = _masked_smooth_l1_with_count(rsa_pred, rsa_target, rsa_supervision)

        loss_vq_total = loss_vq_type + loss_vq_angle + loss_vq_geometry
        total_loss = (
            self.lambda_type * loss_type
            + self.lambda_angle * loss_angle
            + self.lambda_geometry * loss_geometry
            + self.lambda_rsa * loss_rsa
            + self.lambda_vq * loss_vq_total
        )

        assignments = {
            "type": (type_indices, candidate_mask),
            "angle": (angle_indices, candidate_mask),
            "geometry": (geometry_indices, candidate_mask),
        }
        self.last_assignments = assignments
        self.last_quantized = {
            "type": q_type,
            "angle": q_angle,
            "geometry": q_geometry,
        }
        self.last_encoder_h = h

        logs = {
            "loss_type": _float_log(loss_type),
            "loss_angle": _float_log(loss_angle),
            "loss_local": _float_log(loss_local),
            "loss_structure_distance": _float_log(loss_structure_distance),
            "loss_interface_distance": _float_log(loss_interface_distance),
            "loss_distance": _float_log(loss_distance),
            "loss_structure_direction": _float_log(loss_structure_direction),
            "loss_interface_direction": _float_log(loss_interface_direction),
            "loss_direction": _float_log(loss_direction),
            "loss_geometry": _float_log(loss_geometry),
            "loss_rsa": _float_log(loss_rsa),
            "loss_vq_type": _float_log(loss_vq_type),
            "loss_vq_angle": _float_log(loss_vq_angle),
            "loss_vq_geometry": _float_log(loss_vq_geometry),
            "loss_vq_total": _float_log(loss_vq_total),
            "total_loss": _float_log(total_loss),
            "type_count": type_count,
            "angle_count": angle_count,
            "local_count": local_count,
            "structure_distance_count": structure_distance_count,
            "interface_distance_count": interface_distance_count,
            "structure_direction_count": structure_direction_count,
            "interface_direction_count": interface_direction_count,
            "rsa_count": rsa_count,
            "vq_type_candidate_count": float(candidate_mask.sum().item()),
            "vq_angle_candidate_count": float(candidate_mask.sum().item()),
            "vq_geometry_candidate_count": float(candidate_mask.sum().item()),
            "mask_type_count": float(branch_masks["type"].sum().item()),
            "mask_angle_count": float(branch_masks["angle"].sum().item()),
            "mask_geometry_count": float(branch_masks["geometry"].sum().item()),
            "mask_rsa_count": float(branch_masks["rsa"].sum().item()),
            "angle_supervised_count": angle_count,
            "local_supervised_count": local_count,
            "distance_supervised_count": structure_distance_count + interface_distance_count,
            "direction_supervised_count": structure_direction_count + interface_direction_count,
            "rsa_supervised_count": rsa_count,
            "vq_candidate_count": float(candidate_mask.sum().item()),
        }
        return total_loss, logs, assignments
