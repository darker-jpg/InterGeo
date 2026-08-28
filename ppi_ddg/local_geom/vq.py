from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AssignmentStatistics", "StandardVectorQuantizer", "assignment_statistics"]


@dataclass(frozen=True)
class AssignmentStatistics:
    histogram: torch.Tensor
    active_codes: int
    perplexity: torch.Tensor
    max_share: torch.Tensor
    total: int


class StandardVectorQuantizer(nn.Module):
    def __init__(self, embedding_dim: int, num_embeddings: int, commitment_cost: float = 0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_norm = F.normalize(z, p=2, dim=-1)
        weight_norm = F.normalize(self.embedding.weight, p=2, dim=-1)

        distances = (
            z_norm.pow(2).sum(dim=-1, keepdim=True)
            + weight_norm.pow(2).sum(dim=-1).view(*((1,) * (z_norm.ndim - 1)), -1)
            - 2.0 * torch.matmul(z_norm, weight_norm.t())
        )
        indices = torch.argmin(distances, dim=-1)
        quantized = F.normalize(self.embedding(indices), p=2, dim=-1)

        final_mask = mask.to(dtype=torch.bool, device=z.device)
        if final_mask.any():
            q_loss = F.mse_loss(quantized[final_mask], z_norm.detach()[final_mask])
            e_loss = F.mse_loss(z_norm[final_mask], quantized.detach()[final_mask])
            loss = q_loss + self.commitment_cost * e_loss
        else:
            loss = z_norm.sum() * 0.0

        quantized = z_norm + (quantized - z_norm).detach()
        return quantized, loss, indices


def assignment_statistics(indices: torch.Tensor, mask: torch.Tensor, K: int) -> AssignmentStatistics:
    final_mask = mask.to(dtype=torch.bool, device=indices.device)
    histogram = torch.bincount(indices[final_mask].reshape(-1), minlength=K)
    total = int(final_mask.sum().item())
    if total > 0:
        probs = histogram.to(dtype=torch.float32) / float(total)
        active = histogram > 0
        entropy = -(probs[active] * probs[active].log()).sum()
        perplexity = entropy.exp()
        max_share = probs.max()
        active_codes = int(active.sum().item())
    else:
        perplexity = torch.zeros((), device=indices.device, dtype=torch.float32)
        max_share = torch.zeros((), device=indices.device, dtype=torch.float32)
        active_codes = 0
    return AssignmentStatistics(
        histogram=histogram,
        active_codes=active_codes,
        perplexity=perplexity,
        max_share=max_share,
        total=total,
    )
