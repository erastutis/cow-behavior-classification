from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        prob = torch.sigmoid(logits)
        pt = prob * targets + (1 - prob) * (1 - targets)
        focal = (1 - pt).pow(self.gamma) * bce
        return focal.mean()


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, eps: float = 1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        pos_loss = targets * torch.log(prob.clamp(min=self.eps)) * (1 - prob).pow(self.gamma_pos)
        neg_loss = (1 - targets) * torch.log((1 - prob).clamp(min=self.eps)) * prob.pow(self.gamma_neg)
        return -(pos_loss + neg_loss).mean()


class CombinedLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.asl = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0)
        self.focal = FocalLoss(gamma=2.0, pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return 0.5 * self.asl(logits, targets) + 0.5 * self.focal(logits, targets)


def compute_pos_weight(labels: torch.Tensor, max_weight: float | None = None) -> torch.Tensor:
    pos = labels.sum(dim=0).clamp(min=1)
    neg = labels.shape[0] - pos
    w = neg / pos
    if max_weight is not None:
        w = torch.clamp(w, max=max_weight)
    return w
