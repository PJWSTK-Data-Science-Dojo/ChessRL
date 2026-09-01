"""Losses and latent diagnostics used by the EfficientZeroV2 learner."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from luna.ezv2_networks import SimSiamProjector


def soft_ce_with_support(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1)


def piece_class_targets(observation: torch.Tensor) -> torch.Tensor:
    if observation.ndim != 4 or observation.shape[-1] < 12:
        raise ValueError(
            "Piece reconstruction observations must have shape (batch, 8, 8, planes>=12), "
            f"got {tuple(observation.shape)}"
        )
    piece_planes = observation[..., :12]
    occupied = piece_planes.amax(dim=-1) > 0.5
    piece_class = piece_planes.argmax(dim=-1) + 1
    return torch.where(occupied, piece_class, torch.zeros_like(piece_class)).long()


def piece_reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expected_shape = (target.shape[0], 13, target.shape[1], target.shape[2])
    if tuple(logits.shape) != expected_shape:
        raise ValueError(f"Piece reconstruction logits must have shape {expected_shape}, got {tuple(logits.shape)}")
    per_square = F.cross_entropy(logits, target, reduction="none")
    occupied = (target > 0).to(per_square.dtype)
    empty = 1.0 - occupied
    occupied_loss = (per_square * occupied).sum(dim=(-2, -1)) / occupied.sum(dim=(-2, -1)).clamp(min=1.0)
    empty_loss = (per_square * empty).sum(dim=(-2, -1)) / empty.sum(dim=(-2, -1)).clamp(min=1.0)
    return 0.5 * (occupied_loss + empty_loss)


def raw_latent_health_metrics(prefix: str, latent: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        detached = latent.detach().float()
        flattened = detached.flatten(1)
        return {
            f"train/latent_{prefix}_batch_feature_std": float(flattened.std(dim=0, unbiased=False).mean().item()),
            f"train/latent_{prefix}_spatial_std": float(detached.std(dim=(-2, -1), unbiased=False).mean().item()),
        }


def piece_reconstruction_accuracy_metrics(
    prefix: str,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        prediction = logits.detach().argmax(dim=1)
        correct = prediction == target
        occupied = target > 0
        occupied_total = int(occupied.sum().item())
        occupied_accuracy = float(correct[occupied].float().mean().item()) if occupied_total > 0 else 0.0
        return {
            f"train/reconstruction_{prefix}_accuracy": float(correct.float().mean().item()),
            f"train/reconstruction_{prefix}_occupied_accuracy": occupied_accuracy,
        }


def simsiam_loss(
    projector: SimSiamProjector,
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
) -> torch.Tensor:
    predicted_projection = projector.project(predicted_latent)
    prediction = projector.predict(predicted_projection)
    with torch.no_grad():
        target_projection = projector.project(target_latent)
    prediction = F.normalize(prediction, dim=1)
    target_projection = F.normalize(target_projection, dim=1)
    return 1.0 - (prediction * target_projection).sum(dim=1)


def latent_health_metrics(
    projector: SimSiamProjector,
    predicted_latent: torch.Tensor,
    target_latent: torch.Tensor,
) -> dict[str, float]:
    with torch.no_grad():
        predicted = predicted_latent.detach().float()
        target = target_latent.detach().float()
        projected_predicted = projector.project(predicted_latent.detach()).float()
        projected_target = projector.project(target_latent.detach()).float()
        normalized_predicted = F.normalize(projected_predicted, dim=1)
        normalized_target = F.normalize(projected_target, dim=1)
        alignment = (normalized_predicted * normalized_target).sum(dim=1).mean()
        projector_batch_std = projected_target.std(dim=0, unbiased=False).mean()
        off_diagonal = _off_diagonal_cosine(normalized_target)
        return {
            **raw_latent_health_metrics("predicted", predicted),
            **raw_latent_health_metrics("target", target),
            "train/projector_target_batch_std": float(projector_batch_std.item()),
            "train/projector_target_offdiag_cosine": float(off_diagonal.item()),
            "train/consistency_cosine_alignment": float(alignment.item()),
        }


def _off_diagonal_cosine(normalized: torch.Tensor) -> torch.Tensor:
    batch = normalized.shape[0]
    if batch <= 1:
        return torch.zeros((), device=normalized.device)
    similarities = normalized @ normalized.T
    off_diagonal_sum = similarities.abs().sum() - similarities.diagonal().abs().sum()
    return off_diagonal_sum / (batch * (batch - 1))
