"""Inference paths used by latent MCTS and external engine adapters."""

from __future__ import annotations

import numpy as np
import torch

from luna.ezv2_networks import action_index_to_planes
from luna.network_runtime import pinned_h2d_float32
from luna.network_types import NetworkRuntime, RecurrentBatchResult


def persist_compiled_latent(network: NetworkRuntime, latent: torch.Tensor) -> torch.Tensor:
    """Detach outputs from CUDA Graph buffers retained across compiled calls."""
    return latent.clone() if network._mcts_inference_compiled else latent


def encode_action_planes(network: NetworkRuntime, actions: torch.Tensor) -> torch.Tensor:
    if network._action_plane_lookup is None:
        return action_index_to_planes(actions, network.device)
    return torch.index_select(network._action_plane_lookup, 0, actions)


def batched_initial_inference(
    network: NetworkRuntime,
    observations: np.ndarray,
    valid_moves: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    board_tensor = pinned_h2d_float32(observations.astype(np.float32, copy=False), network.device)
    valid_tensor = pinned_h2d_float32(valid_moves.astype(np.float32, copy=False), network.device)
    network.nnet.eval()
    with torch.inference_mode(), _autocast(network):
        latent, log_policy, values = network._mcts_initial_inference(board_tensor, valid_tensor)
    return (
        torch.exp(log_policy).float().cpu().numpy(),
        values.float().cpu().numpy(),
        persist_compiled_latent(network, latent),
    )


def batched_recurrent_inference(
    network: NetworkRuntime,
    latents: torch.Tensor,
    actions: list[int],
    *,
    valid_masks: list[np.ndarray | None] | None = None,
    policy_topk: int | None = None,
) -> RecurrentBatchResult:
    action_tensor = torch.as_tensor(actions, dtype=torch.long, device=network.device)
    action_planes = encode_action_planes(network, action_tensor)
    valid_tensor = _stack_valid_masks(network, valid_masks)
    network.nnet.eval()
    with torch.inference_mode(), _autocast(network):
        next_latent, reward, log_policy, value = network._mcts_recurrent_inference(
            latents,
            action_planes,
            valid_tensor,
        )
    policy_width = int(log_policy.shape[1])
    topk = _policy_topk(policy_width, policy_topk, valid_masks)
    values = value.float().cpu().numpy()
    rewards = reward.float().cpu().numpy()
    persistent_latent = persist_compiled_latent(network, next_latent)
    if topk >= policy_width:
        policies = torch.exp(log_policy).float().cpu().numpy()
        return RecurrentBatchResult(policies, None, None, values, rewards, persistent_latent)
    top_log_policy, top_indices = torch.topk(log_policy, k=topk, dim=1)
    probabilities = torch.softmax(top_log_policy.float(), dim=1)
    return RecurrentBatchResult(
        None,
        top_indices.cpu().numpy().astype(np.int32),
        probabilities.cpu().numpy().astype(np.float32),
        values,
        rewards,
        persistent_latent,
    )


def predict_with_latent(
    network: NetworkRuntime,
    board: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, float, torch.Tensor]:
    board_tensor = torch.as_tensor(board, dtype=torch.float32, device=network.device).view(
        1,
        network.board_x,
        network.board_y,
        network.board_z,
    )
    valid_tensor = torch.as_tensor(valid, dtype=torch.float32, device=network.device)
    if valid_tensor.dim() == 1:
        valid_tensor = valid_tensor.unsqueeze(0)
    network.nnet.eval()
    with torch.inference_mode(), _autocast(network):
        latent, log_policy, value = network._mcts_initial_inference(board_tensor, valid_tensor)
    return (
        torch.exp(log_policy).float().cpu().numpy()[0],
        float(value.item()),
        persist_compiled_latent(network, latent),
    )


def recurrent_predict(
    network: NetworkRuntime,
    latent: torch.Tensor,
    action: int,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, float, torch.Tensor]:
    action_tensor = torch.tensor([action], device=network.device)
    action_planes = encode_action_planes(network, action_tensor)
    valid_tensor = None
    if valid_mask is not None:
        valid_tensor = torch.as_tensor(valid_mask, dtype=torch.float32, device=network.device).unsqueeze(0)
    network.nnet.eval()
    with torch.inference_mode(), _autocast(network):
        next_latent, reward, log_policy, value = network._mcts_recurrent_inference(
            latent,
            action_planes,
            valid_tensor,
        )
    return (
        torch.exp(log_policy).float().cpu().numpy()[0],
        float(value.item()),
        float(reward.item()),
        persist_compiled_latent(network, next_latent),
    )


def _autocast(network: NetworkRuntime) -> torch.autocast:
    return torch.autocast(
        "cuda",
        enabled=network._learner.mixed_precision and network.device.type == "cuda",
        dtype=network._amp_dtype,
    )


def _stack_valid_masks(
    network: NetworkRuntime,
    valid_masks: list[np.ndarray | None] | None,
) -> torch.Tensor | None:
    if not valid_masks:
        return None
    masks = np.ones((len(valid_masks), network.action_size), dtype=np.float32)
    for index, mask in enumerate(valid_masks):
        if mask is not None:
            masks[index] = mask
    return torch.as_tensor(masks, dtype=torch.float32, device=network.device)


def _policy_topk(
    policy_width: int,
    requested: int | None,
    valid_masks: list[np.ndarray | None] | None,
) -> int:
    limit = requested if requested is not None else policy_width
    if valid_masks and all(mask is not None for mask in valid_masks):
        limit = max(int(np.count_nonzero(mask)) for mask in valid_masks if mask is not None)
    elif valid_masks:
        max_legal = max((int(np.count_nonzero(mask)) for mask in valid_masks if mask is not None), default=0)
        limit = max(limit, max_legal)
    return policy_width if limit <= 0 else min(limit, policy_width)
