"""Neural Network Wrapper -- EfficientZeroV2 learner with unroll training."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast

from .ezv2_model import EZV2Networks, _scale_latent, scalar_to_support
from .game.luna_game import ChessGame
from .replay_buffer import PrioritizedReplayBuffer
from .targets import build_unroll_targets, collate_batch
from .utils import AverageMeter, dotdict

log = logging.getLogger(__name__)

NNET_ARGS = dotdict(
    {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "epochs": 20,
        "batch_size": 64,
        "num_channels": 128,
        "support_size": 10,
        "repr_blocks": 4,
        "dyn_blocks": 2,
        "mixed_precision": True,
        "unroll_steps": 5,
        "td_steps": 10,
        "discount": 0.997,
        "policy_loss_weight": 1.0,
        "value_loss_weight": 0.25,
        "reward_loss_weight": 1.0,
        "consistency_loss_weight": 2.0,
    }
)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LunaNetwork:
    """EfficientZeroV2 learner with persistent optimizer, mixed-precision, and unroll training."""

    def __init__(self, game: ChessGame) -> None:
        self.device = _get_device()
        self.board_x, self.board_y, self.board_z = game.get_board_size()
        self.action_size = game.get_action_size()

        self.nnet = EZV2Networks(game, NNET_ARGS).to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=NNET_ARGS.lr, weight_decay=NNET_ARGS.weight_decay)
        self.scaler = GradScaler("cuda", enabled=NNET_ARGS.mixed_precision and self.device.type == "cuda")

    def train_ezv2(self, replay: PrioritizedReplayBuffer, steps: int) -> dict[str, float]:
        self.nnet.train()

        total_loss_m = AverageMeter()
        pi_loss_m = AverageMeter()
        v_loss_m = AverageMeter()
        r_loss_m = AverageMeter()
        consist_loss_m = AverageMeter()
        step_time_m = AverageMeter()

        unroll = NNET_ARGS.unroll_steps
        td = NNET_ARGS.td_steps
        bs = NNET_ARGS.batch_size
        support = NNET_ARGS.support_size

        for step in range(1, steps + 1):
            t0 = time.time()
            batch, is_weights, tree_indices = replay.sample(bs, unroll)

            batch_targets = [
                build_unroll_targets(traj, pos, unroll, td, NNET_ARGS.discount) for traj, pos in batch
            ]
            collated = collate_batch(batch_targets)

            obs = torch.as_tensor(collated["observations"], dtype=torch.float32, device=self.device)
            valid = torch.as_tensor(collated["valid_masks"], dtype=torch.float32, device=self.device)
            t_values = torch.as_tensor(collated["target_values"], dtype=torch.float32, device=self.device)
            t_rewards = torch.as_tensor(collated["target_rewards"], dtype=torch.float32, device=self.device)
            t_policies = torch.as_tensor(collated["target_policies"], dtype=torch.float32, device=self.device)
            obs_unroll = torch.as_tensor(collated["observations_unroll"], dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(collated["actions"], dtype=torch.long, device=self.device)
            is_w = torch.as_tensor(is_weights, dtype=torch.float32, device=self.device)

            with autocast("cuda", enabled=self.scaler.is_enabled()):
                latent, log_pi_0, _ = self.nnet.initial_inference_with_latent(obs, valid)

                loss_pi = -(t_policies[:, 0] * log_pi_0).sum(dim=1)

                v_target_0 = scalar_to_support(t_values[:, 0], support).to(self.device)
                loss_v_pred = _soft_ce_with_support(self._raw_value_logits(latent), v_target_0)

                loss_r_total = torch.zeros(bs, device=self.device)
                loss_pi_total = loss_pi
                loss_v_total = loss_v_pred
                loss_consist_total = torch.zeros(bs, device=self.device)

                current_latent = latent
                for k in range(unroll):
                    action_oh = F.one_hot(actions[:, k], self.action_size).float()
                    next_latent_raw, r_logits = self.nnet.dynamics(current_latent, action_oh)
                    next_latent = _scale_latent(next_latent_raw)
                    policy_logits_k, _ = self.nnet.prediction(next_latent)
                    log_pi_k = F.log_softmax(policy_logits_k, dim=1)

                    r_target = scalar_to_support(t_rewards[:, k], support).to(self.device)
                    loss_r = _soft_ce_with_support(r_logits, r_target)

                    loss_pi_k = -(t_policies[:, k + 1] * log_pi_k).sum(dim=1)

                    v_target_k = scalar_to_support(t_values[:, k + 1], support).to(self.device)
                    loss_v_k = _soft_ce_with_support(self._raw_value_logits(next_latent), v_target_k)

                    obs_target_k = obs_unroll[:, k + 1]
                    target_latent = self.nnet.representation(self.nnet._obs_to_planes(obs_target_k)).detach()
                    target_latent = _scale_latent(target_latent)
                    loss_consist = F.mse_loss(next_latent, target_latent, reduction="none").mean(dim=[1, 2, 3])

                    loss_r_total = loss_r_total + loss_r
                    loss_pi_total = loss_pi_total + loss_pi_k
                    loss_v_total = loss_v_total + loss_v_k
                    loss_consist_total = loss_consist_total + loss_consist

                    current_latent = next_latent

                scale = 1.0 / (unroll + 1)
                total = (
                    NNET_ARGS.policy_loss_weight * loss_pi_total
                    + NNET_ARGS.value_loss_weight * loss_v_total
                    + NNET_ARGS.reward_loss_weight * loss_r_total
                    + NNET_ARGS.consistency_loss_weight * loss_consist_total
                ) * scale

                weighted = (total * is_w).mean()

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(weighted).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            td_errors = total.detach().cpu().numpy()
            replay.update_priorities(tree_indices, td_errors)

            total_loss_m.update(weighted.item(), bs)
            pi_loss_m.update(loss_pi_total.mean().item(), bs)
            v_loss_m.update(loss_v_total.mean().item(), bs)
            r_loss_m.update(loss_r_total.mean().item(), bs)
            consist_loss_m.update(loss_consist_total.mean().item(), bs)
            step_time_m.update(time.time() - t0)

            if step % 50 == 0 or step == steps:
                log.info(
                    "(step %d/%d) %.3fs | loss=%.4f pi=%.4f v=%.4f r=%.4f c=%.4f",
                    step,
                    steps,
                    step_time_m.avg,
                    total_loss_m.avg,
                    pi_loss_m.avg,
                    v_loss_m.avg,
                    r_loss_m.avg,
                    consist_loss_m.avg,
                )

        return {
            "total": total_loss_m.avg,
            "policy": pi_loss_m.avg,
            "value": v_loss_m.avg,
            "reward": r_loss_m.avg,
            "consistency": consist_loss_m.avg,
        }

    def _raw_value_logits(self, latent: torch.Tensor) -> torch.Tensor:
        return self.nnet.prediction.value_head(latent)

    def train(self, examples: list[Any]) -> None:
        for epoch in range(NNET_ARGS.epochs):
            log.info("EPOCH %d / %d", epoch + 1, NNET_ARGS.epochs)
            self.nnet.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()
            batch_time = AverageMeter()
            end = time.time()

            n_batches = max(1, len(examples) // NNET_ARGS.batch_size)
            for batch_idx in range(n_batches):
                sample_ids = np.random.randint(len(examples), size=NNET_ARGS.batch_size)
                boards, pis, vs, valids = list(zip(*[examples[i] for i in sample_ids]))

                boards_t = torch.as_tensor(np.array(boards), dtype=torch.float32, device=self.device)
                target_pis = torch.as_tensor(np.array(pis), dtype=torch.float32, device=self.device)
                target_vs = torch.as_tensor(np.array(vs), dtype=torch.float32, device=self.device)
                target_valids = torch.as_tensor(np.array(valids), dtype=torch.float32, device=self.device)

                with autocast("cuda", enabled=self.scaler.is_enabled()):
                    out_pi, out_v = self.nnet.initial_inference(boards_t, target_valids)
                    l_pi = self.loss_pi(target_pis, out_pi)
                    l_v = self.loss_v(target_vs, out_v)
                    total_loss = l_pi + l_v

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                pi_losses.update(l_pi.item(), boards_t.size(0))
                v_losses.update(l_v.item(), boards_t.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if (batch_idx + 1) % 50 == 0 or batch_idx == n_batches - 1:
                    log.info(
                        "(%d: %d/%d) Batch: %.3fs | Loss_pi: %.4f | Loss_v: %.4f",
                        epoch + 1,
                        batch_idx + 1,
                        n_batches,
                        batch_time.avg,
                        pi_losses.avg,
                        v_losses.avg,
                    )

    def predict(self, board_and_valid: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        board, valid = board_and_valid
        board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device)
        valid_t = torch.as_tensor(valid, dtype=torch.float32, device=self.device)
        board_t = board_t.view(1, self.board_x, self.board_y, self.board_z)

        self.nnet.eval()
        with torch.no_grad():
            pi, v = self.nnet.initial_inference(board_t, valid_t)

        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]

    def predict_with_latent(self, board: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, torch.Tensor]:
        board_t = torch.as_tensor(board, dtype=torch.float32, device=self.device)
        valid_t = torch.as_tensor(valid, dtype=torch.float32, device=self.device)
        board_t = board_t.view(1, self.board_x, self.board_y, self.board_z)

        self.nnet.eval()
        with torch.no_grad():
            latent, log_pi, v = self.nnet.initial_inference_with_latent(board_t, valid_t)

        return torch.exp(log_pi).data.cpu().numpy()[0], float(v.item()), latent

    def recurrent_predict(self, latent: torch.Tensor, action: int) -> tuple[np.ndarray, float, float, torch.Tensor]:
        action_oh = F.one_hot(torch.tensor([action], device=self.device), self.action_size).float()
        self.nnet.eval()
        with torch.no_grad():
            next_latent, reward, log_pi, v = self.nnet.recurrent_inference(latent, action_oh)
        return (
            torch.exp(log_pi).data.cpu().numpy()[0],
            float(v.item()),
            float(reward.item()),
            next_latent,
        )

    def loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(targets * outputs) / targets.size(0)

    def loss_v(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size(0)

    def save_checkpoint(self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar") -> None:
        filepath = os.path.join(folder, filename)
        os.makedirs(folder, exist_ok=True)
        torch.save(
            {
                "state_dict": self.nnet.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar") -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model in path {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.nnet.load_state_dict(checkpoint["state_dict"], strict=False)
        if "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except (ValueError, KeyError):
                log.warning("Could not restore optimizer state, starting fresh.")

    def print(self, game: ChessGame) -> None:
        print(self.nnet)


def _soft_ce_with_support(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft categorical support targets."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1)
