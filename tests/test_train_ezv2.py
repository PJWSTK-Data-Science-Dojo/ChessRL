"""Regression tests for EfficientZeroV2 training loop."""

from collections.abc import Sequence
from pathlib import Path

import chess
import numpy as np
import pytest
import torch
from torch.amp import GradScaler

from luna.config import EzV2LearnerConfig, MCTSParams, TrainingRunConfig
from luna.game.chess_game import ACTION_SIZE, OBS_PLANES, ChessGame
from luna.network import LunaNetwork, _scale_gradient
from luna.replay_buffer import PrioritizedReplayBuffer, Trajectory


def _make_trajectory(length: int = 4) -> Trajectory:
    return Trajectory(
        observations=[np.random.randn(8, 8, OBS_PLANES).astype(np.float32) for _ in range(length)],
        actions=[np.random.randint(0, min(256, ACTION_SIZE)) for _ in range(length)],
        rewards=np.zeros(length, dtype=np.float32),
        root_policies=[np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32) for _ in range(length)],
        root_values=np.zeros(length, dtype=np.float32),
        valids=np.ones((length, ACTION_SIZE), dtype=np.float32),
    )


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_scale_gradient_preserves_forward_and_scales_backward(scale: float) -> None:
    source = torch.tensor([1.5, -2.0], requires_grad=True)
    upstream = torch.tensor([3.0, -4.0])

    scaled = _scale_gradient(source, scale)
    torch.testing.assert_close(scaled, source)
    (scaled * upstream).sum().backward()

    assert source.grad is not None
    torch.testing.assert_close(source.grad, upstream * scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_ezv2_increments_global_step_per_optimizer_step() -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        batch_size=2,
        grad_accum_steps=2,
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        lr=3e-3,
    )
    nnet = LunaNetwork(game, learner)
    replay = PrioritizedReplayBuffer(capacity=500)

    for _ in range(8):
        replay.save_trajectory(_make_trajectory(length=12))

    g0 = int(nnet._global_step)
    run_params = TrainingRunConfig(num_mcts_sims=2)
    nnet.train_ezv2(
        replay,
        steps=4,
        discount=0.997,
        mcts_for_reanalyze=run_params,
    )
    assert nnet._global_step == g0 + 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_reanalyze_disables_async_prefetch_paths() -> None:
    game = ChessGame()
    learner = EzV2LearnerConfig(
        reanalyze_mcts_sims=2,
        reanalyze_prob=1.0,
        mixed_value_td_until_step=0,
        batch_size=2,
        num_channels=16,
        repr_blocks=1,
        dyn_blocks=1,
        proj_dim=32,
        lr=1e-3,
    )
    nnet = LunaNetwork(game, learner)
    assert not nnet._async_batch_prefetch()


def test_reanalysis_warmup_keeps_plain_replay_prefetch_enabled(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.reanalyze_mcts_sims = 2
    small_learner_config.reanalyze_prob = 1.0
    small_learner_config.mixed_value_td_until_step = 10
    network = LunaNetwork(ChessGame(), small_learner_config)

    assert network._async_batch_prefetch(upcoming_steps=9)
    assert not network._async_batch_prefetch(upcoming_steps=10)


def test_zero_replay_workers_disable_background_prefetch(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.dataloader_workers = 0
    network = LunaNetwork(ChessGame(), small_learner_config)

    assert network._prefetch_executor is None
    assert not network._async_batch_prefetch()


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("grad_accum_steps", 0, "grad_accum_steps must be positive"),
        ("dataloader_workers", -1, "dataloader_workers cannot be negative"),
        ("support_size", 0, "support_size must be an integer of at least 1"),
        ("grad_clip_norm", 0.0, "grad_clip_norm must be positive and finite"),
        ("grad_clip_norm", float("inf"), "grad_clip_norm must be positive and finite"),
    ],
)
def test_invalid_execution_settings_fail_loudly(
    small_learner_config: EzV2LearnerConfig,
    field_name: str,
    value: object,
    message: str,
) -> None:
    setattr(small_learner_config, field_name, value)

    with pytest.raises(ValueError, match=message):
        LunaNetwork(ChessGame(), small_learner_config)


def test_training_rejects_zero_unroll_horizon(
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.unroll_steps = 0
    small_learner_config.dataloader_workers = 0
    network = LunaNetwork(ChessGame(), small_learner_config)
    replay = PrioritizedReplayBuffer(capacity=2)
    replay.save_trajectory(_make_trajectory(length=1))

    with pytest.raises(ValueError, match="unroll_steps must be positive"):
        network.train_ezv2(replay, steps=1)


def test_reported_total_loss_is_invariant_to_identical_gradient_accumulation() -> None:
    np.random.seed(23)
    trajectory = _make_trajectory(length=1)

    def train_once(grad_accum_steps: int) -> float:
        torch.manual_seed(7)
        learner = EzV2LearnerConfig(
            device="cpu",
            batch_size=1,
            unroll_steps=1,
            td_steps=1,
            num_channels=16,
            repr_blocks=1,
            dyn_blocks=1,
            proj_dim=32,
            lr=0.0,
            lr_min=0.0,
            mixed_precision=False,
            grad_accum_steps=grad_accum_steps,
            dataloader_workers=0,
        )
        network = LunaNetwork(ChessGame(), learner)
        replay = PrioritizedReplayBuffer(capacity=2)
        replay.save_trajectory(trajectory)
        return network.train_ezv2(replay, steps=1)["total"]

    single_microbatch = train_once(1)
    accumulated_microbatches = train_once(2)

    assert accumulated_microbatches == pytest.approx(single_microbatch, rel=1e-5)


def test_reanalysis_restores_training_mode_and_uses_direct_sve(
    monkeypatch: pytest.MonkeyPatch,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    game = ChessGame()
    small_learner_config.batch_size = 1
    small_learner_config.reanalyze_mcts_sims = 2
    small_learner_config.reanalyze_prob = 1.0
    small_learner_config.reanalyze_policy = True
    small_learner_config.mixed_value_td_until_step = 0
    nnet = LunaNetwork(game, small_learner_config)
    replay = PrioritizedReplayBuffer(capacity=4)
    replay.save_trajectory(_make_trajectory(length=1))

    fresh_value = 0.375

    class _FakeBatchedMCTS:
        def __init__(self, _game: ChessGame, network: LunaNetwork, _params: MCTSParams) -> None:
            self.network = network

        def search_batch(
            self,
            boards: list[chess.Board],
            temp: float,
            *,
            add_exploration_noise: bool | Sequence[bool] | None,
        ) -> list[tuple[np.ndarray, float, None, None]]:
            assert temp == 1.0
            assert add_exploration_noise is False
            self.network.nnet.eval()
            policy = np.full(ACTION_SIZE, 1.0 / ACTION_SIZE, dtype=np.float32)
            return [(policy, fresh_value, None, None) for _ in boards]

    monkeypatch.setattr("luna.network.BatchedMCTS", _FakeBatchedMCTS)
    nnet.nnet.train()

    collated, _weights, _indices = nnet._prepare_batch(
        replay,
        bs=1,
        unroll=0,
        td=5,
        discount=1.0,
        training_step=0,
        mcts_for_reanalyze=TrainingRunConfig(num_mcts_sims=2),
    )

    assert nnet.nnet.training
    assert collated["target_values"][0, 0] == pytest.approx(fresh_value)


def test_checkpoint_contains_architecture_metadata(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.recurrent_gradient_scale = 0.4
    nnet = LunaNetwork(chess_game, small_learner_config)
    nnet._global_step = 17
    nnet._trainer_iteration = 6
    nnet.scaler = GradScaler("cpu", init_scale=1024.0, growth_interval=123, enabled=True)
    expected_scaler_state = nnet.scaler.state_dict()
    nnet.save_checkpoint(str(tmp_path), "metadata.pth.tar")

    checkpoint = torch.load(
        tmp_path / "metadata.pth.tar",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["format_version"] == 2
    assert checkpoint["global_step"] == 17
    assert checkpoint["trainer_iteration"] == 6
    assert checkpoint["scaler"] == expected_scaler_state
    assert checkpoint["model_spec"]["action_size"] == chess_game.get_action_size()
    assert checkpoint["model_spec"]["observation_shape"] == list(chess_game.get_board_size())
    assert checkpoint["learner_config"]["num_channels"] == small_learner_config.num_channels
    assert checkpoint["learner_config"]["recurrent_gradient_scale"] == pytest.approx(0.4)

    restored = LunaNetwork.from_checkpoint(
        chess_game,
        tmp_path / "metadata.pth.tar",
        device="cpu",
    )
    assert restored._global_step == 17
    assert restored._trainer_iteration == 6
    assert restored._learner.num_channels == small_learner_config.num_channels
    assert restored._learner.recurrent_gradient_scale == pytest.approx(0.4)

    resumed = LunaNetwork(chess_game, small_learner_config)
    resumed.scaler = GradScaler("cpu", init_scale=32.0, growth_interval=7, enabled=True)
    resumed.load_checkpoint(str(tmp_path), "metadata.pth.tar", load_optimizer=True)
    assert resumed.scaler.state_dict() == expected_scaler_state


def test_learning_rate_continues_from_checkpoint_global_step(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    small_learner_config.batch_size = 2
    small_learner_config.unroll_steps = 1
    small_learner_config.td_steps = 1
    small_learner_config.mixed_precision = False
    small_learner_config.lr = 1e-3
    small_learner_config.lr_min = 1e-5
    small_learner_config.lr_warmup_steps = 4

    nnet = LunaNetwork(chess_game, small_learner_config)
    nnet._global_step = 7
    nnet.save_checkpoint(str(tmp_path), "resume.pth.tar")

    restored = LunaNetwork(chess_game, small_learner_config)
    restored.load_checkpoint(str(tmp_path), "resume.pth.tar", load_optimizer=False)
    replay = PrioritizedReplayBuffer(capacity=8)
    replay.save_trajectory(_make_trajectory(length=4))
    expected_lr = restored._lr_schedule(step_in_run=8, total_steps=20)

    restored.train_ezv2(replay, steps=1, total_train_steps=20)

    assert restored._global_step == 8
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_checkpoint_loader_rejects_legacy_and_mismatched_model_specs(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    legacy_path = tmp_path / "legacy.pth.tar"
    torch.save({"state_dict": network.nnet.state_dict()}, legacy_path)

    with pytest.raises(ValueError, match="only format version 2"):
        network.load_checkpoint(str(tmp_path), legacy_path.name)

    valid_path = tmp_path / "valid.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    mismatched = torch.load(valid_path, map_location="cpu", weights_only=True)
    mismatched["model_spec"]["action_size"] += 1
    mismatch_path = tmp_path / "mismatch.pth.tar"
    torch.save(mismatched, mismatch_path)

    with pytest.raises(ValueError, match="model specification"):
        network.load_checkpoint(str(tmp_path), mismatch_path.name)


@pytest.mark.parametrize("missing_field", ["optimizer", "scaler", "global_step", "trainer_iteration"])
def test_checkpoint_loader_rejects_incomplete_v2_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    missing_field: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "complete.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    del checkpoint[missing_field]
    incomplete_path = tmp_path / f"missing-{missing_field}.pth.tar"
    torch.save(checkpoint, incomplete_path)

    with pytest.raises(ValueError, match="missing required fields"):
        network.load_checkpoint(str(tmp_path), incomplete_path.name, load_optimizer=False)


def test_checkpoint_loader_rejects_incompatible_training_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    valid_path = tmp_path / "valid-training-state.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint["optimizer"] = {"invalid": True}
    corrupt_path = tmp_path / "invalid-training-state.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(RuntimeError, match="training state is incompatible"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=True)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_loader_rejects_mismatched_resume_semantics(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-learner-config.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint["learner_config"]["unroll_steps"] += 1
    mismatch_path = tmp_path / "mismatched-learner-config.pth.tar"
    torch.save(checkpoint, mismatch_path)

    with pytest.raises(ValueError, match=r"differs in fields.*unroll_steps"):
        network.load_checkpoint(str(tmp_path), mismatch_path.name, load_optimizer=False)


def test_checkpoint_loader_rejects_corrupt_learner_metadata(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-metadata.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint["learner_config"] = "corrupt"
    corrupt_path = tmp_path / "corrupt-metadata.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="string-keyed mapping"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)


@pytest.mark.parametrize("field_name", ["optimizer", "scaler"])
def test_checkpoint_loader_rejects_invalid_training_state_containers_for_inference(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
    field_name: str,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-containers.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint[field_name] = None
    corrupt_path = tmp_path / f"invalid-{field_name}.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="optimizer and scaler states must be mappings"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)


def test_checkpoint_counter_validation_precedes_model_mutation(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-counter.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    original = {name: tensor.detach().clone() for name, tensor in network.nnet.state_dict().items()}
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    first_name = next(iter(checkpoint["state_dict"]))
    checkpoint["state_dict"][first_name] = checkpoint["state_dict"][first_name] + 1
    checkpoint["global_step"] = -1
    corrupt_path = tmp_path / "negative-counter.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="non-negative integer"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)

    for name, tensor in network.nnet.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_checkpoint_loader_rejects_non_tensor_model_state(
    tmp_path: Path,
    chess_game: ChessGame,
    small_learner_config: EzV2LearnerConfig,
) -> None:
    network = LunaNetwork(chess_game, small_learner_config)
    valid_path = tmp_path / "valid-model-state.pth.tar"
    network.save_checkpoint(str(tmp_path), valid_path.name)
    checkpoint = torch.load(valid_path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"] = {"invalid": None}
    corrupt_path = tmp_path / "non-tensor-state.pth.tar"
    torch.save(checkpoint, corrupt_path)

    with pytest.raises(ValueError, match="state_dict must map string names to tensors"):
        network.load_checkpoint(str(tmp_path), corrupt_path.name, load_optimizer=False)
