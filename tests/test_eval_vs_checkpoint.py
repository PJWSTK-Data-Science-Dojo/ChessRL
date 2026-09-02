"""Tests for the checkpoint-arena CLI boundary."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eval_vs_checkpoint import EvalVsCheckpointCli, _protocol, _result_path, run
from luna.game.checkpoint_arena import (
    CheckpointArenaResult,
    CheckpointArenaScores,
    CheckpointIdentity,
)


def _result(config: EvalVsCheckpointCli, scores: CheckpointArenaScores) -> CheckpointArenaResult:
    return CheckpointArenaResult(
        checkpoint_a=CheckpointIdentity(str(config.checkpoint_a), "a" * 64),
        checkpoint_b=CheckpointIdentity(str(config.checkpoint_b), "b" * 64),
        protocol=_protocol(config),
        scores=scores,
    )


def test_cli_defaults_pin_comparable_mcts_protocol(tmp_path: Path) -> None:
    config = EvalVsCheckpointCli(tmp_path / "a.pth.tar", tmp_path / "b.pth.tar", "latent")

    protocol = _protocol(config)

    assert protocol.games == 20
    assert protocol.max_ply == 256
    assert protocol.mcts.search_mode == "gumbel"
    assert protocol.mcts.tree_state_mode == "latent"
    assert protocol.mcts.num_mcts_sims == 32
    assert protocol.mcts.gumbel_max_considered_actions == 8
    assert protocol.mcts.dir_noise is False


def test_cli_can_pin_exact_state_search(tmp_path: Path) -> None:
    config = EvalVsCheckpointCli(
        tmp_path / "a.pth.tar",
        tmp_path / "b.pth.tar",
        "exact",
    )

    assert _protocol(config).mcts.tree_state_mode == "exact"


def test_cli_writes_result_without_loading_models(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"
    config = EvalVsCheckpointCli(
        tmp_path / "a.pth.tar",
        tmp_path / "b.pth.tar",
        "latent",
        output=output,
        games=4,
    )
    result = _result(config, CheckpointArenaScores(1, 2, 1))

    with patch("eval_vs_checkpoint.evaluate_checkpoints", return_value=result):
        exit_code = run(config)

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["checkpoint_b_passed"] is True
    assert json.loads(capsys.readouterr().out) == payload


def test_cli_returns_distinct_exit_code_when_candidate_regresses(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    config = EvalVsCheckpointCli(
        tmp_path / "a.pth.tar",
        tmp_path / "b.pth.tar",
        "latent",
        output=output,
        games=4,
    )
    result = _result(config, CheckpointArenaScores(3, 0, 1))

    with patch("eval_vs_checkpoint.evaluate_checkpoints", return_value=result):
        exit_code = run(config)

    assert exit_code == 3
    assert output.is_file()


def test_cli_refuses_to_overwrite_an_input_checkpoint(tmp_path: Path) -> None:
    checkpoint_a = tmp_path / "a.pth.tar"
    config = EvalVsCheckpointCli(
        checkpoint_a,
        tmp_path / "b.pth.tar",
        "latent",
        output=checkpoint_a,
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        _result_path(config)
