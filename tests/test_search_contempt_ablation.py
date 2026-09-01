"""Tests for the fixed-weight Search-contempt ablation."""

from pathlib import Path

import chess
import pytest

from ablate_search_contempt import _arms, _counterbalanced_order
from luna.search_contempt_ablation import SearchContemptAblationCli, trajectory_metrics, validate_cli
from tests.conftest import TrajectoryFactory


def test_arm_grid_keeps_current_and_low_noise_controls() -> None:
    cli = SearchContemptAblationCli(Path("model.pt"), Path("report.json"), node_limits=(2, 4))

    arms = _arms(cli)

    assert [(arm.name, arm.visit_limit, arm.temperature_ply) for arm in arms] == [
        ("current-control", None, 257),
        ("low-noise-control", None, 40),
        ("search-contempt-2", 2, 40),
        ("search-contempt-4", 4, 40),
    ]


def test_trajectory_metrics_report_diversity_outcomes_and_activation(
    make_trajectory: TrajectoryFactory,
) -> None:
    white_win = make_trajectory(16, termination=chess.Termination.CHECKMATE)
    white_win.rewards[-1] = -1.0
    white_win.search_contempt_opponent_selections = 8
    white_win.search_contempt_thompson_selections = 4
    white_win.search_contempt_frozen_nodes = 2
    draw = make_trajectory(16, termination=chess.Termination.STALEMATE)
    unique = make_trajectory(16, truncated=True)
    unique.actions[0] = 1

    metrics = trajectory_metrics([white_win, draw, unique], elapsed=2.0)

    assert metrics.games == 3
    assert metrics.positions == 48
    assert metrics.positions_per_second == 24.0
    assert metrics.white_wins == 1
    assert metrics.draws == 1
    assert metrics.truncated == 1
    assert metrics.thompson_fraction == 0.5
    assert metrics.frozen_nodes == 2
    assert metrics.repeated_prefix_8_fraction == pytest.approx(2 / 3)
    assert metrics.repeated_prefix_16_fraction == pytest.approx(2 / 3)
    assert metrics.terminations == {"checkmate": 1, "stalemate": 1}


@pytest.mark.parametrize("node_limits", [(), (0,), (-1,)])
def test_invalid_node_limit_grids_are_rejected(node_limits: tuple[int, ...]) -> None:
    cli = SearchContemptAblationCli(Path("model.pt"), Path("report.json"), node_limits=node_limits)

    with pytest.raises(ValueError, match="node_limits"):
        validate_cli(cli)


def test_arm_order_is_rotated_between_seeds() -> None:
    cli = SearchContemptAblationCli(Path("model.pt"), Path("report.json"), node_limits=(2, 4, 8))
    arms = _arms(cli)

    orders = [[arm.name for arm in _counterbalanced_order(arms, index)] for index in range(3)]

    assert [order[0] for order in orders] == ["current-control", "low-noise-control", "search-contempt-2"]


def test_existing_report_is_rejected_before_the_ablation(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    output.write_text("{}", encoding="utf-8")
    cli = SearchContemptAblationCli(tmp_path / "model.pt", output)

    with pytest.raises(FileExistsError, match="already exists"):
        validate_cli(cli)
