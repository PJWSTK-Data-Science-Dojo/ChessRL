from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest
import zstandard

from luna.game.chess_game import SIDE_TO_MOVE_PLANE, ChessGame, mirror_move, move_to_action
from luna.pgn_dataset import PgnDatasetConfig, load_pgn_dataset


def _game_text(
    moves: str,
    *,
    result: str = "1-0",
    white_elo: str = "2400",
    black_elo: str = "2350",
    extra_headers: str = "",
) -> str:
    headers = (
        f'[Event "Synthetic"]\n[Site "test"]\n[White "Expert A"]\n[Black "Expert B"]\n'
        f'[WhiteElo "{white_elo}"]\n[BlackElo "{black_elo}"]\n[Result "{result}"]\n{extra_headers}'
    )
    return f"{headers}\n{moves} {result}\n\n"


def _write_pgn(tmp_path: Path, *records: str, compressed: bool = False) -> Path:
    text = "".join(records)
    if compressed:
        path = tmp_path / "games.pgn.zst"
        path.write_bytes(zstandard.ZstdCompressor().compress(text.encode("utf-8")))
        return path
    path = tmp_path / "games.pgn"
    path.write_text(text, encoding="utf-8")
    return path


def _config(
    *,
    max_positions: int = 100,
    validation_fraction: float = 0.0,
    split_seed: int = 0,
    min_game_plies: int = 1,
    max_game_plies: int = 512,
) -> PgnDatasetConfig:
    return PgnDatasetConfig(
        min_player_elo=2000,
        max_positions=max_positions,
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        min_game_plies=min_game_plies,
        max_game_plies=max_game_plies,
    )


def test_loads_canonical_expert_targets_for_both_colours(tmp_path: Path, chess_game: ChessGame) -> None:
    path = _write_pgn(tmp_path, _game_text("1. e4 e5"))

    dataset = load_pgn_dataset(path, _config(), chess_game)

    trajectory = dataset.train_trajectories[0]
    white_action = move_to_action(chess.Move.from_uci("e2e4"))
    black_action = move_to_action(mirror_move(chess.Move.from_uci("e7e5")))
    np.testing.assert_array_equal(trajectory.actions, [white_action, black_action])
    np.testing.assert_array_equal(trajectory.root_values, [1.0, -1.0])
    np.testing.assert_array_equal(trajectory.rewards, [0.0, -1.0])
    assert np.all(trajectory.observations[:, :, :, SIDE_TO_MOVE_PLANE] == 1.0)
    assert np.all(trajectory.root_policies.sum(axis=1) == 1.0)
    assert np.all(trajectory.root_policies[np.arange(2), trajectory.actions] == 1.0)
    assert np.all(trajectory.valids[np.arange(2), trajectory.actions])
    assert dataset.stats.train_positions == 2
    assert dataset.stats.result_fallback_positions == 2


def test_uses_node_evaluations_as_side_to_move_wdl_values(tmp_path: Path, chess_game: ChessGame) -> None:
    moves = "{ [%eval 0.25] } 1. e4 { [%eval 0.50] } e5"
    path = _write_pgn(tmp_path, _game_text(moves))

    dataset = load_pgn_dataset(path, _config(), chess_game)

    values = dataset.train_trajectories[0].root_values
    white_expectation = chess.engine.Cp(25).wdl(model="sf16", ply=0).expectation()
    black_expectation = chess.engine.Cp(-50).wdl(model="sf16", ply=1).expectation()
    assert values[0] == pytest.approx(2.0 * white_expectation - 1.0, abs=1e-6)
    assert values[1] == pytest.approx(2.0 * black_expectation - 1.0, abs=1e-6)
    assert dataset.stats.engine_evaluated_positions == 2
    assert dataset.stats.result_fallback_positions == 0


def test_preserves_underpromotion_action_encoding(tmp_path: Path, chess_game: ChessGame) -> None:
    moves = "1. a4 h5 2. a5 h4 3. a6 h3 4. axb7 hxg2 5. bxa8=R"
    path = _write_pgn(tmp_path, _game_text(moves))

    dataset = load_pgn_dataset(path, _config(), chess_game)

    trajectory = dataset.train_trajectories[0]
    promotion = move_to_action(chess.Move.from_uci("b7a8r"))
    assert trajectory.actions[-1] == promotion
    assert trajectory.root_policies[-1, promotion] == 1.0
    assert trajectory.valids[-1, promotion]


def test_records_exact_terminal_outcome_and_reward(tmp_path: Path, chess_game: ChessGame) -> None:
    path = _write_pgn(tmp_path, _game_text("1. f3 e5 2. g4 Qh4#", result="0-1"))

    trajectory = load_pgn_dataset(path, _config(), chess_game).train_trajectories[0]

    assert trajectory.termination is chess.Termination.CHECKMATE
    assert trajectory.rewards[-1] == 1.0
    np.testing.assert_array_equal(trajectory.root_values, [-1.0, 1.0, -1.0, 1.0])


def test_does_not_treat_an_unclaimed_threefold_as_terminal(tmp_path: Path, chess_game: ChessGame) -> None:
    moves = "1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8"
    path = _write_pgn(tmp_path, _game_text(moves))

    dataset = load_pgn_dataset(path, _config(min_game_plies=8), chess_game)

    assert dataset.stats.games_loaded == 1
    assert dataset.train_trajectories[0].termination is None


def test_filters_non_expert_and_low_quality_games(tmp_path: Path, chess_game: ChessGame) -> None:
    records = (
        _game_text("1. e4 e5", extra_headers='[Termination "Normal"]\n'),
        _game_text("1. d4 d5", white_elo="1900"),
        _game_text("1. c4 c5", result="*"),
        _game_text("1. Nf3 Nf6", extra_headers='[Variant "Atomic"]\n'),
        _game_text("1. g3 g6", extra_headers='[WhiteTitle "BOT"]\n'),
        _game_text("1. b3", extra_headers='[Termination "Normal"]\n'),
        _game_text("1. f4 e5", extra_headers='[Termination "Time forfeit"]\n'),
        _game_text("1. e4 e5 2. Nf3", extra_headers='[Termination "Normal"]\n'),
        _game_text("1. e4 e5", extra_headers=f'[SetUp "1"]\n[FEN "{chess.STARTING_FEN}"]\n'),
        _game_text("1. f3 e5 2. g4 Qh4#"),
    )
    path = _write_pgn(tmp_path, *records)

    dataset = load_pgn_dataset(path, _config(min_game_plies=2, max_game_plies=2), chess_game)

    assert dataset.stats.games_scanned == 10
    assert dataset.stats.games_loaded == 1
    assert dataset.stats.games_filtered == 9


def test_capacity_stops_only_before_a_complete_game(tmp_path: Path, chess_game: ChessGame) -> None:
    path = _write_pgn(
        tmp_path,
        _game_text("1. e4 e5"),
        _game_text("1. d4 d5"),
        _game_text("1. c4 c5"),
    )

    dataset = load_pgn_dataset(path, _config(max_positions=5), chess_game)

    assert tuple(trajectory.game_length for trajectory in dataset.train_trajectories) == (2, 2)
    assert dataset.stats.train_positions == 4
    assert dataset.stats.capacity_skipped_games == 1
    assert dataset.stats.limit_reached is True


def test_deduplicates_identical_games(tmp_path: Path, chess_game: ChessGame) -> None:
    first = _game_text("1. e4 e5")
    second = first.replace('[Site "test"]', '[Site "duplicate"]')
    path = _write_pgn(tmp_path, first, second)

    dataset = load_pgn_dataset(path, _config(), chess_game)

    assert dataset.stats.games_scanned == 2
    assert dataset.stats.games_loaded == 1
    assert dataset.stats.duplicate_games == 1


def test_game_level_split_is_deterministic_and_disjoint(tmp_path: Path, chess_game: ChessGame) -> None:
    records = tuple(
        _game_text(moves)
        for moves in (
            "1. e4 e5",
            "1. d4 d5",
            "1. c4 c5",
            "1. Nf3 Nf6",
            "1. g3 g6",
            "1. b3 b6",
            "1. f4 f5",
            "1. Nc3 Nc6",
        )
    )
    path = _write_pgn(tmp_path, *records)
    config = _config(validation_fraction=0.5, split_seed=17)

    first = load_pgn_dataset(path, config, chess_game)
    second = load_pgn_dataset(path, config, chess_game)

    first_train = {trajectory.actions.tobytes() for trajectory in first.train_trajectories}
    first_validation = {trajectory.actions.tobytes() for trajectory in first.validation_trajectories}
    assert first_train
    assert first_validation
    assert first_train.isdisjoint(first_validation)
    assert first_train == {trajectory.actions.tobytes() for trajectory in second.train_trajectories}
    assert first_validation == {trajectory.actions.tobytes() for trajectory in second.validation_trajectories}


def test_reads_zstandard_compressed_pgn(tmp_path: Path, chess_game: ChessGame) -> None:
    path = _write_pgn(tmp_path, _game_text("1. e4 e5"), compressed=True)

    dataset = load_pgn_dataset(path, _config(), chess_game)

    assert dataset.stats.games_loaded == 1
    assert dataset.train_trajectories[0].game_length == 2


def test_rejects_invalid_dataset_configuration() -> None:
    with pytest.raises(ValueError, match="min_player_elo"):
        PgnDatasetConfig(min_player_elo=-1)
    with pytest.raises(ValueError, match="max_positions"):
        PgnDatasetConfig(max_positions=0)
    with pytest.raises(ValueError, match="validation_fraction"):
        PgnDatasetConfig(validation_fraction=1.1)
    with pytest.raises(ValueError, match="min_game_plies"):
        PgnDatasetConfig(min_game_plies=0)
    with pytest.raises(ValueError, match="max_game_plies"):
        PgnDatasetConfig(min_game_plies=10, max_game_plies=9)


def test_rejects_unsupported_file_extension(tmp_path: Path, chess_game: ChessGame) -> None:
    path = tmp_path / "games.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.pgn"):
        load_pgn_dataset(path, _config(), chess_game)
