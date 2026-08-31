"""Tests for balanced arena execution and player boundaries."""

from collections.abc import Callable

import chess
import pytest

from luna.game.arena import Arena
from luna.game.chess_game import ChessGame


def _first_legal_action(game: ChessGame) -> Callable[[chess.Board], int]:
    def player(board: chess.Board) -> int:
        return int(game.get_valid_moves(board, 1).nonzero()[0][0])

    return player


@pytest.mark.parametrize("action", [4288, -4289, 1.5, True])
def test_play_game_rejects_malformed_player_actions(action: object) -> None:
    game = ChessGame()

    def malformed_player(_board: chess.Board) -> object:
        return action

    with pytest.raises(ValueError, match=r"[Aa]ction"):
        Arena(malformed_player, _first_legal_action(game), game).play_game(max_ply=1)


def test_play_games_restores_player_assignment_after_second_half_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    game = ChessGame()
    player_one = _first_legal_action(game)
    player_two = _first_legal_action(game)
    arena = Arena(player_one, player_two, game)
    calls = 0

    def fail_in_second_half(*, verbose: bool = False, **_kwargs: object) -> float:
        nonlocal calls
        del verbose
        calls += 1
        if calls == 2:
            raise RuntimeError("second half failed")
        return 0.0

    monkeypatch.setattr(arena, "play_game", fail_in_second_half)

    with pytest.raises(RuntimeError, match="second half failed"):
        arena.play_games(2)

    assert arena.player1 is player_one
    assert arena.player2 is player_two
