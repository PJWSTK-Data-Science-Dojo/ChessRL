"""Arena where 2 players fight against each other."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from tqdm import tqdm

from .chess_game import ChessGame

if TYPE_CHECKING:
    import chess

_WIN_THRESHOLD = 0.5
_DRAW_THRESHOLD = 1e-8


class Arena:
    """An Arena class where any 2 agents can be pit against each other."""

    game: ChessGame
    player1: Callable[..., Any]
    player2: Callable[..., Any]

    def __init__(
        self,
        player1: Callable[..., Any],
        player2: Callable[..., Any],
        game: ChessGame,
        display: Callable[[chess.Board], None] | None = None,
    ) -> None:
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display: Callable[[chess.Board], None] | None = display

    def play_game(self, verbose: bool = False, max_ply: int | None = None) -> float:
        """Execute one episode. Returns +1 if player1 wins, -1 if player2 wins, else small draw value.

        If ``max_ply`` is set and reached without a terminal outcome, returns ``0.0`` (draw).
        """
        players = {1: self.player1, -1: self.player2}
        current_player = 1
        board = self.game.get_init_board()
        turn_count = 0
        while abs(self.game.get_game_ended(board, current_player)) < _DRAW_THRESHOLD:
            if max_ply is not None and turn_count >= max_ply:
                return 0.0
            turn_count += 1
            if verbose:
                if self.display is None:
                    raise ValueError("display callback required for verbose mode")
                logger.info("Turn {} Player {}", turn_count, current_player)
                self.display(board)
            canonical_board = self.game.get_canonical_form(board, current_player)
            action = players[current_player](canonical_board)

            valids = self.game.get_valid_moves(canonical_board, 1)

            if valids[action] == 0:
                logger.error("Action {} is not valid!", action)
                logger.debug("valids = {}", valids)
                raise ValueError(f"Action {action} is not valid")
            board, current_player = self.game.get_next_state(board, current_player, action)
        if verbose:
            if self.display is None:
                raise ValueError("display callback required for verbose mode")
            logger.info(
                "Game over: Turn {} Result {}",
                turn_count,
                self.game.get_game_ended(board, 1),
            )
            self.display(board)
        return current_player * self.game.get_game_ended(board, current_player)

    @staticmethod
    def _classify_result(result: float) -> int:
        """Classify game result: +1 for p1 win, -1 for p2 win, 0 for draw."""
        if result > _WIN_THRESHOLD:
            return 1
        if result < -_WIN_THRESHOLD:
            return -1
        return 0

    def play_games(self, num: int, verbose: bool = False) -> tuple[int, int, int]:
        """Play num games, swapping colors halfway. Returns (p1_wins, p2_wins, draws)."""
        half = num // 2
        player_one_wins = 0
        player_two_wins = 0
        draws = 0
        for _ in tqdm(range(half), desc="Arena.play_games (1)"):
            classification = self._classify_result(self.play_game(verbose=verbose))
            if classification == 1:
                player_one_wins += 1
            elif classification == -1:
                player_two_wins += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num - half), desc="Arena.play_games (2)"):
            classification = self._classify_result(self.play_game(verbose=verbose))
            if classification == -1:
                player_one_wins += 1
            elif classification == 1:
                player_two_wins += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1
        return player_one_wins, player_two_wins, draws
