"""Arena where 2 players fight against each other."""

from collections.abc import Callable
from operator import index
from typing import Any

import chess
from loguru import logger
from tqdm import tqdm

from luna.game.chess_game import ChessGame, player_from_turn

_WIN_THRESHOLD = 0.5


class Arena:
    """Play balanced matches between two action-producing chess agents."""

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

    def play_game(
        self,
        verbose: bool = False,
        max_ply: int | None = None,
        initial_board: chess.Board | None = None,
    ) -> float:
        """Execute one episode. Returns +1 if player1 wins, -1 if player2 wins, or 0 for a draw.

        If ``max_ply`` is set and reached without a terminal outcome, returns ``0.0`` (draw).
        """
        players = {1: self.player1, -1: self.player2}
        board = self.game.get_init_board() if initial_board is None else initial_board.copy(stack=True)
        current_player = player_from_turn(board.turn)
        turn_count = 0
        while self.game.get_game_outcome(board, current_player) is None:
            if max_ply is not None and turn_count >= max_ply:
                return 0.0
            turn_count += 1
            if verbose:
                if self.display is None:
                    raise ValueError("display callback required for verbose mode")
                logger.info("Turn {} Player {}", turn_count, current_player)
                self.display(board)
            canonical_board = self.game.get_canonical_form(board, current_player)
            raw_action = players[current_player](canonical_board)
            if isinstance(raw_action, bool):
                raise ValueError(f"Player returned a non-integer action: {raw_action!r}")
            try:
                action = index(raw_action)
            except TypeError as exc:
                raise ValueError(f"Player returned a non-integer action: {raw_action!r}") from exc

            valids = self.game.get_valid_moves(canonical_board, 1)

            if not 0 <= action < len(valids) or valids[action] == 0:
                logger.error("Action {} is not valid!", action)
                logger.debug("valids = {}", valids)
                raise ValueError(f"Action {action} is not valid")
            current_player = self.game.push_action(board, current_player, action)
        if verbose:
            if self.display is None:
                raise ValueError("display callback required for verbose mode")
            logger.info(
                "Game over: Turn {} Result {}",
                turn_count,
                self.game.get_game_outcome(board, 1),
            )
            self.display(board)
        outcome = self.game.get_game_outcome(board, current_player)
        if outcome is None:
            raise RuntimeError("Arena stopped before reaching a terminal position")
        return current_player * outcome

    @staticmethod
    def _classify_result(result: float) -> int:
        if result > _WIN_THRESHOLD:
            return 1
        if result < -_WIN_THRESHOLD:
            return -1
        return 0

    def play_games(self, num: int, verbose: bool = False) -> tuple[int, int, int]:
        """Play both color assignments and return wins for each agent plus draws."""
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
        try:
            for _ in tqdm(range(num - half), desc="Arena.play_games (2)"):
                classification = self._classify_result(self.play_game(verbose=verbose))
                if classification == -1:
                    player_one_wins += 1
                elif classification == 1:
                    player_two_wins += 1
                else:
                    draws += 1
        finally:
            self.player1, self.player2 = self.player2, self.player1
        return player_one_wins, player_two_wins, draws
