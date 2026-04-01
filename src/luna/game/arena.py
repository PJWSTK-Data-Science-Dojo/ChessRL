"""Arena where 2 players fight against each other."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from tqdm import tqdm

from .luna_game import ChessGame

if TYPE_CHECKING:
    import chess

log = logging.getLogger(__name__)


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
        self.display = display

    def play_game(self, verbose: bool = False) -> float:
        """Execute one episode. Returns +1 if player1 wins, -1 if player2 wins, else draw value."""
        players = [self.player2, None, self.player1]
        current_player = 1
        board = self.game.get_init_board()
        turn_count = 0
        while self.game.get_game_ended(board, current_player) == 0:
            turn_count += 1
            if verbose:
                assert self.display
                print("Turn ", str(turn_count), "Player ", str(current_player))
                self.display(board)
            canonical_board = self.game.get_canonical_form(board, current_player)
            action = players[current_player + 1](canonical_board)

            valids = self.game.get_valid_moves(canonical_board, 1)

            if valids[action] == 0:
                log.error("Action %d is not valid!", action)
                log.debug("valids = %s", valids)
                assert valids[action] > 0
            board, current_player = self.game.get_next_state(board, current_player, action)
        if verbose:
            assert self.display
            print("Game over: Turn ", str(turn_count), "Result ", str(self.game.get_game_ended(board, 1)))
            self.display(board)
        return current_player * self.game.get_game_ended(board, current_player)

    def play_games(self, num: int, verbose: bool = False) -> tuple[int, int, int]:
        """Play num games, swapping colors halfway. Returns (p1_wins, p2_wins, draws)."""
        num = int(num / 2)
        player_one_wins = 0
        player_two_wins = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.play_games (1)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == 1:
                player_one_wins += 1
            elif game_result == -1:
                player_two_wins += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.play_games (2)"):
            game_result = self.play_game(verbose=verbose)
            if game_result == -1:
                player_one_wins += 1
            elif game_result == 1:
                player_two_wins += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1
        return player_one_wins, player_two_wins, draws

