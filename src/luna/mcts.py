import logging
import math
import numpy as np
import chess
from omegaconf import DictConfig, OmegaConf
from collections import deque

# Luna
from .game.luna_game import ChessGame, who, _flip_action_index
from .NNet import Luna_Network # Import for type hinting

EPS = 1e-8
log = logging.getLogger(__name__)

class MCTS(object):
    """
        This class handles the MCTS tree. Updated for OmegaConf.
    """

    game: ChessGame
    nnet: Luna_Network
    # Store the MCTS config specific to this instance
    cfg: DictConfig

    Qsa: dict
    Nsa: dict
    Ns: dict
    Ps: dict
    Es: dict
    Vs: dict
    state_info: dict

    # Change __init__ signature
    def __init__(self, game: ChessGame, nnet: Luna_Network, mcts_cfg: DictConfig) -> None:
        super(MCTS, self).__init__()

        self.game = game
        self.nnet = nnet
        self.cfg = mcts_cfg # Store the MCTS-specific config
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.state_info = {}

    def getActionProb(self, root_board: chess.Board, root_history: deque, temp=1) -> list:
        """
        Performs numMCTSSims simulations starting from root_board with root_history.
        Returns: policy vector proportional to Nsa[(s,a)]**(1./temp) for the root state (canonical perspective).
        """
        canonical_root_board = self.game.getCanonicalForm(root_board, who(root_board.turn))
        s_root = self.game.stringRepresentation(canonical_root_board)

        self.state_info[s_root] = (root_board.copy(), deque(root_history, maxlen=self.game.history_len))

        # Access numMCTSSims from self.cfg
        for i in range(self.cfg.numMCTSSims):
            self.search(s_root)

        counts = np.array([self.Nsa.get((s_root, a), 0) for a in range(self.game.getActionSize())], dtype=np.float32)

        # --- Apply Dirichlet Noise (ONLY at the root) ---
        # Access dir_noise, dirichlet_alpha, dirichlet_epsilon from self.cfg
        if self.cfg.dir_noise and temp > 0:
             alpha = self.cfg.dirichlet_alpha
             epsilon = self.cfg.dirichlet_epsilon
             num_actions = self.game.getActionSize()

             valids_root_canonical = self.Vs.get(s_root)
             if valids_root_canonical is None:
                  root_board_actual, _ = self.state_info.get(s_root, (None, None))
                  if root_board_actual is not None:
                       canonical_board_for_valids = self.game.getCanonicalForm(root_board_actual, who(root_board_actual.turn))
                       valids_root_canonical = self.game.getValidMoves(canonical_board_for_valids, 1)
                       self.Vs[s_root] = valids_root_canonical
                  else:
                       log.error(f"getActionProb: Could not retrieve actual root board for state key {s_root} to compute valid moves.")
                       # Fallback to assuming all moves are valid if necessary, but this indicates an issue
                       valids_root_canonical = np.ones(num_actions, dtype=np.int32) # Fallback


             counts_valid_only = counts * valids_root_canonical
             sum_counts_valid = np.sum(counts_valid_only)

             if sum_counts_valid > EPS:
                  base_probs_valid = counts_valid_only / sum_counts_valid
                  dirichlet_noise_full = np.random.dirichlet([alpha] * num_actions)
                  dirichlet_noise_valid = dirichlet_noise_full * valids_root_canonical
                  mixed_probs_valid = (1 - epsilon) * base_probs_valid + epsilon * dirichlet_noise_valid
                  # Replace original counts with the new noisy distribution, ensuring invalid moves are zero
                  counts = mixed_probs_valid * valids_root_canonical

                  sum_counts_with_noise = np.sum(counts)
                  if sum_counts_with_noise > EPS:
                      counts /= sum_counts_with_noise
                  else:
                      log.warning(f"getActionProb: Policy sum zero after noise and masking for root {s_root}. Falling back to uniform over valids.")
                      num_valid = np.sum(valids_root_canonical)
                      counts = (valids_root_canonical / num_valid) if num_valid > 0 else np.zeros_like(counts)
             else:
                  log.warning(f"getActionProb: Sum of valid counts is zero for root {s_root}. Cannot apply noise. Falling back to uniform over valids.")
                  num_valid = np.sum(valids_root_canonical)
                  counts = (valids_root_canonical / num_valid) if num_valid > 0 else np.zeros_like(counts)

        # --- End Dirichlet Noise Application ---

        if temp == 0:
            max_count = np.max(counts)
            if max_count == 0:
                 valids = self.Vs.get(s_root)
                 if valids is None:
                     root_board_actual, _ = self.state_info.get(s_root, (None, None))
                     if root_board_actual is not None:
                          canonical_board_for_valids = self.game.getCanonicalForm(root_board_actual, who(root_board_actual.turn))
                          valids = self.game.getValidMoves(canonical_board_for_valids, 1)
                          self.Vs[s_root] = valids
                     else:
                          log.error(f"getActionProb temp=0: Could not retrieve actual root board for state key {s_root} to compute valid moves.")
                          valids = np.ones(self.game.getActionSize(), dtype=np.int32)

                 num_valid = np.sum(valids)
                 probs_array = (valids / num_valid) if num_valid > 0 else np.zeros_like(counts)
            else:
                bestAs = np.array(np.argwhere(counts == max_count)).flatten()
                valids = self.Vs.get(s_root)
                if valids is None:
                     root_board_actual, _ = self.state_info.get(s_root, (None, None))
                     if root_board_actual is not None:
                          canonical_board_for_valids = self.game.getCanonicalForm(root_board_actual, who(root_board_actual.turn))
                          valids = self.game.getValidMoves(canonical_board_for_valids, 1)
                          self.Vs[s_root] = valids
                     else:
                           log.error(f"getActionProb temp=0 (selection): Could not retrieve actual root board for state key {s_root} to compute valid moves.")
                           valids = np.ones(self.game.getActionSize(), dtype=np.int32)

                valid_bestAs = [a for a in bestAs if valids[a] == 1]
                if not valid_bestAs:
                     log.error(f"getActionProb temp=0: Best actions ({bestAs}) are not valid for state {s_root}. Falling back to uniform over all valids.")
                     num_valid = np.sum(valids)
                     probs_array = (valids / num_valid) if num_valid > 0 else np.zeros_like(counts)
                else:
                     bestA = np.random.choice(valid_bestAs)
                     probs_array = np.zeros_like(counts)
                     probs_array[bestA] = 1.0
        else:
            counts = counts ** (1. / temp)
            counts_sum = np.sum(counts)
            if counts_sum < EPS:
                log.warning(f"getActionProb (temp={temp}): Sum of counts is zero for state {s_root}. Returning uniform over valids.")
                valids = self.Vs.get(s_root)
                if valids is None:
                     root_board_actual, _ = self.state_info.get(s_root, (None, None))
                     if root_board_actual is not None:
                          canonical_board_for_valids = self.game.getCanonicalForm(root_board_actual, who(root_board_actual.turn))
                          valids = self.game.getValidMoves(canonical_board_for_valids, 1)
                          self.Vs[s_root] = valids
                     else:
                          log.error(f"getActionProb temp>0: Could not retrieve actual root board for state key {s_root} to compute valid moves.")
                          valids = np.ones(self.game.getActionSize(), dtype=np.int32)

                num_valid = np.sum(valids)
                probs_array = (valids / num_valid) if num_valid > 0 else np.zeros_like(counts)
            else:
                probs_array = counts / counts_sum

        final_sum = np.sum(probs_array)
        if abs(final_sum - 1.0) > 1e-6 and final_sum > EPS:
             probs_array /= final_sum
        elif final_sum <= EPS:
              log.warning(f"getActionProb (temp={temp}): Final policy sum ({final_sum}) is zero. Returning zero policy.")
              probs_array = np.zeros_like(probs_array)


        return probs_array.tolist()

    def search(self, s: str) -> float:
        """Performs one MCTS simulation from the state represented by canonical key 's'."""
        if s not in self.Es:
             state_data = self.state_info.get(s)
             if state_data is None:
                 log.error(f"MCTS Search Error (recursive call): State key '{s}' not found in state_info.")
                 return 0.0

             current_board, current_history = state_data
             current_player = who(current_board.turn)
             self.Es[s] = self.game.getGameEnded(current_board, 1)

        game_ended_result_p1 = self.Es[s]

        state_data = self.state_info.get(s) # Re-get state data for player info
        if state_data is None:
             log.error(f"MCTS Search Error (recursive call): State key '{s}' not found in state_info during terminal check.")
             return 0.0
        current_board, _ = state_data
        current_player = who(current_board.turn)


        if game_ended_result_p1 != 0:
            return game_ended_result_p1 * current_player

        # --- Leaf Node Expansion ---
        if s not in self.Ps:
            state_data = self.state_info.get(s)
            if state_data is None:
                 log.error(f"MCTS Search Error (recursive call): State key '{s}' not found in state_info during expansion.")
                 return 0.0

            current_board, current_history = state_data
            current_player = who(current_board.turn)

            try:
                 canonical_board = self.game.getCanonicalForm(current_board, current_player)
                 canonical_history = self.game.getCanonicalHistory(current_history, current_player)
                 nn_input_array_canonical = self.game.toArray(canonical_history)
                 valids_canonical = self.game.getValidMoves(canonical_board, 1)

                 # Predict using the NN (nnet already has access to its model_cfg for device)
                 policy_pred_canonical, v_pred = self.nnet.predict((nn_input_array_canonical, valids_canonical))

                 self.Ps[s] = policy_pred_canonical
                 self.Vs[s] = valids_canonical
                 self.Ns[s] = 0

                 value = -v_pred

            except Exception as e:
                log.error(f"Error during MCTS leaf expansion state {s} (canonical): {e}", exc_info=True)
                value = 0.0

            return value
        # --- End Leaf Node Expansion ---

        # --- Selection (UCT) ---
        valids_canonical = self.Vs.get(s)
        policy_canonical = self.Ps.get(s)

        if valids_canonical is None or policy_canonical is None:
             log.error(f"MCTS Selection Error: State {s} (canonical) inconsistent (in Ns but not Ps/Vs). Terminating search path.")
             return 0.0

        if self.Es[s] != 0:
             state_data = self.state_info.get(s)
             if state_data is None:
                  log.error(f"MCTS Search Error (recursive call): State key '{s}' not found in state_info during terminal check.")
                  return 0.0
             current_board, _ = state_data
             current_player = who(current_board.turn)
             return self.Es[s] * current_player


        cur_best = -float('inf')
        best_act = -1
        current_N_s = self.Ns.get(s, 0)

        # Access cpuct from self.cfg
        cpuct_value = self.cfg.cpuct

        for a in range(self.game.getActionSize()):
            if valids_canonical[a]:
                 q_val = self.Qsa.get((s, a), 0.0)
                 p_sa = policy_canonical[a]
                 n_sa = self.Nsa.get((s, a), 0)

                 uct_score = q_val + cpuct_value * p_sa * math.sqrt(current_N_s + EPS) / (1 + n_sa)

                 if uct_score > cur_best:
                     cur_best = uct_score
                     best_act = a

        action_canonical = best_act

        if action_canonical == -1:
             log.error(f"MCTS selection failed: No valid action found for canonical state {s}. Valids sum: {np.sum(valids_canonical)}. Terminating search path.")
             return 0.0

        # --- Perform the move to get the next state ---
        state_data = self.state_info.get(s)
        if state_data is None:
             log.error(f"MCTS Search Error (recursive call): State key '{s}' not found in state_info before making move.")
             return 0.0

        current_board_actual, current_history_actual = state_data
        current_player_actual = who(current_board_actual.turn)

        action_actual = action_canonical
        if current_player_actual == -1:
             action_actual = _flip_action_index(action_canonical)

        previous_board_fen = current_board_actual.fen()

        next_board_actual, next_player_actual, next_history_actual = self.game.getNextState(current_board_actual, current_player_actual, action_actual, current_history_actual)

        # --- CHECK FOR ILLEGAL MOVE ---
        if next_player_actual == current_player_actual and next_board_actual.fen() == previous_board_fen:
            log.error(f"MCTS Error: getNextState returned original state after action {action_actual} on actual board {previous_board_fen}. Illegal move selected by MCTS! Ending episode path.")
            return 0.0

        assert next_player_actual == -current_player_actual, f"MCTS Error: Player did not flip after legal move! Current: {current_player_actual}, Next: {next_player_actual}. Board: {current_board_actual.fen()}"

        canonical_next_board = self.game.getCanonicalForm(next_board_actual, next_player_actual)
        next_s = self.game.stringRepresentation(canonical_next_board)

        # Store the state info for the next state if not already present
        if next_s not in self.state_info:
             self.state_info[next_s] = (next_board_actual.copy(), deque(next_history_actual, maxlen=self.game.history_len))
             if next_s not in self.Es:
                  self.Es[next_s] = self.game.getGameEnded(next_board_actual, 1)

        # Recursive call
        value = self.search(next_s)

        # --- Backpropagation ---
        backup_value_p1 = -value

        if (s, action_canonical) in self.Qsa:
            self.Qsa[(s, action_canonical)] = (self.Nsa[(s, action_canonical)] * self.Qsa[(s, action_canonical)] + backup_value_p1) / (self.Nsa[(s, action_canonical)] + 1)
            self.Nsa[(s, action_canonical)] += 1
        else:
            self.Qsa[(s, action_canonical)] = backup_value_p1
            self.Nsa[(s, action_canonical)] = 1

        self.Ns[s] = self.Ns.get(s, 0) + 1

        return -value