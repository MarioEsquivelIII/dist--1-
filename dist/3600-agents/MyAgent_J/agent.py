"""
Tournament agent for CS3600 Spring 2026.

Strategy:
  - HMM (via RatBelief) tracks the rat using noise and distance observations.
  - Negamax with alpha-beta pruning (iterative deepening) chooses board moves.
  - Each turn we decide: search for rat (if expected value is high enough) OR
    make the best board move found by the search.
  - Heuristic: score differential + carpet potential + weighted primed runway.
"""

from collections.abc import Callable
from typing import Tuple
import time

import numpy as np

from game import board as board_mod, move as move_mod
from game.enums import (
    Direction, MoveType, Cell,
    BOARD_SIZE, CARPET_POINTS_TABLE,
    loc_after_direction,
)
from game.move import Move

from .rat_belief import RatBelief


# ---------------------------------------------------------------------------
# Heuristic weights (tunable)
# ---------------------------------------------------------------------------
W_SCORE      = 1.0   # score difference
W_MY_CARPET  = 0.45  # my best available carpet value
W_OPP_CARPET = 0.25  # opponent's best carpet value (penalty)
W_RUNWAY     = 0.08  # my total primed squares in reach

# Search threshold: only search when EV > this value
# EV = 6*P - 2;  EV > THRESHOLD  ↔  P > (THRESHOLD + 2) / 6
SEARCH_EV_THRESHOLD = 1.2

# Max moves to consider at each node (top-K after ordering)
MAX_MOVES_PER_NODE = 18

# Keep at least this many seconds in reserve before timing out
TIME_BUFFER = 8.0


class PlayerAgent:
    """
    Required entry points: __init__, commentate, play.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief = RatBelief(transition_matrix)
        self.turn_number = 0          # counts our own turns (0-indexed)
        self._prev_opp_search = (None, False)

    # ------------------------------------------------------------------
    def commentate(self):
        best_cell, best_prob = self.rat_belief.get_best_cell()
        return f"Turn {self.turn_number} | rat belief peak {best_prob:.3f} @ {best_cell}"

    # ------------------------------------------------------------------
    def play(self, board, sensor_data: Tuple, time_left: Callable):
        noise, observed_dist = sensor_data
        worker_pos = board.player_worker.get_location()

        # ---- 1. Update rat HMM (order matters!) -----------------------
        # Rat moves once per game-turn. We play every other game-turn.
        # The rat resets (1000-step spawn) whenever EITHER player catches it.
        #
        # Case A: We caught the rat on our LAST turn.
        #   → new rat: 1000 spawn moves, then 1 move before opp turn, 1 before ours
        #   → reset belief, predict() × 2
        #
        # Case B: Opponent caught the rat on THEIR last turn.
        #   → new rat: 1000 spawn moves, then 1 move before our turn
        #   → reset belief, predict() × 1
        #
        # Case C: No catch (normal).
        #   → rat moved once before opp turn + once before our turn  (2 moves)
        #   → but on turn_number == 0 the rat only moved once before us
        #   → predict() × (1 if first turn else 2)

        my_loc, my_found = board.player_search
        opp_loc, opp_found = board.opponent_search
        opp_search_new = (opp_loc is not None and
                          opp_loc != self._prev_opp_search[0])

        if my_found and my_loc is not None:
            # We caught rat last turn; reset, then 2 more rat moves.
            self.rat_belief.reset_to_spawn()
            n_predict = 2
        elif opp_search_new and opp_found:
            # Opponent caught rat on their last turn; reset, then 1 rat move.
            self.rat_belief.reset_to_spawn()
            n_predict = 1
        else:
            n_predict = 1 if self.turn_number == 0 else 2

        for _ in range(n_predict):
            self.rat_belief.predict()

        self.rat_belief.update(noise, observed_dist, worker_pos, board)

        # Incorporate opponent's negative search (cell ruled out).
        if opp_search_new and not opp_found:
            self.rat_belief.update_from_search(opp_loc, False)

        self._prev_opp_search = (opp_loc, opp_found)
        self.turn_number += 1

        # ---- 2. Rat search decision -----------------------------------
        best_rat_cell, best_rat_prob = self.rat_belief.get_best_cell()
        search_ev = 6.0 * best_rat_prob - 2.0   # E[points] from searching best cell

        turns_left = board.player_worker.turns_left
        time_remaining = time_left()

        # Compare search EV against expected board move value.
        valid_moves = board.get_valid_moves(exclude_search=True)

        if search_ev > SEARCH_EV_THRESHOLD:
            best_board_imm = self._best_immediate_value(valid_moves)
            if search_ev >= best_board_imm or best_rat_prob > 0.72:
                return Move.search(best_rat_cell)

        if not valid_moves:
            # Completely blocked; must search somewhere.
            return Move.search(best_rat_cell)

        # ---- 3. Board move via iterative-deepening negamax ------------
        safe_time = max(0.0, time_remaining - TIME_BUFFER)
        time_per_turn = safe_time / max(turns_left, 1)
        search_time = min(time_per_turn * 0.80, 4.5)

        return self._choose_move(board, valid_moves, search_time)

    # ------------------------------------------------------------------
    # Move selection
    # ------------------------------------------------------------------

    def _choose_move(self, board, valid_moves, time_budget: float):
        start = time.time()

        # Greedy baseline (depth-0).
        best_move = self._greedy_pick(valid_moves)

        for depth in range(1, 7):
            elapsed = time.time() - start
            if elapsed >= time_budget * 0.55:
                break
            move, _ = self._negamax(
                board, depth,
                float('-inf'), float('inf'),
                start, time_budget * 0.92
            )
            if move is not None:
                best_move = move
            if time.time() - start >= time_budget:
                break

        return best_move

    def _negamax(self, board, depth, alpha, beta, start_time, deadline):
        """
        Negamax with alpha-beta pruning.
        Returns (best_move, value_for_current_player).
        After reverse_perspective(), player_worker is the side to move.
        """
        if time.time() - start_time > deadline:
            return None, self._heuristic(board)

        if depth == 0 or board.is_game_over():
            return None, self._heuristic(board)

        valid_moves = board.get_valid_moves(exclude_search=True)
        if not valid_moves:
            return None, self._heuristic(board)

        # Order and limit moves for efficiency.
        ordered = self._order_moves(valid_moves, board)[:MAX_MOVES_PER_NODE]

        best_move = ordered[0]
        best_val  = float('-inf')

        for m in ordered:
            if time.time() - start_time > deadline:
                break

            child = board.forecast_move(m, check_ok=False)
            if child is None:
                continue
            child.reverse_perspective()

            _, val = self._negamax(child, depth - 1, -beta, -alpha,
                                   start_time, deadline)
            val = -val   # negate: child returns value for *that* player

            if val > best_val:
                best_val = val
                best_move = m
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break   # alpha-beta cutoff

        return best_move, best_val

    # ------------------------------------------------------------------
    # Heuristic
    # ------------------------------------------------------------------

    def _heuristic(self, board):
        """
        Evaluate from the current player's perspective.
        After an even number of reverse_perspective() calls, player_worker
        is our actual worker; after an odd number it is the opponent's.
        Negamax handles the sign flip automatically.
        """
        my_pts  = board.player_worker.get_points()
        opp_pts = board.opponent_worker.get_points()
        score_diff = float(my_pts - opp_pts)

        my_pos  = board.player_worker.get_location()
        opp_pos = board.opponent_worker.get_location()

        my_carpet,  my_reach  = self._carpet_stats(board, my_pos)
        opp_carpet, opp_reach = self._carpet_stats(board, opp_pos)

        return (W_SCORE     * score_diff
                + W_MY_CARPET  * my_carpet
                - W_OPP_CARPET * opp_carpet
                + W_RUNWAY     * (my_reach - opp_reach))

    def _carpet_stats(self, board, pos):
        """
        From `pos`, scan 4 directions for contiguous primed squares.
        Returns (best_carpet_value, total_primed_squares_reachable).
        """
        pm = board._primed_mask
        pw = board.player_worker.get_location()
        ow = board.opponent_worker.get_location()

        best_val = 0
        total    = 0

        for direction in Direction:
            count = 0
            cur = pos
            while count < BOARD_SIZE - 1:
                nxt = loc_after_direction(cur, direction)
                if not board.is_valid_cell(nxt):
                    break
                bit = 1 << (nxt[1] * BOARD_SIZE + nxt[0])
                if not (pm & bit):
                    break
                if nxt == pw or nxt == ow:
                    break
                count += 1
                cur = nxt

            total += count
            if 1 <= count <= 7:
                v = CARPET_POINTS_TABLE[count]
                if v > best_val:
                    best_val = v

        return float(best_val), float(total)

    # ------------------------------------------------------------------
    # Move ordering & helpers
    # ------------------------------------------------------------------

    def _order_moves(self, moves, board):
        """Sort moves: high-value carpet first, then prime, then plain."""
        def key(m):
            if m.move_type == MoveType.CARPET:
                # Higher carpet length → lower sort key (goes first).
                return -float(CARPET_POINTS_TABLE.get(m.roll_length, 0))
            elif m.move_type == MoveType.PRIME:
                # Prime toward the longest run of existing primed squares.
                pos = board.player_worker.get_location()
                nxt = loc_after_direction(pos, m.direction)
                run = self._primed_run_after(board, nxt, m.direction)
                return -(0.5 + 0.1 * run)
            else:
                # Plain: prefer moving toward primed clusters (rough heuristic).
                return 0.0
        return sorted(moves, key=key)

    def _primed_run_after(self, board, start, direction):
        """Count consecutive primed squares starting at `start` in `direction`."""
        count = 0
        cur = start
        pm = board._primed_mask
        while count < BOARD_SIZE - 1:
            bit = 1 << (cur[1] * BOARD_SIZE + cur[0])
            if not (pm & bit):
                break
            count += 1
            nxt = loc_after_direction(cur, direction)
            if not board.is_valid_cell(nxt):
                break
            cur = nxt
        return count

    def _greedy_pick(self, valid_moves):
        """Return the highest-immediate-value move (fast, no tree search)."""
        best_move = valid_moves[0]
        best_val  = float('-inf')
        for m in valid_moves:
            v = self._immediate_value(m)
            if v > best_val:
                best_val = v
                best_move = m
        return best_move

    def _immediate_value(self, move) -> float:
        if move.move_type == MoveType.CARPET:
            return float(CARPET_POINTS_TABLE.get(move.roll_length, 0))
        elif move.move_type == MoveType.PRIME:
            return 1.0
        return 0.0

    def _best_immediate_value(self, valid_moves) -> float:
        """Return the best immediate point gain available from valid_moves."""
        best = 0.0
        for m in valid_moves:
            v = self._immediate_value(m)
            if v > best:
                best = v
        return best
