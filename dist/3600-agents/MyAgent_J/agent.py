import random
import numpy as np
import time as time_module
from collections.abc import Callable

from game import board as board_module
from game import enums
from game.enums import Direction, MoveType, Cell, BOARD_SIZE, CARPET_POINTS_TABLE
from game.move import Move


DIRECTIONS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

DIR_DELTAS = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

# For fast iteration without dict lookup
DIR_LIST = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT

# Noise emission probabilities: Cell type -> noise index -> probability
NOISE_TABLE = np.array([
    [0.7,  0.15, 0.15],  # SPACE=0
    [0.1,  0.8,  0.1 ],  # PRIMED=1
    [0.1,  0.1,  0.8 ],  # CARPET=2
    [0.5,  0.3,  0.2 ],  # BLOCKED=3
], dtype=np.float64)

# Carpet points lookup (index = roll length)
CARPET_PTS = [0, -1, 2, 4, 6, 10, 15, 21]


class PlayerAgent:
    def __init__(self, board_state, transition_matrix=None, time_left: Callable = None):
        self._init_ok = False
        try:
            self._real_init(board_state, transition_matrix)
            self._init_ok = True
        except Exception:
            pass

    def _real_init(self, board_state, transition_matrix):
        # --- Transition Matrix Setup ---
        if transition_matrix is not None:
            try:
                self.T = np.array(transition_matrix, dtype=np.float64)
            except Exception:
                self.T = np.array(
                    [[float(transition_matrix[i][j]) for j in range(64)] for i in range(64)],
                    dtype=np.float64,
                )
        else:
            self.T = np.eye(64, dtype=np.float64)

        # Compute stationary distribution
        T_1000 = np.linalg.matrix_power(self.T, 1000)
        self.stationary = T_1000[0].copy()
        s = self.stationary.sum()
        self.stationary = self.stationary / s if s > 0 else np.ones(64) / 64

        self.belief = self.stationary.copy()
        self.first_turn = True
        self.turn_number = 0

        # --- Precompute Manhattan Distances ---
        coords = np.array([(i % 8, i // 8) for i in range(64)])
        self.manhattan = np.abs(
            coords[:, None, :] - coords[None, :, :]
        ).sum(axis=2).astype(np.int32)

        # --- Distance Observation Likelihood Table ---
        max_d = 14
        self.dist_like = np.zeros((max_d + 3, max_d + 1), dtype=np.float64)
        for actual in range(max_d + 1):
            for offset, prob in zip([-1, 0, 1, 2], [0.12, 0.70, 0.12, 0.06]):
                reported = max(0, actual + offset)
                if reported < self.dist_like.shape[0]:
                    self.dist_like[reported, actual] += prob

        # Precompute blocked mask
        self._blocked = board_state._blocked_mask

        # --- Precompute static board potential ---
        self.static_potential = [[0]*8 for _ in range(8)]
        for y in range(8):
            for x in range(8):
                if self._blocked & (1 << (y * 8 + x)):
                    continue
                h_len = 1
                lx = x - 1
                while lx >= 0 and not (self._blocked & (1 << (y * 8 + lx))):
                    h_len += 1
                    lx -= 1
                rx = x + 1
                while rx < 8 and not (self._blocked & (1 << (y * 8 + rx))):
                    h_len += 1
                    rx += 1
                v_len = 1
                uy = y - 1
                while uy >= 0 and not (self._blocked & (1 << (uy * 8 + x))):
                    v_len += 1
                    uy -= 1
                dy_ = y + 1
                while dy_ < 8 and not (self._blocked & (1 << (dy_ * 8 + x))):
                    v_len += 1
                    dy_ += 1
                self.static_potential[y][x] = min(max(h_len, v_len), 7)

        # Transposition table
        self.tt = {}

    def commentate(self):
        return ""

    # =========================================================================
    # HMM Rat Tracker
    # =========================================================================

    def _apply_transition_with_opp_search(self, opp_search):
        self.belief = self.belief @ self.T
        if opp_search[0] is not None and not opp_search[1]:
            idx = opp_search[0][1] * 8 + opp_search[0][0]
            self.belief[idx] = 0.0
            s = self.belief.sum()
            if s > 0:
                self.belief /= s

    def _update_hmm(self, board_state, sensor_data):
        if board_state.opponent_search[1]:
            self.belief = self.stationary.copy()
            self.belief = self.belief @ self.T
        elif board_state.player_search[1]:
            self.belief = self.stationary.copy()
            self._apply_transition_with_opp_search(board_state.opponent_search)
            self.belief = self.belief @ self.T
        elif self.first_turn and board_state.turn_count == 0:
            self.first_turn = False
            self.belief = self.belief @ self.T
        else:
            self.first_turn = False
            self._apply_transition_with_opp_search(board_state.opponent_search)
            self.belief = self.belief @ self.T

        noise_idx = int(sensor_data[0])
        est_d = int(sensor_data[1])

        blocked = board_state._blocked_mask
        primed = board_state._primed_mask
        carpet = board_state._carpet_mask
        cell_types = np.zeros(64, dtype=np.int32)
        for j in range(64):
            bit = 1 << j
            if blocked & bit:
                cell_types[j] = 3
            elif primed & bit:
                cell_types[j] = 1
            elif carpet & bit:
                cell_types[j] = 2

        self.belief *= NOISE_TABLE[cell_types, noise_idx]

        w_pos = board_state.player_worker.get_location()
        w_idx = w_pos[1] * 8 + w_pos[0]
        actual_dists = self.manhattan[w_idx]

        if est_d < self.dist_like.shape[0]:
            safe_dists = np.minimum(actual_dists, self.dist_like.shape[1] - 1)
            dist_lik = self.dist_like[est_d, safe_dists]
            dist_lik[actual_dists >= self.dist_like.shape[1]] = 0
            self.belief *= dist_lik

        self.belief = self.belief * 0.995 + 0.005 / 64
        total = self.belief.sum()
        self.belief = self.belief / total if total > 1e-300 else np.ones(64) / 64

    # =========================================================================
    # Board Scanning Helpers
    # =========================================================================

    def _count_primes_dir(self, primed, x, y, dx, dy, opp_x, opp_y, my_x, my_y):
        count = 0
        cx, cy = x + dx, y + dy
        while 0 <= cx < 8 and 0 <= cy < 8:
            if (cx == opp_x and cy == opp_y) or (cx == my_x and cy == my_y):
                break
            if not (primed & (1 << (cy * 8 + cx))):
                break
            count += 1
            cx += dx
            cy += dy
        return count

    # =========================================================================
    # Precompute per-turn data
    # =========================================================================

    def _precompute_position_scores(self, bs):
        carpet = bs._carpet_mask
        primed = bs._primed_mask
        blocked = bs._blocked_mask
        self._pos_quality = [[0.0]*8 for _ in range(8)]
        for y in range(8):
            for x in range(8):
                pot = self.static_potential[y][x]
                if pot < 3:
                    continue
                bit = 1 << (y * 8 + x)
                if carpet & bit:
                    continue
                self._pos_quality[y][x] = CARPET_PTS[pot] if pot <= 7 else 21

        # Precompute prime segment proximity advantage
        my_x, my_y = bs.player_worker.get_location()
        opp_x, opp_y = bs.opponent_worker.get_location()
        self._prime_proximity_adv = 0.0

        for axis in range(2):  # 0=horizontal, 1=vertical
            for outer in range(8):
                inner = 0
                while inner < 8:
                    y, x = (outer, inner) if axis == 0 else (inner, outer)
                    if primed & (1 << (y * 8 + x)):
                        start = inner
                        while inner < 8:
                            y2, x2 = (outer, inner) if axis == 0 else (inner, outer)
                            if not (primed & (1 << (y2 * 8 + x2))):
                                break
                            inner += 1
                        seg_len = inner - start
                        if seg_len >= 2:
                            pts = CARPET_PTS[min(seg_len, 7)]
                            if pts > 0:
                                my_d = opp_d = 99
                                for end_inner in (start - 1, inner):
                                    ey, ex = (outer, end_inner) if axis == 0 else (end_inner, outer)
                                    if 0 <= ex < 8 and 0 <= ey < 8 and not (blocked & (1 << (ey * 8 + ex))):
                                        my_d = min(my_d, abs(ex - my_x) + abs(ey - my_y))
                                        opp_d = min(opp_d, abs(ex - opp_x) + abs(ey - opp_y))
                                if my_d < 99:
                                    self._prime_proximity_adv += pts * (1.0/(opp_d+1) - 1.0/(my_d+1)) * 0.15
                    else:
                        inner += 1

    # =========================================================================
    # Heuristic Evaluation
    # =========================================================================

    def _heuristic(self, bs, is_my_turn):
        if is_my_turn:
            my_w, opp_w = bs.player_worker, bs.opponent_worker
        else:
            my_w, opp_w = bs.opponent_worker, bs.player_worker

        my_x, my_y = my_w.get_location()
        opp_x, opp_y = opp_w.get_location()
        primed = bs._primed_mask
        blocked = bs._blocked_mask
        carpet = bs._carpet_mask
        occupied = primed | carpet | blocked

        my_pts = my_w.get_points()
        opp_pts = opp_w.get_points()
        turns_left = my_w.turns_left

        # ---- 1. Score margin ----
        score_diff = float(my_pts - opp_pts)
        if turns_left <= 3:
            val = score_diff * 2.5
        elif turns_left <= 8:
            val = score_diff * 1.8
        elif turns_left <= 20:
            val = score_diff * 1.3
        else:
            val = score_diff * 1.0

        # ---- 2. Immediate carpet potential ----
        my_best_carpet = 0
        my_second_carpet = 0
        opp_best_carpet = 0

        for dx, dy in DIR_LIST:
            length = self._count_primes_dir(primed, my_x, my_y, dx, dy, opp_x, opp_y, my_x, my_y)
            if length >= 1:
                pts = CARPET_PTS[min(length, 7)]
                if pts > my_best_carpet:
                    my_second_carpet = my_best_carpet
                    my_best_carpet = pts
                elif pts > my_second_carpet:
                    my_second_carpet = pts

            length = self._count_primes_dir(primed, opp_x, opp_y, dx, dy, opp_x, opp_y, my_x, my_y)
            if length >= 1:
                pts = CARPET_PTS[min(length, 7)]
                if pts > opp_best_carpet:
                    opp_best_carpet = pts

        # Value carpet potential highly - near-guaranteed points next move
        val += my_best_carpet * 0.90
        val += my_second_carpet * 0.15
        val -= opp_best_carpet * 0.85

        # ---- 3. Building potential (phase-aware) ----
        if turns_left <= 2:
            build_factor = 0.0
        elif turns_left <= 5:
            build_factor = 0.15
        elif turns_left <= 10:
            build_factor = 0.30
        else:
            build_factor = 0.40

        my_bit = 1 << (my_y * 8 + my_x)
        on_space = not bool(occupied & my_bit)

        best_build = 0.0
        if on_space and build_factor > 0:
            for i in range(4):
                dx, dy = DIR_LIST[i]
                odx, ody = DIR_LIST[i ^ 1]
                primes_behind = self._count_primes_dir(primed, my_x, my_y, odx, ody, opp_x, opp_y, my_x, my_y)

                spaces_ahead = 0
                cx, cy = my_x + dx, my_y + dy
                while 0 <= cx < 8 and 0 <= cy < 8:
                    if cx == opp_x and cy == opp_y:
                        break
                    if occupied & (1 << (cy * 8 + cx)):
                        break
                    spaces_ahead += 1
                    cx += dx
                    cy += dy

                future_line = primes_behind + 1 + spaces_ahead
                if future_line >= 2:
                    if primes_behind >= 1:
                        build_val = CARPET_PTS[min(primes_behind + 1, 7)] * build_factor
                    else:
                        build_val = CARPET_PTS[min(future_line, 7)] * build_factor * 0.30
                    if build_val > best_build:
                        best_build = build_val
        val += best_build

        # ---- 4. Opponent building potential (threat) ----
        opp_bit = 1 << (opp_y * 8 + opp_x)
        if not bool(occupied & opp_bit):
            for i in range(4):
                odx, ody = DIR_LIST[i ^ 1]
                pb = self._count_primes_dir(primed, opp_x, opp_y, odx, ody, opp_x, opp_y, my_x, my_y)
                if pb >= 1:
                    val -= CARPET_PTS[min(pb + 1, 7)] * 0.35

        # ---- 5. Position quality ----
        val += self._pos_quality[my_y][my_x] * 0.03
        val -= self._pos_quality[opp_y][opp_x] * 0.02
        for dx, dy in DIR_LIST:
            nx, ny = my_x + dx, my_y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                val += self._pos_quality[ny][nx] * 0.01

        # ---- 6. Mobility ----
        mob = 0
        for dx, dy in DIR_LIST:
            nx, ny = my_x + dx, my_y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                nbit = 1 << (ny * 8 + nx)
                if not ((blocked | primed) & nbit):
                    if not (nx == opp_x and ny == opp_y):
                        mob += 1
        val += mob * 0.12
        if mob == 0:
            val -= 4.0  # Trapped is catastrophic

        # ---- 7. Prime stealing opportunity ----
        if my_best_carpet >= 4:
            val += 0.5

        # ---- 8. Prime segment proximity (precomputed) ----
        val += self._prime_proximity_adv

        return val

    # =========================================================================
    # Move Ordering
    # =========================================================================

    def _score_move_fast(self, m, bs, my_x, my_y):
        if m.move_type == MoveType.CARPET:
            return 1000 + CARPET_PTS[min(m.roll_length, 7)] * 10
        elif m.move_type == MoveType.PRIME:
            dx, dy = DIR_DELTAS[m.direction]
            nx, ny = my_x + dx, my_y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                return 100 + self.static_potential[ny][nx]
            return 100
        else:  # PLAIN
            dx, dy = DIR_DELTAS[m.direction]
            nx, ny = my_x + dx, my_y + dy
            if 0 <= nx < 8 and 0 <= ny < 8:
                return self.static_potential[ny][nx]
            return 0

    def _order_moves(self, moves, bs):
        my_x, my_y = bs.player_worker.get_location()
        return sorted(moves, key=lambda m: self._score_move_fast(m, bs, my_x, my_y), reverse=True)

    # =========================================================================
    # Minimax with Alpha-Beta + Iterative Deepening
    # =========================================================================

    def _board_hash(self, bs):
        return (
            bs._primed_mask,
            bs._carpet_mask,
            bs.player_worker.position,
            bs.opponent_worker.position,
            bs.player_worker.get_points(),
            bs.opponent_worker.get_points(),
        )

    def _minimax(self, bs, depth, is_my_turn, alpha, beta, deadline):
        if time_module.perf_counter() > deadline:
            return self._heuristic(bs, is_my_turn), None

        if depth == 0 or bs.is_game_over():
            return self._heuristic(bs, is_my_turn), None

        bh = self._board_hash(bs)
        tt_key = (bh, depth, is_my_turn)
        tt_entry = self.tt.get(tt_key)
        if tt_entry is not None:
            return tt_entry

        moves = bs.get_valid_moves()
        if not moves:
            return self._heuristic(bs, is_my_turn), None

        moves = self._order_moves(moves, bs)
        best_move = moves[0]

        if is_my_turn:
            best_val = float('-inf')
            for m in moves:
                if time_module.perf_counter() > deadline:
                    break
                child = bs.forecast_move(m, False)
                if child is None:
                    continue
                child.reverse_perspective()
                v, _ = self._minimax(child, depth - 1, False, alpha, beta, deadline)
                if v > best_val:
                    best_val = v
                    best_move = m
                if v > alpha:
                    alpha = v
                if beta <= alpha:
                    break
        else:
            best_val = float('inf')
            for m in moves:
                if time_module.perf_counter() > deadline:
                    break
                child = bs.forecast_move(m, False)
                if child is None:
                    continue
                child.reverse_perspective()
                v, _ = self._minimax(child, depth - 1, True, alpha, beta, deadline)
                if v < best_val:
                    best_val = v
                    best_move = m
                if v < beta:
                    beta = v
                if beta <= alpha:
                    break

        result = (best_val, best_move)
        self.tt[tt_key] = result
        return result

    def _iterative_deepening(self, board_state, moves, deadline, max_depth):
        best_move = moves[0]
        best_val = float('-inf')

        for depth in range(1, max_depth + 1):
            if time_module.perf_counter() > deadline:
                break

            current_best_move = moves[0]
            current_best_val = float('-inf')
            alpha = float('-inf')

            for m in moves:
                if time_module.perf_counter() > deadline:
                    break
                child = board_state.forecast_move(m, False)
                if child is None:
                    continue
                child.reverse_perspective()
                v, _ = self._minimax(
                    child, depth - 1, False, alpha, float('inf'), deadline
                )
                if v > current_best_val:
                    current_best_val = v
                    current_best_move = m
                if v > alpha:
                    alpha = v

            if current_best_val > float('-inf'):
                best_val = current_best_val
                best_move = current_best_move
                moves = [best_move] + [m for m in moves if not self._moves_equal(m, best_move)]

        return best_val, best_move

    def _moves_equal(self, a, b):
        if a.move_type != b.move_type:
            return False
        if a.move_type == MoveType.SEARCH:
            return a.search_loc == b.search_loc
        if a.direction != b.direction:
            return False
        if a.move_type == MoveType.CARPET:
            return a.roll_length == b.roll_length
        return True

    # =========================================================================
    # Rat Search Expected Value
    # =========================================================================

    def _evaluate_rat_search(self, board_state, best_movement_val, deadline, depth):
        max_prob = float(self.belief.max())
        best_rat_idx = int(np.argmax(self.belief))
        rat_x, rat_y = best_rat_idx % 8, best_rat_idx // 8

        score_diff = float(board_state.player_worker.get_points() - board_state.opponent_worker.get_points())
        turns_left = board_state.player_worker.turns_left

        # Dynamic threshold based on game state
        if score_diff < -5:
            threshold = 0.12  # Behind: take more risks
        elif score_diff < 0:
            threshold = 0.16
        elif score_diff > 5 and turns_left <= 5:
            threshold = 0.28  # Ahead late: be conservative
        else:
            threshold = 0.20

        if max_prob < threshold:
            return None

        # Don't risk search if safely ahead with few turns left
        if turns_left <= 2 and score_diff > 3:
            return None

        p = max_prob
        raw_ev = p * 4.0 - (1.0 - p) * 2.0

        if raw_ev < -0.5:
            return None

        if time_module.perf_counter() > deadline:
            score_diff = float(board_state.player_worker.get_points() - board_state.opponent_worker.get_points())
            incremental = best_movement_val - score_diff
            if raw_ev > incremental:
                return Move.search((rat_x, rat_y))
            return None

        search_depth = min(depth, 2)

        success = board_state.get_copy()
        success.player_worker.increment_points(4)
        success.end_turn(0)
        success.reverse_perspective()
        sv, _ = self._minimax(
            success, search_depth, False, float('-inf'), float('inf'), deadline
        )

        fail = board_state.get_copy()
        fail.player_worker.decrement_points(2)
        fail.end_turn(0)
        fail.reverse_perspective()
        fv, _ = self._minimax(
            fail, search_depth, False, float('-inf'), float('inf'), deadline
        )

        search_val = p * sv + (1 - p) * fv
        if search_val > best_movement_val:
            return Move.search((rat_x, rat_y))

        # Check top 3 locations
        top3 = np.argsort(self.belief)[-3:][::-1]
        for idx in top3:
            prob = float(self.belief[idx])
            if prob < 0.15:
                break
            ev = prob * sv + (1 - prob) * fv
            if ev > best_movement_val:
                rx, ry = idx % 8, idx // 8
                return Move.search((rx, ry))

        return None

    # =========================================================================
    # Main Play
    # =========================================================================

    def play(self, board_state, sensor_data, time_left):
        if not self._init_ok:
            moves = board_state.get_valid_moves()
            return random.choice(moves) if moves else Move.search((0, 0))
        try:
            return self._real_play(board_state, sensor_data, time_left)
        except Exception:
            moves = board_state.get_valid_moves()
            return random.choice(moves) if moves else Move.search((0, 0))

    def _real_play(self, board_state, sensor_data, time_left):
        self.turn_number += 1
        self._update_hmm(board_state, sensor_data)
        self._precompute_position_scores(board_state)

        # --- Time management ---
        # We have 240 seconds total for 40 turns = 6s/turn average
        # Previous version only used ~31s total - way too conservative
        remaining = time_left()
        turns_remaining = max(board_state.player_worker.turns_left, 1)
        per_turn = remaining / turns_remaining

        # Time allocation: more in mid-game, conservative caps to avoid timeout
        if turns_remaining <= 3:
            budget = min(per_turn * 0.80, remaining * 0.25, 4.0)
        elif turns_remaining <= 10:
            budget = min(per_turn * 0.85, remaining * 0.12, 5.0)
        elif turns_remaining <= 25:
            # Mid-game: invest more time for deeper search
            budget = min(per_turn * 0.85, remaining * 0.08, 5.5)
        else:
            # Early game: save time
            budget = min(per_turn * 0.70, remaining * 0.05, 4.0)

        budget = max(budget, 0.05)
        if remaining < 10.0:
            budget = min(budget, remaining * 0.15)
        elif remaining < 30.0:
            budget = min(budget, 1.5)

        if budget > 4.0:
            max_depth = 8
        elif budget > 2.0:
            max_depth = 7
        elif budget > 1.0:
            max_depth = 6
        elif budget > 0.4:
            max_depth = 5
        elif budget > 0.15:
            max_depth = 4
        elif budget > 0.05:
            max_depth = 3
        else:
            max_depth = 2

        deadline = time_module.perf_counter() + budget

        # --- Get moves ---
        moves = board_state.get_valid_moves()
        if not moves:
            best_rat_idx = int(np.argmax(self.belief))
            return Move.search((best_rat_idx % 8, best_rat_idx // 8))

        moves = self._order_moves(moves, board_state)

        # --- Iterative deepening search ---
        self.tt.clear()
        best_val, best_move = self._iterative_deepening(
            board_state, moves, deadline, max_depth
        )

        # --- Rat search evaluation ---
        search_move = self._evaluate_rat_search(
            board_state, best_val, deadline, max_depth
        )
        if search_move is not None:
            return search_move

        return best_move
