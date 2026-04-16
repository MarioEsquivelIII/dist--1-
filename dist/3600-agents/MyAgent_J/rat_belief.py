"""
HMM-based rat belief tracker.

State space: 64 cells on an 8x8 board (index = y*8 + x).
Transition model: given T[i,j] = P(rat moves from cell i to cell j).
Observation model:
  - Noise type depends on cell type under rat.
  - Observed Manhattan distance has known error distribution.
"""

import numpy as np
from game.enums import Cell, BOARD_SIZE

# P(noise | cell_type): [squeak, scratch, squeal]
NOISE_PROBS = {
    Cell.BLOCKED: [0.5,  0.3,  0.2],
    Cell.SPACE:   [0.7,  0.15, 0.15],
    Cell.PRIMED:  [0.1,  0.8,  0.1],
    Cell.CARPET:  [0.1,  0.1,  0.8],
}

# P(observed_distance = actual + offset)
DISTANCE_ERROR_OFFSETS = (-1, 0, 1, 2)
DISTANCE_ERROR_PROBS   = (0.12, 0.70, 0.12, 0.06)

N = BOARD_SIZE * BOARD_SIZE  # 64


class RatBelief:
    """Maintains a probability distribution over rat positions using HMM filtering."""

    def __init__(self, T):
        """
        Parameters:
            T: 64x64 transition matrix (JAX or numpy array).
               T[i,j] = P(rat moves from cell i to cell j).
        """
        self.T = np.array(T, dtype=np.float64)

        # Compute approximate stationary distribution:
        # rat starts at (0,0) = index 0 and walks 1000 steps before game starts.
        belief = np.zeros(N, dtype=np.float64)
        belief[0] = 1.0
        for _ in range(300):          # 300 steps approaches stationary well
            belief = belief @ self.T
        s = belief.sum()
        self.belief = belief / s if s > 0 else np.ones(N) / N

        # Cache cell-type → noise probability arrays (shape: [N, 3])
        # Built lazily once board is first seen, since board changes.
        self._cached_cell_types = None   # tuple of Cell values, len=N
        self._noise_mat = None           # shape (N, 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self):
        """Propagate belief through the transition model (rat moves one step)."""
        self.belief = self.belief @ self.T
        s = self.belief.sum()
        if s > 1e-12:
            self.belief /= s

    def update(self, noise, observed_dist: int, worker_pos, board):
        """
        Update belief given observations for this turn.

        Parameters:
            noise: Noise enum (0=squeak,1=scratch,2=squeal).
            observed_dist: Observed Manhattan distance (may be clamped/noisy).
            worker_pos: (x, y) of our worker.
            board: Current Board object (used for cell types).
        """
        noise_idx = int(noise)

        # Build noise likelihood per cell (N,)
        cell_types = self._get_cell_types(board)
        noise_like = np.empty(N, dtype=np.float64)
        for i, ct in enumerate(cell_types):
            noise_like[i] = NOISE_PROBS[ct][noise_idx]

        # Build distance likelihood per cell (N,)
        wx, wy = worker_pos
        dist_like = np.empty(N, dtype=np.float64)
        for i in range(N):
            rx = i % BOARD_SIZE
            ry = i // BOARD_SIZE
            actual = abs(wx - rx) + abs(wy - ry)
            dist_like[i] = self._dist_likelihood(observed_dist, actual)

        self.belief *= noise_like * dist_like
        s = self.belief.sum()
        if s > 1e-12:
            self.belief /= s
        else:
            # Degenerate: reset to uniform
            self.belief = np.ones(N, dtype=np.float64) / N

    def reset_to_spawn(self):
        """Reset belief to the approximate distribution after a 1000-step spawn."""
        belief = np.zeros(N, dtype=np.float64)
        belief[0] = 1.0
        for _ in range(300):
            belief = belief @ self.T
        s = belief.sum()
        self.belief = belief / s if s > 0 else np.ones(N) / N

    def update_from_search(self, search_loc, found: bool):
        """
        Update belief after a search action (ours or opponent's).

        Parameters:
            search_loc: (x, y) that was searched.
            found: True if rat was caught there (rat respawns), False otherwise.
        """
        if found:
            # Rat caught and respawned; reset to approximate stationary.
            belief = np.zeros(N, dtype=np.float64)
            belief[0] = 1.0
            for _ in range(300):
                belief = belief @ self.T
            s = belief.sum()
            self.belief = belief / s if s > 0 else np.ones(N) / N
        else:
            # Zero out the searched cell.
            idx = search_loc[1] * BOARD_SIZE + search_loc[0]
            self.belief[idx] = 0.0
            s = self.belief.sum()
            if s > 1e-12:
                self.belief /= s
            else:
                self.belief = np.ones(N, dtype=np.float64) / N

    def get_best_cell(self):
        """Return ((x, y), probability) of the most likely rat location."""
        idx = int(np.argmax(self.belief))
        x = idx % BOARD_SIZE
        y = idx // BOARD_SIZE
        return (x, y), float(self.belief[idx])

    def get_pos_probability(self, pos) -> float:
        idx = pos[1] * BOARD_SIZE + pos[0]
        return float(self.belief[idx])

    def get_belief(self):
        return self.belief

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _dist_likelihood(self, observed: int, actual: int) -> float:
        """P(observed | actual), accounting for the clamp-to-zero rule."""
        if observed == 0:
            # Any error that would produce <= 0 gets mapped to 0.
            p = 0.0
            for prob, offset in zip(DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS):
                if max(0, actual + offset) == 0:
                    p += prob
            return p
        else:
            # Non-zero observed: only one error value can produce it.
            error = observed - actual
            for prob, offset in zip(DISTANCE_ERROR_PROBS, DISTANCE_ERROR_OFFSETS):
                if offset == error:
                    return prob
            return 0.0

    def _get_cell_types(self, board):
        """Return a tuple of Cell values for all 64 cells (cached if unchanged)."""
        # Use bit masks as a fast change-detection key.
        key = (board._primed_mask, board._carpet_mask, board._blocked_mask)
        if self._cached_cell_types is None or key != getattr(self, '_cell_key', None):
            ct = []
            for i in range(N):
                x = i % BOARD_SIZE
                y = i // BOARD_SIZE
                bit = 1 << i
                if board._primed_mask & bit:
                    ct.append(Cell.PRIMED)
                elif board._carpet_mask & bit:
                    ct.append(Cell.CARPET)
                elif board._blocked_mask & bit:
                    ct.append(Cell.BLOCKED)
                else:
                    ct.append(Cell.SPACE)
            self._cached_cell_types = ct
            self._cell_key = key
        return self._cached_cell_types
