"""
NNNU — Neural Network Non-Use (v4 Final)
==========================================
Built by Masamichi & Tamaki

Zero-parameter, zero-regression, zero-learning causal discovery.
Inherits Λ³ core from BANKAI-MD / Lambda_inverse_problem lineage.

Core insight:
  Jumps determine WHEN to look.
  signed_mean determines WHAT is seen.
  Spurious filters determine WHAT to discard.

Architecture:
  1. Λ³ dimensionless displacement (diff / local_std)
  2. Percentile-based jump extraction (adaptive, no fitted threshold)
  3. signed_mean at each lag: mean(target_response × source_sign)
     → Strong causal: high signed_mean at specific lag
     → Weak causal: moderate signed_mean (still detectable!)
     → Noise: signed_mean ≈ 0
  4. Common ancestor filter (structural)
  5. Mediator filter (structural)
  6. Conditional signed_mean (multi-dim interaction)
  7. P-value via t-test on signed responses

"人間が因果ですって言ってるのは、相関性が何回か確認できました。以上。"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return args[0]
        def wrapper(fn):
            return fn
        return wrapper

logger = logging.getLogger("fisj.nnnu")


# ============================================================================
# JIT Core — Λ³ lineage
# ============================================================================


@njit
def _local_std(data_1d: np.ndarray, window: int) -> np.ndarray:
    n = len(data_1d)
    out = np.empty(n)
    half_w = window // 2
    for i in range(n):
        start = max(0, i - half_w)
        end = min(n, i + half_w + 1)
        subset = data_1d[start:end]
        mean = 0.0
        for k in range(len(subset)):
            mean += subset[k]
        mean /= len(subset)
        var = 0.0
        for k in range(len(subset)):
            var += (subset[k] - mean) ** 2
        var /= len(subset)
        out[i] = np.sqrt(var)
    return out


@njit
def _compute_displacement(data: np.ndarray, window: int) -> np.ndarray:
    n_frames, n_dims = data.shape
    disp = np.zeros((n_frames - 1, n_dims))
    for d in range(n_dims):
        lstd = _local_std(data[:, d], window)
        for t in range(n_frames - 1):
            denom = lstd[t + 1] if lstd[t + 1] > 1e-12 else 1e-12
            disp[t, d] = (data[t + 1, d] - data[t, d]) / denom
    return disp


# ============================================================================
# Data class
# ============================================================================


@dataclass
class NNNUResult:
    """Full NNNU output."""
    score_matrix: np.ndarray        # (N, N) continuous causal score
    binary_matrix: np.ndarray       # (N, N) binary adjacency
    lag_matrix: np.ndarray          # (N, N) best lag
    sign_matrix: np.ndarray         # (N, N) causal sign (+1/-1)
    p_matrix: np.ndarray            # (N, N) raw p-values
    q_matrix: np.ndarray            # (N, N) BH-FDR corrected
    jump_counts: np.ndarray         # (N,) jumps per dimension
    raw_score_matrix: np.ndarray    # (N, N) pre-filter scores
    n_dims: int = 0
    n_frames: int = 0
    total_jumps: int = 0


# ============================================================================
# Engine
# ============================================================================


class NNNUEngine:
    """
    Neural Network Non-Use: jump-triggered signed_mean causal discovery.

    Parameters
    ----------
    max_lag : int
        Maximum causal lag (provided by benchmark).
    local_std_window : int
        Λ³ local_std window.
    jump_percentile : float
        Percentile for jump extraction (94.0 = top 6%).
    alpha : float
        Significance threshold for binary adjacency.
    min_jumps : int
        Minimum source jumps required to evaluate an edge.
    """

    def __init__(
        self,
        max_lag: int = 5,
        local_std_window: int = 20,
        jump_percentile: float = 94.0,
        alpha: float = 0.05,
        min_jumps: int = 5,
    ):
        self.max_lag = max_lag
        self.local_std_window = local_std_window
        self.jump_percentile = jump_percentile
        self.alpha = alpha
        self.min_jumps = min_jumps

    def fit(self, data: np.ndarray) -> NNNUResult:
        n_frames, n_dims = data.shape

        # --- Step 1: Λ³ displacement ---
        disp = _compute_displacement(data, self.local_std_window)
        n_disp = len(disp)

        # --- Step 2: Jump extraction (percentile) ---
        jump_frames = {}    # dim -> array of frames
        jump_signs = {}     # dim -> array of signs (+1/-1)
        jump_counts = np.zeros(n_dims, dtype=int)

        for d in range(n_dims):
            abs_d = np.abs(disp[:, d])
            thr = np.percentile(abs_d, self.jump_percentile)
            frames = np.where(abs_d > thr)[0]
            jump_frames[d] = frames
            jump_signs[d] = np.sign(disp[frames, d]).astype(int)
            jump_counts[d] = len(frames)

        # --- Step 3: signed_mean scoring (all pairs × all lags) ---
        score_matrix = np.zeros((n_dims, n_dims))
        lag_matrix = np.zeros((n_dims, n_dims), dtype=int)
        sign_matrix = np.zeros((n_dims, n_dims))
        p_matrix = np.ones((n_dims, n_dims))
        # Store per-lag data for conditioning step
        all_signed_responses = {}  # (src, tgt, lag) -> array of signed responses

        for src in range(n_dims):
            frames = jump_frames[src]
            signs = jump_signs[src]
            n_jumps = len(frames)

            if n_jumps < self.min_jumps:
                continue

            for tgt in range(n_dims):
                if src == tgt:
                    continue

                best_score = 0.0
                best_lag = 0
                best_sign = 0.0
                best_pval = 1.0

                for lag in range(1, self.max_lag + 1):
                    valid = frames + lag < n_disp
                    if np.sum(valid) < self.min_jumps:
                        continue

                    v_frames = frames[valid]
                    v_signs = signs[valid]
                    responses = disp[v_frames + lag, tgt]

                    # signed response: response × source_sign
                    signed_resp = responses * v_signs
                    signed_mean = np.mean(signed_resp)

                    # Store for conditioning step
                    all_signed_responses[(src, tgt, lag)] = signed_resp

                    # t-test: is signed_mean significantly different from 0?
                    n = len(signed_resp)
                    if n > 2:
                        std = np.std(signed_resp, ddof=1)
                        if std > 1e-12:
                            t_stat = signed_mean / (std / np.sqrt(n))
                            from scipy.stats import t as t_dist
                            pval = float(2.0 * t_dist.sf(abs(t_stat), n - 1))
                        else:
                            pval = 0.0 if abs(signed_mean) > 0 else 1.0
                    else:
                        pval = 1.0

                    if abs(signed_mean) > abs(best_score):
                        best_score = signed_mean
                        best_lag = lag
                        best_sign = 1.0 if signed_mean > 0 else -1.0
                        best_pval = pval

                score_matrix[src, tgt] = abs(best_score)
                lag_matrix[src, tgt] = best_lag
                sign_matrix[src, tgt] = best_sign
                p_matrix[src, tgt] = best_pval

        raw_score_matrix = score_matrix.copy()

        # --- Step 4 & 5: Spurious filter ---
        filtered_score, filtered_lag, filtered_sign = self._spurious_filter(
            score_matrix.copy(), lag_matrix.copy(), sign_matrix.copy(), n_dims,
        )

        # --- Step 6: Conditional signed_mean (p-value update) ---
        _, cond_p = self._conditional_scoring(
            filtered_score, filtered_lag, p_matrix.copy(),
            disp, jump_frames, jump_signs, n_dims, n_disp,
        )

        # --- Step 6.5: BH-FDR correction ---
        q_matrix = self._bh_fdr(cond_p, n_dims)

        # --- Step 7: Score suppress (filter + BH-FDR → score discount) ---
        # Keep raw scores but suppress filtered/non-significant edges
        suppress_floor = 0.05
        out_score = score_matrix.copy()
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                # Spurious filter killed it → suppress
                if filtered_score[i, j] == 0 and score_matrix[i, j] > 0:
                    out_score[i, j] *= suppress_floor
                # BH-FDR not significant → suppress
                if q_matrix[i, j] >= self.alpha:
                    out_score[i, j] *= suppress_floor

        # --- Step 8: Binary (q-value + spurious filter) ---
        binary_matrix = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                if q_matrix[i, j] < self.alpha and filtered_score[i, j] > 0:
                    binary_matrix[i, j] = 1.0

        total_jumps = int(np.sum(jump_counts))
        logger.info(
            f"🎯 NNNU: {n_dims}d, {n_frames}f, "
            f"{total_jumps} jumps, "
            f"{int(np.sum(binary_matrix))} edges"
        )

        # out_score = signed_mean × suppress (filter/BH-FDR → AUC ranking)
        # binary_matrix = hard decision (q-value + spurious → F-measure)
        return NNNUResult(
            score_matrix=out_score,
            binary_matrix=binary_matrix,
            lag_matrix=lag_matrix,
            sign_matrix=sign_matrix.astype(int),
            p_matrix=cond_p,
            q_matrix=q_matrix,
            jump_counts=jump_counts,
            raw_score_matrix=raw_score_matrix,
            n_dims=n_dims,
            n_frames=n_frames,
            total_jumps=total_jumps,
        )

    # ------------------------------------------------------------------
    # Spurious filter
    # ------------------------------------------------------------------

    def _spurious_filter(
        self,
        score_matrix: np.ndarray,
        lag_matrix: np.ndarray,
        sign_matrix: np.ndarray,
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove common ancestor and mediator edges."""
        out_score = score_matrix.copy()
        out_lag = lag_matrix.copy()
        out_sign = sign_matrix.copy()

        for a in range(n_dims):
            for b in range(n_dims):
                if a == b or score_matrix[a, b] <= 0:
                    continue

                removed = False

                # Common ancestor: Z→A and Z→B both stronger
                for z in range(n_dims):
                    if z == a or z == b:
                        continue
                    if (score_matrix[z, a] > score_matrix[a, b]
                            and score_matrix[z, b] > score_matrix[a, b]):
                        expected = lag_matrix[z, a] + lag_matrix[a, b]
                        if abs(expected - lag_matrix[z, b]) <= 1:
                            out_score[a, b] = 0.0
                            out_lag[a, b] = 0
                            out_sign[a, b] = 0.0
                            removed = True
                            break

                if removed:
                    continue

                # Mediator: A→M→B explains A→B
                for m in range(n_dims):
                    if m == a or m == b:
                        continue
                    if score_matrix[a, m] <= 0 or score_matrix[m, b] <= 0:
                        continue
                    mediated = lag_matrix[a, m] + lag_matrix[m, b]
                    if abs(mediated - lag_matrix[a, b]) <= 1:
                        path = min(score_matrix[a, m], score_matrix[m, b])
                        if path >= score_matrix[a, b] * 0.8:
                            out_score[a, b] = 0.0
                            out_lag[a, b] = 0
                            out_sign[a, b] = 0.0
                            break

        return out_score, out_lag, out_sign

    # ------------------------------------------------------------------
    # BH-FDR correction
    # ------------------------------------------------------------------

    @staticmethod
    def _bh_fdr(p_matrix: np.ndarray, n_dims: int) -> np.ndarray:
        """Benjamini-Hochberg FDR correction on directed p-value matrix."""
        pairs = []
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                pairs.append((i, j, p_matrix[i, j]))

        if not pairs:
            return np.ones_like(p_matrix)

        raw = np.array([p for _, _, p in pairs], dtype=float)
        order = np.argsort(raw)
        sorted_p = raw[order]
        m = len(sorted_p)

        adjusted = np.zeros(m, dtype=float)
        adjusted[-1] = sorted_p[-1]
        for k in range(m - 2, -1, -1):
            adjusted[k] = min(adjusted[k + 1], sorted_p[k] * m / (k + 1))
        adjusted = np.clip(adjusted, 0.0, 1.0)

        q = np.ones_like(p_matrix)
        for rank, idx in enumerate(order):
            i, j, _ = pairs[idx]
            q[i, j] = adjusted[rank]
        np.fill_diagonal(q, 1.0)
        return q

    # ------------------------------------------------------------------
    # Conditional scoring (causal-path-aware)
    # ------------------------------------------------------------------

    def _conditional_scoring(
        self,
        score_matrix: np.ndarray,
        lag_matrix: np.ndarray,
        p_matrix: np.ndarray,
        disp: np.ndarray,
        jump_frames: dict,
        jump_signs: dict,
        n_dims: int,
        n_disp: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Causal-path-aware conditional scoring.

        For each edge src→tgt, find established causal targets of src
        (dims where src→M is stronger). If M responded to src's jump,
        then tgt's response might be indirect via M.

        Exclude those frames and recompute signed_mean.
        If it drops → indirect effect → kill or discount.

        Example:
          0→1 (score=1.76, established)
          0→3 (score=0.94, under test)

          Frame 38: dim0 jumped → dim1 responded at lag=2 → dim3 also responded
          → dim3's response may be via dim1 → exclude frame 38
          → clean signed_mean for 0→3 drops → indirect effect confirmed
        """
        from scipy.stats import t as t_dist

        out_score = score_matrix.copy()
        out_p = p_matrix.copy()

        for src in range(n_dims):
            # Find established downstream dims of src (stronger edges FROM src)
            downstream = []
            for m in range(n_dims):
                if m == src:
                    continue
                if score_matrix[src, m] > 0 and lag_matrix[src, m] > 0:
                    downstream.append((m, score_matrix[src, m], lag_matrix[src, m]))

            if not downstream:
                continue

            # Sort by score descending
            downstream.sort(key=lambda x: x[1], reverse=True)

            for tgt in range(n_dims):
                if src == tgt or score_matrix[src, tgt] <= 0:
                    continue

                lag = lag_matrix[src, tgt]
                if lag <= 0:
                    continue

                # Only condition on STRONGER downstream paths
                mediators = [
                    (m, m_lag) for m, m_score, m_lag in downstream
                    if m != tgt and m_score > score_matrix[src, tgt]
                ]

                if not mediators:
                    continue

                frames = jump_frames[src]
                signs = jump_signs[src]

                # For each src jump, check if a mediator also responded
                # If mediator responded → this frame's tgt response might be indirect
                clean_mask = np.ones(len(frames), dtype=bool)

                for m, m_lag in mediators:
                    m_jump_set = set(jump_frames[m].tolist())
                    for idx, f in enumerate(frames):
                        # Did mediator M respond at f + m_lag (±1)?
                        for offset in range(-1, 2):
                            if (f + m_lag + offset) in m_jump_set:
                                clean_mask[idx] = False
                                break

                clean_frames = frames[clean_mask]
                clean_signs = signs[clean_mask]

                if len(clean_frames) < self.min_jumps:
                    # Almost all frames had mediator response → likely indirect
                    out_score[src, tgt] *= 0.1  # Heavy discount
                    out_p[src, tgt] = 1.0
                    continue

                valid = clean_frames + lag < n_disp
                if np.sum(valid) < self.min_jumps:
                    out_score[src, tgt] *= 0.1
                    out_p[src, tgt] = 1.0
                    continue

                v_frames = clean_frames[valid]
                v_signs = clean_signs[valid]
                responses = disp[v_frames + lag, tgt]
                signed_resp = responses * v_signs
                cond_mean = np.mean(signed_resp)

                # t-test
                n = len(signed_resp)
                if n > 2:
                    std = np.std(signed_resp, ddof=1)
                    if std > 1e-12:
                        t_stat = cond_mean / (std / np.sqrt(n))
                        pval = float(2.0 * t_dist.sf(abs(t_stat), n - 1))
                    else:
                        pval = 0.0 if abs(cond_mean) > 0 else 1.0
                else:
                    pval = 1.0

                out_score[src, tgt] = abs(cond_mean)
                out_p[src, tgt] = pval

        return out_score, out_p


# ============================================================================
# Benchmark Adapter
# ============================================================================


class NNNUAdapter:
    """Benchmark adapter for NNNU."""
    method_name = "NNNU"

    def __init__(self, max_lag=5, jump_percentile=94.0, alpha=0.05, min_jumps=5):
        self.kwargs = dict(
            max_lag=max_lag, jump_percentile=jump_percentile,
            alpha=alpha, min_jumps=min_jumps,
        )

    def fit(self, df, cfg=None):
        from FISJ.adapter import MethodOutput
        names = list(df.columns)
        data = df.values.astype(np.float64)
        result = NNNUEngine(**self.kwargs).fit(data)
        return MethodOutput(
            method_name=self.method_name, names=names,
            adjacency_scores=result.score_matrix,
            adjacency_bin=result.binary_matrix.astype(int),
            directed_support=True, lag_support=True, sign_support=True,
            lag_matrix=result.lag_matrix, sign_matrix=result.sign_matrix,
            meta={"total_jumps": result.total_jumps, "p_matrix": result.p_matrix, "q_matrix": result.q_matrix},
        )
