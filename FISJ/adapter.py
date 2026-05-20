"""
FISJ Adapter — unified interface
================================
Built by Masamichi & Tamaki

Single adapter, method-switched. Fusion logic is internal.

Usage:
    >>> from FISJ import FISJAdapter
    >>> adapter = FISJAdapter(method="nnnu_inverse", max_lag=5)
    >>> result = adapter.fit(df)

Methods:
    - 'nnnu':         NNNU only (Level 1, observational)
    - 'nnnu_inverse': NNNU + Inverse + Regime rescue (recommended)
    - 'fusion':       NetworkCore + Inverse (legacy, AUC-strongest)
    - 'inverse':      Inverse only

Lag matrix convention (unified across all methods):
    lag_matrix[i, j] = k >= 0   ← directed edge i→j with time delay k
    lag_matrix[i, j] = 0        ← no edge from i to j

    NetworkAnalyzerCore natively uses anti-symmetric (signed lag) format;
    this adapter normalizes it to the unified positive-only convention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import local_std_1d, benjamini_hochberg_per_source
from .engines import (
    NNNUEngine,
    InverseCausalEngine, InverseCausalEngineConfig,
    NetworkAnalyzerCore,
    GenericRegimeDetector, GenericRegimeConfig,
)

logger = logging.getLogger("fisj.adapter")


@dataclass
class MethodOutput:
    """Standard output for any FISJ method."""
    method_name: str
    names: list[str]
    adjacency_scores: np.ndarray
    adjacency_bin: np.ndarray | None = None
    lag_matrix: np.ndarray | None = None
    sign_matrix: np.ndarray | None = None
    directed_support: bool = True
    lag_support: bool = True
    sign_support: bool = True
    meta: dict = field(default_factory=dict)


class FISJAdapter:
    """
    Unified causal discovery adapter.

    Parameters
    ----------
    method : str
        One of 'nnnu', 'nnnu_inverse', 'fusion', 'inverse'.
    max_lag : int
        Maximum causal lag. Strictly enforced (adaptive=False for fairness).
    solver : str
        Inverse engine solver ('ridge', 'lasso', 'auto').
    alpha : float
        Significance threshold.
    delta_percentile : float
        Jump extraction percentile.
    suppress_floor : float
        Score discount factor for filtered edges.
    regime_aware : bool
        Enable regime rescue (nnnu_inverse only).
    network_adaptive : bool
        If True, NetworkAnalyzerCore auto-adjusts max_lag.
        Default False for benchmark fairness (max_lag strictly enforced).
    nnnu_adaptive : bool
        If True, NNNU auto-tunes its internal parameters (percentile, windows).
        Default True — this is internal tuning, doesn't violate max_lag.
    """

    SUPPORTED_METHODS = ("nnnu", "nnnu_inverse", "fusion", "inverse")

    def __init__(
        self,
        method: str = "nnnu_inverse",
        max_lag: int = 5,
        solver: str = "ridge",
        alpha: float = 0.05,
        delta_percentile: float = 90.0,
        suppress_floor: float = 0.05,
        regime_aware: bool = True,
        n_regimes: int = 3,
        min_segment_length: int = 40,
        network_adaptive: bool = False,
        nnnu_adaptive: bool = True,
        method_name: str | None = None,
    ):
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method!r}. Choose from {self.SUPPORTED_METHODS}"
            )

        self.method = method
        self.max_lag = max_lag
        self.solver = solver
        self.alpha = alpha
        self.delta_percentile = delta_percentile
        self.suppress_floor = suppress_floor
        self.regime_aware = regime_aware
        self.n_regimes = n_regimes
        self.min_segment_length = min_segment_length
        self.network_adaptive = network_adaptive
        self.nnnu_adaptive = nnnu_adaptive
        self.method_name = method_name or method.upper()

    def fit(self, df: pd.DataFrame, cfg=None) -> MethodOutput:
        if self.method == "nnnu":
            return self._run_nnnu(df)
        elif self.method == "nnnu_inverse":
            return self._run_nnnu_inverse(df)
        elif self.method == "fusion":
            return self._run_fusion(df)
        elif self.method == "inverse":
            return self._run_inverse(df)

    # ==================================================================
    # Method: NNNU only
    # ==================================================================

    def _run_nnnu(self, df):
        names = list(df.columns)
        data = df.values.astype(np.float64)

        engine = NNNUEngine(
            max_lag=self.max_lag,
            delta_percentile=self.delta_percentile,
            alpha=self.alpha,
            adaptive=self.nnnu_adaptive,
        )
        r = engine.fit(data)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=r.score_matrix,
            adjacency_bin=r.binary_matrix.astype(int),
            lag_matrix=r.lag_matrix,
            sign_matrix=r.sign_matrix,
            meta={
                "q_matrix": r.q_matrix,
                "consistency_matrix": r.consistency_matrix,
                "total_jumps": r.total_jumps,
            },
        )

    # ==================================================================
    # Method: NNNU_Inverse (OR-combined hybrid)
    # ==================================================================
    # Combines two complementary causal-discovery philosophies:
    #
    #   NNNU (Layer 1):    Consistency-based — "do they move together?"
    #                      Strong for high-SNR causal jumps.
    #                      Weak for low-SNR continuous coupling.
    #
    #   Inverse (Layer 2): Uniqueness-based — "is this the only explanation?"
    #                      Strong even for low-SNR coupling (Ridge solver
    #                      estimates coefficients directly).
    #                      Can be fooled by confounders (spurious DI).
    #
    # The OR-combined version takes the maximum of both signals and adds
    # an agreement bonus when both engines agree. NNNU's textbook filter
    # is used to suppress confounder-induced spurious DI values.
    # ==================================================================

    def _run_nnnu_inverse(self, df):
        names = list(df.columns)
        n = len(names)
        data = df.values.astype(np.float64)
        n_frames = data.shape[0]

        # Layer 1: NNNU (consistency-based)
        engine = NNNUEngine(
            max_lag=self.max_lag,
            delta_percentile=self.delta_percentile,
            alpha=self.alpha,
            adaptive=self.nnnu_adaptive,
        )
        nnnu_r = engine.fit(data)

        # Layer 1.5: Regime rescue (score-only)
        regime_labels = None
        if self.regime_aware and n_frames >= self.min_segment_length * 2:
            nnnu_r, regime_labels = self._regime_rescue(data, nnnu_r, n, n_frames)

        # Layer 2: Inverse (uniqueness-based)
        ice_config = InverseCausalEngineConfig(
            max_lag=self.max_lag,
            ar_lag=1,
            solver=self.solver,
            standardize=True,
            include_intercept=True,
            validation_fraction=0.25,
            use_backward_check=True,
            refit_on_drop=False,
            residualize_ar=True,
            compute_direct_irreducibility=True,
        )
        ice_r = InverseCausalEngine(ice_config).fit(data, dimension_names=names)

        di_matrix = (
            ice_r.direct_score_matrix.copy()
            if ice_r.direct_score_matrix is not None
            else np.zeros((n, n))
        )

        # ============================================================
        # Step 1: Propagate NNNU's textbook filter to Inverse DI
        # ============================================================
        # NNNU's spurious filter identifies common-ancestor and
        # mediator patterns. When NNNU has high confidence (q < alpha
        # at raw level) and filtered the edge out, we discount the
        # corresponding DI value because Inverse alone can be fooled
        # by confounders.
        di_filtered = self._propagate_nnnu_filter_to_inverse(
            di_matrix, nnnu_r, n,
        )

        # ============================================================
        # Step 2: Normalize both signals
        # ============================================================
        nnnu_norm = self._normalize_score(nnnu_r.score_matrix)
        di_norm = self._normalize_score(di_filtered)

        # ============================================================
        # Step 3: OR-combine — max() + agreement bonus
        # ============================================================
        # Take whichever engine is more confident, and add a bonus
        # when both engines agree (uniqueness AND consistency).
        scores = self._or_combine(nnnu_norm, di_norm, bonus=0.3)
        np.fill_diagonal(scores, 0.0)

        # ============================================================
        # Step 4: Binary — NNNU strict + Inverse rescue
        # ============================================================
        binary = nnnu_r.binary_matrix.astype(int).copy()
        binary = self._inverse_rescue_binary(
            binary, di_norm, nnnu_r, n,
        )

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=binary,
            lag_matrix=nnnu_r.lag_matrix,
            sign_matrix=nnnu_r.sign_matrix,
            meta={
                "q_matrix": nnnu_r.q_matrix,
                "consistency_matrix": nnnu_r.consistency_matrix,
                "di_matrix": di_matrix,
                "di_filtered": di_filtered,
                "nnnu_score_normalized": nnnu_norm,
                "di_normalized": di_norm,
                "total_jumps": nnnu_r.total_jumps,
                "regime_labels": regime_labels,
            },
        )

    # ==================================================================
    # OR-combination primitives
    # ==================================================================

    def _propagate_nnnu_filter_to_inverse(
        self, di_matrix: np.ndarray, nnnu_r, n: int,
    ) -> np.ndarray:
        """
        Apply NNNU's spurious-edge filter to Inverse DI scores.

        When NNNU has detected an edge at the raw level (raw_score > 0)
        but then explicitly filtered it as spurious (final score = 0)
        AND has high statistical confidence in that judgment
        (q-value < alpha), the corresponding Inverse DI value is
        discounted. This handles confounder-induced uniqueness signals
        that Inverse alone cannot distinguish from true causality.
        """
        raw = nnnu_r.raw_score_matrix
        final = nnnu_r.score_matrix
        q_matrix = nnnu_r.q_matrix

        out = di_matrix.copy()
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # NNNU's textbook filter said "spurious" with confidence
                is_filtered = raw[i, j] > 0 and final[i, j] == 0
                if is_filtered and q_matrix[i, j] < self.alpha:
                    out[i, j] *= self.suppress_floor
        return out

    @staticmethod
    def _normalize_score(score_matrix: np.ndarray) -> np.ndarray:
        """Normalize a score matrix to [0, 1] by its max."""
        out = np.maximum(score_matrix, 0.0)
        peak = out.max()
        if peak > 1e-10:
            out = out / peak
        return out

    @staticmethod
    def _or_combine(a: np.ndarray, b: np.ndarray, bonus: float = 0.3) -> np.ndarray:
        """
        OR-combine two normalized score matrices.

        scores = max(a, b) + bonus * (a * b)

        - max() takes the more confident engine's verdict per edge
        - agreement bonus rewards edges where both engines agree
        """
        combined = np.maximum(a, b)
        agreement = a * b
        return combined + bonus * agreement

    def _inverse_rescue_binary(
        self, binary: np.ndarray, di_norm: np.ndarray, nnnu_r, n: int,
        di_threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Add edges to binary where Inverse is highly confident AND
        NNNU didn't explicitly filter them as spurious.

        Inverse DI normalized >= di_threshold and NNNU's raw signal > 0
        but not flagged as spurious → add to binary.
        """
        raw = nnnu_r.raw_score_matrix
        final = nnnu_r.score_matrix
        q_matrix = nnnu_r.q_matrix

        out = binary.copy()
        for i in range(n):
            for j in range(n):
                if i == j or out[i, j] == 1:
                    continue
                if di_norm[i, j] < di_threshold:
                    continue
                # NNNU has at least a hint of signal
                if raw[i, j] <= 0:
                    continue
                # NNNU has NOT confidently filtered as spurious
                is_confident_spurious = (
                    raw[i, j] > 0 and final[i, j] == 0
                    and q_matrix[i, j] < self.alpha
                )
                if not is_confident_spurious:
                    out[i, j] = 1
        return out


    # ==================================================================
    # Method: Fusion (NetworkCore + Inverse) — legacy
    # ==================================================================
    # Normalizes NetworkAnalyzerCore's signed-lag matrix to the unified
    # positive-only lag convention, ensuring consistency with NNNU output.

    def _run_fusion(self, df):
        names = list(df.columns)
        n = len(names)
        data = df.values.astype(np.float64)

        # NetworkCore — adaptive=False by default to strictly enforce max_lag
        network = NetworkAnalyzerCore(
            max_lag=self.max_lag,
            adaptive=self.network_adaptive,
        )
        net_r = network.analyze(data, dimension_names=names)

        # Inverse
        ice_config = InverseCausalEngineConfig(
            max_lag=self.max_lag, ar_lag=1, solver=self.solver,
            standardize=True, include_intercept=True,
            validation_fraction=0.25, use_backward_check=True,
            refit_on_drop=False, residualize_ar=True,
            compute_direct_irreducibility=True,
        )
        ice_r = InverseCausalEngine(ice_config).fit(data, dimension_names=names)

        # ============================================================
        # Reconstruct direction-aware matrices from causal_network.
        # NetworkAnalyzerCore's causal_lag_matrix uses anti-symmetric
        # signed-lag format; we extract directed edges via the
        # high-level causal_network list and rebuild positive-only
        # matrices that match the NNNU convention.
        # ============================================================
        lag_matrix = np.zeros((n, n), dtype=int)
        binary = np.zeros((n, n), dtype=int)
        net_score = np.zeros((n, n))
        sign_matrix = np.zeros((n, n), dtype=int)

        # Directed causal links — positive lag, directional
        for link in net_r.causal_network:
            i, j = link.from_dim, link.to_dim
            binary[i, j] = 1
            lag_matrix[i, j] = int(link.lag)
            net_score[i, j] = float(link.strength)
            sign_matrix[i, j] = 1 if link.correlation >= 0 else -1

        # Undirected sync links — fill both directions (no lag)
        for link in net_r.sync_network:
            i, j = link.from_dim, link.to_dim
            binary[i, j] = 1
            binary[j, i] = 1
            # Don't overwrite if a directed causal link already set the score
            if net_score[i, j] == 0:
                net_score[i, j] = float(link.strength)
                sign_matrix[i, j] = 1 if link.correlation >= 0 else -1
            if net_score[j, i] == 0:
                net_score[j, i] = float(link.strength)
                sign_matrix[j, i] = 1 if link.correlation >= 0 else -1

        # Apply Inverse DI gate (soft amplification of confirmed edges)
        scores = self._apply_di_gate(net_score, ice_r.direct_score_matrix, n)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=binary,
            lag_matrix=lag_matrix,
            sign_matrix=sign_matrix,
            meta={
                "di_matrix": ice_r.direct_score_matrix,
                "sync_matrix": net_r.sync_matrix,
                "causal_matrix": net_r.causal_matrix,
                "adaptive_params": net_r.adaptive_params,
                "n_sync_links": net_r.n_sync_links,
                "n_causal_links": net_r.n_causal_links,
            },
        )

    # ==================================================================
    # Method: Inverse only
    # ==================================================================

    def _run_inverse(self, df):
        names = list(df.columns)
        data = df.values.astype(np.float64)
        n = len(names)

        ice_config = InverseCausalEngineConfig(
            max_lag=self.max_lag, ar_lag=1, solver=self.solver,
            standardize=True, include_intercept=True,
            validation_fraction=0.25, use_backward_check=True,
            refit_on_drop=False, residualize_ar=True,
            compute_direct_irreducibility=True,
        )
        ice_r = InverseCausalEngine(ice_config).fit(data, dimension_names=names)

        scores = ice_r.direct_score_matrix if ice_r.direct_score_matrix is not None else np.zeros((n, n))
        binary = (scores > 0).astype(int)
        np.fill_diagonal(binary, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=binary,
            lag_matrix=np.zeros((n, n), dtype=int),
            sign_matrix=np.sign(scores).astype(int),
            meta={"di_matrix": scores},
        )

    # ==================================================================
    # Fusion primitives (internal)
    # ==================================================================

    @staticmethod
    def _apply_di_gate(base_score, di_matrix, n_dims, base=0.3):
        """Apply Inverse DI as a soft gate to base scores."""
        if di_matrix is None:
            return base_score.copy()

        di_norm = np.maximum(di_matrix, 0.0)
        di_max = di_norm.max()
        if di_max > 0:
            di_norm = di_norm / di_max
        gate = base + (1.0 - base) * di_norm
        np.fill_diagonal(gate, 0.0)
        return base_score * gate

    def _apply_suppress(self, scores, q_matrix, consistency_matrix, n_dims):
        """Apply q-value + consistency suppress."""
        min_cons = 0.70 if n_dims <= 5 else 0.65
        out = scores.copy()
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                if q_matrix[i, j] >= self.alpha:
                    out[i, j] *= self.suppress_floor
                if consistency_matrix[i, j] <= min_cons:
                    out[i, j] *= self.suppress_floor
        np.fill_diagonal(out, 0.0)
        return out

    def _regime_rescue(self, data, nnnu_r, n_dims, n_frames):
        """
        Detect regimes and re-score undetected edges per regime.
        Score-only rescue (does not modify binary).
        """
        from scipy.stats import t as t_dist, binom

        config = GenericRegimeConfig(
            n_regimes=self.n_regimes,
            min_segment_length=self.min_segment_length,
        )
        detector = GenericRegimeDetector(config)
        labels = detector.detect(data)
        segments = detector.build_segments(labels)

        if len(segments) <= 1:
            return nnnu_r, labels

        # Recompute displacement
        disp = np.zeros((n_frames - 1, n_dims))
        for d in range(n_dims):
            lstd = local_std_1d(data[:, d], 20)
            disp[:, d] = np.diff(data[:, d]) / (lstd[1:] + 1e-10)

        # Jumps
        jump_frames = {}
        jump_signs = {}
        for d in range(n_dims):
            abs_d = np.abs(disp[:, d])
            thr = np.percentile(abs_d, self.delta_percentile)
            frames = np.where(abs_d > thr)[0]
            jump_frames[d] = frames
            jump_signs[d] = np.sign(disp[frames, d]).astype(int)

        score_matrix = nnnu_r.score_matrix.copy()
        rescued = 0

        for src in range(n_dims):
            for tgt in range(n_dims):
                if src == tgt:
                    continue
                if nnnu_r.binary_matrix[src, tgt] > 0:
                    continue

                best_score = 0.0

                for seg in segments:
                    s, e = seg.start, min(seg.end, len(disp))
                    if e - s < 30:
                        continue

                    for lag in range(1, self.max_lag + 1):
                        if lag >= e - s:
                            continue

                        src_d = disp[s:e - lag, src]
                        tgt_d = disp[s + lag:e, tgt]
                        if len(src_d) < 10:
                            continue

                        signed_resp = tgt_d * np.sign(src_d)
                        signed_mean = float(np.mean(signed_resp))

                        # Jump consistency within segment
                        j_f = jump_frames[src]
                        j_s = jump_signs[src]
                        mask = (j_f >= s) & (j_f + lag < e)
                        j_seg = j_f[mask]
                        s_seg = j_s[mask]

                        consistency = 0.5
                        if len(j_seg) >= 3:
                            j_resp = disp[j_seg + lag, tgt]
                            j_signed = j_resp * s_seg
                            same_rate = float(np.mean(j_signed > 0))
                            consistency = max(same_rate, 1 - same_rate)

                        combined = abs(signed_mean) * (1.0 + (consistency - 0.5) * 2.0)

                        # Strict: cons > 0.90 + both t-test and binomial pass
                        if consistency <= 0.90:
                            continue

                        nn = len(signed_resp)
                        t_pval = 1.0
                        if nn > 5:
                            std = float(np.std(signed_resp, ddof=1))
                            if std > 1e-12:
                                t_stat = signed_mean / (std / np.sqrt(nn))
                                t_pval = float(2.0 * t_dist.sf(abs(t_stat), nn - 1))

                        b_pval = 1.0
                        if len(j_seg) >= 3:
                            n_same = int(round(consistency * len(j_seg)))
                            b_pval = float(2.0 * binom.sf(n_same - 1, len(j_seg), 0.5))

                        if max(t_pval, b_pval) < self.alpha and combined > best_score:
                            best_score = combined

                if best_score > score_matrix[src, tgt]:
                    score_matrix[src, tgt] = best_score
                    rescued += 1

        if rescued > 0:
            nnnu_r.score_matrix = score_matrix
            logger.info(f"   🔄 Regime rescue: {rescued} edges score-boosted")

        return nnnu_r, labels


# Convenience function for CauseMe-style submission
def run_fisj(data, max_lag=3, method="nnnu_inverse", **kwargs):
    """
    CauseMe-style runner.

    Returns
    -------
    scores : ndarray (N, N)
    lags : ndarray (N, N) — positive values only (direction-aware)
    pvals : ndarray (N, N)
    """
    n_dims = data.shape[1]
    df = pd.DataFrame(data, columns=[f"V{i}" for i in range(n_dims)])
    adapter = FISJAdapter(method=method, max_lag=max_lag, **kwargs)
    result = adapter.fit(df)
    scores = result.adjacency_scores
    lags = result.lag_matrix if result.lag_matrix is not None else np.zeros((n_dims, n_dims), dtype=int)
    pvals = result.meta.get("q_matrix", np.ones((n_dims, n_dims)))
    return scores, lags, pvals


# ==================================================================
# Top-K binary utility
# ==================================================================

def make_topk_binary(scores: np.ndarray, k: int) -> np.ndarray:
    """
    Convert score matrix to binary by selecting the top-K edges.

    FISJ's default ``adjacency_bin`` is conservative — it returns
    only edges that pass statistical significance, spurious-edge
    filtering, and directional consistency. This is good for
    production use ("when FISJ says yes, it means yes") but
    pessimistic for benchmarks where weak-signal datasets push
    the strict binary toward zero detections even when the score
    ranking is meaningful.

    Top-K binary uses the score ranking directly: the K highest
    off-diagonal scores become edges, regardless of the strict
    threshold. This matches the evaluation protocol of CauseMe,
    TimeGraph, and most published causal-discovery benchmarks,
    where the true edge count is part of the experiment metadata.

    Parameters
    ----------
    scores : np.ndarray, shape (n, n)
        Score matrix from a ``FISJAdapter.fit()`` call's
        ``adjacency_scores`` field.
    k : int
        Number of edges to retain (typically the true edge count
        of the ground-truth graph, provided by the benchmark).

    Returns
    -------
    np.ndarray, shape (n, n), dtype=int
        Binary adjacency matrix with at most ``k`` ones placed at
        the positions of the largest off-diagonal scores. The
        diagonal is always zero.

    Examples
    --------
    >>> result = adapter.fit(df)
    >>> binary_topk = make_topk_binary(result.adjacency_scores, k=true_edge_count)
    >>> # Use binary_topk in benchmark evaluation
    """
    if scores is None or scores.size == 0:
        return np.zeros((0, 0), dtype=int)

    n = scores.shape[0]
    binary = np.zeros((n, n), dtype=int)

    if k <= 0:
        return binary

    # Collect off-diagonal positive scores
    flat = []
    for i in range(n):
        for j in range(n):
            if i != j and scores[i, j] > 0:
                flat.append((float(scores[i, j]), i, j))

    if not flat:
        return binary

    flat.sort(reverse=True)

    # Take top-K (or fewer if not enough positive scores)
    for _, i, j in flat[:k]:
        binary[i, j] = 1

    return binary
