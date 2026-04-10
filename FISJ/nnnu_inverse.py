"""
NNNU_Inverse Adapter
=====================
Built by Masamichi & Tamaki

Three-layer causal discovery: no partial correlation, no precision matrix.

  Layer 1 (NNNU):     All-frame signed_mean → 100% causal candidate extraction
  Layer 2 (Inverse):  Source-drop DI → Cascade separation + interventional evidence
  Layer 3 (Regime):   GenericRegimeDetector → Event-driven data rescue

Architecture:
  1. NNNU full-frame scoring (statistical power for normal data)
  2. Regime detection → per-regime NNNU re-scoring (rescue event-driven)
  3. score = max(full_score, regime_score) × DI_gate × suppress
  4. Inverse DI gate → Cascade separation
  5. Suppress (q-value + consistency)

"main.py 合掌 🙏"
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .nnnu import NNNUEngine, _calculate_local_std_1d
from .inverse_causal_engine import InverseCausalEngine, InverseCausalEngineConfig
from .network_analyzer_core_v2 import GenericRegimeDetector, GenericRegimeConfig

logger = logging.getLogger("fisj.nnnu_inverse")


class NNNUInverseAdapter:
    """
    NNNU + Inverse Engine + Regime Detection fusion.

    Three layers, no partial correlation:
      L1:   NNNU full-frame signed_mean (observational, all data)
      L1.5: Regime-aware NNNU (rescue event-driven patterns)
      L2:   Inverse Engine DI (interventional, cascade separation)

    Fusion:
      score = max(full_score, regime_score) × DI_gate × suppress
    """

    method_name = "NNNU_Inverse"

    def __init__(
        self,
        max_lag: int = 5,
        solver: str = "ridge",
        alpha: float = 0.05,
        delta_percentile: float = 90.0,
        suppress_floor: float = 0.05,
        regime_aware: bool = True,
        n_regimes: int = 3,
        min_segment_length: int = 40,
        method_name: str | None = None,
    ):
        self.max_lag = max_lag
        self.solver = solver
        self.alpha = alpha
        self.delta_percentile = delta_percentile
        self.suppress_floor = suppress_floor
        self.regime_aware = regime_aware
        self.n_regimes = n_regimes
        self.min_segment_length = min_segment_length
        if method_name is not None:
            self.method_name = method_name

    def fit(self, df: pd.DataFrame, cfg=None):
        from .adapter import MethodOutput

        names = list(df.columns)
        n = len(names)
        state_vectors = df.values.astype(np.float64)
        n_frames = state_vectors.shape[0]

        # === Layer 1: NNNU (observational, full-frame) ===
        nnnu = NNNUEngine(
            max_lag=self.max_lag,
            delta_percentile=self.delta_percentile,
            alpha=self.alpha,
            adaptive=True,
        )
        nnnu_result = nnnu.fit(state_vectors)

        # === Layer 1.5: Regime-aware rescue ===
        regime_labels = None
        if self.regime_aware and n_frames >= self.min_segment_length * 2:
            nnnu_result, regime_labels = self._regime_rescue(
                state_vectors, nnnu_result, n, n_frames,
            )

        # === Layer 2: Inverse (interventional) ===
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
        ice_result = InverseCausalEngine(ice_config).fit(
            state_vectors, dimension_names=names,
        )

        # === Fusion: max(full, regime) × DI_gate × suppress ===
        nnnu_score = nnnu_result.score_matrix.copy()
        nnnu_q = nnnu_result.q_matrix.copy()
        consistency = nnnu_result.consistency_matrix.copy()

        # DI gate (soft): 0.3 base + 0.7 × normalized DI
        di_matrix = ice_result.direct_score_matrix
        if di_matrix is not None:
            di_norm = np.maximum(di_matrix, 0.0)
            di_max = di_norm.max()
            if di_max > 0:
                di_norm = di_norm / di_max
            di_gate = 0.3 + 0.7 * di_norm
        else:
            di_gate = np.ones((n, n))
        np.fill_diagonal(di_gate, 0.0)

        # Fused score
        fused = nnnu_score * di_gate

        # Suppress: q-value + consistency
        min_consistency = 0.70 if n <= 5 else 0.65
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nnnu_q[i, j] >= self.alpha:
                    fused[i, j] *= self.suppress_floor
                if consistency[i, j] <= min_consistency:
                    fused[i, j] *= self.suppress_floor

        np.fill_diagonal(fused, 0.0)

        # Binary: q-value + consistency
        binary = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (nnnu_q[i, j] < self.alpha
                        and consistency[i, j] > min_consistency
                        and nnnu_result.binary_matrix[i, j] > 0):
                    binary[i, j] = 1

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=fused,
            adjacency_bin=binary,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            lag_matrix=nnnu_result.lag_matrix,
            sign_matrix=nnnu_result.sign_matrix,
            meta={
                "nnnu_q_matrix": nnnu_q,
                "nnnu_raw_score": nnnu_result.raw_score_matrix,
                "consistency_matrix": consistency,
                "di_matrix": di_matrix,
                "n_binary_edges": int(np.sum(binary)),
                "nnnu_total_jumps": nnnu_result.total_jumps,
                "regime_labels": regime_labels,
            },
        )

    # ------------------------------------------------------------------
    # Layer 1.5: Regime-aware rescue
    # ------------------------------------------------------------------

    def _regime_rescue(
        self,
        state_vectors: np.ndarray,
        nnnu_result,
        n_dims: int,
        n_frames: int,
    ):
        """
        Detect regimes and re-score undetected edges per regime.
        If per-regime score is stronger, rescue the edge.

        Solves: event-driven data where full-frame signed_mean
        gets diluted by non-event periods.
        """
        from scipy.stats import t as t_dist, binom

        # Detect regimes
        regime_config = GenericRegimeConfig(
            n_regimes=self.n_regimes,
            min_segment_length=self.min_segment_length,
        )
        detector = GenericRegimeDetector(regime_config)
        labels = detector.detect(state_vectors)
        segments = detector.build_segments(labels)

        if len(segments) <= 1:
            return nnnu_result, labels

        # Displacement for per-regime scoring
        disp = np.zeros((n_frames - 1, n_dims))
        for d in range(n_dims):
            lstd = _calculate_local_std_1d(state_vectors[:, d], 20)
            disp[:, d] = np.diff(state_vectors[:, d]) / (lstd[1:] + 1e-10)

        # Jumps for consistency
        jump_frames_all = {}
        jump_signs_all = {}
        for d in range(n_dims):
            abs_d = np.abs(disp[:, d])
            thr = np.percentile(abs_d, self.delta_percentile)
            frames = np.where(abs_d > thr)[0]
            jump_frames_all[d] = frames
            jump_signs_all[d] = np.sign(disp[frames, d]).astype(int)

        # Try to rescue undetected edges
        rescued = 0
        score_matrix = nnnu_result.score_matrix.copy()
        raw_score = nnnu_result.raw_score_matrix.copy()
        p_matrix = nnnu_result.p_matrix.copy()
        q_matrix = nnnu_result.q_matrix.copy()
        lag_matrix = nnnu_result.lag_matrix.copy()
        sign_matrix = nnnu_result.sign_matrix.copy()
        consistency_matrix = nnnu_result.consistency_matrix.copy()
        binary_matrix = nnnu_result.binary_matrix.copy()

        min_consistency = 0.70 if n_dims <= 5 else 0.65

        for src in range(n_dims):
            for tgt in range(n_dims):
                if src == tgt:
                    continue

                # Only rescue edges NOT already detected
                if binary_matrix[src, tgt] > 0:
                    continue

                best_seg_score = 0.0
                best_seg_lag = 0
                best_seg_sign = 0.0
                best_seg_pval = 1.0
                best_seg_cons = 0.5

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

                        src_signs = np.sign(src_d)
                        signed_resp = tgt_d * src_signs
                        signed_mean = np.mean(signed_resp)

                        # Jump consistency within segment
                        j_frames = jump_frames_all[src]
                        j_signs = jump_signs_all[src]
                        seg_mask = (j_frames >= s) & (j_frames + lag < e)
                        j_seg = j_frames[seg_mask]
                        s_seg = j_signs[seg_mask]

                        consistency = 0.5
                        if len(j_seg) >= 3:
                            j_resp = disp[j_seg + lag, tgt]
                            j_signed = j_resp * s_seg
                            same_rate = np.mean(j_signed > 0)
                            consistency = max(same_rate, 1 - same_rate)

                        cons_bonus = (consistency - 0.5) * 2.0
                        combined = abs(signed_mean) * (1.0 + cons_bonus)

                        # t-test on segment frames
                        nn = len(signed_resp)
                        t_pval = 1.0
                        if nn > 5:
                            std = np.std(signed_resp, ddof=1)
                            if std > 1e-12:
                                t_stat = signed_mean / (std / np.sqrt(nn))
                                t_pval = float(2.0 * t_dist.sf(abs(t_stat), nn - 1))
                            else:
                                t_pval = 0.0

                        # Binomial test on jump consistency in segment
                        b_pval = 1.0
                        if len(j_seg) >= 3 and consistency > 0.5:
                            n_same = int(round(consistency * len(j_seg)))
                            b_pval = float(2.0 * binom.sf(n_same - 1, len(j_seg), 0.5))

                        # Rescue requires BOTH tests to pass (strict)
                        pval = max(t_pval, b_pval)
                        # Rescue-specific: higher consistency bar
                        rescue_min_cons = 0.90

                        if (combined > best_seg_score
                                and consistency > rescue_min_cons
                                and pval < self.alpha):
                            best_seg_score = combined
                            best_seg_lag = lag
                            best_seg_sign = 1.0 if signed_mean > 0 else -1.0
                            best_seg_pval = pval
                            best_seg_cons = consistency

                # Rescue if regime score beats full-frame score
                if (best_seg_score > score_matrix[src, tgt]
                        and best_seg_pval < self.alpha
                        and best_seg_cons > 0.90):
                    score_matrix[src, tgt] = best_seg_score
                    raw_score[src, tgt] = best_seg_score
                    lag_matrix[src, tgt] = best_seg_lag
                    sign_matrix[src, tgt] = int(best_seg_sign)
                    p_matrix[src, tgt] = best_seg_pval
                    consistency_matrix[src, tgt] = best_seg_cons
                    # NOTE: Do NOT set binary — spurious filter was not re-applied
                    rescued += 1

        if rescued > 0:
            q_matrix = NNNUEngine._bh_fdr(p_matrix, n_dims)
            logger.info(f"   🔄 Regime rescue: {rescued} edges rescued")

        # Update result in-place
        nnnu_result.score_matrix = score_matrix
        nnnu_result.raw_score_matrix = raw_score
        nnnu_result.p_matrix = p_matrix
        nnnu_result.q_matrix = q_matrix
        nnnu_result.lag_matrix = lag_matrix
        nnnu_result.sign_matrix = sign_matrix.astype(int)
        nnnu_result.consistency_matrix = consistency_matrix
        nnnu_result.binary_matrix = binary_matrix

        return nnnu_result, labels
