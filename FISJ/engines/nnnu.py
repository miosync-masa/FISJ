"""
NNNU — Neural Network Non-Use (v6)
====================================
Built by Masamichi & Tamaki

Zero-parameter, zero-regression, zero-learning causal discovery.

Architecture:
  1. Λ³ event extraction (adaptive percentile + window)
  2. ALL-frame signed_mean: mean(disp_target × sign(disp_source))
     → Score from ALL frames (statistical power)
     → Consistency from JUMP frames (signal quality)
  3. Spurious filter (common ancestor + mediator, ±2 frame window)
  4. Conditional scoring (causal-path-aware exclusion)
  5. Per-source BH-FDR (independent experiments)
  6. Suppress scoring (filter + BH-FDR + consistency → discount)

"人間が因果ですって言ってるのは、相関性が何回か確認できました。以上。"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..core import (
    local_std_1d,
    rho_t_1d,
    extract_lambda3_events,
    benjamini_hochberg_per_source,
)

logger = logging.getLogger("fisj.engines.nnnu")


@dataclass
class NNNUResult:
    """Full NNNU output."""
    score_matrix: np.ndarray
    binary_matrix: np.ndarray
    lag_matrix: np.ndarray
    sign_matrix: np.ndarray
    p_matrix: np.ndarray
    q_matrix: np.ndarray
    jump_counts: np.ndarray
    raw_score_matrix: np.ndarray
    consistency_matrix: np.ndarray | None = None
    rho_t: np.ndarray | None = None
    adaptive_params: dict | None = None
    n_dims: int = 0
    n_frames: int = 0
    total_jumps: int = 0


class NNNUEngine:
    """
    Neural Network Non-Use causal discovery engine.

    Parameters
    ----------
    max_lag : int
        Maximum causal lag.
    local_std_window : int
        Λ³ local_std window (adaptive may override).
    rho_t_window : int
        ρT window (adaptive may override).
    delta_percentile : float
        Jump extraction percentile (adaptive may override).
    alpha : float
        Significance threshold for BH-FDR.
    min_jumps : int
        Minimum source jumps to evaluate an edge.
    adaptive : bool
        Enable adaptive parameter tuning.
    """

    def __init__(
        self,
        max_lag: int = 5,
        local_std_window: int = 20,
        rho_t_window: int = 30,
        delta_percentile: float = 90.0,
        alpha: float = 0.05,
        min_jumps: int = 5,
        adaptive: bool = True,
    ):
        self.max_lag = max_lag
        self.local_std_window_hint = local_std_window
        self.rho_t_window_hint = rho_t_window
        self.delta_percentile_hint = delta_percentile
        self.alpha = alpha
        self.min_jumps = min_jumps
        self.adaptive = adaptive

        self.local_std_window = local_std_window
        self.rho_t_window = rho_t_window
        self.delta_percentile = delta_percentile

    def fit(self, data: np.ndarray) -> NNNUResult:
        n_frames, n_dims = data.shape
        n_disp = n_frames - 1

        # --- Step 1: Λ³ event extraction ---
        events_pos, events_neg, rho_t, disp, _ = extract_lambda3_events(
            data,
            local_std_window=self.local_std_window,
            rho_t_window=self.rho_t_window,
            delta_percentile=self.delta_percentile,
        )

        # --- Step 1.5: Adaptive tuning ---
        adaptive_params = None
        if self.adaptive:
            adaptive_params = self._compute_adaptive_parameters(
                data, events_pos, events_neg, rho_t, n_frames, n_dims,
            )
            self.local_std_window = adaptive_params["local_std_window"]
            self.rho_t_window = adaptive_params["rho_t_window"]
            self.delta_percentile = adaptive_params["delta_percentile"]
            events_pos, events_neg, rho_t, disp, _ = extract_lambda3_events(
                data,
                local_std_window=self.local_std_window,
                rho_t_window=self.rho_t_window,
                delta_percentile=self.delta_percentile,
            )

        # --- Step 2: Jump frames ---
        jump_frames, jump_signs, jump_counts = self._build_jumps(
            events_pos, events_neg, disp, n_dims,
        )

        # --- Step 3: ALL-frame signed_mean × jump consistency ---
        (score_matrix, lag_matrix, sign_matrix,
         p_matrix, consistency_matrix) = self._score_edges(
            disp, jump_frames, jump_signs, n_dims, n_disp,
        )
        raw_score_matrix = score_matrix.copy()

        # --- Step 4: Spurious filter (ancestor + mediator) ---
        filtered_score, filtered_lag, _ = self._spurious_filter(
            score_matrix.copy(), lag_matrix.copy(), sign_matrix.copy(),
            events_pos, events_neg, n_dims,
        )

        # --- Step 5: Conditional scoring (causal-path-aware) ---
        _, cond_p = self._conditional_scoring(
            filtered_score, filtered_lag, p_matrix.copy(),
            disp, jump_frames, jump_signs, n_dims, n_disp,
        )

        # --- Step 6: Per-source BH-FDR ---
        q_matrix = benjamini_hochberg_per_source(cond_p)

        # --- Step 7: Suppress + binary ---
        score_matrix, binary_matrix = self._apply_suppress_and_binary(
            score_matrix, filtered_score, q_matrix, consistency_matrix, n_dims,
        )

        total_jumps = int(np.sum(jump_counts))
        logger.info(
            f"🎯 NNNU: {n_dims}d, {n_frames}f, {total_jumps} jumps, "
            f"{int(np.sum(binary_matrix))} edges"
        )

        return NNNUResult(
            score_matrix=score_matrix,
            binary_matrix=binary_matrix,
            lag_matrix=lag_matrix,
            sign_matrix=sign_matrix.astype(int),
            p_matrix=cond_p,
            q_matrix=q_matrix,
            jump_counts=jump_counts,
            raw_score_matrix=raw_score_matrix,
            consistency_matrix=consistency_matrix,
            rho_t=rho_t,
            adaptive_params=adaptive_params,
            n_dims=n_dims,
            n_frames=n_frames,
            total_jumps=total_jumps,
        )

    # ==================================================================
    # Step 2: Jump extraction
    # ==================================================================

    @staticmethod
    def _build_jumps(events_pos, events_neg, disp, n_dims):
        jump_frames = {}
        jump_signs = {}
        jump_counts = np.zeros(n_dims, dtype=int)

        for d in range(n_dims):
            fp = np.where(events_pos[:, d] > 0)[0]
            fn = np.where(events_neg[:, d] > 0)[0]
            all_f = np.concatenate([fp, fn])
            all_s = np.concatenate([
                np.ones(len(fp), dtype=int),
                -np.ones(len(fn), dtype=int),
            ])
            order = np.argsort(all_f)
            jump_frames[d] = all_f[order]
            jump_signs[d] = all_s[order]
            jump_counts[d] = len(all_f)

        return jump_frames, jump_signs, jump_counts

    # ==================================================================
    # Step 3: All-frame scoring + jump-based consistency
    # ==================================================================

    def _score_edges(self, disp, jump_frames, jump_signs, n_dims, n_disp):
        from scipy.stats import t as t_dist

        score_matrix = np.zeros((n_dims, n_dims))
        lag_matrix = np.zeros((n_dims, n_dims), dtype=int)
        sign_matrix = np.zeros((n_dims, n_dims))
        p_matrix = np.ones((n_dims, n_dims))
        consistency_matrix = np.zeros((n_dims, n_dims))

        for src in range(n_dims):
            frames = jump_frames[src]
            signs = jump_signs[src]

            for tgt in range(n_dims):
                if src == tgt:
                    continue

                best_score = 0.0
                best_lag = 0
                best_sign = 0.0
                best_pval = 1.0
                best_consistency = 0.5

                for lag in range(1, self.max_lag + 1):
                    if lag >= n_disp:
                        continue

                    # ALL frames: scoring + t-test
                    src_disp = disp[:-lag, src]
                    tgt_disp = disp[lag:, tgt]
                    signed_resp = tgt_disp * np.sign(src_disp)
                    signed_mean = float(np.mean(signed_resp))

                    n = len(signed_resp)
                    if n > 2:
                        std = float(np.std(signed_resp, ddof=1))
                        if std > 1e-12:
                            t_stat = signed_mean / (std / np.sqrt(n))
                            pval = float(2.0 * t_dist.sf(abs(t_stat), n - 1))
                        else:
                            pval = 0.0 if abs(signed_mean) > 0 else 1.0
                    else:
                        pval = 1.0

                    # JUMP frames: consistency
                    consistency = 0.5
                    if len(frames) >= self.min_jumps:
                        valid = frames + lag < n_disp
                        if np.sum(valid) >= self.min_jumps:
                            j_frames = frames[valid]
                            j_signs = signs[valid]
                            j_resp = disp[j_frames + lag, tgt]
                            j_signed = j_resp * j_signs
                            same_rate = float(np.mean(j_signed > 0))
                            consistency = max(same_rate, 1 - same_rate)

                    cons_bonus = (consistency - 0.5) * 2.0
                    combined = abs(signed_mean) * (1.0 + cons_bonus)

                    if combined > best_score:
                        best_score = combined
                        best_lag = lag
                        best_sign = 1.0 if signed_mean > 0 else -1.0
                        best_pval = pval
                        best_consistency = consistency

                score_matrix[src, tgt] = best_score
                lag_matrix[src, tgt] = best_lag
                sign_matrix[src, tgt] = best_sign
                p_matrix[src, tgt] = best_pval
                consistency_matrix[src, tgt] = best_consistency

        return score_matrix, lag_matrix, sign_matrix, p_matrix, consistency_matrix

    # ==================================================================
    # Step 4: Spurious filter
    # ==================================================================

    def _spurious_filter(
        self, score_matrix, lag_matrix, sign_matrix,
        events_pos, events_neg, n_dims,
    ):
        events_all = np.minimum(events_pos + events_neg, 1.0)
        out_score = score_matrix.copy()
        out_lag = lag_matrix.copy()
        out_sign = sign_matrix.copy()

        for a in range(n_dims):
            for b in range(n_dims):
                if a == b or score_matrix[a, b] <= 0:
                    continue

                lag_ab = int(lag_matrix[a, b])
                removed = False

                # Common ancestor
                for z in range(n_dims):
                    if z == a or z == b:
                        continue
                    if score_matrix[z, a] <= score_matrix[a, b]:
                        continue
                    if score_matrix[z, b] <= score_matrix[a, b]:
                        continue

                    lag_za = int(lag_matrix[z, a])
                    lag_zb = int(lag_matrix[z, b])
                    if abs((lag_za + lag_ab) - lag_zb) > max(1, lag_zb // 3):
                        continue

                    prob_with, prob_without = self._conditional_propagation(
                        events_all, a, b, z, lag_ab,
                    )
                    a_ev = events_all[:-lag_ab, a] if lag_ab > 0 else events_all[:, a]
                    z_active = self._z_activity_mask(events_all, z, len(a_ev))
                    n_without = int(np.sum((a_ev > 0) & (z_active == 0)))

                    prob_suspicious = (
                        prob_without < score_matrix[a, b] * 0.5
                        and prob_with > prob_without * 1.5
                    )
                    insufficient = n_without < 3

                    if prob_suspicious or insufficient:
                        out_score[a, b] = 0.0
                        out_lag[a, b] = 0
                        out_sign[a, b] = 0.0
                        removed = True
                        break

                if removed:
                    continue

                # Mediator
                for m in range(n_dims):
                    if m == a or m == b:
                        continue
                    if score_matrix[a, m] <= 0 or score_matrix[m, b] <= 0:
                        continue

                    mediated = int(lag_matrix[a, m]) + int(lag_matrix[m, b])
                    lag_diff = abs(mediated - lag_ab)

                    if lag_diff <= 1:
                        path_str = min(score_matrix[a, m], score_matrix[m, b])
                        if path_str >= score_matrix[a, b]:
                            out_score[a, b] = 0.0
                            out_lag[a, b] = 0
                            out_sign[a, b] = 0.0
                            break

                    if lag_diff <= max(2, lag_ab // 2):
                        prob_with, prob_without = self._conditional_propagation(
                            events_all, a, b, m, lag_ab,
                        )
                        if (prob_without < score_matrix[a, b] * 0.4
                                and prob_with > prob_without * 2.0):
                            out_score[a, b] = 0.0
                            out_lag[a, b] = 0
                            out_sign[a, b] = 0.0
                            break

        return out_score, out_lag, out_sign

    @staticmethod
    def _conditional_propagation(events_all, a, b, z, lag):
        if lag >= len(events_all) or lag <= 0:
            return 0.0, 0.0

        a_events = events_all[:-lag, a]
        b_events = events_all[lag:, b]

        z_active = np.zeros(len(a_events))
        for t in range(len(a_events)):
            z_start = max(0, t - 2)
            z_end = min(len(events_all), t + 3)
            if np.any(events_all[z_start:z_end, z] > 0):
                z_active[t] = 1.0

        m_with = (a_events > 0) & (z_active > 0)
        n_with = np.sum(m_with)
        p_with = float(np.sum(m_with * b_events) / n_with) if n_with > 0 else 0.0

        m_without = (a_events > 0) & (z_active == 0)
        n_without = np.sum(m_without)
        p_without = float(np.sum(m_without * b_events) / n_without) if n_without > 0 else 0.0

        return p_with, p_without

    @staticmethod
    def _z_activity_mask(events_all, z, length):
        z_active = np.zeros(length)
        for t in range(length):
            z_start = max(0, t - 2)
            z_end = min(len(events_all), t + 3)
            if np.any(events_all[z_start:z_end, z] > 0):
                z_active[t] = 1.0
        return z_active

    # ==================================================================
    # Step 5: Conditional scoring
    # ==================================================================

    def _conditional_scoring(
        self, score_matrix, lag_matrix, p_matrix,
        disp, jump_frames, jump_signs, n_dims, n_disp,
    ):
        from scipy.stats import t as t_dist

        out_score = score_matrix.copy()
        out_p = p_matrix.copy()

        for src in range(n_dims):
            downstream = [
                (m, score_matrix[src, m], int(lag_matrix[src, m]))
                for m in range(n_dims)
                if m != src and score_matrix[src, m] > 0 and lag_matrix[src, m] > 0
            ]
            if not downstream:
                continue
            downstream.sort(key=lambda x: x[1], reverse=True)

            for tgt in range(n_dims):
                if src == tgt or score_matrix[src, tgt] <= 0:
                    continue

                lag = int(lag_matrix[src, tgt])
                if lag <= 0:
                    continue

                mediators = [
                    (m, m_lag) for m, m_score, m_lag in downstream
                    if m != tgt and m_score > score_matrix[src, tgt]
                ]
                if not mediators:
                    continue

                frames = jump_frames[src]
                signs = jump_signs[src]

                clean_mask = np.ones(len(frames), dtype=bool)
                for m, m_lag in mediators:
                    m_jump_set = set(jump_frames[m].tolist())
                    for idx, f in enumerate(frames):
                        for offset in range(-1, 2):
                            if (f + m_lag + offset) in m_jump_set:
                                clean_mask[idx] = False
                                break

                clean_frames = frames[clean_mask]
                clean_signs = signs[clean_mask]

                if len(clean_frames) < self.min_jumps:
                    out_score[src, tgt] *= 0.1
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
                cond_mean = float(np.mean(signed_resp))

                n = len(signed_resp)
                if n > 2:
                    std = float(np.std(signed_resp, ddof=1))
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

    # ==================================================================
    # Step 7: Suppress + binary
    # ==================================================================

    def _apply_suppress_and_binary(
        self, score_matrix, filtered_score, q_matrix, consistency_matrix, n_dims,
    ):
        suppress_floor = 0.05
        min_consistency = 0.70 if n_dims <= 5 else 0.65

        out_score = score_matrix.copy()
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                if filtered_score[i, j] == 0 and score_matrix[i, j] > 0:
                    out_score[i, j] *= suppress_floor
                if q_matrix[i, j] >= self.alpha:
                    out_score[i, j] *= suppress_floor
                if consistency_matrix[i, j] <= min_consistency:
                    out_score[i, j] *= suppress_floor

        binary = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                if (q_matrix[i, j] < self.alpha
                        and filtered_score[i, j] > 0
                        and consistency_matrix[i, j] > min_consistency):
                    binary[i, j] = 1.0

        return out_score, binary

    # ==================================================================
    # Adaptive parameters
    # ==================================================================

    def _compute_adaptive_parameters(
        self, state_vectors, events_pos, events_neg, rho_t, n_frames, n_dims,
    ):
        events_all = np.minimum(events_pos + events_neg, 1.0)
        event_density = float(np.mean(events_all))

        cofiring_rates = []
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                cofiring_rates.append(float(np.mean(events_all[:, i] * events_all[:, j])))
        mean_cofiring = float(np.mean(cofiring_rates)) if cofiring_rates else 0.0

        rho_t_means = np.mean(rho_t, axis=0)
        rho_t_overall = float(np.mean(rho_t_means))
        rho_t_cv = (
            float(np.std(rho_t_means) / (rho_t_overall + 1e-10))
            if rho_t_overall > 1e-10 else 0.0
        )

        temporal_changes = np.diff(state_vectors, axis=0)
        temporal_vol = float(np.mean(np.std(temporal_changes, axis=0)))
        global_std = float(np.std(state_vectors))
        vol_ratio = temporal_vol / (global_std + 1e-10)

        pct = self.delta_percentile_hint
        if event_density > 0.15:
            pct = min(pct + 2.0, 97.0)
        elif event_density < 0.03:
            pct = max(pct - 3.0, 80.0)
        if mean_cofiring > 0.02:
            pct = min(pct + 1.0, 97.0)
        if n_dims > 10:
            pct = min(pct + 1.0, 97.0)

        window_scale = 1.0
        if n_frames < 200:
            window_scale *= 0.7
        elif n_frames > 1000:
            window_scale *= 1.3
        if vol_ratio > 1.5:
            window_scale *= 0.8

        local_std_window = int(
            np.clip(self.local_std_window_hint * window_scale, 5, n_frames // 5)
        )
        rho_t_window = int(
            np.clip(self.rho_t_window_hint * window_scale, 5, n_frames // 5)
        )

        return {
            "delta_percentile": float(pct),
            "local_std_window": local_std_window,
            "rho_t_window": rho_t_window,
            "diagnostics": {
                "event_density": event_density,
                "mean_cofiring": mean_cofiring,
                "rho_t_cv": rho_t_cv,
                "vol_ratio": vol_ratio,
                "n_frames": n_frames,
                "n_dims": n_dims,
            },
        }
