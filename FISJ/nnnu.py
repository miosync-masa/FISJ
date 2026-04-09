"""
NNNU — Neural Network Non-Use (v5)
====================================
Built by Masamichi & Tamaki

Zero-parameter, zero-regression, zero-learning causal discovery.
Λ³ core inherited from BANKAI-MD / GETTER One lineage.

v5: GETTER One component integration
  - _calculate_local_std_1d / _calculate_rho_t_1d (Λ³ native)
  - _extract_lambda3_events (ΔΛC event extraction)
  - _compute_adaptive_parameters (data-driven auto-tuning)
  - _conditional_propagation (±2 frame window spurious filter)

Architecture:
  1. Λ³ event extraction (adaptive percentile + window)
  2. signed_mean at each lag: mean(target_displacement × source_sign)
     → Jumps determine WHEN to look
     → signed_mean determines WHAT is seen
  3. Common ancestor filter (conditional propagation probability)
  4. Mediator filter (lag consistency)
  5. Conditional scoring (causal-path-aware frame exclusion)
  6. BH-FDR correction
  7. Suppress scoring (filter + BH-FDR → score discount for AUC)

"人間が因果ですって言ってるのは、相関性が何回か確認できました。以上。"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger("fisj.nnnu")


# ============================================================================
# Λ³ 1D Functions — from GETTER One lineage
# ============================================================================


def _calculate_local_std_1d(data: np.ndarray, window: int) -> np.ndarray:
    """局所標準偏差（対称窓）— 無次元化の分母。GETTER One 由来。"""
    n = len(data)
    local_std = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        subset = data[start:end]
        if len(subset) > 0:
            local_std[i] = np.std(subset)
    return local_std


def _calculate_rho_t_1d(data: np.ndarray, window: int) -> np.ndarray:
    """1次元テンション密度 (ρT) — 過去窓の局所標準偏差。GETTER One 由来。"""
    n = len(data)
    rho_t = np.zeros(n)
    for i in range(n):
        start = max(0, i - window)
        end = i + 1
        subset = data[start:end]
        if len(subset) > 1:
            rho_t[i] = np.std(subset)
    return rho_t


# ============================================================================
# Data class
# ============================================================================


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


# ============================================================================
# Engine
# ============================================================================


class NNNUEngine:
    """
    Neural Network Non-Use: jump-triggered signed_mean causal discovery.

    Parameters
    ----------
    max_lag : int
        Maximum causal lag (provided by benchmark or user).
    local_std_window : int
        Λ³ local_std window (hint; adaptive may override).
    rho_t_window : int
        ρT window (hint; adaptive may override).
    delta_percentile : float
        Percentile for jump extraction (hint; adaptive may override).
    alpha : float
        Significance threshold for binary adjacency (BH-FDR q-value).
    min_jumps : int
        Minimum source jumps required to evaluate an edge.
    adaptive : bool
        Enable adaptive parameter tuning from data characteristics.
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

        # Runtime values (may be overridden by adaptive)
        self.local_std_window = local_std_window
        self.rho_t_window = rho_t_window
        self.delta_percentile = delta_percentile

    def fit(self, data: np.ndarray) -> NNNUResult:
        n_frames, n_dims = data.shape

        # --- Step 1: Λ³ event extraction (GETTER One lineage) ---
        events_pos, events_neg, rho_t, disp, local_std = (
            self._extract_lambda3_events(data, n_frames, n_dims)
        )
        n_disp = n_frames - 1

        # --- Step 1.5: Adaptive parameter tuning ---
        adaptive_params = None
        if self.adaptive:
            adaptive_params = self._compute_adaptive_parameters(
                data, events_pos, events_neg, rho_t, n_frames, n_dims,
            )
            self.local_std_window = adaptive_params["local_std_window"]
            self.rho_t_window = adaptive_params["rho_t_window"]
            self.delta_percentile = adaptive_params["delta_percentile"]

            # Re-extract with adaptive params
            events_pos, events_neg, rho_t, disp, local_std = (
                self._extract_lambda3_events(data, n_frames, n_dims)
            )

        # --- Step 2: Build jump frame/sign arrays ---
        jump_frames = {}
        jump_signs = {}
        jump_counts = np.zeros(n_dims, dtype=int)

        for d in range(n_dims):
            frames_pos = np.where(events_pos[:, d] > 0)[0]
            frames_neg = np.where(events_neg[:, d] > 0)[0]
            all_frames = np.concatenate([frames_pos, frames_neg])
            all_signs = np.concatenate([
                np.ones(len(frames_pos), dtype=int),
                -np.ones(len(frames_neg), dtype=int),
            ])
            order = np.argsort(all_frames)
            jump_frames[d] = all_frames[order]
            jump_signs[d] = all_signs[order]
            jump_counts[d] = len(all_frames)

        # --- Step 3: signed_mean × directional consistency scoring ---
        score_matrix = np.zeros((n_dims, n_dims))
        lag_matrix = np.zeros((n_dims, n_dims), dtype=int)
        sign_matrix = np.zeros((n_dims, n_dims))
        p_matrix = np.ones((n_dims, n_dims))
        consistency_matrix = np.zeros((n_dims, n_dims))

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
                best_consistency = 0.5

                for lag in range(1, self.max_lag + 1):
                    valid = frames + lag < n_disp
                    if np.sum(valid) < self.min_jumps:
                        continue

                    v_frames = frames[valid]
                    v_signs = signs[valid]
                    responses = disp[v_frames + lag, tgt]
                    signed_resp = responses * v_signs
                    signed_mean = np.mean(signed_resp)

                    # Directional consistency: how consistently same direction?
                    same_sign_rate = np.mean(signed_resp > 0)
                    # Adjusted: works for both positive and negative causation
                    consistency = max(same_sign_rate, 1 - same_sign_rate)
                    # Bonus: 0.5 → 0.0, 0.7 → 0.4, 0.9 → 0.8, 1.0 → 1.0
                    consistency_bonus = (consistency - 0.5) * 2.0

                    # Combined score: magnitude × directional consistency
                    combined = abs(signed_mean) * (1.0 + consistency_bonus)

                    # t-test
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

        raw_score_matrix = score_matrix.copy()

        # --- Step 4 & 5: Spurious filter (GETTER One conditional propagation) ---
        filtered_score, filtered_lag, filtered_sign = self._spurious_filter(
            score_matrix.copy(), lag_matrix.copy(), sign_matrix.copy(),
            events_pos, events_neg, n_dims,
        )

        # --- Step 6: Conditional scoring (causal-path-aware) ---
        _, cond_p = self._conditional_scoring(
            filtered_score, filtered_lag, p_matrix.copy(),
            disp, jump_frames, jump_signs, n_dims, n_disp,
        )

        # --- Step 6.5: BH-FDR correction ---
        q_matrix = self._bh_fdr(cond_p, n_dims)

        # --- Step 7: Suppress scoring (filter + BH-FDR → score discount) ---
        suppress_floor = 0.05
        out_score = score_matrix.copy()
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                if filtered_score[i, j] == 0 and score_matrix[i, j] > 0:
                    out_score[i, j] *= suppress_floor
                if q_matrix[i, j] >= self.alpha:
                    out_score[i, j] *= suppress_floor

        # --- Step 8: Binary adjacency ---
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

        return NNNUResult(
            score_matrix=out_score,
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

    # ------------------------------------------------------------------
    # Step 1: Λ³ Event Extraction (GETTER One lineage)
    # ------------------------------------------------------------------

    def _extract_lambda3_events(
        self,
        state_vectors: np.ndarray,
        n_frames: int,
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Λ³ event extraction — directly from GETTER One.

        Returns
        -------
        events_pos    : (n_frames-1, n_dims) positive ΔΛC events
        events_neg    : (n_frames-1, n_dims) negative ΔΛC events
        rho_t         : (n_frames, n_dims) tension density
        disp          : (n_frames-1, n_dims) dimensionless displacement
        local_std     : (n_frames, n_dims) local standard deviation
        """
        n_diff = n_frames - 1
        events_pos = np.zeros((n_diff, n_dims))
        events_neg = np.zeros((n_diff, n_dims))
        rho_t = np.zeros((n_frames, n_dims))
        disp = np.zeros((n_diff, n_dims))
        local_std_all = np.zeros((n_frames, n_dims))

        for d in range(n_dims):
            series = state_vectors[:, d]

            # diff → local_std で無次元化 → percentile → binary events
            diff = np.diff(series)
            lstd = _calculate_local_std_1d(series, self.local_std_window)
            lstd_diff = lstd[1:]
            score = np.abs(diff) / (lstd_diff + 1e-10)
            threshold = np.percentile(score, self.delta_percentile)

            jump_mask = score > threshold
            events_pos[:, d] = ((diff > 0) & jump_mask).astype(float)
            events_neg[:, d] = ((diff < 0) & jump_mask).astype(float)

            # Dimensionless displacement (for signed_mean)
            disp[:, d] = diff / (lstd_diff + 1e-10)

            # ρT
            rho_t[:, d] = _calculate_rho_t_1d(series, self.rho_t_window)

            # local_std
            local_std_all[:, d] = lstd

        return events_pos, events_neg, rho_t, disp, local_std_all

    # ------------------------------------------------------------------
    # Step 1.5: Adaptive Parameter Computation (GETTER One lineage)
    # ------------------------------------------------------------------

    def _compute_adaptive_parameters(
        self,
        state_vectors: np.ndarray,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        rho_t: np.ndarray,
        n_frames: int,
        n_dims: int,
    ) -> dict:
        """
        Data-driven adaptive parameter tuning — from GETTER One.

        Adjusts: delta_percentile, local_std_window, rho_t_window
        Based on: event_density, cofiring_rate, ρT variability, volatility
        """
        # Event density
        events_all = np.minimum(events_pos + events_neg, 1.0)
        event_density = float(np.mean(events_all))

        # Co-firing rate
        cofiring_rates = []
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                cofiring_rates.append(float(np.mean(events_all[:, i] * events_all[:, j])))
        mean_cofiring = float(np.mean(cofiring_rates)) if cofiring_rates else 0.0

        # ρT variability
        rho_t_means = np.mean(rho_t, axis=0)
        rho_t_overall = float(np.mean(rho_t_means))
        rho_t_cv = (
            float(np.std(rho_t_means) / (rho_t_overall + 1e-10))
            if rho_t_overall > 1e-10 else 0.0
        )

        # Temporal volatility
        temporal_changes = np.diff(state_vectors, axis=0)
        temporal_vol = float(np.mean(np.std(temporal_changes, axis=0)))
        global_std = float(np.std(state_vectors))
        vol_ratio = temporal_vol / (global_std + 1e-10)

        # === Adaptive delta_percentile ===
        pct = self.delta_percentile_hint
        if event_density > 0.15:
            pct = min(pct + 2.0, 97.0)  # Too many events → stricter
        elif event_density < 0.03:
            pct = max(pct - 3.0, 80.0)  # Too few events → relaxer
        if mean_cofiring > 0.02:
            pct = min(pct + 1.0, 97.0)  # High co-firing → stricter
        if n_dims > 10:
            pct = min(pct + 1.0, 97.0)  # High dim → stricter (more multiple testing)

        # === Adaptive windows ===
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

        params = {
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

        logger.info(
            f"   🔧 NNNU Adaptive: pct={pct:.1f}% "
            f"(hint={self.delta_percentile_hint}), "
            f"lstd_w={local_std_window}, rho_w={rho_t_window}"
        )

        return params

    # ------------------------------------------------------------------
    # Steps 4 & 5: Spurious Filter (GETTER One conditional propagation)
    # ------------------------------------------------------------------

    def _spurious_filter(
        self,
        score_matrix: np.ndarray,
        lag_matrix: np.ndarray,
        sign_matrix: np.ndarray,
        events_pos: np.ndarray,
        events_neg: np.ndarray,
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Spurious edge removal using GETTER One's conditional propagation.

        Pattern 1: Common Ancestor
          Z→A and Z→B exist → A→B is spurious if
          P(B|A, Z not fired) drops significantly

        Pattern 2: Mediator
          A→M→B exists → A→B is spurious if
          lag(A→M) + lag(M→B) ≈ lag(A→B)
        """
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

                # --- Common Ancestor (GETTER One style ±2 window) ---
                for z in range(n_dims):
                    if z == a or z == b:
                        continue
                    if score_matrix[z, a] <= score_matrix[a, b]:
                        continue
                    if score_matrix[z, b] <= score_matrix[a, b]:
                        continue

                    # Lag consistency
                    lag_za = int(lag_matrix[z, a])
                    lag_zb = int(lag_matrix[z, b])
                    if abs((lag_za + lag_ab) - lag_zb) > max(1, lag_zb // 3):
                        continue

                    # Conditional propagation (±2 frame window)
                    prob_with, prob_without = self._conditional_propagation(
                        events_all, a, b, z, lag_ab,
                    )

                    # Check sample count
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

                # --- Mediator ---
                for m in range(n_dims):
                    if m == a or m == b:
                        continue
                    if score_matrix[a, m] <= 0 or score_matrix[m, b] <= 0:
                        continue

                    mediated = int(lag_matrix[a, m]) + int(lag_matrix[m, b])
                    if abs(mediated - lag_ab) > max(2, lag_ab // 2):
                        continue

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
    def _conditional_propagation(
        events_all: np.ndarray,
        a: int, b: int, z: int, lag: int,
    ) -> tuple[float, float]:
        """
        Conditional propagation probability — from GETTER One.
        P(B(t+lag) | A(t), Z active/inactive within ±2 frames)
        """
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

        mask_with = (a_events > 0) & (z_active > 0)
        n_with = np.sum(mask_with)
        prob_with = float(np.sum(mask_with * b_events) / n_with) if n_with > 0 else 0.0

        mask_without = (a_events > 0) & (z_active == 0)
        n_without = np.sum(mask_without)
        prob_without = float(np.sum(mask_without * b_events) / n_without) if n_without > 0 else 0.0

        return prob_with, prob_without

    @staticmethod
    def _z_activity_mask(events_all: np.ndarray, z: int, length: int) -> np.ndarray:
        """Build Z activity mask with ±2 frame window."""
        z_active = np.zeros(length)
        for t in range(length):
            z_start = max(0, t - 2)
            z_end = min(len(events_all), t + 3)
            if np.any(events_all[z_start:z_end, z] > 0):
                z_active[t] = 1.0
        return z_active

    # ------------------------------------------------------------------
    # Step 6: Conditional scoring (causal-path-aware)
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
        Re-score edges by excluding frames where established
        downstream mediators also responded.
        """
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
                cond_mean = np.mean(signed_resp)

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

    # ------------------------------------------------------------------
    # BH-FDR correction
    # ------------------------------------------------------------------

    @staticmethod
    def _bh_fdr(p_matrix: np.ndarray, n_dims: int) -> np.ndarray:
        """Benjamini-Hochberg FDR correction."""
        pairs = []
        for i in range(n_dims):
            for j in range(n_dims):
                if i != j:
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


# ============================================================================
# Benchmark Adapter
# ============================================================================


class NNNUAdapter:
    """Benchmark adapter for NNNU."""
    method_name = "NNNU"

    def __init__(
        self,
        max_lag=5,
        delta_percentile=90.0,
        alpha=0.05,
        min_jumps=5,
        adaptive=True,
    ):
        self.kwargs = dict(
            max_lag=max_lag,
            delta_percentile=delta_percentile,
            alpha=alpha,
            min_jumps=min_jumps,
            adaptive=adaptive,
        )

    def fit(self, df, cfg=None):
        from FISJ.adapter import MethodOutput
        names = list(df.columns)
        data = df.values.astype(np.float64)
        result = NNNUEngine(**self.kwargs).fit(data)
        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=result.score_matrix,
            adjacency_bin=result.binary_matrix.astype(int),
            directed_support=True,
            lag_support=True,
            sign_support=True,
            lag_matrix=result.lag_matrix,
            sign_matrix=result.sign_matrix,
            meta={
                "total_jumps": result.total_jumps,
                "p_matrix": result.p_matrix,
                "q_matrix": result.q_matrix,
            },
        )
