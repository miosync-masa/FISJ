from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .main import NetworkAnalyzerCore, DimensionLink, NetworkResult

logger = logging.getLogger("getter_one.analysis.network_analyzer_core_v2")


# ============================================================================
# NOTE
# ----------------------------------------------------------------------------
# This file is designed as a new version layer on top of the user's existing
# NetworkAnalyzerCore implementation.
#
# Expected integration:
#   1) Keep the original DimensionLink / NetworkResult / CooperativeEventNetwork
#      dataclasses and NetworkAnalyzerCore class in scope.
#   2) Add this file in the same module/package, or paste it below the original
#      implementation.
#   3) Instantiate NetworkAnalyzerCoreV2 instead of NetworkAnalyzerCore.
#
# Main goals:
#   - fully domain-agnostic regime detection
#   - contiguous regime segmentation (no time-order destruction)
#   - loose candidate detection + inverse-problem refinement
#   - CauseMe-oriented precision recovery after high-recall edge harvesting
# ============================================================================


# ============================================================================
# New Data Classes
# ============================================================================


@dataclass
class GenericRegimeConfig:
    """Domain-agnostic regime detection configuration."""

    n_regimes: int = 3
    min_segment_length: int = 40
    smooth_window: int = 5
    feature_windows: tuple[int, ...] = (5, 20, 50)
    random_state: int = 42
    n_init: int = 8
    max_iter: int = 100
    zscore_features: bool = True
    allow_single_regime_fallback: bool = True


@dataclass
class InverseRefinementConfig:
    """Configuration for inverse-problem refinement of candidate causal edges."""

    enabled: bool = True
    kernel_max_lag: int = 8
    autoregressive_lag: int = 1
    ridge_alpha: float = 1e-2
    smooth_alpha: float = 1e-2
    min_delta_mse: float = 1e-4
    min_confidence: float = 0.05
    validation_fraction: float = 0.25
    asymmetry_weight: float = 0.25
    refit_after_removal: bool = True
    benefit_percentile: float = 40.0


@dataclass
class RefinedEdgeEvidence:
    """Evidence summary for one refined causal edge."""

    from_dim: int
    to_dim: int
    chosen_lag: int
    kernel_norm: float
    delta_mse_forward: float
    delta_mse_backward: float
    asymmetry: float
    confidence: float


@dataclass
class RegimeSegment:
    """One contiguous time segment assigned to a single regime."""

    regime_id: int
    start: int
    end: int  # exclusive
    n_frames: int


@dataclass
class RegimeAwareNetworkResult:
    """Top-level result for regime-aware + inverse-refined analysis."""

    base_result: Any
    final_result: Any
    regime_labels: np.ndarray
    regime_segments: list[RegimeSegment] = field(default_factory=list)
    regime_results: dict[int, list[Any]] = field(default_factory=dict)
    refined_evidence: list[RefinedEdgeEvidence] = field(default_factory=list)
    edge_frequency: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_mean_confidence: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_mean_lag: dict[tuple[int, int], float] = field(default_factory=dict)
    regime_transition_matrix: np.ndarray | None = None


# ============================================================================
# Generic Regime Detector
# ============================================================================


class GenericRegimeDetector:
    """
    Fully domain-agnostic regime detector using only multivariate time-series.

    Design principles:
      - no finance-specific labels or assumptions
      - use generic dynamical features extracted from state_vectors
      - keep contiguous segments only (never squeeze disjoint timestamps together)
    """

    def __init__(self, config: GenericRegimeConfig | None = None):
        self.config = config or GenericRegimeConfig()

    def detect(self, state_vectors: np.ndarray) -> np.ndarray:
        if state_vectors.ndim != 2:
            raise ValueError("state_vectors must have shape (n_frames, n_dims)")

        n_frames, n_dims = state_vectors.shape
        if n_frames < max(12, self.config.min_segment_length):
            if self.config.allow_single_regime_fallback:
                return np.zeros(n_frames, dtype=int)
            raise ValueError("Not enough frames for regime detection")

        X = self._extract_features(state_vectors)
        if self.config.zscore_features:
            X = self._zscore_matrix(X)

        k = int(np.clip(self.config.n_regimes, 1, max(1, n_frames // self.config.min_segment_length)))
        if k <= 1:
            return np.zeros(n_frames, dtype=int)

        labels = self._kmeans(X, k)
        labels = self._smooth_labels(labels, self.config.smooth_window)
        return labels.astype(int)

    def build_segments(self, labels: np.ndarray) -> list[RegimeSegment]:
        if len(labels) == 0:
            return []

        segments: list[RegimeSegment] = []
        start = 0
        current = int(labels[0])

        for t in range(1, len(labels)):
            if int(labels[t]) != current:
                end = t
                if end - start >= self.config.min_segment_length:
                    segments.append(
                        RegimeSegment(
                            regime_id=current,
                            start=start,
                            end=end,
                            n_frames=end - start,
                        )
                    )
                start = t
                current = int(labels[t])

        end = len(labels)
        if end - start >= self.config.min_segment_length:
            segments.append(
                RegimeSegment(
                    regime_id=current,
                    start=start,
                    end=end,
                    n_frames=end - start,
                )
            )

        return segments

    def transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        if len(labels) == 0:
            return np.zeros((0, 0))
        n_regimes = int(np.max(labels)) + 1
        mat = np.zeros((n_regimes, n_regimes), dtype=float)
        for i in range(len(labels) - 1):
            a = int(labels[i])
            b = int(labels[i + 1])
            mat[a, b] += 1.0
        row_sums = mat.sum(axis=1, keepdims=True)
        return mat / (row_sums + 1e-12)

    def _extract_features(self, state_vectors: np.ndarray) -> np.ndarray:
        n_frames, n_dims = state_vectors.shape
        diffs = np.diff(state_vectors, axis=0, prepend=state_vectors[[0]])

        # Per-frame generic features
        mean_abs = np.mean(np.abs(state_vectors), axis=1)
        std_abs = np.std(state_vectors, axis=1)
        mean_step = np.mean(np.abs(diffs), axis=1)
        std_step = np.std(diffs, axis=1)
        energy = np.mean(state_vectors ** 2, axis=1)
        step_energy = np.mean(diffs ** 2, axis=1)
        dim_coherence = self._rolling_cross_dim_coherence(state_vectors, window=20)
        low_freq_ratio = self._rolling_low_frequency_ratio(state_vectors)

        feature_list = [
            mean_abs,
            std_abs,
            mean_step,
            std_step,
            energy,
            step_energy,
            dim_coherence,
            low_freq_ratio,
        ]

        for w in self.config.feature_windows:
            feature_list.append(self._rolling_mean(mean_step, w))
            feature_list.append(self._rolling_std(mean_step, w))
            feature_list.append(self._rolling_mean(energy, w))
            feature_list.append(self._rolling_std(energy, w))

        X = np.column_stack(feature_list)
        X[~np.isfinite(X)] = 0.0
        return X

    @staticmethod
    def _zscore_matrix(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        return (X - mu) / sigma

    @staticmethod
    def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - window + 1)
            out[i] = np.mean(x[start : i + 1])
        return out

    @staticmethod
    def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            start = max(0, i - window + 1)
            out[i] = np.std(x[start : i + 1])
        return out

    def _rolling_cross_dim_coherence(self, state_vectors: np.ndarray, window: int) -> np.ndarray:
        n_frames, n_dims = state_vectors.shape
        if n_dims < 2:
            return np.zeros(n_frames, dtype=float)

        out = np.zeros(n_frames, dtype=float)
        for t in range(n_frames):
            start = max(0, t - window + 1)
            block = state_vectors[start : t + 1]
            if len(block) < 3:
                continue
            corr = np.corrcoef(block.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            triu = corr[np.triu_indices(n_dims, k=1)]
            out[t] = np.mean(np.abs(triu)) if len(triu) > 0 else 0.0
        return out

    def _rolling_low_frequency_ratio(self, state_vectors: np.ndarray) -> np.ndarray:
        n_frames, n_dims = state_vectors.shape
        window = max(16, min(64, n_frames // 4 if n_frames >= 20 else n_frames))
        out = np.zeros(n_frames, dtype=float)
        for t in range(n_frames):
            start = max(0, t - window + 1)
            block = state_vectors[start : t + 1]
            if len(block) < 8:
                continue
            fft_mag = np.abs(np.fft.rfft(block, axis=0))
            cutoff = max(1, fft_mag.shape[0] // 4)
            low = np.sum(fft_mag[:cutoff])
            total = np.sum(fft_mag) + 1e-12
            out[t] = low / total
        return out

    def _kmeans(self, X: np.ndarray, k: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_state)
        best_labels = np.zeros(len(X), dtype=int)
        best_inertia = np.inf

        for _ in range(self.config.n_init):
            idx = rng.choice(len(X), size=k, replace=False)
            centers = X[idx].copy()

            for _ in range(self.config.max_iter):
                d2 = self._squared_distances(X, centers)
                labels = np.argmin(d2, axis=1)

                new_centers = centers.copy()
                for j in range(k):
                    mask = labels == j
                    if np.any(mask):
                        new_centers[j] = np.mean(X[mask], axis=0)
                    else:
                        new_centers[j] = X[rng.integers(0, len(X))]

                shift = np.linalg.norm(new_centers - centers)
                centers = new_centers
                if shift < 1e-8:
                    break

            inertia = np.sum(np.min(self._squared_distances(X, centers), axis=1))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

        return best_labels

    @staticmethod
    def _squared_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)

    def _smooth_labels(self, labels: np.ndarray, window: int) -> np.ndarray:
        if window <= 1 or len(labels) < 3:
            return labels
        half = window // 2
        out = labels.copy()
        for i in range(len(labels)):
            start = max(0, i - half)
            end = min(len(labels), i + half + 1)
            vals, counts = np.unique(labels[start:end], return_counts=True)
            out[i] = vals[np.argmax(counts)]
        return out


# ============================================================================
# Inverse-Problem Causal Refiner
# ============================================================================


class InverseCausalRefiner:
    """
    Refine a high-recall candidate graph by solving a node-wise inverse problem.

    Strategy:
      1) harvest candidate edges with low-ish threshold upstream
      2) for each target node, fit a lagged linear reconstruction model
      3) remove one parent at a time and measure validation error increase
      4) keep only edges that materially improve future reconstruction
      5) optionally compare forward vs backward time for directional asymmetry

    This is intentionally lightweight and domain-agnostic.
    """

    def __init__(self, config: InverseRefinementConfig | None = None):
        self.config = config or InverseRefinementConfig()

    def refine(
        self,
        state_vectors: np.ndarray,
        candidate_links: list[Any],
        dimension_names: list[str],
    ) -> tuple[list[Any], list[RefinedEdgeEvidence]]:
        if not self.config.enabled or len(candidate_links) == 0:
            return candidate_links, []

        n_frames, n_dims = state_vectors.shape
        by_target: dict[int, list[Any]] = {}
        for link in candidate_links:
            by_target.setdefault(link.to_dim, []).append(link)

        kept_links: list[Any] = []
        evidence_list: list[RefinedEdgeEvidence] = []

        for target, incoming in by_target.items():
            try:
                target_kept, target_evidence = self._refine_one_target(
                    state_vectors=state_vectors,
                    target=target,
                    incoming_links=incoming,
                    dimension_names=dimension_names,
                )
                kept_links.extend(target_kept)
                evidence_list.extend(target_evidence)
            except Exception as exc:  # pragma: no cover - safe fallback path
                logger.warning(
                    "Inverse refinement failed for target %s: %s. Falling back to original links.",
                    target,
                    exc,
                )
                kept_links.extend(incoming)

        return kept_links, evidence_list

    def _refine_one_target(
        self,
        state_vectors: np.ndarray,
        target: int,
        incoming_links: list[Any],
        dimension_names: list[str],
    ) -> tuple[list[Any], list[RefinedEdgeEvidence]]:
        max_lag = self.config.kernel_max_lag
        ar_lag = self.config.autoregressive_lag

        parents = sorted({link.from_dim for link in incoming_links})
        if len(parents) == 0:
            return [], []

        y, X, groups, group_lags = self._build_design_matrix(
            state_vectors=state_vectors,
            target=target,
            parents=parents,
            kernel_max_lag=max_lag,
            autoregressive_lag=ar_lag,
        )
        if len(y) < 20 or X.shape[1] == 0:
            return incoming_links, []

        split = int(len(y) * (1.0 - self.config.validation_fraction))
        split = int(np.clip(split, 8, len(y) - 4))
        y_tr, y_va = y[:split], y[split:]
        X_tr, X_va = X[:split], X[split:]

        w_full = self._fit_penalized(X_tr, y_tr, groups)
        pred_full = X_va @ w_full
        mse_full = self._mse(y_va, pred_full)

        backward_state = state_vectors[::-1].copy()
        y_b, X_b, groups_b, _ = self._build_design_matrix(
            state_vectors=backward_state,
            target=target,
            parents=parents,
            kernel_max_lag=max_lag,
            autoregressive_lag=ar_lag,
        )
        use_backward = len(y_b) >= 20 and X_b.shape[1] > 0
        if use_backward:
            split_b = int(len(y_b) * (1.0 - self.config.validation_fraction))
            split_b = int(np.clip(split_b, 8, len(y_b) - 4))
            yb_tr, yb_va = y_b[:split_b], y_b[split_b:]
            Xb_tr, Xb_va = X_b[:split_b], X_b[split_b:]
            w_full_b = self._fit_penalized(Xb_tr, yb_tr, groups_b)
            mse_full_b = self._mse(yb_va, Xb_va @ w_full_b)
        else:
            mse_full_b = mse_full
            w_full_b = None

        evidence: list[RefinedEdgeEvidence] = []
        retained_edges: list[Any] = []
        delta_values: list[float] = []

        parent_to_cols = self._group_to_parent_columns(groups)
        parent_to_cols_b = self._group_to_parent_columns(groups_b) if use_backward else {}

        parent_kernel_norm: dict[int, float] = {}
        parent_best_lag: dict[int, int] = {}
        for p in parents:
            cols = parent_to_cols.get(p, [])
            if not cols:
                parent_kernel_norm[p] = 0.0
                parent_best_lag[p] = 1
                continue
            kernel = w_full[cols]
            parent_kernel_norm[p] = float(np.linalg.norm(kernel))
            best_idx = int(np.argmax(np.abs(kernel)))
            parent_best_lag[p] = int(group_lags[p][best_idx])

        for p in parents:
            cols = parent_to_cols.get(p, [])
            if not cols:
                continue

            if self.config.refit_after_removal:
                keep_cols = [c for c in range(X_tr.shape[1]) if c not in cols]
                X_tr_r = X_tr[:, keep_cols]
                X_va_r = X_va[:, keep_cols]
                groups_r = self._remap_groups_after_column_drop(groups, keep_cols)
                if X_tr_r.shape[1] == 0:
                    mse_removed = np.var(y_va)
                else:
                    w_removed = self._fit_penalized(X_tr_r, y_tr, groups_r)
                    mse_removed = self._mse(y_va, X_va_r @ w_removed)
            else:
                w_masked = w_full.copy()
                w_masked[cols] = 0.0
                mse_removed = self._mse(y_va, X_va @ w_masked)

            delta_forward = float(mse_removed - mse_full)
            delta_values.append(delta_forward)

            if use_backward:
                cols_b = parent_to_cols_b.get(p, [])
                if cols_b:
                    if self.config.refit_after_removal:
                        keep_cols_b = [c for c in range(Xb_tr.shape[1]) if c not in cols_b]
                        Xb_tr_r = Xb_tr[:, keep_cols_b]
                        Xb_va_r = Xb_va[:, keep_cols_b]
                        groups_b_r = self._remap_groups_after_column_drop(groups_b, keep_cols_b)
                        if Xb_tr_r.shape[1] == 0:
                            mse_removed_b = np.var(yb_va)
                        else:
                            wb_removed = self._fit_penalized(Xb_tr_r, yb_tr, groups_b_r)
                            mse_removed_b = self._mse(yb_va, Xb_va_r @ wb_removed)
                    else:
                        wb_masked = w_full_b.copy()
                        wb_masked[cols_b] = 0.0
                        mse_removed_b = self._mse(yb_va, Xb_va @ wb_masked)
                    delta_backward = float(mse_removed_b - mse_full_b)
                else:
                    delta_backward = 0.0
            else:
                delta_backward = 0.0

            asymmetry = max(0.0, delta_forward - delta_backward)
            confidence = max(0.0, delta_forward) + self.config.asymmetry_weight * asymmetry

            evidence.append(
                RefinedEdgeEvidence(
                    from_dim=p,
                    to_dim=target,
                    chosen_lag=parent_best_lag[p],
                    kernel_norm=parent_kernel_norm[p],
                    delta_mse_forward=delta_forward,
                    delta_mse_backward=delta_backward,
                    asymmetry=asymmetry,
                    confidence=confidence,
                )
            )

        if not evidence:
            return incoming_links, []

        confidence_values = np.array([ev.confidence for ev in evidence], dtype=float)
        cutoff = np.percentile(confidence_values, self.config.benefit_percentile)
        cutoff = max(cutoff, self.config.min_confidence)

        original_by_parent = {link.from_dim: link for link in incoming_links}
        for ev in evidence:
            if ev.delta_mse_forward < self.config.min_delta_mse:
                continue
            if ev.confidence < cutoff:
                continue

            base_link = original_by_parent[ev.from_dim]
            new_link = type(base_link)(
                from_dim=base_link.from_dim,
                to_dim=base_link.to_dim,
                from_name=base_link.from_name,
                to_name=base_link.to_name,
                link_type=base_link.link_type,
                strength=max(base_link.strength, ev.confidence),
                correlation=base_link.correlation,
                lag=ev.chosen_lag,
            )
            retained_edges.append(new_link)

        return retained_edges, evidence

    def _build_design_matrix(
        self,
        state_vectors: np.ndarray,
        target: int,
        parents: list[int],
        kernel_max_lag: int,
        autoregressive_lag: int,
    ) -> tuple[np.ndarray, np.ndarray, dict[int, list[int]], dict[int, list[int]]]:
        n_frames, n_dims = state_vectors.shape
        max_lag = max(kernel_max_lag, autoregressive_lag)
        if n_frames <= max_lag + 2:
            return np.zeros(0), np.zeros((0, 0)), {}, {}

        rows = []
        y = []
        groups: dict[int, list[int]] = {}
        group_lags: dict[int, list[int]] = {}

        # Column layout: first optional target AR block, then one kernel block per parent.
        col = 0
        if autoregressive_lag > 0:
            groups[-1] = list(range(col, col + autoregressive_lag))
            group_lags[-1] = list(range(1, autoregressive_lag + 1))
            col += autoregressive_lag
        for p in parents:
            groups[p] = list(range(col, col + kernel_max_lag))
            group_lags[p] = list(range(1, kernel_max_lag + 1))
            col += kernel_max_lag

        for t in range(max_lag, n_frames):
            row = []
            if autoregressive_lag > 0:
                for lag in range(1, autoregressive_lag + 1):
                    row.append(state_vectors[t - lag, target])
            for p in parents:
                for lag in range(1, kernel_max_lag + 1):
                    row.append(state_vectors[t - lag, p])
            rows.append(row)
            y.append(state_vectors[t, target])

        X = np.asarray(rows, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        X = self._zscore_design(X)
        y_arr = self._zscore_vector(y_arr)
        return y_arr, X, groups, group_lags

    def _fit_penalized(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: dict[int, list[int]],
    ) -> np.ndarray:
        p = X.shape[1]
        if p == 0:
            return np.zeros(0)

        xtx = X.T @ X
        xty = X.T @ y

        reg = self.config.ridge_alpha * np.eye(p)

        # Smoothness penalty across lags inside each parent kernel.
        for gid, cols in groups.items():
            if gid == -1:
                continue
            if len(cols) < 2:
                continue
            for a, b in zip(cols[:-1], cols[1:]):
                reg[a, a] += self.config.smooth_alpha
                reg[b, b] += self.config.smooth_alpha
                reg[a, b] -= self.config.smooth_alpha
                reg[b, a] -= self.config.smooth_alpha

        try:
            w = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xtx + reg) @ xty
        return w

    @staticmethod
    def _group_to_parent_columns(groups: dict[int, list[int]]) -> dict[int, list[int]]:
        return {gid: cols for gid, cols in groups.items() if gid != -1}

    @staticmethod
    def _remap_groups_after_column_drop(
        groups: dict[int, list[int]],
        keep_cols: list[int],
    ) -> dict[int, list[int]]:
        mapping = {old: new for new, old in enumerate(keep_cols)}
        out: dict[int, list[int]] = {}
        for gid, cols in groups.items():
            new_cols = [mapping[c] for c in cols if c in mapping]
            if new_cols:
                out[gid] = new_cols
        return out

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _zscore_design(X: np.ndarray) -> np.ndarray:
        X = X.copy()
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        return (X - mu) / sigma

    @staticmethod
    def _zscore_vector(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x)
        sigma = np.std(x)
        if sigma < 1e-12:
            sigma = 1.0
        return (x - mu) / sigma


# ============================================================================
# Aggregation Helpers
# ============================================================================


class _AggregatedLink:
    """Internal helper used before converting back to user-level DimensionLink."""

    __slots__ = ("from_dim", "to_dim", "strengths", "correlations", "lags")

    def __init__(self, from_dim: int, to_dim: int):
        self.from_dim = from_dim
        self.to_dim = to_dim
        self.strengths: list[float] = []
        self.correlations: list[float] = []
        self.lags: list[int] = []

    def add(self, strength: float, correlation: float, lag: int = 0) -> None:
        self.strengths.append(float(strength))
        self.correlations.append(float(correlation))
        self.lags.append(int(lag))


# ============================================================================
# New Version: Regime-aware + Inverse-refined analyzer
# ============================================================================


class NetworkAnalyzerCoreV2(NetworkAnalyzerCore):
    """
    New version targeting noisy, non-stationary benchmark settings such as CauseMe.

    Pipeline
    --------
    1. Run the original analyzer on the full sequence with intentionally high recall.
    2. Detect generic regimes and extract contiguous segments.
    3. Run the original analyzer on each sufficiently long segment.
    4. Aggregate causal evidence across segments.
    5. Refine the candidate causal graph with a node-wise inverse problem.
    """

    def __init__(
        self,
        *args,
        regime_config: GenericRegimeConfig | None = None,
        inverse_config: InverseRefinementConfig | None = None,
        candidate_threshold_scale: float = 0.75,
        min_segment_causal_frequency: float = 0.20,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.regime_detector = GenericRegimeDetector(regime_config)
        self.inverse_refiner = InverseCausalRefiner(inverse_config)
        self.candidate_threshold_scale = candidate_threshold_scale
        self.min_segment_causal_frequency = min_segment_causal_frequency

    def analyze_regime_aware(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str] | None = None,
        window: int | None = None,
    ) -> RegimeAwareNetworkResult:
        n_frames, n_dims = state_vectors.shape
        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        # ------------------------------------------------------------------
        # Step 1: high-recall candidate detection on full data
        # ------------------------------------------------------------------
        original_sync_threshold = self.sync_threshold
        original_causal_threshold = self.causal_threshold
        self.sync_threshold = max(0.05, original_sync_threshold * self.candidate_threshold_scale)
        self.causal_threshold = max(0.05, original_causal_threshold * self.candidate_threshold_scale)
        try:
            base_result = super().analyze(state_vectors, dimension_names, window)
        finally:
            self.sync_threshold = original_sync_threshold
            self.causal_threshold = original_causal_threshold

        # ------------------------------------------------------------------
        # Step 2: generic regimes + contiguous segments
        # ------------------------------------------------------------------
        regime_labels = self.regime_detector.detect(state_vectors)
        segments = self.regime_detector.build_segments(regime_labels)
        transition_matrix = self.regime_detector.transition_matrix(regime_labels)

        # ------------------------------------------------------------------
        # Step 3: segment-wise analysis
        # ------------------------------------------------------------------
        regime_results: dict[int, list[Any]] = {}
        segment_causal_counts: dict[tuple[int, int], int] = {}
        segment_sync_counts: dict[tuple[int, int], int] = {}
        agg_causal: dict[tuple[int, int], _AggregatedLink] = {}
        agg_sync: dict[tuple[int, int], _AggregatedLink] = {}

        for seg in segments:
            seg_data = state_vectors[seg.start : seg.end]
            if len(seg_data) < max(12, self.regime_detector.config.min_segment_length):
                continue
            seg_result = super().analyze(seg_data, dimension_names, window=len(seg_data))
            regime_results.setdefault(seg.regime_id, []).append(seg_result)

            seen_causal = set()
            for link in seg_result.causal_network:
                key = (link.from_dim, link.to_dim)
                agg_causal.setdefault(key, _AggregatedLink(*key)).add(
                    strength=link.strength,
                    correlation=link.correlation,
                    lag=link.lag,
                )
                seen_causal.add(key)
            for key in seen_causal:
                segment_causal_counts[key] = segment_causal_counts.get(key, 0) + 1

            seen_sync = set()
            for link in seg_result.sync_network:
                a, b = sorted((link.from_dim, link.to_dim))
                key = (a, b)
                agg_sync.setdefault(key, _AggregatedLink(a, b)).add(
                    strength=link.strength,
                    correlation=link.correlation,
                    lag=0,
                )
                seen_sync.add(key)
            for key in seen_sync:
                segment_sync_counts[key] = segment_sync_counts.get(key, 0) + 1

        n_segments = max(1, len(segments))

        # ------------------------------------------------------------------
        # Step 4: aggregate candidate edges from segment evidence + base result
        # ------------------------------------------------------------------
        candidate_causal_links = self._aggregate_candidate_causal_links(
            agg_causal=agg_causal,
            segment_counts=segment_causal_counts,
            n_segments=n_segments,
            base_result=base_result,
            dimension_names=dimension_names,
        )
        candidate_sync_links = self._aggregate_candidate_sync_links(
            agg_sync=agg_sync,
            segment_counts=segment_sync_counts,
            n_segments=n_segments,
            base_result=base_result,
            dimension_names=dimension_names,
        )

        # ------------------------------------------------------------------
        # Step 5: inverse-problem refinement for precision recovery
        # ------------------------------------------------------------------
        refined_links, evidence = self.inverse_refiner.refine(
            state_vectors=state_vectors,
            candidate_links=candidate_causal_links,
            dimension_names=dimension_names,
        )

        final_result = self._build_final_network_result(
            base_result=base_result,
            sync_links=candidate_sync_links,
            causal_links=refined_links,
            dimension_names=dimension_names,
            state_vectors=state_vectors,
        )

        edge_frequency = {
            key: segment_causal_counts.get(key, 0) / n_segments for key in agg_causal.keys()
        }
        edge_mean_confidence = {
            (ev.from_dim, ev.to_dim): ev.confidence for ev in evidence
        }
        edge_mean_lag = {
            key: float(np.mean(agg_causal[key].lags)) if agg_causal[key].lags else 0.0
            for key in agg_causal.keys()
        }

        return RegimeAwareNetworkResult(
            base_result=base_result,
            final_result=final_result,
            regime_labels=regime_labels,
            regime_segments=segments,
            regime_results=regime_results,
            refined_evidence=evidence,
            edge_frequency=edge_frequency,
            edge_mean_confidence=edge_mean_confidence,
            edge_mean_lag=edge_mean_lag,
            regime_transition_matrix=transition_matrix,
        )

    def _aggregate_candidate_causal_links(
        self,
        agg_causal: dict[tuple[int, int], _AggregatedLink],
        segment_counts: dict[tuple[int, int], int],
        n_segments: int,
        base_result: Any,
        dimension_names: list[str],
    ) -> list[Any]:
        out: dict[tuple[int, int], Any] = {}

        # Segment-supported links
        for key, bucket in agg_causal.items():
            freq = segment_counts.get(key, 0) / n_segments
            if freq < self.min_segment_causal_frequency:
                continue
            i, j = key
            out[key] = DimensionLink(
                from_dim=i,
                to_dim=j,
                from_name=dimension_names[i],
                to_name=dimension_names[j],
                link_type="causal",
                strength=float(np.mean(bucket.strengths) * (0.5 + 0.5 * freq)),
                correlation=float(np.mean(bucket.correlations)),
                lag=max(1, int(round(np.mean(bucket.lags)))) if bucket.lags else 1,
            )

        # Full-data candidates are also kept as weak proposals
        for link in base_result.causal_network:
            key = (link.from_dim, link.to_dim)
            if key not in out:
                out[key] = DimensionLink(
                    from_dim=link.from_dim,
                    to_dim=link.to_dim,
                    from_name=link.from_name,
                    to_name=link.to_name,
                    link_type="causal",
                    strength=max(0.05, 0.8 * link.strength),
                    correlation=link.correlation,
                    lag=link.lag,
                )

        return list(out.values())

    def _aggregate_candidate_sync_links(
        self,
        agg_sync: dict[tuple[int, int], _AggregatedLink],
        segment_counts: dict[tuple[int, int], int],
        n_segments: int,
        base_result: Any,
        dimension_names: list[str],
    ) -> list[Any]:
        out: dict[tuple[int, int], Any] = {}

        for key, bucket in agg_sync.items():
            freq = segment_counts.get(key, 0) / n_segments
            if freq < self.min_segment_causal_frequency:
                continue
            i, j = key
            out[key] = DimensionLink(
                from_dim=i,
                to_dim=j,
                from_name=dimension_names[i],
                to_name=dimension_names[j],
                link_type="sync",
                strength=float(np.mean(bucket.strengths) * (0.5 + 0.5 * freq)),
                correlation=float(np.mean(bucket.correlations)),
                lag=0,
            )

        for link in base_result.sync_network:
            key = tuple(sorted((link.from_dim, link.to_dim)))
            if key not in out:
                i, j = key
                out[key] = DimensionLink(
                    from_dim=i,
                    to_dim=j,
                    from_name=dimension_names[i],
                    to_name=dimension_names[j],
                    link_type="sync",
                    strength=link.strength,
                    correlation=link.correlation,
                    lag=0,
                )

        return list(out.values())

    def _build_final_network_result(
        self,
        base_result: Any,
        sync_links: list[Any],
        causal_links: list[Any],
        dimension_names: list[str],
        state_vectors: np.ndarray,
    ) -> Any:
        n_dims = state_vectors.shape[1]
        pattern = self._identify_pattern(sync_links, causal_links)
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)
        drivers, followers = self._identify_causal_structure(causal_links, n_dims)

        sync_matrix = np.zeros((n_dims, n_dims), dtype=float)
        causal_matrix = np.zeros((n_dims, n_dims), dtype=float)
        lag_matrix = np.zeros((n_dims, n_dims), dtype=int)

        for link in sync_links:
            i, j = link.from_dim, link.to_dim
            sync_matrix[i, j] = link.correlation
            sync_matrix[j, i] = link.correlation

        for link in causal_links:
            i, j = link.from_dim, link.to_dim
            causal_matrix[i, j] = link.strength
            causal_matrix[j, i] = link.strength
            lag_matrix[i, j] = link.lag
            lag_matrix[j, i] = -link.lag

        return NetworkResult(
            sync_network=sync_links,
            causal_network=causal_links,
            sync_matrix=sync_matrix,
            causal_matrix=causal_matrix,
            causal_lag_matrix=lag_matrix,
            pattern=pattern,
            hub_dimensions=hub_dims,
            hub_names=[dimension_names[d] for d in hub_dims],
            causal_drivers=drivers,
            causal_followers=followers,
            driver_names=[dimension_names[d] for d in drivers],
            follower_names=[dimension_names[d] for d in followers],
            n_dims=n_dims,
            n_sync_links=len(sync_links),
            n_causal_links=len(causal_links),
            dimension_names=dimension_names,
            adaptive_params=getattr(base_result, "adaptive_params", None),
        )


# ============================================================================
# Convenience usage sketch
# ============================================================================
#
# analyzer = NetworkAnalyzerCoreV2(
#     sync_threshold=0.45,
#     causal_threshold=0.30,
#     max_lag=12,
#     adaptive=True,
#     regime_config=GenericRegimeConfig(
#         n_regimes=3,
#         min_segment_length=40,
#     ),
#     inverse_config=InverseRefinementConfig(
#         enabled=True,
#         kernel_max_lag=8,
#         ridge_alpha=1e-2,
#         smooth_alpha=1e-2,
#         min_delta_mse=1e-4,
#         min_confidence=0.05,
#     ),
# )
# result_v2 = analyzer.analyze_regime_aware(state_vectors, dimension_names)
# final_network = result_v2.final_result
# evidence = result_v2.refined_evidence
#
# For CauseMe-style evaluation, final_network.causal_network is the key output.
# Use evidence to rank edges by confidence when sweeping thresholds for ROC/AUC.
# ============================================================================
