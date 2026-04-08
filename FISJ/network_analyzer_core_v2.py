"""
Network Analyzer Core V2 - Regime-Aware Analysis
=================================================
Built by Masamichi & Tamaki

A regime-aware extension of NetworkAnalyzerCore that:
  1. Detects generic dynamical regimes via contiguous segmentation
  2. Runs segment-wise causal analysis
  3. Exports segment frequency matrix for downstream fusion

Inverse-problem causal refinement is handled by InverseCausalEngine (standalone).
Score fusion is handled by score_fusion.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .main import NetworkAnalyzerCore, DimensionLink, NetworkResult

logger = logging.getLogger("fisj.network_analyzer_core_v2")


# ============================================================================
# Data Classes
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
class RegimeSegment:
    """One contiguous time segment assigned to a single regime."""

    regime_id: int
    start: int
    end: int  # exclusive
    n_frames: int


@dataclass
class RegimeAwareResult:
    """Result of regime-aware network analysis."""

    # Full-data analysis result
    base_result: NetworkResult

    # Regime info
    regime_labels: np.ndarray
    regime_segments: list[RegimeSegment] = field(default_factory=list)
    regime_results: dict[int, list[NetworkResult]] = field(default_factory=dict)
    regime_transition_matrix: np.ndarray | None = None

    # Segment frequency: how often each directed edge appears across segments
    # Shape: (n_dims, n_dims), values in [0, 1]
    causal_frequency_matrix: np.ndarray | None = None
    sync_frequency_matrix: np.ndarray | None = None

    # Per-edge aggregated statistics from segments
    edge_mean_strength: dict[tuple[int, int], float] = field(default_factory=dict)
    edge_mean_lag: dict[tuple[int, int], float] = field(default_factory=dict)

    n_segments: int = 0


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

        k = int(np.clip(
            self.config.n_regimes, 1,
            max(1, n_frames // self.config.min_segment_length),
        ))
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
                    segments.append(RegimeSegment(
                        regime_id=current, start=start, end=end, n_frames=end - start,
                    ))
                start = t
                current = int(labels[t])

        end = len(labels)
        if end - start >= self.config.min_segment_length:
            segments.append(RegimeSegment(
                regime_id=current, start=start, end=end, n_frames=end - start,
            ))
        return segments

    def transition_matrix(self, labels: np.ndarray) -> np.ndarray:
        if len(labels) == 0:
            return np.zeros((0, 0))
        n_regimes = int(np.max(labels)) + 1
        mat = np.zeros((n_regimes, n_regimes), dtype=float)
        for i in range(len(labels) - 1):
            mat[int(labels[i]), int(labels[i + 1])] += 1.0
        row_sums = mat.sum(axis=1, keepdims=True)
        return mat / (row_sums + 1e-12)

    # --- Feature extraction ---

    def _extract_features(self, state_vectors: np.ndarray) -> np.ndarray:
        n_frames, n_dims = state_vectors.shape
        diffs = np.diff(state_vectors, axis=0, prepend=state_vectors[[0]])

        mean_abs = np.mean(np.abs(state_vectors), axis=1)
        std_abs = np.std(state_vectors, axis=1)
        mean_step = np.mean(np.abs(diffs), axis=1)
        std_step = np.std(diffs, axis=1)
        energy = np.mean(state_vectors ** 2, axis=1)
        step_energy = np.mean(diffs ** 2, axis=1)
        dim_coherence = self._rolling_cross_dim_coherence(state_vectors, window=20)
        low_freq_ratio = self._rolling_low_frequency_ratio(state_vectors)

        feature_list = [
            mean_abs, std_abs, mean_step, std_step,
            energy, step_energy, dim_coherence, low_freq_ratio,
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
        mu = np.mean(X, axis=0)
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-12] = 1.0
        return (X - mu) / sigma

    @staticmethod
    def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            out[i] = np.mean(x[max(0, i - window + 1): i + 1])
        return out

    @staticmethod
    def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            out[i] = np.std(x[max(0, i - window + 1): i + 1])
        return out

    def _rolling_cross_dim_coherence(self, sv: np.ndarray, window: int) -> np.ndarray:
        n_frames, n_dims = sv.shape
        if n_dims < 2:
            return np.zeros(n_frames, dtype=float)
        out = np.zeros(n_frames, dtype=float)
        for t in range(n_frames):
            block = sv[max(0, t - window + 1): t + 1]
            if len(block) < 3:
                continue
            corr = np.corrcoef(block.T)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            triu = corr[np.triu_indices(n_dims, k=1)]
            out[t] = np.mean(np.abs(triu)) if len(triu) > 0 else 0.0
        return out

    def _rolling_low_frequency_ratio(self, sv: np.ndarray) -> np.ndarray:
        n_frames = sv.shape[0]
        window = max(16, min(64, n_frames // 4 if n_frames >= 20 else n_frames))
        out = np.zeros(n_frames, dtype=float)
        for t in range(n_frames):
            block = sv[max(0, t - window + 1): t + 1]
            if len(block) < 8:
                continue
            fft_mag = np.abs(np.fft.rfft(block, axis=0))
            cutoff = max(1, fft_mag.shape[0] // 4)
            out[t] = np.sum(fft_mag[:cutoff]) / (np.sum(fft_mag) + 1e-12)
        return out

    # --- K-means (no sklearn dependency) ---

    def _kmeans(self, X: np.ndarray, k: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_state)
        best_labels = np.zeros(len(X), dtype=int)
        best_inertia = np.inf

        for _ in range(self.config.n_init):
            idx = rng.choice(len(X), size=k, replace=False)
            centers = X[idx].copy()
            for _ in range(self.config.max_iter):
                d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels = np.argmin(d2, axis=1)
                new_centers = centers.copy()
                for j in range(k):
                    mask = labels == j
                    if np.any(mask):
                        new_centers[j] = np.mean(X[mask], axis=0)
                    else:
                        new_centers[j] = X[rng.integers(0, len(X))]
                if np.linalg.norm(new_centers - centers) < 1e-8:
                    break
                centers = new_centers
            inertia = np.sum(np.min(
                np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1,
            ))
            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
        return best_labels

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
# Regime-Aware Analyzer
# ============================================================================


class NetworkAnalyzerCoreV2(NetworkAnalyzerCore):
    """
    Regime-aware extension of NetworkAnalyzerCore.

    Pipeline:
      1. Run base analyzer on full data
      2. Detect regimes → contiguous segments
      3. Run base analyzer per segment
      4. Compute segment frequency matrix (how often each edge appears)
      5. Export everything for downstream fusion with InverseCausalEngine
    """

    def __init__(
        self,
        *args: Any,
        regime_config: GenericRegimeConfig | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.regime_detector = GenericRegimeDetector(regime_config)

    def analyze_regime_aware(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str] | None = None,
        window: int | None = None,
    ) -> RegimeAwareResult:
        n_frames, n_dims = state_vectors.shape
        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        # Step 1: full-data analysis
        base_result = super().analyze(state_vectors, dimension_names, window)

        # Step 2: regime detection + segmentation
        regime_labels = self.regime_detector.detect(state_vectors)
        segments = self.regime_detector.build_segments(regime_labels)
        transition_matrix = self.regime_detector.transition_matrix(regime_labels)

        # Step 3: segment-wise analysis
        regime_results: dict[int, list[NetworkResult]] = {}
        causal_counts = np.zeros((n_dims, n_dims), dtype=float)
        sync_counts = np.zeros((n_dims, n_dims), dtype=float)
        causal_strengths: dict[tuple[int, int], list[float]] = {}
        causal_lags: dict[tuple[int, int], list[int]] = {}
        n_analyzed = 0

        for seg in segments:
            seg_data = state_vectors[seg.start: seg.end]
            if len(seg_data) < max(12, self.regime_detector.config.min_segment_length):
                continue

            seg_result = super().analyze(seg_data, dimension_names, window=len(seg_data))
            regime_results.setdefault(seg.regime_id, []).append(seg_result)
            n_analyzed += 1

            seen_causal: set[tuple[int, int]] = set()
            for link in seg_result.causal_network:
                key = (link.from_dim, link.to_dim)
                if key not in seen_causal:
                    causal_counts[key] += 1.0
                    seen_causal.add(key)
                causal_strengths.setdefault(key, []).append(link.strength)
                causal_lags.setdefault(key, []).append(link.lag)

            seen_sync: set[tuple[int, int]] = set()
            for link in seg_result.sync_network:
                for key in [(link.from_dim, link.to_dim), (link.to_dim, link.from_dim)]:
                    if key not in seen_sync:
                        sync_counts[key] += 1.0
                        seen_sync.add(key)

        # Step 4: frequency matrices
        n_seg = max(1, n_analyzed)
        causal_freq = causal_counts / n_seg
        sync_freq = sync_counts / n_seg
        np.fill_diagonal(causal_freq, 0.0)
        np.fill_diagonal(sync_freq, 0.0)

        edge_mean_strength = {
            k: float(np.mean(v)) for k, v in causal_strengths.items()
        }
        edge_mean_lag = {
            k: float(np.mean(v)) for k, v in causal_lags.items()
        }

        logger.info(
            f"🔀 Regime-aware: {len(segments)} segments, "
            f"{n_analyzed} analyzed, "
            f"causal edges seen: {int(np.sum(causal_counts > 0))}"
        )

        return RegimeAwareResult(
            base_result=base_result,
            regime_labels=regime_labels,
            regime_segments=segments,
            regime_results=regime_results,
            regime_transition_matrix=transition_matrix,
            causal_frequency_matrix=causal_freq,
            sync_frequency_matrix=sync_freq,
            edge_mean_strength=edge_mean_strength,
            edge_mean_lag=edge_mean_lag,
            n_segments=n_analyzed,
        )
