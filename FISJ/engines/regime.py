"""
GenericRegimeDetector — domain-agnostic regime detection
=========================================================
Built by Masamichi & Tamaki

Extracted from network_analyzer_core_v2.py.
Used by NNNU_Inverse for event-driven data rescue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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


class GenericRegimeDetector:
    """
    Fully domain-agnostic regime detector.

    Uses generic dynamical features (rolling mean/std/coherence)
    + k-means clustering + smoothing to assign each frame to a regime.

    Output: contiguous segments where causality patterns may differ.
    """

    def __init__(self, config: GenericRegimeConfig | None = None):
        self.config = config or GenericRegimeConfig()

    def detect(self, state_vectors: np.ndarray) -> np.ndarray:
        if state_vectors.ndim != 2:
            raise ValueError("state_vectors must have shape (n_frames, n_dims)")

        n_frames, _ = state_vectors.shape
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
                        regime_id=current, start=start, end=end,
                        n_frames=end - start,
                    ))
                start = t
                current = int(labels[t])

        end = len(labels)
        if end - start >= self.config.min_segment_length:
            segments.append(RegimeSegment(
                regime_id=current, start=start, end=end,
                n_frames=end - start,
            ))
        return segments

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, state_vectors: np.ndarray) -> np.ndarray:
        n_frames, n_dims = state_vectors.shape
        diffs = np.diff(state_vectors, axis=0, prepend=state_vectors[[0]])

        mean_abs = np.mean(np.abs(state_vectors), axis=1)
        std_abs = np.std(state_vectors, axis=1)
        mean_step = np.mean(np.abs(diffs), axis=1)
        std_step = np.std(diffs, axis=1)

        features = [mean_abs, std_abs, mean_step, std_step]

        for w in self.config.feature_windows:
            features.append(self._rolling_mean(mean_step, w))
            features.append(self._rolling_std(mean_step, w))

        features.append(self._rolling_cross_dim_coherence(state_vectors, 20))
        features.append(self._rolling_low_frequency_ratio(state_vectors))

        return np.column_stack(features)

    @staticmethod
    def _zscore_matrix(X: np.ndarray) -> np.ndarray:
        mu = np.mean(X, axis=0, keepdims=True)
        sd = np.std(X, axis=0, keepdims=True) + 1e-12
        return (X - mu) / sd

    @staticmethod
    def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.zeros(n)
        for i in range(n):
            s = max(0, i - window // 2)
            e = min(n, i + window // 2 + 1)
            out[i] = np.mean(x[s:e])
        return out

    @staticmethod
    def _rolling_std(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.zeros(n)
        for i in range(n):
            s = max(0, i - window // 2)
            e = min(n, i + window // 2 + 1)
            if e > s + 1:
                out[i] = np.std(x[s:e])
        return out

    def _rolling_cross_dim_coherence(
        self, sv: np.ndarray, window: int,
    ) -> np.ndarray:
        n_frames = sv.shape[0]
        out = np.zeros(n_frames)
        for i in range(n_frames):
            s = max(0, i - window // 2)
            e = min(n_frames, i + window // 2 + 1)
            sub = sv[s:e]
            if sub.shape[0] > 2 and sub.shape[1] > 1:
                c = np.corrcoef(sub.T)
                if c.ndim == 2:
                    iu = np.triu_indices_from(c, k=1)
                    out[i] = float(np.nanmean(np.abs(c[iu]))) if iu[0].size > 0 else 0.0
        return out

    @staticmethod
    def _rolling_low_frequency_ratio(sv: np.ndarray) -> np.ndarray:
        n_frames, n_dims = sv.shape
        out = np.zeros(n_frames)
        if n_frames < 32:
            return out
        for d in range(n_dims):
            fft = np.fft.rfft(sv[:, d] - np.mean(sv[:, d]))
            pwr = np.abs(fft) ** 2
            total = pwr.sum() + 1e-12
            low = pwr[: len(pwr) // 8].sum()
            out += low / total
        out /= n_dims
        return np.full(n_frames, float(np.mean(out)))

    # ------------------------------------------------------------------
    # k-means
    # ------------------------------------------------------------------

    def _kmeans(self, X: np.ndarray, k: int) -> np.ndarray:
        rng = np.random.default_rng(self.config.random_state)
        n = X.shape[0]
        best_labels = np.zeros(n, dtype=int)
        best_inertia = np.inf

        for _ in range(self.config.n_init):
            idx = rng.choice(n, size=k, replace=False)
            centers = X[idx].copy()

            for _ in range(self.config.max_iter):
                d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
                labels = np.argmin(d2, axis=1)
                new_centers = np.zeros_like(centers)
                for c in range(k):
                    mask = labels == c
                    if np.any(mask):
                        new_centers[c] = X[mask].mean(axis=0)
                    else:
                        new_centers[c] = X[rng.integers(0, n)]
                if np.allclose(centers, new_centers, atol=1e-6):
                    centers = new_centers
                    break
                centers = new_centers

            d2 = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(d2, axis=1)
            inertia = float(np.sum(np.min(d2, axis=1)))

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()

        return best_labels

    @staticmethod
    def _smooth_labels(labels: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return labels
        n = len(labels)
        out = labels.copy()
        half = window // 2
        for i in range(n):
            s = max(0, i - half)
            e = min(n, i + half + 1)
            sub = labels[s:e]
            vals, counts = np.unique(sub, return_counts=True)
            out[i] = int(vals[np.argmax(counts)])
        return out
