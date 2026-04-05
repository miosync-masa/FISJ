
"""
Network Analyzer Core - Domain-Agnostic (Standalone, Adaptive, Multi-Scale Ensemble)
====================================================================================

Built by Masamichi & Tamaki

This module provides a domain-agnostic multidimensional network analysis engine
based on partial correlation, lagged causality estimation, adaptive windowing,
and multi-scale ensemble aggregation.

Core capabilities
-----------------
1. Single-scale network analysis on multivariate time series
2. Adaptive parameter tuning based on data diagnostics
3. Adaptive window derivation from shared metrics
4. Multi-scale ensemble network construction with temporal support statistics
5. Event-centered network analysis for cooperative transitions

Notes
-----
- Synchronization links are undirected.
- Causal links are directed and lag-aware.
- Multi-scale ensemble analysis aggregates edge evidence across both scales
  and overlapping temporal segments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("getter_one.analysis.network_analyzer_core")


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DimensionLink:
    """A single-scale network link between two dimensions."""

    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    link_type: str  # "sync" or "causal"
    strength: float
    correlation: float
    lag: int = 0


@dataclass
class EnsembleLink:
    """
    Aggregated network link across temporal segments and/or multiple scales.

    Attributes
    ----------
    strength : float
        Aggregated mean link strength.
    correlation : float
        Aggregated signed correlation.
    lag : float
        Mean lag for causal links. Zero for synchronization links.
    support : int
        Number of temporal segments in which the link appeared.
    support_ratio : float
        Fraction of all temporal segments (across the relevant ensemble scope)
        in which the link appeared.
    scale_support : int
        Number of scales in which the link appeared.
    scale_support_ratio : float
        Fraction of scales in which the link appeared.
    strength_std : float
        Standard deviation of link strength.
    correlation_std : float
        Standard deviation of signed correlation.
    lag_std : float
        Standard deviation of lag for causal links.
    sign_consistency : float
        Fraction of weighted support carrying the dominant sign.
    direction_confidence : float
        For causal links, support of the stored direction divided by support of
        both directions combined.
    scales : list[int]
        Window sizes contributing to this link.
    scale_labels : list[str]
        Human-readable labels for the contributing scales.
    """

    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    link_type: str  # "sync" or "causal"
    strength: float
    correlation: float
    lag: float = 0.0

    support: int = 0
    support_ratio: float = 0.0
    scale_support: int = 0
    scale_support_ratio: float = 0.0

    strength_std: float = 0.0
    correlation_std: float = 0.0
    lag_std: float = 0.0

    sign_consistency: float = 1.0
    direction_confidence: float = 1.0

    scales: list[int] = field(default_factory=list)
    scale_labels: list[str] = field(default_factory=list)


@dataclass
class NetworkResult:
    """Container for single-scale network analysis results."""

    sync_network: list[DimensionLink] = field(default_factory=list)
    causal_network: list[DimensionLink] = field(default_factory=list)

    sync_matrix: np.ndarray | None = None
    causal_matrix: np.ndarray | None = None
    causal_lag_matrix: np.ndarray | None = None

    pattern: str = "unknown"
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    causal_drivers: list[int] = field(default_factory=list)
    causal_followers: list[int] = field(default_factory=list)
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    n_dims: int = 0
    n_sync_links: int = 0
    n_causal_links: int = 0
    dimension_names: list[str] = field(default_factory=list)

    analysis_window: int = 0
    local_std_window: int = 0
    max_lag: int = 0
    segment_start: int | None = None
    segment_end: int | None = None

    adaptive_params: dict | None = None


@dataclass
class ScaleEnsembleResult:
    """Ensemble-aggregated result for one temporal scale."""

    scale: float
    analysis_window: int
    local_std_window: int
    max_lag: int
    stride: int

    sync_network: list[EnsembleLink] = field(default_factory=list)
    causal_network: list[EnsembleLink] = field(default_factory=list)

    pattern: str = "unknown"
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    causal_drivers: list[int] = field(default_factory=list)
    causal_followers: list[int] = field(default_factory=list)
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    n_segments: int = 0


@dataclass
class MultiScaleNetworkResult:
    """Container for multi-scale ensemble network analysis results."""

    scale_results: list[ScaleEnsembleResult] = field(default_factory=list)

    sync_network: list[EnsembleLink] = field(default_factory=list)
    causal_network: list[EnsembleLink] = field(default_factory=list)

    robust_sync_network: list[EnsembleLink] = field(default_factory=list)
    robust_causal_network: list[EnsembleLink] = field(default_factory=list)

    scale_specific_sync_network: list[EnsembleLink] = field(default_factory=list)
    scale_specific_causal_network: list[EnsembleLink] = field(default_factory=list)

    pattern: str = "unknown"
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    causal_drivers: list[int] = field(default_factory=list)
    causal_followers: list[int] = field(default_factory=list)
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    n_dims: int = 0
    n_scales: int = 0
    total_segments: int = 0
    dimension_names: list[str] = field(default_factory=list)

    adaptive_params: dict | None = None


@dataclass
class CooperativeEventNetwork:
    """Network structure around a cooperative event."""

    event_frame: int
    event_timestamp: str | None = None
    delta_lambda_c: float = 0.0

    network: NetworkResult | MultiScaleNetworkResult | None = None

    initiator_dims: list[int] = field(default_factory=list)
    initiator_names: list[str] = field(default_factory=list)
    propagation_order: list[int] = field(default_factory=list)


# ============================================================================
# Network Analyzer Core
# ============================================================================


class NetworkAnalyzerCore:
    """
    Domain-agnostic multidimensional network analyzer with adaptive multi-scale
    ensemble support.

    Parameters
    ----------
    sync_threshold : float
        Baseline threshold for synchronization links.
    causal_threshold : float
        Baseline threshold for causal links.
    max_lag : int
        Baseline maximum lag (in frames) for causal estimation.
    adaptive : bool
        Whether to enable data-driven adaptive thresholds and windows.
    local_std_window : int
        Baseline rolling window used for local standard deviation.
    analysis_window_hint : int | None
        Optional baseline analysis window hint. If omitted, the analyzer
        derives a reasonable default from the data length and baseline lag.
    multiscale_scales : tuple[float, ...]
        Relative scales used to generate multi-scale window sizes.
    multiscale_stride_ratio : float
        Fraction of each scale window used as the default stride in
        multi-scale sliding-window analysis.
    """

    def __init__(
        self,
        sync_threshold: float = 0.5,
        causal_threshold: float = 0.4,
        max_lag: int = 12,
        adaptive: bool = True,
        local_std_window: int = 20,
        analysis_window_hint: int | None = None,
        multiscale_scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
        multiscale_stride_ratio: float = 0.5,
    ):
        self.sync_threshold_hint = float(sync_threshold)
        self.causal_threshold_hint = float(causal_threshold)
        self.max_lag_hint = int(max_lag)
        self.local_std_window_hint = int(local_std_window)
        self.analysis_window_hint = analysis_window_hint
        self.adaptive = bool(adaptive)

        self.multiscale_scales = tuple(sorted(set(float(s) for s in multiscale_scales)))
        self.multiscale_stride_ratio = float(multiscale_stride_ratio)

        self.sync_threshold = float(sync_threshold)
        self.causal_threshold = float(causal_threshold)
        self.max_lag = int(max_lag)
        self.local_std_window = int(local_std_window)

        self.adaptive_params: dict[str, Any] | None = None

        logger.info(
            "NetworkAnalyzerCore initialized "
            "(sync>%.3f, causal>%.3f, max_lag=%d, adaptive=%s, "
            "local_std_window=%d, analysis_window_hint=%s)",
            self.sync_threshold_hint,
            self.causal_threshold_hint,
            self.max_lag_hint,
            self.adaptive,
            self.local_std_window_hint,
            self.analysis_window_hint,
        )

    # ========================================================================
    # Adaptive Diagnostics
    # ========================================================================

    def _compute_adaptive_metrics(self, state_vectors: np.ndarray) -> dict[str, float]:
        """
        Extract shared adaptive diagnostics from the input series.

        The same metrics are reused for threshold adaptation, lag adaptation,
        and adaptive window derivation.

        Returns
        -------
        dict[str, float]
            Shared diagnostics:
            - global_volatility
            - temporal_volatility
            - correlation_complexity
            - local_variation
            - low_freq_ratio
            - mean_abs_corr
        """
        n_frames, n_dims = state_vectors.shape

        global_std = float(np.std(state_vectors))
        global_mean = float(np.mean(np.abs(state_vectors)))
        volatility_ratio = global_std / (global_mean + 1e-10)

        if n_frames > 1:
            temporal_changes = np.diff(state_vectors, axis=0)
            temporal_volatility = float(np.mean(np.std(temporal_changes, axis=0)))
        else:
            temporal_volatility = 0.0

        if n_dims > 1:
            corr_matrix = np.corrcoef(state_vectors.T)
            triu = corr_matrix[np.triu_indices(n_dims, k=1)]
            triu = triu[~np.isnan(triu)]
            mean_abs_corr = float(np.mean(np.abs(triu))) if len(triu) > 0 else 0.0
            correlation_complexity = 1.0 - mean_abs_corr if len(triu) > 0 else 0.5
        else:
            mean_abs_corr = 0.0
            correlation_complexity = 0.5

        base_window = max(10, min(n_frames, max(10, n_frames // 10)))
        local_volatilities: list[float] = []
        stride = max(1, base_window // 2)
        for i in range(0, max(1, n_frames - base_window + 1), stride):
            window_data = state_vectors[i : i + base_window]
            if len(window_data) > 0:
                local_volatilities.append(float(np.std(window_data)))

        if len(local_volatilities) > 1:
            volatility_variation = float(
                np.std(local_volatilities) / (np.mean(local_volatilities) + 1e-10)
            )
        else:
            volatility_variation = 0.5

        if n_frames > 2:
            fft_mag = np.abs(np.fft.fft(state_vectors, axis=0))
            low_cutoff = max(1, n_frames // 10)
            high_cutoff = max(2, n_frames // 2)
            low_freq_ratio = float(
                np.sum(fft_mag[:low_cutoff]) / (np.sum(fft_mag[:high_cutoff]) + 1e-10)
            )
        else:
            low_freq_ratio = 0.5

        return {
            "global_std": float(global_std),
            "global_mean_abs": float(global_mean),
            "global_volatility": float(volatility_ratio),
            "temporal_volatility": float(temporal_volatility),
            "correlation_complexity": float(correlation_complexity),
            "local_variation": float(volatility_variation),
            "low_freq_ratio": float(low_freq_ratio),
            "mean_abs_corr": float(mean_abs_corr),
        }

    def _derive_adaptive_thresholds(
        self,
        metrics: dict[str, float],
        n_frames: int,
        n_dims: int,
    ) -> dict[str, float | int]:
        """
        Derive thresholds and lag range from shared adaptive metrics.
        """
        global_std = metrics["global_std"]
        global_volatility = metrics["global_volatility"]
        temporal_volatility = metrics["temporal_volatility"]
        corr_complexity = metrics["correlation_complexity"]
        low_freq_ratio = metrics["low_freq_ratio"]
        mean_abs_corr = metrics["mean_abs_corr"]

        sync_adj = 0.0
        if global_volatility > 2.0:
            sync_adj += 0.10
        elif global_volatility < 0.3:
            sync_adj -= 0.10

        if mean_abs_corr > 0.6:
            sync_adj += 0.10
        elif mean_abs_corr < 0.2:
            sync_adj -= 0.10

        if metrics["local_variation"] > 1.0:
            sync_adj += 0.05

        sync_threshold = float(
            np.clip(self.sync_threshold_hint + sync_adj, 0.15, 0.85)
        )

        causal_adj = 0.0
        if temporal_volatility < global_std * 0.3:
            causal_adj -= 0.10
        elif temporal_volatility > global_std * 2.0:
            causal_adj += 0.10

        if n_dims > 50:
            causal_adj += 0.10
        elif n_dims <= 5:
            causal_adj -= 0.05

        if corr_complexity > 0.7:
            causal_adj -= 0.05

        causal_threshold = float(
            np.clip(self.causal_threshold_hint + causal_adj, 0.15, 0.80)
        )

        scale_factor = 1.0
        if low_freq_ratio > 0.8:
            scale_factor *= 1.5
        elif low_freq_ratio < 0.3:
            scale_factor *= 0.7

        if corr_complexity > 0.7:
            scale_factor *= 1.3

        if temporal_volatility > global_std * 2.0:
            scale_factor *= 0.8

        raw_lag = int(round(self.max_lag_hint * scale_factor))
        max_lag = int(np.clip(raw_lag, 2, max(4, n_frames // 5)))

        return {
            "sync_threshold": sync_threshold,
            "causal_threshold": causal_threshold,
            "max_lag": max_lag,
            "threshold_scale_factor": float(scale_factor),
        }

    def _derive_adaptive_windows(
        self,
        metrics: dict[str, float],
        n_frames: int,
    ) -> dict[str, Any]:
        """
        Derive adaptive window sizes from shared adaptive metrics.

        This reuses the same metrics that drive the threshold adaptation. The
        resulting windows are used for local standard deviation, event analysis,
        and multi-scale ensemble generation.
        """
        if self.analysis_window_hint is None:
            base_window = max(24, min(n_frames, max(self.max_lag_hint * 8, n_frames // 5)))
        else:
            base_window = int(np.clip(self.analysis_window_hint, 12, n_frames))

        if n_frames > 300:
            size_adjusted_base = base_window
        elif n_frames > 100:
            size_adjusted_base = int(round(base_window * 0.8))
        else:
            size_adjusted_base = int(round(base_window * 0.65))

        size_adjusted_base = max(size_adjusted_base, max(12, n_frames // 20))

        scale_factor = 1.0
        if metrics["global_volatility"] > 2.0:
            scale_factor *= 0.8
        elif metrics["global_volatility"] < 0.3:
            scale_factor *= 1.5

        if metrics["temporal_volatility"] > metrics["global_std"] * 2.0:
            scale_factor *= 0.9
        elif metrics["temporal_volatility"] < metrics["global_std"] * 0.3:
            scale_factor *= 1.4

        if metrics["correlation_complexity"] > 0.7:
            scale_factor *= 1.2
        elif metrics["correlation_complexity"] < 0.3:
            scale_factor *= 0.9

        if metrics["local_variation"] > 1.0:
            scale_factor *= 0.85

        if metrics["low_freq_ratio"] > 0.8:
            scale_factor *= 1.4
        elif metrics["low_freq_ratio"] < 0.3:
            scale_factor *= 0.8

        analysis_window = int(np.clip(
            round(size_adjusted_base * scale_factor),
            max(12, self.max_lag_hint * 4),
            n_frames,
        ))

        local_std_window = int(
            np.clip(
                round(max(self.local_std_window_hint, analysis_window * 0.35)),
                5,
                max(5, analysis_window // 2),
            )
        )

        event_window_before = int(np.clip(round(analysis_window * 0.75), 8, n_frames))
        event_window_after = int(np.clip(round(analysis_window * 0.25), 4, n_frames))

        multiscale_windows = sorted(
            {
                int(np.clip(round(analysis_window * scale), max(12, local_std_window), n_frames))
                for scale in self.multiscale_scales
            }
        )
        multiscale_windows.append(analysis_window)
        if n_frames > analysis_window and n_frames >= max(48, int(round(analysis_window * 1.25))):
            multiscale_windows.append(n_frames)
        multiscale_windows = sorted(set(multiscale_windows))

        return {
            "analysis_window": analysis_window,
            "local_std_window": local_std_window,
            "event_window_before": event_window_before,
            "event_window_after": event_window_after,
            "multiscale_windows": multiscale_windows,
            "window_scale_factor": float(scale_factor),
        }

    def _compute_adaptive_parameters(self, state_vectors: np.ndarray) -> dict[str, Any]:
        """
        Compute all adaptive parameters from a single shared metric extraction.
        """
        n_frames, n_dims = state_vectors.shape
        metrics = self._compute_adaptive_metrics(state_vectors)
        thresholds = self._derive_adaptive_thresholds(metrics, n_frames, n_dims)
        windows = self._derive_adaptive_windows(metrics, n_frames)

        params = {
            "metrics": metrics,
            "thresholds": thresholds,
            "windows": windows,
            # Convenience aliases
            "sync_threshold": thresholds["sync_threshold"],
            "causal_threshold": thresholds["causal_threshold"],
            "max_lag": thresholds["max_lag"],
            "analysis_window": windows["analysis_window"],
            "local_std_window": windows["local_std_window"],
            "multiscale_windows": windows["multiscale_windows"],
            "volatility_metrics": metrics,
        }

        logger.info(
            "Adaptive params: sync_th=%.3f, causal_th=%.3f, max_lag=%d, "
            "analysis_window=%d, local_std_window=%d",
            params["sync_threshold"],
            params["causal_threshold"],
            params["max_lag"],
            params["analysis_window"],
            params["local_std_window"],
        )
        logger.info(
            "Adaptive metrics: global_vol=%.3f, temporal_vol=%.3f, "
            "corr_complexity=%.3f, local_var=%.3f, low_freq=%.3f, mean_abs_corr=%.3f",
            metrics["global_volatility"],
            metrics["temporal_volatility"],
            metrics["correlation_complexity"],
            metrics["local_variation"],
            metrics["low_freq_ratio"],
            metrics["mean_abs_corr"],
        )
        return params

    # ========================================================================
    # Public API
    # ========================================================================

    def analyze(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str] | None = None,
        window: int | None = None,
        use_adaptive_window: bool = False,
    ) -> NetworkResult:
        """
        Run single-scale network analysis.

        Parameters
        ----------
        state_vectors : np.ndarray of shape (n_frames, n_dims)
            Input multivariate time series.
        dimension_names : list[str] | None
            Optional dimension names.
        window : int | None
            Number of most recent frames to analyze. If omitted, the full series
            is used unless `use_adaptive_window=True`.
        use_adaptive_window : bool
            If True and `window` is None, use the adaptively derived analysis
            window instead of the full series length.
        """
        n_frames, n_dims = state_vectors.shape
        dimension_names = self._resolve_dimension_names(dimension_names, n_dims)

        adaptive_params = self._compute_adaptive_parameters(state_vectors) if self.adaptive else None

        sync_threshold = (
            adaptive_params["sync_threshold"] if adaptive_params else self.sync_threshold_hint
        )
        causal_threshold = (
            adaptive_params["causal_threshold"] if adaptive_params else self.causal_threshold_hint
        )
        max_lag = adaptive_params["max_lag"] if adaptive_params else self.max_lag_hint
        local_std_window = (
            adaptive_params["local_std_window"] if adaptive_params else self.local_std_window_hint
        )

        if window is not None:
            analysis_window = int(np.clip(window, 8, n_frames))
        elif use_adaptive_window and adaptive_params is not None:
            analysis_window = int(adaptive_params["analysis_window"])
        else:
            analysis_window = n_frames

        segment, start, end = self._select_recent_segment(state_vectors, analysis_window)

        logger.info(
            "Analyzing single-scale network: n_dims=%d, n_frames=%d, "
            "segment=[%d:%d], sync>%.3f, causal>%.3f, max_lag=%d, local_std_window=%d",
            n_dims,
            n_frames,
            start,
            end,
            sync_threshold,
            causal_threshold,
            max_lag,
            local_std_window,
        )

        self.sync_threshold = float(sync_threshold)
        self.causal_threshold = float(causal_threshold)
        self.max_lag = int(max_lag)
        self.local_std_window = int(local_std_window)
        self.adaptive_params = adaptive_params

        result = self._analyze_core(
            segment,
            dimension_names=dimension_names,
            sync_threshold=float(sync_threshold),
            causal_threshold=float(causal_threshold),
            max_lag=int(max_lag),
            local_std_window=int(local_std_window),
            adaptive_params=adaptive_params,
            segment_start=start,
            segment_end=end,
        )
        self._print_summary(result)
        return result

    def analyze_multiscale(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str] | None = None,
        base_window: int | None = None,
        scales: tuple[float, ...] | None = None,
        stride_ratio: float | None = None,
        min_scale_support_ratio: float = 0.7,
        min_segment_support_ratio: float = 0.3,
    ) -> MultiScaleNetworkResult:
        """
        Run multi-scale ensemble network analysis.

        The algorithm proceeds in three nested stages:

        1. Shared adaptive metrics are computed once.
        2. Multiple temporal scales are generated from an adaptive base window.
        3. For each scale, overlapping temporal segments are analyzed and
           aggregated into scale-specific ensemble edges.
        4. Scale-specific ensembles are merged into a global multi-scale
           ensemble graph.

        Parameters
        ----------
        state_vectors : np.ndarray
            Input multivariate time series.
        dimension_names : list[str] | None
            Optional dimension names.
        base_window : int | None
            Optional base window. If omitted, the adaptive base window is used.
        scales : tuple[float, ...] | None
            Relative scale multipliers. If omitted, the configured defaults are
            used.
        stride_ratio : float | None
            Sliding stride as a fraction of each scale window.
        min_scale_support_ratio : float
            Minimum fraction of scales for a link to be considered robust.
        min_segment_support_ratio : float
            Minimum fraction of temporal segments for a link to be considered
            robust.

        Returns
        -------
        MultiScaleNetworkResult
            Global multi-scale ensemble network result.
        """
        n_frames, n_dims = state_vectors.shape
        dimension_names = self._resolve_dimension_names(dimension_names, n_dims)

        adaptive_params = self._compute_adaptive_parameters(state_vectors) if self.adaptive else None
        base_thresholds = adaptive_params["thresholds"] if adaptive_params else {
            "sync_threshold": self.sync_threshold_hint,
            "causal_threshold": self.causal_threshold_hint,
            "max_lag": self.max_lag_hint,
        }
        base_windows = adaptive_params["windows"] if adaptive_params else {
            "analysis_window": min(n_frames, self.analysis_window_hint or max(24, n_frames // 5)),
            "local_std_window": self.local_std_window_hint,
            "multiscale_windows": [min(n_frames, self.analysis_window_hint or max(24, n_frames // 5))],
        }

        windows = self._resolve_multiscale_windows(
            n_frames=n_frames,
            base_window=base_window or base_windows["analysis_window"],
            adaptive_windows=base_windows.get("multiscale_windows"),
            scales=scales,
        )
        stride_ratio = float(self.multiscale_stride_ratio if stride_ratio is None else stride_ratio)

        scale_results: list[ScaleEnsembleResult] = []
        total_segments = 0
        resolved_scale_factors = self._window_to_scale_factors(
            windows=windows,
            base_window=base_window or base_windows["analysis_window"],
        )

        logger.info(
            "Analyzing multi-scale network: n_dims=%d, n_frames=%d, windows=%s",
            n_dims,
            n_frames,
            windows,
        )

        for window_size, scale_factor in zip(windows, resolved_scale_factors):
            scale_params = self._derive_scale_runtime_parameters(
                window_size=window_size,
                base_window=base_window or base_windows["analysis_window"],
                base_local_std_window=base_windows["local_std_window"],
                base_max_lag=int(base_thresholds["max_lag"]),
                base_sync_threshold=float(base_thresholds["sync_threshold"]),
                base_causal_threshold=float(base_thresholds["causal_threshold"]),
            )
            stride = max(1, int(round(window_size * stride_ratio)))
            segments = self._generate_sliding_segments(n_frames, window_size, stride)
            total_segments += len(segments)

            segment_results: list[NetworkResult] = []
            for start, end in segments:
                segment = state_vectors[start:end]
                segment_result = self._analyze_core(
                    segment,
                    dimension_names=dimension_names,
                    sync_threshold=scale_params["sync_threshold"],
                    causal_threshold=scale_params["causal_threshold"],
                    max_lag=scale_params["max_lag"],
                    local_std_window=scale_params["local_std_window"],
                    adaptive_params=adaptive_params,
                    segment_start=start,
                    segment_end=end,
                )
                segment_results.append(segment_result)

            scale_result = self._aggregate_scale_result(
                segment_results=segment_results,
                dimension_names=dimension_names,
                scale=float(scale_factor),
                analysis_window=int(window_size),
                local_std_window=int(scale_params["local_std_window"]),
                max_lag=int(scale_params["max_lag"]),
                stride=int(stride),
            )
            scale_results.append(scale_result)

        result = self._aggregate_multiscale_result(
            scale_results=scale_results,
            dimension_names=dimension_names,
            n_dims=n_dims,
            min_scale_support_ratio=min_scale_support_ratio,
            min_segment_support_ratio=min_segment_support_ratio,
            adaptive_params=adaptive_params,
            total_segments=total_segments,
        )

        self.adaptive_params = adaptive_params
        self._print_multiscale_summary(result)
        return result

    def analyze_event_network(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        window_before: int | None = None,
        window_after: int | None = None,
        dimension_names: list[str] | None = None,
        multiscale: bool = False,
        scales: tuple[float, ...] | None = None,
        stride_ratio: float | None = None,
        min_scale_support_ratio: float = 0.7,
        min_segment_support_ratio: float = 0.3,
    ) -> CooperativeEventNetwork:
        """
        Analyze a local neighborhood around an event.

        If `multiscale=True`, the event neighborhood is analyzed with the full
        multi-scale ensemble pipeline.
        """
        n_frames, n_dims = state_vectors.shape
        dimension_names = self._resolve_dimension_names(dimension_names, n_dims)
        adaptive_params = self._compute_adaptive_parameters(state_vectors) if self.adaptive else None

        if window_before is None:
            if adaptive_params is not None:
                window_before = int(adaptive_params["windows"]["event_window_before"])
            else:
                window_before = 24
        if window_after is None:
            if adaptive_params is not None:
                window_after = int(adaptive_params["windows"]["event_window_after"])
            else:
                window_after = 6

        start = max(0, event_frame - window_before)
        end = min(n_frames, event_frame + window_after)
        local_data = state_vectors[start:end]

        if multiscale:
            network: NetworkResult | MultiScaleNetworkResult = self.analyze_multiscale(
                local_data,
                dimension_names=dimension_names,
                scales=scales,
                stride_ratio=stride_ratio,
                min_scale_support_ratio=min_scale_support_ratio,
                min_segment_support_ratio=min_segment_support_ratio,
            )
        else:
            network = self.analyze(
                local_data,
                dimension_names=dimension_names,
                window=len(local_data),
                use_adaptive_window=False,
            )

        initiators = self._identify_initiators(
            state_vectors, event_frame, int(window_before), n_dims
        )
        propagation = self._estimate_propagation_order(
            state_vectors, event_frame, int(window_before), n_dims
        )

        return CooperativeEventNetwork(
            event_frame=event_frame,
            network=network,
            initiator_dims=initiators,
            initiator_names=[dimension_names[d] for d in initiators],
            propagation_order=propagation,
        )

    # ========================================================================
    # Core Analysis
    # ========================================================================

    def _analyze_core(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str],
        sync_threshold: float,
        causal_threshold: float,
        max_lag: int,
        local_std_window: int,
        adaptive_params: dict[str, Any] | None = None,
        segment_start: int | None = None,
        segment_end: int | None = None,
    ) -> NetworkResult:
        """
        Internal single-scale analysis with explicit runtime parameters.
        """
        n_frames, n_dims = state_vectors.shape

        if n_frames < 3:
            empty_matrix = np.zeros((n_dims, n_dims))
            return NetworkResult(
                sync_network=[],
                causal_network=[],
                sync_matrix=empty_matrix.copy(),
                causal_matrix=empty_matrix.copy(),
                causal_lag_matrix=np.zeros((n_dims, n_dims), dtype=int),
                pattern="independent",
                hub_dimensions=[],
                hub_names=[],
                causal_drivers=[],
                causal_followers=[],
                driver_names=[],
                follower_names=[],
                n_dims=n_dims,
                n_sync_links=0,
                n_causal_links=0,
                dimension_names=dimension_names,
                analysis_window=n_frames,
                local_std_window=local_std_window,
                max_lag=max_lag,
                segment_start=segment_start,
                segment_end=segment_end,
                adaptive_params=adaptive_params,
            )

        local_std = self._compute_local_std(state_vectors, local_std_window)
        correlations = self._compute_correlations(
            state_vectors=state_vectors,
            local_std=local_std,
            max_lag=max_lag,
        )

        sync_links, causal_links = self._build_networks(
            correlations=correlations,
            dimension_names=dimension_names,
            sync_threshold=sync_threshold,
            causal_threshold=causal_threshold,
        )
        causal_links = self._filter_spurious_edges(causal_links)

        pattern = self._identify_pattern(sync_links, causal_links)
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)
        drivers, followers = self._identify_causal_structure(causal_links, n_dims)

        return NetworkResult(
            sync_network=sync_links,
            causal_network=causal_links,
            sync_matrix=correlations["sync"],
            causal_matrix=correlations["max_lagged"],
            causal_lag_matrix=correlations["best_lag"],
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
            analysis_window=n_frames,
            local_std_window=local_std_window,
            max_lag=max_lag,
            segment_start=segment_start,
            segment_end=segment_end,
            adaptive_params=adaptive_params,
        )

    # ========================================================================
    # Signal Processing
    # ========================================================================

    @staticmethod
    def _compute_local_std(
        state_vectors: np.ndarray,
        window_size: int,
    ) -> np.ndarray:
        """
        Compute local standard deviation using a rolling symmetric window.
        """
        n_frames, n_dims = state_vectors.shape
        window_size = int(np.clip(window_size, 3, max(3, n_frames)))
        half_w = window_size // 2
        local_std = np.zeros_like(state_vectors, dtype=float)

        for d in range(n_dims):
            series = state_vectors[:, d]
            for t in range(n_frames):
                start = max(0, t - half_w)
                end = min(n_frames, t + half_w + 1)
                local_std[t, d] = np.std(series[start:end])

        return local_std

    def _compute_correlations(
        self,
        state_vectors: np.ndarray,
        local_std: np.ndarray,
        max_lag: int,
    ) -> dict[str, np.ndarray]:
        """
        Compute synchronization and lagged causality matrices.
        """
        n_frames, n_dims = state_vectors.shape

        sync_matrix = np.zeros((n_dims, n_dims))
        max_lagged_matrix = np.zeros((n_dims, n_dims))
        best_lag_matrix = np.zeros((n_dims, n_dims), dtype=int)

        if n_frames < 3:
            return {
                "sync": sync_matrix,
                "max_lagged": max_lagged_matrix,
                "best_lag": best_lag_matrix,
            }

        raw_displacement = np.diff(state_vectors, axis=0)
        local_std_diff = local_std[1:]
        displacement = raw_displacement / (local_std_diff + 1e-10)

        use_partial = n_dims >= 3 and len(displacement) >= 3

        if use_partial:
            sync_matrix = self._compute_partial_corr_precision(displacement)
        else:
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    ts_i = displacement[:, i]
                    ts_j = displacement[:, j]
                    if np.std(ts_i) < 1e-10 or np.std(ts_j) < 1e-10:
                        continue
                    corr = np.corrcoef(ts_i, ts_j)[0, 1]
                    corr = 0.0 if np.isnan(corr) else float(corr)
                    sync_matrix[i, j] = corr
                    sync_matrix[j, i] = corr

        lag_upper = min(max_lag, len(displacement) - 2)
        if lag_upper < 1:
            return {
                "sync": sync_matrix,
                "max_lagged": max_lagged_matrix,
                "best_lag": best_lag_matrix,
            }

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                best_corr = 0.0
                best_lag = 0

                for lag in range(1, lag_upper + 1):
                    corr_ij = self._lagged_partial_corr(
                        displacement=displacement,
                        src=i,
                        dst=j,
                        lag=lag,
                        use_partial=use_partial,
                    )
                    if abs(corr_ij) > abs(best_corr):
                        best_corr = corr_ij
                        best_lag = lag

                    corr_ji = self._lagged_partial_corr(
                        displacement=displacement,
                        src=j,
                        dst=i,
                        lag=lag,
                        use_partial=use_partial,
                    )
                    if abs(corr_ji) > abs(best_corr):
                        best_corr = corr_ji
                        best_lag = -lag

                max_lagged_matrix[i, j] = best_corr
                max_lagged_matrix[j, i] = best_corr
                best_lag_matrix[i, j] = best_lag
                best_lag_matrix[j, i] = -best_lag

        return {
            "sync": sync_matrix,
            "max_lagged": max_lagged_matrix,
            "best_lag": best_lag_matrix,
        }

    @staticmethod
    def _compute_partial_corr_precision(data: np.ndarray) -> np.ndarray:
        """
        Compute partial correlation from the precision matrix.

        pcorr(i, j) = -P[i, j] / sqrt(P[i, i] * P[j, j])
        """
        if data.ndim != 2 or data.shape[1] < 2:
            n_dims = data.shape[1] if data.ndim == 2 else 1
            return np.zeros((n_dims, n_dims))

        n_dims = data.shape[1]
        cov = np.cov(data.T)
        cov = np.atleast_2d(cov)
        cov += np.eye(n_dims) * 1e-8

        try:
            precision = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov)

        diag = np.sqrt(np.abs(np.diag(precision)))
        diag[diag < 1e-10] = 1e-10

        partial_corr = np.zeros((n_dims, n_dims))
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                pc = -precision[i, j] / (diag[i] * diag[j])
                pc = float(np.clip(pc, -1.0, 1.0))
                partial_corr[i, j] = pc
                partial_corr[j, i] = pc

        return partial_corr

    @staticmethod
    def _lagged_partial_corr(
        displacement: np.ndarray,
        src: int,
        dst: int,
        lag: int,
        use_partial: bool,
    ) -> float:
        """
        Compute lagged partial correlation with multi-lag conditioning.
        """
        n_samples, n_dims = displacement.shape
        if lag >= n_samples - 1:
            return 0.0

        ts_src = displacement[:-lag, src]
        ts_dst = displacement[lag:, dst]
        n = min(len(ts_src), len(ts_dst))

        ts_src = ts_src[:n]
        ts_dst = ts_dst[:n]

        if np.std(ts_src) < 1e-10 or np.std(ts_dst) < 1e-10:
            return 0.0

        if not use_partial or n_dims < 3:
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)

        other_dims = [d for d in range(n_dims) if d != src and d != dst]
        if not other_dims:
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)

        max_cond_lag = min(lag + 2, n - 1)
        z_parts: list[np.ndarray] = []
        for cl in range(max_cond_lag + 1):
            start_s = max(0, cl)
            end_s = start_s + n
            if end_s <= len(displacement) - lag + cl:
                z_lag = displacement[start_s:end_s, :][:, other_dims]
                if len(z_lag) >= n:
                    z_parts.append(z_lag[:n])

        if not z_parts:
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)

        z = np.hstack(z_parts)

        z_std = np.std(z, axis=0)
        valid_cols = z_std > 1e-10
        if not np.any(valid_cols):
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)
        z = z[:, valid_cols]

        if z.shape[1] > max(1, z.shape[0] // 2):
            try:
                u, s, _ = np.linalg.svd(z, full_matrices=False)
                cumvar = np.cumsum(s**2) / (np.sum(s**2) + 1e-10)
                n_keep = max(1, int(np.searchsorted(cumvar, 0.95)) + 1)
                z = u[:, :n_keep] * s[:n_keep]
            except np.linalg.LinAlgError:
                pass

        try:
            z_pinv = np.linalg.pinv(z)
            resid_src = ts_src - z @ (z_pinv @ ts_src)
            resid_dst = ts_dst - z @ (z_pinv @ ts_dst)
        except np.linalg.LinAlgError:
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)

        if np.std(resid_src) < 1e-10 or np.std(resid_dst) < 1e-10:
            return 0.0

        corr = np.corrcoef(resid_src, resid_dst)[0, 1]
        return 0.0 if np.isnan(corr) else float(corr)

    # ========================================================================
    # Network Construction
    # ========================================================================

    @staticmethod
    def _build_networks(
        correlations: dict[str, np.ndarray],
        dimension_names: list[str],
        sync_threshold: float,
        causal_threshold: float,
    ) -> tuple[list[DimensionLink], list[DimensionLink]]:
        """
        Build synchronization and causal links from correlation matrices.
        """
        n_dims = len(dimension_names)
        sync_links: list[DimensionLink] = []
        causal_links: list[DimensionLink] = []

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                sync_corr = float(correlations["sync"][i, j])
                causal_corr = float(correlations["max_lagged"][i, j])
                lag = int(correlations["best_lag"][i, j])

                if abs(sync_corr) > sync_threshold:
                    sync_links.append(
                        DimensionLink(
                            from_dim=i,
                            to_dim=j,
                            from_name=dimension_names[i],
                            to_name=dimension_names[j],
                            link_type="sync",
                            strength=abs(sync_corr),
                            correlation=sync_corr,
                            lag=0,
                        )
                    )

                if abs(causal_corr) > causal_threshold and abs(causal_corr) > abs(sync_corr) * 1.1:
                    if lag > 0:
                        from_d, to_d = i, j
                        lag_used = lag
                    else:
                        from_d, to_d = j, i
                        lag_used = abs(lag)

                    causal_links.append(
                        DimensionLink(
                            from_dim=from_d,
                            to_dim=to_d,
                            from_name=dimension_names[from_d],
                            to_name=dimension_names[to_d],
                            link_type="causal",
                            strength=abs(causal_corr),
                            correlation=causal_corr,
                            lag=lag_used,
                        )
                    )

        return sync_links, causal_links

    @staticmethod
    def _filter_spurious_edges(causal_links: list[DimensionLink]) -> list[DimensionLink]:
        """
        Remove likely spurious causal edges caused by common ancestors.
        """
        if len(causal_links) < 3:
            return causal_links

        link_map: dict[tuple[int, int], DimensionLink] = {}
        for link in causal_links:
            key = (link.from_dim, link.to_dim)
            if key not in link_map or link.strength > link_map[key].strength:
                link_map[key] = link

        filtered: list[DimensionLink] = []
        for link in causal_links:
            a, b = link.from_dim, link.to_dim
            remove_link = False

            for (z_src, z_dst), z_link_a in link_map.items():
                if z_dst != a:
                    continue
                z = z_src
                if z in (a, b):
                    continue

                z_link_b = link_map.get((z, b))
                if z_link_b is None:
                    continue

                if z_link_a.strength > link.strength and z_link_b.strength > link.strength:
                    remove_link = True
                    break

            if not remove_link:
                filtered.append(link)

        return filtered

    # ========================================================================
    # Multi-Scale Ensemble Aggregation
    # ========================================================================

    def _aggregate_scale_result(
        self,
        segment_results: list[NetworkResult],
        dimension_names: list[str],
        scale: float,
        analysis_window: int,
        local_std_window: int,
        max_lag: int,
        stride: int,
    ) -> ScaleEnsembleResult:
        """
        Aggregate per-segment results into a scale-specific ensemble.
        """
        n_dims = len(dimension_names)
        n_segments = len(segment_results)

        sync_links = self._aggregate_links_within_scale(
            segment_results=segment_results,
            dimension_names=dimension_names,
            link_type="sync",
            analysis_window=analysis_window,
            n_segments=n_segments,
        )
        causal_links = self._aggregate_links_within_scale(
            segment_results=segment_results,
            dimension_names=dimension_names,
            link_type="causal",
            analysis_window=analysis_window,
            n_segments=n_segments,
        )

        pattern = self._identify_pattern(sync_links, causal_links)
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)
        drivers, followers = self._identify_causal_structure(causal_links, n_dims)

        return ScaleEnsembleResult(
            scale=scale,
            analysis_window=analysis_window,
            local_std_window=local_std_window,
            max_lag=max_lag,
            stride=stride,
            sync_network=sync_links,
            causal_network=causal_links,
            pattern=pattern,
            hub_dimensions=hub_dims,
            hub_names=[dimension_names[d] for d in hub_dims],
            causal_drivers=drivers,
            causal_followers=followers,
            driver_names=[dimension_names[d] for d in drivers],
            follower_names=[dimension_names[d] for d in followers],
            n_segments=n_segments,
        )

    def _aggregate_links_within_scale(
        self,
        segment_results: list[NetworkResult],
        dimension_names: list[str],
        link_type: str,
        analysis_window: int,
        n_segments: int,
    ) -> list[EnsembleLink]:
        """
        Aggregate raw links across temporal segments within one scale.
        """
        buckets: dict[tuple[int, int], list[DimensionLink]] = {}
        for result in segment_results:
            links = result.sync_network if link_type == "sync" else result.causal_network
            for link in links:
                if link_type == "sync":
                    key = self._canonical_sync_key(link.from_dim, link.to_dim)
                else:
                    key = (link.from_dim, link.to_dim)
                buckets.setdefault(key, []).append(link)

        aggregated: dict[tuple[int, int], EnsembleLink] = {}
        for key, records in buckets.items():
            strengths = np.array([rec.strength for rec in records], dtype=float)
            corrs = np.array([rec.correlation for rec in records], dtype=float)
            lags = np.array([rec.lag for rec in records], dtype=float)

            support = len(records)
            support_ratio = support / max(n_segments, 1)

            sign_consistency = self._dominant_sign_ratio(corrs, np.ones_like(corrs))

            from_dim, to_dim = key
            aggregated[key] = EnsembleLink(
                from_dim=from_dim,
                to_dim=to_dim,
                from_name=dimension_names[from_dim],
                to_name=dimension_names[to_dim],
                link_type=link_type,
                strength=float(np.mean(strengths)) if len(strengths) > 0 else 0.0,
                correlation=float(np.mean(corrs)) if len(corrs) > 0 else 0.0,
                lag=float(np.mean(lags)) if link_type == "causal" and len(lags) > 0 else 0.0,
                support=support,
                support_ratio=float(support_ratio),
                scale_support=1,
                scale_support_ratio=1.0,
                strength_std=float(np.std(strengths)) if len(strengths) > 1 else 0.0,
                correlation_std=float(np.std(corrs)) if len(corrs) > 1 else 0.0,
                lag_std=float(np.std(lags)) if link_type == "causal" and len(lags) > 1 else 0.0,
                sign_consistency=float(sign_consistency),
                direction_confidence=1.0,
                scales=[analysis_window],
                scale_labels=[f"w={analysis_window}"],
            )

        if link_type == "causal":
            for key, link in aggregated.items():
                reverse = aggregated.get((key[1], key[0]))
                reverse_support = reverse.support if reverse is not None else 0
                total = link.support + reverse_support
                link.direction_confidence = float(link.support / total) if total > 0 else 1.0

        return sorted(
            aggregated.values(),
            key=lambda lnk: (lnk.support_ratio, lnk.strength),
            reverse=True,
        )

    def _aggregate_multiscale_result(
        self,
        scale_results: list[ScaleEnsembleResult],
        dimension_names: list[str],
        n_dims: int,
        min_scale_support_ratio: float,
        min_segment_support_ratio: float,
        adaptive_params: dict[str, Any] | None,
        total_segments: int,
    ) -> MultiScaleNetworkResult:
        """
        Aggregate scale-specific ensembles into a global multi-scale ensemble.
        """
        sync_network = self._aggregate_links_across_scales(
            scale_results=scale_results,
            dimension_names=dimension_names,
            link_type="sync",
            total_segments=total_segments,
        )
        causal_network = self._aggregate_links_across_scales(
            scale_results=scale_results,
            dimension_names=dimension_names,
            link_type="causal",
            total_segments=total_segments,
        )

        robust_sync_network = [
            link
            for link in sync_network
            if link.scale_support_ratio >= min_scale_support_ratio
            and link.support_ratio >= min_segment_support_ratio
            and link.sign_consistency >= 0.60
        ]
        robust_causal_network = [
            link
            for link in causal_network
            if link.scale_support_ratio >= min_scale_support_ratio
            and link.support_ratio >= min_segment_support_ratio
            and link.direction_confidence >= 0.60
        ]

        scale_specific_sync_network = [
            link
            for link in sync_network
            if link.scale_support == 1 and link.support_ratio >= min_segment_support_ratio / 2.0
        ]
        scale_specific_causal_network = [
            link
            for link in causal_network
            if link.scale_support == 1
            and link.direction_confidence >= 0.60
            and link.support_ratio >= min_segment_support_ratio / 2.0
        ]

        topology_sync = robust_sync_network if robust_sync_network else sync_network
        topology_causal = robust_causal_network if robust_causal_network else causal_network

        pattern = self._identify_pattern(topology_sync, topology_causal)
        hub_dims = self._detect_hubs(topology_sync, topology_causal, n_dims)
        drivers, followers = self._identify_causal_structure(topology_causal, n_dims)

        return MultiScaleNetworkResult(
            scale_results=scale_results,
            sync_network=sync_network,
            causal_network=causal_network,
            robust_sync_network=robust_sync_network,
            robust_causal_network=robust_causal_network,
            scale_specific_sync_network=scale_specific_sync_network,
            scale_specific_causal_network=scale_specific_causal_network,
            pattern=pattern,
            hub_dimensions=hub_dims,
            hub_names=[dimension_names[d] for d in hub_dims],
            causal_drivers=drivers,
            causal_followers=followers,
            driver_names=[dimension_names[d] for d in drivers],
            follower_names=[dimension_names[d] for d in followers],
            n_dims=n_dims,
            n_scales=len(scale_results),
            total_segments=total_segments,
            dimension_names=dimension_names,
            adaptive_params=adaptive_params,
        )

    def _aggregate_links_across_scales(
        self,
        scale_results: list[ScaleEnsembleResult],
        dimension_names: list[str],
        link_type: str,
        total_segments: int,
    ) -> list[EnsembleLink]:
        """
        Aggregate scale-specific ensemble links into a global ensemble graph.

        Each scale contributes with weight equal to the edge support ratio inside
        that scale, so short-window scales with many segments do not dominate
        solely because they generate more segment-level samples.
        """
        buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
        n_scales = len(scale_results)

        for scale_result in scale_results:
            links = scale_result.sync_network if link_type == "sync" else scale_result.causal_network
            for link in links:
                if link_type == "sync":
                    key = self._canonical_sync_key(link.from_dim, link.to_dim)
                else:
                    key = (link.from_dim, link.to_dim)

                buckets.setdefault(key, []).append(
                    {
                        "strength": link.strength,
                        "correlation": link.correlation,
                        "lag": link.lag,
                        "support": link.support,
                        "support_ratio": link.support_ratio,
                        "analysis_window": scale_result.analysis_window,
                    }
                )

        aggregated: dict[tuple[int, int], EnsembleLink] = {}
        for key, records in buckets.items():
            support = int(sum(rec["support"] for rec in records))
            scale_support = len(records)
            support_ratio = float(support / max(total_segments, 1))
            scale_support_ratio = float(scale_support / max(n_scales, 1))

            weights = np.array([rec["support_ratio"] for rec in records], dtype=float)
            if np.sum(weights) < 1e-10:
                weights = np.ones_like(weights)

            strengths = np.array([rec["strength"] for rec in records], dtype=float)
            corrs = np.array([rec["correlation"] for rec in records], dtype=float)
            lags = np.array([rec["lag"] for rec in records], dtype=float)
            windows = [int(rec["analysis_window"]) for rec in records]

            strength_mean, strength_std = self._weighted_mean_std(strengths, weights)
            corr_mean, corr_std = self._weighted_mean_std(corrs, weights)
            lag_mean, lag_std = self._weighted_mean_std(lags, weights)

            sign_consistency = self._dominant_sign_ratio(corrs, weights)

            from_dim, to_dim = key
            aggregated[key] = EnsembleLink(
                from_dim=from_dim,
                to_dim=to_dim,
                from_name=dimension_names[from_dim],
                to_name=dimension_names[to_dim],
                link_type=link_type,
                strength=float(strength_mean),
                correlation=float(corr_mean),
                lag=float(lag_mean) if link_type == "causal" else 0.0,
                support=support,
                support_ratio=support_ratio,
                scale_support=scale_support,
                scale_support_ratio=scale_support_ratio,
                strength_std=float(strength_std),
                correlation_std=float(corr_std),
                lag_std=float(lag_std) if link_type == "causal" else 0.0,
                sign_consistency=float(sign_consistency),
                direction_confidence=1.0,
                scales=sorted(set(windows)),
                scale_labels=[f"w={w}" for w in sorted(set(windows))],
            )

        if link_type == "causal":
            for key, link in aggregated.items():
                reverse = aggregated.get((key[1], key[0]))
                reverse_support = reverse.support if reverse is not None else 0
                total_direction_support = link.support + reverse_support
                link.direction_confidence = (
                    float(link.support / total_direction_support)
                    if total_direction_support > 0
                    else 1.0
                )

        return sorted(
            aggregated.values(),
            key=lambda lnk: (lnk.scale_support_ratio, lnk.support_ratio, lnk.strength),
            reverse=True,
        )

    # ========================================================================
    # Pattern / Hub / Role Analysis
    # ========================================================================

    @staticmethod
    def _identify_pattern(
        sync_links: list[DimensionLink | EnsembleLink],
        causal_links: list[DimensionLink | EnsembleLink],
    ) -> str:
        """Identify a coarse-grained network pattern."""
        n_sync = len(sync_links)
        n_causal = len(causal_links)

        if n_sync == 0 and n_causal == 0:
            return "independent"
        if n_sync > n_causal * 2:
            return "parallel"
        if n_causal > n_sync * 2:
            return "cascade"
        return "mixed"

    @staticmethod
    def _detect_hubs(
        sync_links: list[DimensionLink | EnsembleLink],
        causal_links: list[DimensionLink | EnsembleLink],
        n_dims: int,
    ) -> list[int]:
        """Detect hub dimensions using weighted connectivity."""
        connectivity = np.zeros(n_dims, dtype=float)

        for link in list(sync_links) + list(causal_links):
            connectivity[link.from_dim] += link.strength
            connectivity[link.to_dim] += link.strength

        if np.max(connectivity) <= 0:
            return []

        threshold = float(np.mean(connectivity) + np.std(connectivity))
        hubs = np.where(connectivity > threshold)[0].tolist()
        return sorted(hubs, key=lambda idx: connectivity[idx], reverse=True)

    @staticmethod
    def _identify_causal_structure(
        causal_links: list[DimensionLink | EnsembleLink],
        n_dims: int,
    ) -> tuple[list[int], list[int]]:
        """Identify likely causal drivers and followers."""
        out_degree = np.zeros(n_dims, dtype=float)
        in_degree = np.zeros(n_dims, dtype=float)

        for link in causal_links:
            out_degree[link.from_dim] += link.strength
            in_degree[link.to_dim] += link.strength

        drivers: list[int] = []
        followers: list[int] = []

        for d in range(n_dims):
            if out_degree[d] > 0 and out_degree[d] > in_degree[d] * 1.5:
                drivers.append(d)
            elif in_degree[d] > 0 and in_degree[d] > out_degree[d] * 1.5:
                followers.append(d)

        return (
            sorted(drivers, key=lambda d: out_degree[d], reverse=True),
            sorted(followers, key=lambda d: in_degree[d], reverse=True),
        )

    # ========================================================================
    # Event Analysis Helpers
    # ========================================================================

    @staticmethod
    def _identify_initiators(
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """
        Estimate which dimensions initiated a cooperative event.
        """
        start = max(0, event_frame - lookback)
        pre_event = state_vectors[start : event_frame + 1]

        if len(pre_event) < 3:
            return []

        displacement = np.diff(pre_event, axis=0)
        scores = np.zeros(n_dims)

        for d in range(n_dims):
            abs_disp = np.abs(displacement[:, d])
            weights = np.linspace(0.5, 2.0, len(abs_disp))
            scores[d] = np.sum(abs_disp * weights)

        threshold = float(np.mean(scores) + np.std(scores))
        initiators = np.where(scores > threshold)[0]
        return sorted(initiators, key=lambda d: scores[d], reverse=True)

    @staticmethod
    def _estimate_propagation_order(
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """
        Estimate propagation order via onset threshold crossing.
        """
        start = max(0, event_frame - lookback)
        window = state_vectors[start : event_frame + 1]

        if len(window) < 3:
            return list(range(n_dims))

        displacement = np.abs(np.diff(window, axis=0))
        onset_frames = np.full(n_dims, len(displacement), dtype=int)

        for d in range(n_dims):
            series = displacement[:, d]
            threshold = float(np.mean(series) + 1.5 * np.std(series))
            exceeding = np.where(series > threshold)[0]
            if len(exceeding) > 0:
                onset_frames[d] = int(exceeding[0])

        return list(np.argsort(onset_frames))

    # ========================================================================
    # Utility Helpers
    # ========================================================================

    @staticmethod
    def _resolve_dimension_names(
        dimension_names: list[str] | None,
        n_dims: int,
    ) -> list[str]:
        """Resolve dimension names."""
        if dimension_names is None:
            return [f"dim_{i}" for i in range(n_dims)]
        if len(dimension_names) != n_dims:
            raise ValueError(
                f"dimension_names length ({len(dimension_names)}) "
                f"does not match n_dims ({n_dims})"
            )
        return dimension_names

    @staticmethod
    def _select_recent_segment(
        state_vectors: np.ndarray,
        window: int,
    ) -> tuple[np.ndarray, int, int]:
        """
        Select the most recent segment of length `window`.
        """
        n_frames = len(state_vectors)
        if window >= n_frames:
            return state_vectors, 0, n_frames
        start = n_frames - window
        end = n_frames
        return state_vectors[start:end], start, end

    @staticmethod
    def _generate_sliding_segments(
        n_frames: int,
        window_size: int,
        stride: int,
    ) -> list[tuple[int, int]]:
        """
        Generate overlapping temporal segments.
        """
        window_size = int(np.clip(window_size, 3, max(3, n_frames)))
        stride = max(1, int(stride))

        if window_size >= n_frames:
            return [(0, n_frames)]

        starts = list(range(0, n_frames - window_size + 1, stride))
        if starts[-1] != n_frames - window_size:
            starts.append(n_frames - window_size)

        starts = sorted(set(starts))
        return [(start, start + window_size) for start in starts]

    def _resolve_multiscale_windows(
        self,
        n_frames: int,
        base_window: int,
        adaptive_windows: list[int] | None = None,
        scales: tuple[float, ...] | None = None,
    ) -> list[int]:
        """
        Resolve concrete multi-scale window sizes.
        """
        if scales is None and adaptive_windows is not None:
            windows = [int(np.clip(w, 12, n_frames)) for w in adaptive_windows]
            return sorted(set(windows))

        base_window = int(np.clip(base_window, 12, n_frames))
        scales = self.multiscale_scales if scales is None else tuple(sorted(set(scales)))

        windows = [
            int(np.clip(round(base_window * scale), 12, n_frames))
            for scale in scales
        ]
        windows.append(base_window)
        if n_frames > base_window and n_frames >= max(48, int(round(base_window * 1.25))):
            windows.append(n_frames)
        return sorted(set(windows))

    @staticmethod
    def _window_to_scale_factors(
        windows: list[int],
        base_window: int,
    ) -> list[float]:
        """Convert window sizes into relative scale factors."""
        base_window = max(1, int(base_window))
        return [float(window / base_window) for window in windows]

    @staticmethod
    def _derive_scale_runtime_parameters(
        window_size: int,
        base_window: int,
        base_local_std_window: int,
        base_max_lag: int,
        base_sync_threshold: float,
        base_causal_threshold: float,
    ) -> dict[str, Any]:
        """
        Derive runtime parameters for a specific scale.

        Shorter windows become slightly stricter because estimates are noisier.
        Larger windows relax thresholds slightly and permit a longer lag range.
        """
        scale_ratio = window_size / max(base_window, 1)

        if scale_ratio < 1.0:
            threshold_adj = 0.05
        elif scale_ratio > 2.0:
            threshold_adj = -0.02
        else:
            threshold_adj = 0.0

        sync_threshold = float(np.clip(base_sync_threshold + threshold_adj, 0.15, 0.90))
        causal_threshold = float(np.clip(base_causal_threshold + threshold_adj, 0.15, 0.85))

        local_std_window = int(
            np.clip(
                round(base_local_std_window * np.sqrt(scale_ratio)),
                5,
                max(5, window_size // 2),
            )
        )

        max_lag = int(
            np.clip(
                round(base_max_lag * np.sqrt(scale_ratio)),
                1,
                max(1, window_size // 4),
            )
        )

        return {
            "sync_threshold": sync_threshold,
            "causal_threshold": causal_threshold,
            "local_std_window": local_std_window,
            "max_lag": max_lag,
        }

    @staticmethod
    def _canonical_sync_key(i: int, j: int) -> tuple[int, int]:
        """Canonical key for undirected synchronization edges."""
        return (i, j) if i < j else (j, i)

    @staticmethod
    def _weighted_mean_std(
        values: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """
        Compute weighted mean and weighted standard deviation.
        """
        if len(values) == 0:
            return 0.0, 0.0
        weights = np.asarray(weights, dtype=float)
        values = np.asarray(values, dtype=float)

        if np.sum(weights) <= 1e-10:
            weights = np.ones_like(values)

        mean = float(np.average(values, weights=weights))
        var = float(np.average((values - mean) ** 2, weights=weights))
        return mean, float(np.sqrt(max(var, 0.0)))

    @staticmethod
    def _dominant_sign_ratio(values: np.ndarray, weights: np.ndarray) -> float:
        """
        Ratio of support carried by the dominant sign.
        """
        if len(values) == 0:
            return 1.0

        weights = np.asarray(weights, dtype=float)
        if np.sum(weights) <= 1e-10:
            weights = np.ones_like(values, dtype=float)

        pos = float(np.sum(weights[values > 0]))
        neg = float(np.sum(weights[values < 0]))
        zero = float(np.sum(weights[values == 0]))
        total = pos + neg + zero
        if total <= 1e-10:
            return 1.0

        dominant = max(pos, neg, zero)
        return float(dominant / total)

    # ========================================================================
    # Logging
    # ========================================================================

    @staticmethod
    def _print_summary(result: NetworkResult) -> None:
        """Print a compact summary for single-scale analysis."""
        logger.info("=" * 72)
        logger.info("Single-Scale Network Analysis Summary")
        logger.info("=" * 72)
        logger.info("Pattern: %s", result.pattern)
        logger.info("Segment: [%s:%s]", result.segment_start, result.segment_end)
        logger.info("Analysis window: %d", result.analysis_window)
        logger.info("Local std window: %d", result.local_std_window)
        logger.info("Max lag: %d", result.max_lag)
        logger.info("Sync links: %d", result.n_sync_links)
        logger.info("Causal links: %d", result.n_causal_links)

        if result.hub_names:
            logger.info("Hub dimensions: %s", ", ".join(result.hub_names))
        if result.driver_names:
            logger.info("Causal drivers: %s", ", ".join(result.driver_names))
        if result.follower_names:
            logger.info("Causal followers: %s", ", ".join(result.follower_names))

        if result.sync_network:
            logger.info("Sync Network:")
            for link in sorted(result.sync_network, key=lambda lnk: lnk.strength, reverse=True):
                sign = "+" if link.correlation > 0 else "-"
                logger.info(
                    "  %s <-> %s: %s%.3f",
                    link.from_name,
                    link.to_name,
                    sign,
                    link.strength,
                )

        if result.causal_network:
            logger.info("Causal Network:")
            for link in sorted(result.causal_network, key=lambda lnk: lnk.strength, reverse=True):
                logger.info(
                    "  %s -> %s: %.3f (lag=%d)",
                    link.from_name,
                    link.to_name,
                    link.strength,
                    link.lag,
                )

    @staticmethod
    def _print_multiscale_summary(result: MultiScaleNetworkResult) -> None:
        """Print a compact summary for multi-scale ensemble analysis."""
        logger.info("=" * 72)
        logger.info("Multi-Scale Ensemble Network Analysis Summary")
        logger.info("=" * 72)
        logger.info("Pattern: %s", result.pattern)
        logger.info("Scales: %d", result.n_scales)
        logger.info("Total segments: %d", result.total_segments)
        logger.info("Aggregated sync links: %d", len(result.sync_network))
        logger.info("Aggregated causal links: %d", len(result.causal_network))
        logger.info("Robust sync links: %d", len(result.robust_sync_network))
        logger.info("Robust causal links: %d", len(result.robust_causal_network))
        logger.info("Scale-specific sync links: %d", len(result.scale_specific_sync_network))
        logger.info("Scale-specific causal links: %d", len(result.scale_specific_causal_network))

        if result.hub_names:
            logger.info("Hub dimensions: %s", ", ".join(result.hub_names))
        if result.driver_names:
            logger.info("Causal drivers: %s", ", ".join(result.driver_names))
        if result.follower_names:
            logger.info("Causal followers: %s", ", ".join(result.follower_names))

        if result.robust_sync_network:
            logger.info("Robust Sync Network:")
            for link in sorted(
                result.robust_sync_network,
                key=lambda lnk: (lnk.scale_support_ratio, lnk.support_ratio, lnk.strength),
                reverse=True,
            ):
                sign = "+" if link.correlation > 0 else "-"
                logger.info(
                    "  %s <-> %s: %s%.3f | support=%.2f | scales=%s",
                    link.from_name,
                    link.to_name,
                    sign,
                    link.strength,
                    link.support_ratio,
                    link.scale_labels,
                )

        if result.robust_causal_network:
            logger.info("Robust Causal Network:")
            for link in sorted(
                result.robust_causal_network,
                key=lambda lnk: (lnk.scale_support_ratio, lnk.support_ratio, lnk.strength),
                reverse=True,
            ):
                logger.info(
                    "  %s -> %s: %.3f | lag=%.2f | support=%.2f | dir_conf=%.2f | scales=%s",
                    link.from_name,
                    link.to_name,
                    link.strength,
                    link.lag,
                    link.support_ratio,
                    link.direction_confidence,
                    link.scale_labels,
                )
