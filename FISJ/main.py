"""
Network Analyzer Core - Domain-Agnostic (Standalone)
====================================================
Built by Masamichi & Tamaki

A standalone multidimensional network analysis engine based on
partial correlation, lagged causality estimation, and common-ancestor
filtering.

This module detects synchronization and causal structure across
dimensions in multivariate time-series data and returns the results
in a network-friendly representation.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("getter_one.analysis.network_analyzer_core")


# ============================================
# Data Classes
# ============================================


@dataclass
class DimensionLink:
    """A network link between two dimensions."""

    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    link_type: str  # 'sync' or 'causal'
    strength: float  # absolute magnitude of the correlation
    correlation: float  # signed correlation value
    lag: int = 0  # lag in frames for causal links


@dataclass
class NetworkResult:
    """Container for full network analysis results."""

    sync_network: list[DimensionLink] = field(default_factory=list)
    causal_network: list[DimensionLink] = field(default_factory=list)

    # Raw correlation matrices
    sync_matrix: np.ndarray | None = None  # (n_dims, n_dims)
    causal_matrix: np.ndarray | None = None  # (n_dims, n_dims), max lagged corr
    causal_lag_matrix: np.ndarray | None = None  # (n_dims, n_dims), optimal lag

    # Network-level characteristics
    pattern: str = "unknown"  # 'parallel', 'cascade', 'mixed', 'independent'
    hub_dimensions: list[int] = field(default_factory=list)
    hub_names: list[str] = field(default_factory=list)

    # Causal structure summary
    causal_drivers: list[int] = field(default_factory=list)
    causal_followers: list[int] = field(default_factory=list)
    driver_names: list[str] = field(default_factory=list)
    follower_names: list[str] = field(default_factory=list)

    # Metadata
    n_dims: int = 0
    n_sync_links: int = 0
    n_causal_links: int = 0
    dimension_names: list[str] = field(default_factory=list)

    # Adaptive analysis parameters, if adaptive mode is enabled
    adaptive_params: dict | None = None


@dataclass
class CooperativeEventNetwork:
    """Local network structure around a cooperative event."""

    event_frame: int
    event_timestamp: str | None = None
    delta_lambda_c: float = 0.0

    # Network snapshot around the event
    network: NetworkResult | None = None

    # Event-specific annotations
    initiator_dims: list[int] = field(default_factory=list)
    initiator_names: list[str] = field(default_factory=list)
    propagation_order: list[int] = field(default_factory=list)


# ============================================
# Network Analyzer Core
# ============================================


class NetworkAnalyzerCore:
    """
    Domain-agnostic multidimensional network analysis engine.

    This class detects synchronization and causal relationships among
    dimensions in an N-dimensional time series and exports the result
    in a form suitable for downstream visualization or structural analysis.

    Internally, the analyzer computes a rolling local standard deviation
    and uses it to normalize displacement signals, enabling scale-invariant
    partial correlation analysis without any external module dependency.

    Parameters
    ----------
    sync_threshold : float
        Threshold for synchronization links. If adaptive=True, this is used
        as a prior hint rather than a fixed cutoff.
    causal_threshold : float
        Threshold for causal links. If adaptive=True, this is used as a
        prior hint rather than a fixed cutoff.
    max_lag : int
        Maximum lag used for causal estimation. If adaptive=True, this is
        treated as a hint.
    adaptive : bool
        Whether to enable data-driven adaptive parameter tuning.
    local_std_window : int
        Rolling window size used to compute local standard deviation.
    """

    def __init__(
        self,
        sync_threshold: float = 0.5,
        causal_threshold: float = 0.4,
        max_lag: int = 12,
        adaptive: bool = True,
        local_std_window: int = 20,
    ):
        # User-provided hints (used as priors in adaptive mode)
        self.sync_threshold_hint = sync_threshold
        self.causal_threshold_hint = causal_threshold
        self.max_lag_hint = max_lag
        self.adaptive = adaptive
        self.local_std_window = local_std_window

        # Runtime parameters (possibly overwritten adaptively)
        self.sync_threshold = sync_threshold
        self.causal_threshold = causal_threshold
        self.max_lag = max_lag

        # Stores adaptive parameter estimation results
        self.adaptive_params: dict | None = None

        logger.info(
            f"✅ NetworkAnalyzerCore initialized "
            f"(sync>{sync_threshold}, causal>{causal_threshold}, "
            f"max_lag={max_lag}, adaptive={adaptive}, "
            f"local_std_window={local_std_window})"
        )

    # ================================================================
    # Internal: Local Std Computation
    # ================================================================

    def _compute_local_std(
        self,
        state_vectors: np.ndarray,
    ) -> np.ndarray:
        """
        Compute local standard deviation using a rolling window.

        For each frame and each dimension, the standard deviation is
        computed over a symmetric local window of size `local_std_window`.
        This gives a local scale estimate for each dimension and is later
        used to normalize displacement into a dimensionless signal.

        Parameters
        ----------
        state_vectors : np.ndarray of shape (n_frames, n_dims)
            Multidimensional input time series.

        Returns
        -------
        np.ndarray of shape (n_frames, n_dims)
            Local standard deviation at each frame and dimension.
        """
        n_frames, n_dims = state_vectors.shape
        w = self.local_std_window
        half_w = w // 2
        local_std = np.zeros_like(state_vectors)

        for d in range(n_dims):
            series = state_vectors[:, d]
            for t in range(n_frames):
                start = max(0, t - half_w)
                end = min(n_frames, t + half_w + 1)
                local_std[t, d] = np.std(series[start:end])

        return local_std

    # ================================================================
    # Adaptive Parameter Computation
    # ================================================================

    def _compute_adaptive_parameters(
        self,
        state_vectors: np.ndarray,
    ) -> dict:
        """
        Compute data-driven adaptive analysis parameters.

        The analyzer dynamically adjusts thresholds and maximum lag based on
        five signal characteristics:

        1. Global volatility
        2. Temporal volatility
        3. Inter-dimensional correlation complexity
        4. Local nonstationarity
        5. Spectral low-frequency dominance

        Parameters
        ----------
        state_vectors : np.ndarray of shape (n_frames, n_dims)
            Multidimensional input time series.

        Returns
        -------
        dict
            Dictionary containing:
            - sync_threshold
            - causal_threshold
            - max_lag
            - scale_factor
            - volatility_metrics
        """
        n_frames, n_dims = state_vectors.shape

        # 1. Global volatility
        global_std = np.std(state_vectors)
        global_mean = np.mean(np.abs(state_vectors))
        volatility_ratio = global_std / (global_mean + 1e-10)

        # 2. Temporal volatility (frame-to-frame change)
        temporal_changes = np.diff(state_vectors, axis=0)
        temporal_volatility = np.mean(np.std(temporal_changes, axis=0))

        # 3. Complexity of the inter-dimensional correlation structure
        if n_dims > 1:
            corr_matrix = np.corrcoef(state_vectors.T)
            triu = corr_matrix[np.triu_indices(n_dims, k=1)]
            triu = triu[~np.isnan(triu)]
            correlation_complexity = (
                1.0 - np.mean(np.abs(triu)) if len(triu) > 0 else 0.5
            )
            mean_abs_corr = np.mean(np.abs(triu)) if len(triu) > 0 else 0.0
        else:
            correlation_complexity = 0.5
            mean_abs_corr = 0.0

        # 4. Local volatility variation
        base_window = max(10, n_frames // 10)
        local_volatilities = []
        for i in range(0, n_frames - base_window, max(1, base_window // 2)):
            window_data = state_vectors[i : i + base_window]
            local_volatilities.append(np.std(window_data))

        if len(local_volatilities) > 1:
            volatility_variation = np.std(local_volatilities) / (
                np.mean(local_volatilities) + 1e-10
            )
        else:
            volatility_variation = 0.5

        # 5. Spectral structure (estimate dominant low-frequency behavior)
        fft_mag = np.abs(np.fft.fft(state_vectors, axis=0))
        low_cutoff = max(1, n_frames // 10)
        high_cutoff = max(2, n_frames // 2)
        low_freq_ratio = np.sum(fft_mag[:low_cutoff]) / (
            np.sum(fft_mag[:high_cutoff]) + 1e-10
        )

        # Adaptive sync_threshold
        sync_adj = 0.0
        if volatility_ratio > 2.0:
            sync_adj += 0.1
        elif volatility_ratio < 0.3:
            sync_adj -= 0.1

        if mean_abs_corr > 0.6:
            sync_adj += 0.1
        elif mean_abs_corr < 0.2:
            sync_adj -= 0.1

        if volatility_variation > 1.0:
            sync_adj += 0.05

        sync_threshold = np.clip(self.sync_threshold_hint + sync_adj, 0.15, 0.85)

        # Adaptive causal_threshold
        causal_adj = 0.0
        if temporal_volatility < global_std * 0.3:
            causal_adj -= 0.1
        elif temporal_volatility > global_std * 2.0:
            causal_adj += 0.1

        if n_dims > 50:
            causal_adj += 0.1
        elif n_dims <= 5:
            causal_adj -= 0.05

        if correlation_complexity > 0.7:
            causal_adj -= 0.05

        causal_threshold = np.clip(self.causal_threshold_hint + causal_adj, 0.15, 0.80)

        # Adaptive max_lag
        scale_factor = 1.0
        if low_freq_ratio > 0.8:
            scale_factor *= 1.5
        elif low_freq_ratio < 0.3:
            scale_factor *= 0.7

        if correlation_complexity > 0.7:
            scale_factor *= 1.3

        if temporal_volatility > global_std * 2.0:
            scale_factor *= 0.8

        raw_lag = int(self.max_lag_hint * scale_factor)
        max_lag = int(np.clip(raw_lag, 3, max(5, n_frames // 5)))

        params = {
            "sync_threshold": float(sync_threshold),
            "causal_threshold": float(causal_threshold),
            "max_lag": max_lag,
            "scale_factor": float(scale_factor),
            "volatility_metrics": {
                "global_volatility": float(volatility_ratio),
                "temporal_volatility": float(temporal_volatility),
                "correlation_complexity": float(correlation_complexity),
                "local_variation": float(volatility_variation),
                "low_freq_ratio": float(low_freq_ratio),
                "mean_abs_corr": float(mean_abs_corr),
            },
        }

        logger.info(
            f"   Adaptive params: "
            f"sync_th={sync_threshold:.3f} "
            f"(hint={self.sync_threshold_hint}), "
            f"causal_th={causal_threshold:.3f} "
            f"(hint={self.causal_threshold_hint}), "
            f"max_lag={max_lag} (hint={self.max_lag_hint})"
        )
        logger.info(
            f"   Volatility: global={volatility_ratio:.3f}, "
            f"temporal={temporal_volatility:.3f}, "
            f"corr_complexity={correlation_complexity:.3f}, "
            f"local_var={volatility_variation:.3f}, "
            f"low_freq={low_freq_ratio:.3f}, "
            f"mean_corr={mean_abs_corr:.3f}"
        )

        return params

    # ================================================================
    # Main Entry Point
    # ================================================================

    def analyze(
        self,
        state_vectors: np.ndarray,
        dimension_names: list[str] | None = None,
        window: int | None = None,
    ) -> NetworkResult:
        """
        Run full-network analysis on a multidimensional time series.

        Parameters
        ----------
        state_vectors : np.ndarray of shape (n_frames, n_dims)
            Input state-vector time series.
        dimension_names : list[str], optional
            Human-readable names for each dimension.
        window : int, optional
            Window length used for correlation analysis. If None, all frames
            are used.

        Returns
        -------
        NetworkResult
            Full network analysis result.
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        if window is None:
            window = n_frames

        # Compute adaptive parameters if enabled
        if self.adaptive:
            self.adaptive_params = self._compute_adaptive_parameters(state_vectors)
            self.sync_threshold = self.adaptive_params["sync_threshold"]
            self.causal_threshold = self.adaptive_params["causal_threshold"]
            self.max_lag = self.adaptive_params["max_lag"]

        logger.info(
            f"🔍 Analyzing {n_dims}-dimensional network "
            f"({n_frames} frames, window={window}, "
            f"sync>{self.sync_threshold:.3f}, "
            f"causal>{self.causal_threshold:.3f}, "
            f"max_lag={self.max_lag})"
        )

        # 1. Compute local standard deviation internally
        local_std = self._compute_local_std(state_vectors)

        # 2. Compute correlations with local-std-based normalization
        correlations = self._compute_correlations(state_vectors, window, local_std)

        # 3. Build raw network structures
        sync_links, causal_links = self._build_networks(correlations, dimension_names)

        # 3.5 Remove spurious causal edges induced by common ancestors
        causal_links = self._filter_spurious_edges(causal_links)

        # 4. Identify global network pattern
        pattern = self._identify_pattern(sync_links, causal_links)

        # 5. Detect hub dimensions
        hub_dims = self._detect_hubs(sync_links, causal_links, n_dims)

        # 6. Infer causal role structure (drivers / followers)
        drivers, followers = self._identify_causal_structure(causal_links, n_dims)

        result = NetworkResult(
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
            adaptive_params=self.adaptive_params,
        )

        self._print_summary(result)
        return result

    def analyze_event_network(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        window_before: int = 24,
        window_after: int = 6,
        dimension_names: list[str] | None = None,
    ) -> CooperativeEventNetwork:
        """
        Analyze the local network structure around a cooperative event.

        This method examines causal organization in a window around the event
        and estimates which dimensions most likely initiated the event.

        Parameters
        ----------
        state_vectors : np.ndarray
            Input state-vector time series.
        event_frame : int
            Frame index at which the event occurs.
        window_before : int
            Number of frames to include before the event.
        window_after : int
            Number of frames to include after the event.
        dimension_names : list[str], optional
            Human-readable names for each dimension.

        Returns
        -------
        CooperativeEventNetwork
            Local network snapshot and event-specific annotations.
        """
        n_frames, n_dims = state_vectors.shape

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        start = max(0, event_frame - window_before)
        end = min(n_frames, event_frame + window_after)
        local_data = state_vectors[start:end]

        # Local network analysis
        network = self.analyze(local_data, dimension_names, window=len(local_data))

        # Estimate event initiators
        initiators = self._identify_initiators(
            state_vectors, event_frame, window_before, n_dims
        )

        # Estimate propagation order
        propagation = self._estimate_propagation_order(
            state_vectors, event_frame, window_before, n_dims
        )

        return CooperativeEventNetwork(
            event_frame=event_frame,
            network=network,
            initiator_dims=initiators,
            initiator_names=[dimension_names[d] for d in initiators],
            propagation_order=propagation,
        )

    # ================================================================
    # Correlation Computation
    # ================================================================

    def _compute_correlations(
        self,
        state_vectors: np.ndarray,
        window: int,
        local_std: np.ndarray,
    ) -> dict:
        """
        Compute synchronization and causal correlations for all dimension pairs.

        For n_dims >= 3, partial correlation is used to suppress confounding
        effects. For n_dims < 3, the method falls back to pairwise correlation.

        Displacement is internally normalized by local standard deviation,
        yielding scale-invariant partial correlation estimates.

        Parameters
        ----------
        state_vectors : np.ndarray
            Input state-vector time series.
        window : int
            Number of frames to use.
        local_std : np.ndarray
            Precomputed local standard deviation.

        Returns
        -------
        dict
            Dictionary with:
            - "sync": synchronous correlation matrix
            - "max_lagged": strongest lagged correlation matrix
            - "best_lag": matrix of optimal lags
        """
        n_frames, n_dims = state_vectors.shape
        w = min(window, n_frames)

        sync_matrix = np.zeros((n_dims, n_dims))
        max_lagged_matrix = np.zeros((n_dims, n_dims))
        best_lag_matrix = np.zeros((n_dims, n_dims), dtype=int)

        # First-order displacement vectors
        raw_displacement = np.diff(state_vectors[:w], axis=0)

        # Normalize by local standard deviation
        local_std_diff = local_std[1:w]
        displacement = raw_displacement / (local_std_diff + 1e-10)
        logger.info("   📐 Using dimensionless displacement (internal local_std)")

        use_partial = n_dims >= 3

        # Synchronous partial correlation using a precision-matrix approach
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
                    if np.isnan(corr):
                        corr = 0.0
                    sync_matrix[i, j] = corr
                    sync_matrix[j, i] = corr

        # Lagged partial correlation for causal estimation
        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                best_corr = 0.0
                best_lag = 0

                for lag in range(1, min(self.max_lag + 1, len(displacement) - 1)):
                    # i -> j (i leads by `lag` frames)
                    corr_ij = self._lagged_partial_corr(
                        displacement, i, j, lag, use_partial
                    )
                    if abs(corr_ij) > abs(best_corr):
                        best_corr = corr_ij
                        best_lag = lag

                    # j -> i (j leads by `lag` frames)
                    corr_ji = self._lagged_partial_corr(
                        displacement, j, i, lag, use_partial
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
        Compute the partial correlation matrix from the precision matrix.

        The relation used is:

            pcorr(i, j) = -P[i, j] / sqrt(P[i, i] * P[j, j])

        where P is the inverse covariance matrix.

        Parameters
        ----------
        data : np.ndarray
            Input data matrix of shape (n_samples, n_dims).

        Returns
        -------
        np.ndarray
            Partial correlation matrix of shape (n_dims, n_dims).
        """
        n_dims = data.shape[1]
        cov = np.cov(data.T)

        # Regularization for near-singular covariance matrices
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
                pc = np.clip(pc, -1.0, 1.0)
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
        Compute lagged partial correlation using a multi-lag residual approach.

        When estimating src -> dst with a given lag, this method removes the
        effects of all other dimensions at multiple lag values. This helps
        suppress indirect lag-mediated confounding.

        Parameters
        ----------
        displacement : np.ndarray
            Dimensionless displacement signal.
        src : int
            Source dimension index.
        dst : int
            Destination dimension index.
        lag : int
            Positive lag in frames.
        use_partial : bool
            Whether to use partial conditioning or fall back to pairwise
            correlation.

        Returns
        -------
        float
            Lagged partial correlation value.
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

        # Condition on all other dimensions across multiple lag offsets
        other_dims = [d for d in range(n_dims) if d != src and d != dst]
        if not other_dims:
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)

        max_cond_lag = min(lag + 2, n - 1)
        z_parts = []
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

        # Remove nearly constant conditioning columns
        z_std = np.std(z, axis=0)
        valid_cols = z_std > 1e-10
        if not np.any(valid_cols):
            corr = np.corrcoef(ts_src, ts_dst)[0, 1]
            return 0.0 if np.isnan(corr) else float(corr)
        z = z[:, valid_cols]

        # Reduce dimensionality if conditioning variables are too collinear
        if z.shape[1] > z.shape[0] // 2:
            try:
                u, s, _ = np.linalg.svd(z, full_matrices=False)
                cumvar = np.cumsum(s**2) / np.sum(s**2)
                n_keep = max(1, int(np.searchsorted(cumvar, 0.95)) + 1)
                z = u[:, :n_keep] * s[:n_keep]
            except np.linalg.LinAlgError:
                pass

        # Regress out the conditioning variables
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

    # ================================================================
    # Network Construction
    # ================================================================

    def _build_networks(
        self,
        correlations: dict,
        dimension_names: list[str],
    ) -> tuple[list[DimensionLink], list[DimensionLink]]:
        """
        Build synchronization and causal links from correlation matrices.
        """
        n_dims = len(dimension_names)
        sync_links = []
        causal_links = []

        for i in range(n_dims):
            for j in range(i + 1, n_dims):
                sync_corr = correlations["sync"][i, j]
                causal_corr = correlations["max_lagged"][i, j]
                lag = correlations["best_lag"][i, j]

                # Synchronization link
                if abs(sync_corr) > self.sync_threshold:
                    sync_links.append(
                        DimensionLink(
                            from_dim=i,
                            to_dim=j,
                            from_name=dimension_names[i],
                            to_name=dimension_names[j],
                            link_type="sync",
                            strength=abs(sync_corr),
                            correlation=sync_corr,
                        )
                    )

                # Causal link
                # Only accept causality if lagged correlation is meaningfully
                # stronger than synchronous correlation
                if (
                    abs(causal_corr) > self.causal_threshold
                    and abs(causal_corr) > abs(sync_corr) * 1.1
                ):
                    # Determine direction from the lag sign
                    if lag > 0:
                        from_d, to_d = i, j
                    else:
                        from_d, to_d = j, i
                        lag = abs(lag)

                    causal_links.append(
                        DimensionLink(
                            from_dim=from_d,
                            to_dim=to_d,
                            from_name=dimension_names[from_d],
                            to_name=dimension_names[to_d],
                            link_type="causal",
                            strength=abs(causal_corr),
                            correlation=causal_corr,
                            lag=lag,
                        )
                    )

        return sync_links, causal_links

    def _filter_spurious_edges(
        self,
        causal_links: list[DimensionLink],
    ) -> list[DimensionLink]:
        """
        Remove spurious causal edges induced by common ancestors.

        If A -> B is detected, but there exists a common ancestor Z such that
        both Z -> A and Z -> B exist and Z -> A is stronger than A -> B,
        then A -> B is treated as a likely confounded edge and removed.
        """
        if len(causal_links) < 3:
            return causal_links

        # Fast lookup map: (from_dim, to_dim) -> strongest link
        link_map: dict[tuple[int, int], DimensionLink] = {}
        for link in causal_links:
            key = (link.from_dim, link.to_dim)
            if key not in link_map or link.strength > link_map[key].strength:
                link_map[key] = link

        filtered = []
        n_removed = 0

        for link in causal_links:
            a, b = link.from_dim, link.to_dim
            has_common_ancestor = False

            for (z_src, z_dst), z_link_a in link_map.items():
                if z_dst != a:
                    continue
                z = z_src
                if z in (a, b):
                    continue

                z_link_b = link_map.get((z, b))
                if z_link_b is None:
                    continue

                if z_link_a.strength > link.strength:
                    has_common_ancestor = True
                    logger.debug(
                        f"   🔍 Spurious edge removed: "
                        f"{link.from_name}→{link.to_name} "
                        f"(confounder: {z_link_a.from_name}, "
                        f"strength {link.strength:.3f} < "
                        f"{z_link_a.strength:.3f}, {z_link_b.strength:.3f})"
                    )
                    break

            if has_common_ancestor:
                n_removed += 1
            else:
                filtered.append(link)

        if n_removed > 0:
            logger.info(
                f"   🔍 Spurious edge filter: {n_removed} removed, "
                f"{len(filtered)} retained"
            )

        return filtered

    # ================================================================
    # Pattern Identification / Hub Detection / Causal Roles
    # ================================================================

    def _identify_pattern(
        self,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
    ) -> str:
        """Identify the coarse-grained network pattern."""
        n_sync = len(sync_links)
        n_causal = len(causal_links)

        if n_sync == 0 and n_causal == 0:
            return "independent"
        elif n_sync > n_causal * 2:
            return "parallel"
        elif n_causal > n_sync * 2:
            return "cascade"
        else:
            return "mixed"

    def _detect_hubs(
        self,
        sync_links: list[DimensionLink],
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> list[int]:
        """
        Detect hub dimensions based on aggregate weighted connectivity.
        """
        connectivity = np.zeros(n_dims)

        for link in sync_links + causal_links:
            connectivity[link.from_dim] += link.strength
            connectivity[link.to_dim] += link.strength

        if np.max(connectivity) == 0:
            return []

        threshold = np.mean(connectivity) + np.std(connectivity)
        hubs = np.where(connectivity > threshold)[0].tolist()

        return sorted(hubs, key=lambda d: connectivity[d], reverse=True)

    def _identify_causal_structure(
        self,
        causal_links: list[DimensionLink],
        n_dims: int,
    ) -> tuple[list[int], list[int]]:
        """
        Identify causal driver and follower dimensions.
        """
        out_degree = np.zeros(n_dims)
        in_degree = np.zeros(n_dims)

        for link in causal_links:
            out_degree[link.from_dim] += link.strength
            in_degree[link.to_dim] += link.strength

        drivers = []
        followers = []

        for d in range(n_dims):
            if out_degree[d] > 0 and out_degree[d] > in_degree[d] * 1.5:
                drivers.append(d)
            elif in_degree[d] > 0 and in_degree[d] > out_degree[d] * 1.5:
                followers.append(d)

        return (
            sorted(drivers, key=lambda d: out_degree[d], reverse=True),
            sorted(followers, key=lambda d: in_degree[d], reverse=True),
        )

    # ================================================================
    # Event Analysis Helpers
    # ================================================================

    def _identify_initiators(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """
        Estimate which dimensions initiated a cooperative event.

        The method scores dimensions by how early and strongly they begin
        moving within the pre-event window.
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

        threshold = np.mean(scores) + np.std(scores)
        initiators = np.where(scores > threshold)[0]

        return sorted(initiators, key=lambda d: scores[d], reverse=True)

    def _estimate_propagation_order(
        self,
        state_vectors: np.ndarray,
        event_frame: int,
        lookback: int,
        n_dims: int,
    ) -> list[int]:
        """
        Estimate event propagation order across dimensions.

        Dimensions are ordered by the first frame at which their displacement
        exceeds a dimension-specific onset threshold.
        """
        start = max(0, event_frame - lookback)
        window = state_vectors[start : event_frame + 1]

        if len(window) < 3:
            return list(range(n_dims))

        displacement = np.abs(np.diff(window, axis=0))

        onset_frames = np.full(n_dims, len(displacement))

        for d in range(n_dims):
            series = displacement[:, d]
            threshold = np.mean(series) + 1.5 * np.std(series)

            exceeding = np.where(series > threshold)[0]
            if len(exceeding) > 0:
                onset_frames[d] = exceeding[0]

        return list(np.argsort(onset_frames))

    # ================================================================
    # Output
    # ================================================================

    def _print_summary(self, result: NetworkResult) -> None:
        """Print a compact summary of the analysis result."""
        logger.info("=" * 50)
        logger.info("Network Analysis Summary")
        logger.info("=" * 50)
        logger.info(f"  Pattern: {result.pattern}")
        logger.info(f"  Sync links: {result.n_sync_links}")
        logger.info(f"  Causal links: {result.n_causal_links}")

        if result.hub_names:
            logger.info(f"  Hub dimensions: {', '.join(result.hub_names)}")

        if result.driver_names:
            logger.info(f"  Causal drivers: {', '.join(result.driver_names)}")

        if result.follower_names:
            logger.info(f"  Causal followers: {', '.join(result.follower_names)}")

        if result.sync_network:
            logger.info("  Sync Network:")
            for link in sorted(
                result.sync_network,
                key=lambda lnk: lnk.strength,
                reverse=True,
            ):
                sign = "+" if link.correlation > 0 else "−"
                logger.info(
                    f"    {link.from_name} ↔ {link.to_name}: "
                    f"{sign}{link.strength:.3f}"
                )

        if result.causal_network:
            logger.info("  Causal Network:")
            for link in sorted(
                result.causal_network,
                key=lambda lnk: lnk.strength,
                reverse=True,
            ):
                logger.info(
                    f"    {link.from_name} → {link.to_name}: "
                    f"{link.strength:.3f} (lag={link.lag})"
                )
