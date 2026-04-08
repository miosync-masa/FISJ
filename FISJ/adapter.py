"""
FISJ Adapter - Benchmark-Ready Interface
==========================================
Built by Masamichi & Tamaki

Two adapters:
  - FISJAdapter       : Original NetworkAnalyzerCore-based (partial correlation)
  - FISJInverseAdapter: InverseCausalEngine-based (auto Ridge/Lasso + DI)

Usage
-----
>>> from FISJ import FISJInverseAdapter
>>> adapter = FISJInverseAdapter()
>>> result = adapter.fit(df)
>>> result.adjacency_scores   # (n, n) continuous causal scores
>>> result.lag_matrix          # (n, n) optimal lag
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .main import NetworkAnalyzerCore
from .inverse_causal_engine import InverseCausalEngine, InverseCausalEngineConfig
from .score_fusion import fuse_scores


@dataclass
class MethodOutput:
    """
    Standardized output format for causal discovery benchmarks.

    Stores the results of any causal estimation method in a unified
    structure suitable for cross-method evaluation.
    """

    method_name: str
    names: list[str]
    adjacency_scores: np.ndarray  # (n, n) continuous causal scores
    adjacency_bin: np.ndarray  # (n, n) binary adjacency matrix

    directed_support: bool = False
    lag_support: bool = False
    sign_support: bool = False

    lag_matrix: np.ndarray | None = None  # (n, n)
    sign_matrix: np.ndarray | None = None  # (n, n)

    meta: dict = field(default_factory=dict)

    def undirected_bin(self) -> np.ndarray | None:
        if self.adjacency_bin is None:
            return None
        return ((self.adjacency_bin + self.adjacency_bin.T) > 0).astype(int)


# ============================================================================
# Original adapter (NetworkAnalyzerCore)
# ============================================================================


class FISJAdapter:
    """
    Benchmark adapter for FISJ using NetworkAnalyzerCore.

    Accepts a pandas DataFrame, runs NetworkAnalyzerCore, and returns
    a MethodOutput. No external pipeline dependency.
    """

    method_name = "FISJ"

    def __init__(
        self,
        sync_threshold: float = 0.3,
        causal_threshold: float = 0.25,
        max_lag: int = 8,
        adaptive: bool = True,
        local_std_window: int = 20,
        method_name: str | None = None,
    ):
        self.sync_threshold = sync_threshold
        self.causal_threshold = causal_threshold
        self.max_lag = max_lag
        self.adaptive = adaptive
        self.local_std_window = local_std_window

        if method_name is not None:
            self.method_name = method_name

    def fit(
        self,
        df: pd.DataFrame,
        cfg: object | None = None,
    ) -> MethodOutput:
        """
        Run causal network analysis on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Each column represents one time-series dimension.
        cfg : object, optional
            Benchmark framework configuration (accepted for interface
            compatibility but not used).

        Returns
        -------
        MethodOutput
            Standardized analysis result.
        """
        names = list(df.columns)
        n = len(names)
        state_vectors = df.values.astype(np.float64)

        analyzer = NetworkAnalyzerCore(
            sync_threshold=self.sync_threshold,
            causal_threshold=self.causal_threshold,
            max_lag=self.max_lag,
            adaptive=self.adaptive,
            local_std_window=self.local_std_window,
        )

        result = analyzer.analyze(
            state_vectors=state_vectors,
            dimension_names=names,
        )

        # --- Convert NetworkResult to benchmark format ---
        scores = np.abs(result.sync_matrix) + np.abs(result.causal_matrix)
        np.fill_diagonal(scores, 0.0)

        adjacency = np.zeros((n, n), dtype=int)
        lagm = np.zeros((n, n), dtype=int)
        signm = np.zeros((n, n), dtype=int)

        for link in result.sync_network:
            adjacency[link.from_dim, link.to_dim] = 1
            adjacency[link.to_dim, link.from_dim] = 1
            sign = int(np.sign(link.correlation))
            signm[link.from_dim, link.to_dim] = sign
            signm[link.to_dim, link.from_dim] = sign

        for link in result.causal_network:
            adjacency[link.from_dim, link.to_dim] = 1
            lagm[link.from_dim, link.to_dim] = link.lag
            signm[link.from_dim, link.to_dim] = int(np.sign(link.correlation))

        np.fill_diagonal(adjacency, 0)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            lag_matrix=lagm,
            sign_matrix=signm,
            meta={
                "pattern": result.pattern,
                "adaptive": result.adaptive_params,
                "n_sync_links": result.n_sync_links,
                "n_causal_links": result.n_causal_links,
                "hub_names": result.hub_names,
                "driver_names": result.driver_names,
                "follower_names": result.follower_names,
            },
        )


# ============================================================================
# Inverse Causal Engine adapter (auto Ridge/Lasso + Direct Irreducibility)
# ============================================================================


class FISJInverseAdapter:
    """
    Benchmark adapter for FISJ using InverseCausalEngine.

    Auto-selects solver (Ridge/Lasso) and post-processing (Direct
    Irreducibility) based on data characteristics via DataGate.

    Parameters
    ----------
    max_lag : int
        Maximum causal lag to consider.
    solver : str
        "auto" (recommended), "ridge", or "lasso".
    score_mode : str
        Score composition mode: "mixed", "block_norm", "delta_mse".
    binary_threshold : float | None
        Threshold on adjacency_scores for binary adjacency.
        If None, uses detected links from engine.
    method_name : str | None
        Override the default method name in output.
    **engine_kwargs
        Additional keyword arguments passed to InverseCausalEngineConfig.
    """

    method_name = "FISJ-Inverse"

    def __init__(
        self,
        max_lag: int = 5,
        solver: str = "auto",
        score_mode: str = "mixed",
        binary_threshold: float | None = None,
        method_name: str | None = None,
        **engine_kwargs,
    ):
        self.max_lag = max_lag
        self.solver = solver
        self.score_mode = score_mode
        self.binary_threshold = binary_threshold
        self.engine_kwargs = engine_kwargs

        if method_name is not None:
            self.method_name = method_name

    def fit(
        self,
        df: pd.DataFrame,
        cfg: object | None = None,
    ) -> MethodOutput:
        """
        Run inverse-problem causal analysis on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Each column represents one time-series dimension.
        cfg : object, optional
            Benchmark framework configuration (accepted for interface
            compatibility but not used).

        Returns
        -------
        MethodOutput
            Standardized analysis result with adjacency, lag, and sign
            matrices.
        """
        names = list(df.columns)
        n = len(names)
        state_vectors = df.values.astype(np.float64)

        # Build engine config with defaults + user overrides
        config_params = dict(
            max_lag=self.max_lag,
            ar_lag=1,
            solver=self.solver,
            score_mode=self.score_mode,
            standardize=True,
            include_intercept=True,
            validation_fraction=0.25,
            use_backward_check=True,
            # auto routing handles these via _select_pipeline:
            apply_textbook_filter=False,
            compute_direct_irreducibility=True,
        )
        config_params.update(self.engine_kwargs)

        config = InverseCausalEngineConfig(**config_params)
        engine = InverseCausalEngine(config)
        result = engine.fit(state_vectors, dimension_names=names)

        # --- Score selection ---
        # Use direct_score_matrix (DI) when available, else raw
        if result.direct_score_matrix is not None:
            scores = result.direct_score_matrix.copy()
        else:
            scores = result.score_matrix_unfiltered.copy()
        np.fill_diagonal(scores, 0.0)

        # --- Binary adjacency ---
        if self.binary_threshold is not None:
            adjacency = (scores > self.binary_threshold).astype(int)
        else:
            # Adaptive binarization: percentile of nonzero scores
            nonzero = scores[scores > 0]
            if len(nonzero) > 0:
                thr = float(np.percentile(nonzero, 65))
                adjacency = (scores > thr).astype(int)
            else:
                adjacency = np.zeros((n, n), dtype=int)

        np.fill_diagonal(adjacency, 0)

        # --- Lag and sign matrices ---
        lagm = result.lag_matrix.copy()
        signm = result.sign_matrix.copy()

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            lag_matrix=lagm,
            sign_matrix=signm,
            meta={
                "gate_regime": result.gate.regime,
                "gate_mean_corr": result.gate.mean_corr,
                "gate_max_corr": result.gate.max_corr,
                "gate_feature_sample_ratio": result.gate.feature_sample_ratio,
                "effective_solver": getattr(engine, "_effective_solver", self.solver),
                "n_links": len(result.links),
                "has_di": result.direct_score_matrix is not None,
            },
        )


# ============================================================================
# Fusion adapter (NetworkAnalyzerCore + InverseCausalEngine + suppress fusion)
# ============================================================================


class FISJFusionAdapter:
    """
    Benchmark adapter for FISJ using suppress-mode score fusion.

    Combines:
      - NetworkAnalyzerCore: raw ranking + q-value (statistical significance)
      - InverseCausalEngine: DI (structural necessity)

    The q-value acts as a hard suppressor to kill non-significant edges,
    preserving clean AUC separation while providing theoretically grounded
    binary thresholding for F-measure.

    Parameters
    ----------
    max_lag : int
        Maximum causal lag to consider.
    solver : str
        Solver for InverseCausalEngine: "auto", "ridge", or "lasso".
    alpha : float
        FDR threshold for q-value suppression.
    suppress_floor : float
        Floor value for non-significant edges (0.0 = hard kill).
    method_name : str | None
        Override the default method name in output.
    """

    method_name = "FISJ-Fusion"

    def __init__(
        self,
        max_lag: int = 5,
        solver: str = "auto",
        alpha: float = 0.05,
        suppress_floor: float = 0.05,
        method_name: str | None = None,
    ):
        self.max_lag = max_lag
        self.solver = solver
        self.alpha = alpha
        self.suppress_floor = suppress_floor

        if method_name is not None:
            self.method_name = method_name

    def fit(
        self,
        df: pd.DataFrame,
        cfg: object | None = None,
    ) -> MethodOutput:
        """
        Run fusion causal analysis on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Each column represents one time-series dimension.
        cfg : object, optional
            Benchmark framework configuration (accepted for interface
            compatibility but not used).

        Returns
        -------
        MethodOutput
            Standardized analysis result.
        """
        names = list(df.columns)
        n = len(names)
        state_vectors = df.values.astype(np.float64)

        # Engine 1: NetworkAnalyzerCore (raw score + q-value source)
        nac = NetworkAnalyzerCore(
            max_lag=self.max_lag,
            adaptive=False,
            p_value_threshold=self.alpha,
        )
        nac_result = nac.analyze(
            state_vectors=state_vectors,
            dimension_names=names,
        )

        # Engine 2: InverseCausalEngine (DI only)
        ice_config = InverseCausalEngineConfig(
            max_lag=self.max_lag,
            ar_lag=1,
            solver=self.solver,
            standardize=True,
            include_intercept=True,
            validation_fraction=0.25,
            use_backward_check=True,
            compute_direct_irreducibility=True,
        )
        ice_result = InverseCausalEngine(ice_config).fit(
            state_vectors, dimension_names=names,
        )

        # Fusion (suppress mode)
        n_samples = state_vectors.shape[0] - self.max_lag
        raw_score = np.abs(nac_result.causal_matrix)
        np.fill_diagonal(raw_score, 0.0)

        fusion = fuse_scores(
            raw_score_matrix=raw_score,
            direct_score_matrix=ice_result.direct_score_matrix,
            causal_matrix=nac_result.causal_matrix,
            lag_matrix=nac_result.causal_lag_matrix,
            n_samples=n_samples,
            max_lag=self.max_lag,
            fusion_mode="suppress",
            alpha=self.alpha,
            suppress_floor=self.suppress_floor,
        )

        scores = fusion.fused_score_matrix.copy()
        adjacency = fusion.binary_matrix.astype(int)

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=scores,
            adjacency_bin=adjacency,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            lag_matrix=ice_result.lag_matrix,
            sign_matrix=ice_result.sign_matrix,
            meta={
                "fusion_mode": "suppress",
                "alpha": self.alpha,
                "suppress_floor": self.suppress_floor,
                "n_sig_edges": int(np.sum(fusion.q_matrix < self.alpha)),
                "n_binary_edges": int(np.sum(adjacency)),
                "has_di": ice_result.direct_score_matrix is not None,
            },
        )
