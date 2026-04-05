"""
FISJ Adapter - Benchmark-Ready Interface
==========================================
Built by Masamichi & Tamaki

A thin adapter that wraps NetworkAnalyzerCore for direct use with
causal discovery benchmark frameworks. No external pipeline dependency.

Usage
-----
>>> from FISJ import FISJAdapter
>>> adapter = FISJAdapter()
>>> result = adapter.fit(df)
>>> result.adjacency_bin   # (n, n) binary adjacency
>>> result.lag_matrix       # (n, n) optimal lag
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .main import NetworkAnalyzerCore


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


class FISJAdapter:
    """
    Benchmark adapter for FISJ.

    Accepts a pandas DataFrame, runs NetworkAnalyzerCore, and returns
    a MethodOutput. No external pipeline dependency.

    Parameters
    ----------
    sync_threshold : float
        Synchronization threshold (adaptive hint if adaptive=True).
    causal_threshold : float
        Causal link threshold (adaptive hint if adaptive=True).
    max_lag : int
        Maximum lag in frames (adaptive hint if adaptive=True).
    adaptive : bool
        Enable data-driven adaptive parameter tuning.
    local_std_window : int
        Rolling window size for local standard deviation computation.
    method_name : str | None
        Override the default method name in output.
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
            Standardized analysis result with adjacency, lag, and sign
            matrices.
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
