"""
FISJ — Find Insight Structural Junction
========================================
Built by Masamichi & Tamaki

Causal discovery from time series via Λ³ structural analysis.

Quick start:
    >>> from FISJ import FISJAdapter
    >>> adapter = FISJAdapter(method="nnnu_inverse", max_lag=5)
    >>> result = adapter.fit(df)
    >>> scores = result.adjacency_scores
    >>> binary = result.adjacency_bin

Methods:
    - 'nnnu':         Zero-parameter causal discovery (Sign=1.000)
    - 'nnnu_inverse': NNNU + Inverse + Regime rescue (recommended)
    - 'fusion':       NetworkCore + Inverse (AUC-strongest, legacy)
    - 'inverse':      Inverse engine only

Engines (advanced use):
    >>> from FISJ.engines import NNNUEngine, InverseCausalEngine

Benchmark utilities:
    >>> from FISJ import make_topk_binary
    >>> binary_topk = make_topk_binary(result.adjacency_scores, k=expected_edges)
"""

from .adapter import FISJAdapter, MethodOutput, run_fisj, make_topk_binary

# Engine direct access
from .engines import (
    NNNUEngine, NNNUResult,
    InverseCausalEngine, InverseCausalEngineConfig,
    NetworkAnalyzerCore,
    GenericRegimeDetector, GenericRegimeConfig,
)

# Core primitives
from .core import (
    local_std_1d,
    rho_t_1d,
    extract_lambda3_events,
    benjamini_hochberg_per_source,
)

__version__ = "0.9.9"

__all__ = [
    # Main interface
    "FISJAdapter",
    "MethodOutput",
    "run_fisj",
    "make_topk_binary",
    # Engines
    "NNNUEngine",
    "NNNUResult",
    "InverseCausalEngine",
    "InverseCausalEngineConfig",
    "NetworkAnalyzerCore",
    "GenericRegimeDetector",
    "GenericRegimeConfig",
    # Core
    "local_std_1d",
    "rho_t_1d",
    "extract_lambda3_events",
    "benjamini_hochberg_per_source",
]
