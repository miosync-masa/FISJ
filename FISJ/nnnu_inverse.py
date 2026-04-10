"""
NNNU_Inverse Adapter
=====================
Built by Masamichi & Tamaki

Two-engine causal discovery: no partial correlation, no precision matrix.

  NNNU (Level 1):  All-frame signed_mean → 100% causal candidate extraction
  Inverse (Level 2): Source-drop DI → Cascade separation + interventional evidence

"Neural Network Non-Use × Inverse Scattering"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .nnnu import NNNUEngine
from .inverse_causal_engine import InverseCausalEngine, InverseCausalEngineConfig


class NNNUInverseAdapter:
    """
    NNNU + Inverse Engine fusion.

    NNNU provides:
      - All-frame signed_mean scoring (statistical power)
      - Jump-based directional consistency (signal quality)
      - Per-source BH-FDR (binary decision)
      - Spurious filter (common ancestor + mediator)

    Inverse Engine provides:
      - Source-drop DI (cascade separation)
      - Interventional evidence (Level 2)

    Fusion:
      score = NNNU_score × DI_gate × suppress
    """

    method_name = "NNNU_Inverse"

    def __init__(
        self,
        max_lag: int = 5,
        solver: str = "ridge",
        alpha: float = 0.05,
        delta_percentile: float = 90.0,
        suppress_floor: float = 0.05,
        method_name: str | None = None,
    ):
        self.max_lag = max_lag
        self.solver = solver
        self.alpha = alpha
        self.delta_percentile = delta_percentile
        self.suppress_floor = suppress_floor
        if method_name is not None:
            self.method_name = method_name

    def fit(self, df: pd.DataFrame, cfg=None):
        from .adapter import MethodOutput

        names = list(df.columns)
        n = len(names)
        state_vectors = df.values.astype(np.float64)

        # === Engine 1: NNNU (Level 1 — observational) ===
        nnnu = NNNUEngine(
            max_lag=self.max_lag,
            delta_percentile=self.delta_percentile,
            alpha=self.alpha,
            adaptive=True,
        )
        nnnu_result = nnnu.fit(state_vectors)

        # === Engine 2: Inverse (Level 2 — interventional) ===
        ice_config = InverseCausalEngineConfig(
            max_lag=self.max_lag,
            ar_lag=1,
            solver=self.solver,
            standardize=True,
            include_intercept=True,
            validation_fraction=0.25,
            use_backward_check=True,
            refit_on_drop=False,
            residualize_ar=True,
            compute_direct_irreducibility=True,
        )
        ice_result = InverseCausalEngine(ice_config).fit(
            state_vectors, dimension_names=names,
        )

        # === Fusion: NNNU score × DI gate × suppress ===
        nnnu_score = nnnu_result.score_matrix.copy()
        nnnu_q = nnnu_result.q_matrix.copy()
        consistency = nnnu_result.consistency_matrix.copy()

        # DI gate (soft): 0.3 base + 0.7 × normalized DI
        di_matrix = ice_result.direct_score_matrix
        if di_matrix is not None:
            di_norm = np.maximum(di_matrix, 0.0)
            di_max = di_norm.max()
            if di_max > 0:
                di_norm = di_norm / di_max
            di_gate = 0.3 + 0.7 * di_norm
        else:
            di_gate = np.ones((n, n))
        np.fill_diagonal(di_gate, 0.0)

        # Fused score
        fused = nnnu_score * di_gate

        # Suppress: q-value + consistency
        min_consistency = 0.70 if n <= 5 else 0.65
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if nnnu_q[i, j] >= self.alpha:
                    fused[i, j] *= self.suppress_floor
                if consistency[i, j] <= min_consistency:
                    fused[i, j] *= self.suppress_floor

        np.fill_diagonal(fused, 0.0)

        # Binary: NNNU q-value + consistency + DI confirmation
        binary = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if (nnnu_q[i, j] < self.alpha
                        and consistency[i, j] > min_consistency
                        and nnnu_result.binary_matrix[i, j] > 0):
                    binary[i, j] = 1

        return MethodOutput(
            method_name=self.method_name,
            names=names,
            adjacency_scores=fused,
            adjacency_bin=binary,
            directed_support=True,
            lag_support=True,
            sign_support=True,
            lag_matrix=nnnu_result.lag_matrix,
            sign_matrix=nnnu_result.sign_matrix,
            meta={
                "nnnu_q_matrix": nnnu_q,
                "nnnu_raw_score": nnnu_result.raw_score_matrix,
                "consistency_matrix": consistency,
                "di_matrix": di_matrix,
                "n_binary_edges": int(np.sum(binary)),
                "nnnu_total_jumps": nnnu_result.total_jumps,
            },
        )
