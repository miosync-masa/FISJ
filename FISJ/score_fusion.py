"""
Score Fusion Module
===================
Built by Masamichi & Tamaki

Merges two independent causal evidence streams:
  - NetworkAnalyzerCore  → statistical significance (q-value via Fisher z + BH-FDR)
  - InverseCausalEngine  → structural necessity (DI / direct_score)

Design principle:
  - AUC/PR  → soft fusion (rank-normalized geometric mean)
  - Binary  → AND condition (q < α ∧ fused_score > τ)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

logger = logging.getLogger("fisj.score_fusion")


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class FusionResult:
    """Output of the score fusion module."""

    # Core outputs
    fused_score_matrix: np.ndarray       # AUC用 soft fusion score
    binary_matrix: np.ndarray            # AND条件による二値グラフ

    # Intermediate rank-normalized scores
    s_raw: np.ndarray                    # ranknorm of inverse engine raw score
    s_stat: np.ndarray                   # ranknorm of -log10(q)
    s_struct: np.ndarray                 # ranknorm of DI direct_score
    s_freq: np.ndarray | None = None     # ranknorm of segment frequency

    # Statistical significance from NetworkAnalyzerCore path
    p_matrix: np.ndarray = field(default_factory=lambda: np.zeros(0))
    q_matrix: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Consensus ranking (ablation / baseline用)
    consensus_score_matrix: np.ndarray = field(default_factory=lambda: np.zeros(0))

    # Metadata
    n_dims: int = 0
    weights: dict[str, float] = field(default_factory=dict)


# ============================================================================
# Significance helpers (ported from NetworkAnalyzerCore)
# ============================================================================


def _pcorr_pvalue(r: float, n: int, k: int) -> float:
    """
    P-value for partial correlation via Fisher z-transform.

    z = atanh(r) * sqrt(n - k - 3)
    p = 2 * (1 - Φ(|z|))
    """
    dof = n - k - 3
    if dof < 1 or abs(r) >= 1.0:
        return 1.0 if abs(r) < 1.0 else 0.0
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(dof)
    return float(2.0 * stats.norm.sf(abs(z)))


def _bh_fdr_matrix(p_matrix: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to a directed p-value matrix.

    Diagonal entries are left at 1.0.
    """
    n = p_matrix.shape[0]
    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
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
# Rank normalization
# ============================================================================


def _ranknorm(matrix: np.ndarray) -> np.ndarray:
    """
    Rank-normalize off-diagonal entries to [0, 1].

    Ties get averaged ranks. Diagonal stays 0.
    """
    n = matrix.shape[0]
    vals = []
    indices = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vals.append(matrix[i, j])
            indices.append((i, j))

    if not vals:
        return np.zeros_like(matrix)

    vals = np.array(vals, dtype=float)
    m = len(vals)

    # Rank with tie-averaging via argsort-of-argsort
    order = np.argsort(vals)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(m, dtype=float)

    # Normalize to [0, 1]
    if m > 1:
        ranks = ranks / (m - 1)

    out = np.zeros_like(matrix)
    for idx, (i, j) in enumerate(indices):
        out[i, j] = ranks[idx]
    return out


# ============================================================================
# Main fusion function
# ============================================================================


def compute_causal_q_matrix(
    causal_matrix: np.ndarray,
    lag_matrix: np.ndarray,
    n_samples: int,
    n_dims: int,
    max_lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute directed p-value and q-value matrices from causal correlation.

    Uses the same Fisher z + Bonferroni-within-pair + BH-FDR pipeline
    as NetworkAnalyzerCore._build_networks, but outputs full directed
    matrices instead of filtered link lists.

    Parameters
    ----------
    causal_matrix : (N, N) partial correlation matrix (symmetric, signed)
    lag_matrix : (N, N) best lag matrix (signed: positive = row leads)
    n_samples : effective sample count
    n_dims : number of dimensions
    max_lag : maximum lag tested

    Returns
    -------
    p_matrix : (N, N) raw p-values (directed)
    q_matrix : (N, N) BH-FDR corrected q-values (directed)
    """
    p_matrix = np.ones((n_dims, n_dims), dtype=float)

    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            causal_corr = causal_matrix[i, j]
            lag = lag_matrix[i, j]

            abs_lag = abs(int(lag))
            n_lagged = max(1, n_samples - abs_lag)
            k_causal = max(0, min(n_dims - 2, n_lagged // 4))
            p_raw = _pcorr_pvalue(causal_corr, n_lagged, k_causal)

            # Bonferroni within pair: tested max_lag × 2 directions
            n_lag_tests = max_lag * 2
            p_corrected = min(1.0, p_raw * n_lag_tests)

            # Assign to directed edge based on lag sign
            if lag > 0:
                p_matrix[i, j] = p_corrected
            elif lag < 0:
                p_matrix[j, i] = p_corrected
            else:
                # lag == 0: assign to both (conservative)
                p_matrix[i, j] = p_corrected
                p_matrix[j, i] = p_corrected

    q_matrix = _bh_fdr_matrix(p_matrix)
    return p_matrix, q_matrix


def fuse_scores(
    # --- From InverseCausalEngine ---
    raw_score_matrix: np.ndarray,
    direct_score_matrix: np.ndarray | None,
    # --- From NetworkAnalyzerCore ---
    causal_matrix: np.ndarray,
    lag_matrix: np.ndarray,
    n_samples: int,
    max_lag: int,
    # --- From V2 Regime-Aware (optional) ---
    frequency_matrix: np.ndarray | None = None,
    # --- Fusion parameters ---
    w_raw: float = 0.50,
    w_stat: float = 0.25,
    w_struct: float = 0.25,
    w_freq: float = 0.0,
    alpha: float = 0.05,
    tau_fused: float | None = None,
    eps: float = 1e-300,
) -> FusionResult:
    """
    Fuse independent causal evidence streams.

    Supports 3-evidence (raw + stat + struct) or 4-evidence
    (+ segment frequency) geometric mean fusion.

    When frequency_matrix is provided and w_freq > 0, weights are
    automatically renormalized to sum to 1.

    AUC score: rank-normalized geometric mean
        S = s_raw^w_r · s_stat^w_q · s_struct^w_d [· s_freq^w_f]

    Binary graph: AND condition
        edge = (q < alpha) AND (S > tau)

    Parameters
    ----------
    raw_score_matrix : inverse engine の score_matrix_unfiltered
    direct_score_matrix : inverse engine の direct_score_matrix (DI)
    causal_matrix : NetworkAnalyzerCore の causal_matrix
    lag_matrix : NetworkAnalyzerCore の causal_lag_matrix
    n_samples : effective sample count for p-value computation
    max_lag : maximum lag tested
    frequency_matrix : V2 regime-aware の causal_frequency_matrix
    w_raw, w_stat, w_struct, w_freq : geometric mean weights
    alpha : FDR threshold for binary graph
    tau_fused : fused score threshold for binary graph (auto if None)
    eps : floor for log(q) computation

    Returns
    -------
    FusionResult
    """
    n_dims = raw_score_matrix.shape[0]

    # --- Step 1: q-value matrix from NetworkAnalyzerCore's correlation ---
    p_matrix, q_matrix = compute_causal_q_matrix(
        causal_matrix=causal_matrix,
        lag_matrix=lag_matrix,
        n_samples=n_samples,
        n_dims=n_dims,
        max_lag=max_lag,
    )

    # --- Step 2: Prepare DI score ---
    struct_matrix = (
        direct_score_matrix.copy()
        if direct_score_matrix is not None
        else raw_score_matrix.copy()
    )
    np.fill_diagonal(struct_matrix, 0.0)

    # --- Step 3: Rank-normalize all sources ---
    s_raw = _ranknorm(np.maximum(raw_score_matrix, 0.0))

    neg_log_q = -np.log10(q_matrix + eps)
    np.fill_diagonal(neg_log_q, 0.0)
    s_stat = _ranknorm(neg_log_q)

    s_struct = _ranknorm(np.maximum(struct_matrix, 0.0))

    # Frequency (4th evidence, optional)
    use_freq = frequency_matrix is not None and w_freq > 0
    s_freq = None
    if use_freq:
        s_freq = _ranknorm(np.maximum(frequency_matrix, 0.0))

    # --- Step 4: Normalize weights ---
    if use_freq:
        w_total = w_raw + w_stat + w_struct + w_freq
        wr = w_raw / w_total
        wq = w_stat / w_total
        wd = w_struct / w_total
        wf = w_freq / w_total
    else:
        w_total = w_raw + w_stat + w_struct
        wr = w_raw / w_total
        wq = w_stat / w_total
        wd = w_struct / w_total
        wf = 0.0

    # --- Step 5: Geometric mean fusion ---
    fused = (
        np.power(np.maximum(s_raw, eps), wr)
        * np.power(np.maximum(s_stat, eps), wq)
        * np.power(np.maximum(s_struct, eps), wd)
    )
    if use_freq and s_freq is not None:
        fused *= np.power(np.maximum(s_freq, eps), wf)
    np.fill_diagonal(fused, 0.0)

    # --- Step 6: Binary graph (AND condition) ---
    if tau_fused is None:
        nonzero = fused[fused > 0]
        tau_fused = float(np.percentile(nonzero, 50)) if len(nonzero) > 0 else 0.0

    binary = np.zeros((n_dims, n_dims), dtype=float)
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                continue
            if q_matrix[i, j] < alpha and fused[i, j] > tau_fused:
                binary[i, j] = 1.0

    # --- Step 7: Consensus ranking (Borda-style backup) ---
    consensus = wr * s_raw + wq * s_stat + wd * s_struct
    if use_freq and s_freq is not None:
        consensus += wf * s_freq
    np.fill_diagonal(consensus, 0.0)

    weights = {"raw": wr, "stat": wq, "struct": wd}
    if use_freq:
        weights["freq"] = wf

    logger.info(
        f"🔗 Score Fusion: n_dims={n_dims}, "
        f"weights={weights}, "
        f"q<{alpha}: {int(np.sum(q_matrix < alpha))} edges, "
        f"binary: {int(np.sum(binary))} edges, "
        f"tau_fused={tau_fused:.4f}"
    )

    return FusionResult(
        fused_score_matrix=fused,
        binary_matrix=binary,
        s_raw=s_raw,
        s_stat=s_stat,
        s_struct=s_struct,
        s_freq=s_freq,
        p_matrix=p_matrix,
        q_matrix=q_matrix,
        consensus_score_matrix=consensus,
        n_dims=n_dims,
        weights=weights,
    )
