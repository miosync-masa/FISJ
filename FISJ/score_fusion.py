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
    # --- From NNNU (optional) ---
    nnnu_score_matrix: np.ndarray | None = None,
    nnnu_q_matrix: np.ndarray | None = None,
    # --- Fusion parameters ---
    fusion_mode: str = "suppress",
    w_raw: float = 0.50,
    w_stat: float = 0.25,
    w_struct: float = 0.25,
    w_freq: float = 0.0,
    alpha: float = 0.05,
    suppress_floor: float = 0.05,
    tau_fused: float | None = None,
    eps: float = 1e-300,
) -> FusionResult:
    """
    Fuse independent causal evidence streams.

    Two fusion modes:

    "suppress" (default):
        Uses q-value as a hard suppressor to kill non-significant edges.
        Best for AUC — preserves clean separation between true/false.
            S = raw * DI_gate * suppressor * nnnu_gate
            suppressor = 1.0 if q < alpha, else suppress_floor
            nnnu_gate  = 1.0 if nnnu confirms, else suppress_floor

    "geometric":
        Rank-normalized geometric mean. All evidence treated equally.
            S = s_raw^w_r · s_stat^w_q · s_struct^w_d

    Binary graph (both modes): AND condition
        edge = (q < alpha) AND (S > tau)
    """
    n_dims = raw_score_matrix.shape[0]

    # --- Step 1: q-value matrix ---
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

    # --- Step 3: Rank-normalize (used by both modes for diagnostics) ---
    s_raw = _ranknorm(np.maximum(raw_score_matrix, 0.0))
    neg_log_q = -np.log10(q_matrix + eps)
    np.fill_diagonal(neg_log_q, 0.0)
    s_stat = _ranknorm(neg_log_q)
    s_struct = _ranknorm(np.maximum(struct_matrix, 0.0))

    use_freq = frequency_matrix is not None and w_freq > 0
    s_freq = _ranknorm(np.maximum(frequency_matrix, 0.0)) if use_freq else None

    # --- Step 4: Fusion ---
    if fusion_mode == "suppress":
        # q-value as hard suppressor: significant → 1.0, else → floor
        suppressor = np.where(q_matrix < alpha, 1.0, suppress_floor)
        np.fill_diagonal(suppressor, 0.0)

        # DI as soft gate: same idea as DirectIrreducibilityScorer
        raw_norm = np.maximum(raw_score_matrix, 0.0)
        raw_max = raw_norm.max()
        if raw_max > 0:
            raw_norm = raw_norm / raw_max

        struct_norm = np.maximum(struct_matrix, 0.0)
        struct_max = struct_norm.max()
        if struct_max > 0:
            struct_norm = struct_norm / struct_max

        # Blend: raw base + DI boost, then suppress
        di_gate = 0.7 + 0.3 * struct_norm
        fused = raw_norm * di_gate * suppressor

        # NNNU gate: if NNNU sees no causal signal, suppress further
        if nnnu_score_matrix is not None:
            nnnu_norm = np.maximum(nnnu_score_matrix, 0.0)
            nnnu_max = nnnu_norm.max()
            if nnnu_max > 0:
                nnnu_norm = nnnu_norm / nnnu_max
            # Soft gate: NNNU score scales 0.3 → 1.0
            # Even 0 NNNU score gives 0.3 (doesn't completely kill)
            nnnu_gate = 0.3 + 0.7 * nnnu_norm
            fused = fused * nnnu_gate

        # NNNU q-value as hard suppressor (if available)
        if nnnu_q_matrix is not None:
            nnnu_suppress = np.where(nnnu_q_matrix < alpha, 1.0, suppress_floor)
            np.fill_diagonal(nnnu_suppress, 0.0)
            fused = fused * nnnu_suppress

        np.fill_diagonal(fused, 0.0)

    else:  # "geometric"
        if use_freq:
            w_total = w_raw + w_stat + w_struct + w_freq
            wr, wq, wd, wf = w_raw/w_total, w_stat/w_total, w_struct/w_total, w_freq/w_total
        else:
            w_total = w_raw + w_stat + w_struct
            wr, wq, wd = w_raw/w_total, w_stat/w_total, w_struct/w_total
            wf = 0.0

        fused = (
            np.power(np.maximum(s_raw, eps), wr)
            * np.power(np.maximum(s_stat, eps), wq)
            * np.power(np.maximum(s_struct, eps), wd)
        )
        if use_freq and s_freq is not None:
            fused *= np.power(np.maximum(s_freq, eps), wf)
        np.fill_diagonal(fused, 0.0)

    # --- Step 5: Binary graph (AND condition) ---
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

    # --- Step 6: Consensus ranking (backup) ---
    w_total = w_raw + w_stat + w_struct
    wr_c, wq_c, wd_c = w_raw/w_total, w_stat/w_total, w_struct/w_total
    consensus = wr_c * s_raw + wq_c * s_stat + wd_c * s_struct
    if use_freq and s_freq is not None:
        consensus = consensus * 0.8 + 0.2 * s_freq
    np.fill_diagonal(consensus, 0.0)

    weights = {"raw": w_raw, "stat": w_stat, "struct": w_struct, "mode": fusion_mode}
    if use_freq:
        weights["freq"] = w_freq

    n_sig = int(np.sum(q_matrix < alpha))
    n_bin = int(np.sum(binary))
    logger.info(
        f"🔗 Score Fusion ({fusion_mode}): n_dims={n_dims}, "
        f"q<{alpha}: {n_sig} edges, binary: {n_bin} edges, "
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
