"""
FISJ core — Λ³ shared primitives
================================
Built by Masamichi & Tamaki

Λ³ (Lambda-cubed) framework primitives:
  - Dimensionless displacement: diff / local_std
  - Jump extraction: percentile threshold
  - Tension density (ρT): local variability
"""

from __future__ import annotations

import numpy as np


def local_std_1d(data: np.ndarray, window: int) -> np.ndarray:
    """Symmetric-window local standard deviation. Λ³ normalization denominator."""
    n = len(data)
    out = np.zeros(n)
    for i in range(n):
        s = max(0, i - window)
        e = min(n, i + window + 1)
        sub = data[s:e]
        if len(sub) > 0:
            out[i] = np.std(sub)
    return out


def rho_t_1d(data: np.ndarray, window: int) -> np.ndarray:
    """1D tension density (ρT) — past-window local standard deviation."""
    n = len(data)
    out = np.zeros(n)
    for i in range(n):
        s = max(0, i - window)
        sub = data[s:i + 1]
        if len(sub) > 1:
            out[i] = np.std(sub)
    return out


def extract_lambda3_events(
    state_vectors: np.ndarray,
    local_std_window: int = 20,
    rho_t_window: int = 30,
    delta_percentile: float = 90.0,
):
    """
    Extract Λ³ events from multivariate time series.

    Returns
    -------
    events_pos : (n_frames-1, n_dims) positive ΔΛC events
    events_neg : (n_frames-1, n_dims) negative ΔΛC events
    rho_t      : (n_frames, n_dims) tension density
    disp       : (n_frames-1, n_dims) dimensionless displacement
    local_std  : (n_frames, n_dims) local standard deviation
    """
    n_frames, n_dims = state_vectors.shape
    n_diff = n_frames - 1

    events_pos = np.zeros((n_diff, n_dims))
    events_neg = np.zeros((n_diff, n_dims))
    rho_t = np.zeros((n_frames, n_dims))
    disp = np.zeros((n_diff, n_dims))
    lstd_all = np.zeros((n_frames, n_dims))

    for d in range(n_dims):
        series = state_vectors[:, d]
        diff = np.diff(series)
        lstd = local_std_1d(series, local_std_window)
        lstd_diff = lstd[1:]
        score = np.abs(diff) / (lstd_diff + 1e-10)
        threshold = np.percentile(score, delta_percentile)

        jump_mask = score > threshold
        events_pos[:, d] = ((diff > 0) & jump_mask).astype(float)
        events_neg[:, d] = ((diff < 0) & jump_mask).astype(float)

        disp[:, d] = diff / (lstd_diff + 1e-10)
        rho_t[:, d] = rho_t_1d(series, rho_t_window)
        lstd_all[:, d] = lstd

    return events_pos, events_neg, rho_t, disp, lstd_all


def benjamini_hochberg_per_source(p_matrix: np.ndarray) -> np.ndarray:
    """
    Per-source Benjamini-Hochberg FDR correction.

    For each source dimension, correct N-1 tests independently.
    This is statistically correct because each source's jumps
    are independent experiments.

    Avoids over-correction when N is large (global BH would
    correct over N*(N-1) tests).
    """
    n_dims = p_matrix.shape[0]
    q = np.ones_like(p_matrix)

    for src in range(n_dims):
        targets = [(tgt, p_matrix[src, tgt])
                   for tgt in range(n_dims) if tgt != src]
        if not targets:
            continue

        raw = np.array([p for _, p in targets], dtype=float)
        order = np.argsort(raw)
        sorted_p = raw[order]
        m = len(sorted_p)

        adjusted = np.zeros(m, dtype=float)
        adjusted[-1] = sorted_p[-1]
        for k in range(m - 2, -1, -1):
            adjusted[k] = min(adjusted[k + 1], sorted_p[k] * m / (k + 1))
        adjusted = np.clip(adjusted, 0.0, 1.0)

        for rank, idx in enumerate(order):
            tgt, _ = targets[idx]
            q[src, tgt] = adjusted[rank]

    np.fill_diagonal(q, 1.0)
    return q
