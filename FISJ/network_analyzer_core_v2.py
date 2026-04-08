from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger("getter_one.analysis.inverse_causal_engine")


# ============================================================================
# Inverse Causal Engine
# ----------------------------------------------------------------------------
# Core idea
#   For each target dimension j, solve a single inverse problem:
#       x_j(t) <- own past + all other dimensions' lagged histories
#   Then interpret each source block of lag coefficients as a causal kernel.
#
# CauseMe-oriented design goals
#   - use the benchmark-provided max_lag directly
#   - produce continuous-valued edge scores for AUC/ROC evaluation
#   - direction emerges naturally from past -> present prediction
#   - avoid pairwise hypothesis tests / BH-FDR as the main engine
#   - optionally apply textbook post-filters for common-ancestor / mediation
# ============================================================================


# ============================================================================
# Data classes
# ============================================================================


@dataclass
class InverseCausalLink:
    """Directed causal link inferred from one source block."""

    from_dim: int
    to_dim: int
    from_name: str
    to_name: str
    strength: float
    signed_peak: float
    best_lag: int
    block_norm: float
    delta_mse_forward: float = 0.0
    delta_mse_backward: float = 0.0
    asymmetry: float = 0.0
    confidence: float = 0.0


@dataclass
class TargetFitSummary:
    """Fit summary for a single target node."""

    target_dim: int
    baseline_mse: float
    null_mse: float
    intercept: float
    self_kernel: np.ndarray
    source_kernels: dict[int, np.ndarray] = field(default_factory=dict)
    source_delta_mse_forward: dict[int, float] = field(default_factory=dict)
    source_delta_mse_backward: dict[int, float] = field(default_factory=dict)
    source_asymmetry: dict[int, float] = field(default_factory=dict)
    source_confidence: dict[int, float] = field(default_factory=dict)


@dataclass
class InverseCausalResult:
    """Full engine output."""

    links: list[InverseCausalLink]
    score_matrix: np.ndarray
    lag_matrix: np.ndarray
    sign_matrix: np.ndarray
    confidence_matrix: np.ndarray
    block_norm_matrix: np.ndarray
    delta_mse_matrix: np.ndarray
    target_summaries: list[TargetFitSummary]
    dimension_names: list[str]
    max_lag: int
    ar_lag: int


@dataclass
class InverseCausalEngineConfig:
    """Configuration for the inverse-problem causal engine."""

    max_lag: int
    ar_lag: int = 1
    alpha_ridge: float = 1e-2
    alpha_smooth: float = 1e-2
    standardize: bool = True
    include_intercept: bool = True
    validation_fraction: float = 0.25
    min_train_size: int = 40
    min_effect: float = 1e-6
    score_mix: float = 0.65
    confidence_mix: float = 0.35
    asymmetry_weight: float = 0.25
    use_backward_check: bool = True
    refit_on_drop: bool = True
    prune_by_confidence: bool = False
    confidence_quantile: float = 0.50
    apply_textbook_filter: bool = True
    common_ancestor_strength_ratio: float = 0.95
    mediated_path_strength_ratio: float = 0.85
    lag_tolerance: int = 1
    eps: float = 1e-12


# ============================================================================
# Engine
# ============================================================================


class InverseCausalEngine:
    """
    One-shot inverse-problem causal discovery engine.

    Workflow
    --------
    1. For each target j, build a lagged design matrix using:
         - own autoregressive lags
         - all other dimensions from lag 1..max_lag
    2. Solve a single regularized inverse problem for that target.
    3. Interpret each source block as a causal kernel.
    4. Convert each block into:
         - strength     = block norm
         - best_lag     = lag of maximum absolute coefficient
         - sign         = sign at peak lag
    5. Optionally refine with validation delta-MSE and backward-time asymmetry.
    6. Optionally apply textbook structural filters.

    Notes
    -----
    - This engine is fully domain-agnostic.
    - It is designed to emit continuous scores suitable for AUC/ROC evaluation.
    - The diagonal is always zero in the returned score matrix (self AR is modeled
      internally but not reported as causal output).
    """

    def __init__(self, config: InverseCausalEngineConfig):
        if config.max_lag < 1:
            raise ValueError("max_lag must be >= 1")
        if config.ar_lag < 0:
            raise ValueError("ar_lag must be >= 0")
        self.config = config

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def fit(self, state_vectors: np.ndarray, dimension_names: Optional[list[str]] = None) -> InverseCausalResult:
        """Fit the engine and return continuous causal scores."""
        state_vectors = np.asarray(state_vectors, dtype=float)
        if state_vectors.ndim != 2:
            raise ValueError("state_vectors must have shape (n_frames, n_dims)")

        n_frames, n_dims = state_vectors.shape
        if n_frames <= max(self.config.max_lag, self.config.ar_lag) + 5:
            raise ValueError("Not enough frames for the requested lag structure")

        if dimension_names is None:
            dimension_names = [f"dim_{i}" for i in range(n_dims)]

        block_norm = np.zeros((n_dims, n_dims), dtype=float)
        score = np.zeros((n_dims, n_dims), dtype=float)
        lag = np.zeros((n_dims, n_dims), dtype=int)
        sign = np.zeros((n_dims, n_dims), dtype=float)
        confidence = np.zeros((n_dims, n_dims), dtype=float)
        delta_mse = np.zeros((n_dims, n_dims), dtype=float)
        summaries: list[TargetFitSummary] = []

        for target in range(n_dims):
            summary = self._fit_one_target(state_vectors, target)
            summaries.append(summary)

            for source, kernel in summary.source_kernels.items():
                kernel = np.asarray(kernel, dtype=float)
                if kernel.size == 0:
                    continue
                bn = float(np.linalg.norm(kernel))
                peak_idx = int(np.argmax(np.abs(kernel)))
                best_lag = peak_idx + 1
                peak_val = float(kernel[peak_idx])
                d_f = float(summary.source_delta_mse_forward.get(source, 0.0))
                d_b = float(summary.source_delta_mse_backward.get(source, 0.0))
                asym = float(summary.source_asymmetry.get(source, 0.0))
                conf = float(summary.source_confidence.get(source, 0.0))

                block_norm[source, target] = bn
                lag[source, target] = best_lag
                sign[source, target] = np.sign(peak_val)
                confidence[source, target] = conf
                delta_mse[source, target] = d_f

        score = self._compose_score_matrix(block_norm, delta_mse, confidence)
        np.fill_diagonal(score, 0.0)
        np.fill_diagonal(block_norm, 0.0)
        np.fill_diagonal(confidence, 0.0)
        np.fill_diagonal(delta_mse, 0.0)
        np.fill_diagonal(lag, 0)
        np.fill_diagonal(sign, 0.0)

        links = self._links_from_matrices(
            score_matrix=score,
            lag_matrix=lag,
            sign_matrix=sign,
            block_norm_matrix=block_norm,
            delta_mse_matrix=delta_mse,
            confidence_matrix=confidence,
            dimension_names=dimension_names,
        )

        if self.config.prune_by_confidence:
            links = self._prune_by_confidence(links)
            score, lag, sign, block_norm, delta_mse, confidence = self._rebuild_matrices_from_links(
                links=links,
                n_dims=n_dims,
            )

        if self.config.apply_textbook_filter:
            links = self._apply_textbook_filter(links)
            score, lag, sign, block_norm, delta_mse, confidence = self._rebuild_matrices_from_links(
                links=links,
                n_dims=n_dims,
            )

        links.sort(key=lambda x: x.strength, reverse=True)

        return InverseCausalResult(
            links=links,
            score_matrix=score,
            lag_matrix=lag,
            sign_matrix=sign,
            confidence_matrix=confidence,
            block_norm_matrix=block_norm,
            delta_mse_matrix=delta_mse,
            target_summaries=summaries,
            dimension_names=dimension_names,
            max_lag=self.config.max_lag,
            ar_lag=self.config.ar_lag,
        )

    def fit_predict(self, state_vectors: np.ndarray, dimension_names: Optional[list[str]] = None) -> np.ndarray:
        """Convenience method returning the continuous score matrix only."""
        return self.fit(state_vectors, dimension_names).score_matrix

    def predict_adjacency(self, state_vectors: np.ndarray, dimension_names: Optional[list[str]] = None) -> np.ndarray:
        """Alias for benchmark-style usage."""
        return self.fit_predict(state_vectors, dimension_names)

    # ---------------------------------------------------------------------
    # Core fitting
    # ---------------------------------------------------------------------

    def _fit_one_target(self, state_vectors: np.ndarray, target: int) -> TargetFitSummary:
        y, X_full, meta, intercept = self._build_target_problem(state_vectors, target)
        n_samples = len(y)

        if n_samples < max(self.config.min_train_size, 10):
            raise ValueError(
                f"Target {target}: not enough samples after lag embedding ({n_samples})"
            )

        split = int(round(n_samples * (1.0 - self.config.validation_fraction)))
        split = int(np.clip(split, self.config.min_train_size, n_samples - 5))

        y_train, y_val = y[:split], y[split:]
        X_train, X_val = X_full[:split], X_full[split:]

        # Fit full forward model
        w_full = self._solve_regularized(X_train, y_train, meta)
        yhat_val = X_val @ w_full
        baseline_mse = self._mse(y_val, yhat_val)
        null_mse = self._mse(y_val, np.full_like(y_val, np.mean(y_train)))

        # Optional backward model for direction asymmetry
        backward_info = None
        if self.config.use_backward_check:
            backward_series = state_vectors[::-1].copy()
            y_b, X_b, meta_b, _ = self._build_target_problem(backward_series, target)
            if len(y_b) >= max(self.config.min_train_size, 10):
                split_b = int(round(len(y_b) * (1.0 - self.config.validation_fraction)))
                split_b = int(np.clip(split_b, self.config.min_train_size, len(y_b) - 5))
                yb_train, yb_val = y_b[:split_b], y_b[split_b:]
                Xb_train, Xb_val = X_b[:split_b], X_b[split_b:]
                wb_full = self._solve_regularized(Xb_train, yb_train, meta_b)
                mse_b_full = self._mse(yb_val, Xb_val @ wb_full)
                backward_info = {
                    "y_train": yb_train,
                    "y_val": yb_val,
                    "X_train": Xb_train,
                    "X_val": Xb_val,
                    "meta": meta_b,
                    "w_full": wb_full,
                    "baseline_mse": mse_b_full,
                }

        self_kernel = self._extract_self_kernel(w_full, meta)
        source_kernels = self._extract_source_kernels(w_full, meta)

        source_delta_mse_forward: dict[int, float] = {}
        source_delta_mse_backward: dict[int, float] = {}
        source_asymmetry: dict[int, float] = {}
        source_confidence: dict[int, float] = {}

        for source in sorted(source_kernels.keys()):
            cols = meta["source_cols"][source]
            if len(cols) == 0:
                continue

            mse_drop_fwd = self._drop_source_and_score(
                y_train=y_train,
                y_val=y_val,
                X_train=X_train,
                X_val=X_val,
                meta=meta,
                cols_to_drop=cols,
                w_full=w_full,
            )
            delta_fwd = float(mse_drop_fwd - baseline_mse)
            source_delta_mse_forward[source] = delta_fwd

            if backward_info is not None:
                cols_b = backward_info["meta"]["source_cols"].get(source, [])
                if cols_b:
                    mse_drop_bwd = self._drop_source_and_score(
                        y_train=backward_info["y_train"],
                        y_val=backward_info["y_val"],
                        X_train=backward_info["X_train"],
                        X_val=backward_info["X_val"],
                        meta=backward_info["meta"],
                        cols_to_drop=cols_b,
                        w_full=backward_info["w_full"],
                    )
                    delta_bwd = float(mse_drop_bwd - backward_info["baseline_mse"])
                else:
                    delta_bwd = 0.0
            else:
                delta_bwd = 0.0

            source_delta_mse_backward[source] = delta_bwd
            asym = max(0.0, delta_fwd - delta_bwd)
            source_asymmetry[source] = asym
            source_confidence[source] = max(0.0, delta_fwd) + self.config.asymmetry_weight * asym

        return TargetFitSummary(
            target_dim=target,
            baseline_mse=baseline_mse,
            null_mse=null_mse,
            intercept=intercept,
            self_kernel=self_kernel,
            source_kernels=source_kernels,
            source_delta_mse_forward=source_delta_mse_forward,
            source_delta_mse_backward=source_delta_mse_backward,
            source_asymmetry=source_asymmetry,
            source_confidence=source_confidence,
        )

    def _build_target_problem(
        self,
        state_vectors: np.ndarray,
        target: int,
    ) -> tuple[np.ndarray, np.ndarray, dict, float]:
        """Build y, X and metadata for one target node."""
        n_frames, n_dims = state_vectors.shape
        max_back = max(self.config.max_lag, self.config.ar_lag)

        if self.config.standardize:
            state_vectors = self._zscore_matrix(state_vectors)

        rows: list[list[float]] = []
        y: list[float] = []

        source_cols: dict[int, list[int]] = {}
        source_lags: dict[int, list[int]] = {}
        self_cols: list[int] = []

        col_cursor = 0
        if self.config.ar_lag > 0:
            self_cols = list(range(col_cursor, col_cursor + self.config.ar_lag))
            col_cursor += self.config.ar_lag

        for source in range(n_dims):
            if source == target:
                continue
            source_cols[source] = list(range(col_cursor, col_cursor + self.config.max_lag))
            source_lags[source] = list(range(1, self.config.max_lag + 1))
            col_cursor += self.config.max_lag

        for t in range(max_back, n_frames):
            row: list[float] = []

            # Self autoregression
            for l in range(1, self.config.ar_lag + 1):
                row.append(float(state_vectors[t - l, target]))

            # All other sources over the allowed lag window
            for source in range(n_dims):
                if source == target:
                    continue
                for l in range(1, self.config.max_lag + 1):
                    row.append(float(state_vectors[t - l, source]))

            rows.append(row)
            y.append(float(state_vectors[t, target]))

        X = np.asarray(rows, dtype=float)
        y = np.asarray(y, dtype=float)

        intercept = 0.0
        if self.config.include_intercept:
            intercept = float(np.mean(y))
            y = y - intercept

        meta = {
            "target": target,
            "n_dims": n_dims,
            "self_cols": self_cols,
            "source_cols": source_cols,
            "source_lags": source_lags,
            "n_features": X.shape[1],
        }
        return y, X, meta, intercept

    def _solve_regularized(self, X: np.ndarray, y: np.ndarray, meta: dict) -> np.ndarray:
        """
        Solve
            argmin ||y - Xw||^2 + alpha_ridge ||w||^2 + alpha_smooth * lag_smoothness

        lag_smoothness is applied independently to each block (self block and each source block)
        using a discrete first-difference penalty.
        """
        p = X.shape[1]
        if p == 0:
            return np.zeros(0, dtype=float)

        xtx = X.T @ X
        xty = X.T @ y
        reg = self.config.alpha_ridge * np.eye(p, dtype=float)

        def add_smooth_block(cols: list[int]) -> None:
            if len(cols) < 2 or self.config.alpha_smooth <= 0:
                return
            lam = self.config.alpha_smooth
            for a, b in zip(cols[:-1], cols[1:]):
                reg[a, a] += lam
                reg[b, b] += lam
                reg[a, b] -= lam
                reg[b, a] -= lam

        add_smooth_block(meta["self_cols"])
        for source, cols in meta["source_cols"].items():
            add_smooth_block(cols)

        try:
            w = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xtx + reg) @ xty
        return np.asarray(w, dtype=float)

    def _drop_source_and_score(
        self,
        y_train: np.ndarray,
        y_val: np.ndarray,
        X_train: np.ndarray,
        X_val: np.ndarray,
        meta: dict,
        cols_to_drop: list[int],
        w_full: np.ndarray,
    ) -> float:
        """Measure validation MSE after removing one source block."""
        if len(cols_to_drop) == 0:
            return self._mse(y_val, X_val @ w_full)

        if self.config.refit_on_drop:
            keep_cols = [c for c in range(X_train.shape[1]) if c not in set(cols_to_drop)]
            if len(keep_cols) == 0:
                return self._mse(y_val, np.zeros_like(y_val))

            Xtr = X_train[:, keep_cols]
            Xva = X_val[:, keep_cols]
            meta_reduced = self._reduce_meta(meta, keep_cols)
            w_red = self._solve_regularized(Xtr, y_train, meta_reduced)
            return self._mse(y_val, Xva @ w_red)

        w_masked = w_full.copy()
        w_masked[cols_to_drop] = 0.0
        return self._mse(y_val, X_val @ w_masked)

    # ---------------------------------------------------------------------
    # Score composition
    # ---------------------------------------------------------------------

    def _compose_score_matrix(
        self,
        block_norm: np.ndarray,
        delta_mse: np.ndarray,
        confidence: np.ndarray,
    ) -> np.ndarray:
        """
        Build the final continuous score matrix.

        Default behavior:
          score = score_mix * normalized(block_norm)
                + (1-score_mix) * normalized(positive delta_mse)
        with optional confidence contribution blended inside delta component.
        """
        bn = self._normalize_positive(block_norm)
        dm = self._normalize_positive(np.maximum(delta_mse, 0.0))
        cf = self._normalize_positive(np.maximum(confidence, 0.0))

        delta_part = (1.0 - self.config.confidence_mix) * dm + self.config.confidence_mix * cf
        score = self.config.score_mix * bn + (1.0 - self.config.score_mix) * delta_part
        return np.asarray(score, dtype=float)

    # ---------------------------------------------------------------------
    # Link extraction / pruning / textbook filters
    # ---------------------------------------------------------------------

    def _links_from_matrices(
        self,
        score_matrix: np.ndarray,
        lag_matrix: np.ndarray,
        sign_matrix: np.ndarray,
        block_norm_matrix: np.ndarray,
        delta_mse_matrix: np.ndarray,
        confidence_matrix: np.ndarray,
        dimension_names: list[str],
    ) -> list[InverseCausalLink]:
        n_dims = score_matrix.shape[0]
        links: list[InverseCausalLink] = []
        for source in range(n_dims):
            for target in range(n_dims):
                if source == target:
                    continue
                strength = float(score_matrix[source, target])
                if strength <= self.config.min_effect:
                    continue
                links.append(
                    InverseCausalLink(
                        from_dim=source,
                        to_dim=target,
                        from_name=dimension_names[source],
                        to_name=dimension_names[target],
                        strength=strength,
                        signed_peak=float(sign_matrix[source, target]),
                        best_lag=int(lag_matrix[source, target]),
                        block_norm=float(block_norm_matrix[source, target]),
                        delta_mse_forward=float(delta_mse_matrix[source, target]),
                        delta_mse_backward=0.0,  # reconstructed later if needed
                        asymmetry=0.0,
                        confidence=float(confidence_matrix[source, target]),
                    )
                )
        return links

    def _prune_by_confidence(self, links: list[InverseCausalLink]) -> list[InverseCausalLink]:
        if not links:
            return links
        confs = np.array([max(0.0, link.confidence) for link in links], dtype=float)
        cutoff = float(np.quantile(confs, self.config.confidence_quantile))
        cutoff = max(cutoff, self.config.min_effect)
        return [link for link in links if link.confidence >= cutoff]

    def _apply_textbook_filter(self, links: list[InverseCausalLink]) -> list[InverseCausalLink]:
        """
        Remove likely indirect links using two classic patterns.

        Pattern 1: Common ancestor
            z->a and z->b exist, and a->b is weaker with lag consistency.

        Pattern 2: Mediation
            a->m and m->b exist, and their lag sum approximately matches a->b.

        This filter is intentionally conservative and only removes weaker links.
        """
        if len(links) < 3:
            return links

        link_map: dict[tuple[int, int], InverseCausalLink] = {}
        for link in links:
            key = (link.from_dim, link.to_dim)
            if key not in link_map or link.strength > link_map[key].strength:
                link_map[key] = link

        retained: list[InverseCausalLink] = []

        for link in links:
            a, b = link.from_dim, link.to_dim
            remove = False

            # Common ancestor: z->a and z->b both stronger than a->b
            for (z, x), z_to_a in link_map.items():
                if x != a or z in (a, b):
                    continue
                z_to_b = link_map.get((z, b))
                if z_to_b is None:
                    continue

                stronger = (
                    z_to_a.strength >= link.strength * self.config.common_ancestor_strength_ratio
                    and z_to_b.strength >= link.strength * self.config.common_ancestor_strength_ratio
                )
                lag_consistent = abs((z_to_a.best_lag + link.best_lag) - z_to_b.best_lag) <= self.config.lag_tolerance

                if stronger and lag_consistent:
                    remove = True
                    break

            # Mediation: a->m and m->b explain a->b
            if not remove:
                for (src, m), a_to_m in link_map.items():
                    if src != a or m in (a, b):
                        continue
                    m_to_b = link_map.get((m, b))
                    if m_to_b is None:
                        continue

                    path_strength = min(a_to_m.strength, m_to_b.strength)
                    strength_ok = path_strength >= link.strength * self.config.mediated_path_strength_ratio
                    lag_ok = abs((a_to_m.best_lag + m_to_b.best_lag) - link.best_lag) <= self.config.lag_tolerance

                    if strength_ok and lag_ok:
                        remove = True
                        break

            if not remove:
                retained.append(link)

        return retained

    def _rebuild_matrices_from_links(
        self,
        links: list[InverseCausalLink],
        n_dims: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        score = np.zeros((n_dims, n_dims), dtype=float)
        lag = np.zeros((n_dims, n_dims), dtype=int)
        sign = np.zeros((n_dims, n_dims), dtype=float)
        block_norm = np.zeros((n_dims, n_dims), dtype=float)
        delta_mse = np.zeros((n_dims, n_dims), dtype=float)
        confidence = np.zeros((n_dims, n_dims), dtype=float)

        for link in links:
            i, j = link.from_dim, link.to_dim
            score[i, j] = link.strength
            lag[i, j] = link.best_lag
            sign[i, j] = link.signed_peak
            block_norm[i, j] = link.block_norm
            delta_mse[i, j] = link.delta_mse_forward
            confidence[i, j] = link.confidence

        return score, lag, sign, block_norm, delta_mse, confidence

    # ---------------------------------------------------------------------
    # Kernel extraction helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _extract_self_kernel(w: np.ndarray, meta: dict) -> np.ndarray:
        cols = meta["self_cols"]
        if not cols:
            return np.zeros(0, dtype=float)
        return np.asarray(w[cols], dtype=float)

    @staticmethod
    def _extract_source_kernels(w: np.ndarray, meta: dict) -> dict[int, np.ndarray]:
        out: dict[int, np.ndarray] = {}
        for source, cols in meta["source_cols"].items():
            out[source] = np.asarray(w[cols], dtype=float)
        return out

    @staticmethod
    def _reduce_meta(meta: dict, keep_cols: list[int]) -> dict:
        mapping = {old: new for new, old in enumerate(keep_cols)}
        self_cols = [mapping[c] for c in meta["self_cols"] if c in mapping]
        source_cols = {}
        for source, cols in meta["source_cols"].items():
            new_cols = [mapping[c] for c in cols if c in mapping]
            if new_cols:
                source_cols[source] = new_cols
        return {
            "target": meta["target"],
            "n_dims": meta["n_dims"],
            "self_cols": self_cols,
            "source_cols": source_cols,
            "source_lags": meta["source_lags"],
            "n_features": len(keep_cols),
        }

    # ---------------------------------------------------------------------
    # Numerical helpers
    # ---------------------------------------------------------------------

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_true) == 0:
            return 0.0
        diff = y_true - y_pred
        return float(np.mean(diff * diff))

    @staticmethod
    def _normalize_positive(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        x = np.maximum(x, 0.0)
        xmax = float(np.max(x)) if x.size > 0 else 0.0
        if xmax <= 0:
            return np.zeros_like(x)
        return x / xmax

    @staticmethod
    def _zscore_matrix(x: np.ndarray) -> np.ndarray:
        mu = np.mean(x, axis=0, keepdims=True)
        sd = np.std(x, axis=0, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return (x - mu) / sd


# ============================================================================
# Benchmark-friendly convenience function
# ============================================================================


def predict_adjacency(data: np.ndarray, max_lag: int) -> np.ndarray:
    """
    CauseMe-style convenience entrypoint.

    Parameters
    ----------
    data : np.ndarray, shape (T, N)
        Multivariate time-series.
    max_lag : int
        Benchmark-provided maximum lag.

    Returns
    -------
    np.ndarray, shape (N, N)
        Continuous adjacency score matrix. Entry [i, j] represents i -> j.
    """
    engine = InverseCausalEngine(
        InverseCausalEngineConfig(
            max_lag=max_lag,
            ar_lag=1,
            alpha_ridge=1e-2,
            alpha_smooth=1e-2,
            standardize=True,
            include_intercept=True,
            validation_fraction=0.25,
            min_train_size=max(20, 2 * max_lag),
            score_mix=0.70,
            confidence_mix=0.35,
            asymmetry_weight=0.25,
            use_backward_check=True,
            refit_on_drop=False,
            prune_by_confidence=False,
            apply_textbook_filter=True,
        )
    )
    return engine.fit_predict(data)


# ============================================================================
# Usage sketch
# ============================================================================
#
# config = InverseCausalEngineConfig(
#     max_lag=5,
#     ar_lag=1,
#     alpha_ridge=1e-2,
#     alpha_smooth=1e-2,
#     standardize=True,
#     validation_fraction=0.25,
#     apply_textbook_filter=True,
# )
#
# engine = InverseCausalEngine(config)
# result = engine.fit(state_vectors, dimension_names=[...])
# scores = result.score_matrix          # continuous AUC-ready scores, i -> j
# best_lags = result.lag_matrix         # best lag per directed edge
# links = result.links                  # sorted edge list
#
# For CauseMe-like interfaces:
# scores = predict_adjacency(state_vectors, max_lag=5)
# ============================================================================
