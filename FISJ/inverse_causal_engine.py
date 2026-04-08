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
class DataGate:
    """Pre-scan data characteristics for adaptive pipeline routing."""

    mean_corr: float          # 次元間平均絶対相関
    max_corr: float           # 最大ペア相関
    n_dims: int
    n_frames: int
    feature_sample_ratio: float  # 特徴量数 / サンプル数
    regime: str               # "weak", "moderate", "strong"


@dataclass
class InverseCausalResult:
    """Full engine output."""

    links: list[InverseCausalLink]
    score_matrix: np.ndarray
    score_matrix_unfiltered: np.ndarray  # フィルタ前（AUC用）
    lag_matrix: np.ndarray
    sign_matrix: np.ndarray
    confidence_matrix: np.ndarray
    block_norm_matrix: np.ndarray
    delta_mse_matrix: np.ndarray
    gate: DataGate
    target_summaries: list[TargetFitSummary]
    dimension_names: list[str]
    max_lag: int
    ar_lag: int
    # --- Direct Irreducibility (post-solver layer) ---
    necessity_matrix: np.ndarray | None = None
    indirect_support_matrix: np.ndarray | None = None
    direct_irreducibility_matrix: np.ndarray | None = None
    direct_score_matrix: np.ndarray | None = None


@dataclass
class InverseCausalEngineConfig:
    """Configuration for the inverse-problem causal engine."""

    max_lag: int
    ar_lag: int = 1
    alpha_ridge: float = 1e-2
    alpha_smooth: float = 1e-2
    solver: str = "lasso"  # "ridge", "lasso", "auto"
    lasso_cv_folds: int = 5
    lasso_max_iter: int = 5000
    adaptive_regularization: bool = True  # N/T依存の正則化
    standardize: bool = True
    include_intercept: bool = True
    validation_fraction: float = 0.25
    min_train_size: int = 40
    min_effect: float = 1e-6
    score_mode: str = "block_norm"  # "block_norm", "delta_mse", "mixed"
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
    residualize_ar: bool = True  # 自己回帰を先に引いてからcross-node solve
    eps: float = 1e-12
    # --- Direct Irreducibility ---
    compute_direct_irreducibility: bool = True
    di_alpha_raw: float = 0.35
    di_lambda_indirect: float = 1.0
    di_lag_tau: float = 1.0


# ============================================================================
# Direct Irreducibility Scorer (post-solver layer)
# ============================================================================


class DirectIrreducibilityScorer:
    """
    Re-score edges by *necessity* and *indirect substitutability*.

    Does NOT re-solve the inverse problem.  Takes the existing
    score_matrix_unfiltered, lag_matrix, delta_mse_matrix and
    target_summaries, then computes:

      1) Necessity  N_ij  — how much validation error increases when
         source i is dropped from target j's model.
      2) Indirect support  I_ij  — strongest lag-consistent two-step
         mediator path that could replace edge i→j.
      3) Direct irreducibility  D_ij = N / (N + λI + ε)
      4) Final direct score  S_dir = (α·raw_norm + (1-α)·nec_norm) · D
    """

    def __init__(
        self,
        alpha_raw: float = 0.35,
        lambda_indirect: float = 1.0,
        lag_tau: float = 1.0,
        eps: float = 1e-12,
    ):
        self.alpha_raw = alpha_raw
        self.lambda_indirect = lambda_indirect
        self.lag_tau = lag_tau
        self.eps = eps

    def compute(
        self,
        score_matrix_unfiltered: np.ndarray,
        lag_matrix: np.ndarray,
        delta_mse_matrix: np.ndarray,
        target_summaries: list,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (necessity, indirect_support, irreducibility, direct_score)."""
        raw = score_matrix_unfiltered.copy()
        lag = lag_matrix.copy()
        delta = np.maximum(delta_mse_matrix.copy(), 0.0)
        n_dims = raw.shape[0]

        # 1) Necessity
        necessity = np.zeros_like(raw)
        for target in range(n_dims):
            s = target_summaries[target]
            denom = max(self.eps, s.null_mse - s.baseline_mse)
            necessity[:, target] = delta[:, target] / denom
        necessity = np.maximum(necessity, 0.0)
        np.fill_diagonal(necessity, 0.0)

        # 2) Indirect support
        indirect = np.zeros_like(raw)
        for i in range(n_dims):
            for j in range(n_dims):
                if i == j:
                    continue
                best = 0.0
                lij = lag[i, j]
                for m in range(n_dims):
                    if m == i or m == j:
                        continue
                    sim, smj = raw[i, m], raw[m, j]
                    if sim <= 0.0 or smj <= 0.0:
                        continue
                    path_str = min(sim, smj)
                    lag_gap = abs(lij - (lag[i, m] + lag[m, j]))
                    support = path_str * np.exp(
                        -lag_gap / max(self.lag_tau, self.eps)
                    )
                    if support > best:
                        best = support
                indirect[i, j] = best
        np.fill_diagonal(indirect, 0.0)

        # 3) Direct irreducibility
        irreducibility = necessity / (
            necessity + self.lambda_indirect * indirect + self.eps
        )
        irreducibility = np.clip(irreducibility, 0.0, 1.0)
        np.fill_diagonal(irreducibility, 0.0)

        # 4) Final direct score
        raw_norm = self._normalize(raw)
        nec_norm = self._normalize(necessity)
        mixed = self.alpha_raw * raw_norm + (1.0 - self.alpha_raw) * nec_norm
        direct_score = mixed * irreducibility
        np.fill_diagonal(direct_score, 0.0)

        return necessity, indirect, irreducibility, direct_score

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        xmax = x.max()
        return x / xmax if xmax > 0 else np.zeros_like(x)


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

        # --- Gate: データ特性のpre-scan ---
        gate = self._compute_gate(state_vectors)
        logger.info(
            f"Gate: regime={gate.regime} mean_corr={gate.mean_corr:.3f} "
            f"N={gate.n_dims} T={gate.n_frames} feat/sample={gate.feature_sample_ratio:.2f}"
        )

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

        # フィルタ前のscoreを保存（AUC用連続値ランキング）
        score_unfiltered = score.copy()

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

        # --- Direct Irreducibility (post-solver layer) ---
        nec_mat = ind_mat = irr_mat = dir_mat = None
        if self.config.compute_direct_irreducibility:
            scorer = DirectIrreducibilityScorer(
                alpha_raw=self.config.di_alpha_raw,
                lambda_indirect=self.config.di_lambda_indirect,
                lag_tau=max(self.config.di_lag_tau, float(self.config.lag_tolerance)),
                eps=self.config.eps,
            )
            nec_mat, ind_mat, irr_mat, dir_mat = scorer.compute(
                score_matrix_unfiltered=score_unfiltered,
                lag_matrix=lag,
                delta_mse_matrix=delta_mse,
                target_summaries=summaries,
            )
            logger.info(
                f"   🎯 Direct Irreducibility: "
                f"mean_D={irr_mat.mean():.4f}, "
                f"max_D={irr_mat.max():.4f}, "
                f"nonzero={np.count_nonzero(dir_mat)}"
            )

        return InverseCausalResult(
            links=links,
            score_matrix=score,
            score_matrix_unfiltered=score_unfiltered,
            lag_matrix=lag,
            sign_matrix=sign,
            confidence_matrix=confidence,
            block_norm_matrix=block_norm,
            delta_mse_matrix=delta_mse,
            gate=gate,
            target_summaries=summaries,
            dimension_names=dimension_names,
            max_lag=self.config.max_lag,
            ar_lag=self.config.ar_lag,
            necessity_matrix=nec_mat,
            indirect_support_matrix=ind_mat,
            direct_irreducibility_matrix=irr_mat,
            direct_score_matrix=dir_mat,
        )

    def fit_predict(self, state_vectors: np.ndarray, dimension_names: Optional[list[str]] = None) -> np.ndarray:
        """Convenience method returning the continuous score matrix only."""
        return self.fit(state_vectors, dimension_names).score_matrix

    def predict_adjacency(self, state_vectors: np.ndarray, dimension_names: Optional[list[str]] = None) -> np.ndarray:
        """Alias for benchmark-style usage."""
        return self.fit_predict(state_vectors, dimension_names)

    # ---------------------------------------------------------------------
    # Gate: data pre-scan
    # ---------------------------------------------------------------------

    def _compute_gate(self, state_vectors: np.ndarray) -> DataGate:
        """
        Pre-scan data characteristics before solving.

        Computes correlation structure to determine regime:
          weak:     mean_corr < 0.10
          moderate: 0.10 <= mean_corr < 0.20
          strong:   mean_corr >= 0.20
        """
        n_frames, n_dims = state_vectors.shape

        # Correlation matrix
        corr = np.corrcoef(state_vectors.T)
        upper = corr[np.triu_indices(n_dims, k=1)]
        mean_corr = float(np.mean(np.abs(upper)))
        max_corr = float(np.max(np.abs(upper))) if len(upper) > 0 else 0.0

        # Feature/sample ratio
        n_features = self.config.ar_lag + (n_dims - 1) * self.config.max_lag
        n_samples = n_frames - max(self.config.max_lag, self.config.ar_lag)
        feat_ratio = n_features / max(n_samples, 1)

        # Regime classification
        if mean_corr < 0.10:
            regime = "weak"
        elif mean_corr < 0.20:
            regime = "moderate"
        else:
            regime = "strong"

        return DataGate(
            mean_corr=mean_corr,
            max_corr=max_corr,
            n_dims=n_dims,
            n_frames=n_frames,
            feature_sample_ratio=feat_ratio,
            regime=regime,
        )

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

        # --- Two-stage AR residualization ---
        if self.config.residualize_ar and meta["self_cols"]:
            self_cols = meta["self_cols"]
            X_self_train = X_train[:, self_cols]
            X_self_val = X_val[:, self_cols]

            # Stage 1: fit self-AR
            w_ar = np.linalg.lstsq(X_self_train, y_train, rcond=None)[0]
            self_kernel = w_ar.copy()

            # Compute residuals
            y_train_resid = y_train - X_self_train @ w_ar
            y_val_resid = y_val - X_self_val @ w_ar

            # Stage 2: solve cross-node on residuals (no self cols)
            source_col_list = []
            for cols in meta["source_cols"].values():
                source_col_list.extend(cols)

            if source_col_list:
                X_src_train = X_train[:, source_col_list]
                X_src_val = X_val[:, source_col_list]

                # Remap meta for source-only solve
                meta_src = self._reduce_meta(meta, source_col_list)
                meta_src["self_cols"] = []  # no self in this stage

                w_src = self._solve_regularized(X_src_train, y_train_resid, meta_src)

                # Reconstruct full weight vector
                w_full = np.zeros(X_full.shape[1])
                w_full[self_cols] = w_ar
                for idx, col in enumerate(source_col_list):
                    w_full[col] = w_src[idx]

                yhat_val = X_val @ w_full
                baseline_mse = self._mse(y_val, yhat_val)
            else:
                w_full = np.zeros(X_full.shape[1])
                w_full[self_cols] = w_ar
                baseline_mse = self._mse(y_val, X_self_val @ w_ar)

            null_mse = self._mse(y_val, np.full_like(y_val, np.mean(y_train)))

        else:
            # --- Original single-stage solve ---
            w_full = self._solve_regularized(X_train, y_train, meta)
            yhat_val = X_val @ w_full
            baseline_mse = self._mse(y_val, yhat_val)
            null_mse = self._mse(y_val, np.full_like(y_val, np.mean(y_train)))
            self_kernel = self._extract_self_kernel(w_full, meta)

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

        # self_kernel は上のif/elseで設定済み
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

        solver modes:
          "ridge" : Ridge regression (L2, all coefficients shrink)
          "lasso" : LassoCV (L1, automatic sparsity + cross-validated alpha)
          "auto"  : Lasso if n_dims > 5, else Ridge
        """
        n_samples = X.shape[0]
        p = X.shape[1]
        if p == 0:
            return np.zeros(0, dtype=float)

        # Determine solver
        solver = self.config.solver
        if solver == "auto":
            n_dims = meta.get("n_dims", 3)
            solver = "lasso" if n_dims > 5 else "ridge"

        # --- Lasso path ---
        if solver == "lasso":
            return self._solve_lasso(X, y)

        # --- Ridge path ---
        if self.config.adaptive_regularization and n_samples > 0 and p > 0:
            adaptive_scale = float(np.sqrt(np.log(max(p, 2)) / max(n_samples, 1)))
            alpha_ridge = self.config.alpha_ridge * adaptive_scale / 0.1
            alpha_smooth = self.config.alpha_smooth * adaptive_scale / 0.1
        else:
            alpha_ridge = self.config.alpha_ridge
            alpha_smooth = self.config.alpha_smooth

        xtx = X.T @ X
        xty = X.T @ y
        reg = alpha_ridge * np.eye(p, dtype=float)

        def add_smooth_block(cols: list[int]) -> None:
            if len(cols) < 2 or alpha_smooth <= 0:
                return
            for a, b in zip(cols[:-1], cols[1:]):
                reg[a, a] += alpha_smooth
                reg[b, b] += alpha_smooth
                reg[a, b] -= alpha_smooth
                reg[b, a] -= alpha_smooth

        add_smooth_block(meta["self_cols"])
        for source, cols in meta["source_cols"].items():
            add_smooth_block(cols)

        try:
            w = np.linalg.solve(xtx + reg, xty)
        except np.linalg.LinAlgError:
            w = np.linalg.pinv(xtx + reg) @ xty
        return np.asarray(w, dtype=float)

    def _solve_lasso(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Solve with LassoCV (L1 regularization with cross-validated alpha).

        L1 naturally produces sparse solutions — noise sources get exactly
        zero coefficients, while true causal sources retain non-zero weights.
        """
        from sklearn.linear_model import LassoCV

        try:
            lasso = LassoCV(
                cv=self.config.lasso_cv_folds,
                max_iter=self.config.lasso_max_iter,
                n_jobs=-1,
            ).fit(X, y)
            return np.asarray(lasso.coef_, dtype=float)
        except Exception:
            # Fallback to Ridge if Lasso fails
            p = X.shape[1]
            xtx = X.T @ X
            xty = X.T @ y
            reg = self.config.alpha_ridge * np.eye(p)
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

        score_mode controls the composition:
          "block_norm" : block norm only (simplest)
          "delta_mse"  : delta MSE only (prediction importance)
          "mixed"      : weighted combination of all components
          "anomaly"    : block_norm with anomaly detection
                         (noise suppression + signal boost)
        """
        mode = self.config.score_mode

        if mode == "block_norm":
            return self._normalize_positive(block_norm)

        if mode == "delta_mse":
            return self._normalize_positive(np.maximum(delta_mse, 0.0))

        if mode == "anomaly":
            return self._anomaly_score(block_norm)

        # mixed (default fallback)
        bn = self._normalize_positive(block_norm)
        dm = self._normalize_positive(np.maximum(delta_mse, 0.0))
        cf = self._normalize_positive(np.maximum(confidence, 0.0))

        delta_part = (1.0 - self.config.confidence_mix) * dm + self.config.confidence_mix * cf
        score = self.config.score_mix * bn + (1.0 - self.config.score_mix) * delta_part
        return np.asarray(score, dtype=float)

    @staticmethod
    def _anomaly_score(block_norm: np.ndarray) -> np.ndarray:
        """
        Anomaly-based scoring: treat each target's incoming block_norms
        as a distribution, and identify signal (outliers) vs noise.

        For each target column j:
          - compute median and MAD of incoming block_norms
          - z_robust = (bn - median) / MAD
          - signal: z > 2 → boost (keep original + extra)
          - noise:  z < 1 → suppress (shrink toward zero)
          - border: linear interpolation

        This naturally adapts to the number of dimensions:
          N=3:  2 sources → outlier is obvious
          N=10: 9 sources → noise floor is well-estimated

        Result is re-normalized to [0, 1].
        """
        n_dims = block_norm.shape[0]
        score = np.zeros_like(block_norm)

        for target in range(n_dims):
            # Incoming block_norms for this target
            col = block_norm[:, target].copy()
            col[target] = 0.0  # self is always zero

            # Get non-zero values (sources only)
            sources = [i for i in range(n_dims) if i != target]
            vals = col[sources]

            if len(vals) < 2 or np.max(vals) < 1e-10:
                continue

            # Robust statistics
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med)))
            if mad < 1e-10:
                mad = float(np.std(vals)) * 0.6745  # fallback
            if mad < 1e-10:
                # All values nearly identical → no signal
                continue

            for src in sources:
                bn = col[src]
                z = (bn - med) / mad

                if z > 3.0:
                    # Strong signal → boost
                    score[src, target] = bn * (1.0 + z * 0.5)
                elif z > 1.5:
                    # Moderate signal → keep with mild boost
                    blend = (z - 1.5) / 1.5  # 0→1
                    score[src, target] = bn * (1.0 + blend * z * 0.3)
                elif z > 0.5:
                    # Border → keep but no boost
                    score[src, target] = bn * 0.5
                else:
                    # Noise → suppress
                    score[src, target] = bn * 0.1

        # Normalize to [0, 1]
        smax = np.max(score)
        if smax > 0:
            score = score / smax

        return score

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
                        delta_mse_backward=0.0,
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
            adaptive_regularization=True,
            standardize=True,
            include_intercept=True,
            validation_fraction=0.25,
            min_train_size=max(20, 2 * max_lag),
            score_mode="block_norm",
            asymmetry_weight=0.25,
            use_backward_check=True,
            refit_on_drop=False,
            prune_by_confidence=False,
            apply_textbook_filter=False,  # AUC用はフィルタなし
        )
    )
    result = engine.fit(data)
    return result.score_matrix_unfiltered
