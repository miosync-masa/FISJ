"""
CauseMe-Compatible Local Benchmark
===================================
CauseMeと同じ指標（AUC, AUC-PR, F-measure, FPR, TPR, TLR）で
逆問題エンジンを手動テストする。

CauseMeの100件/日制限を回避して高速にアブレーション可能。

Usage:
    python causeme_local_bench.py
"""

import sys
import time
import logging
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

logging.disable(logging.CRITICAL)

from inverse_causal_engine import InverseCausalEngine, InverseCausalEngineConfig


# ============================================================
# CauseMe-style VAR data generator
# ============================================================

def generate_var_data(
    n_dims: int = 3,
    T: int = 300,
    max_lag: int = 5,
    coupling_strength: float = 0.3,
    noise_scale: float = 0.5,
    sparsity: float = 0.3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate VAR-style data similar to CauseMe's logistic-deterministic.

    Returns
    -------
    data : (T, n_dims)
    true_adj : (n_dims, n_dims) binary ground truth
    true_lag : (n_dims, n_dims) true lag for each edge
    """
    rng = np.random.default_rng(seed)

    # Generate sparse causal structure
    true_adj = np.zeros((n_dims, n_dims), dtype=int)
    true_lag = np.zeros((n_dims, n_dims), dtype=int)
    true_coeff = np.zeros((n_dims, n_dims))

    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                continue
            if rng.random() < sparsity:
                true_adj[i, j] = 1
                true_lag[i, j] = rng.integers(1, max_lag + 1)
                true_coeff[i, j] = rng.choice([-1, 1]) * coupling_strength * rng.uniform(0.5, 1.5)

    # Generate data
    data = np.zeros((T + 50, n_dims))  # burn-in
    for t in range(max_lag, T + 50):
        for j in range(n_dims):
            # Self autoregression
            data[t, j] = 0.5 * data[t - 1, j]
            # Causal influences
            for i in range(n_dims):
                if true_adj[i, j] == 1:
                    lag = true_lag[i, j]
                    if t - lag >= 0:
                        data[t, j] += true_coeff[i, j] * data[t - lag, i]
            # Noise
            data[t, j] += rng.normal(scale=noise_scale)

    return data[50:], true_adj, true_lag


# ============================================================
# CauseMe-style scenario definitions
# ============================================================

SCENARIOS = {}

# Logistic-deterministic equivalent
for n in [3, 5, 10]:
    for T in [150, 300]:
        for strength_name, strength in [("weak", 0.15), ("medium", 0.3), ("strong", 0.6)]:
            name = f"logistic_N-{n}_T-{T}_{strength_name}"
            SCENARIOS[name] = {
                "n_dims": n, "T": T, "max_lag": 5,
                "coupling_strength": strength,
                "noise_scale": 0.5, "sparsity": 0.3,
            }

# Largenoise equivalent
for n in [10]:
    for T in [300]:
        for strength_name, strength in [("weak", 0.15), ("medium", 0.3), ("strong", 0.6)]:
            name = f"largenoise_N-{n}_T-{T}_{strength_name}"
            SCENARIOS[name] = {
                "n_dims": n, "T": T, "max_lag": 5,
                "coupling_strength": strength,
                "noise_scale": 2.0, "sparsity": 0.3,
            }


# ============================================================
# CauseMe-compatible evaluation metrics
# ============================================================

@dataclass
class CauseMeMetrics:
    """CauseMeと同じ指標セット"""
    AUC: float = 0.0
    AUC_PR: float = 0.0
    F_measure: float = 0.0
    FPR: float = 0.0
    TPR: float = 0.0
    TLR: float = 0.0  # True Lag Rate


def evaluate_causeme(
    scores: np.ndarray,
    lags: np.ndarray,
    true_adj: np.ndarray,
    true_lag: np.ndarray,
) -> CauseMeMetrics:
    """
    CauseMeと同じ方法で評価。

    - AUC: scores のランキング評価
    - AUC-PR: precision-recall AUC
    - F-measure: 最適閾値でのF1
    - FPR/TPR: 最適閾値での偽陽性率/真陽性率
    - TLR: 検出したエッジのうちラグが正しい割合
    """
    n_dims = scores.shape[0]
    mask = ~np.eye(n_dims, dtype=bool)

    y_true = true_adj[mask].flatten()
    y_scores = scores[mask].flatten()

    # Handle edge cases
    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return CauseMeMetrics()

    if np.max(y_scores) == np.min(y_scores):
        return CauseMeMetrics()

    # AUC
    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.5

    # AUC-PR
    try:
        auc_pr = average_precision_score(y_true, y_scores)
    except ValueError:
        auc_pr = 0.0

    # F-measure at optimal threshold
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_scores)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-10)
    best_idx = np.argmax(f1_arr)

    if best_idx < len(thresholds):
        best_threshold = thresholds[best_idx]
    else:
        best_threshold = 0.5

    y_pred = (y_scores >= best_threshold).astype(int)
    f_measure = float(f1_score(y_true, y_pred, zero_division=0))

    # FPR, TPR
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # TLR: True Lag Rate (detected edges with correct lag)
    detected_true = (y_pred == 1) & (y_true == 1)
    pred_adj = np.zeros_like(true_adj)
    pred_adj[mask] = y_pred

    n_correct_lag = 0
    n_detected_true = 0
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j:
                continue
            if pred_adj[i, j] == 1 and true_adj[i, j] == 1:
                n_detected_true += 1
                if lags[i, j] == true_lag[i, j]:
                    n_correct_lag += 1

    tlr = n_correct_lag / n_detected_true if n_detected_true > 0 else 0.0

    return CauseMeMetrics(
        AUC=float(auc),
        AUC_PR=float(auc_pr),
        F_measure=float(f_measure),
        FPR=float(fpr),
        TPR=float(tpr),
        TLR=float(tlr),
    )


# ============================================================
# Benchmark runner
# ============================================================

def run_benchmark(
    config_overrides: dict | None = None,
    n_repeats: int = 5,
    scenarios: dict | None = None,
):
    """
    全シナリオでベンチマーク実行。

    Parameters
    ----------
    config_overrides : dict
        InverseCausalEngineConfig のフィールドを上書き
    n_repeats : int
        各シナリオの繰り返し回数
    scenarios : dict
        テストするシナリオ（Noneなら全部）
    """
    if scenarios is None:
        scenarios = SCENARIOS

    print("=" * 80)
    print(f"  CauseMe-Compatible Local Benchmark")
    print(f"  {n_repeats} repeats × {len(scenarios)} scenarios")
    if config_overrides:
        print(f"  Overrides: {config_overrides}")
    print("=" * 80)

    all_results = []

    for sname, sparams in scenarios.items():
        metrics_list = []
        t0 = time.time()

        for rep in range(n_repeats):
            data, true_adj, true_lag = generate_var_data(
                seed=42 + rep * 7, **sparams
            )

            max_lag = sparams["max_lag"]
            base_config = dict(
                max_lag=max_lag,
                ar_lag=1,
                alpha_ridge=1e-2,
                alpha_smooth=1e-2,
                adaptive_regularization=False,
                standardize=True,
                include_intercept=True,
                validation_fraction=0.25,
                min_train_size=max(20, 2 * max_lag),
                score_mode="mixed",
                score_mix=0.70,
                confidence_mix=0.35,
                asymmetry_weight=0.25,
                use_backward_check=True,
                refit_on_drop=False,
                prune_by_confidence=False,
                residualize_ar=False,
                apply_textbook_filter=True,
            )

            if config_overrides:
                base_config.update(config_overrides)

            config = InverseCausalEngineConfig(**base_config)
            engine = InverseCausalEngine(config)

            try:
                result = engine.fit(
                    data,
                    dimension_names=[f"V{i}" for i in range(sparams["n_dims"])],
                )

                scores = result.score_matrix_unfiltered.copy()
                np.fill_diagonal(scores, 0.0)
                lags = result.lag_matrix.copy()

                m = evaluate_causeme(scores, lags, true_adj, true_lag)
                metrics_list.append(m)
            except Exception as e:
                print(f"  ⚠️ {sname} rep={rep}: {e}")

        elapsed = time.time() - t0

        if metrics_list:
            avg = CauseMeMetrics(
                AUC=np.mean([m.AUC for m in metrics_list]),
                AUC_PR=np.mean([m.AUC_PR for m in metrics_list]),
                F_measure=np.mean([m.F_measure for m in metrics_list]),
                FPR=np.mean([m.FPR for m in metrics_list]),
                TPR=np.mean([m.TPR for m in metrics_list]),
                TLR=np.mean([m.TLR for m in metrics_list]),
            )
        else:
            avg = CauseMeMetrics()

        all_results.append({"name": sname, "metrics": avg, "time": elapsed})

        print(
            f"  {sname:45s} | AUC={avg.AUC:.3f} PR={avg.AUC_PR:.3f} "
            f"F1={avg.F_measure:.3f} FPR={avg.FPR:.3f} TPR={avg.TPR:.3f} "
            f"TLR={avg.TLR:.3f} | {elapsed:.1f}s"
        )

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY BY CONDITION")
    print("=" * 80)

    # Group by strength
    for strength in ["weak", "medium", "strong"]:
        subset = [r for r in all_results if strength in r["name"]]
        if subset:
            avg_auc = np.mean([r["metrics"].AUC for r in subset])
            avg_pr = np.mean([r["metrics"].AUC_PR for r in subset])
            avg_f1 = np.mean([r["metrics"].F_measure for r in subset])
            print(f"  {strength:10s}: AUC={avg_auc:.3f} AUC-PR={avg_pr:.3f} F1={avg_f1:.3f}")

    # Group by N
    for n in [3, 5, 10]:
        subset = [r for r in all_results if f"N-{n}_" in r["name"]]
        if subset:
            avg_auc = np.mean([r["metrics"].AUC for r in subset])
            print(f"  N={n:2d}      : AUC={avg_auc:.3f}")

    # Group by T
    for T in [150, 300]:
        subset = [r for r in all_results if f"T-{T}_" in r["name"]]
        if subset:
            avg_auc = np.mean([r["metrics"].AUC for r in subset])
            print(f"  T={T:3d}     : AUC={avg_auc:.3f}")

    return all_results


# ============================================================
# Ablation helper
# ============================================================

def run_ablation(n_repeats: int = 5, scenarios: dict | None = None):
    """
    複数設定を自動で比較するアブレーション。
    """
    configs = {
        "baseline": {},
        "residualize_ar": {"residualize_ar": True},
        "block_norm": {"score_mode": "block_norm"},
        "adaptive_reg": {"adaptive_regularization": True},
        "residualize+block": {"residualize_ar": True, "score_mode": "block_norm"},
        "no_filter": {"apply_textbook_filter": False},
    }

    # デフォルトは strong のみ（速度のため）
    if scenarios is None:
        scenarios = {
            k: v for k, v in SCENARIOS.items()
            if "strong" in k and "T-300" in k
        }

    print("=" * 80)
    print("  ABLATION STUDY")
    print(f"  Scenarios: {list(scenarios.keys())}")
    print("=" * 80)

    results = {}
    for config_name, overrides in configs.items():
        print(f"\n--- {config_name} ---")
        r = run_benchmark(
            config_overrides=overrides,
            n_repeats=n_repeats,
            scenarios=scenarios,
        )
        avg_auc = np.mean([x["metrics"].AUC for x in r])
        avg_pr = np.mean([x["metrics"].AUC_PR for x in r])
        results[config_name] = {"auc": avg_auc, "auc_pr": avg_pr, "detail": r}

    print("\n" + "=" * 80)
    print("  ABLATION RESULTS")
    print("=" * 80)
    for name, r in sorted(results.items(), key=lambda x: x[1]["auc"], reverse=True):
        print(f"  {name:25s}: AUC={r['auc']:.3f} AUC-PR={r['auc_pr']:.3f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bench", choices=["bench", "ablation"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--strong-only", action="store_true")
    args = parser.parse_args()

    if args.strong_only:
        scenarios = {k: v for k, v in SCENARIOS.items() if "strong" in k}
    else:
        scenarios = None

    if args.mode == "ablation":
        run_ablation(n_repeats=args.repeats, scenarios=scenarios)
    else:
        run_benchmark(n_repeats=args.repeats, scenarios=scenarios)
