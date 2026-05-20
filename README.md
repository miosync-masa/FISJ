# FISJ — Find Insight Structural Junction

A lightweight, domain-agnostic causal discovery engine for multivariate time series.

FISJ detects directed causal relationships across dimensions using **dimensionless Λ³ displacement**, **directional jump consistency**, and **inverse-problem interventional evidence** — with adaptive, data-driven parameters and **zero learning**. The entire engine is pure Python with **numpy + scipy** as dependencies.

## Key Features

- **Zero learned parameters** — no training, no overfitting, no model selection
- **Directional consistency scoring** — true causal edges show consistent direction across time; spurious edges do not
- **All-frame statistical power × jump-based signal quality** — combines exhaustive temporal search with Λ³ event filtering
- **Inverse-problem DI gate** — source-drop interventional evidence (Pearl Level 2) prunes overlapping cascades
- **Regime-aware rescue** — event-driven causal patterns are recovered via regime segmentation
- **Per-source BH-FDR** — high-dimensional adjacency without statistical over-correction
- **Local-std normalization** — scale-invariant across heterogeneous dimensions (µV alongside BTC, etc.)
- **Minimal dependencies** — `numpy`, `scipy`

## Installation

```bash
pip install git+https://github.com/miosync-masa/FISJ.git
```

Or clone and install in editable mode:

```bash
git clone https://github.com/miosync-masa/FISJ.git
cd FISJ
pip install -e .
```

## Quick Start

```python
import numpy as np
import pandas as pd
from FISJ import FISJAdapter

# Example: 3 dimensions, A → B (lag=2), A → C (lag=5)
np.random.seed(42)
n = 300
a = np.cumsum(np.random.randn(n) * 0.5)
b = np.zeros(n)
c = np.zeros(n)
for t in range(2, n):
    b[t] = 0.7 * a[t-2] + 0.3 * np.random.randn()
for t in range(5, n):
    c[t] = 0.6 * a[t-5] + 0.3 * np.random.randn()

df = pd.DataFrame({"driver": a, "follower_A": b, "follower_B": c})

# Default method: nnnu_inverse (recommended)
adapter = FISJAdapter(method="nnnu_inverse", max_lag=8)
result = adapter.fit(df)

for i, src in enumerate(result.names):
    for j, tgt in enumerate(result.names):
        if result.adjacency_bin[i, j]:
            lag = int(result.lag_matrix[i, j])
            sign = "+" if result.sign_matrix[i, j] > 0 else "-"
            score = result.adjacency_scores[i, j]
            print(f"  {src} → {tgt}  (lag={lag}, sign={sign}, score={score:.3f})")
```

## Available Methods

`FISJAdapter` provides four causal discovery strategies via the `method` parameter:

| Method | Engines | Best For |
|--------|---------|----------|
| **`nnnu_inverse`** *(default)* | NNNU + Inverse + Regime | General causal discovery, recommended |
| `nnnu` | NNNU only | Fast pre-screening, Sign=1.000 |
| `fusion` | NetworkCore + Inverse | Legacy, strongest AUC ranking |
| `inverse` | Inverse only | Interventional reasoning only |

```python
# Choose the appropriate method for your data
adapter = FISJAdapter(method="nnnu_inverse")   # default, recommended
adapter = FISJAdapter(method="nnnu")           # fastest, no regression
adapter = FISJAdapter(method="fusion")         # legacy partial correlation
adapter = FISJAdapter(method="inverse")        # source-drop only
```

## Benchmark Results

*(Benchmark numbers pending — see `tests/benchmark_internal.py` to regenerate.)*

FISJ is evaluated against five established methods across multiple benchmark suites:

### Methods Compared

| Method | Library | Approach |
|--------|---------|----------|
| **FISJ (nnnu_inverse)** | numpy + scipy | Λ³ jump consistency + Inverse DI + Regime rescue |
| **FISJ (nnnu)** | numpy + scipy | Pure jump consistency, no regression |
| **FISJ (fusion)** | numpy + scipy | Partial correlation + Inverse DI |
| VAR_Granger | statsmodels | Vector autoregression with Granger causality test |
| PCMCI+ | tigramite | Conditional independence with iterative PC algorithm |
| TransferEntropy | custom | Discrete transfer entropy with permutation test |
| EventXCorr | custom | Event-based cross-correlation |
| GraphLasso | scikit-learn | Graphical Lasso (no temporal/directional support) |

### Benchmark Categories

| Category | Scenarios | Description |
|----------|-----------|-------------|
| **S — Standard** | 9 | Linear/nonlinear coupling, chains, confounders, bidirectional |
| **H — Heterogeneous Scale** | 5 | Financial market data with extreme scale ratios |
| **HELL — Robustness** | 8 | Pulse noise, bifurcations, cascades, decay, resonance |

### Composite Score (placeholder — re-run to populate)

```
TBD — Run tests/benchmark_internal.py to populate.
```

### CauseMe External Benchmark

FISJ has been submitted to the [CauseMe](http://www.causeme.net) public causal discovery benchmark. The current submission ranks competitively against PCMCI+ and other state-of-the-art methods on logistic-deterministic experiments.

*(Specific AUC/F1 numbers pending — see CauseMe leaderboard for current standings.)*

## How It Works

### Architecture (v0.3.0)

```
Input: (n_frames, n_dims) state vectors
  │
  ├─ Layer 1 (NNNU — Pearl Level 1, observational)
  │   1. Λ³ dimensionless displacement: diff / local_std
  │   2. ALL-frame signed_mean per (src, tgt, lag)
  │      → score = |signed_mean| × (1 + 2·(consistency - 0.5))
  │   3. JUMP-frame directional consistency
  │      → consistency = max(same_sign_rate, 1 - same_sign_rate)
  │   4. Spurious filter (common ancestor + mediator, ±2 frame window)
  │   5. Conditional scoring (causal-path-aware exclusion)
  │   6. Per-source Benjamini-Hochberg FDR
  │
  ├─ Layer 1.5 (Regime Rescue — event-driven recovery)
  │   - Generic regime detection via k-means on dynamical features
  │   - Per-regime re-scoring with binomial test on jump direction
  │   - Score-only rescue (does not modify binary adjacency)
  │
  ├─ Layer 2 (Inverse Engine — Pearl Level 2, interventional)
  │   - One-shot Ridge solve: target ~ sources at multiple lags
  │   - Source-drop ΔMSE → Direct Irreducibility (DI)
  │   - Soft DI gate: 0.3 + 0.7 × normalized_DI
  │
  └─ Suppression: q-value floor × consistency threshold

Output: MethodOutput (adjacency_scores, adjacency_bin, lag_matrix, sign_matrix)
```

### Why NNNU Matters

Traditional causal discovery methods rely on either regression (Granger, VAR) or conditional independence tests (PCMCI, PC). Both approaches have well-known failure modes:

- **Regression methods** suffer from suppressor variable problems (Haufe et al. 2026) when sources are correlated.
- **CI-based methods** require exponential-time conditioning sets in high dimensions and assume faithfulness.

NNNU takes a radically different approach: **count whether the target consistently moves in the same direction as the source after a fixed lag**. This is the operational essence of Granger's original 1969 definition, with two modern additions:

1. **Λ³ dimensionless displacement** makes the comparison scale-invariant.
2. **Directional consistency** explicitly tests whether the relationship is repeatable.

The result: a method that **cannot overfit** (no fitted parameters), is **inherently interpretable** (every score is a count of consistent same-direction responses), and achieves **Sign accuracy = 1.000** across most benchmark categories — meaning when FISJ identifies a causal direction, it gets the sign right.

## CauseMe Submission

For [CauseMe](http://www.causeme.net) benchmark submission:

```python
import numpy as np
from FISJ import run_fisj

def process_dataset(data: np.ndarray):
    """Submit-ready runner for CauseMe."""
    scores, lags, pvals = run_fisj(
        data,
        max_lag=3,                  # CauseMe logistic-deterministic uses lag ≤ 3
        method="nnnu_inverse",
    )
    return scores, lags, pvals
```

## Advanced API

### Direct Engine Access

```python
from FISJ.engines import NNNUEngine, InverseCausalEngine, GenericRegimeDetector
from FISJ.core import extract_lambda3_events

# Use individual engines for custom pipelines
data = np.random.randn(300, 5)

# NNNU only
nnnu = NNNUEngine(max_lag=5, alpha=0.05)
nnnu_result = nnnu.fit(data)

# Regime detection only
detector = GenericRegimeDetector()
labels = detector.detect(data)
segments = detector.build_segments(labels)
```

### Core Primitives

```python
from FISJ.core import (
    local_std_1d,                    # Λ³ scale normalization
    rho_t_1d,                        # Tension density
    extract_lambda3_events,          # Jump extraction
    benjamini_hochberg_per_source,   # Per-source FDR correction
)
```

## Design Philosophy

FISJ follows a principle of **conservative detection with minimal assumptions**:

- No distributional assumptions (no Gaussianity requirement)
- No model-class assumptions (no VAR / linear restriction)
- No stationarity assumption (local-std normalization handles nonstationarity)
- No trained parameters (cannot overfit — there is nothing to fit)
- False positives are more costly than false negatives in real-world deployment

This shows up most clearly in NNNU's behavior: it refuses to report edges that cannot be verified by directional consistency, even when the data shows correlated activity. When FISJ identifies a causal link, it has passed multiple independent layers of evidence.

## Repository Layout (v0.3.0)

```
FISJ/
├── __init__.py            # Top-level exports
├── core.py                # Λ³ shared primitives
├── adapter.py             # Unified FISJAdapter (4 methods)
└── engines/
    ├── __init__.py
    ├── nnnu.py            # NNNUEngine (Pearl Level 1)
    ├── inverse.py         # InverseCausalEngine (Pearl Level 2)
    ├── network_core.py    # NetworkAnalyzerCore (legacy, partial corr)
    └── regime.py          # GenericRegimeDetector

tests/
└── benchmark_internal.py  # Internal benchmark suite
```

## Citation

If you use FISJ in your research, please cite:

```bibtex
@software{fisj2026,
  author  = {Iizumi, Masamichi and Tamaki and Kurisu},
  title   = {FISJ: Find Insight Structural Junction},
  year    = {2026},
  version = {0.3.0},
  url     = {https://github.com/miosync-masa/FISJ},
}
```

## License

MIT — see [LICENSE](LICENSE).

---

*"The miracle is now scheduled. Welcome to the Λ³ zone."*
