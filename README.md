# FISJ — Find Insight Structural Junction

A lightweight, domain-agnostic causal discovery engine for multivariate time series.

FISJ detects synchronization and causal relationships across dimensions using partial correlation, lagged causality estimation, and common-ancestor filtering — with adaptive, data-driven thresholds. The entire engine is ~600 lines of Python with **numpy as the only dependency**.

## Key Features

- **Partial correlation via precision matrix** — suppresses confounding variables automatically
- **Lagged causal estimation with multi-lag residual conditioning** — detects directed, time-delayed causality
- **Common-ancestor filter** — removes spurious edges induced by shared upstream drivers
- **Adaptive thresholds** — sync/causal thresholds and max lag are tuned from five data-driven diagnostics (volatility, temporal variation, correlation complexity, nonstationarity, spectral structure)
- **Local-std normalization** — displacement signals are normalized by rolling local standard deviation, enabling scale-invariant analysis across heterogeneous dimensions (e.g., µV and mmHg in the same dataset)
- **Minimal dependencies** — `numpy` only (core); `pandas` for the optional benchmark adapter

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
from FISJ import NetworkAnalyzerCore

# Example: 3 dimensions, A → B (lag=2), A → C (lag=5)
np.random.seed(42)
n = 200
a = np.cumsum(np.random.randn(n) * 0.5)
b = np.zeros(n)
c = np.zeros(n)
for t in range(2, n):
    b[t] = 0.7 * a[t-2] + 0.3 * np.random.randn()
for t in range(5, n):
    c[t] = 0.6 * a[t-5] + 0.3 * np.random.randn()

state = np.column_stack([a, b, c])

analyzer = NetworkAnalyzerCore(adaptive=True)
result = analyzer.analyze(state, dimension_names=["driver", "follower_A", "follower_B"])

print(f"Pattern: {result.pattern}")
print(f"Drivers: {result.driver_names}")
for link in result.causal_network:
    print(f"  {link.from_name} → {link.to_name} (lag={link.lag}, strength={link.strength:.3f})")
```

Output:
```
Pattern: cascade
Drivers: ['driver']
  driver → follower_A (lag=2, strength=0.632)
  driver → follower_B (lag=5, strength=0.320)
```

## Benchmark Adapter

For integration with causal discovery benchmarks:

```python
import pandas as pd
from FISJ import FISJAdapter

adapter = FISJAdapter()
result = adapter.fit(df)  # pd.DataFrame with time series columns

result.adjacency_bin   # (n, n) binary adjacency matrix
result.lag_matrix      # (n, n) estimated lag matrix
result.sign_matrix     # (n, n) sign of causal coupling
```

## Benchmark Results

FISJ was evaluated against five established methods across four benchmark suites (20 repeats each). All benchmarks use synthetic data with known ground-truth causal structures.

### Methods Compared

| Method | Library | Approach |
|--------|---------|----------|
| **FISJ** | numpy (~600 lines) | Partial correlation + lagged causality + common-ancestor filter |
| VAR_Granger | statsmodels | Vector autoregression with Granger causality test |
| PCMCI+ | tigramite | Conditional independence with iterative PC algorithm |
| TransferEntropy | custom | Discrete transfer entropy with permutation test |
| EventXCorr | custom | Event-based cross-correlation |
| GraphLasso | scikit-learn | Graphical Lasso (no temporal/directional support) |

### Composite Scores

| Category | Scenarios | FISJ | VAR | PCMCI+ | TE | EventXCorr | GraphLasso |
|----------|-----------|------|-----|--------|-----|------------|------------|
| **S — Standard** | 9 | **0.874** | 0.863 | 0.860 | 0.466 | 0.525 | 0.152 |
| **H — Heterogeneous Scale** | 5 | **0.706** | 0.672 | 0.602 | 0.312 | 0.402 | 0.123 |
| **HELL — Robustness** | 8 | **0.857** | 0.642 | 0.632 | 0.425 | 0.298 | 0.120 |
| **M — Medical Vitals** | 7 | 0.661 | 0.554 | **0.674** | 0.332 | 0.330 | 0.127 |

FISJ ranks **#1 in three out of four categories** and #2 in the fourth (Medical, within 0.013 of PCMCI+).

### Robustness Under Adversarial Conditions (HELL Mode)

The HELL benchmark injects pulse noise, phase jumps, bifurcations, cascading failures, resonance distortion, structural decay, and combinations thereof into causally structured data. This tests whether methods maintain accuracy when signal quality degrades severely.

| Metric | FISJ | VAR_Granger | PCMCI+ |
|--------|------|-------------|--------|
| Composite | **0.857** | 0.642 | 0.632 |
| F1 (directed) | **0.758** | 0.507 | 0.476 |
| Lag MAE | **0.095** | 0.468 | 0.537 |
| Sign accuracy | **0.974** | 0.952 | 0.943 |
| Spurious rate | **0.028** | 0.383 | 0.297 |

Under adversarial conditions, FISJ's lead over the second-place method expands from +0.011 (Standard) to **+0.215** (HELL). The gap is driven by FISJ's near-zero spurious rate (0.028 vs 0.297–0.383 for competitors), indicating that the common-ancestor filter and local-std normalization provide strong resilience to distributional corruption.

### Heterogeneous Scale (H Series)

The H series uses financial-market-inspired data with scale ratios up to 260,000× (e.g., EUR/USD at 0.003 vs BTC at 800.0). VAR_Granger produces numerical instability errors (`leading minor not positive definite`) on these datasets. FISJ handles them without modification due to internal local-std normalization.

### Medical Vital Signs (M Series)

Seven neurophysiologically validated scenarios (designed with clinical physiology references) test causal discovery across heterogeneous vital signs: EEG (µV), ECG (mV), temperature (°C), heart rate (bpm), blood pressure (mmHg), and SpO2 (%). Scenarios include fever cascades, hemorrhagic shock, seizure cascades, autonomic arousal, drug response, and gradual deterioration (early sepsis).

Per-scenario results (F1 directed):

| Scenario | FISJ | PCMCI+ | VAR | Clinical Significance |
|----------|------|--------|-----|----------------------|
| M1 Fever cascade | **0.627** | 0.416 | 0.470 | Thermoregulatory reflex |
| M2 Hemorrhagic shock | **0.741** | 0.646 | 0.631 | Emergency triage |
| M3 Autonomic arousal | 0.238 | **0.299** | 0.124 | Weak EEG→HR coupling |
| M4 Drug response | **0.659** | 0.604 | 0.574 | Pharmacological monitoring |
| M5 Seizure cascade | 0.663 | **0.711** | 0.521 | Neurological emergency |
| M6 Gradual deterioration | **0.508** | 0.377 | 0.401 | Early warning (sub-threshold) |

FISJ wins 4 of 6 causal scenarios. PCMCI+ leads in M3 and M5 (EEG-origin with weak coupling). FISJ's advantage is largest in M2 (shock) and M6 (gradual deterioration) — the two scenarios most relevant to clinical early warning systems.

## How It Works

### Architecture

```
Input: (n_frames, n_dims) state vectors
  │
  ├─ 1. Local Std Computation
  │     Rolling window → per-frame, per-dim scale estimate
  │
  ├─ 2. Displacement & Normalization
  │     diff(state) / local_std → dimensionless displacement
  │
  ├─ 3. Synchronous Partial Correlation
  │     Precision matrix → pcorr(i,j) = -P[i,j]/√(P[i,i]·P[j,j])
  │
  ├─ 4. Lagged Partial Correlation
  │     Multi-lag residual conditioning → directed causal links
  │
  ├─ 5. Common-Ancestor Filter
  │     If Z→A and Z→B both stronger than A→B, remove A→B
  │
  └─ 6. Network Assembly
        Pattern identification, hub detection, driver/follower roles

Output: NetworkResult (sync_network, causal_network, matrices, metadata)
```

### Adaptive Parameter Tuning

Five signal diagnostics are computed from the input data and used to adjust analysis parameters:

| Diagnostic | Drives | Effect |
|------------|--------|--------|
| Global volatility | sync_threshold | High → stricter threshold |
| Temporal volatility | causal_threshold | High → stricter threshold |
| Correlation complexity | causal_threshold, max_lag | High → relax threshold, extend lag |
| Local nonstationarity | sync_threshold | High → stricter threshold |
| Spectral low-freq ratio | max_lag | High → longer lag range |

## API Reference

### `NetworkAnalyzerCore`

```python
NetworkAnalyzerCore(
    sync_threshold=0.5,      # Baseline sync threshold (adaptive hint)
    causal_threshold=0.4,    # Baseline causal threshold (adaptive hint)
    max_lag=12,              # Baseline max lag (adaptive hint)
    adaptive=True,           # Enable data-driven parameter tuning
    local_std_window=20,     # Rolling window for local std
)
```

#### `.analyze(state_vectors, dimension_names=None, window=None)`

Run network analysis on multivariate time series.

Returns `NetworkResult` with:
- `sync_network` / `causal_network` — lists of `DimensionLink`
- `sync_matrix` / `causal_matrix` / `causal_lag_matrix` — raw correlation matrices
- `pattern` — `"parallel"`, `"cascade"`, `"mixed"`, or `"independent"`
- `hub_dimensions` / `hub_names` — highly connected dimensions
- `causal_drivers` / `causal_followers` — directed role assignments

### `FISJAdapter`

Benchmark-compatible adapter. Takes a `pd.DataFrame`, returns `MethodOutput` with adjacency, lag, and sign matrices.

## Design Philosophy

FISJ follows a principle of **conservative detection with minimal assumptions**:

- No distributional assumptions (no Gaussianity requirement)
- No model class assumptions (no VAR/linear restriction)
- No stationarity assumption (local-std normalization handles nonstationarity)
- False positives are more costly than false negatives in real-world applications

This is reflected in the benchmark results: FISJ consistently achieves the lowest spurious rate across all conditions (0.013–0.058), meaning **when FISJ reports a causal link, it is almost certainly real**.

## Citation

If you use FISJ in your research, please cite:

```bibtex
@software{fisj2025,
  author = {Iizumi, Masamichi,Tamaki,Kurisu},
  title = {FISJ: Find Insight Structural Junction},
  year = {2025},
  url = {https://github.com/miosync-masa/FISJ}
}
```

## License

MIT
