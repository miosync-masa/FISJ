"""
benchmark
"""

import math
import time
import warnings

import numpy as np
import pandas as pd
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sklearn.covariance import GraphicalLassoCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler

_HAS_STATSMODELS = False
try:
    from statsmodels.tsa.api import VAR

    _HAS_STATSMODELS = True
except ImportError:
    pass

_HAS_TIGRAMITE = False
try:
    import tigramite.data_processing as pp  # noqa: F401
    from tigramite.independence_tests.parcorr import ParCorr  # noqa: F401
    from tigramite.pcmci import PCMCI  # noqa: F401

    _HAS_TIGRAMITE = True
except ImportError:
    pass

from FISJ.adapter import FISJInverseAdapter
from FISJ.inverse_causal_engine import InverseCausalEngine, InverseCausalEngineConfig
from FISJ.main import NetworkAnalyzerCore
from FISJ.score_fusion import fuse_scores

# ── Data Structures ──
# =====================================================================
# Data structures (from benchmark framework)
# =====================================================================

@dataclass
class ScenarioConfig:
    T: int = 400
    burn_in: int = 100
    n_series: int = 3
    max_lag: int = 8
    noise_scale: float = 0.35
    regime_split: int = 200
    event_percentile: float = 97.0
    zero_inflation_prob: float = 0.88
    seed: int = 42

@dataclass
class GroundTruth:
    names: list[str]
    adjacency: np.ndarray
    lag_matrix: np.ndarray | None = None
    sign_matrix: np.ndarray | None = None
    notes: str = ""
    regime_boundaries: list[int] | None = None
    adjacency_by_regime: list[np.ndarray] | None = None
    lag_by_regime: list[np.ndarray] | None = None
    sign_by_regime: list[np.ndarray] | None = None
    low_corr_edges: list[tuple[int, int]] | None = None
    forbidden_edges: list[tuple[int, int]] | None = None

    def undirected_adjacency(self) -> np.ndarray:
        return ((self.adjacency + self.adjacency.T) > 0).astype(int)

@dataclass
class MethodOutput:
    method_name: str
    names: list[str]
    adjacency_scores: np.ndarray | None = None
    adjacency_bin: np.ndarray | None = None
    directed_support: bool = True
    lag_support: bool = False
    sign_support: bool = False
    regime_support: bool = False
    lag_matrix: np.ndarray | None = None
    sign_matrix: np.ndarray | None = None
    regime_boundaries: list[int] | None = None
    adjacency_by_regime: list[np.ndarray] | None = None
    lag_by_regime: list[np.ndarray] | None = None
    sign_by_regime: list[np.ndarray] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def undirected_bin(self) -> np.ndarray | None:
        if self.adjacency_bin is None:
            return None
        return ((self.adjacency_bin + self.adjacency_bin.T) > 0).astype(int)


# =====================================================================
# Utility functions
# =====================================================================

def _rng(seed): return np.random.default_rng(seed)

def _diff_events(x, percentile=97.0):
    d = np.diff(x, prepend=x[0])
    thr = np.percentile(np.abs(d), percentile)
    return (np.abs(d) > thr).astype(int)

def _shifted_overlap(a, b, lag):
    if lag > 0: return a[:-lag], b[lag:]
    if lag < 0: return a[-lag:], b[:lag]
    return a, b

def _quantile_discretize(x, n_bins=4):
    x = np.asarray(x)
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if len(edges) <= 2:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        if xmin == xmax: return np.zeros_like(x, dtype=int)
        edges = np.linspace(xmin, xmax, n_bins + 1)
    return np.digitize(x, edges[1:-1], right=False).astype(int)

def _te_discrete(source, target, lag=1, n_bins=4):
    if lag < 1: raise ValueError("lag must be >= 1")
    x = _quantile_discretize(source, n_bins)
    y = _quantile_discretize(target, n_bins)
    t0 = max(lag, 1)
    y_t, y_prev, x_lag = y[t0:], y[t0-1:-1], x[t0-lag:len(x)-lag]
    n = min(len(y_t), len(y_prev), len(x_lag))
    y_t, y_prev, x_lag = y_t[:n], y_prev[:n], x_lag[:n]
    if n <= 5: return 0.0
    c_xyz = Counter(zip(x_lag.tolist(), y_prev.tolist(), y_t.tolist()))
    c_xy = Counter(zip(x_lag.tolist(), y_prev.tolist()))
    c_yz = Counter(zip(y_prev.tolist(), y_t.tolist()))
    c_y = Counter(y_prev.tolist())
    te = 0.0
    for (xs, yp, yt), c in c_xyz.items():
        p_xyz = c / n
        p1 = c / c_xy[(xs, yp)]
        p2 = c_yz[(yp, yt)] / c_y[yp]
        if p1 > 0 and p2 > 0: te += p_xyz * math.log(p1 / p2)
    return float(te)

def _event_sync_score(a_events, b_events, lag):
    aa, bb = _shifted_overlap(a_events, b_events, lag)
    return float(np.mean(aa * bb)) if len(aa) else 0.0

def _binarize_adjacency_from_scores(scores, threshold=None, percentile=None, symmetric=False):
    s = np.asarray(scores).copy()
    np.fill_diagonal(s, 0.0)
    vals = s[~np.eye(s.shape[0], dtype=bool)]
    if threshold is None:
        percentile = percentile or 75.0
        threshold = float(np.percentile(vals, percentile)) if len(vals) else 0.0
    out = (s >= threshold).astype(int)
    np.fill_diagonal(out, 0)
    if symmetric:
        out = ((out + out.T) > 0).astype(int)
        np.fill_diagonal(out, 0)
    return out

def _precision_recall_f1(y_true, y_pred):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

def _boundary_f1(true_b, pred_b, tolerance=5):
    true_b, pred_b = list(true_b or []), list(pred_b or [])
    if not true_b and not pred_b: return 1.0
    if not true_b or not pred_b: return 0.0
    mt, mp = set(), set()
    for i, tb in enumerate(true_b):
        for j, pb in enumerate(pred_b):
            if abs(tb - pb) <= tolerance and j not in mp:
                mt.add(i); mp.add(j); break
    tp = len(mt); fp = len(pred_b) - tp; fn = len(true_b) - tp
    p = tp/(tp+fp) if (tp+fp) else 0.0
    r = tp/(tp+fn) if (tp+fn) else 0.0
    return 2*p*r/(p+r) if (p+r) else 0.0


# =====================================================================
# Scenario generators
# =====================================================================

SCALES = {
    "EUR_USD": 0.003,
    "USD_JPY": 0.5,
    "Gold": 15.0,
    "Nikkei": 200.0,
    "BTC": 800.0,
    "VIX": 3.0,
    "Oil": 1.5,
    "SP500": 20.0,
}

SCALE_NAMES = list(SCALES.keys())
SCALE_VALUES = list(SCALES.values())


def generate_hetero_chain(cfg, seed):
    rng = _rng(seed); T = cfg.T + cfg.burn_in; n = 8
    x = np.zeros((T, n)); scales = SCALE_VALUES
    for t in range(1, T):
        x[t, 0] = 0.70 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale * scales[0])
        b_from_a = 1.10 * x[t - 2, 0] / scales[0] * scales[4] if t - 2 >= 0 else 0.0
        x[t, 4] = 0.45 * x[t - 1, 4] + b_from_a + rng.normal(scale=cfg.noise_scale * scales[4])
        c_from_b = 0.95 * x[t - 3, 4] / scales[4] * scales[3] if t - 3 >= 0 else 0.0
        x[t, 3] = 0.50 * x[t - 1, 3] + c_from_b + rng.normal(scale=cfg.noise_scale * scales[3])
        for d in [1, 2, 5, 6, 7]:
            x[t, d] = 0.60 * x[t - 1, d] + rng.normal(scale=cfg.noise_scale * scales[d])
    x = x[cfg.burn_in:]
    adj = np.zeros((n, n), dtype=int); adj[0, 4] = 1; adj[4, 3] = 1
    lagm = np.zeros((n, n), dtype=int); lagm[0, 4] = 2; lagm[4, 3] = 3
    signm = np.zeros((n, n), dtype=int); signm[0, 4] = 1; signm[4, 3] = 1
    return pd.DataFrame(x, columns=SCALE_NAMES), GroundTruth(
        names=SCALE_NAMES, adjacency=adj, lag_matrix=lagm, sign_matrix=signm,
        forbidden_edges=[(0, 3), (3, 0), (3, 4), (4, 0)], notes="Hetero chain EUR_USD→BTC→Nikkei",
    )


def generate_hetero_confounder(cfg, seed):
    rng = _rng(seed); T = cfg.T + cfg.burn_in; n = 8
    x = np.zeros((T, n)); scales = SCALE_VALUES
    for t in range(1, T):
        x[t, 5] = 0.80 * x[t - 1, 5] + rng.normal(scale=cfg.noise_scale * scales[5])
        g_from_v = 1.20 * x[t - 1, 5] / scales[5] * scales[2] if t >= 1 else 0.0
        x[t, 2] = 0.50 * x[t - 1, 2] + g_from_v + rng.normal(scale=cfg.noise_scale * scales[2])
        o_from_v = -1.05 * x[t - 2, 5] / scales[5] * scales[6] if t - 2 >= 0 else 0.0
        x[t, 6] = 0.55 * x[t - 1, 6] + o_from_v + rng.normal(scale=cfg.noise_scale * scales[6])
        for d in [0, 1, 3, 4, 7]:
            x[t, d] = 0.60 * x[t - 1, d] + rng.normal(scale=cfg.noise_scale * scales[d])
    x = x[cfg.burn_in:]
    adj = np.zeros((n, n), dtype=int); adj[5, 2] = 1; adj[5, 6] = 1
    lagm = np.zeros((n, n), dtype=int); lagm[5, 2] = 1; lagm[5, 6] = 2
    signm = np.zeros((n, n), dtype=int); signm[5, 2] = 1; signm[5, 6] = -1
    return pd.DataFrame(x, columns=SCALE_NAMES), GroundTruth(
        names=SCALE_NAMES, adjacency=adj, lag_matrix=lagm, sign_matrix=signm,
        forbidden_edges=[(2, 6), (6, 2)], notes="Hetero confounder VIX→Gold,Oil",
    )


def generate_hetero_bidirectional(cfg, seed):
    rng = _rng(seed); T = cfg.T + cfg.burn_in; n = 8
    x = np.zeros((T, n)); scales = SCALE_VALUES
    for t in range(1, T):
        sp_from_usd = 0.90 * x[t - 2, 1] / scales[1] * scales[7] if t - 2 >= 0 else 0.0
        usd_from_sp = 0.60 * x[t - 3, 7] / scales[7] * scales[1] if t - 3 >= 0 else 0.0
        x[t, 1] = 0.55 * x[t - 1, 1] + usd_from_sp + rng.normal(scale=cfg.noise_scale * scales[1])
        x[t, 7] = 0.50 * x[t - 1, 7] + sp_from_usd + rng.normal(scale=cfg.noise_scale * scales[7])
        for d in [0, 2, 3, 4, 5, 6]:
            x[t, d] = 0.60 * x[t - 1, d] + rng.normal(scale=cfg.noise_scale * scales[d])
    x = x[cfg.burn_in:]
    adj = np.zeros((n, n), dtype=int); adj[1, 7] = 1; adj[7, 1] = 1
    lagm = np.zeros((n, n), dtype=int); lagm[1, 7] = 2; lagm[7, 1] = 3
    signm = np.zeros((n, n), dtype=int); signm[1, 7] = 1; signm[7, 1] = 1
    return pd.DataFrame(x, columns=SCALE_NAMES), GroundTruth(
        names=SCALE_NAMES, adjacency=adj, lag_matrix=lagm, sign_matrix=signm,
        notes="Hetero bidirectional USD_JPY<->SP500",
    )


def generate_hetero_event_cascade(cfg, seed):
    rng = _rng(seed); T = cfg.T + cfg.burn_in; n = 8
    x = np.zeros((T, n)); scales = SCALE_VALUES
    for t in range(1, T):
        x[t, 0] = 0.70 * x[t - 1, 0] + rng.normal(scale=cfg.noise_scale * scales[0])
        if rng.random() < 0.04:
            x[t, 0] += rng.choice([-1, 1]) * rng.uniform(2.0, 4.0) * scales[0]
        x[t, 1] = 0.60 * x[t - 1, 1] + rng.normal(scale=cfg.noise_scale * scales[1])
        if t - 2 >= 1:
            eur_jump = abs(x[t - 2, 0] - x[t - 2 - 1, 0])
            if eur_jump > 1.5 * scales[0]:
                x[t, 1] += 0.8 * np.sign(x[t - 2, 0] - x[t - 2 - 1, 0]) / scales[0] * scales[1] * rng.uniform(1.5, 3.0)
        x[t, 3] = 0.55 * x[t - 1, 3] + rng.normal(scale=cfg.noise_scale * scales[3])
        if t - 3 >= 1:
            usd_jump = abs(x[t - 3, 1] - x[t - 3 - 1, 1])
            if usd_jump > 1.5 * scales[1]:
                x[t, 3] += 1.2 * np.sign(x[t - 3, 1] - x[t - 3 - 1, 1]) / scales[1] * scales[3] * rng.uniform(1.0, 2.0)
        for d in [2, 4, 5, 6, 7]:
            x[t, d] = 0.60 * x[t - 1, d] + rng.normal(scale=cfg.noise_scale * scales[d])
    x = x[cfg.burn_in:]
    adj = np.zeros((n, n), dtype=int); adj[0, 1] = 1; adj[1, 3] = 1
    lagm = np.zeros((n, n), dtype=int); lagm[0, 1] = 2; lagm[1, 3] = 3
    signm = np.zeros((n, n), dtype=int); signm[0, 1] = 1; signm[1, 3] = 1
    return pd.DataFrame(x, columns=SCALE_NAMES), GroundTruth(
        names=SCALE_NAMES, adjacency=adj, lag_matrix=lagm, sign_matrix=signm,
        forbidden_edges=[(0, 3), (3, 0), (3, 1), (1, 0)], notes="Hetero event cascade EUR→USD→Nikkei",
    )


def generate_hetero_null(cfg, seed):
    rng = _rng(seed); T = cfg.T + cfg.burn_in; n = 8
    x = np.zeros((T, n)); scales = SCALE_VALUES
    for t in range(1, T):
        for d in range(n):
            x[t, d] = 0.65 * x[t - 1, d] + rng.normal(scale=cfg.noise_scale * scales[d])
    x = x[cfg.burn_in:]
    return pd.DataFrame(x, columns=SCALE_NAMES), GroundTruth(
        names=SCALE_NAMES, adjacency=np.zeros((n, n), dtype=int),
        lag_matrix=np.zeros((n, n), dtype=int), sign_matrix=np.zeros((n, n), dtype=int),
        notes="Hetero null (independent)",
    )

def generate_null_independent(cfg, seed):
    rng = _rng(seed); n = cfg.n_series; T = cfg.T + cfg.burn_in
    x = np.zeros((T, n))
    for t in range(1, T): x[t] = 0.65*x[t-1] + rng.normal(scale=cfg.noise_scale, size=n)
    x = x[cfg.burn_in:]; names = [f"X{i+1}" for i in range(n)]
    return pd.DataFrame(x, columns=names), GroundTruth(names=names, adjacency=np.zeros((n,n),dtype=int), lag_matrix=np.zeros((n,n),dtype=int), sign_matrix=np.zeros((n,n),dtype=int), notes="Null")

def generate_delayed_directional(cfg, seed):
    rng = _rng(seed); T = cfg.T+cfg.burn_in; lag=3; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.75*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        x[t,2]=0.55*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
        eff=1.15*x[t-lag,0] if t-lag>=0 else 0.0
        x[t,1]=0.60*x[t-1,1]+eff+0.15*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=lag
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Delayed A->B")

def generate_asymmetric_coupling(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        x[t,1]=0.45*x[t-1,1]+1.35*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        x[t,0]+=0.08*x[t-1,1]
        x[t,2]=0.60*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","Noise"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=1
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Asymmetric A->B")

def generate_event_driven_delayed(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; lag=3; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        if rng.random()<0.04: x[t,0]+=rng.choice([-1,1])*rng.uniform(2.0,4.0)
        x[t,1]=0.60*x[t-1,1]+rng.normal(scale=cfg.noise_scale)
        if t-lag>=1:
            a_jump=abs(x[t-lag,0]-x[t-lag-1,0])
            if a_jump>1.5: x[t,1]+=0.8*np.sign(x[t-lag,0]-x[t-lag-1,0])*rng.uniform(1.5,3.0)
        x[t,2]=0.55*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=lag
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Event-driven delayed A->B")

def generate_event_driven_asymmetric(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        if rng.random()<0.05: x[t,0]+=rng.choice([-1,1])*rng.uniform(2.5,5.0)
        x[t,1]=0.50*x[t-1,1]+rng.normal(scale=cfg.noise_scale)
        if t>=2:
            a_jump=abs(x[t-1,0]-x[t-2,0])
            if a_jump>1.5: x[t,1]+=2.5*np.sign(x[t-1,0]-x[t-2,0])*rng.uniform(0.8,1.2)
        if t>=2:
            b_jump=abs(x[t-1,1]-x[t-2,1])
            if b_jump>1.5: x[t,0]+=0.05*np.sign(x[t-1,1]-x[t-2,1])
        x[t,2]=0.55*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","Noise"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=1
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Event asymmetric A->B")

def generate_confounder(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,2]=0.80*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
        x[t,0]=0.50*x[t-1,0]+1.20*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
        x[t,1]=0.55*x[t-1,1]+(-1.05*x[t-2,2] if t-2>=0 else 0.0)+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[2,0]=1; adj[2,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[2,0]=1; lagm[2,1]=2
    signm=np.zeros((3,3),dtype=int); signm[2,0]=1; signm[2,1]=-1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,forbidden_edges=[(0,1),(1,0)],notes="Confounder C->A,B")

def generate_bidirectional(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        a_from_b = 0.6*x[t-3,1] if t-3>=0 else 0.0
        b_from_a = 0.9*x[t-2,0] if t-2>=0 else 0.0
        x[t,0]=0.55*x[t-1,0]+a_from_b+rng.normal(scale=cfg.noise_scale)
        x[t,1]=0.50*x[t-1,1]+b_from_a+rng.normal(scale=cfg.noise_scale)
        x[t,2]=0.60*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","Noise"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1; adj[1,0]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=2; lagm[1,0]=3
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1; signm[1,0]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Bidirectional A<->B")

def generate_chain(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.70*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        b_from_a = 1.10*x[t-2,0] if t-2>=0 else 0.0
        x[t,1]=0.45*x[t-1,1]+b_from_a+rng.normal(scale=cfg.noise_scale)
        c_from_b = 0.95*x[t-2,1] if t-2>=0 else 0.0
        x[t,2]=0.50*x[t-1,2]+c_from_b+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","C"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1; adj[1,2]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=2; lagm[1,2]=2
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1; signm[1,2]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,forbidden_edges=[(0,2),(2,0),(2,1),(1,0)],notes="Chain A->B->C")

def generate_nonlinear(cfg, seed):
    rng=_rng(seed); T=cfg.T+cfg.burn_in; lag=2; x=np.zeros((T,3))
    for t in range(1,T):
        x[t,0]=0.65*x[t-1,0]+rng.normal(scale=cfg.noise_scale)
        a_sq = x[t-lag,0]**2 if t-lag>=0 else 0.0
        x[t,1]=0.45*x[t-1,1]+0.80*np.sign(x[t-lag,0])*a_sq+rng.normal(scale=cfg.noise_scale) if t-lag>=0 else 0.45*x[t-1,1]+rng.normal(scale=cfg.noise_scale)
        x[t,2]=0.60*x[t-1,2]+rng.normal(scale=cfg.noise_scale)
    x=x[cfg.burn_in:]; names=["A","B","Noise"]
    adj=np.zeros((3,3),dtype=int); adj[0,1]=1
    lagm=np.zeros((3,3),dtype=int); lagm[0,1]=lag
    signm=np.zeros((3,3),dtype=int); signm[0,1]=1
    return pd.DataFrame(x,columns=names), GroundTruth(names=names,adjacency=adj,lag_matrix=lagm,sign_matrix=signm,notes="Nonlinear A²->B")


# ============================================================
# Hell Mode Data Generator
# ============================================================

class HellModeGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _pulse(self, series, intensity=3.0):
        s = series.copy(); n = len(s)
        for idx in self.rng.choice(n, size=min(3, n), replace=False):
            s[idx] += self.rng.standard_normal() * intensity
            for offset in range(1, 4):
                if idx + offset < n: s[idx + offset] += s[idx] * np.exp(-0.5 * offset) * 0.3
        return s

    def _phase_jump(self, series, intensity=3.0):
        s = series.copy(); n = len(s)
        jump = self.rng.integers(n // 3, 2 * n // 3)
        s[jump:] += intensity * self.rng.choice([-1, 1])
        s[jump] += self.rng.standard_normal() * intensity * 2
        return s

    def _bifurcation(self, series, intensity=2.0):
        s = series.copy(); n = len(s); split = n // 2
        for i in range(split, n):
            t = (i - split) / (n - split)
            s[i] += intensity * np.sqrt(t) * self.rng.standard_normal() * (1 if i % 2 == 0 else -1)
        return s

    def _cascade(self, series, intensity=2.0):
        s = series.copy(); n = len(s); start = self.rng.integers(0, n // 2)
        s[start] += intensity * 3
        for i in range(start + 1, min(start + 15, n)):
            s[i] += s[i - 1] * 0.5 * np.exp(-0.3 * (i - start)) + self.rng.standard_normal() * 0.3
        return s

    def _resonance(self, series, intensity=2.0):
        s = series.copy(); fft = np.fft.fft(s)
        res_freq = self.rng.integers(1, max(2, len(fft) // 4))
        fft[res_freq] *= intensity * 3
        if len(fft) - res_freq > 0: fft[-res_freq] *= intensity * 3
        return np.real(np.fft.ifft(fft))

    def _structural_decay(self, series, intensity=2.0):
        s = series.copy(); n = len(s); mid = n // 2
        for i in range(mid, n):
            decay = np.exp(-intensity * (i - mid) / (n - mid))
            s[i] = s[i] * decay + (1 - decay) * self.rng.standard_normal() * np.std(s[:mid])
        return s

    def _multi_hell(self, series, intensity=3.0):
        return self._cascade(self._phase_jump(self._pulse(series, intensity), intensity * 0.5), intensity * 0.7)

    def _progressive_hell(self, series, intensity=3.0):
        s = series.copy(); n = len(s)
        for i in range(n):
            hf = (i / n) ** 2
            s[i] += self.rng.standard_normal() * intensity * hf
            if self.rng.random() < hf * 0.1: s[i] *= self.rng.choice([-1, 1]) * self.rng.uniform(2, 5)
        return s

    def generate_scenario(self, name, T=400, n_series=5):
        noise_scale = 0.5
        A, B, C = np.zeros(T), np.zeros(T), np.zeros(T)
        for t in range(T):
            A[t] = 0.6 * (A[t - 1] if t > 0 else 0) + self.rng.standard_normal() * noise_scale
            if t >= 2: B[t] = 0.5 * (B[t - 1] if t > 0 else 0) + 0.7 * A[t - 2] + self.rng.standard_normal() * noise_scale
            if t >= 3: C[t] = 0.4 * (C[t - 1] if t > 0 else 0) + 0.6 * A[t - 3] + self.rng.standard_normal() * noise_scale
        extras = []
        for _ in range(n_series - 3):
            s = np.zeros(T)
            for t in range(1, T): s[t] = 0.5 * s[t - 1] + self.rng.standard_normal() * noise_scale
            extras.append(s)
        all_series = [A, B, C] + extras
        hell_fns = {
            "H1_pulse": lambda s: self._pulse(s, 5.0), "H2_phase_jump": lambda s: self._phase_jump(s, 4.0),
            "H3_bifurcation": lambda s: self._bifurcation(s, 3.0), "H4_cascade": lambda s: self._cascade(s, 4.0),
            "H5_resonance": lambda s: self._resonance(s, 3.0), "H6_decay": lambda s: self._structural_decay(s, 2.0),
            "H7_multi_hell": self._multi_hell, "H8_progressive": self._progressive_hell,
        }
        corrupted = [np.real(hell_fns[name](s)).astype(np.float64) for s in all_series]
        data = np.nan_to_num(np.column_stack(corrupted), nan=0.0, posinf=1e6, neginf=-1e6)
        columns = [f"X{i}" for i in range(data.shape[1])]
        gt = {"edges": [("X0", "X1", 2), ("X0", "X2", 3)], "forbidden": [("X1", "X0"), ("X2", "X0"), ("X1", "X2"), ("X2", "X1")]}
        return pd.DataFrame(data=data, columns=columns, dtype=np.float64), gt

def _make_hell_scenario(hell_name):
    def generator(cfg, seed):
        gen = HellModeGenerator(seed=seed)
        df, gt_dict = gen.generate_scenario(hell_name, T=cfg.T, n_series=5)
        names = list(df.columns); n = len(names)
        adj = np.zeros((n, n), dtype=int); lagm = np.zeros((n, n), dtype=int); signm = np.zeros((n, n), dtype=int)
        name_to_idx = {name: i for i, name in enumerate(names)}
        for src, dst, lag in gt_dict["edges"]:
            i, j = name_to_idx[src], name_to_idx[dst]; adj[i, j] = 1; lagm[i, j] = lag; signm[i, j] = 1
        forbidden = [(name_to_idx[s], name_to_idx[d]) for s, d in gt_dict["forbidden"]]
        return df, GroundTruth(names=names, adjacency=adj, lag_matrix=lagm, sign_matrix=signm, forbidden_edges=forbidden, notes=f"Hell Mode: {hell_name}")
    return generator


# ── Scenario Registry (ON/OFF切り替え) ──

ALL_SCENARIOS = {
    "S0_null": generate_null_independent,
    "S1_delayed": generate_delayed_directional,
    "S2_asymmetric": generate_asymmetric_coupling,
    "S3_bidirectional": generate_bidirectional,
    "S4_chain": generate_chain,
    "S5_confounder": generate_confounder,
    "S6_nonlinear": generate_nonlinear,
    "S7_event_delayed": generate_event_driven_delayed,
    "S8_event_asym": generate_event_driven_asymmetric,
    "H0_hetero_null": generate_hetero_null,
    "H1_hetero_chain": generate_hetero_chain,
    "H2_hetero_confounder": generate_hetero_confounder,
    "H3_hetero_bidirectional": generate_hetero_bidirectional,
    "H4_hetero_event_cascade": generate_hetero_event_cascade,
    **{f"HELL_{name}": _make_hell_scenario(name)
       for name in ["H1_pulse", "H2_phase_jump", "H3_bifurcation",
                     "H4_cascade", "H5_resonance", "H6_decay",
                     "H7_multi_hell", "H8_progressive"]},
}

# ↓ ここでON/OFFを切り替え（コメントアウトで無効化）
ENABLED = {
    "S0_null",
    "S1_delayed",
    "S2_asymmetric",
    "S3_bidirectional",
    "S4_chain",
    "S5_confounder",
    "S6_nonlinear",
    "S7_event_delayed",
    "S8_event_asym",
    #"H0_hetero_null",
    #"H1_hetero_chain",
    #"H2_hetero_confounder",
    #"H3_hetero_bidirectional",
    #"H4_hetero_event_cascade",
    #"HELL_H1_pulse",
    #"HELL_H2_phase_jump",
    #"HELL_H3_bifurcation",
    #"HELL_H4_cascade",
    #"HELL_H5_resonance",
    #"HELL_H6_decay",
    #"HELL_H7_multi_hell",
    #"HELL_H8_progressive",
}

SCENARIOS = {k: v for k, v in ALL_SCENARIOS.items() if k in ENABLED}


# ── Adapters ──

class FISJFusionAdapter:
    """3-engine fusion: InverseCausalEngine (raw+DI) + NetworkAnalyzerCore (q-value)."""
    method_name = "FISJ-Fusion"

    def __init__(self, max_lag=8, solver="auto", w_raw=0.50, w_stat=0.25, w_struct=0.25):
        self.max_lag = max_lag
        self.solver = solver
        self.w_raw = w_raw
        self.w_stat = w_stat
        self.w_struct = w_struct

    def fit(self, df, cfg=None):
        names = list(df.columns); n = len(names)
        data = df.values.astype(np.float64)

        # Engine 1: InverseCausalEngine
        ice_cfg = InverseCausalEngineConfig(
            max_lag=self.max_lag, ar_lag=1, solver=self.solver,
            standardize=True, include_intercept=True, validation_fraction=0.25,
            use_backward_check=True, compute_direct_irreducibility=True,
        )
        ice = InverseCausalEngine(ice_cfg)
        ice_result = ice.fit(data, dimension_names=names)

        # Engine 2: NetworkAnalyzerCore
        nac = NetworkAnalyzerCore(max_lag=self.max_lag, adaptive=False, p_value_threshold=0.05)
        nac_result = nac.analyze(data, dimension_names=names)

        # Fusion
        n_samples = data.shape[0] - self.max_lag
        fusion = fuse_scores(
            raw_score_matrix=ice_result.score_matrix_unfiltered,
            direct_score_matrix=ice_result.direct_score_matrix,
            causal_matrix=nac_result.causal_matrix,
            lag_matrix=nac_result.causal_lag_matrix,
            n_samples=n_samples,
            max_lag=self.max_lag,
            w_raw=self.w_raw, w_stat=self.w_stat, w_struct=self.w_struct,
        )

        scores = fusion.fused_score_matrix.copy()
        adjacency = fusion.binary_matrix.astype(int)

        return MethodOutput(
            method_name=self.method_name, names=names,
            adjacency_scores=scores, adjacency_bin=adjacency,
            directed_support=True, lag_support=True, sign_support=True,
            lag_matrix=ice_result.lag_matrix, sign_matrix=ice_result.sign_matrix,
        )


class VARGrangerAdapter:
    method_name = "VAR_Granger"
    def __init__(self, maxlags=8, alpha=0.05): self.maxlags = maxlags; self.alpha = alpha
    def fit(self, df, cfg):
        names = list(df.columns); n = len(names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = VAR(df.copy()); results = model.fit(maxlags=min(self.maxlags, max(1, len(df) // 10)), ic="aic")
        scores = np.zeros((n, n)); adj = np.zeros((n, n), dtype=int)
        lagm = np.zeros((n, n), dtype=int); signm = np.zeros((n, n), dtype=int)
        params = results.params.copy()
        for i, src in enumerate(names):
            for j, dst in enumerate(names):
                if i == j: continue
                try: test = results.test_causality(caused=dst, causing=[src], kind="f"); pval = float(test.pvalue)
                except: pval = 1.0
                scores[i, j] = -math.log10(max(pval, 1e-12)); adj[i, j] = int(pval < self.alpha)
                cv, cl = [], []
                for lag in range(1, results.k_ar + 1):
                    rn = f"L{lag}.{src}"
                    if rn in params.index and dst in params.columns: cv.append(float(params.loc[rn, dst])); cl.append(lag)
                if cv: idx = int(np.argmax(np.abs(cv))); lagm[i, j] = cl[idx]; signm[i, j] = int(np.sign(cv[idx]))
        np.fill_diagonal(scores, 0); np.fill_diagonal(adj, 0)
        return MethodOutput(method_name=self.method_name, names=names, adjacency_scores=scores, adjacency_bin=adj, directed_support=True, lag_support=True, sign_support=True, lag_matrix=lagm, sign_matrix=signm)


class TransferEntropyAdapter:
    method_name = "TransferEntropy"
    def __init__(self, max_lag=8, n_perm=30): self.max_lag = max_lag; self.n_perm = n_perm
    def fit(self, df, cfg):
        names = list(df.columns); n = len(names); x = df.to_numpy(); rng = _rng(12345)
        scores = np.zeros((n, n)); adj = np.zeros((n, n), dtype=int); lagm = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                best_te, best_lag = -np.inf, 1
                for lag in range(1, min(self.max_lag, len(df) - 2) + 1):
                    te = _te_discrete(x[:, i], x[:, j], lag=lag)
                    if te > best_te: best_te = te; best_lag = lag
                null = [_te_discrete(rng.permutation(x[:, i]), x[:, j], lag=best_lag) for _ in range(self.n_perm)]
                pval = (1 + np.sum(np.array(null) >= best_te)) / (self.n_perm + 1)
                scores[i, j] = max(best_te, 0); lagm[i, j] = best_lag; adj[i, j] = int(pval < 0.05)
        np.fill_diagonal(scores, 0); np.fill_diagonal(adj, 0)
        return MethodOutput(method_name=self.method_name, names=names, adjacency_scores=scores, adjacency_bin=adj, directed_support=True, lag_support=True, sign_support=False, lag_matrix=lagm)


class GraphicalLassoAdapter:
    method_name = "GraphLasso"
    def __init__(self, edge_thr=0.03): self.edge_thr = edge_thr
    def fit(self, df, cfg):
        names = list(df.columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore"); m = GraphicalLassoCV(); m.fit(StandardScaler().fit_transform(df.to_numpy()))
        p = np.abs(m.precision_); np.fill_diagonal(p, 0)
        adj = ((p >= self.edge_thr).astype(int) + ((p >= self.edge_thr).astype(int)).T > 0).astype(int); np.fill_diagonal(adj, 0)
        return MethodOutput(method_name=self.method_name, names=names, adjacency_scores=p, adjacency_bin=adj, directed_support=False, lag_support=False, sign_support=False)


class EventXCorrAdapter:
    method_name = "EventXCorr"
    def __init__(self, max_lag=8, n_perm=30): self.max_lag = max_lag; self.n_perm = n_perm
    def fit(self, df, cfg):
        names = list(df.columns); n = len(names); x = df.to_numpy(); rng = _rng(54321)
        events = np.column_stack([_diff_events(x[:, i]) for i in range(n)])
        scores = np.zeros((n, n)); adj = np.zeros((n, n), dtype=int); lagm = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                bs, bl = -np.inf, 1
                for lag in range(1, min(self.max_lag, len(df) - 2) + 1):
                    s = _event_sync_score(events[:, i], events[:, j], lag)
                    if s > bs: bs = s; bl = lag
                null = [_event_sync_score(rng.permutation(events[:, i]), events[:, j], lag=bl) for _ in range(self.n_perm)]
                pval = (1 + np.sum(np.array(null) >= bs)) / (self.n_perm + 1)
                scores[i, j] = max(bs, 0); lagm[i, j] = bl; adj[i, j] = int(pval < 0.05)
        np.fill_diagonal(scores, 0); np.fill_diagonal(adj, 0)
        return MethodOutput(method_name=self.method_name, names=names, adjacency_scores=scores, adjacency_bin=adj, directed_support=True, lag_support=True, sign_support=False, lag_matrix=lagm)


class PCMCIPlusAdapter:
    method_name = "PCMCIPlus"
    def __init__(self, tau_max=8, pc_alpha=0.05, verbosity=0): self.tau_max = tau_max; self.pc_alpha = pc_alpha; self.verbosity = verbosity
    def fit(self, df, cfg):
        if not _HAS_TIGRAMITE: raise RuntimeError("tigramite not installed")
        names = list(df.columns); x = df.to_numpy(dtype=float); n = len(names)
        tg_df = pp.DataFrame(x, var_names=names)
        pcmci = PCMCI(dataframe=tg_df, cond_ind_test=ParCorr(significance="analytic"), verbosity=self.verbosity)
        results = pcmci.run_pcmciplus(tau_min=0, tau_max=self.tau_max, pc_alpha=self.pc_alpha)
        graph = results.get("graph"); val_matrix = results.get("val_matrix")
        adjacency = np.zeros((n, n), dtype=int); scores = np.zeros((n, n), dtype=float)
        lagm = np.zeros((n, n), dtype=int); signm = np.zeros((n, n), dtype=int)
        if graph is not None:
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    best_abs, best_tau, best_sign, found = 0.0, 0, 0, False
                    for tau in range(1, min(self.tau_max, graph.shape[2] - 1) + 1):
                        g = graph[i, j, tau]
                        if isinstance(g, bytes): g = g.decode()
                        if g is None: g = ""
                        if str(g).strip() != "":
                            found = True; val = float(val_matrix[i, j, tau]) if val_matrix is not None else 0.0
                            if abs(val) >= best_abs: best_abs = abs(val); best_tau = tau; best_sign = int(np.sign(val)) if val != 0 else 0
                    adjacency[i, j] = int(found); scores[i, j] = best_abs; lagm[i, j] = best_tau; signm[i, j] = best_sign
        np.fill_diagonal(adjacency, 0); np.fill_diagonal(scores, 0.0)
        return MethodOutput(method_name=self.method_name, names=names, adjacency_scores=scores, adjacency_bin=adjacency, directed_support=True, lag_support=True, sign_support=True, lag_matrix=lagm, sign_matrix=signm, meta={"pc_alpha": self.pc_alpha, "tau_max": self.tau_max})


# ── Evaluation ──

def evaluate(output, gt):
    metrics = {}; n = len(gt.names); mask = ~np.eye(n, dtype=bool)
    if output.adjacency_bin is not None:
        pred_u = output.undirected_bin(); true_u = gt.undirected_adjacency()
        metrics["edge_f1_undir"] = _precision_recall_f1(true_u[mask], pred_u[mask])["f1"]
        if output.directed_support:
            prf = _precision_recall_f1(gt.adjacency[mask], output.adjacency_bin[mask])
            metrics["edge_f1_dir"] = prf["f1"]; metrics["edge_prec_dir"] = prf["precision"]; metrics["edge_rec_dir"] = prf["recall"]
        else: metrics["edge_f1_dir"] = np.nan
        if output.lag_support and output.lag_matrix is not None and gt.lag_matrix is not None:
            errs = [abs(int(output.lag_matrix[s, d]) - int(gt.lag_matrix[s, d])) for s, d in np.argwhere(gt.adjacency == 1) if output.adjacency_bin[s, d] == 1]
            metrics["lag_mae"] = float(np.mean(errs)) if errs else np.nan
        else: metrics["lag_mae"] = np.nan
        if output.sign_support and output.sign_matrix is not None and gt.sign_matrix is not None:
            hits = [int(np.sign(output.sign_matrix[s, d]) == np.sign(gt.sign_matrix[s, d])) for s, d in np.argwhere(gt.adjacency == 1) if output.adjacency_bin[s, d] == 1]
            metrics["sign_acc"] = float(np.mean(hits)) if hits else np.nan
        else: metrics["sign_acc"] = np.nan
        if gt.forbidden_edges:
            fp = [int(output.adjacency_bin[s, d] == 1) for s, d in gt.forbidden_edges]
            metrics["spurious_rate"] = float(np.mean(fp))
        else: metrics["spurious_rate"] = np.nan
    return metrics


# ── CauseMe Metrics ──

@dataclass
class CauseMeMetrics:
    """CauseMeと同じ指標セット"""
    AUC: float = 0.0
    AUC_PR: float = 0.0
    F_measure: float = 0.0
    FPR: float = 0.0
    TPR: float = 0.0
    TLR: float = 0.0


def evaluate_causeme(
    scores: np.ndarray,
    lags: np.ndarray,
    true_adj: np.ndarray,
    true_lag: np.ndarray,
) -> CauseMeMetrics:
    """CauseMeと同じ方法で評価。"""
    n_dims = scores.shape[0]
    mask = ~np.eye(n_dims, dtype=bool)
    y_true = true_adj[mask].flatten()
    y_scores = scores[mask].flatten()

    if np.sum(y_true) == 0 or np.sum(y_true) == len(y_true):
        return CauseMeMetrics()
    if np.max(y_scores) == np.min(y_scores):
        return CauseMeMetrics()

    try: auc = roc_auc_score(y_true, y_scores)
    except ValueError: auc = 0.5

    try: auc_pr = average_precision_score(y_true, y_scores)
    except ValueError: auc_pr = 0.0

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_scores)
    f1_arr = 2 * precision_arr * recall_arr / (precision_arr + recall_arr + 1e-10)
    best_idx = np.argmax(f1_arr)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    y_pred = (y_scores >= best_threshold).astype(int)
    f_measure = float(f1_score(y_true, y_pred, zero_division=0))

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    pred_adj = np.zeros_like(true_adj)
    pred_adj[mask] = y_pred
    n_correct_lag = 0
    n_detected_true = 0
    for i in range(n_dims):
        for j in range(n_dims):
            if i == j: continue
            if pred_adj[i, j] == 1 and true_adj[i, j] == 1:
                n_detected_true += 1
                if lags[i, j] == true_lag[i, j]:
                    n_correct_lag += 1
    tlr = n_correct_lag / n_detected_true if n_detected_true > 0 else 0.0

    return CauseMeMetrics(AUC=float(auc), AUC_PR=float(auc_pr), F_measure=float(f_measure), FPR=float(fpr), TPR=float(tpr), TLR=float(tlr))


# ── Main ──

def run_benchmark(n_repeats=5):
    import logging; logging.disable(logging.CRITICAL)
    print("=" * 65)
    print("  FISJ vs Baselines")
    print(f"  {n_repeats} repeats × {len(SCENARIOS)} scenarios")
    print("=" * 65)

    cfg = ScenarioConfig()
    methods = [
        FISJInverseAdapter(max_lag=8, solver="auto", apply_textbook_filter=True),
        FISJFusionAdapter(max_lag=8, solver="auto"),
        VARGrangerAdapter(),
        TransferEntropyAdapter(n_perm=30),
        GraphicalLassoAdapter(),
        EventXCorrAdapter(n_perm=30),
    ]
    if _HAS_TIGRAMITE: methods.append(PCMCIPlusAdapter(tau_max=8))
    else: print("  ⚠️ tigramite not installed — skipping PCMCI+")

    all_rows = []
    for sname, gen_fn in SCENARIOS.items():
        print(f"\n  {sname}...", end=" ", flush=True)
        for rep in range(n_repeats):
            df, gt = gen_fn(cfg, cfg.seed + rep * 7)
            for adapter in methods:
                t0 = time.time()
                try:
                    out = adapter.fit(df, cfg)
                    elapsed = time.time() - t0
                    m = evaluate(out, gt); m["scenario"] = sname; m["method"] = adapter.method_name; m["repeat"] = rep; m["time_s"] = elapsed
                    # CauseMe metrics
                    if out.adjacency_scores is not None:
                        lag_mat = out.lag_matrix if out.lag_matrix is not None else np.zeros_like(out.adjacency_scores, dtype=int)
                        true_lag = gt.lag_matrix if gt.lag_matrix is not None else np.zeros_like(gt.adjacency, dtype=int)
                        cm = evaluate_causeme(out.adjacency_scores, lag_mat, gt.adjacency, true_lag)
                        m["AUC"] = cm.AUC; m["AUC_PR"] = cm.AUC_PR; m["F_measure"] = cm.F_measure
                        m["FPR"] = cm.FPR; m["TPR"] = cm.TPR; m["TLR"] = cm.TLR
                    all_rows.append(m)
                except Exception as e: print(f"⚠️{adapter.method_name}:{e}", end=" ")
        print("✅")

    results = pd.DataFrame(all_rows)
    mcols = [c for c in results.columns if c not in ("scenario", "method", "repeat", "time_s")]

    print("\n" + "=" * 65)
    print("  RESULTS: Per-Method Average")
    print("=" * 65)
    summary = results.groupby("method")[mcols + ["time_s"]].mean()
    print(summary.to_string(float_format="%.3f"))

    print("\n" + "=" * 65)
    print("  COMPOSITE SCORE")
    print("=" * 65)
    composite = {}
    for method in summary.index:
        r = summary.loc[method]
        f1d = r.get("edge_f1_dir", 0) if not np.isnan(r.get("edge_f1_dir", np.nan)) else 0
        f1u = r.get("edge_f1_undir", 0) if not np.isnan(r.get("edge_f1_undir", np.nan)) else 0
        lag = r.get("lag_mae", np.nan); lag_s = 1 / (1 + lag) if not np.isnan(lag) else 0
        sign = r.get("sign_acc", np.nan); sign_s = sign if not np.isnan(sign) else 0
        spur = r.get("spurious_rate", np.nan); spur_s = (1 - spur) if not np.isnan(spur) else 0.5
        c = (f1d * 1.5 + f1u * 1.0 + lag_s * 1.0 + sign_s * 0.8 + spur_s * 0.8) / 5.1
        composite[method] = c
        print(f"  {method:20s}: {c:.3f}  (F1d={f1d:.3f} F1u={f1u:.3f} Lag={lag_s:.3f} Sign={sign_s:.3f} Spur={spur_s:.3f} | {r['time_s']:.2f}s)")

    print("\n  🏆 Ranking:")
    for rank, (method, score) in enumerate(sorted(composite.items(), key=lambda x: x[1], reverse=True), 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
        print(f"    {medal} {rank}. {method}: {score:.3f}")

    # CauseMe metrics
    cm_cols = ["AUC", "AUC_PR", "F_measure", "FPR", "TPR", "TLR"]
    if all(c in results.columns for c in cm_cols):
        print("\n" + "=" * 65)
        print("  CAUSEME METRICS (Per-Method Average)")
        print("=" * 65)
        cm_summary = results.groupby("method")[cm_cols].mean()
        print(cm_summary.to_string(float_format="%.3f"))

        print("\n  🏆 CauseMe Ranking (by AUC):")
        for rank, (method, auc) in enumerate(sorted(cm_summary["AUC"].items(), key=lambda x: x[1], reverse=True), 1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "  ")
            pr = cm_summary.loc[method, "AUC_PR"]
            f1 = cm_summary.loc[method, "F_measure"]
            print(f"    {medal} {rank}. {method}: AUC={auc:.3f} PR={pr:.3f} F1={f1:.3f}")

    return results


if __name__ == "__main__":
    run_benchmark(n_repeats=5)
