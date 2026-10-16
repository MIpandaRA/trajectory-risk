"""Is a score a probability? Reliability, ECE, Brier, ranking metrics,
recalibration, and a cluster bootstrap that respects correlated samples.

A risk score is only a probability of the event it was built to estimate if,
among all the times it says p, the event happens a fraction p of the time.
That is CALIBRATION, and it is separate from DISCRIMINATION (ranking
dangerous above safe: AUROC, PR-AUC). A score can rank perfectly and still
be badly miscalibrated; recalibration (Platt, isotonic) fixes calibration
without changing the ranking.

Samples here are not independent: consecutive 5 Hz updates of one pair are
near copies, and recordings differ (one has no events at all). Confidence
intervals therefore resample whole CLUSTERS (recordings). Every metric is
computed from per-cluster sufficient statistics, so one bootstrap replicate
is a weight vector over clusters, not a re-sort of millions of rows.

Binning:
* equal-width bins: [0, 0.1), [0.1, 0.2), ... (standard ECE);
* equal-count bins (adaptive ECE): quantile edges, but a tied score value is
  never split across bins. With thousands of exact zeros, all zeros share
  bin 0 and the remaining bins split the rest, so bins can be unequal.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

EPS = 1e-4   # logit clipping for Platt scaling


# -- binning -------------------------------------------------------------
def equal_width_edges(n_bins: int = 10) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def equal_count_edges(scores: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Edges for (up to) n_bins bins of about equal count, never splitting a
    tied value. Greedy: each bin takes 1/(bins left) of the samples left,
    extended to include every copy of its last value. A big tie (thousands of
    exact zeros) becomes one bin and the remaining bins split the rest evenly,
    instead of quantile edges collapsing onto the tie. Used with
    ``assign_bins`` (right-closed inner edges)."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    N = len(s)
    inner, start, left = [], 0, n_bins
    while left > 1 and start < N:
        idx = start + int(np.ceil((N - start) / left)) - 1
        v = s[idx]
        end = int(np.searchsorted(s, v, side="right"))
        if end >= N:
            break
        inner.append(v)
        start, left = end, left - 1
    return np.concatenate([[min(0.0, s[0]) if N else 0.0], inner, [max(1.0, s[-1]) if N else 1.0]])


def assign_bins(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Bin b holds edges[b-1] < s <= edges[b] for inner edges; bin 0 holds
    everything <= edges[1]. A tie is one value, so it is one bin."""
    return np.searchsorted(edges[1:-1], np.asarray(scores, dtype=np.float64), side="left")


# -- sufficient statistics per cluster -----------------------------------
@dataclass
class Prepared:
    """Per-(bin, cluster) and per-(score level, cluster) sums for one score."""
    n_clusters: int
    edges: np.ndarray
    # binned: shape (n_bins, n_clusters)
    n: np.ndarray
    s: np.ndarray
    s2: np.ndarray
    y: np.ndarray
    sy: np.ndarray
    # ranking: unique score levels ascending, shape (n_levels, n_clusters)
    levels: np.ndarray
    pos: np.ndarray
    neg: np.ndarray


def prepare(scores, labels, clusters, edges: np.ndarray) -> Prepared:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    c = np.asarray(clusters, dtype=np.int64)
    C = int(c.max()) + 1 if len(c) else 0
    b = assign_bins(s, edges)
    B = len(edges) - 1
    idx = b * C + c
    sums = lambda v: np.bincount(idx, weights=v, minlength=B * C).reshape(B, C)
    levels, inv = np.unique(s, return_inverse=True)
    L = len(levels)
    lidx = inv * C + c
    pos = np.bincount(lidx, weights=y, minlength=L * C).reshape(L, C)
    neg = np.bincount(lidx, weights=1 - y, minlength=L * C).reshape(L, C)
    return Prepared(C, edges, sums(np.ones_like(s)), sums(s), sums(s * s), sums(y), sums(s * y), levels, pos, neg)


def _w(p: Prepared, w: Optional[np.ndarray]) -> np.ndarray:
    return np.ones(p.n_clusters) if w is None else np.asarray(w, dtype=np.float64)


# -- metrics (all take optional cluster weights w) ------------------------
def reliability(p: Prepared, w=None) -> Dict[str, np.ndarray]:
    """Per bin: count, mean score, observed frequency (empty bins NaN)."""
    w = _w(p, w)
    n = p.n @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"count": n, "mean_score": (p.s @ w) / n, "observed": (p.y @ w) / n}


def ece(p: Prepared, w=None) -> float:
    """sum_b n_b / N * |observed_b - mean_score_b| over the prepared bins."""
    r = reliability(p, w)
    ok = r["count"] > 0
    return float(np.sum(r["count"][ok] * np.abs(r["observed"][ok] - r["mean_score"][ok])) / r["count"].sum())


def brier(p: Prepared, w=None) -> Dict[str, float]:
    """Brier score and its decomposition over the prepared bins
    (Stephenson et al. 2008):

        BS = REL - RES + UNC + WBV - WBC

    REL  reliability   sum n_b (sbar_b - obar_b)^2 / N   (0 = calibrated)
    RES  resolution    sum n_b (obar_b - obar)^2 / N     (higher = more informative)
    UNC  uncertainty   obar (1 - obar)                   (base rate only)
    WBV  within-bin variance of the scores
    WBC  2 x within-bin covariance of score and outcome
    With scores constant inside each bin, WBV = WBC = 0 and this is Murphy's
    classic three-term decomposition. The identity is exact either way.
    """
    w = _w(p, w)
    n, s, s2, y, sy = (a @ w for a in (p.n, p.s, p.s2, p.y, p.sy))
    N = n.sum()
    ok = n > 0
    sb, ob = s[ok] / n[ok], y[ok] / n[ok]
    obar = y.sum() / N
    bs = (s2.sum() - 2 * sy.sum() + y.sum()) / N            # y^2 = y for binary labels
    rel = np.sum(n[ok] * (sb - ob) ** 2) / N
    res = np.sum(n[ok] * (ob - obar) ** 2) / N
    unc = obar * (1 - obar)
    wbv = np.sum(s2[ok] - n[ok] * sb ** 2) / N
    wbc = 2 * np.sum(sy[ok] - n[ok] * sb * ob) / N
    return {"brier": float(bs), "reliability": float(rel), "resolution": float(res), "uncertainty": float(unc),
            "within_bin_variance": float(wbv), "within_bin_covariance": float(wbc), "base_rate": float(obar)}


def auroc(p: Prepared, w=None) -> float:
    """P(score of a positive > score of a negative), ties count half."""
    w = _w(p, w)
    pos, neg = p.pos @ w, p.neg @ w
    P, N = pos.sum(), neg.sum()
    if P == 0 or N == 0:
        return float("nan")
    neg_below = np.cumsum(neg) - neg
    return float(np.sum(pos * (neg_below + 0.5 * neg)) / (P * N))


def pr_auc(p: Prepared, w=None) -> float:
    """Average precision (sklearn's definition): sum over thresholds, from
    the highest score down, of precision x the recall gained there. Tied
    scores form one threshold."""
    w = _w(p, w)
    pos, neg = (p.pos @ w)[::-1], (p.neg @ w)[::-1]
    P = pos.sum()
    if P == 0:
        return float("nan")
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    return float(np.sum(pos / P * tp / np.maximum(tp + fp, 1e-300)))


# -- recalibration -------------------------------------------------------
def logit(s, eps: float = EPS):
    s = np.clip(np.asarray(s, dtype=np.float64), eps, 1 - eps)
    return np.log(s / (1 - s))


def fit_platt(scores, labels, eps: float = EPS, iters: int = 100) -> Tuple[float, float]:
    """Logistic regression y ~ sigmoid(a * logit(clip(s)) + b) by Newton's
    method, unregularised. -> (a, b). a = 1, b = 0 is the identity."""
    x = logit(scores, eps)
    y = np.asarray(labels, dtype=np.float64)
    X = np.stack([x, np.ones_like(x)], axis=1)
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (y - p)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-9 * np.eye(2)
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(beta[0]), float(beta[1])


def apply_platt(scores, params: Tuple[float, float], eps: float = EPS) -> np.ndarray:
    a, b = params
    return 1 / (1 + np.exp(-(a * logit(scores, eps) + b)))


def fit_isotonic(scores, labels):
    """Non-decreasing step function from score to observed frequency
    (sklearn IsotonicRegression, clipped to [0, 1] and to the fitted range)."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    iso.fit(np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64))
    return iso


# -- cluster bootstrap ---------------------------------------------------
def cluster_bootstrap(stat: Callable[[np.ndarray], float], n_clusters: int, n_boot: int = 1000, seed: int = 0,
                      q: Sequence[float] = (2.5, 97.5)) -> Dict[str, float]:
    """Resample clusters with replacement: replicate weights are multinomial
    counts over the clusters. ``stat(weights)`` -> number. Returns the point
    estimate (all weights 1), percentile CI and how many clusters it rests on.
    NaN replicates (e.g. no positives drawn) are dropped and counted."""
    rng = np.random.default_rng(seed)
    point = stat(np.ones(n_clusters))
    reps = np.array([stat(rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters)).astype(float))
                     for _ in range(n_boot)])
    ok = reps[~np.isnan(reps)]
    lo, hi = np.percentile(ok, q) if len(ok) else (float("nan"), float("nan"))
    return {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi), "n_clusters": n_clusters,
            "n_boot": n_boot, "nan_replicates": int(len(reps) - len(ok))}


def cluster_bootstrap_many(stats: Callable[[np.ndarray], Dict[str, float]], n_clusters: int, n_boot: int = 1000,
                           seed: int = 0, q: Sequence[float] = (2.5, 97.5)) -> Dict[str, Dict[str, float]]:
    """Like ``cluster_bootstrap`` for several statistics at once: every
    statistic sees the SAME replicate weights, so their CIs are consistent
    and the data are prepared once. ``stats(weights)`` -> {name: number}."""
    rng = np.random.default_rng(seed)
    point = stats(np.ones(n_clusters))
    draws = [stats(rng.multinomial(n_clusters, np.full(n_clusters, 1 / n_clusters)).astype(float))
             for _ in range(n_boot)]
    out = {}
    for k, v in point.items():
        reps = np.array([d[k] for d in draws], dtype=float)
        ok = reps[~np.isnan(reps)]
        lo, hi = np.percentile(ok, q) if len(ok) else (float("nan"), float("nan"))
        out[k] = {"point": float(v), "ci_lo": float(lo), "ci_hi": float(hi), "n_clusters": n_clusters,
                  "nan_replicates": int(len(reps) - len(ok))}
    return out


__all__ = ["cluster_bootstrap_many", "equal_width_edges", "equal_count_edges", "assign_bins", "Prepared", "prepare", "reliability", "ece",
           "brier", "auroc", "pr_auc", "logit", "fit_platt", "apply_platt", "fit_isotonic", "cluster_bootstrap"]
