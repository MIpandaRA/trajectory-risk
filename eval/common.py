"""Shared helpers for the offline experiments."""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Tuple

import numpy as np

from trajectory_risk.protocol import Tick, load_recording
from trajectory_risk.tracking import Decimator

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCENARIO_DIR = os.path.join(ROOT, "data", "scenarios")
CACHE_DIR = os.path.join(ROOT, "eval", "cache")
RESULTS_DIR = os.path.join(ROOT, "eval", "results")

SCENARIOS = {  # name -> map
    "t11_quiet": "T11",
    "t11_crossing": "T11",
    "t11_busy": "T11",
    "onenorth": "OneNorthRoundabout",
}


def load_ticks(name: str) -> List[Tick]:
    _, ticks = load_recording(os.path.join(SCENARIO_DIR, f"{name}.jsonl.gz"))
    return ticks


def ground_truth(ticks: List[Tick], period_s: float, agent_type: str) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """{agent id: {grid index: (x, y)}} on the same decimation grid the live
    pipeline uses, built from every tick in the scenario."""
    dec = Decimator(period_s)
    out: Dict[int, Dict[int, Tuple[float, float]]] = {}
    for t in ticks:
        k = dec.index_for(t.t_s)
        if k is None:
            continue
        for a in t.agents:
            if a.type == agent_type:
                out.setdefault(a.id, {})[k] = (a.x_m, a.y_m)
    return out


def future_of(gt, agent_id: int, index: int, horizon: int):
    """(horizon, 2) actual future after ``index``, or None if the agent left."""
    track = gt.get(agent_id, {})
    pts = [track.get(index + i) for i in range(1, horizon + 1)]
    if any(p is None for p in pts):
        return None
    return np.array(pts, dtype=np.float64)


def save_cache(name: str, obj) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_cache(name: str):
    with open(os.path.join(CACHE_DIR, f"{name}.pkl"), "rb") as f:
        return pickle.load(f)


def ade_fde(pred: np.ndarray, gt: np.ndarray) -> Tuple[float, float]:
    """Average and final Euclidean displacement error between (H, 2) paths."""
    d = np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)
    return float(d.mean()), float(d[-1])


def bootstrap_ci(values, n: int = 2000, seed: int = 0, q=(2.5, 97.5)):
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, q)
    return float(lo), float(hi)
