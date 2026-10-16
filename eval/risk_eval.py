"""E5: does the collision-risk alert fire when it should, and early enough?

Ground truth comes from the scenario: a (vehicle, pedestrian) pair is truly
DANGEROUS at time t if their actual positions come within the safety buffer
(2.2 m) at any moment in the next 3 s.

The pipeline is replayed with the cached predictions injected on the live
schedule, so the risk scores are exactly what the dashboard would have shown.
Then, offline:

* AUROC of the raw score against the danger label (all pairs, every 0.2 s);
* event level: a danger EVENT is a continuous stretch where a pair is
  dangerous. It counts as caught if the alert is on at any point between 3 s
  before the stretch starts and the moment of closest approach. Warning time
  = closest-approach time minus the alert's first firing.
* false alarms: alert onsets with no danger label within the following 3 s,
  per minute of scenario.

The hysteresis is applied offline over a grid of thresholds, so one replay
covers the whole sweep.

    python -m eval.risk_eval
"""
from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

from trajectory_risk.pipeline import LivePipeline
from trajectory_risk.risk import DEFAULT_SAFETY_BUFFER_M, HysteresisAlert

from .accuracy import write_csv
from .common import RESULTS_DIR, load_cache, load_ticks

HORIZON_S = 3.0
SCENARIOS = ("t11_crossing", "t11_busy", "onenorth")


class Cached:
    """A predictor that returns what the real model returned for this exact request."""

    def __init__(self, results, agent_type, step_dt_s, ob_horizon, name):
        self.by_key = {(r.agent_id, r.sample_index): r.candidates for r in results if r.agent_type == agent_type}
        self.step_dt_s, self.ob_horizon, self.name = step_dt_s, ob_horizon, name

    @property
    def buffer_len(self):
        return self.ob_horizon + 2

    def predict(self, req):
        return self.by_key.get((req.agent_id, req.sample_index), [])


def positions_by_time(ticks):
    """{agent id: (times array, xy array)} at full 60 Hz."""
    acc = defaultdict(lambda: ([], []))
    kinds = {}
    for t in ticks:
        for a in t.agents:
            acc[a.id][0].append(t.t_s)
            acc[a.id][1].append((a.x_m, a.y_m))
            kinds[a.id] = a.type
    return {i: (np.array(ts), np.array(xy)) for i, (ts, xy) in acc.items()}, kinds


def true_min_distance(track_v, track_p, t0, horizon=HORIZON_S):
    tv, xv = track_v
    tp, xp = track_p
    lo, hi = t0, t0 + horizon
    common = np.intersect1d(np.round(tv[(tv >= lo) & (tv <= hi)], 4), np.round(tp[(tp >= lo) & (tp <= hi)], 4))
    if len(common) == 0:
        return float("inf"), float("nan")
    iv = np.searchsorted(np.round(tv, 4), common)
    ip = np.searchsorted(np.round(tp, 4), common)
    d = np.linalg.norm(xv[iv] - xp[ip], axis=1)
    j = int(np.argmin(d))
    return float(d[j]), float(common[j])


def replay_scores(scenario: str, models: str):
    """-> list of (t, veh_id, ped_id, risk) at every 0.2 s risk update."""
    cache = load_cache(f"pred_{scenario}_{models}")
    res = cache["results"]
    pipe = LivePipeline(Cached(res, "vehicle", 0.5, 5, "cached_vehicle"),
                        Cached(res, "pedestrian", 0.4, 8, "cached_pedestrian"))
    ticks = load_ticks(scenario)
    out = []
    for t in ticks:
        pipe.step_sync(t)
        if abs(pipe._last_risk_s - t.t_s) < 1e-9:
            out += [(t.t_s, r.veh_id, r.ped_id, r.risk) for r in pipe.risk_rows]
    return out, ticks


def label(scores, ticks):
    tracks, _ = positions_by_time(ticks)
    labelled = []
    for t, v, p, risk in scores:
        d, t_cpa = true_min_distance(tracks[v], tracks[p], t)
        labelled.append((t, v, p, risk, d < DEFAULT_SAFETY_BUFFER_M, t_cpa))
    return labelled


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def event_metrics(labelled, duration_min: float, enter_threshold: float, enter_ticks: int,
                  clear_threshold: float = None, clear_ticks: int = 5) -> Dict:
    clear_threshold = enter_threshold / 3 if clear_threshold is None else clear_threshold
    series = defaultdict(list)
    for row in labelled:
        series[(row[1], row[2])].append(row)
    caught = total = false_alarms = 0
    warnings: List[float] = []
    for key, rows in series.items():
        rows.sort()
        alert = HysteresisAlert(enter_threshold, enter_ticks, min(clear_threshold, enter_threshold), clear_ticks)
        active = [alert.update(r[3]) for r in rows]
        times = np.array([r[0] for r in rows])
        danger = np.array([r[4] for r in rows])
        # danger events: maximal runs of consecutive dangerous updates
        i = 0
        while i < len(rows):
            if not danger[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and danger[j + 1] and times[j + 1] - times[j] < 0.25:
                j += 1
            total += 1
            t_cpa = np.nanmin([r[5] for r in rows[i:j + 1]])
            window = (times >= times[i] - HORIZON_S) & (times <= t_cpa)
            fired = np.where(window & np.array(active))[0]
            if len(fired):
                caught += 1
                warnings.append(float(t_cpa - times[fired[0]]))
            i = j + 1
        # false alarms: rising edges with no danger in the next 3 s
        for k in range(len(active)):
            if active[k] and (k == 0 or not active[k - 1]):
                ahead = (times >= times[k]) & (times <= times[k] + HORIZON_S)
                if not danger[ahead].any():
                    false_alarms += 1
    return dict(events=total, caught=caught, recall=caught / total if total else float("nan"),
                false_alarms=false_alarms, false_alarms_per_min=false_alarms / duration_min,
                warning_s_median=float(np.median(warnings)) if warnings else float("nan"))


def main() -> None:
    summary, sweep = [], []
    for models in ("learned", "cv"):
        all_lab, minutes = [], 0.0
        for sc in SCENARIOS:
            scores, ticks = replay_scores(sc, models)
            lab = label(scores, ticks)
            all_lab += [(f"{sc}:{t}", v, p, r, d, c) for (t, v, p, r, d, c) in lab]
            minutes += (ticks[-1].t_s - ticks[0].t_s) / 60
            a = np.array([r[3] for r in lab])
            y = np.array([r[4] for r in lab])
            m = event_metrics(lab, (ticks[-1].t_s - ticks[0].t_s) / 60, 0.15, 3)
            summary.append(dict(scenario=sc, models=models, pair_updates=len(lab), dangerous_share=y.mean(),
                                auroc=auroc(a, y), **m))
            print(summary[-1], flush=True)
        # threshold sweep pooled over scenarios (keys made unique per scenario)
        pooled = []
        for (t, v, p, r, d, c) in all_lab:
            sc, tt = t.split(":")
            pooled.append((float(tt), f"{sc}-{v}", p, r, d, c))
        for thr in (0.02, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6):
            for ticks_ in (1, 3):
                m = event_metrics(pooled, minutes, thr, ticks_)
                sweep.append(dict(models=models, enter_threshold=thr, enter_ticks=ticks_, **m))
        y = np.array([r[4] for r in pooled])
        a = np.array([r[3] for r in pooled])
        summary.append(dict(scenario="ALL", models=models, pair_updates=len(pooled), dangerous_share=y.mean(),
                            auroc=auroc(a, y), **event_metrics(pooled, minutes, 0.15, 3)))
        print(summary[-1], flush=True)
    write_csv(os.path.join(RESULTS_DIR, "risk_summary.csv"), summary)
    write_csv(os.path.join(RESULTS_DIR, "risk_threshold_sweep.csv"), sweep)


if __name__ == "__main__":
    main()
