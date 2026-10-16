"""E1: prediction accuracy, learned models vs constant velocity.

For every cached prediction, the actual future comes from the scenario itself
(the simulator is ground truth). Reported per scenario, agent type and model:

* top-1 ADE / FDE: the highest-confidence candidate only. This is what the
  dashboard draws and what a downstream system would act on.
* minADE_k / minFDE_k: the best of all k candidates. An oracle metric: it
  says one candidate was close, not that the system knew which one.
* off-road rate (vehicles): share of top-1 predicted points that fall off the
  drivable area, counted only where the true future stays on it.

Vehicle predictions are also split into "straight" and "turning" (true future
heading changes by more than 20 degrees), since that is where map context
should matter.

    python -m eval.accuracy
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import numpy as np

from trajectory_risk.maps import MapBundle

from .common import RESULTS_DIR, SCENARIOS, ade_fde, bootstrap_ci, future_of, ground_truth, load_cache, load_ticks

TURN_DEG = 20.0


def heading_change_deg(path: np.ndarray, start: np.ndarray) -> float:
    pts = np.vstack([start, path])
    d0, d1 = pts[1] - pts[0], pts[-1] - pts[-2]
    if np.linalg.norm(d0) < 1e-6 or np.linalg.norm(d1) < 1e-6:
        return 0.0
    a = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    return abs(math.degrees((a + math.pi) % (2 * math.pi) - math.pi))


def per_prediction_rows(scenario: str, models: str):
    cache = load_cache(f"pred_{scenario}_{models}")
    ticks = load_ticks(scenario)
    bundle = MapBundle.load(SCENARIOS[scenario])
    gts = {"vehicle": ground_truth(ticks, 0.5, "vehicle"), "pedestrian": ground_truth(ticks, 0.4, "pedestrian")}
    rows = []
    for r in cache["results"]:
        if not r.candidates:
            continue
        H = len(r.candidates[0].trajectory)
        gt = future_of(gts[r.agent_type], r.agent_id, r.sample_index, H)
        if gt is None:
            continue  # agent left the scene before the horizon ended
        now = np.array(gts[r.agent_type][r.agent_id][r.sample_index])
        top_ade, top_fde = ade_fde(r.candidates[0].trajectory, gt)
        all_ade_fde = [ade_fde(c.trajectory, gt) for c in r.candidates]
        row = dict(scenario=scenario, models=models, agent_type=r.agent_type, model=r.model,
                   agent_id=r.agent_id, sample_index=r.sample_index, k=len(r.candidates),
                   top1_ade=top_ade, top1_fde=top_fde,
                   min_ade=min(a for a, _ in all_ade_fde), min_fde=min(f for _, f in all_ade_fde),
                   top1_conf=r.candidates[0].confidence, runtime_ms=r.runtime_ms,
                   turn_deg=heading_change_deg(gt, now), offroad=float("nan"))
        if r.agent_type == "vehicle":
            gt_on = bundle.is_drivable(gt)
            if gt_on.all():
                row["offroad"] = float(1.0 - bundle.is_drivable(r.candidates[0].trajectory).mean())
        rows.append(row)
    return rows


def summarise(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["agent_type"], r["models"])].append(r)
        if r["agent_type"] == "vehicle":
            motion = "turning" if r["turn_deg"] > TURN_DEG else "straight"
            groups[(r["scenario"] + f"|{motion}", r["agent_type"], r["models"])].append(r)
        # Vehicle traffic is deterministic (no randomness in routing), so t11_busy
        # has exactly the same vehicles as t11_crossing; count them once in ALL.
        if not (r["agent_type"] == "vehicle" and r["scenario"] == "t11_busy"):
            groups[("ALL", r["agent_type"], r["models"])].append(r)
    out = []
    for (sc, at, models), rs in sorted(groups.items()):
        top = np.array([x["top1_ade"] for x in rs])
        lo, hi = bootstrap_ci(top)
        off = [x["offroad"] for x in rs if not math.isnan(x["offroad"])]
        out.append(dict(scenario=sc, agent_type=at, models=models, n=len(rs), k=rs[0]["k"],
                        top1_ade=top.mean(), top1_ade_ci_lo=lo, top1_ade_ci_hi=hi,
                        top1_fde=np.mean([x["top1_fde"] for x in rs]),
                        min_ade=np.mean([x["min_ade"] for x in rs]), min_fde=np.mean([x["min_fde"] for x in rs]),
                        offroad=np.mean(off) if off else float("nan"),
                        runtime_ms_median=float(np.median([x["runtime_ms"] for x in rs])),
                        runtime_ms_p95=float(np.percentile([x["runtime_ms"] for x in rs], 95))))
    return out


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    rows = []
    for sc in SCENARIOS:
        for models in ("cv", "learned"):
            try:
                rows += per_prediction_rows(sc, models)
            except FileNotFoundError:
                print(f"[accuracy] no cache for {sc}/{models}, skipping")
    write_csv(os.path.join(RESULTS_DIR, "accuracy_per_prediction.csv"), rows)
    summary = summarise(rows)
    write_csv(os.path.join(RESULTS_DIR, "accuracy_summary.csv"), summary)
    for s in summary:
        if "|" in s["scenario"]:
            continue
        print(f"{s['scenario']:14s} {s['agent_type']:10s} {s['models']:8s} n={s['n']:5d}  "
              f"top1 ADE {s['top1_ade']:.2f} [{s['top1_ade_ci_lo']:.2f},{s['top1_ade_ci_hi']:.2f}] "
              f"FDE {s['top1_fde']:.2f}  minADE {s['min_ade']:.2f} minFDE {s['min_fde']:.2f}  "
              f"off-road {s['offroad']:.3f}")


if __name__ == "__main__":
    main()
