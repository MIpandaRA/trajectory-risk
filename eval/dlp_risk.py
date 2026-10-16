"""DLP risk scores per pair-update, joined to the true event labels.

Replays each recording through LivePipeline with the CACHED predictions
injected on the live schedule (exactly what the dashboard would have held),
and at every 5 Hz risk update scores every vehicle-pedestrian pair where both
have a held prediction. All scores come from one candidate-CPA computation
(risk.candidate_cpas), for every pre-declared combination:

    buffer d in {2.0, 2.5 (primary), 3.0} m
    x  (a) "close"              = original pairwise_risk
       (b) "close and moving"   = motion-aware variant, vehicle > 0.5 m/s (primary)
       (b1)                     = motion-aware variant, vehicle > 1.0 m/s (secondary)

Rows whose true future is not fully observed (eval/dlp_labels censoring)
have no label and are dropped.

    python -m eval.dlp_risk --split calibration --models learned cv
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .common import CACHE_DIR
from .dlp_labels import RISK_EVERY, cached_labels, overlap_artifact, splits
from .dlp_predict import cache_path, iter_ticks

BUFFERS_M = (2.0, 2.5, 3.0)
SPEEDS_MPS = (0.5, 1.0)
SCORE_NAMES = [f"{kind}_d{d:g}" for d in BUFFERS_M for kind in ("a", "b", "b1")]


class Cached:
    """A predictor that returns what the real model returned for this exact request."""

    def __init__(self, results, agent_type, step_dt_s, ob_horizon, name):
        from trajectory_risk.predictors.base import Candidate
        self.by_key = {(r["id"], r["k_idx"]): [Candidate(t.astype(np.float64), float(c))
                                              for t, c in zip(r["traj"], r["conf"])]
                       for r in results if r["type"] == agent_type}
        self.step_dt_s, self.ob_horizon, self.name = step_dt_s, ob_horizon, name

    @property
    def buffer_len(self):
        return self.ob_horizon + 2

    def predict(self, req):
        return self.by_key.get((req.agent_id, req.sample_index), [])


def risk_path(name: str, models: str) -> str:
    return os.path.join(CACHE_DIR, f"dlp_risk_{name}_{models}.npz")


def pair_key(t_s, veh, ped):
    """(t, veh, ped) -> one int64 for joining (5 Hz index, ids < 10^4)."""
    return (np.round(np.asarray(t_s) * 5).astype(np.int64) * 10 ** 8 + np.asarray(veh, np.int64) * 10 ** 4
            + np.asarray(ped, np.int64))


def score_recording(name: str, models: str) -> str:
    import pickle

    from trajectory_risk.pipeline import LivePipeline
    from trajectory_risk.risk import candidate_cpas

    with open(cache_path(name, models), "rb") as f:
        res = pickle.load(f)["results"]
    veh = Cached(res, "vehicle", 0.5, 5, f"{models}_vehicle")
    ped = Cached(res, "pedestrian", 0.4, 8, f"{models}_pedestrian")
    pipe = LivePipeline(veh, ped)
    cols = {k: [] for k in ["t_s", "veh", "ped", *SCORE_NAMES]}
    for tick in iter_ticks(name):
        for q in pipe.ingest(tick):
            pipe.accept(pipe.predict_now(q))
        if tick.seq % RISK_EVERY:
            continue
        hv, hp = pipe.held("vehicle"), pipe.held("pedestrian")
        for vid, v in hv.items():
            for pid, p in hp.items():
                vc, pc, d_all, moving = candidate_cpas(v, p, tick.t_s, SPEEDS_MPS)
                w = vc[:, None] * pc[None, :]
                cols["t_s"].append(tick.t_s)
                cols["veh"].append(vid)
                cols["ped"].append(pid)
                for d in BUFFERS_M:
                    cols[f"a_d{d:g}"].append(min(1.0, float(np.sum(w * (d_all < d)))))
                    cols[f"b_d{d:g}"].append(min(1.0, float(np.sum(w * (moving[0.5] < d)))))
                    cols[f"b1_d{d:g}"].append(min(1.0, float(np.sum(w * (moving[1.0] < d)))))
    out = {k: np.asarray(v) for k, v in cols.items()}
    # Join the true labels; unlabeled rows (unobserved future) are dropped.
    L = cached_labels(name)
    lk = pair_key(L["t_s"], L["veh"], L["ped"])
    order = np.argsort(lk)
    rk = pair_key(out["t_s"], out["veh"], out["ped"])
    pos = np.searchsorted(lk[order], rk)
    pos = np.clip(pos, 0, len(lk) - 1)
    hit = lk[order][pos] == rk
    li = order[pos[hit]]
    out = {k: v[hit] for k, v in out.items()}
    for k in ("d_now", "d_min", "d_min_moving", "d_min_moving_1"):
        out[k] = L[k][li]
    out["artifact"] = overlap_artifact(L)[li]
    out["_scored"], out["_labelled"] = np.array(len(rk)), np.array(int(hit.sum()))
    path = risk_path(name, models)
    np.savez_compressed(path + ".tmp.npz", **out)
    os.replace(path + ".tmp.npz", path)
    return f"{name} {models}: {len(rk)} pair-updates scored, {int(hit.sum())} with a fully observed future"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["learned", "cv"])
    ap.add_argument("--split", nargs="+", default=["calibration"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    sp = splits()
    jobs = [(n, m) for s in args.split for n in sp[s] for m in args.models
            if os.path.exists(cache_path(n, m)) and (args.force or not os.path.exists(risk_path(n, m)))]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(score_recording, n, m) for n, m in jobs]):
            print(f"[dlp_risk] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
