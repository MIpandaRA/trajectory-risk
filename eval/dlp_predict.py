"""DLP step 2 data: run the live pipeline synchronously over each converted
recording and cache every prediction (same schedule as the dashboard:
vehicles every 1.0 s, pedestrians every 0.8 s).

Deployed configuration: ContextVAE N = 500, k = 10 on maps/DLP (built from
the calibration split only); SocialVAE N = 740, k = 20; plus the
constant-velocity baseline. Seeded: each predictor's RNG is seed + call
count, so a recording replays identically. One cache file per
(recording, models); an existing file is skipped, so an interrupted run
resumes at the next recording.

    python -m eval.dlp_predict --models cv learned --split calibration test --workers 4
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .common import CACHE_DIR
from .dlp_labels import DLP_DIR, splits

SEED = 0
RAW_SAMPLES_VEHICLE = 500      # dashboard default (docs/REPORT.md section 5)
MAP_NAME = "DLP"


def iter_ticks(name: str, max_seconds: float = 0.0):
    """Stream validated ticks from the converted recording (no full list in memory)."""
    from trajectory_risk.protocol import Tick
    with gzip.open(os.path.join(DLP_DIR, f"{name}.jsonl.gz"), "rt") as f:
        for line in f:
            msg = json.loads(line)
            if msg.get("type") != "tick":
                continue
            tick = Tick.from_dict(msg)
            if max_seconds and tick.t_s > max_seconds:
                return
            yield tick


def compact_result(r) -> dict:
    return {"type": r.agent_type, "id": r.agent_id, "k_idx": r.sample_index, "anchor": r.anchor_time_s,
            "dt": r.step_dt_s, "model": r.model, "ms": r.runtime_ms,
            "traj": np.array([c.trajectory for c in r.candidates], dtype=np.float32),
            "conf": np.array([c.confidence for c in r.candidates], dtype=np.float32)}


def compact_request(q) -> dict:
    return {"id": q.agent_id, "k_idx": q.sample_index, "anchor": q.anchor_time_s,
            "positions": q.positions.astype(np.float32), "headings": q.headings.astype(np.float32),
            "neighbours": [n.astype(np.float32) for n in q.neighbours]}


def cache_path(name: str, models: str) -> str:
    return os.path.join(CACHE_DIR, f"dlp_pred_{name}_{models}.pkl")


def run(name: str, models: str, threads: int = 2, max_seconds: float = 0.0, out: str = None) -> str:
    import pickle

    import torch

    from trajectory_risk.maps import MapBundle
    from trajectory_risk.pipeline import LivePipeline
    from trajectory_risk.predictors import factory

    torch.set_num_threads(threads)
    bundle = MapBundle.load(MAP_NAME)
    veh, ped = factory.build(bundle, models, raw_samples=RAW_SAMPLES_VEHICLE, seed=SEED)
    pipe = LivePipeline(veh, ped)
    results, vreqs = [], []
    t0 = time.time()
    for tick in iter_ticks(name, max_seconds):
        for q in pipe.ingest(tick):        # the risk pass is separate (eval/dlp_risk.py)
            res = pipe.predict_now(q)
            pipe.accept(res)
            results.append(compact_result(res))
            if q.agent_type == "vehicle":
                vreqs.append(compact_request(q))
    payload = {"recording": name, "models": models, "map": MAP_NAME, "seed": SEED,
               "raw_samples_vehicle": RAW_SAMPLES_VEHICLE, "results": results, "vehicle_requests": vreqs,
               "wall_s": time.time() - t0, "torch_threads": threads, "max_seconds": max_seconds}
    path = out or cache_path(name, models)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(path + ".tmp", path)    # atomic: a killed run never leaves a half cache
    return f"{name} {models}: {len(results)} predictions in {payload['wall_s']:.0f} s"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["cv", "learned"])
    ap.add_argument("--split", nargs="+", default=["calibration", "test"])
    ap.add_argument("--recordings", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-seconds", type=float, default=0.0, help="benchmark: stop after this much recording time")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    sp = splits()
    names = args.recordings or [n for s in args.split for n in sp[s]]
    jobs = [(n, m) for m in args.models for n in names
            if args.max_seconds or not os.path.exists(cache_path(n, m))]
    print(f"[dlp_predict] {len(jobs)} jobs ({len(names) * len(args.models) - len(jobs)} cached)", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run, n, m, args.threads, args.max_seconds, args.out) for n, m in jobs]
        for fut in as_completed(futs):
            print(f"[dlp_predict] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
