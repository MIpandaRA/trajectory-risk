"""E1 data: run the live pipeline synchronously over every scenario and cache
every prediction it makes (same schedule as the dashboard: vehicles every
1.0 s, pedestrians every 0.8 s). The accuracy, risk and latency analyses all
read these caches, so the expensive model calls happen once.

    python -m eval.run_predictions                 # learned + cv, all scenarios
    python -m eval.run_predictions --models cv --scenarios t11_quiet
"""
from __future__ import annotations

import argparse
import time

import torch

from trajectory_risk.maps import MapBundle
from trajectory_risk.pipeline import LivePipeline
from trajectory_risk.predictors import factory

from .common import SCENARIOS, load_ticks, save_cache


def run(scenario: str, models: str, seed: int = 0, **factory_kw):
    bundle = MapBundle.load(SCENARIOS[scenario])
    veh, ped = factory.build(bundle, models, seed=seed, **factory_kw)
    pipe = LivePipeline(veh, ped)
    results = []
    t0 = time.time()
    for tick in load_ticks(scenario):
        results.extend(pipe.step_sync(tick))
    return {"scenario": scenario, "models": models, "results": results,
            "wall_s": time.time() - t0, "torch_threads": torch.get_num_threads()}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["cv", "learned"])
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    for models in args.models:
        for sc in args.scenarios:
            out = run(sc, models)
            path = save_cache(f"pred_{sc}_{models}", out)
            n = len(out["results"])
            print(f"[run_predictions] {sc:14s} {models:8s} {n:5d} predictions in {out['wall_s']:.0f}s -> {path}",
                  flush=True)


if __name__ == "__main__":
    main()
