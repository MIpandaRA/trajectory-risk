"""Generate a scenario recording headlessly (much faster than real time).

    traj-generate data/scenarios/t11_busy.jsonl.gz --map T11 --num-pedestrians 10 --seconds 180 --seed 3

Same seed and settings -> byte-identical file, so scenarios are reproducible
from the command line in the README rather than opaque data files.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

from ..protocol import hello
from ..sim import WorldConfig, generate


def write_recording(path: str, ticks, map_name: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    opener = (lambda p: gzip.GzipFile(p, "wb", mtime=0)) if path.endswith(".gz") else (lambda p: open(p, "wb"))
    with opener(path) as f:
        f.write((json.dumps(hello(map_name, source="generator"), separators=(",", ":")) + "\n").encode())
        for t in ticks:
            f.write((json.dumps(t.to_dict(), separators=(",", ":")) + "\n").encode())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_path")
    ap.add_argument("--map", default="T11")
    ap.add_argument("--num-vehicles", type=int, default=0)
    ap.add_argument("--num-pedestrians", type=int, default=6)
    ap.add_argument("--crossing-bias", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--warmup", type=float, default=10.0)
    ap.add_argument("--vehicle-stagger", type=float, default=6.0, help="seconds between routes' first vehicles")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = WorldConfig(args.map, args.num_vehicles, args.num_pedestrians,
                      crossing_bias=args.crossing_bias, vehicle_stagger_s=args.vehicle_stagger, seed=args.seed)
    ticks = generate(cfg, args.seconds, args.warmup)
    write_recording(args.out_path, ticks, args.map)
    print(f"[generate] {len(ticks)} ticks -> {args.out_path}")


if __name__ == "__main__":
    main()
