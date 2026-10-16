"""Assign DLP recordings to calibration / test splits, once, before any model runs.

    python scripts/make_dlp_splits.py                 # writes data/dlp/splits.json
    python scripts/make_dlp_splits.py --add-fov       # later: add each recording's drone x-range

The split is by recording, not by frame or track: frames of one recording
share the drone position, the parked cars and the people, so splitting
inside a recording would leak. The assignment is a seeded permutation, so it
is reproducible and was not chosen by looking at any data or model output.
--add-fov only annotates the existing split (from data/dlp/<rec>.meta.json);
it never reassigns.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUT = os.path.join(ROOT, "data", "dlp", "splits.json")
GATE_XY_M = (15.0, 75.0)   # the lot's only entrance; see docs/data_dlp.md
GATE_MARGIN_M = 3.0


def assign(names, seed: int, n_calibration: int) -> dict:
    """numpy.random.default_rng(seed).permutation over the sorted names; the
    first n_calibration go to calibration, the rest to test."""
    names = sorted(names)
    perm = np.random.default_rng(seed).permutation(len(names))
    shuffled = [names[i] for i in perm]
    return {"calibration": sorted(shuffled[:n_calibration]), "test": sorted(shuffled[n_calibration:]),
            "permutation": shuffled}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-calibration", type=int, default=10)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--add-fov", action="store_true")
    args = ap.parse_args(argv)

    if args.add_fov:
        with open(args.out) as f:
            splits = json.load(f)
        data_dir = os.path.dirname(args.out)
        fov = {}
        for name in splits["calibration"] + splits["test"]:
            with open(os.path.join(data_dir, f"{name}.meta.json")) as f:
                m = json.load(f)
            (x0, x1), (y0, y1) = m["fov_x_m"], m["fov_y_m"]
            gx, gy = GATE_XY_M
            fov[name] = {"fov_x_m": [x0, x1], "fov_y_m": [y0, y1],
                         "gate_visible": bool(x0 <= gx - GATE_MARGIN_M and y1 >= gy - 5.0)}
        splits["recordings"] = fov
        splits["fov_note"] = ("fov_x_m / fov_y_m = extent of agent positions in the recording (a lower bound "
                              "on the drone's view); gate_visible = that box reaches x <= "
                              f"{GATE_XY_M[0] - GATE_MARGIN_M} m and y >= {GATE_XY_M[1] - 5.0} m, so the entrance "
                              f"at {GATE_XY_M} is in view.")
    else:
        names = [f"DJI_{i:04d}" for i in range(args.first, args.last + 1)]
        splits = {"seed": args.seed, "method": "numpy.random.default_rng(seed).permutation over the sorted "
                                                f"names; first {args.n_calibration} -> calibration, rest -> test",
                  "reason": "Decided before any model output was seen; split by recording to avoid leakage "
                            "between correlated frames (same drone position, parked cars and people).",
                  "created": datetime.date.today().isoformat(),
                  **assign(names, args.seed, args.n_calibration)}
        if os.path.exists(args.out):
            raise SystemExit(f"{args.out} exists; the split is fixed. Delete it deliberately to redo it.")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(splits, f, indent=1)
        f.write("\n")
    print(json.dumps({k: splits[k] for k in ("calibration", "test")}))


if __name__ == "__main__":
    main()
