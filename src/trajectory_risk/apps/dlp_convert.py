"""Convert Dragon Lake Parking recordings into replayable tick recordings.

    python scripts/download_dlp.py DJI_0001                    # raw JSON -> data/dlp/raw/
    traj-dlp-convert DJI_0001                                  # -> data/dlp/DJI_0001.jsonl.gz (+ .meta.json)
    traj-dlp-convert DJI_0001 DJI_0012 --build-map             # also (re)build maps/DLP/ from both
    traj-dlp-convert DJI_0001 --build-map --map-variant aisles --no-convert   # maps/DLP_aisles/
    traj-dlp-convert --split calibration --build-map --no-convert              # map from data/dlp/splits.json
    traj-replay data/dlp/DJI_0001.jsonl.gz                     # then: traj-dashboard --map DLP

The .jsonl.gz has the same format as data/scenarios/*.jsonl.gz (hello, then
one tick per 1/60 s), so traj-replay and the evaluation read it unchanged.
The .meta.json next to it holds what the tick protocol has no field for:
DLP tokens and labels per id, censoring at the drone's field-of-view edge,
reversing intervals, and the parked cars that were kept out of the stream.
Output is deterministic: same raw files and options -> identical bytes.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

from ..datasets import dlp, dlp_map
from ..maps import MAPS_DIR, REPO_ROOT
from ..protocol import hello

DEFAULT_RAW = os.path.join(REPO_ROOT, "data", "dlp", "raw")
DEFAULT_OUT = os.path.join(REPO_ROOT, "data", "dlp")


def write_conversion(out_dir: str, conv: dlp.Conversion, map_name: str = "DLP") -> str:
    """-> path of the .jsonl.gz written."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{conv.name}.jsonl.gz")
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write((json.dumps(hello(map_name, conv.rate_hz, source=f"dlp:{conv.name}"),
                            separators=(",", ":")) + "\n").encode())
        for tick in dlp.iter_ticks(conv):
            f.write((json.dumps(tick.to_dict(), separators=(",", ":")) + "\n").encode())
    with open(os.path.join(out_dir, f"{conv.name}.meta.json"), "w") as f:
        json.dump(conv.meta, f, indent=1)
        f.write("\n")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="*", help="e.g. DJI_0001")
    ap.add_argument("--split", default=None,
                    help="use the recordings of this split in --splits (e.g. calibration); "
                         "recorded in map_meta.json when building the map")
    ap.add_argument("--splits", default=os.path.join(DEFAULT_OUT, "splits.json"))
    ap.add_argument("--raw", default=DEFAULT_RAW, help="folder with the DLP JSON files")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cyclists", choices=dlp.CYCLIST_POLICIES, default="drop",
                    help="what bicycles become (default: left out; see docs/data_dlp.md)")
    ap.add_argument("--no-convert", action="store_true", help="only build the map")
    ap.add_argument("--build-map", action="store_true", help="build the map bundle from these recordings")
    ap.add_argument("--map-variant", choices=dlp_map.VARIANTS, default="lot",
                    help="drivable = the whole paved lot (default) or aisles only (ablation)")
    ap.add_argument("--map-name", default=None, help="default: DLP, or DLP_aisles for --map-variant aisles")
    ap.add_argument("--maps-dir", default=MAPS_DIR)
    args = ap.parse_args(argv)

    map_name = args.map_name or ("DLP" if args.map_variant == "lot" else "DLP_aisles")
    names, provenance = list(args.recordings), None
    if args.split:
        with open(args.splits) as f:
            splits = json.load(f)
        names += [n for n in splits[args.split] if n not in names]
        provenance = {"built_from_split": args.split,
                      "split_file": os.path.relpath(args.splits, REPO_ROOT),
                      "split_seed": splits.get("seed"),
                      "excluded_recordings": sorted(n for k, v in splits.items()
                                                    if k in ("calibration", "test") and k != args.split
                                                    for n in v)}
    if not names:
        ap.error("give recordings or --split")
    raws = []
    for name in names:
        rec = dlp.load_raw(args.raw, name)
        if args.build_map:  # a raw recording is ~0.5-1 GB in memory; keep only if needed
            raws.append(rec)
        if args.no_convert:
            continue
        conv = dlp.convert(rec, cyclists=args.cyclists)
        path = write_conversion(args.out, conv, map_name)
        print(f"[dlp] {name}: {len(conv.tracks)} agents, {conv.k_end + 1} ticks -> {path}")
    if args.build_map:
        out_dir = os.path.join(args.maps_dir, map_name)
        mask = dlp_map.build_bundle(raws, out_dir, args.map_variant, provenance=provenance)
        print(f"[dlp] map {map_name} ({args.map_variant}): {int(mask.sum())} m2 drivable -> {out_dir}")


if __name__ == "__main__":
    main()
