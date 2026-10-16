"""Download selected Dragon Lake Parking (DLP) recordings from Zenodo.

    python scripts/download_dlp.py DJI_0001 DJI_0012
    python scripts/download_dlp.py DJI_0001 --out data/dlp/raw

Each recording is five JSON files (agents, frames, instances, obstacles,
scene). The full dataset is ~8 GB, so only the recordings you name are
fetched. Downloads resume from a partial file (HTTP Range) and a stalled
connection is retried rather than waited on forever: the instances files are
60-480 MB and Zenodo connections do stall. Every file is checked against the
md5 the Zenodo record publishes; a mismatch deletes the file.

Data: Shen, Pan, Govindarajan, Borrelli, "Dragon Lake Parking Dataset",
https://zenodo.org/records/10084683, CC0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RECORD_API = "https://zenodo.org/api/records/10084683"
PARTS = ("agents", "frames", "instances", "obstacles", "scene")
CHUNK = 1 << 20


def record_files() -> dict:
    """-> {filename: (size, md5, url)} from the Zenodo record."""
    with urllib.request.urlopen(RECORD_API, timeout=60) as r:
        rec = json.load(r)
    out = {}
    for f in rec["files"]:
        algo, digest = f["checksum"].split(":", 1)
        assert algo == "md5", f["checksum"]
        out[f["key"]] = (int(f["size"]), digest, f["links"]["self"])
    return out


def md5sum(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: str, size: int, retries: int = 20, timeout_s: float = 60.0) -> None:
    """Append to ``dest`` from its current length until it reaches ``size``.
    A read that sees no bytes for ``timeout_s`` raises, and the next attempt
    resumes from where the file ends."""
    for attempt in range(1, retries + 1):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= size:
            return
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                if have and r.status != 206:  # server ignored Range: start over
                    have = 0
                with open(dest, "ab" if have else "wb") as f:
                    for block in iter(lambda: r.read(CHUNK), b""):
                        f.write(block)
        except (urllib.error.URLError, OSError) as e:  # includes socket timeouts
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  attempt {attempt}: {e!r} at {got / 1e6:.1f}/{size / 1e6:.1f} MB, resuming")
            time.sleep(min(30, 2 * attempt))
    if os.path.getsize(dest) < size:
        sys.exit(f"gave up on {dest}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+", help="e.g. DJI_0001")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "dlp", "raw"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    files = record_files()
    for rec in args.recordings:
        for part in PARTS:
            name = f"{rec}_{part}.json"
            if name not in files:
                sys.exit(f"{name} is not in the Zenodo record")
            size, md5, url = files[name]
            dest = os.path.join(args.out, name)
            if os.path.exists(dest) and os.path.getsize(dest) == size and md5sum(dest) == md5:
                print(f"[ok]   {name}")
                continue
            if os.path.exists(dest) and os.path.getsize(dest) > size:
                os.remove(dest)
            print(f"[get]  {name} ({size / 1e6:.1f} MB)")
            fetch(url, dest, size)
            if md5sum(dest) != md5:
                os.remove(dest)
                sys.exit(f"md5 mismatch for {name}; deleted, rerun to fetch again")
            print(f"[ok]   {name} md5 verified")
    print("done")


if __name__ == "__main__":
    main()
