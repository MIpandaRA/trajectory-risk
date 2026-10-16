"""Record everything a producer broadcasts into a .jsonl(.gz) file.

    traj-record data/scenarios/my_run.jsonl.gz --seconds 60

Each line is {"recv_t": <wall clock>, "msg": <message>}, so a replay can
reproduce the original pacing.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import socket
import time

from ..transport import DEFAULT_HOST, DEFAULT_PORT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_path")
    ap.add_argument("--seconds", type=float, default=60.0, help="stop after this much simulation time")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args(argv)

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    sock = socket.create_connection((args.host, args.port), timeout=10.0)
    fh = sock.makefile("r", encoding="utf-8")
    opener = gzip.open if args.out_path.endswith(".gz") else open
    n, t_first = 0, None
    with opener(args.out_path, "wt", encoding="utf-8") as out:
        for line in fh:  # one JSON object per line: readline() does the framing
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            out.write(json.dumps({"recv_t": round(time.time(), 4), "msg": msg}, separators=(",", ":")) + "\n")
            if msg.get("type") == "tick":
                n += 1
                t = msg.get("t_s", msg.get("t", 0.0))
                t_first = t if t_first is None else t_first
                if t - t_first >= args.seconds:
                    break
    sock.close()
    print(f"[record] wrote {n} ticks to {args.out_path}")


if __name__ == "__main__":
    main()
