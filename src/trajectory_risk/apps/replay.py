"""Stand in for the simulator: serve a recording over TCP.

    traj-replay data/scenarios/t11_crossing.jsonl.gz
    traj-dashboard --map T11          # in another terminal

The dashboard cannot tell this from the live simulator. Pacing follows the
ticks' simulation time (``--speed 2`` for double speed). With ``--loop`` the
dashboard sees time jump backwards and resets its tracks, as designed.
"""
from __future__ import annotations

import argparse
import time

from ..protocol import hello, load_recording
from ..transport import DEFAULT_HOST, DEFAULT_PORT, LiveStateServer


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-wait", action="store_true", help="start immediately instead of waiting for a consumer")
    args = ap.parse_args(argv)

    hello_msg, ticks = load_recording(args.recording)
    if hello_msg is None:
        hello_msg = hello("unknown", source="replay")
    hello_msg = dict(hello_msg, source="replay")
    span = ticks[-1].t_s - ticks[0].t_s
    print(f"[replay] {len(ticks)} ticks, {span:.1f} s of simulation, map={hello_msg.get('map')}")

    server = LiveStateServer(args.host, args.port, hello=hello_msg)
    if not args.no_wait:
        print("[replay] waiting for a consumer...")
        while server.n_clients == 0:
            server.accept_pending()
            time.sleep(0.05)
    try:
        while True:
            t0_wall, t0_sim = time.perf_counter(), ticks[0].t_s
            for tick in ticks:
                delay = t0_wall + (tick.t_s - t0_sim) / args.speed - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                server.accept_pending()
                server.broadcast(tick.to_dict())
            print("[replay] finished")
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    finally:
        server.close()


if __name__ == "__main__":
    main()
