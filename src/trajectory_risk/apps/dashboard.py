"""Window 2: live predictions, collision risk and alerts.

    traj-dashboard --map T11                 # ContextVAE + SocialVAE
    traj-dashboard --map T11 --models cv     # constant velocity, no weights needed

Keys: D debug overlay (latency, superseded jobs, fps), A toggle alternate
candidates, Esc quit.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import time

from ..maps import MapBundle
from ..pipeline import LivePipeline, PipelineConfig
from ..predictors import factory
from ..protocol import ProtocolError, Tick
from ..risk import DEFAULT_SAFETY_BUFFER_M
from ..runtime.worker import InferenceWorker, split_torch_threads
from ..transport import DEFAULT_HOST, DEFAULT_PORT, LiveStateClient


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", default="T11", help="must match the producer's map")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--models", choices=["learned", "cv"], default="learned")
    ap.add_argument("--raw-samples", type=int, default=500,
                    help="ContextVAE N. 500 keeps 99%% of N=2000's best-of-10 accuracy at ~40%% of the cost "
                         "(docs/REPORT.md section 5)")
    ap.add_argument("--ped-raw-multiplier", type=int, default=None, help="SocialVAE N = this x 20 (default 37)")
    ap.add_argument("--predict-every", type=int, default=2, help="vehicle re-prediction interval, in 0.5 s samples")
    ap.add_argument("--predict-every-ped", type=int, default=2, help="pedestrian interval, in 0.4 s samples")
    ap.add_argument("--safety-buffer-m", type=float, default=DEFAULT_SAFETY_BUFFER_M)
    ap.add_argument("--risk-enter-threshold", type=float, default=0.6)
    ap.add_argument("--risk-enter-ticks", type=int, default=1)
    ap.add_argument("--risk-clear-threshold", type=float, default=0.2)
    ap.add_argument("--risk-clear-ticks", type=int, default=5)
    ap.add_argument("--inject-delay-ms", type=float, default=0, help="stress test: slow every inference down")
    ap.add_argument("--log-predictions", default=None, help="write every prediction and risk row to this .jsonl")
    ap.add_argument("--save-frames", default=None, help="save every Nth frame as PNG into this folder")
    ap.add_argument("--save-every", type=int, default=4)
    ap.add_argument("--max-seconds", type=float, default=0, help="quit after this much simulation time")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--debug", action="store_true", help="start with the debug overlay on")
    return ap.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    from ..dashboard.render import CANVAS_H, CANVAS_W, DashboardRenderer

    bundle = MapBundle.load(args.map)
    print(f"[dashboard] loading {args.models} models...")
    veh, ped = factory.build(bundle, args.models, raw_samples=args.raw_samples,
                             ped_raw_multiplier=args.ped_raw_multiplier)
    split_torch_threads(2)
    cfg = PipelineConfig(args.predict_every, args.predict_every_ped, 0.2, args.safety_buffer_m,
                         args.risk_enter_threshold, args.risk_enter_ticks,
                         args.risk_clear_threshold, args.risk_clear_ticks)
    pipe = LivePipeline(veh, ped, cfg)
    delay = args.inject_delay_ms / 1000.0
    workers = {"vehicle": InferenceWorker(veh, "vehicle", delay),
               "pedestrian": InferenceWorker(ped, "pedestrian", delay)}
    log = open(args.log_predictions, "w", encoding="utf-8") if args.log_predictions else None

    pygame.init()
    screen = pygame.display.set_mode((CANVAS_W, CANVAS_H))
    pygame.display.set_caption("Window 2 - predictions and collision risk")
    renderer = DashboardRenderer(bundle)
    client = LiveStateClient(args.host, args.port)
    clock = pygame.time.Clock()
    show_alt, debug = True, args.debug
    frame_i, bad_msgs = 0, 0
    if args.save_frames:
        os.makedirs(args.save_frames, exist_ok=True)

    try:
        while True:
            clock.tick(60)
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    return
                if e.type == pygame.KEYDOWN and e.key == pygame.K_d:
                    debug = not debug
                if e.type == pygame.KEYDOWN and e.key == pygame.K_a:
                    show_alt = not show_alt

            while True:
                try:
                    msg = client.queue.get_nowait()
                except queue.Empty:
                    break
                if msg.get("type") != "tick":
                    continue
                try:
                    tick = Tick.from_dict(msg)
                except ProtocolError as e:
                    bad_msgs += 1
                    if bad_msgs <= 5:
                        print(f"[dashboard] dropped invalid tick: {e}")
                    continue
                for req in pipe.ingest(tick):
                    workers[req.agent_type].submit(req)

            for w in workers.values():
                for res in w.drain():
                    if pipe.accept(res) and log:
                        log.write(json.dumps(_pred_record(res)) + "\n")
            rows = pipe.update_risk()
            if rows is not None and log:
                for r in rows:
                    log.write(json.dumps({"kind": "risk", "t_s": round(pipe.now_s, 3), "veh": r.veh_id,
                                          "ped": r.ped_id, "risk": round(r.risk, 4),
                                          "d_min": round(r.d_min, 3), "active": r.active}) + "\n")

            status = {"fps": clock.get_fps(), "source": f"{args.host}:{args.port}",
                      "vehicle_stats": workers["vehicle"].stats, "pedestrian_stats": workers["pedestrian"].stats}
            renderer.draw(screen, pipe, status, show_alt, debug)
            pygame.display.flip()
            frame_i += 1
            if args.save_frames and frame_i % args.save_every == 0 and pipe.now_s > 0:
                pygame.image.save(screen, os.path.join(args.save_frames, f"frame_{frame_i:06d}.png"))
            if args.max_seconds and pipe.now_s >= args.max_seconds:
                return
    finally:
        client.close()
        for w in workers.values():
            w.close()
        if log:
            log.close()
        pygame.quit()


def _pred_record(res) -> dict:
    return {"kind": "pred", "agent_type": res.agent_type, "id": res.agent_id, "sample_index": res.sample_index,
            "anchor_time_s": round(res.anchor_time_s, 3), "runtime_ms": round(res.runtime_ms, 1),
            "model": res.model,
            "candidates": [{"conf": round(float(c.confidence), 4), "xy": c.trajectory.round(3).tolist()}
                           for c in res.candidates]}


if __name__ == "__main__":
    main()
