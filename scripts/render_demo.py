"""Render the dashboard offline: a GIF/MP4 for the README.

Uses the same pipeline and the same renderer as the live dashboard, with
predictions computed synchronously (or taken from eval/cache if the
experiments have already run, which makes this take seconds).

    python scripts/render_demo.py --scenario t11_crossing --start 40 --seconds 14 --out demo/demo.gif

Real data (Dragon Lake Parking): a converted recording on maps/DLP with the
cached ContextVAE + SocialVAE predictions (eval/dlp_predict.py). The risk
panel shows the CALIBRATED risk: the motion-aware score (b) at 2.5 m mapped
through the isotonic map fitted on the calibration recordings. The alert
uses the demo setting chosen on those recordings
(eval/results/dlp_demo_alert_setting.json; docs/dlp_evaluation.md section 11).
The view is a fixed-size window around --center.

    python scripts/render_demo.py --dlp DJI_0003 --start 51.2 --seconds 18 --center 27.3 31.0 --out demo/dlp_demo.gif
    python scripts/render_demo.py --dlp DJI_0003 --start 51.2 --center 27.3 31.0 --preview 6 8 10 --out preview.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from trajectory_risk.dashboard.render import CANVAS_H, CANVAS_W, DashboardRenderer, init_headless  # noqa: E402
from trajectory_risk.maps import MapBundle  # noqa: E402
from trajectory_risk.pipeline import LivePipeline  # noqa: E402
from trajectory_risk.predictors import factory  # noqa: E402


class CalibratedPipeline(LivePipeline):
    """LivePipeline whose risk rows and alerts use a calibrated probability:
    transform(raw motion-aware score). Risk is only computed from risk_from_s
    on, to skip the minutes before a clip (alerts need a few updates of
    history, which the caller provides by starting earlier)."""

    def __init__(self, veh, ped, cfg, transform, risk_from_s: float):
        super().__init__(veh, ped, cfg)
        self.transform, self.risk_from_s = transform, risk_from_s

    def update_risk(self, force: bool = False):
        from trajectory_risk.risk import RiskRow, pairwise_risk
        if self.now_s < self.risk_from_s - 1e-9:
            return None
        if not force and self.now_s - self._last_risk_s < self.cfg.risk_interval_s - 1e-9:
            return None
        self._last_risk_s = self.now_s
        rows, live = [], set()
        for vid, v in self.held("vehicle").items():
            for pid, p in self.held("pedestrian").items():
                raw, d_min = pairwise_risk(v, p, self.now_s, self.cfg.safety_buffer_m,
                                           vehicle_min_speed_mps=self.cfg.risk_vehicle_min_speed_mps)
                risk = float(self.transform(raw))
                key = (vid, pid)
                live.add(key)
                alert = self.alerts.get(key) or self.alerts.setdefault(key, self._new_alert())
                rows.append(RiskRow(vid, pid, risk, d_min, alert.update(risk)))
        for key in [k for k in self.alerts if k not in live]:
            del self.alerts[key]
        rows.sort(key=lambda r: r.risk, reverse=True)
        self.risk_rows = rows
        return rows


def zoomed(bundle: MapBundle, cx: float, cy: float, width_m: float) -> MapBundle:
    """The same map, viewed through a 4:3 window of width_m centred on (cx, cy)."""
    meta = dict(bundle.meta, world_origin_x=cx - width_m / 2, world_origin_y=cy - width_m * 3 / 8,
                world_width_m=width_m, world_height_m=width_m * 3 / 4)
    return MapBundle(bundle.name, bundle.root, meta)


def dlp_setup(args):
    import json
    import pickle

    import numpy as np

    from eval.dlp_predict import cache_path, iter_ticks
    from eval.dlp_risk import Cached
    from trajectory_risk.pipeline import PipelineConfig

    with open(os.path.join(ROOT, "eval", "results", "dlp_demo_alert_setting.json")) as f:
        c = json.load(f)
    with open(cache_path(args.dlp, "learned"), "rb") as f:
        res = pickle.load(f)["results"]
    veh = Cached(res, "vehicle", 0.5, 5, "contextvae")
    ped = Cached(res, "pedestrian", 0.4, 8, "socialvae")
    cfg = PipelineConfig(safety_buffer_m=2.5, risk_vehicle_min_speed_mps=0.5,
                         enter_threshold=c["enter_threshold"], enter_ticks=c["enter_ticks"],
                         clear_threshold=c["clear_threshold"], clear_ticks=c["clear_ticks"])
    x, y = np.array(c["isotonic_x"]), np.array(c["isotonic_y"])
    # Start risk on the same 5 Hz grid as the live pipeline, labels and alert
    # simulation (multiples of 0.2 s from t = 0); an offset start shifts every
    # update by up to 0.1 s and changes which updates cross the threshold.
    first = max(0.0, np.floor((args.start - 5.0) / cfg.risk_interval_s) * cfg.risk_interval_s)
    pipe = CalibratedPipeline(veh, ped, cfg, lambda r: np.interp(r, x, y), risk_from_s=first)
    bundle = MapBundle.load("DLP")
    if args.center:
        bundle = zoomed(bundle, args.center[0], args.center[1], args.view_m)
    status = {"source": f"DLP {args.dlp} (real data)", "risk_label": "calibrated risk",
              "risk_note": f"alert at >= {c['enter_threshold']:.2f} for {c['enter_ticks']} updates",
              "alert_threshold": c["enter_threshold"]}
    return pipe, bundle, iter_ticks(args.dlp), status


def main() -> None:
    from eval.common import SCENARIOS, load_cache, load_ticks
    from eval.risk_eval import Cached

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="t11_crossing")
    ap.add_argument("--start", type=float, default=40.0, help="seconds into the scenario")
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.65)
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "demo.gif"))
    ap.add_argument("--still", default=None, help="also save one full-size PNG here...")
    ap.add_argument("--still-at", type=float, default=7.0, help="...taken this many seconds into the clip")
    ap.add_argument("--dlp", default=None, help="a converted DLP recording, e.g. DJI_0003 (real data)")
    ap.add_argument("--center", type=float, nargs=2, default=None, help="DLP: view centre x y (m)")
    ap.add_argument("--view-m", type=float, default=60.0, help="DLP: view width in metres")
    ap.add_argument("--preview", type=float, nargs="+", default=None,
                    help="instead of a video, save a contact sheet of stills at these clip seconds to --out")
    args = ap.parse_args()

    init_headless()
    import pygame
    if args.dlp:
        pipe, bundle, ticks, status = dlp_setup(args)
        t0 = args.start
    else:
        bundle = MapBundle.load(SCENARIOS[args.scenario])
        try:
            res = load_cache(f"pred_{args.scenario}_learned")["results"]
            veh = Cached(res, "vehicle", 0.5, 5, "contextvae")
            ped = Cached(res, "pedestrian", 0.4, 8, "socialvae")
            print("[demo] using cached ContextVAE/SocialVAE predictions")
        except FileNotFoundError:
            veh, ped = factory.build(bundle, "learned")
            print("[demo] running models live (slower)")
        pipe = LivePipeline(veh, ped)
        ticks = load_ticks(args.scenario)
        t0 = ticks[0].t_s + args.start
        status = {"source": f"replay: {args.scenario}"}
    renderer = DashboardRenderer(bundle)
    surf = pygame.Surface((CANVAS_W, CANVAS_H))
    if args.preview:
        stills = []
        want = sorted(t0 + p for p in args.preview)
        for tick in ticks:
            pipe.step_sync(tick)
            if want and tick.t_s >= want[0] - 1e-9:
                renderer.draw(surf, pipe, status)
                stills.append(pygame.transform.smoothscale(surf, (CANVAS_W // 2, CANVAS_H // 2)))
                want.pop(0)
            if not want:
                break
        sheet = pygame.Surface((CANVAS_W // 2 * len(stills), CANVAS_H // 2))
        for i, st in enumerate(stills):
            sheet.blit(st, (i * CANVAS_W // 2, 0))
        pygame.image.save(sheet, args.out)
        print(f"[demo] preview with {len(stills)} stills -> {args.out}")
        return
    every = max(1, round(60 / args.fps))
    out_size = (int(CANVAS_W * args.scale) // 2 * 2, int(CANVAS_H * args.scale) // 2 * 2)
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for i, tick in enumerate(ticks):
            pipe.step_sync(tick)
            if tick.t_s < t0 or i % every:
                continue
            if tick.t_s > t0 + args.seconds:
                break
            renderer.draw(surf, pipe, status)
            if args.still and abs(tick.t_s - (t0 + args.still_at)) < 1 / args.fps:
                pygame.image.save(surf, args.still)
            pygame.image.save(pygame.transform.smoothscale(surf, out_size), os.path.join(tmp, f"f{n:05d}.png"))
            n += 1
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pattern = os.path.join(tmp, "f%05d.png")
        if args.out.endswith(".gif"):
            pal = os.path.join(tmp, "palette.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern,
                            "-vf", "palettegen=max_colors=128:stats_mode=diff", pal], check=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern, "-i", pal,
                            "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                            "-loop", "0", args.out], check=True)
        else:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", args.out], check=True)
    print(f"[demo] {n} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
