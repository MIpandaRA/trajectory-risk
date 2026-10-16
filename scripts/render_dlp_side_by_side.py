"""Render a DLP scene twice, side by side, as one GIF for the README:

* left:  a semantic view of the car park, drawn only from the CC0 recordings:
         aisles (grey), parked cars that never move (blue), parking spaces
         that are empty in this recording (green), tracked vehicles at their
         real size and heading (yellow) and pedestrians (pink);
* right: the live dashboard on the same seconds (predictions, calibrated
         risk and alerts), exactly as scripts/render_demo.py draws it.

Both halves show the same 60 m x 45 m window at the same scale, frame for
frame, so every agent sits at the same height on both sides.

Parking spaces are estimated, not drawn from the DLP toolkit's map (GPL-3):
parked-car sightings from all 16 recordings are grouped into rows, and each
row is fitted with a regular grid of spaces. Spaces that no car used in any
recording can be missing. Aisles come from maps/DLP_aisles (built from the
calibration recordings), simplified to straight-edged polygons.

    python scripts/render_dlp_side_by_side.py --dlp DJI_0003 --start 51.2 --seconds 18 \
        --center 27.3 31.0 --out demo/dlp_side_by_side.gif
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from trajectory_risk.maps import MapBundle  # noqa: E402

DLP_DIR = os.path.join(ROOT, "data", "dlp")
SS = 3                                   # supersampling for smooth edges
BG, AISLE, PARKED, EMPTY, MOVING, PED = "#000000", "#b9b9b9", "#2442d6", "#1fae4b", "#f2d01c", "#ff5cc8"
PANEL_BG, INK, MUTED = (34, 34, 40), (235, 235, 235), (160, 160, 168)
STALL_DEPTH_M = 5.0
ROW_GAP_M = 1.5                          # parked-car centres further apart than this in y are different rows
SAME_STALL_M = 1.2                       # sightings closer than this in x are the same space
SEGMENT_BREAK = 4.0                      # a gap of more than this many spaces starts a new segment of a row


def load_render_demo():
    spec = importlib.util.spec_from_file_location("render_demo", os.path.join(ROOT, "scripts", "render_demo.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- layout (static) --------------------------------------------------------
def parked_sightings():
    """(x, y, heading, length, width) of every parked car in every converted recording."""
    out = []
    for p in sorted(glob.glob(os.path.join(DLP_DIR, "DJI_*.meta.json"))):
        with open(p) as f:
            for o in json.load(f)["obstacles"]:
                out.append((o["x_m"], o["y_m"], o["heading_rad"], o["length_m"], o["width_m"]))
    return np.array(out, dtype=np.float64)


def _clusters(values, gap):
    """Split sorted values wherever consecutive ones are more than gap apart."""
    order = np.argsort(values)
    v = values[order]
    breaks = np.where(np.diff(v) > gap)[0] + 1
    return [order[s] for s in np.split(np.arange(len(v)), breaks)]


def estimate_stalls(parked, aisle_drivable):
    """-> list of (x, y, width) parking spaces, all perpendicular to the aisles.

    Only near-vertical cars (99% of DLP's parked cars) define spaces. Each row
    is split into segments at long gaps, and each segment is fitted with a
    regular grid x = x0 + k * pitch, so a space that was empty in some
    recordings is still found if neighbouring spaces were used."""
    h = np.mod(parked[:, 2], np.pi)
    vert = parked[np.abs(h - np.pi / 2) < math.radians(20)]
    stalls = []
    for row_idx in _clusters(vert[:, 1], ROW_GAP_M):
        if len(row_idx) < 10:
            continue
        row = vert[row_idx]
        y = float(np.median(row[:, 1]))
        centres = np.array([row[i, 0].mean() for i in _clusters(row[:, 0], SAME_STALL_M)])
        centres.sort()
        d = np.diff(centres)
        steps = d[(d > 2.2) & (d < 3.4)]
        pitch = float(np.median(steps)) if len(steps) else 2.7
        for seg in np.split(centres, np.where(d > SEGMENT_BREAK * pitch)[0] + 1):
            k = np.round((seg - seg[0]) / pitch)
            if len(seg) >= 2 and k[-1] > 0:
                p, x0 = np.polyfit(k, seg, 1)
            else:
                p, x0 = pitch, float(seg[0])
            for kk in range(int(k[-1]) + 1):
                x = x0 + kk * p
                # drop spaces that would sit on an aisle
                xs, ys = np.meshgrid(np.linspace(x - p / 3, x + p / 3, 5), np.linspace(y - 2, y + 2, 5))
                if aisle_drivable(np.stack([xs, ys], -1)).mean() < 0.25:
                    stalls.append((x, y, p))
    return stalls


def aisle_polygons(bundle, x0, y0, w, h, res=0.1):
    """Straight-edged polygons (world metres) of the aisles in a window."""
    xs = x0 + (np.arange(int(w / res)) + 0.5) * res
    ys = y0 + (np.arange(int(h / res)) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    mask = bundle.is_drivable(np.stack([gx, gy], -1)).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(1.0 / res), int(1.0 / res)))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c, hc in zip(contours, hierarchy[0] if hierarchy is not None else []):
        c = cv2.approxPolyDP(c, 0.6 / res, True)[:, 0, :].astype(np.float64)
        world = np.stack([x0 + (c[:, 0] + 0.5) * res, y0 + (c[:, 1] + 0.5) * res], -1)
        polys.append((world, hc[3] >= 0))           # (polygon, is_hole)
    return polys


# -- drawing ------------------------------------------------------------------
class SemanticView:
    def __init__(self, name, cx, cy, view_m, map_px, caption_h):
        self.name, self.map_w = name, map_px[0]
        self.view_w, self.view_h = view_m, view_m * 3 / 4
        self.x0, self.y0 = cx - view_m / 2, cy - view_m * 3 / 8
        self.ppm = map_px[0] / view_m
        self.size = (map_px[0], map_px[1] + caption_h)
        aisles = MapBundle.load("DLP_aisles")
        m = 5.0
        self.polys = aisle_polygons(aisles, self.x0 - m, self.y0 - m, self.view_w + 2 * m, self.view_h + 2 * m)
        self.stalls = estimate_stalls(parked_sightings(), aisles.is_drivable)
        with open(os.path.join(DLP_DIR, f"{name}.meta.json")) as f:
            meta = json.load(f)
        self.obstacles = meta["obstacles"]
        self.agents = meta["agents"]
        self.font = self._font(13)
        self.small = self._font(11)
        self.label = self._font(9 * SS)

    @staticmethod
    def _font(size):
        for path in ("/usr/share/fonts/google-noto/NotoSans-Regular.ttf",
                     "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.isfile(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    def px(self, x, y):
        """World metres -> supersampled map pixels (y down)."""
        return ((x - self.x0) * self.ppm * SS, (self.view_h - (y - self.y0)) * self.ppm * SS)

    def box(self, x, y, heading, length, width):
        c, s = math.cos(heading), math.sin(heading)
        pts = [(x + c * dl - s * dw, y + s * dl + c * dw)
               for dl, dw in ((length / 2, width / 2), (length / 2, -width / 2),
                              (-length / 2, -width / 2), (-length / 2, width / 2))]
        return [self.px(*p) for p in pts]

    def draw(self, tick) -> Image.Image:
        W, H = int(self.map_w * SS), int(self.view_h * self.ppm * SS)
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        for poly, hole in sorted(self.polys, key=lambda p: p[1]):
            d.polygon([self.px(x, y) for x, y in poly], fill=BG if hole else AISLE)
        moving = [a for a in tick.agents if a.type == "vehicle"]
        for x, y, p in self.stalls:
            occupied = any(abs(o["x_m"] - x) < p / 2 and abs(o["y_m"] - y) < STALL_DEPTH_M / 2 for o in self.obstacles)
            occupied |= any(abs(a.x_m - x) < p / 2 and abs(a.y_m - y) < STALL_DEPTH_M / 2 for a in moving)
            if not occupied:
                d.polygon(self.box(x, y, math.pi / 2, STALL_DEPTH_M - 0.5, p - 0.45), fill=EMPTY)
        for o in self.obstacles:
            d.polygon(self.box(o["x_m"], o["y_m"], o["heading_rad"], o["length_m"], o["width_m"]), fill=PARKED)
        for a in tick.agents:
            if a.type == "vehicle":
                length, width = self.agents[str(a.id)]["size_m"]
                d.polygon(self.box(a.x_m, a.y_m, a.heading_rad, length, width), fill=MOVING, outline="#000000")
            else:
                cx, cy = self.px(a.x_m, a.y_m)
                r = 0.45 * self.ppm * SS
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=PED, outline="#000000")
        for a in tick.agents:                       # labels last, so nothing covers them
            cx, cy = self.px(a.x_m, a.y_m)
            tag = ("V" if a.type == "vehicle" else "P") + str(a.id)
            d.text((cx + 0.7 * self.ppm * SS, cy - 1.6 * self.ppm * SS), tag, fill="#ffffff", font=self.label,
                   stroke_width=2, stroke_fill="#000000")
        # scale bar: 10 m, bottom-left of the map
        bx, by = 12 * SS, H - 14 * SS
        d.rectangle((bx, by, bx + 10 * self.ppm * SS, by + 3 * SS), fill="#ffffff")
        d.text((bx, by - 16 * SS), "10 m", fill="#ffffff", font=self._font(10 * SS))
        img = img.resize((self.map_w, H // SS), Image.LANCZOS)

        out = Image.new("RGB", self.size, PANEL_BG)
        out.paste(img, (0, 0))
        c = ImageDraw.Draw(out)
        y = img.height + 8
        c.text((10, y), f"Scene: Dragon Lake Parking {self.name} (real drone data)   t = {tick.t_s:5.1f} s",
               fill=INK, font=self.font)
        x = 10
        for colour, text in ((MOVING, "tracked vehicle"), (PED, "pedestrian"), (PARKED, "parked (never moves)"),
                             (EMPTY, "empty space"), (AISLE, "aisle")):
            c.rectangle((x, y + 28, x + 11, y + 39), fill=colour)
            c.text((x + 16, y + 26), text, fill=INK, font=self.small)
            x += 16 + c.textlength(text, font=self.small) + 14
        c.text((10, y + 50), "Parking spaces estimated from parked-car positions in 16 recordings;",
               fill=MUTED, font=self.small)
        c.text((10, y + 66), "spaces no car ever used can be missing. Data: DLP, CC0.", fill=MUTED, font=self.small)
        c.text((10, y + 90), "Right: what the system predicts on the same seconds.", fill=INK, font=self.small)
        return out


# -- main ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dlp", default="DJI_0003")
    ap.add_argument("--start", type=float, default=51.2)
    ap.add_argument("--seconds", type=float, default=18.0)
    ap.add_argument("--center", type=float, nargs=2, default=(27.3, 31.0))
    ap.add_argument("--view-m", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=float, default=0.65)
    ap.add_argument("--out", default=os.path.join(ROOT, "demo", "dlp_side_by_side.gif"))
    ap.add_argument("--still-at", type=float, default=None, help="only save one PNG, this many seconds into the clip")
    args = ap.parse_args()

    rd = load_render_demo()
    rd.init_headless()
    import pygame
    pipe, bundle, ticks, status = rd.dlp_setup(args)
    renderer = rd.DashboardRenderer(bundle)
    surf = pygame.Surface((rd.CANVAS_W, rd.CANVAS_H))
    right_size = (int(rd.CANVAS_W * args.scale) // 2 * 2, int(rd.CANVAS_H * args.scale) // 2 * 2)
    map_px = (int(800 * args.scale), int(600 * args.scale))
    left = SemanticView(args.dlp, args.center[0], args.center[1], args.view_m, map_px,
                        right_size[1] - map_px[1])

    t0, every = args.start, max(1, round(60 / args.fps))
    with tempfile.TemporaryDirectory() as tmp:
        n = 0
        for i, tick in enumerate(ticks):
            pipe.step_sync(tick)
            if tick.t_s < t0 or i % every:
                continue
            if tick.t_s > t0 + args.seconds:
                break
            if args.still_at is not None and abs(tick.t_s - (t0 + args.still_at)) >= 1 / args.fps:
                continue
            renderer.draw(surf, pipe, status)
            right = Image.frombytes("RGB", (rd.CANVAS_W, rd.CANVAS_H), pygame.image.tostring(surf, "RGB"))
            right = right.resize(right_size, Image.LANCZOS)
            frame = Image.new("RGB", (left.size[0] + 6 + right_size[0], right_size[1]), PANEL_BG)
            frame.paste(left.draw(tick), (0, 0))
            frame.paste(right, (left.size[0] + 6, 0))
            if args.still_at is not None:
                frame.save(args.out)
                print(f"[side-by-side] still at {tick.t_s:.2f} s -> {args.out}")
                return
            frame.save(os.path.join(tmp, f"f{n:05d}.png"))
            n += 1
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        pattern, pal = os.path.join(tmp, "f%05d.png"), os.path.join(tmp, "palette.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern,
                        "-vf", "palettegen=max_colors=128:stats_mode=diff", pal], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", pattern, "-i", pal,
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                        "-loop", "0", args.out], check=True)
    print(f"[side-by-side] {n} frames -> {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
