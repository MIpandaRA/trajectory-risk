"""Pure drawing: takes the pipeline's state and paints one frame onto a
pygame Surface. No sockets, no threads, no model calls, so the live dashboard
and the offline demo renderer draw identical frames."""
from __future__ import annotations

import math
from typing import Dict, Optional

import cv2
import numpy as np
import pygame
from PIL import Image

from ..maps import MapBundle
from ..risk import future_points

MAP_W, MAP_H = 800, 600
SIDE_W = 240
PANEL_H = 200
CANVAS_W, CANVAS_H = MAP_W + SIDE_W, MAP_H + PANEL_H

BG = (24, 24, 28)
PANEL_BG = (34, 34, 40)
SIDE_BG = (30, 24, 26)
TEXT = (225, 225, 230)
MUTED = (150, 150, 160)
VEH = (0, 200, 200)
PED = (235, 90, 205)
VEH_PRED = (255, 150, 30)
PED_PRED = (190, 120, 255)
ALERT = (240, 60, 60)
DRIVABLE = (60, 200, 90)
GOOD = (90, 220, 120)


def world_to_view(Hv: np.ndarray, x, y):
    r = Hv[0, 0] * x + Hv[0, 1] * y + Hv[0, 2]
    c = Hv[1, 0] * x + Hv[1, 1] * y + Hv[1, 2]
    return c, r  # pygame (x, y) = (col, row)


class DashboardRenderer:
    def __init__(self, bundle: MapBundle, fonts: Optional[dict] = None, basemap: str = "semantic"):
        """basemap: "semantic" draws the road raster the model sees (default,
        no third-party imagery); "photo" uses maps/<name>/basemap.png if present."""
        import os
        self.bundle = bundle
        self.Hv = bundle.view_homography(MAP_W, MAP_H)
        self.ppm_view = MAP_W / bundle.width_m
        if basemap == "photo" and os.path.isfile(bundle.basemap_path):
            img = np.array(Image.open(bundle.basemap_path).convert("RGB"))
            img = cv2.resize(img, (MAP_W, MAP_H), interpolation=cv2.INTER_AREA)
            self.overlay = self._drivable_overlay()
        else:
            img = bundle.render_basemap(MAP_W, MAP_H)
            self.overlay = None
        self.basemap = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        self.f = fonts or {
            "sm": pygame.font.Font(None, 18), "md": pygame.font.Font(None, 22),
            "lg": pygame.font.Font(None, 28), "label": pygame.font.Font(None, 16)}

    def _drivable_overlay(self) -> pygame.Surface:
        cols, rows = np.meshgrid(np.arange(MAP_W) + 0.5, np.arange(MAP_H) + 0.5)
        A, b = self.Hv[:2, :2], self.Hv[:2, 2]
        rc = np.stack([rows.ravel(), cols.ravel()])
        xy = np.linalg.inv(A) @ (rc - b[:, None])
        drv = self.bundle.is_drivable(xy.T).reshape(MAP_H, MAP_W).astype(np.float32)
        drv = cv2.GaussianBlur(drv, (13, 13), 3.0)
        rgba = np.zeros((MAP_H, MAP_W, 4), dtype=np.uint8)
        rgba[..., :3] = DRIVABLE
        rgba[..., 3] = (drv * 0.28 * 255).astype(np.uint8)
        return pygame.image.frombuffer(rgba.tobytes(), (MAP_W, MAP_H), "RGBA").copy()

    # ------------------------------------------------------------------
    def draw(self, surf: pygame.Surface, pipeline, status: Dict, show_alternates: bool = True,
             debug: bool = False) -> None:
        now = pipeline.now_s
        surf.fill(BG)
        surf.blit(self.basemap, (0, 0))
        if self.overlay is not None:
            surf.blit(self.overlay, (0, 0))

        rows = []
        for kind, colour, pcol in (("vehicle", VEH, VEH_PRED), ("pedestrian", PED, PED_PRED)):
            for aid, tr in sorted(pipeline.stores[kind].tracks.items()):
                if tr.last_state is None:
                    continue
                s = tr.last_state
                pred = tr.prediction
                if pred is not None and pred.candidates:
                    cands = pred.candidates if show_alternates else pred.candidates[:1]
                    for rank, c in reversed(list(enumerate(cands))):
                        _, pts = future_points(c.trajectory, pred.anchor_time_s, pred.step_dt_s, now)
                        if not len(pts):
                            continue
                        line = [world_to_view(self.Hv, s.x_m, s.y_m)] + [world_to_view(self.Hv, *p) for p in pts]
                        if rank == 0:
                            pygame.draw.lines(surf, pcol, False, line, 3)
                            pygame.draw.circle(surf, pcol, [int(v) for v in line[-1]], 4)
                        elif c.confidence >= 0.05:
                            faded = tuple(int(0.45 * a + 0.55 * b) for a, b in zip(pcol, (60, 60, 60)))
                            pygame.draw.lines(surf, faded, False, line, 1)
                self._marker(surf, kind, s, colour)
                rows.append((kind, s, pred, tr))

        for r in pipeline.risk_rows:
            if not r.active:
                continue
            v = pipeline.stores["vehicle"].tracks.get(r.veh_id)
            p = pipeline.stores["pedestrian"].tracks.get(r.ped_id)
            if v and p and v.last_state and p.last_state:
                a = world_to_view(self.Hv, v.last_state.x_m, v.last_state.y_m)
                b = world_to_view(self.Hv, p.last_state.x_m, p.last_state.y_m)
                pygame.draw.line(surf, ALERT, a, b, 3)
                pygame.draw.circle(surf, ALERT, [int(t) for t in b], 14, 2)

        self._side_panel(surf, pipeline, status)
        self._bottom_panel(surf, rows, pipeline, status)
        if debug:
            self._debug(surf, pipeline, status)

    def _marker(self, surf, kind, s, colour):
        cx, cy = world_to_view(self.Hv, s.x_m, s.y_m)
        label = self.f["label"].render(str(s.id), True, (10, 10, 10))
        if kind == "vehicle":
            w, h = max(16, 4.4 * self.ppm_view), max(10, 2.0 * self.ppm_view)
            body = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.rect(body, colour, (0, 0, w, h), border_radius=4)
            rot = pygame.transform.rotate(body, math.degrees(s.heading_rad))
            surf.blit(rot, rot.get_rect(center=(cx, cy)))
        else:
            pygame.draw.circle(surf, colour, (int(cx), int(cy)), 6)
        surf.blit(label, label.get_rect(center=(cx, cy)))

    def _side_panel(self, surf, pipeline, status: Optional[Dict] = None):
        """status may set "risk_label" (default "score"), "risk_note" (a line
        under the title) and "alert_threshold" (a tick on each risk bar)."""
        status = status or {}
        x0 = MAP_W
        pygame.draw.rect(surf, SIDE_BG, (x0, 0, SIDE_W, MAP_H))
        alerting = [r for r in pipeline.risk_rows if r.active]
        title = self.f["lg"].render("Collision risk", True, ALERT if alerting else TEXT)
        surf.blit(title, (x0 + 14, 14))
        sub = f"{len(alerting)} alert{'s' if len(alerting) != 1 else ''}   {len(pipeline.risk_rows)} pairs scored"
        surf.blit(self.f["sm"].render(sub, True, MUTED), (x0 + 14, 44))
        y = 72
        if status.get("risk_note"):
            surf.blit(self.f["sm"].render(status["risk_note"], True, MUTED), (x0 + 14, 60))
            y = 84
        thr = status.get("alert_threshold")
        label = status.get("risk_label", "score")
        shown = [r for r in pipeline.risk_rows if r.risk > 0.01 or r.active][:11]
        if not shown:
            surf.blit(self.f["md"].render("no risky pairs", True, MUTED), (x0 + 14, y))
        for r in shown:
            col = ALERT if r.active else TEXT
            tag = ""
            if r.active:
                # Display only: hysteresis keeps an alert on for a few updates after the
                # risk drops below the enter threshold; say so rather than look wrong.
                tag = "   ALERT (clearing)" if thr is not None and r.risk < thr else "   ALERT"
            surf.blit(self.f["md"].render(f"V{r.veh_id} x P{r.ped_id}" + tag, True, col), (x0 + 14, y))
            bar_w = int((SIDE_W - 28) * min(1.0, r.risk))
            pygame.draw.rect(surf, (70, 50, 55), (x0 + 14, y + 20, SIDE_W - 28, 6))
            pygame.draw.rect(surf, col, (x0 + 14, y + 20, bar_w, 6))
            if thr is not None:
                tx = x0 + 14 + int((SIDE_W - 28) * thr)
                pygame.draw.line(surf, TEXT, (tx, y + 17), (tx, y + 28), 1)
            cpa = f"{r.d_min:.1f} m" if math.isfinite(r.d_min) else "-"
            surf.blit(self.f["sm"].render(f"{label} {r.risk:.2f}   closest {cpa}", True, MUTED), (x0 + 14, y + 29))
            y += 47

    def _bottom_panel(self, surf, rows, pipeline, status):
        y0 = MAP_H
        pygame.draw.rect(surf, PANEL_BG, (0, y0, CANVAS_W, PANEL_H))
        head = (f"t = {pipeline.now_s:6.1f} s    vehicles {len(pipeline.stores['vehicle'].tracks)}    "
                f"pedestrians {len(pipeline.stores['pedestrian'].tracks)}    {status.get('source', '')}")
        surf.blit(self.f["md"].render(head, True, TEXT), (16, y0 + 10))
        cols = [16, 70, 150, 290, 400, 560, 660, 760]
        names = ["ID", "type", "position (m)", "speed", "predicted end (m)", "conf.", "pred. age", "model"]
        for x, n in zip(cols, names):
            surf.blit(self.f["sm"].render(n, True, MUTED), (x, y0 + 36))
        pygame.draw.line(surf, (70, 70, 80), (16, y0 + 52), (CANVAS_W - 16, y0 + 52))
        max_rows = 6
        for i, (kind, s, pred, tr) in enumerate(rows[:max_rows]):
            y = y0 + 58 + i * 22
            period = pipeline.stores[kind].period_s
            spd = tr.speed_mps(period)
            if pred is not None and pred.candidates:
                end = pred.candidates[0].trajectory[-1]
                vals = [f"({end[0]:.1f}, {end[1]:.1f})", f"{pred.candidates[0].confidence:.2f}",
                        f"{pipeline.now_s - pred.anchor_time_s:.1f} s", pred.model]
                colour = TEXT
            else:
                vals = ["warming up", "-", "-", "-"]
                colour = MUTED
            cells = [str(s.id), "veh" if kind == "vehicle" else "ped", f"({s.x_m:.1f}, {s.y_m:.1f})",
                     f"{spd * 3.6:.1f} km/h" if spd is not None else "-"] + vals
            for x, v in zip(cols, cells):
                surf.blit(self.f["sm"].render(v, True, colour), (x, y))
        if len(rows) > max_rows:
            surf.blit(self.f["sm"].render(f"+ {len(rows) - max_rows} more agents", True, MUTED),
                      (16, y0 + 58 + max_rows * 22))
        legend = [("vehicle", VEH), ("pedestrian", PED), ("vehicle path", VEH_PRED),
                  ("pedestrian path", PED_PRED), ("alert", ALERT)]
        x = CANVAS_W - 470
        for name, col in legend:
            pygame.draw.rect(surf, col, (x, y0 + 14, 12, 12))
            x += 16
            if name:
                t = self.f["sm"].render(name, True, MUTED)
                surf.blit(t, (x, y0 + 13))
                x += t.get_width() + 12

    def _debug(self, surf, pipeline, status):
        lines = [f"fps {status.get('fps', 0):.0f}"]
        for kind in ("vehicle", "pedestrian"):
            st = status.get(f"{kind}_stats")
            if st is None:
                continue
            ms = st.recent_ms()
            med = sorted(ms)[len(ms) // 2] if ms else 0
            lines.append(f"{kind}: {med:.0f} ms median, {st.completed} done, {st.superseded} superseded")
        h = 22 * len(lines) + 12
        box = pygame.Surface((370, h), pygame.SRCALPHA)
        box.fill((0, 0, 0, 170))
        for i, line in enumerate(lines):
            box.blit(self.f["sm"].render(line, True, GOOD), (10, 8 + 22 * i))
        surf.blit(box, (10, 10))


def init_headless() -> None:
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()


__all__ = ["DashboardRenderer", "CANVAS_W", "CANVAS_H", "init_headless"]
