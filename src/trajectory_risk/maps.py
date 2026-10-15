"""A map bundle is one folder under ``maps/``:

    map_meta.json      world extent (metres), pixels_per_metre, optional world origin
    waypoints.json     vehicle route graph, in SIMULATOR pixels (y down)
    semantic_map.pkl   (C x H x W float array in [-1, 1], H) where H maps world
                       (x, y, 1) -> raster (row, col). Channel 0 > 0 means drivable.
                       This is what ContextVAE's map encoder sees.
    basemap.png        picture for the dashboard background
    roboflow_export/   (optional) COCO polygons; used for the pedestrian walkable mask
"""
from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import numpy as np

from .geometry import ScreenFrame

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAPS_DIR = os.environ.get("TRAJ_RISK_MAPS", os.path.join(REPO_ROOT, "maps"))

SIM_CANVAS_W, SIM_CANVAS_H = 800, 600


@dataclass
class MapBundle:
    name: str
    root: str
    meta: dict = field(repr=False)

    @classmethod
    def load(cls, name: str, maps_dir: Optional[str] = None) -> "MapBundle":
        root = os.path.join(maps_dir or MAPS_DIR, name)
        with open(os.path.join(root, "map_meta.json")) as f:
            meta = json.load(f)
        return cls(name, root, meta)

    # -- paths -------------------------------------------------------------
    @property
    def waypoints_path(self) -> str:
        return os.path.join(self.root, "waypoints.json")

    @property
    def semantic_map_path(self) -> str:
        return os.path.join(self.root, "semantic_map.pkl")

    @property
    def basemap_path(self) -> str:
        return os.path.join(self.root, "basemap.png")

    @property
    def roboflow_dir(self) -> str:
        return os.path.join(self.root, "roboflow_export")

    # -- extent ------------------------------------------------------------
    @property
    def x_min(self) -> float:
        return float(self.meta.get("world_origin_x", 0.0))

    @property
    def y_min(self) -> float:
        return float(self.meta.get("world_origin_y", 0.0))

    @property
    def width_m(self) -> float:
        return float(self.meta["world_width_m"])

    @property
    def height_m(self) -> float:
        return float(self.meta["world_height_m"])

    @property
    def screen_frame(self) -> ScreenFrame:
        """Simulator canvas (800 x 600 px) <-> world."""
        return ScreenFrame(float(self.meta["pixels_per_metre"]), SIM_CANVAS_H, self.x_min, self.y_min)

    # -- semantic raster ---------------------------------------------------
    @cached_property
    def semantic(self):
        """-> (semantic_map C x H x W float32, H 3x3 world->(row, col))."""
        with open(self.semantic_map_path, "rb") as f:
            sem, H = pickle.load(f)
        return np.asarray(sem, dtype=np.float32), np.asarray(H, dtype=np.float64)

    def world_to_raster(self, xy: np.ndarray) -> np.ndarray:
        """(..., 2) world metres -> (..., 2) raster (row, col), float."""
        _, H = self.semantic
        xy = np.asarray(xy, dtype=np.float64)
        r = H[0, 0] * xy[..., 0] + H[0, 1] * xy[..., 1] + H[0, 2]
        c = H[1, 0] * xy[..., 0] + H[1, 1] * xy[..., 1] + H[1, 2]
        return np.stack([r, c], axis=-1)

    def is_drivable(self, xy: np.ndarray) -> np.ndarray:
        """(..., 2) world points -> bool array. Outside the raster counts as not drivable."""
        sem, _ = self.semantic
        rc = np.floor(self.world_to_raster(xy)).astype(int)
        r, c = rc[..., 0], rc[..., 1]
        inside = (r >= 0) & (r < sem.shape[1]) & (c >= 0) & (c < sem.shape[2])
        out = np.zeros(r.shape, dtype=bool)
        out[inside] = sem[0, r[inside], c[inside]] > 0
        return out

    def view_homography(self, view_w: int, view_h: int) -> np.ndarray:
        """3x3 world (x, y, 1) -> dashboard view (row, col) for a view of
        view_w x view_h pixels showing the whole map extent."""
        sx = view_w / self.width_m
        sy = view_h / self.height_m
        return np.array([
            [0.0, -sy, sy * (self.y_min + self.height_m)],
            [sx, 0.0, -sx * self.x_min],
            [0.0, 0.0, 1.0],
        ])

    def render_basemap(self, view_w: int, view_h: int) -> np.ndarray:
        """A clean (view_h, view_w, 3) uint8 picture of the semantic raster:
        road, crossings and lane markings. Used as the dashboard background so
        the repo does not depend on third-party imagery."""
        import cv2
        sem, _ = self.semantic
        Hv = self.view_homography(view_w, view_h)
        cols, rows = np.meshgrid(np.arange(view_w) + 0.5, np.arange(view_h) + 0.5)
        A, b = Hv[:2, :2], Hv[:2, 2]
        xy = (np.linalg.inv(A) @ (np.stack([rows.ravel(), cols.ravel()]) - b[:, None])).T
        rc = self.world_to_raster(xy).reshape(view_h, view_w, 2).astype(np.float32)
        def sample(ch, interp=cv2.INTER_NEAREST):
            return cv2.remap(sem[ch], rc[..., 1] - 0.5, rc[..., 0] - 0.5, interp,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
        zoom = view_w / self.width_m  # view pixels per raster pixel (rasters are 1 px per metre)
        blur = lambda a: cv2.GaussianBlur(a.astype(np.float32), (0, 0), max(0.8, 0.35 * zoom))
        ch0 = sample(0)
        road = blur(ch0 > -0.75)[..., None]
        crossing = blur((ch0 > 0.75) | (np.abs(ch0 + 0.5) < 0.1))[..., None]
        lines = blur((sample(1) > 0) | (sample(2) > 0))[..., None] * min(1.0, 3.0 / zoom)
        img = np.full((view_h, view_w, 3), (34, 38, 44), dtype=np.float32)
        img = img * (1 - road) + np.array((72, 76, 84), np.float32) * road
        img = img * (1 - 0.55 * crossing) + np.array((200, 200, 205), np.float32) * 0.55 * crossing
        img = img * (1 - 0.35 * lines) + np.array((210, 190, 120), np.float32) * 0.35 * lines
        return np.clip(img, 0, 255).astype(np.uint8)

