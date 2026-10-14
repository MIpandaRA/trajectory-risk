"""Every coordinate transform in the system lives here, and nowhere else.

Three frames:

* SCREEN: simulator canvas pixels, origin top-left, y DOWN.
* WORLD: metres, y UP, heading in radians counter-clockwise from +x. This is
  what goes on the wire and into both models.
* EGO: the frame ContextVAE was trained in. Origin at the agent's OLDEST
  observed position, rotated so the agent's heading at that moment points
  along +x.

The y-axis flips exactly once, in ``ScreenFrame.to_world`` (and its inverse).
A second flip anywhere else mirrors every prediction across the x-axis, which
looks plausible on screen and is very hard to spot by eye; the tests check it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rotation_matrix(angle: float) -> np.ndarray:
    """2x2 counter-clockwise rotation by ``angle`` radians."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_about(points: np.ndarray, pivot, angle: float) -> np.ndarray:
    """Rotate (..., 2) points counter-clockwise by ``angle`` around ``pivot``."""
    p = np.asarray(points, dtype=np.float64)
    return (p - pivot) @ rotation_matrix(angle).T + pivot


def rotate_state(state: np.ndarray, pivot, angle: float) -> np.ndarray:
    """Rotate (..., 6) [x, y, vx, vy, ax, ay] rows. Position rotates about the
    pivot; velocity and acceleration are directions, so they only rotate."""
    out = np.asarray(state, dtype=np.float64).copy()
    R = rotation_matrix(angle)
    out[..., 0:2] = (out[..., 0:2] - pivot) @ R.T + pivot
    out[..., 2:4] = out[..., 2:4] @ R.T
    out[..., 4:6] = out[..., 4:6] @ R.T
    return out


@dataclass(frozen=True)
class EgoFrame:
    """World <-> ego for one prediction.

    ContextVAE (following its training code) rotates about the pivot but keeps
    the pivot's world coordinates rather than translating it to the origin, so
    ``to_ego`` here is a rotation about the pivot by ``-heading``.
    """
    pivot: tuple
    heading: float

    @property
    def angle(self) -> float:
        return -self.heading

    def to_ego(self, points: np.ndarray) -> np.ndarray:
        return rotate_about(points, self.pivot, self.angle)

    def to_world(self, points: np.ndarray) -> np.ndarray:
        return rotate_about(points, self.pivot, -self.angle)


@dataclass(frozen=True)
class ScreenFrame:
    """Simulator canvas pixels <-> world metres."""
    pixels_per_metre: float
    canvas_height_px: int
    origin_x_m: float = 0.0
    origin_y_m: float = 0.0

    @property
    def canvas_height_m(self) -> float:
        return self.canvas_height_px / self.pixels_per_metre

    def to_world(self, x_px: float, y_px: float, heading_px: float = 0.0):
        x_m = self.origin_x_m + x_px / self.pixels_per_metre
        y_m = self.origin_y_m + self.canvas_height_m - y_px / self.pixels_per_metre
        return x_m, y_m, -heading_px

    def to_screen(self, x_m: float, y_m: float, heading_rad: float = 0.0):
        x_px = (x_m - self.origin_x_m) * self.pixels_per_metre
        y_px = (self.canvas_height_m - (y_m - self.origin_y_m)) * self.pixels_per_metre
        return x_px, y_px, -heading_rad


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi
