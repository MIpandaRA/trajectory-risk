"""Finite-difference state, exactly as both upstream training pipelines build it.

Important unit detail: upstream ``data.py`` stores velocity as displacement
PER SAMPLE STEP (metres per 0.5 s for ContextVAE, per 0.4 s for SocialVAE),
not metres per second, and acceleration as change in that per-step velocity.
The models therefore only make sense at the cadence they were trained on;
feeding 60 Hz samples would make every agent look ~30x slower.
"""
from __future__ import annotations

import numpy as np

SENTINEL = 1e9  # upstream marker for "neighbour not present at this step"


def extend(positions: np.ndarray) -> np.ndarray:
    """(T, 2) positions, oldest first -> (T, 6) [x, y, vx, vy, ax, ay].

    Backward differences. Row 0 has no velocity and rows 0-1 no acceleration,
    which is why live buffers keep two extra warm-up samples (buffer_len =
    ob_horizon + 2) and only the last ob_horizon rows are fed to a model.
    """
    p = np.asarray(positions, dtype=np.float64)
    out = np.zeros((p.shape[0], 6), dtype=np.float32)
    out[:, :2] = p
    v = np.zeros_like(p)
    v[1:] = p[1:] - p[:-1]
    a = np.zeros_like(p)
    a[1:] = v[1:] - v[:-1]
    out[:, 2:4] = v
    out[:, 4:6] = a
    return out


def neighbour_states(neighbours, buffer_len: int, ob_horizon: int) -> np.ndarray:
    """List of (buffer_len, 2) arrays with NaN where the neighbour was absent
    -> (ob_horizon, Nn, 6), absent steps filled with the upstream sentinel.

    Mirrors the original live wrapper: differences are computed on the
    sentinel-filled positions and then every absent row is reset to the
    sentinel, so a neighbour that appears mid-window gets a (huge, harmless)
    velocity on its first present step, as in the original code.
    """
    rows = []
    for n in neighbours:
        n = np.asarray(n, dtype=np.float64)
        if n.shape != (buffer_len, 2):
            continue
        valid = ~np.isnan(n).any(axis=1)
        st = np.full((buffer_len, 6), SENTINEL, dtype=np.float32)
        if valid.any():
            ext = extend(np.where(np.isnan(n), SENTINEL, n))
            st[valid] = ext[valid]
        rows.append(st[-ob_horizon:])
    if not rows:
        return np.zeros((ob_horizon, 0, 6), dtype=np.float32)
    return np.stack(rows, axis=1)
