"""Show a vehicle predictor fixed obstacles (parked cars) as neighbours.

ContextVAE was trained on nuScenes, whose neighbour lists include parked
vehicles. In the DLP conversion the never-moving cars are deliberately kept
out of the tick stream, so they can never become prediction targets or risk
subjects; this wrapper adds them back on the ONE path where they belong: as
extra rows in each request's neighbour list, standing still at their
position for the whole history window.

Off unless you wrap a predictor with it (the DLP ablation does). The pipeline,
the track stores and the risk code never see these points.
"""
from __future__ import annotations

import dataclasses
from typing import List, Optional

import numpy as np

from .base import Candidate, PredictionRequest


class StaticNeighbours:
    def __init__(self, inner, points_xy: np.ndarray, radius_m: Optional[float] = None):
        """radius_m: only points this close to the agent's newest position are
        added. Defaults to the inner model's ob_radius (ContextVAE ignores
        neighbours beyond it anyway) plus a car length, so no point the model
        would use is cut; the cut only keeps the neighbour tensor small."""
        self.inner = inner
        self.points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if radius_m is None:
            ob_radius = getattr(getattr(inner, "cfg", None), "ob_radius", 30.0)
            radius_m = ob_radius + 5.0
        self.radius_m = radius_m
        self.name = f"{inner.name}+parked"
        self.step_dt_s = inner.step_dt_s
        self.ob_horizon = inner.ob_horizon
        self.pred_horizon = inner.pred_horizon

    @property
    def buffer_len(self) -> int:
        return self.inner.buffer_len

    def with_static(self, request: PredictionRequest) -> PredictionRequest:
        """A copy of the request with the nearby points appended as neighbours."""
        if not len(self.points) or not len(request.positions):
            return request
        d = np.hypot(*(self.points - request.positions[-1]).T)
        n = len(request.positions)
        extra = [np.tile(p, (n, 1)) for p in self.points[d <= self.radius_m]]
        return dataclasses.replace(request, neighbours=list(request.neighbours) + extra)

    def predict(self, request: PredictionRequest) -> List[Candidate]:
        return self.inner.predict(self.with_static(request))


__all__ = ["StaticNeighbours"]
