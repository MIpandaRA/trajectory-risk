"""Constant-velocity baseline: the reference every learned model must beat."""
from __future__ import annotations

from typing import List

import numpy as np

from .base import Candidate, PredictionRequest


class ConstantVelocity:
    """Extrapolates the last observed step. One candidate, confidence 1.0.

    ``velocity_window`` > 1 averages the last few steps, which is more robust
    to tracking noise (the robustness experiment uses this)."""

    def __init__(self, step_dt_s: float, ob_horizon: int, pred_horizon: int,
                 velocity_window: int = 1, name: str = "constant_velocity"):
        self.name = name
        self.step_dt_s = step_dt_s
        self.ob_horizon = ob_horizon
        self.pred_horizon = pred_horizon
        self.velocity_window = velocity_window

    @property
    def buffer_len(self) -> int:
        return self.ob_horizon + 2

    def predict(self, request: PredictionRequest) -> List[Candidate]:
        p = np.asarray(request.positions, dtype=np.float64)
        w = min(self.velocity_window, len(p) - 1)
        v = (p[-1] - p[-1 - w]) / w  # metres per step
        steps = np.arange(1, self.pred_horizon + 1)[:, None]
        return [Candidate(p[-1] + steps * v, 1.0)]
