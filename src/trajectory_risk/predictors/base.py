"""The contract every predictor satisfies, so the pipeline, the risk code and
the dashboard never need to know which model produced a prediction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol

import numpy as np


@dataclass
class Candidate:
    trajectory: np.ndarray  # (pred_horizon, 2) world metres; point i is at anchor + (i+1)*step
    confidence: float       # share of raw samples this candidate represents (NOT calibrated)


@dataclass
class PredictionRequest:
    agent_id: int
    agent_type: str
    sample_index: int        # decimated-grid index of the newest sample
    anchor_time_s: float     # simulation time of the newest sample
    step_dt_s: float         # model cadence (0.5 s ContextVAE, 0.4 s SocialVAE)
    positions: np.ndarray    # (buffer_len, 2) world metres, oldest first
    headings: np.ndarray     # (buffer_len,) radians
    neighbours: List[np.ndarray] = field(default_factory=list)  # each (buffer_len, 2), NaN = absent


@dataclass
class PredictionResult:
    agent_id: int
    agent_type: str
    sample_index: int
    anchor_time_s: float
    step_dt_s: float
    candidates: List[Candidate]
    model: str
    runtime_ms: float = 0.0

    @property
    def top(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


class Predictor(Protocol):
    name: str
    step_dt_s: float
    ob_horizon: int
    pred_horizon: int

    @property
    def buffer_len(self) -> int: ...

    def predict(self, request: PredictionRequest) -> List[Candidate]: ...


def sort_candidates(cands: List[Candidate]) -> List[Candidate]:
    return sorted(cands, key=lambda c: c.confidence, reverse=True)
