"""Per-agent history buffers built from the raw ~60 Hz tick stream.

Each model needs its history at the cadence it was trained on (ContextVAE
0.5 s, SocialVAE 0.4 s). Each agent type therefore gets its own TrackStore
with its own decimation grid, built from the SAME raw stream; one model's
history is never resampled from the other's.

Decimation is by simulation time, not by tick count: sample index k is taken
from the tick whose t_s is within half a tick of k * period. The original
prototype used ``tick % 30 == 0``, which silently assumes the producer runs at
exactly 60 Hz.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from .predictors.base import PredictionRequest, PredictionResult
from .protocol import AgentState, Tick


@dataclass(frozen=True)
class Sample:
    index: int      # position on this store's decimation grid
    t_s: float
    x: float
    y: float
    heading: float


class Decimator:
    def __init__(self, period_s: float, tick_rate_hz: float = 60.0):
        self.period_s = period_s
        self.tolerance_s = 0.5 / tick_rate_hz
        self._last_index: Optional[int] = None

    def index_for(self, t_s: float) -> Optional[int]:
        """Grid index if this tick falls on the grid (and that index was not
        already taken by an earlier tick), else None."""
        k = int(round(t_s / self.period_s))
        if abs(t_s - k * self.period_s) > self.tolerance_s + 1e-9:
            return None
        if self._last_index is not None and k <= self._last_index:
            return None
        self._last_index = k
        return k

    def reset(self) -> None:
        self._last_index = None


class AgentTrack:
    def __init__(self, agent_id: int, agent_type: str, buffer_len: int):
        self.id = agent_id
        self.type = agent_type
        self.buffer_len = buffer_len
        self.samples: Deque[Sample] = deque(maxlen=buffer_len)
        self.last_state: Optional[AgentState] = None
        self.last_seen_t: float = float("-inf")
        self.last_requested_index: Optional[int] = None
        self.prediction: Optional[PredictionResult] = None

    def window(self) -> Optional[List[Sample]]:
        """The full buffer if it is full AND has no gaps, else None."""
        if len(self.samples) < self.buffer_len:
            return None
        items = list(self.samples)
        if any(b.index != a.index + 1 for a, b in zip(items, items[1:])):
            return None
        return items

    def speed_mps(self, period_s: float) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        a, b = self.samples[-2], self.samples[-1]
        dt = (b.index - a.index) * period_s
        return float(np.hypot(b.x - a.x, b.y - a.y) / dt) if dt > 0 else None


class TrackStore:
    """All tracks of one agent type, decimated for one model."""

    def __init__(self, agent_type: str, period_s: float, buffer_len: int,
                 predict_every: int = 2, stale_after_s: float = 1.0, tick_rate_hz: float = 60.0):
        self.agent_type = agent_type
        self.period_s = period_s
        self.buffer_len = buffer_len
        self.predict_every = predict_every
        self.stale_after_s = stale_after_s
        self.decimator = Decimator(period_s, tick_rate_hz)
        self.tracks: Dict[int, AgentTrack] = {}
        self.now_s: float = float("-inf")

    def ingest(self, tick: Tick) -> None:
        self.now_s = tick.t_s
        k = self.decimator.index_for(tick.t_s)
        for a in tick.agents:
            if a.type != self.agent_type:
                continue
            tr = self.tracks.get(a.id)
            if tr is None:
                tr = self.tracks[a.id] = AgentTrack(a.id, a.type, self.buffer_len)
            tr.last_state = a
            tr.last_seen_t = tick.t_s
            if k is not None:
                tr.samples.append(Sample(k, tick.t_s, a.x_m, a.y_m, a.heading_rad))
        # Agents that stopped appearing have left the scene. The prototype kept
        # them forever, so a despawned agent's last prediction kept being
        # scored for collision risk.
        for aid in [aid for aid, tr in self.tracks.items() if tick.t_s - tr.last_seen_t > self.stale_after_s]:
            del self.tracks[aid]

    def build_request(self, track: AgentTrack) -> Optional[PredictionRequest]:
        win = track.window()
        if win is None:
            return None
        indices = [s.index for s in win]
        neighbours = []
        for other in self.tracks.values():
            if other.id == track.id:
                continue
            by_index = {s.index: (s.x, s.y) for s in other.samples}
            if not any(i in by_index for i in indices):
                continue
            neighbours.append(np.array([by_index.get(i, (np.nan, np.nan)) for i in indices], dtype=np.float64))
        return PredictionRequest(
            agent_id=track.id, agent_type=track.type, sample_index=win[-1].index,
            anchor_time_s=win[-1].t_s, step_dt_s=self.period_s,
            positions=np.array([(s.x, s.y) for s in win], dtype=np.float64),
            headings=np.array([s.heading for s in win], dtype=np.float64),
            neighbours=neighbours)

    def due_requests(self) -> List[PredictionRequest]:
        """Requests for every track whose window is complete and whose last
        request is at least ``predict_every`` samples old."""
        out = []
        for tr in self.tracks.values():
            win = tr.window()
            if win is None:
                continue
            k = win[-1].index
            if tr.last_requested_index is not None and k - tr.last_requested_index < self.predict_every:
                continue
            req = self.build_request(tr)
            if req is not None:
                tr.last_requested_index = k
                out.append(req)
        return out

    def accept(self, result: PredictionResult) -> bool:
        tr = self.tracks.get(result.agent_id)
        if tr is None or not result.candidates:
            return False
        if tr.prediction is not None and tr.prediction.sample_index > result.sample_index:
            return False  # an older result finished after a newer one
        tr.prediction = result
        return True
