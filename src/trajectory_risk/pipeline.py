"""The consumer's logic with no GUI and no threads:

    tick -> per-type track stores -> due prediction requests
    prediction results -> held predictions -> pairwise risk -> alerts

The dashboard drives this with background inference workers; the offline
evaluation and the demo renderer drive it synchronously. Same code either
way, so what is measured offline is what runs live.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from .predictors.base import PredictionRequest, PredictionResult, Predictor
from .protocol import Tick
from .risk import DEFAULT_SAFETY_BUFFER_M, HeldPrediction, HysteresisAlert, RiskRow, score_all
from .tracking import TrackStore


@dataclass
class PipelineConfig:
    predict_every_vehicle: int = 2     # decimated samples between requests (2 x 0.5 s = 1 Hz)
    predict_every_pedestrian: int = 2  # 2 x 0.4 s = 1.25 Hz
    risk_interval_s: float = 0.2       # 5 Hz
    safety_buffer_m: float = DEFAULT_SAFETY_BUFFER_M
    # Alert policy chosen from the threshold sweep in docs/REPORT.md section 6
    # (the prototype used 0.15 / 3 updates: 3.5 false alarms per minute).
    enter_threshold: float = 0.6
    enter_ticks: int = 1
    clear_threshold: float = 0.2
    clear_ticks: int = 5
    stale_after_s: float = 1.0
    tick_rate_hz: float = 60.0
    # Motion-aware risk (risk.pairwise_risk): count only instants where the
    # vehicle candidate moves faster than this. None = the original score.
    risk_vehicle_min_speed_mps: Optional[float] = None


class LivePipeline:
    def __init__(self, vehicle_predictor: Predictor, pedestrian_predictor: Predictor,
                 config: Optional[PipelineConfig] = None):
        self.cfg = config or PipelineConfig()
        self.predictors = {"vehicle": vehicle_predictor, "pedestrian": pedestrian_predictor}
        self._build()

    def _build(self) -> None:
        c = self.cfg
        self.stores: Dict[str, TrackStore] = {
            "vehicle": TrackStore("vehicle", self.predictors["vehicle"].step_dt_s,
                                  self.predictors["vehicle"].buffer_len, c.predict_every_vehicle,
                                  c.stale_after_s, c.tick_rate_hz),
            "pedestrian": TrackStore("pedestrian", self.predictors["pedestrian"].step_dt_s,
                                     self.predictors["pedestrian"].buffer_len, c.predict_every_pedestrian,
                                     c.stale_after_s, c.tick_rate_hz),
        }
        self.alerts: Dict[tuple, HysteresisAlert] = {}
        self.risk_rows: List[RiskRow] = []
        self.now_s = float("-inf")
        self._last_risk_s = float("-inf")
        self.resets = 0

    def reset(self) -> None:
        self._build()
        self.resets += 1

    # -- ingest ------------------------------------------------------------
    def ingest(self, tick: Tick) -> List[PredictionRequest]:
        if tick.t_s < self.now_s - 1e-9:
            self.reset()  # time went backwards: a replay restarted
        self.now_s = tick.t_s
        requests: List[PredictionRequest] = []
        for store in self.stores.values():
            store.ingest(tick)
            requests.extend(store.due_requests())
        return requests

    def accept(self, result: PredictionResult) -> bool:
        return self.stores[result.agent_type].accept(result)

    def predict_now(self, request: PredictionRequest) -> PredictionResult:
        """Synchronous inference (offline evaluation, demo rendering)."""
        pred = self.predictors[request.agent_type]
        t0 = time.perf_counter()
        cands = pred.predict(request)
        return PredictionResult(request.agent_id, request.agent_type, request.sample_index,
                                request.anchor_time_s, request.step_dt_s, cands, pred.name,
                                (time.perf_counter() - t0) * 1000.0)

    def step_sync(self, tick: Tick) -> List[PredictionResult]:
        """ingest + predict every due request inline + update risk."""
        results = [self.predict_now(r) for r in self.ingest(tick)]
        for r in results:
            self.accept(r)
        self.update_risk()
        return results

    # -- risk --------------------------------------------------------------
    def held(self, agent_type: str) -> Dict[int, HeldPrediction]:
        out = {}
        for aid, tr in self.stores[agent_type].tracks.items():
            if tr.prediction is None or tr.last_state is None:
                continue
            out[aid] = HeldPrediction((tr.last_state.x_m, tr.last_state.y_m), tr.prediction.anchor_time_s,
                                      tr.prediction.step_dt_s, tr.prediction.candidates)
        return out

    def _new_alert(self) -> HysteresisAlert:
        c = self.cfg
        return HysteresisAlert(c.enter_threshold, c.enter_ticks, c.clear_threshold, c.clear_ticks)

    def update_risk(self, force: bool = False) -> Optional[List[RiskRow]]:
        if not force and self.now_s - self._last_risk_s < self.cfg.risk_interval_s - 1e-9:
            return None
        self._last_risk_s = self.now_s
        self.risk_rows = score_all(self.held("vehicle"), self.held("pedestrian"), self.now_s,
                                   self.alerts, self._new_alert, self.cfg.safety_buffer_m,
                                   self.cfg.risk_vehicle_min_speed_mps)
        return self.risk_rows
