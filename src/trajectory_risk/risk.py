"""From two sets of candidate futures to one collision-risk number, then to an alert.

1. Staleness. A held prediction was made at ``anchor_time_s``; point i is the
   prediction for anchor + (i+1) * step. By the time it is used the agent has
   moved on, so only points whose time is still in the future are kept, and the
   path is re-attached to the agent's CURRENT position.
2. Time alignment. Vehicle (0.5 s) and pedestrian (0.4 s) candidates are
   interpolated onto one shared 0.1 s grid over the next 3 s.
3. Closest point of approach (CPA): for each candidate pair, the minimum
   distance at the SAME time. Two paths that cross 5 s apart are safe; a pure
   geometric intersection test would flag them.
4. Risk = sum_ij conf_v_i * conf_p_j * 1[CPA_ij < safety buffer]. Optionally
   (``vehicle_min_speed_mps``, off by default) the CPA only counts instants
   where the vehicle candidate is moving faster than that speed, so the score
   estimates "close AND the vehicle is moving" instead of "close". Multiplying
   the confidences assumes the vehicle and pedestrian futures are
   independent, which is false when they react to each other. Confidences are
   sample shares, not calibrated probabilities, so this is a risk SCORE.
5. Alert policy: hysteresis on the score, so it neither flickers nor fires on
   a single noisy update.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

GRID_STEP_S = 0.1
GRID_HORIZON_S = 3.0

VEHICLE_HALF_WIDTH_M = 0.9   # ~1.8 m wide car
PEDESTRIAN_RADIUS_M = 0.3
DEFAULT_MARGIN_M = 1.0
DEFAULT_SAFETY_BUFFER_M = VEHICLE_HALF_WIDTH_M + PEDESTRIAN_RADIUS_M + DEFAULT_MARGIN_M  # 2.2 m


def future_points(trajectory: np.ndarray, anchor_time_s: float, step_dt_s: float,
                  now_s: float) -> Tuple[np.ndarray, np.ndarray]:
    """-> (times, points) for the trajectory points still ahead of ``now_s``."""
    traj = np.asarray(trajectory, dtype=np.float64)
    times = anchor_time_s + step_dt_s * np.arange(1, len(traj) + 1)
    keep = times > now_s
    return times[keep], traj[keep]


def resample(now_xy, now_s: float, anchor_time_s: float, step_dt_s: float,
             trajectory: np.ndarray, grid_t: np.ndarray) -> np.ndarray:
    """Current position followed by the still-future points, linearly
    interpolated onto ``grid_t``. Past the last point the agent is held still
    (np.interp clamps), a conservative 'assume it stops' fallback."""
    t, pts = future_points(trajectory, anchor_time_s, step_dt_s, now_s)
    times = np.concatenate([[now_s], t])
    xs = np.concatenate([[now_xy[0]], pts[:, 0]])
    ys = np.concatenate([[now_xy[1]], pts[:, 1]])
    return np.stack([np.interp(grid_t, times, xs), np.interp(grid_t, times, ys)], axis=-1)


def closest_approach(a: np.ndarray, b: np.ndarray) -> Tuple[float, int]:
    """Two (T, 2) time-aligned paths -> (minimum distance, grid index where it happens)."""
    d = np.linalg.norm(np.asarray(a) - np.asarray(b), axis=-1)
    i = int(np.argmin(d))
    return float(d[i]), i


def cpa_constant_velocity(p1, v1, p2, v2, horizon_s: float) -> Tuple[float, float]:
    """Analytic CPA for two straight-line movers -> (distance, time), time clamped to [0, horizon]."""
    dp = np.asarray(p2, float) - np.asarray(p1, float)
    dv = np.asarray(v2, float) - np.asarray(v1, float)
    vv = float(dv @ dv)
    t = 0.0 if vv < 1e-12 else float(np.clip(-(dp @ dv) / vv, 0.0, horizon_s))
    return float(np.linalg.norm(dp + dv * t)), t


@dataclass
class HeldPrediction:
    """What the risk code needs from a track: where the agent is now and its
    latest prediction."""
    now_xy: Tuple[float, float]
    anchor_time_s: float
    step_dt_s: float
    candidates: Sequence  # objects with .trajectory and .confidence


def candidate_paths(h: HeldPrediction, now_s: float, grid_t: np.ndarray):
    pos = np.stack([resample(h.now_xy, now_s, h.anchor_time_s, h.step_dt_s, c.trajectory, grid_t)
                    for c in h.candidates])
    conf = np.array([c.confidence for c in h.candidates], dtype=np.float64)
    return pos, conf


def candidate_cpas(veh: HeldPrediction, ped: HeldPrediction, now_s: float,
                   moving_thresholds_mps: Sequence[float] = (), horizon_s: float = GRID_HORIZON_S,
                   step_s: float = GRID_STEP_S):
    """-> (vehicle confidences (Kv,), pedestrian confidences (Kp,), CPA (Kv, Kp),
    {threshold: CPA over the instants where the vehicle candidate is faster
    than threshold (Kv, Kp), inf if it never is}). Every risk score is a
    confidence-weighted count over one of these matrices."""
    grid_t = now_s + np.arange(0.0, horizon_s, step_s)
    vp, vc = candidate_paths(veh, now_s, grid_t)
    pp, pc = candidate_paths(ped, now_s, grid_t)
    d = np.linalg.norm(vp[:, None] - pp[None, :], axis=-1)                   # Kv x Kp x T
    moving = {}
    if len(moving_thresholds_mps):
        speed = np.linalg.norm(np.gradient(vp, step_s, axis=1), axis=-1)      # Kv x T
        for thr in moving_thresholds_mps:
            moving[thr] = np.where((speed > thr)[:, None, :], d, np.inf).min(axis=-1)
    return vc, pc, d.min(axis=-1), moving


def pairwise_risk(veh: HeldPrediction, ped: HeldPrediction, now_s: float,
                  safety_buffer_m: float = DEFAULT_SAFETY_BUFFER_M,
                  horizon_s: float = GRID_HORIZON_S, step_s: float = GRID_STEP_S,
                  vehicle_min_speed_mps: Optional[float] = None) -> Tuple[float, float]:
    """-> (risk score in [0, 1], smallest CPA over all candidate pairs in metres).

    ``vehicle_min_speed_mps``: if set, a candidate pair counts only if the
    candidates come within the buffer at an instant where the VEHICLE
    candidate moves faster than this. Off by default (the original score)."""
    if not veh.candidates or not ped.candidates:
        return 0.0, float("inf")
    thr = () if vehicle_min_speed_mps is None else (vehicle_min_speed_mps,)
    vc, pc, d_min, moving = candidate_cpas(veh, ped, now_s, thr, horizon_s, step_s)
    d_count = d_min if vehicle_min_speed_mps is None else moving[vehicle_min_speed_mps]
    risk = float(np.sum(vc[:, None] * pc[None, :] * (d_count < safety_buffer_m)))
    return min(risk, 1.0), float(d_min.min())


class HysteresisAlert:
    """Fires after ``enter_ticks`` consecutive updates at or above
    ``enter_threshold``; clears after ``clear_ticks`` consecutive updates below
    ``clear_threshold``. Values in the dead zone between the thresholds reset
    both streaks."""

    def __init__(self, enter_threshold: float = 0.15, enter_ticks: int = 3,
                 clear_threshold: float = 0.05, clear_ticks: int = 5):
        if clear_threshold > enter_threshold:
            raise ValueError("clear_threshold must be <= enter_threshold")
        self.enter_threshold, self.enter_ticks = enter_threshold, enter_ticks
        self.clear_threshold, self.clear_ticks = clear_threshold, clear_ticks
        self.active = False
        self._above = self._below = 0

    def update(self, risk: float) -> bool:
        if risk >= self.enter_threshold:
            self._above, self._below = self._above + 1, 0
        elif risk < self.clear_threshold:
            self._below, self._above = self._below + 1, 0
        else:
            self._above = self._below = 0
        if not self.active and self._above >= self.enter_ticks:
            self.active = True
        elif self.active and self._below >= self.clear_ticks:
            self.active = False
        return self.active


def bayes_alert_threshold(cost_false_alarm: float, cost_miss: float) -> float:
    """Alert when P(danger) > C_fa / (C_fa + C_miss): the action with lower
    expected cost. Only meaningful for a CALIBRATED probability; the risk score
    here is not, so the operating threshold is chosen empirically instead
    (see eval/risk_eval.py). Kept to show the reasoning."""
    return cost_false_alarm / (cost_false_alarm + cost_miss)


@dataclass
class RiskRow:
    veh_id: int
    ped_id: int
    risk: float
    d_min: float
    active: bool


def score_all(vehicles: dict, pedestrians: dict, now_s: float, alerts: dict,
              alert_factory, safety_buffer_m: float = DEFAULT_SAFETY_BUFFER_M,
              vehicle_min_speed_mps: Optional[float] = None) -> List[RiskRow]:
    """vehicles / pedestrians: {id: HeldPrediction}. ``alerts`` holds one
    HysteresisAlert per pair and is pruned to pairs that still exist."""
    rows, live = [], set()
    for vid, v in vehicles.items():
        for pid, p in pedestrians.items():
            risk, d_min = pairwise_risk(v, p, now_s, safety_buffer_m,
                                        vehicle_min_speed_mps=vehicle_min_speed_mps)
            key = (vid, pid)
            live.add(key)
            alert = alerts.get(key)
            if alert is None:
                alert = alerts[key] = alert_factory()
            rows.append(RiskRow(vid, pid, risk, d_min, alert.update(risk)))
    for key in [k for k in alerts if k not in live]:
        del alerts[key]
    rows.sort(key=lambda r: r.risk, reverse=True)
    return rows
