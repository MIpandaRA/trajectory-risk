"""Dragon Lake Parking (DLP) recordings -> the tick protocol.

DLP (Shen, Pan, Govindarajan, Borrelli; CC0; https://zenodo.org/records/10084683)
is drone footage of a 140 x 80 m parking lot, tracked at 25 Hz. One recording
is five JSON files (``DJI_0001_{scene,agents,frames,instances,obstacles}.json``);
docs/data_dlp.md has the measured facts this module relies on.

What has to change on the way to the protocol, and why:

* **Rate.** DLP frames are 0.04 s apart. ``tracking.Decimator`` takes a sample
  when a tick lands within half a 60 Hz tick (8.3 ms) of k x 0.5 s or k x 0.4 s.
  25 Hz frames hit 0.5 s grid points only every 1 s, so a vehicle would get
  half its history samples and silently never fill its window. Every track is
  therefore resampled onto the exact 60 Hz grid t = k / 60.
* **Heading** is the BODY yaw in [-pi, pi] (cars reverse out of stalls, so it
  is not the direction of travel). Interpolating it raw would swing through 0
  when it wraps at +/-pi, so it is unwrapped first.
* **Types.** DLP labels are mapped to "vehicle" / "pedestrian"; cyclists are
  dropped by default (``CYCLIST_POLICIES``); unknown labels raise rather than
  being guessed, for the same reason the protocol rejects unknown types.
* **Parked cars.** Vehicles that never move are listed by DLP as obstacles,
  not agents. They never enter the tick stream (they would be ~160 extra
  prediction targets and false-alarm risk subjects next to every pedestrian
  walking between rows); they are kept in the conversion metadata, where
  ``predictors.static_neighbours`` can feed them to ContextVAE as neighbours.
  Agents that move at some point are kept for their whole track, including
  minutes spent stopped, because a live tracker would see them too.

Coordinates are already world metres, x right, y up, origin at the lot's
bottom-left corner, so ``dlp_to_world`` is the identity for DLP; it exists so
the conversion is stated rather than assumed.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..protocol import AgentState, Tick

SOURCE = "Dragon Lake Parking dataset, Shen, Pan, Govindarajan, Borrelli (UC Berkeley), CC0, https://zenodo.org/records/10084683"
PARTS = ("scene", "agents", "frames", "instances", "obstacles")

# DLP local frame -> world. Measured in docs/data_dlp.md: metres, y up.
DLP_METRES_PER_UNIT = 1.0
DLP_Y_UP = True
LOT_WIDTH_M, LOT_HEIGHT_M = 140.0, 80.0
# The area the map covers, in the same frame. Larger than the lot because
# vehicles and pedestrians are tracked outside it (the approach road and
# pavements): over DJI_0001..0016 agents reach x -19.5..148.4, y -35.1..80.5.
# Conversion counts samples outside it so a new recording that goes further
# is noticed rather than silently driving off the map.
MAPPED_X_M = (-25.0, 155.0)
MAPPED_Y_M = (-40.0, 85.0)

VEHICLE_LABELS = frozenset({"Car", "Medium Vehicle", "Bus", "Truck", "Motorcycle"})
PEDESTRIAN_LABELS = frozenset({"Pedestrian"})
CYCLIST_LABELS = frozenset({"Bicycle"})
IGNORED_LABELS = frozenset({"Undefined"})
CYCLIST_POLICIES = ("drop", "pedestrian", "vehicle")

TICK_RATE_HZ = 60.0
VELOCITY_WINDOW_S = 0.2      # +/- window for finite-difference velocity (5 DLP frames each side)
REVERSING_MIN_SPEED = 0.2    # m/s; below this the sign of motion is tracking noise
HEADING_MIN_SPEED = 0.3      # m/s; below this a derived heading is held, not recomputed
FOV_EDGE_MARGIN_M = 5.0      # a track starting/ending this close to the drone's view edge is censored


# -- raw data ------------------------------------------------------------
@dataclass
class RawTrack:
    token: str
    label: str
    size_m: Tuple[float, float]   # (length, width)
    t: np.ndarray                 # (n,) seconds, increasing
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray           # radians, NaN where the source has none
    speed: np.ndarray             # m/s, NaN where the source has none


@dataclass
class Obstacle:
    label: str
    x: float
    y: float
    heading: float
    length_m: float
    width_m: float


@dataclass
class RawRecording:
    name: str
    frame_times: np.ndarray       # every frame's timestamp, sorted
    tracks: List[RawTrack]
    obstacles: List[Obstacle]


def dlp_to_world(x, y, heading, metres_per_unit: float = DLP_METRES_PER_UNIT, y_up: bool = DLP_Y_UP,
                 height_units: float = LOT_HEIGHT_M):
    """DLP local coordinates -> world metres, y up, heading counter-clockwise
    from +x. With a y-down source the y axis is mirrored, which also mirrors
    every angle (heading -> -heading)."""
    x = np.asarray(x, dtype=np.float64) * metres_per_unit
    y = np.asarray(y, dtype=np.float64)
    h = np.asarray(heading, dtype=np.float64)
    if not y_up:
        y = height_units - y
        h = -h
    return x, y * metres_per_unit, h


def _num(v) -> float:
    return float("nan") if v is None or v == "" else float(v)


def load_raw(raw_dir: str, name: str) -> RawRecording:
    """Read one recording's JSON files into per-agent arrays sorted by time."""
    def read(part):
        with open(os.path.join(raw_dir, f"{name}_{part}.json")) as f:
            return json.load(f)
    agents, frames, instances, obstacles = (read(p) for p in ("agents", "frames", "instances", "obstacles"))
    frame_t = {tok: float(f["timestamp"]) for tok, f in frames.items()}
    rows: Dict[str, list] = defaultdict(list)
    for inst in instances.values():
        x, y = inst["coords"]
        rows[inst["agent_token"]].append((frame_t[inst["frame_token"]], x, y,
                                          _num(inst.get("heading")), _num(inst.get("speed"))))
    tracks = []
    for tok, r in rows.items():
        a = np.array(sorted(r), dtype=np.float64)
        x, y, h = dlp_to_world(a[:, 1], a[:, 2], a[:, 3])
        ag = agents[tok]
        tracks.append(RawTrack(tok, ag["type"], tuple(float(s) for s in ag["size"]), a[:, 0], x, y, h,
                               a[:, 4] * DLP_METRES_PER_UNIT))
    obs = []
    for o in obstacles.values():
        x, y, h = dlp_to_world(o["coords"][0], o["coords"][1], o["heading"])
        obs.append(Obstacle(o["type"], float(x), float(y), float(h),
                            float(o["size"][0]) * DLP_METRES_PER_UNIT, float(o["size"][1]) * DLP_METRES_PER_UNIT))
    return RawRecording(name, np.array(sorted(frame_t.values())), tracks, obs)


# -- small pure functions ------------------------------------------------
def protocol_type(label: str, cyclists: str = "drop") -> Optional[str]:
    """DLP label -> "vehicle" / "pedestrian", or None to leave the agent out."""
    if cyclists not in CYCLIST_POLICIES:
        raise ValueError(f"cyclists must be one of {CYCLIST_POLICIES}, got {cyclists!r}")
    if label in VEHICLE_LABELS:
        return "vehicle"
    if label in PEDESTRIAN_LABELS:
        return "pedestrian"
    if label in CYCLIST_LABELS:
        return None if cyclists == "drop" else cyclists
    if label in IGNORED_LABELS:
        return None
    raise ValueError(f"unknown DLP agent label {label!r}: add it to a label set in datasets/dlp.py")


def wrap_angle(a):
    """-> [-pi, pi)."""
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def grid_indices(t0: float, t1: float, rate_hz: float = TICK_RATE_HZ) -> np.ndarray:
    """Every k with k / rate_hz inside [t0, t1]. The epsilon keeps a source
    time that is a float hair away from a grid point (0.04 * 25 = 1.0000000000000002)
    from losing that point."""
    k0 = math.ceil(t0 * rate_hz - 1e-6)
    k1 = math.floor(t1 * rate_hz + 1e-6)
    return np.arange(k0, k1 + 1, dtype=np.int64)


def resample(t: np.ndarray, rate_hz: float = TICK_RATE_HZ, **series) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Linear interpolation of each named series onto t = k / rate_hz.
    A series called ``heading`` is unwrapped first and wrapped back after, so
    3.1 -> -3.1 passes through pi, not through 0."""
    k = grid_indices(float(t[0]), float(t[-1]), rate_hz)
    tg = np.clip(k / rate_hz, t[0], t[-1])
    out = {}
    for name, v in series.items():
        v = np.asarray(v, dtype=np.float64)
        if name == "heading":
            out[name] = wrap_angle(np.interp(tg, t, np.unwrap(v)))
        else:
            out[name] = np.interp(tg, t, v)
    return k, out


def central_velocity(t: np.ndarray, x: np.ndarray, y: np.ndarray, window_s: float = VELOCITY_WINDOW_S):
    """(vx, vy) per sample from positions +/- window_s apart (one-sided at the
    ends). A window rather than neighbouring frames because DLP positions are
    quantised to 1 cm: at 25 Hz a walking pedestrian moves ~4 cm per frame."""
    lo = np.searchsorted(t, t - window_s, side="left")
    hi = np.searchsorted(t, t + window_s, side="right") - 1
    dt = t[hi] - t[lo]
    safe = np.where(dt > 0, dt, 1.0)
    vx = np.where(dt > 0, (x[hi] - x[lo]) / safe, 0.0)
    vy = np.where(dt > 0, (y[hi] - y[lo]) / safe, 0.0)
    return vx, vy


def heading_from_velocity(t, x, y, min_speed: float = HEADING_MIN_SPEED, window_s: float = VELOCITY_WINDOW_S):
    """Heading = direction of travel, held at its last value while slower than
    ``min_speed`` (a standing pedestrian's jitter would otherwise spin it).
    Samples before the first fast one take the first valid heading; a track
    that never moves faces +x."""
    vx, vy = central_velocity(t, x, y, window_s)
    fast = np.hypot(vx, vy) >= min_speed
    h = np.full(len(t), np.nan)
    h[fast] = np.arctan2(vy[fast], vx[fast])
    if not fast.any():
        return np.zeros(len(t))
    last = h[np.argmax(fast)]
    for i in range(len(h)):
        if np.isnan(h[i]):
            h[i] = last
        else:
            last = h[i]
    return h


def reversing_mask(t, x, y, heading, min_speed: float = REVERSING_MIN_SPEED, window_s: float = VELOCITY_WINDOW_S):
    """True where the velocity points backwards relative to the body heading."""
    vx, vy = central_velocity(t, x, y, window_s)
    moving = np.hypot(vx, vy) > min_speed
    return moving & (vx * np.cos(heading) + vy * np.sin(heading) < 0)


def mask_to_intervals(t: np.ndarray, mask: np.ndarray) -> List[List[float]]:
    """Runs of True -> [[t_start, t_end], ...] (inclusive source times)."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            end = i if m else i - 1
            out.append([round(float(t[start]), 3), round(float(t[end]), 3)])
            start = None
    return out


def in_intervals(t, intervals) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    out = np.zeros(t.shape, dtype=bool)
    for a, b in intervals:
        out |= (t >= a - 1e-9) & (t <= b + 1e-9)
    return out


# -- conversion ----------------------------------------------------------
@dataclass
class ConvertedTrack:
    id: int
    type: str
    label: str
    token: str
    k: np.ndarray                 # 60 Hz grid indices
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    speed: np.ndarray
    reversing: List[List[float]] = field(default_factory=list)  # source-time intervals


@dataclass
class Conversion:
    name: str
    rate_hz: float
    k_end: int                    # last tick index (ticks run 0..k_end)
    tracks: List[ConvertedTrack]
    meta: dict


def fov_x_range(rec: RawRecording) -> Tuple[float, float]:
    xs = np.concatenate([tr.x for tr in rec.tracks]) if rec.tracks else np.zeros(1)
    return float(xs.min()), float(xs.max())


def fov_y_range(rec: RawRecording) -> Tuple[float, float]:
    ys = np.concatenate([tr.y for tr in rec.tracks]) if rec.tracks else np.zeros(1)
    return float(ys.min()), float(ys.max())


def convert(rec: RawRecording, cyclists: str = "drop", rate_hz: float = TICK_RATE_HZ) -> Conversion:
    """Resample every kept track onto the tick grid and collect the metadata
    the evaluation needs (censoring, reversing, parked cars, counts)."""
    t_start, t_end = float(rec.frame_times[0]), float(rec.frame_times[-1])
    frame_dt = float(np.median(np.diff(rec.frame_times))) if len(rec.frame_times) > 1 else 1.0 / rate_hz
    fov = fov_x_range(rec)
    kept = sorted(((protocol_type(tr.label, cyclists), tr) for tr in rec.tracks), key=lambda p: (p[1].t[0], p[1].token))
    tracks, agents_meta, counts = [], {}, defaultdict(lambda: {"agents": 0, "as": None})
    next_id = 1
    for typ, tr in kept:
        c = counts[tr.label]
        c["agents"] += 1
        c["as"] = typ or "dropped"
        if typ is None:
            continue
        heading = tr.heading
        if np.isnan(heading).any():
            heading = heading_from_velocity(tr.t, tr.x, tr.y)
        speed = tr.speed
        if np.isnan(speed).any():
            speed = np.hypot(*central_velocity(tr.t, tr.x, tr.y))
        k, s = resample(tr.t, rate_hz, x=tr.x, y=tr.y, heading=heading, speed=speed)
        if len(k) == 0:  # shorter than one tick
            continue
        rev = mask_to_intervals(tr.t, reversing_mask(tr.t, tr.x, tr.y, heading)) if typ == "vehicle" else []
        tracks.append(ConvertedTrack(next_id, typ, tr.label, tr.token, k, s["x"], s["y"], s["heading"],
                                     s["speed"], rev))
        near_edge = lambda x: min(abs(x - fov[0]), abs(x - fov[1])) <= FOV_EDGE_MARGIN_M
        agents_meta[str(next_id)] = {
            "dlp_token": tr.token, "label": tr.label, "type": typ,
            "size_m": [round(v, 3) for v in tr.size_m],
            "t_first_s": round(float(tr.t[0]), 3), "t_last_s": round(float(tr.t[-1]), 3),
            # Censored: we did not see the real entry/exit, only the edge of the
            # drone's view or of the recording.
            "censored_start": bool(tr.t[0] <= t_start + 1.5 * frame_dt or near_edge(tr.x[0])),
            "censored_end": bool(tr.t[-1] >= t_end - 1.5 * frame_dt or near_edge(tr.x[-1])),
            "reversing_intervals_s": rev,
        }
        next_id += 1
    for o in rec.obstacles:
        counts[f"{o.label} (never moves)"]["agents"] += 1
        counts[f"{o.label} (never moves)"]["as"] = "excluded: parked obstacle"
    meta = {
        "source": SOURCE, "recording": rec.name, "tick_rate_hz": rate_hz, "cyclists": cyclists,
        "source_frame_dt_s": round(frame_dt, 6), "duration_s": round(t_end - t_start, 3),
        "fov_x_m": [round(v, 2) for v in fov],
        "fov_y_m": [round(v, 2) for v in fov_y_range(rec)],
        "samples_outside_mapped_area": int(sum(np.sum((tr.x < MAPPED_X_M[0]) | (tr.x > MAPPED_X_M[1]) |
                                                      (tr.y < MAPPED_Y_M[0]) | (tr.y > MAPPED_Y_M[1]))
                                               for typ, tr in kept if typ is not None)),
        "counts": dict(counts),
        "agents": agents_meta,
        "obstacles": [{"label": o.label, "x_m": round(o.x, 3), "y_m": round(o.y, 3), "heading_rad": round(o.heading, 5),
                       "length_m": round(o.length_m, 3), "width_m": round(o.width_m, 3)} for o in rec.obstacles],
    }
    k_end = int(grid_indices(t_start, t_end, rate_hz)[-1])
    return Conversion(rec.name, rate_hz, k_end, tracks, meta)


def iter_ticks(conv: Conversion) -> Iterator[Tick]:
    """Validated ticks 0..k_end, one per 1/rate s, agents sorted by id.
    Ticks with nobody in view are still emitted so time stays continuous."""
    by_start = sorted(conv.tracks, key=lambda tr: tr.k[0])
    active: List[ConvertedTrack] = []
    nxt = 0
    for k in range(conv.k_end + 1):
        while nxt < len(by_start) and by_start[nxt].k[0] <= k:
            active.append(by_start[nxt])
            nxt += 1
        active = [tr for tr in active if tr.k[-1] >= k]
        agents = []
        for tr in sorted(active, key=lambda tr: tr.id):
            i = k - tr.k[0]
            agents.append(AgentState(tr.id, tr.type, float(tr.x[i]), float(tr.y[i]),
                                     float(tr.heading[i]), float(tr.speed[i])))
        for a in agents:
            a.validate()
        tick = Tick(k, k / conv.rate_hz, agents)
        tick.validate()
        yield tick


def parked_cars_xy(meta: dict) -> np.ndarray:
    """(N, 2) centres of the never-moving vehicles, for StaticNeighbours."""
    pts = [(o["x_m"], o["y_m"]) for o in meta.get("obstacles", [])
           if o["label"] in VEHICLE_LABELS]
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def load_meta(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


__all__ = ["RawTrack", "Obstacle", "RawRecording", "ConvertedTrack", "Conversion", "load_raw", "dlp_to_world",
           "protocol_type", "resample", "grid_indices", "wrap_angle", "central_velocity", "heading_from_velocity",
           "reversing_mask", "mask_to_intervals", "in_intervals", "convert", "iter_ticks", "parked_cars_xy",
           "load_meta", "fov_x_range", "fov_y_range", "SOURCE"]
