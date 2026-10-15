"""Headless simulation of vehicles and pedestrians on one map.

Fixed time step. The prototype advanced physics by the measured frame time
but stamped messages with ``t = tick / 60``; whenever a frame took longer than
1/60 s those two disagreed, so the timestamps lied about how far agents had
moved. Here every step is exactly ``dt`` and ``t_s = seq * dt``, true by
construction. The live producer keeps real time by running as many fixed
steps as wall time requires.

Seeded, so the same config always produces the same scenario.

Internally everything is in simulator pixels (y down), as in the original
sim code; conversion to world metres happens once, in ``tick()``.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..maps import SIM_CANVAS_H, SIM_CANVAS_W, MapBundle
from ..protocol import AgentState, Tick
from .pedestrian import Pedestrian, WalkableArea
from .vehicle import DEFAULT_SPEED, Vehicle
from .waypoints import WaypointGraph

SPAWN_MARGIN = 60     # px behind a route start where a vehicle appears
DESPAWN_MARGIN = 90   # px past the canvas edge where an agent is removed (> SPAWN_MARGIN)


@dataclass
class WorldConfig:
    map_name: str = "T11"
    num_vehicles: int = 0              # 0 = one per route
    num_pedestrians: int = 6
    vehicle_speed_px: float = DEFAULT_SPEED      # 25 px/s: 2.5 m/s on T11 (10 px/m), 12.5 m/s on OneNorth (2 px/m)
    pedestrian_speed_mps: float = 1.3            # metres per second, so it means the same on every map
    pedestrian_lifetime_s: float = 30.0
    routing: str = "deterministic"
    crossing_bias: float = 0.0         # probability a pedestrian's next target is across a road
    vehicle_stagger_s: float = 6.0     # route i's first vehicle enters at i * stagger. The prototype
                                       # spawned all at t=0; T11's routes share an entrance, so the
                                       # cars drove on top of each other.
    seed: int = 0
    dt: float = 1.0 / 60.0


def offscreen_point(x0, y0, heading, margin=SPAWN_MARGIN, w=SIM_CANVAS_W, h=SIM_CANVAS_H):
    """From (x0, y0) along ``heading``, ``margin`` px past the first canvas edge crossed."""
    dx, dy = math.cos(heading), math.sin(heading)
    if not (0 <= x0 <= w and 0 <= y0 <= h):
        cx = -margin if x0 < 0 else (w + margin if x0 > w else x0)
        cy = -margin if y0 < 0 else (h + margin if y0 > h else y0)
        return cx, cy
    crossings = []
    if dx:
        crossings += [(0 - x0) / dx, (w - x0) / dx]
    if dy:
        crossings += [(0 - y0) / dy, (h - y0) / dy]
    t_exit = min((t for t in crossings if t > 0), default=0.0)
    return x0 + (t_exit + margin) * dx, y0 + (t_exit + margin) * dy


def within_bounds(x, y, w=SIM_CANVAS_W, h=SIM_CANVAS_H):
    return -DESPAWN_MARGIN <= x <= w + DESPAWN_MARGIN and -DESPAWN_MARGIN <= y <= h + DESPAWN_MARGIN


def walkable_area_from_semantic_map(bundle: MapBundle) -> WalkableArea:
    """Walkable = NOT drivable, read from the same semantic raster ContextVAE
    and the dashboard use, so there is one source of truth for 'road'."""
    sf = bundle.screen_frame
    xs = np.arange(SIM_CANVAS_W) + 0.5
    ys = np.arange(SIM_CANVAS_H) + 0.5
    X, Y = np.meshgrid(xs, ys)
    wx, wy, _ = sf.to_world(X, Y, 0.0)
    drivable = bundle.is_drivable(np.stack([wx, wy], axis=-1))
    return WalkableArea(~drivable, SIM_CANVAS_W, SIM_CANVAS_H)


class _CrossingPedestrian(Pedestrian):
    """With probability ``bias`` the next target is the mirror point across the
    nearest road, so the straight walk to it crosses traffic."""

    def __init__(self, *a, drivable: np.ndarray, bias: float, **kw):
        self._drivable = drivable
        self._bias = bias
        super().__init__(*a, **kw)

    def _pick_new_target(self):
        if self._bias > 0 and self._rng.random() < self._bias:
            for _ in range(40):
                ang = self._rng.uniform(0, 2 * math.pi)
                dist = self._rng.uniform(80, 260)
                tx, ty = self.x + math.cos(ang) * dist, self.y + math.sin(ang) * dist
                if not self.area.is_walkable(tx, ty):
                    continue
                n = 30
                xs = np.linspace(self.x, tx, n).astype(int).clip(0, SIM_CANVAS_W - 1)
                ys = np.linspace(self.y, ty, n).astype(int).clip(0, SIM_CANVAS_H - 1)
                if self._drivable[ys, xs].any():
                    return tx, ty
        return super()._pick_new_target()


class World:
    def __init__(self, config: WorldConfig, bundle: Optional[MapBundle] = None):
        self.cfg = config
        self.bundle = bundle or MapBundle.load(config.map_name)
        self.frame = self.bundle.screen_frame
        self.graph = WaypointGraph.from_file(self.bundle.waypoints_path)
        self.starts = self.graph.route_starts()
        if not self.starts:
            raise ValueError(f"no routes in {self.bundle.waypoints_path}")
        self.area = walkable_area_from_semantic_map(self.bundle)
        self.rng = random.Random(config.seed)
        self.seq = 0
        self._next_id = 1
        self._route_of = {}
        n_veh = config.num_vehicles or len(self.starts)
        self._scheduled = [(i * config.vehicle_stagger_s, i) for i in range(n_veh)]
        self.vehicles: List[Vehicle] = []
        self._spawn_due()
        self.pedestrians: List[Pedestrian] = [self._spawn_pedestrian() for _ in range(config.num_pedestrians)]

    @property
    def t_s(self) -> float:
        return self.seq * self.cfg.dt

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _spawn_vehicle(self, route_idx: int) -> Vehicle:
        wp = self.starts[route_idx % len(self.starts)]
        nb = self.graph.neighbours_of(wp.id)
        if nb:
            away = math.atan2(-(nb[0].y - wp.y), -(nb[0].x - wp.x))
            sx, sy = offscreen_point(wp.x, wp.y, away)
        else:
            sx, sy = wp.x, wp.y
        vid = self._new_id()
        self._route_of[vid] = route_idx
        return Vehicle(vid, self.graph, sx, sy, wp.id, speed=self.cfg.vehicle_speed_px, routing=self.cfg.routing)

    def _spawn_pedestrian(self) -> Pedestrian:
        ex, ey = self.area.random_point(self.rng)
        sx, sy = offscreen_point(ex, ey, self.rng.uniform(0, 2 * math.pi))
        pid = self._new_id()
        # The prototype set this in px/s (13 px/s), which is a walking pace on
        # T11 (10 px/m) but 6.5 m/s on OneNorth (2 px/m).
        speed_px = self.cfg.pedestrian_speed_mps * self.frame.pixels_per_metre
        kw = dict(speed=speed_px, lifetime_s=self.cfg.pedestrian_lifetime_s,
                  seed=self.cfg.seed * 100003 + pid, initial_target=(ex, ey))
        if self.cfg.crossing_bias > 0:
            return _CrossingPedestrian(pid, sx, sy, self.area, drivable=~self.area.mask,
                                       bias=self.cfg.crossing_bias, **kw)
        return Pedestrian(pid, sx, sy, self.area, **kw)

    def _spawn_due(self) -> None:
        due = [r for t, r in self._scheduled if t <= self.t_s + 1e-9]
        self._scheduled = [(t, r) for t, r in self._scheduled if t > self.t_s + 1e-9]
        self.vehicles += [self._spawn_vehicle(r) for r in due]

    def step(self) -> None:
        dt = self.cfg.dt
        self._spawn_due()
        for v in self.vehicles:
            v.step(dt)
            if v._stopped:  # reached the end of its route: drive on and exit
                v._stopped, v._exiting = False, True
        self.vehicles = [v if within_bounds(v.x, v.y) else self._spawn_vehicle(self._route_of[v.id])
                         for v in self.vehicles]
        for p in self.pedestrians:
            if p.lifetime_expired:
                p.start_exiting(offscreen_point(p.x, p.y, p.heading))
            p.step(dt)
        self.pedestrians = [p if within_bounds(p.x, p.y) else self._spawn_pedestrian() for p in self.pedestrians]
        self.seq += 1

    def tick(self) -> Tick:
        ppm = self.frame.pixels_per_metre
        agents = []
        for kind, items in (("vehicle", self.vehicles), ("pedestrian", self.pedestrians)):
            for a in items:
                x, y, h = self.frame.to_world(a.x, a.y, a.heading)
                agents.append(AgentState(a.id, kind, x, y, h, a.speed / ppm))
        return Tick(self.seq, self.t_s, agents)


def generate(config: WorldConfig, duration_s: float, warmup_s: float = 10.0):
    """Run headless and return the list of ticks (after a warm-up so the
    scene is populated). Tick times keep counting from the start of warm-up."""
    w = World(config)
    n_warm = int(round(warmup_s / config.dt))
    for _ in range(n_warm):
        w.step()
    ticks = []
    for _ in range(int(round(duration_s / config.dt))):
        w.step()
        ticks.append(w.tick())
    return ticks
