"""Pedestrian agent and the walkable area it wanders within.

Walkable-area design: T11's Roboflow export (roboflow_data/T11/train) has
no dedicated sidewalk/walkway layer -- only `driveable_area` (the road) and
a sparse 2-annotation `ped_crossing` layer, not enough to build a real
crossing-permission system from. So v1 keeps it simple: walkable = NOT
driveable, i.e. pedestrians wander freely off the road and never cross onto
it. (A real crossing system -- letting pedestrians briefly enter
`ped_crossing` polygons -- is a reasonable future refinement, not built
here.)

Movement model deliberately differs from Vehicle: no waypoint graph (there's
nothing route-like about open-area foot traffic), no turn-rate limiting
(people change direction quickly, unlike a steered vehicle) -- just
"pick a random walkable point, walk straight to it, repeat" with a capped
lifetime before heading back off-screen, mirroring window1_tracker.py's
vehicle spawn/despawn cycle (enter from off-screen, wander, exit off-screen,
respawn) without needing a route concept to define "the end."
"""
from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import numpy as np

DEFAULT_PED_SPEED = 13.0  # px/s -- 1.3 m/s at 10px/m, a typical walking pace
ARRIVE_RADIUS = 8.0
PED_RADIUS_PX = 5  # drawn size only


class WalkableArea:
    """Wraps a boolean walkable mask (pixel-space, same convention as
    vehicle_main.py's RoadMap.mask) and samples random walkable points
    within it, for pedestrians to wander toward."""

    def __init__(self, walkable_mask: np.ndarray, game_w: int, game_h: int):
        self.mask = walkable_mask
        self.w, self.h = game_w, game_h

    def is_walkable(self, x: float, y: float) -> bool:
        xi, yi = int(x), int(y)
        if xi < 0 or xi >= self.w or yi < 0 or yi >= self.h:
            return False
        return bool(self.mask[yi, xi])

    def random_point(self, rng: random.Random, max_tries: int = 500) -> Tuple[float, float]:
        for _ in range(max_tries):
            x = rng.uniform(0, self.w)
            y = rng.uniform(0, self.h)
            if self.is_walkable(x, y):
                return x, y
        # Degenerate mask (shouldn't happen for T11) -- fall back to centre.
        return self.w / 2.0, self.h / 2.0


class Pedestrian:
    """Wanders the given WalkableArea: walks straight to a random walkable
    point, picks a new one on arrival, repeats -- until `lifetime_s` has
    elapsed, at which point it heads for an off-screen exit point instead
    and despawns once window1_tracker.py's usual off-screen margin check
    catches it (same despawn path vehicles use).
    """

    def __init__(self, ped_id: int, x: float, y: float, area: WalkableArea,
                 speed: float = DEFAULT_PED_SPEED, lifetime_s: float = 30.0,
                 seed: Optional[int] = None, initial_target: Optional[Tuple[float, float]] = None):
        """initial_target: where to head first, e.g. a walkable point just
        inside the visible window when spawned off-screen -- otherwise a
        vehicle-style off-screen spawn would immediately roll the dice on a
        random walkable point that could be anywhere, including back
        off-screen the other side. Defaults to a random walkable point
        (free wandering with no entry phase)."""
        self.id = ped_id
        self.area = area
        self.speed = speed
        self.x, self.y = float(x), float(y)
        self.heading = 0.0
        self._rng = random.Random(seed if seed is not None else ped_id)
        self._age = 0.0
        self._lifetime = lifetime_s
        self._exiting = False
        self.target: Tuple[float, float] = initial_target if initial_target is not None else self._pick_new_target()

    def _pick_new_target(self) -> Tuple[float, float]:
        return self.area.random_point(self._rng)

    def start_exiting(self, exit_point: Tuple[float, float]) -> None:
        """Called by window1_tracker.py once this pedestrian's lifetime is
        up: aim it at an off-screen point and stop picking new wander
        targets, mirroring Vehicle's own `_exiting` state.

        exit_point only sets the HEADING to walk in -- once exiting, step()
        walks straight forever with no arrival/distance check at all
        (exactly Vehicle.step()'s own `_exiting` branch). An earlier
        version instead kept a fixed exit *point* as `self.target` and let
        the usual arrival-clamped movement carry it there -- but that
        point sat just past window1_tracker.py's SPAWN_MARGIN, which is
        LESS than DESPAWN_MARGIN, so the pedestrian would arrive, its
        distance-to-target hit ~0, and step() then moved it 0px forever:
        frozen just inside the despawn zone instead of ever leaving it.
        """
        self._exiting = True
        self.heading = math.atan2(exit_point[1] - self.y, exit_point[0] - self.x)

    def step(self, dt: float) -> None:
        self._age += dt

        if self._exiting:
            self.x += math.cos(self.heading) * self.speed * dt
            self.y += math.sin(self.heading) * self.speed * dt
            return

        dx, dy = self.target[0] - self.x, self.target[1] - self.y
        dist = math.hypot(dx, dy)

        if dist <= ARRIVE_RADIUS:
            self.target = self._pick_new_target()
            dx, dy = self.target[0] - self.x, self.target[1] - self.y
            dist = math.hypot(dx, dy)

        if dist > 1e-6:
            self.heading = math.atan2(dy, dx)
            step_len = min(self.speed * dt, dist)
            self.x += math.cos(self.heading) * step_len
            self.y += math.sin(self.heading) * step_len

    @property
    def lifetime_expired(self) -> bool:
        return self._age >= self._lifetime and not self._exiting
