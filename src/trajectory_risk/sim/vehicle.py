"""A vehicle that drives itself along a shared waypoint graph, with a
turn-rate limit so turns look like steering rather than snapping.

Vendored from the original prototype's sim/ (pygame drawing removed so the
simulation can run headless). Units are SIMULATOR PIXELS; the World converts
to world metres before anything leaves the simulator.
"""
import math
import random
from typing import Optional

from .waypoints import WaypointGraph as WaypointManager

DEFAULT_SPEED = 25.0            # px/s
DEFAULT_TURN_RATE = math.radians(120)  # max heading change per second, for a natural-looking turn
ARRIVE_RADIUS = 10.0            # px: distance to a waypoint counted as "reached"


class Vehicle:
    """Drives from waypoint to waypoint along `waypoint_manager`'s graph.

    Can start from any (x, y), not just a waypoint's own position -- it
    drives to its initial target waypoint first, then follows the graph
    as usual. This is what lets the editor drop a vehicle anywhere on
    the map by dragging, instead of only at exact node positions.

    Reaching a dead end (a waypoint whose only connection is the one just
    arrived from) stops the vehicle there: it does NOT turn around and
    retrace the chain backward, and does NOT keep driving forward into
    whatever's past the dead end either -- it just holds position from
    then on (see `_stopped`). (Two earlier behaviours were tried and
    rejected: driving straight through assumed every route was drawn out
    to the window edge to act as an exit ramp -- for routes that dead-end
    inside the mapped area instead, that drove the vehicle into a wall and
    froze it there, still "trying" to move every tick. Turning around
    avoided that but made the vehicle retrace its whole route backward,
    which isn't wanted either. Stopping in place avoids both. The only
    case that still drives straight through/exits is a fully isolated
    waypoint with zero neighbours, which has nowhere to stop toward either.)

    routing:
        "random"        -- pick a uniformly random connected waypoint each
            time the vehicle arrives, EXCLUDING the node just arrived
            from (so it only actually randomizes at a real fork -- a
            waypoint with 2+ non-backward options; on a plain chain with
            no forks it advances the same way "deterministic" does).
            Uses a per-vehicle seeded RNG so a given vehicle id
            reproduces the same route.
        "deterministic" -- always advance to the first neighbour that
            isn't the node just came from, reversing direction only
            at a dead end. Produces a fixed, repeatable back-and-forth
            patrol of the route.
    """

    def __init__(self, vehicle_id, waypoint_manager: WaypointManager,
                 x: float, y: float, target_waypoint_id: int,
                 speed: float = DEFAULT_SPEED,
                 routing: str = "random", turn_rate: float = DEFAULT_TURN_RATE):
        """
        x, y: the vehicle's starting position. It does NOT need to sit
            exactly on a waypoint -- e.g. wherever the user dropped it
            while placing it interactively.
        target_waypoint_id: the graph node the vehicle drives toward
            first (normally the nearest one to (x, y)). Once it arrives
            there it continues along the graph like any other vehicle.
            Passing a waypoint's own (x, y) and its own id reproduces the
            old "spawn exactly on a node" behaviour.
        """
        self.id = vehicle_id
        self.manager = waypoint_manager
        self.speed = speed
        self.routing = routing
        self.turn_rate = turn_rate

        self.x, self.y = float(x), float(y)
        self.spawn_x, self.spawn_y = self.x, self.y
        self.spawn_target_waypoint_id = target_waypoint_id
        self.heading = self._initial_heading(self.x, self.y, target_waypoint_id)

        self.current_waypoint_id: Optional[int] = None  # not yet arrived at any graph node
        self.target_waypoint_id: Optional[int] = target_waypoint_id
        self._prev_waypoint_id: Optional[int] = None
        # True once the vehicle has hit a fully isolated waypoint (no
        # neighbours at all, not even the one just arrived from) -- it
        # then just keeps driving straight in its current heading (see
        # step()) until something outside (vehicle_ai4.py) notices it has
        # left the map and removes it.
        self._exiting = False
        # True once the vehicle has hit a normal dead end (single
        # neighbour, and that neighbour is the node just arrived from) --
        # it then holds position for good; see _pick_next_target().
        self._stopped = False
        # Seeded per-vehicle id so "random" routing is reproducible run-to-run.
        self._rng = random.Random(vehicle_id)

    def _initial_heading(self, x: float, y: float, target_waypoint_id: Optional[int]) -> float:
        """Direction to face at spawn, before step() has run even once.

        Without this, heading defaulted to a hardcoded 0.0 (due east)
        regardless of where the vehicle actually needed to go, and only
        started correcting itself once the simulation was unpaused (Space)
        and step()'s turn-rate-limited rotation caught up -- for a route
        pointed anywhere other than east, that meant driving straight into
        whatever was immediately east of the spawn point for the first
        stretch of the turn.

        target_waypoint_id's own position is normally enough. But a
        vehicle spawned exactly ON its target (x, y) == that waypoint's
        own position -- e.g. route_start_waypoints() in vehicle_main.py,
        which spawns each startup vehicle on its route's own first node --
        arrives instantly (dist 0 <= ARRIVE_RADIUS) and re-targets before
        ever moving, so the real first direction of travel is one hop
        further: the target's own first neighbour.
        """
        target = self.manager.waypoints.get(target_waypoint_id) if target_waypoint_id is not None else None
        if target is None:
            return 0.0
        dx, dy = target.x - x, target.y - y
        if dx == 0 and dy == 0:
            neighbours = self.manager.neighbours_of(target_waypoint_id)
            if not neighbours:
                return 0.0
            dx, dy = neighbours[0].x - target.x, neighbours[0].y - target.y
            if dx == 0 and dy == 0:
                return 0.0
        return math.atan2(dy, dx)

    def reset(self) -> None:
        """Return the vehicle to its original starting position and
        re-target the waypoint it first drove toward.
        """
        self.x, self.y = self.spawn_x, self.spawn_y
        self.heading = self._initial_heading(self.spawn_x, self.spawn_y, self.spawn_target_waypoint_id)
        self.current_waypoint_id = None
        self.target_waypoint_id = self.spawn_target_waypoint_id
        self._prev_waypoint_id = None
        self._exiting = False
        self._stopped = False

    def _pick_next_target(self) -> None:
        neighbours = self.manager.neighbours_of(self.current_waypoint_id)
        if not neighbours:
            # Fully isolated waypoint -- nowhere to go, not even back the
            # way we came (there was no way in). Nothing left to do but
            # the old straight-through/exit behaviour.
            self.target_waypoint_id = None
            self._exiting = True
            return
        if len(neighbours) == 1 and neighbours[0].id == self._prev_waypoint_id:
            # Dead end: the only neighbour is the node just arrived from.
            # Stop here for good -- neither turn around and retrace the
            # route backward, nor keep driving forward into the wall (see
            # step()'s `self._stopped` check).
            self.target_waypoint_id = None
            self._stopped = True
            return
        candidates = [n for n in neighbours if n.id != self._prev_waypoint_id]
        if not candidates:
            candidates = neighbours
        if self.routing == "deterministic":
            chosen = candidates[0]
        else:
            # Uniformly random among candidates that AREN'T the node just
            # arrived from -- same exclusion as deterministic. Without
            # this, a 2-neighbour interior waypoint (forward/backward, no
            # actual fork) had a 50% chance every single arrival of
            # immediately reversing, which is a symmetric random walk with
            # no net progress -- it wanders near the spawn point instead
            # of ever reaching the route's end. Excluding prev means
            # "random" only diverges from "deterministic" where a real
            # fork exists (>=2 non-backward options); on a fork-free chain
            # like this one, it now behaves the same as deterministic.
            chosen = self._rng.choice(candidates)
        self.target_waypoint_id = chosen.id

    def step(self, dt: float, road_map=None) -> None:
        """Advance one simulation tick: steer toward the current target
        waypoint and move forward, advancing to the next waypoint once
        within ARRIVE_RADIUS of it. A normal dead end sets `_stopped` via
        _pick_next_target() (see there) and the vehicle holds position
        from then on -- `_exiting` only gets set for the rare case of a
        fully isolated waypoint with no neighbours at all, in which case
        this keeps driving straight in the current heading instead.

        road_map: optional RoadMap (see vehicle_ai4.py) whose drivable-area
        mask the vehicle should stay inside, enforced via _try_move() for
        both normal and `_exiting` movement. The one place it's NOT
        enforced is once a step would land outside the map's canvas
        entirely -- relevant if `_exiting` drives the vehicle off the
        window edge, which is what lets it despawn (see vehicle_ai4.py).
        """
        if self._stopped:
            return

        if self._exiting:
            new_x = self.x + math.cos(self.heading) * self.speed * dt
            new_y = self.y + math.sin(self.heading) * self.speed * dt
            self._try_move(new_x, new_y, road_map)
            return

        if self.target_waypoint_id is None:
            if self.current_waypoint_id is not None:
                self._pick_next_target()
            if self.target_waypoint_id is None:
                return  # just discovered a dead end this tick; drive straight from next tick

        target = self.manager.waypoints.get(self.target_waypoint_id)
        if target is None:
            # Target waypoint was deleted mid-drive by the editor; re-route.
            if self.current_waypoint_id is not None:
                self._pick_next_target()
            elif self.manager.waypoints:
                # Lost our initial destination before ever reaching a node --
                # aim at whatever waypoint is still around.
                self.target_waypoint_id = next(iter(self.manager.waypoints))
            return

        dx, dy = target.x - self.x, target.y - self.y
        dist = math.hypot(dx, dy)

        if dist <= ARRIVE_RADIUS:
            self._prev_waypoint_id = self.current_waypoint_id
            self.current_waypoint_id = target.id
            self._pick_next_target()
            return

        desired_heading = math.atan2(dy, dx)
        heading_error = self._wrap_angle(desired_heading - self.heading)
        max_turn = self.turn_rate * dt
        self.heading += max(-max_turn, min(max_turn, heading_error))

        new_x = self.x + math.cos(self.heading) * self.speed * dt
        new_y = self.y + math.sin(self.heading) * self.speed * dt
        self._try_move(new_x, new_y, road_map)

    def _try_move(self, new_x: float, new_y: float, road_map) -> None:
        """Apply the tentative new position, unless road_map says it's a
        non-drivable pixel INSIDE the mapped canvas -- e.g. a building or
        curb -- in which case the position is held for this tick instead
        (heading has already turned a little further toward the target,
        so it keeps re-trying with a corrected approach angle each frame
        rather than cutting across a wall).

        A step that lands outside the canvas entirely is always allowed
        through: that's not a wall, it's the vehicle leaving the mapped
        area near a dead end, which is how it exits the window and gets
        despawned by vehicle_ai4.py.
        """
        if road_map is not None:
            within_canvas = 0 <= new_x < road_map.w and 0 <= new_y < road_map.h
            if within_canvas and not road_map.is_drivable(new_x, new_y):
                return
        self.x, self.y = new_x, new_y

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
