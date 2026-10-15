"""Vehicle route graph (read-only). Positions are simulator pixels, y down.

The original prototype's WaypointManager also contained an interactive
pygame editor; only the parts the simulation needs are kept here.
"""
from __future__ import annotations

import json
from typing import Dict, List


class Waypoint:
    def __init__(self, waypoint_id: int, x: float, y: float, path_id: int = 0):
        self.id = waypoint_id
        self.x, self.y = float(x), float(y)
        self.path_id = path_id
        self.neighbours: List[int] = []

    @classmethod
    def from_dict(cls, d: dict) -> "Waypoint":
        wp = cls(d["id"], d["x"], d["y"], path_id=d.get("path_id", 0))
        wp.neighbours = list(d.get("neighbours", []))
        return wp


class WaypointGraph:
    def __init__(self) -> None:
        self.waypoints: Dict[int, Waypoint] = {}

    @classmethod
    def from_file(cls, path: str) -> "WaypointGraph":
        g = cls()
        with open(path) as f:
            data = json.load(f)
        for d in data.get("waypoints", []):
            wp = Waypoint.from_dict(d)
            g.waypoints[wp.id] = wp
        return g

    def neighbours_of(self, waypoint_id: int) -> List[Waypoint]:
        wp = self.waypoints.get(waypoint_id)
        if wp is None:
            return []
        return [self.waypoints[n] for n in wp.neighbours if n in self.waypoints]

    def route_starts(self) -> List[Waypoint]:
        """The lowest-id waypoint of each path = where each route begins."""
        first: Dict[int, Waypoint] = {}
        for wp in self.waypoints.values():
            if wp.path_id not in first or wp.id < first[wp.path_id].id:
                first[wp.path_id] = wp
        return [first[pid] for pid in sorted(first)]
