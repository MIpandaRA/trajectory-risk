"""Headless traffic simulator (vehicles on a route graph, wandering pedestrians)."""
from .world import World, WorldConfig, generate

__all__ = ["World", "WorldConfig", "generate"]
