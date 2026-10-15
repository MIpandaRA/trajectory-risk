"""ContextVAE network definition, vendored unmodified from
https://github.com/xupei0610/ContextVAE (MIT, Copyright (c) 2023 Pei Xu).

Xu, Hayet, Karamouzas. "Context-Aware Timewise VAEs for Real-Time Vehicle
Trajectory Prediction." IEEE Robotics and Automation Letters 8(9), 2023.
"""
from .model import ContextVAE

__all__ = ["ContextVAE"]
