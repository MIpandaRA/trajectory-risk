"""Checkpoint discovery and loading."""
from __future__ import annotations

import os

import torch

from ..maps import REPO_ROOT

CHECKPOINT_DIR = os.environ.get("TRAJ_RISK_CHECKPOINTS", os.path.join(REPO_ROOT, "checkpoints"))
CONTEXTVAE_CKPT = os.path.join(CHECKPOINT_DIR, "contextvae", "ckpt-best")
SOCIALVAE_CKPT = os.path.join(CHECKPOINT_DIR, "socialvae", "ckpt-best")


def checkpoints_available() -> bool:
    return os.path.isfile(CONTEXTVAE_CKPT) and os.path.isfile(SOCIALVAE_CKPT)


def load_checkpoint(path: str, map_location):
    """torch >= 2.6 defaults to weights_only=True, which refuses these
    checkpoints (they pickle plain Python objects next to the weights);
    torch 1.11 does not know the keyword at all. Try both."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found. See the README for where to get the model weights.")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError as e:
        if "weights_only" not in str(e):
            raise
        return torch.load(path, map_location=map_location)


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
