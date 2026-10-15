"""SocialVAE (pedestrians): map-free, world-frame trajectory sampling.

Differences from ContextVAE, and why:

* no map branch and no heading rotation: SDD (the training set) has no map
  rasters and no heading column, so the model works directly in world metres;
* trained at 2.5 Hz (0.4 s steps), 8 observed / 12 predicted samples (3.2 s / 4.8 s);
* N = fpc x 20 raw samples, reduced to k=20 by Final Position Clustering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from ..third_party.socialvae import SocialVAE
from ..runtime.checkpoints import load_checkpoint
from .base import Candidate, PredictionRequest
from .clustering import final_position_clustering
from .kinematics import extend, neighbour_states


@dataclass
class SocialVAEConfig:
    """Values from upstream config/sdd.py."""
    ob_horizon: int = 8
    pred_horizon: int = 12
    step_dt_s: float = 0.4
    ob_radius: float = 5.0
    hidden_dim: int = 512
    k: int = 20                       # PRED_SAMPLES
    raw_multiplier: Optional[int] = None  # None = the checkpoint's own tuned `fpc` (37 -> 740 raw samples)


class SocialVAEPredictor:
    name = "socialvae"

    def __init__(self, ckpt_path: str, config: Optional[SocialVAEConfig] = None,
                 device: str = "cpu", seed: Optional[int] = None):
        self.cfg = config or SocialVAEConfig()
        self.step_dt_s = self.cfg.step_dt_s
        self.ob_horizon = self.cfg.ob_horizon
        self.pred_horizon = self.cfg.pred_horizon
        self.device = torch.device(device)
        self.seed = seed
        self._calls = 0

        self.model = SocialVAE(horizon=self.cfg.pred_horizon, ob_radius=self.cfg.ob_radius,
                               hidden_dim=self.cfg.hidden_dim).to(self.device)
        state = load_checkpoint(ckpt_path, self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.checkpoint_info = {k: state.get(k) for k in ("epoch", "ade", "fde", "fpc")}
        self.raw_multiplier = self.cfg.raw_multiplier or int(state.get("fpc", 1))

    @property
    def buffer_len(self) -> int:
        return self.ob_horizon + 2

    @property
    def raw_samples(self) -> int:
        return self.raw_multiplier * self.cfg.k

    @torch.no_grad()
    def raw_samples_world(self, request: PredictionRequest, n: Optional[int] = None) -> np.ndarray:
        hist = extend(request.positions)[-self.ob_horizon:]
        nb = neighbour_states(request.neighbours, self.buffer_len, self.ob_horizon)
        x = torch.tensor(hist, dtype=torch.float32, device=self.device).unsqueeze(1)
        nbt = torch.tensor(nb, dtype=torch.float32, device=self.device).unsqueeze(1)
        if self.seed is not None:
            torch.manual_seed(self.seed + self._calls)
        self._calls += 1
        pred = self.model(x, nbt, n_predictions=n or self.raw_samples)  # n x H x 1 x 2
        return pred.squeeze(2).cpu().numpy()

    def predict(self, request: PredictionRequest) -> List[Candidate]:
        if len(request.positions) != self.buffer_len:
            return []
        raw = self.raw_samples_world(request)
        seed = None if self.seed is None else self.seed + self._calls
        return final_position_clustering(raw, self.cfg.k, seed=seed)
