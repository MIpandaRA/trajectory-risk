"""ContextVAE (vehicles): map-conditioned, heading-normalised trajectory sampling.

One prediction = five small steps, each a separate testable function:

    build_history   finite differences on the last ob_horizon samples
    to_ego_frame    rotate history + neighbours by -heading about the OLDEST
                    observed position (what upstream data.py does in training)
    crop_map        cut the semantic raster around that SAME pivot, rotated by
                    that SAME angle, so "forward" means the same thing to the
                    trajectory encoder and the map encoder
    sample          N stochastic decodes from the prior p(z | history, map)
    cluster         k-means -> k candidates, then rotate back to world frame

Get the pivot or angle wrong in only one of history/map and predictions come
out plausible-looking but wrong, with no error raised.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..geometry import EgoFrame, rotate_state
from ..third_party.contextvae import ContextVAE
from .base import Candidate, PredictionRequest
from .clustering import kmeans_trajectories
from .kinematics import extend, neighbour_states
from ..runtime.checkpoints import load_checkpoint


@dataclass
class ContextVAEConfig:
    """Values from the upstream nuScenes configs (nuscenes_eval_3s_k10)."""
    ob_horizon: int = 5          # 5 samples x 0.5 s = 2.5 s of history
    pred_horizon: int = 6        # 6 x 0.5 s = 3 s ahead (checkpoint trained at 12; horizon is a loop count)
    step_dt_s: float = 0.5       # nuScenes 2 Hz
    ob_radius: float = 30.0
    map_size: int = 224
    hidden_dim: int = 512
    map_model: str = "resnet18"
    raw_samples: int = 2000      # N stochastic decodes
    k: int = 10                  # candidates kept after clustering


class ContextVAEPredictor:
    name = "contextvae"

    def __init__(self, ckpt_path: str, semantic_map: np.ndarray, map_H: np.ndarray,
                 config: Optional[ContextVAEConfig] = None, device: str = "cpu",
                 map_mode: str = "real", seed: Optional[int] = None):
        """map_mode: "real"; "nothing" (all -1: no road anywhere, the encoding's
        own value for empty space) for the map ablation; or "blank" (all 0),
        kept only to reproduce the old ablation: 0 is not "no map" but "half
        drivable everywhere" in this [-1, 1] encoding (docs/failures.md #13)."""
        self.cfg = config or ContextVAEConfig()
        self.step_dt_s = self.cfg.step_dt_s
        self.ob_horizon = self.cfg.ob_horizon
        self.pred_horizon = self.cfg.pred_horizon
        self.device = torch.device(device)
        self.seed = seed
        self._calls = 0

        self.model = ContextVAE(horizon=self.cfg.pred_horizon, ob_radius=self.cfg.ob_radius,
                                hidden_dim=self.cfg.hidden_dim, map_model=self.cfg.map_model).to(self.device)
        state = load_checkpoint(ckpt_path, self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.checkpoint_info = {k: state.get(k) for k in ("epoch", "ade", "fde")}

        sem = torch.tensor(np.asarray(semantic_map), dtype=torch.float32)
        if map_mode == "blank":
            sem = torch.zeros_like(sem)
        elif map_mode == "nothing":
            sem = torch.full_like(sem, -1.0)
        elif map_mode != "real":
            raise ValueError(f"unknown map_mode {map_mode!r}")
        self.map_mode = map_mode
        self.semantic_map = sem
        self.map_H = np.asarray(map_H, dtype=np.float64)

        # Crop geometry from upstream data.py (map_scale = 1): the agent sits a
        # quarter of the way up from the bottom-centre of the crop, so the map
        # shows more road ahead than behind. EXT is large enough that rotating
        # the square crop never pulls in empty corners.
        m = self.cfg.map_size
        top, left = m // 2, m // 4
        self.ext = int(np.ceil(m // 4 * ((9 + 4) ** 0.5)))
        self.crop_rows = (self.ext - top, self.ext + (m - top))
        self.crop_cols = (self.ext - left, self.ext + (m - left))

    @property
    def buffer_len(self) -> int:
        return self.ob_horizon + 2

    # -- stages ------------------------------------------------------------
    def build_history(self, request: PredictionRequest):
        """-> (hist (ob_horizon, 6) world frame, ego frame)."""
        hist = extend(request.positions)[-self.ob_horizon:]
        heading_oldest = float(request.headings[-self.ob_horizon])
        frame = EgoFrame(pivot=tuple(hist[0, :2].astype(np.float64)), heading=heading_oldest)
        return hist, frame

    def to_ego_frame(self, hist: np.ndarray, neighbours: np.ndarray, frame: EgoFrame):
        pivot = np.asarray(frame.pivot)
        hist_e = rotate_state(hist, pivot, frame.angle)
        nb_e = rotate_state(neighbours, pivot, frame.angle) if neighbours.shape[1] else neighbours
        return hist_e, nb_e

    def crop_map(self, frame: EgoFrame) -> torch.Tensor:
        """-> (1, 1, C, map_size, map_size), zero-padded past the raster edge."""
        rc = self.map_H @ np.array([frame.pivot[0], frame.pivot[1], 1.0])
        r, c = int(np.floor(rc[0])), int(np.floor(rc[1]))
        _, h, w = self.semantic_map.shape
        e = self.ext
        pad = (max(0, e - c), max(0, c + e - w), max(0, e - r), max(0, r + e - h))  # left, right, top, bottom
        sem = F.pad(self.semantic_map, pad) if any(pad) else self.semantic_map
        r, c = r + pad[2], c + pad[0]
        crop = sem[:, r - e:r + e, c - e:c + e].unsqueeze(0)

        s, cs = np.sin(frame.angle), np.cos(frame.angle)
        theta = torch.tensor([[cs, -s, 0.0], [s, cs, 0.0]], dtype=torch.float32).unsqueeze(0)
        grid = F.affine_grid(theta, crop.size(), align_corners=False)
        rotated = F.grid_sample(crop, grid, align_corners=False)
        out = rotated[..., self.crop_rows[0]:self.crop_rows[1], self.crop_cols[0]:self.crop_cols[1]]
        return out.unsqueeze(0)

    @torch.no_grad()
    def sample(self, hist_e: np.ndarray, nb_e: np.ndarray, map_crop: torch.Tensor, n: int) -> np.ndarray:
        """-> (n, pred_horizon, 2) in the ego frame."""
        x = torch.tensor(hist_e, dtype=torch.float32, device=self.device).unsqueeze(1)    # L x 1 x 6
        nb = torch.tensor(nb_e, dtype=torch.float32, device=self.device).unsqueeze(1)     # L x 1 x Nn x 6
        pred = self.model(x, nb, map_crop.to(self.device), n_predictions=n)               # n x H x 1 x 2
        return pred.squeeze(2).cpu().numpy()

    # -- full prediction ---------------------------------------------------
    def raw_samples_world(self, request: PredictionRequest, n: Optional[int] = None) -> np.ndarray:
        hist, frame = self.build_history(request)
        nb = neighbour_states(request.neighbours, self.buffer_len, self.ob_horizon)
        hist_e, nb_e = self.to_ego_frame(hist, nb, frame)
        if self.seed is not None:
            torch.manual_seed(self.seed + self._calls)
        self._calls += 1
        raw_e = self.sample(hist_e, nb_e, self.crop_map(frame), n or self.cfg.raw_samples)
        return frame.to_world(raw_e)

    def predict(self, request: PredictionRequest) -> List[Candidate]:
        if len(request.positions) != self.buffer_len:
            return []
        raw = self.raw_samples_world(request)
        seed = None if self.seed is None else self.seed + self._calls
        return kmeans_trajectories(raw, self.cfg.k, seed=seed)


def load_contextvae(ckpt_path: str, map_bundle, **kwargs) -> ContextVAEPredictor:
    sem, H = map_bundle.semantic
    return ContextVAEPredictor(ckpt_path, sem, H, **kwargs)


__all__ = ["ContextVAEConfig", "ContextVAEPredictor", "load_contextvae"]
