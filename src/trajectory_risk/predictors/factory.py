"""Build the vehicle and pedestrian predictors by name."""
from __future__ import annotations

from typing import Optional, Tuple

from ..maps import MapBundle
from ..runtime.checkpoints import CONTEXTVAE_CKPT, SOCIALVAE_CKPT, device
from .constant_velocity import ConstantVelocity

VEHICLE_MODELS = ("contextvae", "cv")
PEDESTRIAN_MODELS = ("socialvae", "cv")


def vehicle_predictor(name: str, bundle: MapBundle, raw_samples: Optional[int] = None,
                      k: Optional[int] = None, map_mode: str = "real", seed: Optional[int] = None):
    if name == "cv":
        return ConstantVelocity(0.5, 5, 6, name="cv_vehicle")
    if name == "contextvae":
        from .contextvae import ContextVAEConfig, load_contextvae
        cfg = ContextVAEConfig()
        if raw_samples:
            cfg.raw_samples = raw_samples
        if k:
            cfg.k = k
        return load_contextvae(CONTEXTVAE_CKPT, bundle, config=cfg, device=device(), map_mode=map_mode, seed=seed)
    raise ValueError(f"vehicle model must be one of {VEHICLE_MODELS}")


def pedestrian_predictor(name: str, raw_multiplier: Optional[int] = None, k: Optional[int] = None,
                         seed: Optional[int] = None):
    if name == "cv":
        return ConstantVelocity(0.4, 8, 12, name="cv_pedestrian")
    if name == "socialvae":
        from .socialvae import SocialVAEConfig, SocialVAEPredictor
        cfg = SocialVAEConfig(raw_multiplier=raw_multiplier)
        if k:
            cfg.k = k
        return SocialVAEPredictor(SOCIALVAE_CKPT, config=cfg, device=device(), seed=seed)
    raise ValueError(f"pedestrian model must be one of {PEDESTRIAN_MODELS}")


def build(bundle: MapBundle, models: str = "learned", **kw) -> Tuple[object, object]:
    """models: "learned" (ContextVAE + SocialVAE) or "cv" (constant velocity, no weights needed)."""
    if models == "cv":
        return vehicle_predictor("cv", bundle), pedestrian_predictor("cv")
    return (vehicle_predictor("contextvae", bundle, kw.get("raw_samples"), kw.get("k"), seed=kw.get("seed")),
            pedestrian_predictor("socialvae", kw.get("ped_raw_multiplier"), kw.get("ped_k"), seed=kw.get("seed")))
