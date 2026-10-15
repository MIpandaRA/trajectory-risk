"""Constant velocity vs SocialVAE on the same simulated pedestrians.

Both predictors see exactly the same requests from the same tracker. Each
prediction is scored against what the pedestrian actually did over 4.8 s,
split into "straight" and "turning" (true future heading changes by more than
20 degrees).

    python -m eval.compare_pedestrians
"""
import math

import numpy as np

from eval.common import ade_fde, bootstrap_ci, future_of, ground_truth
from trajectory_risk.predictors import ConstantVelocity
from trajectory_risk.predictors.socialvae import SocialVAEPredictor
from trajectory_risk.runtime.checkpoints import SOCIALVAE_CKPT
from trajectory_risk.sim import WorldConfig, generate
from trajectory_risk.tracking import TrackStore

TURN_DEG = 20.0
# Pedestrian settings of the old repo's three T11 scenes.
SCENES = {"t11_quiet": dict(num_vehicles=2, num_pedestrians=3, seed=1),
          "t11_crossing": dict(num_pedestrians=10, crossing_bias=0.6, seed=2),
          "t11_busy": dict(num_pedestrians=16, crossing_bias=0.3, seed=3)}


def heading_change_deg(future, now):
    """Angle between the first and last step of the true future."""
    pts = np.vstack([now, future])
    d0, d1 = pts[1] - pts[0], pts[-1] - pts[-2]
    if np.linalg.norm(d0) < 1e-6 or np.linalg.norm(d1) < 1e-6:
        return 0.0
    a = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    return abs(math.degrees((a + math.pi) % (2 * math.pi) - math.pi))


models = {"constant velocity": ConstantVelocity(0.4, 8, 12),
          "SocialVAE": SocialVAEPredictor(SOCIALVAE_CKPT, seed=0)}

rows = []  # (model, motion, top1 ADE, top1 FDE, best-of-k ADE)
for scene, kw in SCENES.items():
    ticks = generate(WorldConfig(map_name="T11", **kw), 120.0, 10.0)
    truth = ground_truth(ticks, 0.4, "pedestrian")
    store = TrackStore("pedestrian", 0.4, models["SocialVAE"].buffer_len, predict_every=2)
    for tick in ticks:
        store.ingest(tick)
        for req in store.due_requests():
            actual = future_of(truth, req.agent_id, req.sample_index, 12)
            if actual is None:
                continue
            now = np.array(truth[req.agent_id][req.sample_index])
            motion = "turning" if heading_change_deg(actual, now) > TURN_DEG else "straight"
            for name, model in models.items():
                cands = model.predict(req)
                ade, fde = ade_fde(cands[0].trajectory, actual)
                best = min(ade_fde(c.trajectory, actual)[0] for c in cands)
                rows.append((name, motion, ade, fde, best))
    print(f"[{scene}] done", flush=True)

print(f"\nPedestrians on T11 (3 scenes, 120 s each). Top-1 = the most confident candidate.")
print(f"{'':10s} {'model':18s} {'n':>5s}  {'top-1 ADE':>20s}  {'FDE':>6s}  {'best-of-k ADE':>13s}")
for motion in ("all", "straight", "turning"):
    for name in models:
        sel = [r for r in rows if r[0] == name and (motion == "all" or r[1] == motion)]
        ade = np.array([r[2] for r in sel])
        lo, hi = bootstrap_ci(ade)
        print(f"{motion:10s} {name:18s} {len(sel):5d}  {ade.mean():5.2f} m [{lo:.2f}, {hi:.2f}]  "
              f"{np.mean([r[3] for r in sel]):5.2f}m  {np.mean([r[4] for r in sel]):11.2f} m")
