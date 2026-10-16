import pygame

from trajectory_risk.dashboard.render import CANVAS_H, CANVAS_W, DashboardRenderer, init_headless
from trajectory_risk.maps import MapBundle
from trajectory_risk.pipeline import LivePipeline
from trajectory_risk.predictors import ConstantVelocity
from trajectory_risk.sim import WorldConfig, generate


def test_renders_a_frame_headless():
    init_headless()
    pipe = LivePipeline(ConstantVelocity(0.5, 5, 6), ConstantVelocity(0.4, 8, 12))
    for t in generate(WorldConfig(num_pedestrians=6, seed=1), 8.0, 5.0):
        pipe.step_sync(t)
    surf = pygame.Surface((CANVAS_W, CANVAS_H))
    DashboardRenderer(MapBundle.load("T11")).draw(surf, pipe, {"fps": 60}, debug=True)
    arr = pygame.surfarray.array3d(surf)
    assert arr.std() > 10   # something was actually drawn
