import numpy as np

from trajectory_risk.pipeline import LivePipeline, PipelineConfig
from trajectory_risk.predictors import ConstantVelocity
from trajectory_risk.protocol import AgentState, Tick
from trajectory_risk.sim import WorldConfig, generate


def cv_pipeline(**cfg):
    return LivePipeline(ConstantVelocity(0.5, 5, 6), ConstantVelocity(0.4, 8, 12), PipelineConfig(**cfg))


def collision_course(t):
    """Car going +x along y=0 and a pedestrian walking +y up x=10; both reach (10, 0) at t=6 s."""
    return [AgentState(1, "vehicle", -5 + 2.5 * t, 0.0, 0.0, 2.5),
            AgentState(2, "pedestrian", 10.0, -7.8 + 1.3 * t, np.pi / 2, 1.3)]


def test_alert_fires_before_a_collision_course_meets():
    pipe = cv_pipeline()
    fired_at = None
    for i in range(int(6.0 * 60)):
        t = i / 60
        pipe.step_sync(Tick(i, t, collision_course(t)))
        if fired_at is None and any(r.active for r in pipe.risk_rows):
            fired_at = t
    assert fired_at is not None and fired_at < 5.5   # warning before they meet at t = 6


def test_no_alert_for_separated_traffic():
    pipe = cv_pipeline()
    for i in range(int(8 * 60)):
        t = i / 60
        pipe.step_sync(Tick(i, t, [AgentState(1, "vehicle", 2.5 * t, 0.0, 0.0, 2.5),
                                   AgentState(2, "pedestrian", 1.3 * t, 20.0, 0.0, 1.3)]))
    assert not any(r.active for r in pipe.risk_rows) and pipe.risk_rows[0].risk == 0.0


def test_time_going_backwards_resets_state():
    pipe = cv_pipeline()
    for i in range(300):
        pipe.ingest(Tick(i, i / 60, collision_course(i / 60)))
    assert pipe.stores["vehicle"].tracks
    pipe.ingest(Tick(0, 0.0, collision_course(0.0)))
    assert pipe.resets == 1 and len(pipe.stores["vehicle"].tracks[1].samples) == 1


def test_risk_is_not_scored_for_agents_that_left():
    pipe = cv_pipeline()
    for i in range(int(6 * 60)):
        t = i / 60
        agents = collision_course(t) if t < 4 else collision_course(t)[:1]
        pipe.step_sync(Tick(i, t, agents))
    assert all(r.ped_id != 2 for r in pipe.risk_rows)


def test_runs_on_a_generated_scenario():
    pipe = cv_pipeline()
    n = 0
    for tick in generate(WorldConfig(num_pedestrians=6, crossing_bias=0.5, seed=1), 20.0, 5.0):
        n += len(pipe.step_sync(tick))
    assert n > 50 and pipe.risk_rows
