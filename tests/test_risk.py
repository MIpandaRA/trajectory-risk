import math

import numpy as np
import pytest

from trajectory_risk.predictors.base import Candidate
from trajectory_risk.risk import (HeldPrediction, HysteresisAlert, bayes_alert_threshold, closest_approach,
                                  cpa_constant_velocity, future_points, pairwise_risk)


def line(start, vel, n, dt):
    start, vel = np.asarray(start, float), np.asarray(vel, float)
    return start + vel * dt * np.arange(1, n + 1)[:, None]


def held(start, vel, dt, n, conf=1.0, anchor=0.0):
    return HeldPrediction(tuple(start), anchor, dt, [Candidate(line(start, vel, n, dt), conf)])


def test_same_place_same_time_is_high_risk():
    veh = held((-5, 0), (2.5, 0), 0.5, 6)      # reaches x=0 at t=2
    ped = held((0, -2.6), (0, 1.3), 0.4, 12)   # reaches y=0 at t=2
    risk, d = pairwise_risk(veh, ped, 0.0)
    assert risk == pytest.approx(1.0) and d < 0.5


def test_same_crossing_five_seconds_apart_is_safe():
    """Paths intersect in space but not in time: pure geometry would flag this."""
    veh = held((-5, 0), (2.5, 0), 0.5, 6)                  # crosses x=0 at t=2
    ped = held((0, -9.1), (0, 1.3), 0.4, 12)               # crosses y=0 at t=7
    risk, d = pairwise_risk(veh, ped, 0.0)
    assert risk == 0.0 and d > 2.2


def test_parallel_separated_paths_are_safe():
    veh = held((0, 0), (2.5, 0), 0.5, 6)
    ped = held((0, 5), (1.3, 0), 0.4, 12)
    assert pairwise_risk(veh, ped, 0.0)[0] == 0.0


def test_low_confidence_dangerous_candidate_gives_intermediate_score():
    veh = HeldPrediction((-5, 0), 0.0, 0.5, [Candidate(line((-5, 0), (2.5, 0), 6, 0.5), 0.3),
                                             Candidate(line((-5, 0), (0, 2.5), 6, 0.5), 0.7)])
    ped = held((0, -2.6), (0, 1.3), 0.4, 12)
    risk, _ = pairwise_risk(veh, ped, 0.0)
    assert risk == pytest.approx(0.3)


def test_risk_score_is_bounded():
    veh = HeldPrediction((0, 0), 0.0, 0.5, [Candidate(np.zeros((6, 2)), 0.9)] * 3)
    ped = HeldPrediction((0, 0), 0.0, 0.4, [Candidate(np.zeros((12, 2)), 0.9)] * 3)
    assert pairwise_risk(veh, ped, 0.0)[0] <= 1.0


def test_future_points_drops_the_past():
    traj = line((0, 0), (1, 0), 6, 0.5)       # points at t = 0.5 .. 3.0
    t, pts = future_points(traj, anchor_time_s=10.0, step_dt_s=0.5, now_s=11.2)
    assert np.allclose(t, [11.5, 12.0, 12.5, 13.0]) and len(pts) == 4


def test_stale_prediction_does_not_fold_back():
    """Agent has moved on since the anchor: the resampled path starts at its
    CURRENT position and only moves forward."""
    h = HeldPrediction((2.5, 0), 0.0, 0.5, [Candidate(line((0, 0), (2.5, 0), 6, 0.5), 1.0)])
    from trajectory_risk.risk import candidate_paths
    pos, _ = candidate_paths(h, now_s=1.0, grid_t=1.0 + np.arange(0, 2.0, 0.1))
    assert np.all(np.diff(pos[0, :, 0]) >= -1e-9)


def test_analytic_cpa_matches_sampled():
    p1, v1, p2, v2 = (0, 0), (2, 0), (10, -5), (0, 2)
    d, t = cpa_constant_velocity(p1, v1, p2, v2, 6.0)
    ts = np.linspace(0, 6, 6001)[:, None]
    a, b = np.array(p1) + ts * v1, np.array(p2) + ts * v2
    ds, i = closest_approach(a, b)
    assert d == pytest.approx(ds, abs=1e-3) and t == pytest.approx(ts[i, 0], abs=1e-2)


def test_hysteresis_needs_a_streak_to_enter_and_to_clear():
    a = HysteresisAlert(0.15, 3, 0.05, 5)
    assert [a.update(r) for r in (0.2, 0.2, 0.1, 0.2, 0.2)] == [False] * 5   # dead-zone value resets the streak
    assert a.update(0.2) is True
    assert [a.update(0.0) for _ in range(4)] == [True] * 4
    assert a.update(0.0) is False


def test_bayes_threshold():
    assert bayes_alert_threshold(1, 9) == pytest.approx(0.1)
    assert math.isclose(bayes_alert_threshold(1, 1), 0.5)


# -- motion-aware score (vehicle_min_speed_mps) ---------------------------
def test_motion_aware_score_ignores_a_stationary_vehicle():
    """A pedestrian walking right past a parked car: close, but no moving-vehicle conflict."""
    parked = held((0, 0), (0, 0), 0.5, 6)
    ped = held((-3, 1.0), (1.3, 0), 0.4, 12)
    assert pairwise_risk(parked, ped, 0.0, 2.5)[0] == pytest.approx(1.0)
    assert pairwise_risk(parked, ped, 0.0, 2.5, vehicle_min_speed_mps=0.5)[0] == 0.0


def test_motion_aware_score_is_unchanged_for_a_moving_vehicle():
    veh = held((-5, 0), (2.5, 0), 0.5, 6)
    ped = held((0, -2.6), (0, 1.3), 0.4, 12)
    assert pairwise_risk(veh, ped, 0.0, vehicle_min_speed_mps=0.5) == pairwise_risk(veh, ped, 0.0)


def test_motion_aware_score_counts_only_moving_instants():
    """The car is closest while stopped (t < 1 s) and only 3 m away once it
    drives off: close-while-stopped must not count."""
    path = np.array([[0, 0], [0, 0], [1.5, 0], [3.0, 0], [4.5, 0], [6.0, 0]], float)   # stops 1 s, then 3 m/s
    veh = HeldPrediction((0.0, 0.0), 0.0, 0.5, [Candidate(path, 1.0)])
    walk = np.array([[0.0, 1.0 + 0.6 * i] for i in range(1, 13)])           # 1.5 m/s away from the car
    ped = HeldPrediction((0.0, 1.0), 0.0, 0.4, [Candidate(walk, 1.0)])
    assert pairwise_risk(veh, ped, 0.0, 2.0)[0] == 1.0
    assert pairwise_risk(veh, ped, 0.0, 2.0, vehicle_min_speed_mps=0.5)[0] == 0.0


def test_motion_aware_score_is_off_by_default_in_the_pipeline():
    from trajectory_risk.pipeline import PipelineConfig
    assert PipelineConfig().risk_vehicle_min_speed_mps is None
