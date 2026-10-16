import os

import numpy as np
import pytest

from trajectory_risk.maps import MapBundle

from trajectory_risk.sim import World, WorldConfig, generate


def test_same_seed_same_scenario():
    a = generate(WorldConfig(num_pedestrians=4, seed=7), 5.0, 1.0)
    b = generate(WorldConfig(num_pedestrians=4, seed=7), 5.0, 1.0)
    assert [t.to_dict() for t in a] == [t.to_dict() for t in b]


def test_time_is_exact_fixed_step():
    ticks = generate(WorldConfig(seed=1), 2.0, 0.0)
    assert np.allclose(np.diff([t.t_s for t in ticks]), 1 / 60)


def test_ticks_are_valid_world_frame():
    w = World(WorldConfig(seed=2))
    for _ in range(300):
        w.step()
    t = w.tick()
    t.validate()
    b = w.bundle
    inside = [a for a in t.agents if b.x_min <= a.x_m <= b.x_min + b.width_m and b.y_min <= a.y_m <= b.y_min + b.height_m]
    assert len(inside) >= 1


@pytest.mark.parametrize("m", ["T11", "OneNorthRoundabout"])
def test_pedestrian_speed_is_metres_per_second_on_every_map(m):
    if not os.path.isfile(MapBundle.load(m).semantic_map_path):
        pytest.skip(f"{m} semantic map not downloaded")
    w = World(WorldConfig(map_name=m, num_vehicles=1, num_pedestrians=3, seed=3))
    speeds = {round(a.speed_mps, 3) for a in w.tick().agents if a.type == "pedestrian"}
    assert speeds == {1.3}, (m, speeds)


def test_vehicle_heading_matches_motion_direction():
    """If the y flip were applied twice, heading and displacement would disagree."""
    ticks = generate(WorldConfig(num_pedestrians=0, seed=4), 6.0, 2.0)
    a0 = {a.id: a for a in ticks[0].agents}
    a1 = {a.id: a for a in ticks[30].agents}
    for i in set(a0) & set(a1):
        d = np.array([a1[i].x_m - a0[i].x_m, a1[i].y_m - a0[i].y_m])
        if np.linalg.norm(d) < 0.5:
            continue
        h = np.array([np.cos(a0[i].heading_rad), np.sin(a0[i].heading_rad)])
        assert d @ h / np.linalg.norm(d) > 0.8
