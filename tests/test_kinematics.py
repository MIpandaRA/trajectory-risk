import numpy as np

from trajectory_risk.predictors.kinematics import SENTINEL, extend, neighbour_states


def test_backward_differences_by_hand():
    p = np.array([[0, 0], [1, 0], [3, 0], [6, 1]], dtype=float)
    s = extend(p)
    assert np.allclose(s[1:, 2:4], [[1, 0], [2, 0], [3, 1]])   # per-step velocity
    assert np.allclose(s[2:, 4:6], [[1, 0], [1, 1]])           # change in per-step velocity


def test_velocity_is_per_step_not_per_second():
    """2.5 m/s sampled every 0.5 s is 1.25 m per step: what the models were trained on."""
    p = np.array([[0.0, 0.0], [1.25, 0.0], [2.5, 0.0]])
    assert np.isclose(extend(p)[-1, 2], 1.25)


def test_absent_neighbour_steps_use_sentinel():
    n = np.array([[np.nan, np.nan], [np.nan, np.nan], [0, 0], [1, 0], [2, 0], [3, 0], [4, 0]], dtype=float)
    st = neighbour_states([n], buffer_len=7, ob_horizon=5)
    full = neighbour_states([n], buffer_len=7, ob_horizon=7)
    assert np.all(full[:2, 0] == SENTINEL)        # absent steps are the sentinel
    assert st.shape == (5, 1, 6)
    assert np.allclose(st[0, 0, :2], [0, 0])      # first present step keeps its position...
    assert abs(st[0, 0, 2]) > 1e8                 # ...with a huge velocity, as upstream does
    assert np.allclose(st[-1, 0, :4], [4, 0, 1, 0])


def test_no_neighbours_gives_empty_axis():
    assert neighbour_states([], 7, 5).shape == (5, 0, 6)
