import math

import numpy as np
import pytest

from trajectory_risk.geometry import EgoFrame, ScreenFrame, rotate_about, rotate_state, rotation_matrix


def test_rotation_is_orthonormal():
    R = rotation_matrix(0.7)
    assert np.allclose(R @ R.T, np.eye(2))
    assert np.isclose(np.linalg.det(R), 1.0)


def test_rotation_preserves_distance():
    rng = np.random.default_rng(0)
    p = rng.normal(size=(20, 2)) * 10
    q = rotate_about(p, (3.0, -2.0), 1.234)
    assert np.allclose(np.linalg.norm(p[1:] - p[:-1], axis=1), np.linalg.norm(q[1:] - q[:-1], axis=1))


def test_ego_frame_round_trip_and_heading_points_along_x():
    frame = EgoFrame(pivot=(10.0, 5.0), heading=math.radians(120))
    p = np.array([[10.0, 5.0], [10.0 + math.cos(math.radians(120)), 5.0 + math.sin(math.radians(120))]])
    e = frame.to_ego(p)
    assert np.allclose(e[0], (10.0, 5.0))           # pivot stays put
    assert np.allclose(e[1] - e[0], (1.0, 0.0))     # heading now points along +x
    assert np.allclose(frame.to_world(e), p)


def test_matches_original_undo_rotation():
    """The prototype's undo_rotation(), copied verbatim, must agree with EgoFrame.to_world."""
    def undo_rotation(points, pivot, angle):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        d = points - pivot
        x = cos_a * d[..., 0] + sin_a * d[..., 1]
        y = -sin_a * d[..., 0] + cos_a * d[..., 1]
        return np.stack([x, y], axis=-1) + pivot
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(50, 6, 2)) * 20
    frame = EgoFrame(pivot=(3.0, 4.0), heading=0.9)
    assert np.allclose(frame.to_world(pts), undo_rotation(pts, np.array([3.0, 4.0]), frame.angle))


def test_rotate_state_rotates_velocity_without_translating_it():
    st = np.array([[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]])
    out = rotate_state(st, pivot=np.array([1.0, 0.0]), angle=math.pi / 2)
    assert np.allclose(out[0, :2], (1.0, 0.0))
    assert np.allclose(out[0, 2:4], (0.0, 1.0))
    assert np.allclose(out[0, 4:6], (-1.0, 0.0))


def test_screen_world_flips_y_exactly_once():
    sf = ScreenFrame(pixels_per_metre=10.0, canvas_height_px=600)
    x, y, h = sf.to_world(100.0, 0.0, 0.0)       # top-left area of the screen
    assert (x, y) == (10.0, 60.0)                 # ...is the TOP of the world (y up)
    x2, y2, _ = sf.to_world(100.0, 600.0)
    assert y2 == 0.0
    # moving "down" the screen (heading +90 deg in pixels) is moving -y in the world
    _, _, h = sf.to_world(0, 0, math.pi / 2)
    assert h == pytest.approx(-math.pi / 2)
    assert np.allclose(sf.to_screen(*sf.to_world(123.0, 456.0, 0.3)), (123.0, 456.0, 0.3))


def test_scale_100px_is_10m():
    sf = ScreenFrame(10.0, 600)
    a, b = sf.to_world(0, 300), sf.to_world(100, 300)
    assert b[0] - a[0] == pytest.approx(10.0)


def test_world_origin_offset():
    sf = ScreenFrame(2.0, 600, origin_x_m=0.0, origin_y_m=1000.0)
    assert sf.to_world(0, 600)[:2] == (0.0, 1000.0)
