import numpy as np

from trajectory_risk.predictors import ConstantVelocity
from trajectory_risk.predictors.base import PredictionRequest
from trajectory_risk.predictors.clustering import final_position_clustering, kmeans_trajectories


def req(positions):
    p = np.asarray(positions, float)
    return PredictionRequest(1, "vehicle", 10, 5.0, 0.5, p, np.zeros(len(p)))


def test_constant_velocity_toy_case():
    cv = ConstantVelocity(0.5, 1, 3)
    out = cv.predict(req([(0, 0), (1, 0), (2, 0)]))
    assert len(out) == 1 and out[0].confidence == 1.0
    assert np.allclose(out[0].trajectory, [(3, 0), (4, 0), (5, 0)])


def test_velocity_window_averages_recent_steps():
    # steps of 1 m then 3 m: window 1 uses the last step (3 m), window 2 averages to 2 m
    p = [(0, 0), (1, 0), (4, 0)]
    assert np.allclose(ConstantVelocity(0.5, 1, 2).predict(req(p))[0].trajectory, [(7, 0), (10, 0)])
    assert np.allclose(ConstantVelocity(0.5, 1, 2, velocity_window=2).predict(req(p))[0].trajectory, [(6, 0), (8, 0)])


def test_stationary_agent_stays_put():
    out = ConstantVelocity(0.5, 1, 6).predict(req([(5, 5)] * 3))[0].trajectory
    assert out.shape == (6, 2) and np.allclose(out, (5, 5))


def _two_blobs(rng, n=400):
    left = np.cumsum(np.tile([[-1.0, 1.0]], (6, 1)), axis=0)
    right = np.cumsum(np.tile([[1.0, 1.0]], (6, 1)), axis=0)
    a = left + rng.normal(0, 0.05, size=(n // 4, 6, 2))
    b = right + rng.normal(0, 0.05, size=(3 * n // 4, 6, 2))
    return np.concatenate([a, b])


def test_kmeans_finds_both_modes_with_shares():
    rng = np.random.default_rng(0)
    c = kmeans_trajectories(_two_blobs(rng), 2, seed=0)
    assert np.isclose(sum(x.confidence for x in c), 1.0)
    assert np.isclose(c[0].confidence, 0.75) and c[0].trajectory[-1, 0] > 0
    assert np.isclose(c[1].confidence, 0.25) and c[1].trajectory[-1, 0] < 0


def test_fpc_returns_real_samples_and_shares():
    rng = np.random.default_rng(1)
    s = _two_blobs(rng)
    c = final_position_clustering(s, 2, seed=0)
    assert any(np.allclose(c[0].trajectory, x) for x in s)       # an actual draw, not a centroid
    assert np.isclose(c[0].confidence, 0.75)


def test_mean_of_two_modes_is_a_path_nobody_takes():
    """Why we keep several candidates: the average of 'left' and 'right' goes straight."""
    s = _two_blobs(np.random.default_rng(2))
    mean_end = s.mean(axis=0)[-1]
    ends = s[:, -1]
    assert np.min(np.linalg.norm(ends - mean_end, axis=1)) > 2.0


def _stub_contextvae(monkeypatch, map_mode):
    """A real ContextVAEPredictor (real crop geometry) with the network and
    checkpoint stubbed, on a small fake map."""
    import torch
    from trajectory_risk.geometry import EgoFrame
    from trajectory_risk.predictors import contextvae

    class _Net(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    monkeypatch.setattr(contextvae, "ContextVAE", _Net)
    monkeypatch.setattr(contextvae, "load_checkpoint", lambda *a: {"model": {}})
    sem = np.full((3, 500, 500), -1.0, np.float32)
    sem[0, 200:300, :] = 0.5                                   # a road through the middle
    H = np.array([[0.0, -1.0, 250.0], [1.0, 0.0, 250.0], [0.0, 0.0, 1.0]])
    p = contextvae.ContextVAEPredictor("unused", sem, H, map_mode=map_mode)
    return p.crop_map(EgoFrame((0.0, 0.0), 0.3)).numpy()


def test_no_map_ablation_means_no_road_not_half_road(monkeypatch):
    """failures.md #13: the old "blank" ablation filled the map with 0, which
    in the [-1, 1] encoding is "half drivable everywhere", not "no map". The
    no-map arm must be all -1, the encoding's own value for empty space."""
    real = _stub_contextvae(monkeypatch, "real")
    nothing = _stub_contextvae(monkeypatch, "nothing")
    assert (real > 0).any()
    assert np.allclose(nothing, -1.0, atol=1e-6)            # bilinear rotation rounds to 6e-8
    assert np.all(_stub_contextvae(monkeypatch, "blank") == 0.0)   # kept only to reproduce the old row
