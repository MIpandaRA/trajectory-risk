"""Equivalence with the original prototype. tests/fixtures/original_wrappers_fixture.npz
was produced by running the prototype's inference.py / social_inference.py
(seeded) on a fixed input; the refactored predictors must reproduce it."""
import os

import numpy as np
import pytest
import torch

from conftest import FIXTURES
from trajectory_risk.maps import MapBundle
from trajectory_risk.predictors.base import PredictionRequest
from trajectory_risk.predictors.kinematics import neighbour_states

FIX = np.load(os.path.join(FIXTURES, "original_wrappers_fixture.npz"))

# The fixture was produced on another machine. torch's CPU kernels (MKL/AVX
# code paths, bilinear grid_sample, BLAS reductions) are not bit-identical
# across CPUs, so exact equality is the wrong test. Observed differences on an
# i7-1355U: map crop up to 6e-5, SocialVAE samples up to 1.3e-5 m. A real
# preprocessing bug (wrong pivot, angle or y-flip) moves values by metres.
CROSS_CPU_ATOL = 1e-4


def vehicle_request():
    ego = FIX["ego"]
    return PredictionRequest(1, "vehicle", 10, 5.0, 0.5, ego, np.full(len(ego), float(FIX["heading"])),
                             [FIX["nb1"], FIX["nb2"]])


@pytest.fixture(scope="module")
def contextvae():
    from trajectory_risk.predictors.factory import vehicle_predictor
    return vehicle_predictor("contextvae", MapBundle.load("T11"))


@pytest.mark.needs_weights
def test_contextvae_preprocessing_matches_original(contextvae):
    req = vehicle_request()
    hist, frame = contextvae.build_history(req)
    nb = neighbour_states(req.neighbours, contextvae.buffer_len, contextvae.ob_horizon)
    hist_e, _ = contextvae.to_ego_frame(hist, nb, frame)
    assert np.allclose(hist_e, FIX["cv_hist_rot"], atol=1e-6)
    assert np.allclose(contextvae.crop_map(frame).numpy(), FIX["cv_map_crop"], rtol=0, atol=CROSS_CPU_ATOL)


@pytest.mark.needs_weights
def test_contextvae_samples_match_original(contextvae):
    torch.manual_seed(0)
    raw = contextvae.raw_samples_world(vehicle_request())
    assert np.abs(raw - FIX["cv_raw_world"]).max() < 1e-4


@pytest.mark.needs_weights
def test_contextvae_candidates_well_formed(contextvae):
    c = contextvae.predict(vehicle_request())
    assert len(c) == 10 and np.isclose(sum(x.confidence for x in c), 1.0)
    assert all(a.confidence >= b.confidence for a, b in zip(c, c[1:]))
    assert c[0].trajectory.shape == (6, 2)


@pytest.mark.needs_weights
def test_socialvae_samples_match_original():
    from trajectory_risk.predictors.factory import pedestrian_predictor
    sv = pedestrian_predictor("socialvae")
    ped = FIX["ped"]
    req = PredictionRequest(2, "pedestrian", 10, 4.0, 0.4, ped, np.zeros(len(ped)), [FIX["ped_nb"]])
    torch.manual_seed(0)
    assert np.allclose(sv.raw_samples_world(req), FIX["sv_raw"], rtol=0, atol=CROSS_CPU_ATOL)
    c = sv.predict(req)
    assert len(c) == 20 and c[0].trajectory.shape == (12, 2)
