"""DLP conversion, on a tiny hand-made recording (not the real files).

The fixture is 2 s at DLP's 25 Hz: a car driving +x, a car reversing then
parked for the second half, a pedestrian whose heading wraps through +/-pi,
a cyclist, an "Undefined" agent and one never-moving parked car (obstacle).
"""
import gzip
import json
import math
import os

import numpy as np
import pytest

from trajectory_risk.apps import dlp_convert
from trajectory_risk.datasets import dlp, dlp_map
from trajectory_risk.maps import MapBundle
from trajectory_risk.pipeline import LivePipeline
from trajectory_risk.predictors import ConstantVelocity, PredictionRequest
from trajectory_risk.predictors.static_neighbours import StaticNeighbours
from trajectory_risk.protocol import load_recording
from trajectory_risk.tracking import Decimator

DT = 0.04
N_FRAMES = 51                          # t = 0.00 .. 2.00 s
OBSTACLE = (60.0, 40.0)


def _states(label, t):
    """(x, y, heading, speed) for each fixture agent at time t."""
    if label == "car":
        return 30.0 + 2.0 * t, 30.0, 0.0, 2.0
    if label == "reverser":            # faces +x, backs up at 1 m/s, stops at t = 1
        tt = min(t, 1.0)
        return 50.0 - tt, 32.0, 0.0, 1.0 if t < 1.0 else 0.0
    if label == "ped":                 # walks -x; heading straddles the +/-pi seam
        h = 3.1 if int(round(t / DT)) % 2 == 0 else -3.1
        return 40.0 - 1.2 * t, 45.0, h, 1.2
    if label == "bike":
        return 20.0 + 4.0 * t, 47.0, 0.0, 4.0
    return 70.0, 20.0, 0.0, 0.0        # undefined


FIXTURE_AGENTS = {"car": "Car", "reverser": "Medium Vehicle", "ped": "Pedestrian",
                  "bike": "Bicycle", "undef": "Undefined"}


def write_fixture(raw_dir, name="DJI_TEST"):
    os.makedirs(raw_dir, exist_ok=True)
    frames, instances, agents = {}, {}, {}
    for tok, label in FIXTURE_AGENTS.items():
        agents[tok] = {"agent_token": tok, "scene_token": "s", "type": label, "size": [4.5, 1.9],
                       "first_instance": "", "last_instance": ""}
    for i in range(N_FRAMES):
        t = round(i * DT, 2)
        ftok = f"f{i}"
        frames[ftok] = {"frame_token": ftok, "scene_token": "s", "timestamp": t, "instances": []}
        for tok in FIXTURE_AGENTS:
            x, y, h, s = _states(tok, t)
            itok = f"{tok}{i}"
            instances[itok] = {"instance_token": itok, "agent_token": tok, "frame_token": ftok,
                               "coords": [x, y], "heading": h, "speed": s, "acceleration": [0, 0],
                               "mode": "", "prev": "", "next": ""}
            frames[ftok]["instances"].append(itok)
    obstacles = {"o1": {"obstacle_token": "o1", "scene_token": "s", "type": "Car", "size": [4.6, 2.0],
                        "coords": list(OBSTACLE), "heading": math.pi / 2}}
    scene = {"scene_token": "s", "filename": name, "timestamp": "2020-01-01 00:00:00",
             "first_frame": "f0", "last_frame": f"f{N_FRAMES - 1}", "agents": list(agents),
             "obstacles": list(obstacles)}
    for part, obj in (("scene", scene), ("agents", agents), ("frames", frames),
                      ("instances", instances), ("obstacles", obstacles)):
        with open(os.path.join(raw_dir, f"{name}_{part}.json"), "w") as f:
            json.dump(obj, f)
    return name


@pytest.fixture
def raw(tmp_path):
    name = write_fixture(str(tmp_path / "raw"))
    return dlp.load_raw(str(tmp_path / "raw"), name)


def _ids_by_label(conv):
    return {tr.label: tr.id for tr in conv.tracks}


# -- resampling ----------------------------------------------------------
def test_ticks_land_exactly_on_the_60hz_grid(raw):
    ticks = list(dlp.iter_ticks(dlp.convert(raw)))
    assert [t.seq for t in ticks] == list(range(121))            # 2.0 s * 60 + 1
    assert all(t.t_s == t.seq / 60 for t in ticks)


@pytest.mark.parametrize("period", [0.5, 0.4])
def test_decimator_gets_every_model_sample(raw, period):
    """The reason for resampling: at 25 Hz a 0.5 s grid point has no frame
    within half a 60 Hz tick every other second."""
    dec = Decimator(period)
    hits = [k for t in dlp.iter_ticks(dlp.convert(raw)) if (k := dec.index_for(t.t_s)) is not None]
    assert hits == list(range(int(2.0 / period + 1e-9) + 1))


def test_positions_are_linearly_interpolated(raw):
    conv = dlp.convert(raw)
    car = next(tr for tr in conv.tracks if tr.label == "Car")
    assert np.allclose(car.x, 30.0 + 2.0 * car.k / 60, atol=1e-9)


def test_heading_is_unwrapped_before_interpolating():
    k, s = dlp.resample(np.array([0.0, 0.04]), heading=np.array([3.1, -3.1]))
    mid = s["heading"][1]                                       # t = 1/60, ~40% of the way
    assert abs(abs(mid) - math.pi) < 0.1                        # through pi, not through 0
    assert np.all(s["heading"] >= -math.pi) and np.all(s["heading"] < math.pi)


def test_heading_from_velocity_holds_when_nearly_stationary():
    t = np.arange(0, 3.0, DT)
    x = np.where(t < 1.0, t * 1.0, 1.0)                         # walk +x for 1 s, then stand
    y = np.where(t < 1.0, t * 1.0, 1.0) + np.random.default_rng(0).normal(0, 0.005, len(t)) * (t >= 1.0)
    h = dlp.heading_from_velocity(t, x, y)
    assert np.allclose(h[(t > 0.3) & (t < 0.7)], math.pi / 4, atol=0.05)
    assert np.allclose(h[t > 1.5], h[np.searchsorted(t, 1.5)])  # held, not spun by jitter


# -- units / axes / types ------------------------------------------------
def test_dlp_frame_is_world_metres_y_up():
    x, y, h = dlp.dlp_to_world([3.0], [4.0], [0.5])
    assert (x[0], y[0], h[0]) == (3.0, 4.0, 0.5)


def test_y_down_source_is_mirrored_with_its_headings():
    x, y, h = dlp.dlp_to_world([3.0], [10.0], [0.5], metres_per_unit=0.1, y_up=False, height_units=80.0)
    assert np.allclose([x[0], y[0], h[0]], [0.3, 7.0, -0.5])


@pytest.mark.parametrize("label,cyclists,expected", [
    ("Car", "drop", "vehicle"), ("Medium Vehicle", "drop", "vehicle"), ("Bus", "drop", "vehicle"),
    ("Motorcycle", "drop", "vehicle"), ("Pedestrian", "drop", "pedestrian"),
    ("Bicycle", "drop", None), ("Bicycle", "pedestrian", "pedestrian"), ("Bicycle", "vehicle", "vehicle"),
    ("Undefined", "drop", None)])
def test_type_mapping(label, cyclists, expected):
    assert dlp.protocol_type(label, cyclists) == expected


def test_unknown_label_raises_instead_of_guessing():
    with pytest.raises(ValueError, match="unknown DLP agent label"):
        dlp.protocol_type("Tractor")


def test_cyclists_are_dropped_by_default_and_kept_on_request(raw):
    assert "Bicycle" not in _ids_by_label(dlp.convert(raw))
    conv = dlp.convert(raw, cyclists="pedestrian")
    assert next(tr.type for tr in conv.tracks if tr.label == "Bicycle") == "pedestrian"
    assert conv.meta["counts"]["Bicycle"] == {"agents": 1, "as": "pedestrian"}


# -- parked cars, stopped agents, flags ----------------------------------
def test_parked_obstacles_never_enter_the_stream(raw):
    conv = dlp.convert(raw)
    ticks = list(dlp.iter_ticks(conv))
    seen = {a.id for t in ticks for a in t.agents}
    assert seen == {tr.id for tr in conv.tracks} and len(seen) == 3   # car, reverser, pedestrian
    assert all(math.hypot(a.x_m - OBSTACLE[0], a.y_m - OBSTACLE[1]) > 1 for t in ticks for a in t.agents)
    assert np.allclose(dlp.parked_cars_xy(conv.meta), [OBSTACLE])


def test_stopped_agents_stay_in_the_stream(raw):
    conv = dlp.convert(raw)
    rid = _ids_by_label(conv)["Medium Vehicle"]
    last = list(dlp.iter_ticks(conv))[-1]
    rev = next(a for a in last.agents if a.id == rid)
    assert rev.speed_mps == 0.0 and rev.x_m == pytest.approx(49.0)


def test_reversing_is_flagged_for_vehicles_only(raw):
    meta = dlp.convert(raw).meta["agents"]
    by_label = {m["label"]: m for m in meta.values()}
    (a, b), = by_label["Medium Vehicle"]["reversing_intervals_s"]
    assert a <= 0.04 and 0.8 <= b <= 1.2
    assert by_label["Car"]["reversing_intervals_s"] == []
    assert by_label["Pedestrian"]["reversing_intervals_s"] == []
    assert dlp.in_intervals([0.5, 1.5], [[a, b]]).tolist() == [True, False]


def test_samples_outside_the_mapped_area_are_counted(raw):
    assert dlp.convert(raw).meta["samples_outside_mapped_area"] == 0
    raw.tracks[0].x[:3] = dlp.MAPPED_X_M[0] - 1.0
    assert dlp.convert(raw).meta["samples_outside_mapped_area"] == 3


def test_tracks_cut_by_the_recording_are_censored(raw):
    meta = dlp.convert(raw).meta
    assert all(m["censored_start"] and m["censored_end"] for m in meta["agents"].values())
    assert meta["fov_x_m"][0] < meta["fov_x_m"][1]


# -- CLI output ----------------------------------------------------------
def test_cli_writes_a_replayable_validated_recording(tmp_path):
    name = write_fixture(str(tmp_path / "raw"))
    args = [name, "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "out")]
    dlp_convert.main(args)
    path = tmp_path / "out" / f"{name}.jsonl.gz"
    hello, ticks = load_recording(str(path))                  # validates every tick
    assert hello["map"] == "DLP" and hello["tick_rate_hz"] == 60.0 and len(ticks) == 121
    first = path.read_bytes()
    dlp_convert.main(args)
    assert path.read_bytes() == first                         # deterministic

    pipe = LivePipeline(ConstantVelocity(0.5, 5, 6), ConstantVelocity(0.4, 8, 12))
    for t in ticks:
        pipe.step_sync(t)
    veh = pipe.stores["vehicle"].tracks
    assert {len(tr.samples) for tr in veh.values()} == {5}   # 0, 0.5, ..., 2.0 s: nothing missed
    with open(tmp_path / "out" / f"{name}.meta.json") as f:
        assert len(json.load(f)["obstacles"]) == 1


# -- static neighbours ---------------------------------------------------
class _Spy:
    name, step_dt_s, ob_horizon, pred_horizon, buffer_len = "spy", 0.5, 5, 6, 7

    def __init__(self):
        self.requests = []

    def predict(self, request):
        self.requests.append(request)
        return []


def test_static_neighbours_are_only_ever_neighbours():
    spy = _Spy()
    pred = StaticNeighbours(spy, np.array([[10.0, 0.0], [100.0, 0.0]]), radius_m=30.0)
    req = PredictionRequest(1, "vehicle", 6, 3.0, 0.5, np.zeros((7, 2)), np.zeros(7), [np.ones((7, 2))])
    pred.predict(req)
    sent = spy.requests[0]
    assert len(req.neighbours) == 1                            # caller's request untouched
    assert len(sent.neighbours) == 2 and np.all(sent.neighbours[1] == [10.0, 0.0])
    assert sent.agent_id == 1 and np.all(sent.positions == req.positions)


# -- map -----------------------------------------------------------------
@pytest.fixture
def fixture_map(raw, tmp_path):
    out = tmp_path / "maps" / "DLP"
    dlp_map.build_bundle([raw], str(out))
    return MapBundle.load("DLP", str(tmp_path / "maps"))


def test_map_channels_follow_the_nuscenes_convention(fixture_map):
    sem, H = fixture_map.semantic
    m = dlp_map.MARGIN_M
    assert sem.shape == (3, 125 + 2 * m, 180 + 2 * m)                             # mapped area + margin
    assert set(np.unique(sem[0])) == {-1.0, 0.5}
    assert np.all(sem[1:] == -1.0)
    assert fixture_map.is_drivable(np.array([[31.0, 30.0], [33.0, 30.3]])).all()   # the car's path
    assert fixture_map.is_drivable(np.array([OBSTACLE])).all()                    # a parked car's stall
    assert not fixture_map.is_drivable(np.array([[100.0, 70.0]])).any()           # never driven


def _crop_map_predictor(bundle, monkeypatch):
    """A real ContextVAEPredictor (so ext and crop geometry are the shipped
    ones) with the network and checkpoint stubbed out."""
    pytest.importorskip("torch")
    import torch
    from trajectory_risk.predictors import contextvae

    class _Net(torch.nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    monkeypatch.setattr(contextvae, "ContextVAE", _Net)
    monkeypatch.setattr(contextvae, "load_checkpoint", lambda *a: {"model": {}})
    return contextvae.load_contextvae("unused", bundle)


def _real_dlp_map():
    try:
        b = MapBundle.load("DLP")
        b.semantic
        return b
    except (FileNotFoundError, OSError):
        return None


@pytest.mark.parametrize("which", ["fixture", "maps/DLP"])
def test_crop_map_never_zero_pads_anywhere_in_the_mapped_area(which, fixture_map, monkeypatch):
    """Zero is "half drivable" in this encoding, so a crop that pads would
    invent road. The pad is monotonic in the pivot's row and column, so the
    mapped area's corners and edge midpoints bound every point inside it
    (the mapped area contains the lot and every agent position seen)."""
    bundle = fixture_map if which == "fixture" else _real_dlp_map()
    if bundle is None:
        pytest.skip("maps/DLP not built (traj-dlp-convert ... --build-map)")
    pred = _crop_map_predictor(bundle, monkeypatch)
    from trajectory_risk.geometry import EgoFrame
    from trajectory_risk.predictors import contextvae

    def no_pad(*a, **k):
        raise AssertionError(f"crop_map padded: {a[1]}")
    monkeypatch.setattr(contextvae.F, "pad", no_pad)
    (x0, x1), (y0, y1) = dlp.MAPPED_X_M, dlp.MAPPED_Y_M
    for x in (x0, (x0 + x1) / 2, x1 - 1e-3):
        for y in (y0, (y0 + y1) / 2, y1 - 1e-3):
            for heading in (0.0, 2.0):
                crop = pred.crop_map(EgoFrame((x, y), heading))
                assert crop.shape[-2:] == (224, 224)


# -- splits and map provenance -------------------------------------------
def _script(name):
    import importlib.util
    path = os.path.join(os.path.dirname(__file__), "..", "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_is_the_fixed_seed_0_permutation():
    """data/dlp/splits.json was written from this before any model ran; if
    numpy ever changes default_rng, this fails instead of the split moving."""
    s = _script("make_dlp_splits").assign([f"DJI_{i:04d}" for i in range(1, 17)], seed=0, n_calibration=10)
    assert s["test"] == ["DJI_0002", "DJI_0007", "DJI_0009", "DJI_0010", "DJI_0014", "DJI_0016"]
    assert len(s["calibration"]) == 10 and not set(s["calibration"]) & set(s["test"])


def test_map_built_from_a_split_records_it_and_excludes_the_rest(tmp_path):
    raw = tmp_path / "raw"
    write_fixture(str(raw), "DJI_A")
    write_fixture(str(raw), "DJI_B")
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"seed": 0, "calibration": ["DJI_A"], "test": ["DJI_B"]}))
    dlp_convert.main(["--split", "calibration", "--splits", str(splits), "--raw", str(raw), "--build-map",
                      "--no-convert", "--maps-dir", str(tmp_path / "maps")])
    meta = MapBundle.load("DLP", str(tmp_path / "maps")).meta
    assert meta["built_from"] == ["DJI_A"] and meta["built_from_split"] == "calibration"
    assert meta["excluded_recordings"] == ["DJI_B"] and meta["split_seed"] == 0
