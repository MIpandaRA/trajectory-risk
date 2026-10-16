import numpy as np

from trajectory_risk.protocol import AgentState, Tick
from trajectory_risk.tracking import Decimator, TrackStore


def ticks_for(agents_fn, seconds, rate=60.0):
    return [Tick(i, i / rate, agents_fn(i / rate)) for i in range(int(seconds * rate))]


def straight(aid=1, kind="vehicle", v=2.5, y=0.0):
    return lambda t: [AgentState(aid, kind, v * t, y, 0.0, v)]


def test_decimation_hits_2hz_and_2p5hz_from_60hz():
    for period, expected in ((0.5, 20), (0.4, 25)):
        d = Decimator(period)
        taken = [t.t_s for t in ticks_for(lambda t: [], 10.0) if d.index_for(t.t_s) is not None]
        assert len(taken) == expected
        assert np.allclose(np.diff(taken), period)


def test_decimation_survives_irregular_tick_times():
    """Jittered producer: samples still land on the grid, never twice per slot."""
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.9, 1.1, size=600) / 60.0)
    d = Decimator(0.5)
    idx = [d.index_for(x) for x in t]
    taken = [i for i in idx if i is not None]
    assert len(taken) == len(set(taken))
    assert len(taken) >= 15


def test_window_requires_full_contiguous_buffer():
    store = TrackStore("vehicle", 0.5, buffer_len=7)
    for tick in ticks_for(straight(), 3.0):           # 6 samples: not full yet
        store.ingest(tick)
    assert store.tracks[1].window() is None
    for tick in ticks_for(straight(), 3.6)[180:]:      # 7th sample
        store.ingest(tick)
    assert len(store.tracks[1].window()) == 7


def test_gap_invalidates_window():
    store = TrackStore("vehicle", 0.5, buffer_len=4, stale_after_s=5.0)
    for tick in ticks_for(lambda t: [] if 1.2 < t < 1.8 else straight()(t), 4.0):
        store.ingest(tick)
    w = store.tracks[1].window()
    assert w is not None and [s.index for s in w] == [4, 5, 6, 7]  # only after the gap is refilled


def test_types_are_kept_apart():
    both = lambda t: straight(1, "vehicle")(t) + straight(2, "pedestrian", 1.3, 5.0)(t)
    veh, ped = TrackStore("vehicle", 0.5, 7), TrackStore("pedestrian", 0.4, 10)
    for tick in ticks_for(both, 2.0):
        veh.ingest(tick)
        ped.ingest(tick)
    assert set(veh.tracks) == {1} and set(ped.tracks) == {2}


def test_stale_tracks_are_dropped():
    store = TrackStore("vehicle", 0.5, 7, stale_after_s=1.0)
    for tick in ticks_for(lambda t: straight()(t) if t < 2.0 else [], 4.0):
        store.ingest(tick)
    assert 1 not in store.tracks


def test_request_schedule_and_contents():
    store = TrackStore("vehicle", 0.5, buffer_len=7, predict_every=2)
    two = lambda t: straight(1)(t) + straight(2, y=3.0)(t)
    reqs = []
    for tick in ticks_for(two, 6.0):
        store.ingest(tick)
        reqs += store.due_requests()
    first = [r for r in reqs if r.agent_id == 1]
    assert [r.sample_index for r in first] == [6, 8, 10]  # every 2 samples once the buffer is full
    r = first[0]
    assert r.positions.shape == (7, 2) and np.allclose(np.diff(r.positions[:, 0]), 1.25)
    assert r.anchor_time_s == 3.0 and len(r.neighbours) == 1
