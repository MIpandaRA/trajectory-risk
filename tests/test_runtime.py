import json
import queue
import threading
import time

import numpy as np

from trajectory_risk.apps.generate import write_recording
from trajectory_risk.predictors import ConstantVelocity
from trajectory_risk.predictors.base import PredictionRequest
from trajectory_risk.protocol import Tick, load_recording
from trajectory_risk.runtime.worker import InferenceWorker
from trajectory_risk.sim import WorldConfig, generate
from trajectory_risk.transport import LiveStateClient, LiveStateServer


class SlowPredictor(ConstantVelocity):
    def __init__(self, delay):
        super().__init__(0.5, 5, 6)
        self.delay = delay

    def predict(self, request):
        time.sleep(self.delay)
        return super().predict(request)


def req(aid, k):
    return PredictionRequest(aid, "vehicle", k, k * 0.5, 0.5, np.zeros((7, 2)) + k, np.zeros(7))


def test_slow_inference_supersedes_instead_of_queueing():
    w = InferenceWorker(SlowPredictor(0.2), "vehicle")
    t0 = time.perf_counter()
    for k in range(20):                    # 20 requests for the same agent, much faster than it can serve
        w.submit(req(1, k))
    assert time.perf_counter() - t0 < 0.05  # submit never blocks
    time.sleep(0.6)
    assert w.queued <= 1
    assert w.stats.superseded >= 17         # old windows were dropped, not queued
    results = w.drain()
    assert results and results[-1].sample_index == 19 or w.stats.completed <= 3
    w.close()


def test_worker_survives_a_failing_prediction():
    class Boom(ConstantVelocity):
        def predict(self, request):
            if request.sample_index == 0:
                raise RuntimeError("bad window")
            return super().predict(request)
    w = InferenceWorker(Boom(0.5, 5, 6), "vehicle")
    w.submit(req(1, 0))
    time.sleep(0.05)
    w.submit(req(2, 1))
    time.sleep(0.1)
    assert w.stats.failed == 1 and [r.agent_id for r in w.drain()] == [2]
    w.close()


def test_server_client_round_trip_and_hello_on_connect():
    server = LiveStateServer(port=0, hello={"type": "hello", "map": "T11"})
    client = LiveStateClient(port=server.port)
    deadline = time.time() + 3
    while server.n_clients == 0 and time.time() < deadline:
        server.accept_pending()
        time.sleep(0.01)
    tick = Tick(1, 0.5, [])
    server.broadcast(tick.to_dict())
    got = [client.queue.get(timeout=2), client.queue.get(timeout=2)]
    assert got[0]["type"] == "hello" and Tick.from_dict(got[1]) == tick
    client.close()
    server.close()


def test_generated_recording_round_trips(tmp_path):
    ticks = generate(WorldConfig(num_pedestrians=2, seed=5), 2.0, 0.5)
    p = tmp_path / "s.jsonl.gz"
    write_recording(str(p), ticks, "T11")
    hello, back = load_recording(str(p))
    assert hello["map"] == "T11" and [t.to_dict() for t in back] == [t.to_dict() for t in ticks]


def test_client_queue_drops_oldest_when_full():
    server = LiveStateServer(port=0)
    client = LiveStateClient(port=server.port, maxsize=5)
    while server.n_clients == 0:
        server.accept_pending()
        time.sleep(0.01)
    for i in range(50):
        server.broadcast({"type": "tick", "seq": i})
    time.sleep(0.3)
    seqs = []
    while True:
        try:
            seqs.append(client.queue.get_nowait()["seq"])
        except queue.Empty:
            break
    assert len(seqs) <= 5 and seqs[-1] == 49
    client.close()
    server.close()
