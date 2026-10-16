"""Background inference so the render loop never waits on a model.

Why threads help despite the GIL: PyTorch releases the GIL inside its heavy
operators, so a forward pass on a worker thread runs in parallel with the
pygame loop. Each model gets one worker, and ``torch.set_num_threads`` is
split between them so two forward passes don't oversubscribe the cores.

Overload policy: supersede, don't backlog. If a request for an agent is still
queued when a newer one arrives, the newer one replaces it. A prediction from
an old window is worth less than one from a new window, and a queue that
grows without bound turns a slow model into an ever-increasing delay.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from ..predictors.base import PredictionRequest, PredictionResult


@dataclass
class WorkerStats:
    submitted: int = 0
    superseded: int = 0
    completed: int = 0
    failed: int = 0
    runtimes_ms: List[float] = field(default_factory=list)

    def recent_ms(self, n: int = 50) -> List[float]:
        return self.runtimes_ms[-n:]


class InferenceWorker:
    def __init__(self, predictor, label: str, inject_delay_s: float = 0.0,
                 on_result: Optional[Callable[[PredictionResult], None]] = None):
        self.predictor = predictor
        self.label = label
        self.inject_delay_s = inject_delay_s
        self.stats = WorkerStats()
        self._pending: "OrderedDict[int, PredictionRequest]" = OrderedDict()
        self._results: Dict[int, PredictionResult] = {}
        self._cv = threading.Condition()
        self._stop = False
        self._busy = False
        self._on_result = on_result
        self._thread = threading.Thread(target=self._run, name=f"infer-{label}", daemon=True)
        self._thread.start()

    def submit(self, request: PredictionRequest) -> None:
        """Never blocks."""
        with self._cv:
            self.stats.submitted += 1
            if request.agent_id in self._pending:
                self.stats.superseded += 1
                del self._pending[request.agent_id]
            self._pending[request.agent_id] = request
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                _, req = self._pending.popitem(last=False)
                self._busy = True
            t0 = time.perf_counter()
            try:
                if self.inject_delay_s:
                    time.sleep(self.inject_delay_s)
                cands = self.predictor.predict(req)
                ok = True
            except Exception as e:  # keep the worker alive across one bad window
                print(f"[worker:{self.label}] agent {req.agent_id} failed: {e!r}")
                cands, ok = [], False
            ms = (time.perf_counter() - t0) * 1000.0
            res = PredictionResult(req.agent_id, req.agent_type, req.sample_index, req.anchor_time_s,
                                   req.step_dt_s, cands, getattr(self.predictor, "name", self.label), ms)
            with self._cv:
                self._busy = False
                if ok:
                    self.stats.completed += 1
                    self.stats.runtimes_ms.append(ms)
                    if len(self.stats.runtimes_ms) > 5000:
                        del self.stats.runtimes_ms[:1000]
                    self._results[req.agent_id] = res
                else:
                    self.stats.failed += 1
            if self._on_result is not None and ok:
                self._on_result(res)

    def drain(self) -> List[PredictionResult]:
        with self._cv:
            out = list(self._results.values())
            self._results.clear()
            return out

    @property
    def queued(self) -> int:
        with self._cv:
            return len(self._pending)

    def idle(self) -> bool:
        with self._cv:
            return not self._pending and not self._busy

    def close(self, timeout: float = 3.0) -> None:
        """Stop and wait for an in-flight prediction to finish, so the
        interpreter never shuts down underneath a running torch op."""
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout)


def split_torch_threads(n_workers: int = 2) -> int:
    """Give each inference worker an equal share of the CPU cores."""
    import os

    import torch
    per = max(1, (os.cpu_count() or 2) // max(1, n_workers))
    torch.set_num_threads(per)
    return per
