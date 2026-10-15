"""
Wire format between the producer (simulator / replayer / real tracker)
and the consumer (dashboard, recorder).

Framing: one JSON object per line, UTF-8, newline terminated (JSON Lines).
Two message types:

    {"type": "hello", "schema_version": 2, "map": "T11", "tick_rate_hz": 60, "source": "simulator"}
    {"type": "tick", "schema_version": 2, "seq": 120, "t_s": 2.0,
     "agents": [{"id": 3, "type": "vehicle", "x_m": 26.97, "y_m": 31.53,
                 "heading_rad": 1.57, "speed_mps": 2.5}, ...]}

Conventions (the whole system relies on these, so they live in one place):

* positions are WORLD metres, x right, y UP (never screen pixels);
* heading is radians, math convention (0 = +x, counter-clockwise positive);
* ``t_s`` is SIMULATION time in seconds and is monotonic. Consumers use it,
  never their own wall clock, for decimation and prediction timing, which
  is what makes a recorded run replay deterministically;
* ``id`` is stable for the lifetime of an agent and unique across types.

Schema v1 (the original prototype) used ``x``/``y``/``heading``/``speed``,
``tick`` and a ``t`` derived from the tick count. ``Tick.from_dict`` still
reads v1 so old recordings keep working.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, List, Optional

SCHEMA_VERSION = 2
AGENT_TYPES = ("vehicle", "pedestrian")


class ProtocolError(ValueError):
    """ A message that must not reach the rest of the pipeline."""


@dataclass(frozen=True)
class AgentState:
    id: int
    type: str
    x_m: float
    y_m: float
    heading_rad: float
    speed_mps: float

    def validate(self) -> None:
        if self.type not in AGENT_TYPES:
            # Never silently treat an unknown type as a vehicle: the two types
            # go to different models with different preprocessing.
            raise ProtocolError(f"agent {self.id}: unknown type {self.type!r}")
        for name in ("x_m", "y_m", "heading_rad", "speed_mps"):
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ProtocolError(f"agent {self.id}: {name} must be a finite number, got {v!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "AgentState":
        try:
            if "x_m" in d:
                a = cls(int(d["id"]), str(d["type"]), d["x_m"], d["y_m"], d["heading_rad"], d["speed_mps"])
            else:  # schema v1
                a = cls(int(d["id"]), str(d["type"]), d["x"], d["y"], d["heading"], d["speed"])
        except KeyError as e:
            raise ProtocolError(f"agent record missing field {e.args[0]!r}: {d}") from None
        a.validate()
        return a


@dataclass(frozen=True)
class Tick:
    seq: int
    t_s: float
    agents: List[AgentState] = field(default_factory=list)

    def validate(self) -> None:
        if not math.isfinite(self.t_s):
            raise ProtocolError(f"tick {self.seq}: t_s is not finite")
        ids = [a.id for a in self.agents]
        if len(ids) != len(set(ids)):
            raise ProtocolError(f"tick {self.seq}: duplicate agent ids {sorted(ids)}")

    def to_dict(self) -> dict:
        return {"type": "tick", "schema_version": SCHEMA_VERSION, "seq": self.seq,
                "t_s": round(self.t_s, 6), "agents": [_agent_dict(a) for a in self.agents]}

    @classmethod
    def from_dict(cls, d: dict, tick_rate_hz: float = 60.0) -> "Tick":
        if d.get("type") != "tick":
            raise ProtocolError(f"not a tick message: type={d.get('type')!r}")
        if "seq" in d:
            seq, t_s = int(d["seq"]), float(d["t_s"])
        else:  # schema v1
            seq = int(d["tick"])
            t_s = float(d.get("t", seq / tick_rate_hz))
        tick = cls(seq, t_s, [AgentState.from_dict(a) for a in d.get("agents", [])])
        tick.validate()
        return tick


def _agent_dict(a: AgentState) -> dict:
    d = asdict(a)
    for k in ("x_m", "y_m", "heading_rad", "speed_mps"):
        d[k] = round(float(d[k]), 5)
    return d


def hello(map_name: str, tick_rate_hz: float = 60.0, source: str = "simulator") -> dict:
    return {"type": "hello", "schema_version": SCHEMA_VERSION, "map": map_name,
            "tick_rate_hz": tick_rate_hz, "source": source}


def encode(message: dict) -> bytes:
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


class SequenceChecker:
    """Rejects ticks whose simulation time goes backwards (a replay restart,
    a second producer on the same port, a bug). Equal times are allowed."""

    def __init__(self) -> None:
        self.last_t: Optional[float] = None

    def check(self, tick: Tick) -> bool:
        if self.last_t is not None and tick.t_s < self.last_t:
            return False
        self.last_t = tick.t_s
        return True

    def reset(self) -> None:
        self.last_t = None


def read_jsonl(path: str) -> Iterable[dict]:
    """Yield messages from a recording. Accepts plain or .gz files, and both
    raw lines and the {"recv_t": ..., "msg": {...}} wrapper the recorder writes."""
    import gzip
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["msg"] if "msg" in rec and "type" not in rec else rec


def load_recording(path: str):
    """-> (hello dict or None, list[Tick]) with validation applied."""
    hello_msg, ticks = None, []
    rate = 60.0
    for msg in read_jsonl(path):
        if msg.get("type") == "hello" and hello_msg is None:
            hello_msg = msg
            rate = float(msg.get("tick_rate_hz", 60.0))
        elif msg.get("type") == "tick":
            ticks.append(Tick.from_dict(msg, rate))
    return hello_msg, ticks
