import json
import math

import pytest

from trajectory_risk.protocol import AgentState, ProtocolError, SequenceChecker, Tick, load_recording


def _tick(**over):
    d = {"type": "tick", "schema_version": 2, "seq": 3, "t_s": 0.05,
         "agents": [{"id": 1, "type": "vehicle", "x_m": 1.0, "y_m": 2.0, "heading_rad": 0.1, "speed_mps": 2.5}]}
    d.update(over)
    return d


def test_round_trip():
    t = Tick.from_dict(_tick())
    assert Tick.from_dict(json.loads(json.dumps(t.to_dict()))) == t


def test_reads_schema_v1():
    v1 = {"type": "tick", "tick": 120, "t": 2.0, "dt": 0.016,
          "agents": [{"id": 4, "type": "pedestrian", "x": 1, "y": 2, "heading": 0, "speed": 1.3}]}
    t = Tick.from_dict(v1)
    assert t.seq == 120 and t.t_s == 2.0 and t.agents[0].x_m == 1


def test_missing_field_is_named():
    d = _tick()
    del d["agents"][0]["x_m"]
    with pytest.raises(ProtocolError, match="x"):
        Tick.from_dict(d)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "3.0", None])
def test_non_finite_or_non_numeric_rejected(bad):
    d = _tick()
    d["agents"][0]["y_m"] = bad
    with pytest.raises(ProtocolError):
        Tick.from_dict(d)


def test_unknown_type_is_rejected_not_treated_as_vehicle():
    d = _tick()
    d["agents"][0]["type"] = "bicycle"
    with pytest.raises(ProtocolError, match="unknown type"):
        Tick.from_dict(d)


def test_duplicate_ids_rejected():
    d = _tick()
    d["agents"].append(dict(d["agents"][0]))
    with pytest.raises(ProtocolError, match="duplicate"):
        Tick.from_dict(d)


def test_sequence_checker_rejects_time_going_backwards():
    sc = SequenceChecker()
    assert sc.check(Tick(1, 1.0)) and sc.check(Tick(2, 1.0)) and not sc.check(Tick(3, 0.5))


def test_load_recording_reads_both_wrappers(tmp_path):
    p = tmp_path / "r.jsonl"
    lines = [json.dumps({"recv_t": 1.0, "msg": {"type": "hello", "tick_rate_hz": 60}}),
             json.dumps({"recv_t": 1.1, "msg": _tick()}), json.dumps(_tick(seq=4, t_s=0.07))]
    p.write_text("\n".join(lines))
    hello, ticks = load_recording(str(p))
    assert hello["tick_rate_hz"] == 60 and [t.seq for t in ticks] == [3, 4]
    assert math.isclose(ticks[1].t_s, 0.07)


def test_agent_state_validate_direct():
    with pytest.raises(ProtocolError):
        AgentState(1, "vehicle", 0.0, float("nan"), 0.0, 0.0).validate()


# Every malformed message must raise ProtocolError, never another error type:
# the dashboard catches ProtocolError to drop a bad tick and keep running.
def _without(key):
    d = _tick()
    del d[key]
    return d


def _agent(**over):
    d = _tick()
    d["agents"][0].update(over)
    return d


@pytest.mark.parametrize("msg", [
    _agent(id="abc"),
    _agent(id=None),
    _without("seq"),
    _without("t_s"),
    _tick(seq="three"),
    _tick(agents=None),
    _tick(agents="x"),
    _tick(agents=[5]),
    "not a dict",
], ids=["id-text", "id-null", "v2-missing-seq", "missing-t_s", "seq-text", "agents-null",
        "agents-string", "agent-not-object", "message-not-object"])
def test_malformed_messages_raise_protocol_error(msg):
    with pytest.raises(ProtocolError):
        Tick.from_dict(msg)


def test_v2_message_missing_seq_is_not_misread_as_v1():
    with pytest.raises(ProtocolError, match="'seq'"):
        Tick.from_dict(_without("seq"))
