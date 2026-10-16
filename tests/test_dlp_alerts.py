"""The DLP demo alert simulation (eval/dlp_alerts.simulate) on a hand-made pair."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eval.dlp_alerts import check_option_levels, level_used, simulate, snap_levels  # noqa: E402


def one_pair(probs, positive=()):
    """One vehicle-pedestrian pair at 5 Hz; label positive at the given update indices."""
    n = len(probs)
    d = np.full(n, 10.0)
    d[list(positive)] = 1.0                       # d_min_moving < 2.5 -> positive
    D = {"t_s": np.arange(n) * 0.2, "veh": np.ones(n, int), "ped": np.full(n, 2), "recording_index": np.zeros(n, int),
         "names": ["REC"], "d_min_moving": d}
    return D, np.asarray(probs, float)


def test_two_consecutive_updates_are_needed_to_enter():
    """A single spike enters with enter_ticks=1 but not with 2; two in a row enter both."""
    D, p = one_pair([0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    one = simulate(D, p, 0.35, 0.35 / 3, enter_ticks=1)
    two = simulate(D, p, 0.35, 0.35 / 3, enter_ticks=2)
    assert [o["t"] for o in one["onset_rows"]] == pytest.approx([0.2, 1.4])
    assert [o["t"] for o in two["onset_rows"]] == pytest.approx([1.6])            # fires on the second update in a row
    assert two["false_alerts"].sum() == 1 and one["false_alerts"].sum() == 2


def test_alert_on_during_an_encounter_counts_it_as_caught():
    D, p = one_pair([0.0, 0.0, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], positive=(2, 3, 4))
    s = simulate(D, p, 0.35, 0.35 / 3, enter_ticks=2)
    assert s["events"].sum() == 1 and s["caught"].sum() == 1 and s["false_alerts"].sum() == 0
    assert s["encounters"][0]["first_alert_t"] == pytest.approx(0.6)


def test_a_gap_in_a_pair_restarts_the_streak():
    """Updates 0.2 s apart count as consecutive; a gap does not."""
    D, p = one_pair([0.5, 0.5])
    D["t_s"] = np.array([0.0, 1.0])
    assert simulate(D, p, 0.35, 0.35 / 3, enter_ticks=2)["onsets"] == 0


# -- failures.md #14: step-function scores and threshold labels ------------
def test_float_split_level_is_one_level_and_a_threshold_at_it_includes_it():
    """Isotonic interpolation emits one step as 0.35280374 and 0.35280376. A
    threshold placed at that level must fire on both copies, not on half."""
    raw = np.array([0.30097087, 0.35280374, 0.35280376, 0.43617, 0.4418605, 0.441861])
    p = snap_levels(raw)
    assert p[1] == p[2] == 0.35280374 and p[4] == p[5]
    assert ((p >= 0.35280374) == np.array([0, 1, 1, 1, 1, 1], bool)).all()
    D, _ = one_pair([0.0] * 6)
    whole = simulate(D, p, 0.35280374, 0.1, enter_ticks=1)
    assert whole["onsets"] == 1 and level_used(p, 0.35280374) == 0.35280374
    assert level_used(raw, 0.35280376 - 1e-9) == 0.35280376    # i.e. the label would lie


def test_option_labels_must_match_the_levels_actually_used():
    p = snap_levels(np.array([0.0, 0.30097087, 0.35280374, 0.35280376, 0.43617]))
    check_option_levels(p, {"ok": (0.35280374, "enter/3", 2), "also_ok": (0.43617, "enter/3", 2)})
    with pytest.raises(ValueError, match="lowest level it fires on"):
        check_option_levels(p, {"mislabelled": (0.35, "enter/3", 2)})    # 0.35 actually means >= 0.3528
