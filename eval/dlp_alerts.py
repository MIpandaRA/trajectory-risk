"""DLP demo alert setting: hysteresis thresholds on the calibrated risk.

A DEMO SETTING, not an evaluation result. The dashboard's 0.6 / 0.2
thresholds were tuned in simulation on raw scores and mean nothing on DLP,
where the raw score is ~10x overconfident. For the README replay, the alert
runs on the isotonic-calibrated probability of the pre-registered learned
score (b) (map fitted on ALL calibration pairs, because the dashboard scores
all pairs), keeping the existing hysteresis structure:

    enter after 1 update >= enter_threshold, clear after 5 updates < clear_threshold,
    clear_threshold = enter_threshold / 3 (the same ratio as the simulator's 0.6 / 0.2)

enter_threshold is chosen on the 10 CALIBRATION recordings as the grid value
whose false alerts per minute are closest to 1. The test split is then run
once with that setting, as an exploratory figure.

Definitions (as in eval/risk_eval.py, adapted to the DLP labels):
* false alert: an alert onset with no positive label for that pair within
  the next 3 s;
* event: an encounter (run of positive updates for a pair, gaps <= 1 s);
  caught if the alert is on at any of its updates.

Approximation: the cached risk rows exclude updates whose 3 s future is not
fully observed (5-6%, mostly a track's last 3 s), so hysteresis runs on the
labelled rows of each pair and restarts when a pair's updates have a gap.

    python -m eval.dlp_alerts
"""
from __future__ import annotations

import json
import os

import numpy as np

from .accuracy import write_csv
from .common import RESULTS_DIR
from .dlp_calibration import fit_maps, label, load
from .dlp_labels import DLP_DIR

EVENT = "event_d2.5"
D_STAR, MOVING = 2.5, 0.5
# Demo setting chosen 2026-10-12 on the calibration recordings (option C of the
# table written by main(); docs/dlp_evaluation.md section 11).
# Every enter threshold is a calibrated level (after snap_levels) floored to 6
# decimals, so it sits at or just below the level and a label
# like "0.43617" means ">= that level, including it" (docs/failures.md #14).
OPTIONS = {  # name: (enter, clear rule, enter_ticks)
    "A": (0.078295, "enter/3", 1), "B": (0.51865, "0.02", 1), "C": (0.43617, "enter/3", 2),
    "D": (0.43617, "enter/3", 3), "E": (0.30097, "0.02", 5)}
LEVEL_TOL = 1e-6
CHOSEN = "C"   # approved operating point: 0.88 false alerts/min, 76% of events (calibration)
ENTER_GRID = np.round(np.concatenate([np.arange(0.02, 0.10, 0.01), np.arange(0.10, 0.60, 0.025)]), 4)
ENTER_TICKS, CLEAR_TICKS, CLEAR_RATIO = 1, 5, 1 / 3
TARGET_FA_PER_MIN = 1.0
STEP_S = 0.2


def snap_levels(p: np.ndarray, tol: float = LEVEL_TOL) -> np.ndarray:
    """Merge values closer than tol onto one level (the group's smallest).
    Isotonic predictions interpolate between steps in floating point, so one
    step can come out as 0.35280374 and 0.35280376; a threshold between the
    two splits a level in half (docs/failures.md #14). Rounding does not fix
    this (0.4418605 rounds to 0.44186 and 0.441861)."""
    u, inv = np.unique(np.asarray(p, dtype=np.float64), return_inverse=True)
    if not len(u):
        return np.asarray(p, dtype=np.float64)
    group = np.cumsum(np.r_[True, np.diff(u) > tol]) - 1
    first = np.r_[0, np.flatnonzero(np.diff(group)) + 1]
    return u[first][group][inv]


def level_used(p: np.ndarray, enter: float) -> float:
    """The smallest score value an alert at this threshold can fire on."""
    above = p[p >= enter]
    return float(above.min()) if len(above) else float("nan")


def check_option_levels(p: np.ndarray, options: dict) -> None:
    """Fail loudly if an option's label is not the level it actually uses."""
    for name, (enter, _, _) in options.items():
        used = level_used(p, enter)
        if not np.isclose(used, enter, rtol=0, atol=LEVEL_TOL):
            raise ValueError(f"option {name}: labelled enter {enter} but the lowest level it fires on is {used}")


def calibrated(models_split: dict, iso) -> np.ndarray:
    return snap_levels(iso.predict(models_split["b_d2.5"]))


def simulate(D: dict, prob: np.ndarray, enter: float, clear: float, enter_ticks: int = ENTER_TICKS,
             clear_ticks: int = CLEAR_TICKS) -> dict:
    """Hysteresis per pair over the 5 Hz rows -> counts per recording index."""
    y = label(D, EVENT)
    C = len(D["names"])
    order = np.lexsort((D["t_s"], D["ped"], D["veh"], D["recording_index"]))
    r, v, p, t = (D[k][order] for k in ("recording_index", "veh", "ped", "t_s"))
    pr, yy = prob[order], y[order]
    new_pair = np.r_[True, (np.diff(r) != 0) | (np.diff(v) != 0) | (np.diff(p) != 0)]
    gap = np.r_[False, np.diff(t) > STEP_S + 1e-6] & ~new_pair
    start = new_pair | gap
    # encounters (events): runs of positives, gaps <= 1 s, per pair
    pos_idx = np.flatnonzero(yy)
    enc_start = np.r_[True, (r[pos_idx][1:] != r[pos_idx][:-1]) | (v[pos_idx][1:] != v[pos_idx][:-1]) |
                      (p[pos_idx][1:] != p[pos_idx][:-1]) |
                      (t[pos_idx][1:] - t[pos_idx][:-1] > 1.0 + 1e-6)] if len(pos_idx) else []
    enc_id = np.cumsum(enc_start) - 1 if len(pos_idx) else np.array([], int)
    # only series that ever reach the enter threshold can alert
    series_id = np.cumsum(start) - 1
    hot = np.zeros(series_id.max() + 1, bool)
    hot[series_id[pr >= enter]] = True
    active = np.zeros(len(pr), bool)
    onsets = []
    # walk the hot series (a small fraction of rows)
    rows_hot = np.flatnonzero(hot[series_id])
    state, above, below, prev_sid = False, 0, 0, -1
    for i in rows_hot:
        sid = series_id[i]
        if sid != prev_sid:
            state, above, below, prev_sid = False, 0, 0, sid
        if pr[i] >= enter:
            above, below = above + 1, 0
        elif pr[i] < clear:
            below, above = below + 1, 0
        else:
            above = below = 0
        if not state and above >= enter_ticks:
            state = True
            onsets.append(i)
        elif state and below >= clear_ticks:
            state = False
        active[i] = state
    # false alerts: onset with no positive for the same pair within the next 3 s
    fa = np.zeros(C)
    false_onset = set()
    for i in onsets:
        j = i
        ok = False
        while j < len(pr) and series_id[j] == series_id[i] and t[j] <= t[i] + 3.0 + 1e-6:
            if yy[j]:
                ok = True
                break
            j += 1
        if not ok:
            fa[r[i]] += 1
            false_onset.add(i)
    n_enc = np.zeros(C)
    caught = np.zeros(C)
    if len(pos_idx):
        enc_rec = np.zeros(enc_id.max() + 1, int)
        enc_rec[enc_id] = r[pos_idx]
        enc_hit = np.zeros(enc_id.max() + 1, bool)
        np.logical_or.at(enc_hit, enc_id, active[pos_idx])
        n_enc = np.bincount(enc_rec, minlength=C).astype(float)
        caught = np.bincount(enc_rec, weights=enc_hit.astype(float), minlength=C)
    encounters = []
    if len(pos_idx):
        for e in range(enc_id.max() + 1):
            rows_e = pos_idx[enc_id == e]
            hit = rows_e[active[rows_e]]
            encounters.append({"recording_index": int(r[rows_e[0]]), "veh": int(v[rows_e[0]]), "ped": int(p[rows_e[0]]),
                               "t_first": float(t[rows_e[0]]), "t_last": float(t[rows_e[-1]]),
                               "first_alert_t": float(t[hit[0]]) if len(hit) else float("nan")})
    onset_rows = [{"recording_index": int(r[i]), "veh": int(v[i]), "ped": int(p[i]), "t": float(t[i]),
                   "false": i in false_onset} for i in onsets]
    return {"false_alerts": fa, "events": n_enc, "caught": caught, "onsets": len(onsets),
            "encounters": encounters, "onset_rows": onset_rows}


def minutes(names) -> np.ndarray:
    out = []
    for n in names:
        with open(os.path.join(DLP_DIR, f"{n}.meta.json")) as f:
            out.append(json.load(f)["duration_s"] / 60)
    return np.array(out)


def clear_of(enter: float, rule: str) -> float:
    return enter / 3 if rule == "enter/3" else float(rule)


def conflict_onsets(D: dict, encounters: list) -> np.ndarray:
    """First 60 Hz instant, from an encounter's first update, at which the
    pair is closer than D_STAR with the vehicle faster than MOVING (the
    event's own condition; it exists within 3 s by the label's definition)."""
    from .dlp_labels import TICK_HZ, dense_tracks
    out = np.full(len(encounters), np.nan)
    by_rec = {}
    for i, e in enumerate(encounters):
        by_rec.setdefault(e["recording_index"], []).append(i)
    for ri, idx in by_rec.items():
        tr, _ = dense_tracks(D["names"][ri])
        vj = {int(x): k for k, x in enumerate(tr["vehicle"][0])}
        pj = {int(x): k for k, x in enumerate(tr["pedestrian"][0])}
        _, VX, VY, VS, _ = tr["vehicle"]
        _, PX, PY, _, _ = tr["pedestrian"]
        for i in idx:
            e = encounters[i]
            k0 = int(round(e["t_first"] * TICK_HZ))
            w = slice(k0, k0 + 3 * TICK_HZ)
            a, b = vj[e["veh"]], pj[e["ped"]]
            hit = (np.hypot(VX[a, w] - PX[b, w], VY[a, w] - PY[b, w]) < D_STAR) & (VS[a, w] > MOVING)
            if hit.any():
                out[i] = (k0 + int(np.argmax(hit))) / TICK_HZ
    return out


def summarise(D, s, mins, n_boot=1000):
    """False alerts/min, event recall, median warning time; recording-level bootstrap."""
    from trajectory_risk import calibration as cal
    C = len(D["names"])
    enc = s["encounters"]
    onset = conflict_onsets(D, enc)
    rec = np.array([e["recording_index"] for e in enc], int)
    warn = np.array([o - e["first_alert_t"] for o, e in zip(onset, enc)])     # NaN if missed
    fa = s["false_alerts"]

    def stats(w):
        ww = w[rec]
        ok = ~np.isnan(warn) & (ww > 0)
        med = float(np.median(np.repeat(warn[ok], ww[ok].astype(int)))) if ok.any() else float("nan")
        return {"false_alerts_per_min": float(fa @ w / (mins @ w)),
                "event_recall": float((s["caught"] @ w) / (s["events"] @ w)), "median_warning_s": med}

    b = cal.cluster_bootstrap_many(stats, C, n_boot, seed=0)
    out = {"false_alerts": int(fa.sum()), "minutes": float(mins.sum()), "events": int(s["events"].sum()),
           "caught": int(s["caught"].sum()), "warnings_negative": int(np.nansum(warn < 0)), "n_recordings": C}
    for k, v in b.items():
        out[k], out[f"{k}_lo"], out[f"{k}_hi"] = v["point"], v["ci_lo"], v["ci_hi"]
    return out


def isotonic_map():
    return fit_maps(load("calibration", "learned"))[("b_d2.5", EVENT, "all")]["isotonic"]


def main() -> None:
    """Options A-E on the calibration recordings; writes the chosen demo setting."""
    D = load("calibration", "learned")
    iso = isotonic_map()
    prob = calibrated(D, iso)
    mins = minutes(D["names"])
    check_option_levels(prob, OPTIONS)
    rows = []
    for name, (enter, rule, et) in OPTIONS.items():
        s = simulate(D, prob, enter, clear_of(enter, rule), enter_ticks=et)
        row = {"option": name, "enter_threshold": enter, "level_used": level_used(prob, enter), "clear_rule": rule,
               "clear_threshold": clear_of(enter, rule),
               "enter_ticks": et, "clear_ticks": CLEAR_TICKS, **summarise(D, s, mins)}
        rows.append(row)
        print(f"[dlp_alerts] {name}: FA/min {row['false_alerts_per_min']:.2f} recall {row['event_recall']:.3f} "
              f"median warning {row['median_warning_s']:.2f} s", flush=True)
    write_csv(os.path.join(RESULTS_DIR, "dlp_alert_options_calibration.csv"), rows)
    enter, rule, et = OPTIONS[CHOSEN]
    choice = {"option": CHOSEN, "enter_threshold": enter, "clear_threshold": clear_of(enter, rule),
              "enter_ticks": et, "clear_ticks": CLEAR_TICKS,
              "score": "isotonic(learned b_d2.5), map fitted on all calibration pairs",
              "isotonic_x": [float(x) for x in iso.X_thresholds_], "isotonic_y": [float(y) for y in iso.y_thresholds_],
              "isotonic_max_on_calibration": float(prob.max()),
              "level_snap_tolerance": LEVEL_TOL,
              "chosen_on": "calibration recordings only; demo setting, not an evaluation result"}
    with open(os.path.join(RESULTS_DIR, "dlp_demo_alert_setting.json"), "w") as f:
        json.dump(choice, f, indent=1)


def demo_candidates(n: int = 3, lead_s: float = 8.0, length_s: float = 18.0) -> list:
    """Clip windows for the README replay, by a fixed rule (no hand-picking):
    the FIRST caught event of each of the first n calibration recordings (in
    split order) that have one; the window starts lead_s before its conflict
    instant. Lists every event and alert onset inside the window, including
    misses and false alerts."""
    from .dlp_labels import TICK_HZ, dense_tracks
    with open(os.path.join(RESULTS_DIR, "dlp_demo_alert_setting.json")) as f:
        c = json.load(f)
    D = load("calibration", "learned")
    prob = snap_levels(np.interp(D["b_d2.5"], c["isotonic_x"], c["isotonic_y"]))
    s = simulate(D, prob, c["enter_threshold"], c["clear_threshold"], enter_ticks=c["enter_ticks"],
                 clear_ticks=c["clear_ticks"])
    enc = s["encounters"]
    onset = conflict_onsets(D, enc)
    out = []
    for ri, name in enumerate(D["names"]):
        caught = [(onset[i], e) for i, e in enumerate(enc)
                  if e["recording_index"] == ri and not np.isnan(e["first_alert_t"]) and not np.isnan(onset[i])]
        if not caught:
            continue
        t_c, e = min(caught, key=lambda x: x[0])
        start = max(0.0, t_c - lead_s)
        end = start + length_s
        tr, _ = dense_tracks(name)
        vj = {int(x): k for k, x in enumerate(tr["vehicle"][0])}
        pj = {int(x): k for k, x in enumerate(tr["pedestrian"][0])}
        k = int(round(t_c * TICK_HZ))
        cx = (tr["vehicle"][1][vj[e["veh"]], k] + tr["pedestrian"][1][pj[e["ped"]], k]) / 2
        cy = (tr["vehicle"][2][vj[e["veh"]], k] + tr["pedestrian"][2][pj[e["ped"]], k]) / 2
        ev = [(onset[i], x) for i, x in enumerate(enc) if x["recording_index"] == ri
              and not np.isnan(onset[i]) and start <= onset[i] <= end]
        al = [o for o in s["onset_rows"] if o["recording_index"] == ri and start <= o["t"] <= end]
        out.append({"recording": name, "start_s": round(start, 1), "seconds": length_s, "center_x": round(float(cx), 1),
                    "center_y": round(float(cy), 1), "event": f"V{e['veh']} x P{e['ped']} at {t_c:.1f} s",
                    "events_in_window": len(ev), "events_caught": sum(not np.isnan(x["first_alert_t"]) for _, x in ev),
                    "alert_onsets": len(al), "false_alerts": sum(o["false"] for o in al),
                    "details": "; ".join([f"event V{x['veh']}xP{x['ped']} @{t:.1f}s " + ("caught" if not np.isnan(x["first_alert_t"]) else "MISSED")
                                          for t, x in ev] + [f"alert V{o['veh']}xP{o['ped']} @{o['t']:.1f}s" + (" FALSE" if o["false"] else "")
                                                             for o in al])})
        if len(out) == n:
            break
    write_csv(os.path.join(RESULTS_DIR, "dlp_demo_candidates.csv"), out)
    return out


def run_test_once() -> None:
    """EXPLORATORY: the chosen setting on the test split, once."""
    with open(os.path.join(RESULTS_DIR, "dlp_demo_alert_setting.json")) as f:
        c = json.load(f)
    test = load("test", "learned")
    prob = snap_levels(np.interp(test["b_d2.5"], c["isotonic_x"], c["isotonic_y"]))
    s = simulate(test, prob, c["enter_threshold"], c["clear_threshold"], enter_ticks=c["enter_ticks"],
                 clear_ticks=c["clear_ticks"])
    row = {"split": "test (exploratory)", "option": c["option"], **summarise(test, s, minutes(test["names"]))}
    write_csv(os.path.join(RESULTS_DIR, "dlp_alert_test_exploratory.csv"), [row])
    print(f"[dlp_alerts] test (exploratory): {row}")


if __name__ == "__main__":
    import sys
    if "--test-once" in sys.argv:
        run_test_once()
    elif "--candidates" in sys.argv:
        for row in demo_candidates():
            print(row)
    else:
        main()
