"""DLP ground-truth labels from TRUE positions only (no model): how often do a
vehicle and a pedestrian really come close, and is the vehicle moving then?

This fixes the event definition BEFORE any model output exists. For every
vehicle-pedestrian pair, at 5 Hz (the live risk cadence, every 12th 60 Hz
tick), over the window [t, t + 3 s) at 60 Hz:

    d_min           true minimum centre distance
    moving_at_min   vehicle speed > 0.5 m/s at the time of d_min
    event(d)        exists tau in the window with distance(tau) < d AND vehicle
                    speed(tau) > 0.5 m/s ("moving-vehicle conflict")

A pair-sample is kept only if BOTH agents are present for the whole window:
a track that ends inside it (cut by the drone's view, by the recording, or a
real exit) has an unobserved future, so its label would be a guess.

Calibration recordings only (data/dlp/splits.json); the test split is never read.

    python -m eval.dlp_labels                 # eval/cache/dlp_labels_<rec>.npz, eval/results/dlp_label_*.csv
"""
from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from typing import Dict

import numpy as np

from .accuracy import write_csv
from .common import CACHE_DIR, RESULTS_DIR, ROOT

DLP_DIR = os.path.join(ROOT, "data", "dlp")
TICK_HZ = 60
RISK_EVERY = 12                     # 60 Hz ticks between risk updates (5 Hz)
WINDOW = 3 * TICK_HZ                # [t, t + 3 s) at 60 Hz
MOVING_MPS = 0.5
MOVING_MPS_SECONDARY = 1.0          # pre-declared secondary analysis
THRESHOLDS_M = (1.0, 1.5, 2.0, 2.5, 3.0)
NEAR_M = 15.0
HIST_MAX_M = 20.0
HIST_BIN_M = 0.25
PED_HALF_M = 0.416                  # every DLP pedestrian has the same 0.832 m box
D_STAR_M = 2.5                      # chosen 2026-10-11 (docs/dlp_event_definition.md)
# overlap_artifact rule (docs/dlp_event_definition.md, "Label quality"):
CO_MOVING_MPS = 0.5                 # pedestrian travelling WITH the car, not past it
ARTIFACT_GAP_M = -0.17              # reported only: bodies would interpenetrate (box 0.83 m, body ~0.5 m)


def splits() -> dict:
    with open(os.path.join(DLP_DIR, "splits.json")) as f:
        return json.load(f)


def dense_tracks(name: str):
    """-> {type: (ids, x, y, speed, heading)} with (n_agents, n_ticks) arrays, NaN where absent.
    Reads the jsonl.gz directly: the converter already validated every tick."""
    rows = defaultdict(list)
    n = 0
    with gzip.open(os.path.join(DLP_DIR, f"{name}.jsonl.gz"), "rt") as f:
        for line in f:
            msg = json.loads(line)
            if msg.get("type") != "tick":
                continue
            k = msg["seq"]
            n = max(n, k + 1)
            for a in msg["agents"]:
                rows[(a["type"], a["id"])].append((k, a["x_m"], a["y_m"], a["speed_mps"], a["heading_rad"]))
    out = {}
    for typ in ("vehicle", "pedestrian"):
        ids = sorted(i for t, i in rows if t == typ)
        arr = np.full((4, len(ids), n), np.nan)
        for j, i in enumerate(ids):
            r = np.array(rows[(typ, i)])
            k = r[:, 0].astype(int)
            arr[:, j, k] = r[:, 1:].T
        out[typ] = (np.array(ids), arr[0], arr[1], arr[2], arr[3])
    return out, n


def vehicle_sizes(name: str) -> Dict[int, tuple]:
    with open(os.path.join(DLP_DIR, f"{name}.meta.json")) as f:
        return {int(i): tuple(a["size_m"]) for i, a in json.load(f)["agents"].items() if a["type"] == "vehicle"}


def label_recording(name: str) -> Dict[str, np.ndarray]:
    """One row per kept (anchor, vehicle, pedestrian)."""
    tr, n = dense_tracks(name)
    vid, vx, vy, vs, vh = tr["vehicle"]
    pid, px, py, _, _ = tr["pedestrian"]
    vvx, vvy = np.gradient(vx, 1 / TICK_HZ, axis=1), np.gradient(vy, 1 / TICK_HZ, axis=1)
    pvx, pvy = np.gradient(px, 1 / TICK_HZ, axis=1), np.gradient(py, 1 / TICK_HZ, axis=1)
    size = vehicle_sizes(name)
    half_len = np.array([size[i][0] / 2 for i in vid])
    half_wid = np.array([size[i][1] / 2 for i in vid])
    cols = defaultdict(list)
    dropped = total = 0
    for k in range(0, n - WINDOW + 1, RISK_EVERY):
        w = slice(k, k + WINDOW)
        v_now = ~np.isnan(vx[:, k])
        p_now = ~np.isnan(px[:, k])
        v_ok = v_now & ~np.isnan(vx[:, w]).any(axis=1)
        p_ok = p_now & ~np.isnan(px[:, w]).any(axis=1)
        total += int(v_now.sum() * p_now.sum())
        dropped += int(v_now.sum() * p_now.sum() - v_ok.sum() * p_ok.sum())
        if not v_ok.any() or not p_ok.any():
            continue
        dx = vx[v_ok, None, w] - px[None, p_ok, w]            # V x P x T
        dy = vy[v_ok, None, w] - py[None, p_ok, w]
        d = np.hypot(dx, dy)
        moving = (vs[v_ok, w] > MOVING_MPS)[:, None, :]       # V x 1 x T
        i_min = d.argmin(axis=-1)
        d_min = np.take_along_axis(d, i_min[..., None], -1)[..., 0]
        mv_at_min = np.take_along_axis(np.broadcast_to(moving, d.shape), i_min[..., None], -1)[..., 0]
        V, P = d_min.shape
        cols["t_s"].append(np.full(V * P, k / TICK_HZ))
        cols["veh"].append(np.repeat(vid[v_ok], P))
        cols["ped"].append(np.tile(pid[p_ok], V))
        cols["d_now"].append(d[..., 0].ravel())
        cols["d_min"].append(d_min.ravel())
        cols["moving_at_min"].append(mv_at_min.ravel())
        cols["veh_moving_now"].append(np.repeat(vs[v_ok, k] > MOVING_MPS, P))
        # Smallest distance while the vehicle is moving: event(d) <=> this < d.
        d_mv = np.where(moving, d, np.inf)
        cols["d_min_moving"].append(d_mv.min(axis=-1).ravel())
        cols["d_min_moving_1"].append(np.where((vs[v_ok, w] > MOVING_MPS_SECONDARY)[:, None, :], d, np.inf)
                                      .min(axis=-1).ravel())
        # Where the pedestrian is relative to the car body at the closest MOVING
        # approach: lon along the heading, lat across it, and the real gap
        # between the vehicle box and the pedestrian box (0 = touching).
        j = d_mv.argmin(axis=-1)[..., None]
        ddx = np.take_along_axis(-dx, j, -1)[..., 0]           # pedestrian minus vehicle
        ddy = np.take_along_axis(-dy, j, -1)[..., 0]
        h = np.take_along_axis(np.broadcast_to(vh[v_ok, None, w], d.shape), j, -1)[..., 0]
        lon = ddx * np.cos(h) + ddy * np.sin(h)
        lat = -ddx * np.sin(h) + ddy * np.cos(h)
        hl, hw = half_len[v_ok][:, None], half_wid[v_ok][:, None]
        gap = np.hypot(np.maximum(np.abs(lon) - hl, 0), np.maximum(np.abs(lat) - hw, 0)) - PED_HALF_M
        cols["lon_at_min_moving"].append(lon.ravel())
        cols["lat_at_min_moving"].append(lat.ravel())
        cols["box_gap_at_min_moving"].append(gap.ravel())
        at = lambda a: np.take_along_axis(np.broadcast_to(a, d.shape), j, -1)[..., 0]
        rvx = at(pvx[None, p_ok, w]) - at(vvx[v_ok, None, w])
        rvy = at(pvy[None, p_ok, w]) - at(vvy[v_ok, None, w])
        cols["rel_speed_at_min_moving"].append(np.hypot(rvx, rvy).ravel())
        cols["veh_speed_at_min_moving"].append(at(vs[v_ok, None, w]).ravel())
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["_dropped"], out["_total"] = np.array(dropped), np.array(total)
    return out


def overlap_artifact(L: Dict[str, np.ndarray]) -> np.ndarray:
    """Per pair-update: at the closest MOVING approach the pedestrian is
    inside or touching the car's box (box gap <= 0) AND travelling with it
    (relative speed < CO_MOVING_MPS): people walking with a creeping car, not
    a near miss. Fast passes with overlapping boxes are NOT flagged: replay
    stills show real very close passes measured ~0.5 m too close, which are
    positives at d* = 2.5 m either way. Decided on the calibration split.
    Flagged events stay in the primary analysis; results are also reported
    without them."""
    return (L["box_gap_at_min_moving"] <= 0) & (L["rel_speed_at_min_moving"] < CO_MOVING_MPS)


def cached_labels(name: str) -> Dict[str, np.ndarray]:
    path = os.path.join(CACHE_DIR, f"dlp_labels_{name}.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    lab = label_recording(name)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **lab)
    return lab


def sizes(names) -> Dict[str, np.ndarray]:
    """Half-widths and half-lengths from the DLP size fields (agents in the
    stream + parked obstacles), per type."""
    out = defaultdict(list)
    for name in names:
        with open(os.path.join(DLP_DIR, f"{name}.meta.json")) as f:
            m = json.load(f)
        for a in m["agents"].values():
            out[f"{a['type']}_half_width"].append(a["size_m"][1] / 2)
            out[f"{a['type']}_half_length"].append(a["size_m"][0] / 2)
        for o in m["obstacles"]:
            out["parked_half_width"].append(o["width_m"] / 2)
    return {k: np.array(v) for k, v in out.items()}


def main() -> None:
    names = splits()["calibration"]
    labs = {n: cached_labels(n) for n in names}
    rows, per_rec = [], []
    for subset in ("all", f"within_{NEAR_M:g}m"):
        for thr in THRESHOLDS_M:
            for definition in ("event", "min_and_moving_at_min", "proximity_any_speed"):
                pos = tot = 0
                rates = []
                for n, L in labs.items():
                    keep = np.ones(len(L["d_min"]), bool) if subset == "all" else L["d_now"] < NEAR_M
                    if definition == "event":
                        y = L["d_min_moving"] < thr
                    elif definition == "min_and_moving_at_min":
                        y = (L["d_min"] < thr) & L["moving_at_min"]
                    else:
                        y = L["d_min"] < thr
                    p, t = int(y[keep].sum()), int(keep.sum())
                    pos, tot = pos + p, tot + t
                    rates.append(p / max(1, t))
                    if definition == "event":
                        per_rec.append({"recording": n, "subset": subset, "d_m": thr, "pair_updates": t,
                                        "positives": p, "per_10k": p / max(1, t) * 1e4})
                rows.append({"subset": subset, "d_m": thr, "definition": definition, "pair_updates": tot,
                             "positives": pos, "per_10k": pos / max(1, tot) * 1e4,
                             "per_10k_min_recording": min(rates) * 1e4, "per_10k_max_recording": max(rates) * 1e4,
                             "recordings_with_zero": sum(r == 0 for r in rates)})
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_base_rates.csv"), rows)
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_base_rates_per_recording.csv"), per_rec)

    edges = np.arange(0, HIST_MAX_M + HIST_BIN_M, HIST_BIN_M)
    hist = []
    for subset in ("all", f"within_{NEAR_M:g}m"):
        for state, flag in (("moving", True), ("stationary", False)):
            c = np.zeros(len(edges) - 1, int)
            for L in labs.values():
                keep = (L["moving_at_min"] == flag) & ((L["d_now"] < NEAR_M) if subset != "all" else True)
                c += np.histogram(L["d_min"][keep], edges)[0]
            hist += [{"subset": subset, "vehicle_at_min": state, "d_lo_m": round(a, 2), "d_hi_m": round(b, 2),
                      "count": int(x)} for a, b, x in zip(edges[:-1], edges[1:], c)]
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_min_distance_hist.csv"), hist)

    drops = [{"recording": n, "pair_updates_present": int(L["_total"]), "dropped_unobserved_future": int(L["_dropped"]),
              "kept": len(L["d_min"])} for n, L in labs.items()]
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_censoring.csv"), drops)

    # Geometry of the events: where is the pedestrian and how big is the real
    # gap, for the event at each threshold? (A centre distance means a 1.5 m
    # contact beside a car but 2.8 m in front of it.)
    geo = []
    for thr in THRESHOLDS_M:
        lon = np.concatenate([L["lon_at_min_moving"][L["d_min_moving"] < thr] for L in labs.values()])
        lat = np.concatenate([L["lat_at_min_moving"][L["d_min_moving"] < thr] for L in labs.values()])
        gap = np.concatenate([L["box_gap_at_min_moving"][L["d_min_moving"] < thr] for L in labs.values()])
        ahead = np.abs(lon) > np.abs(lat) * 2.0         # within ~27 deg of the long axis
        geo.append({"d_m": thr, "events": len(gap), "share_front_or_back": float(ahead.mean()) if len(gap) else 0.0,
                    "share_front": float((lon[ahead] > 0).sum() / max(1, len(gap))),
                    "share_touching_or_overlap": float((gap <= 0).mean()) if len(gap) else 0.0,
                    "gap_median_m": float(np.median(gap)) if len(gap) else float("nan"),
                    "gap_p90_m": float(np.percentile(gap, 90)) if len(gap) else float("nan")})
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_event_geometry.csv"), geo)

    # Effective sample size: consecutive 5 Hz updates of one encounter are not
    # independent. An encounter = a run of positive updates for one pair with
    # gaps of at most 1 s.
    enc = []
    for thr in THRESHOLDS_M:
        per = []
        for n, L in labs.items():
            sel = L["d_min_moving"] < thr
            runs = 0
            pairs = defaultdict(list)
            for t, v, p in zip(L["t_s"][sel], L["veh"][sel], L["ped"][sel]):
                pairs[(v, p)].append(t)
            for ts in pairs.values():
                ts = np.sort(ts)
                runs += 1 + int((np.diff(ts) > 1.0 + 1e-6).sum())
            per.append((len(pairs), runs))
        enc.append({"d_m": thr, "positive_updates": int(sum((L["d_min_moving"] < thr).sum() for L in labs.values())),
                    "pairs_with_an_event": sum(p for p, _ in per), "encounters": sum(r for _, r in per),
                    "encounters_min_recording": min(r for _, r in per),
                    "encounters_max_recording": max(r for _, r in per)})
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_encounters.csv"), enc)

    sz = sizes(names)
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_sizes.csv"),
              [{"quantity": k, "n": len(v), "p5": round(float(np.percentile(v, 5)), 3),
                "median": round(float(np.median(v)), 3), "p95": round(float(np.percentile(v, 95)), 3),
                "max": round(float(v.max()), 3)} for k, v in sorted(sz.items())])
    print(f"[dlp_labels] {sum(len(L['d_min']) for L in labs.values())} pair-updates from {len(names)} "
          f"calibration recordings -> {RESULTS_DIR}/dlp_label_*.csv")


if __name__ == "__main__":
    main()
