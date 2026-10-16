"""Label quality: the "impossible overlap" positives of the DLP event.

At d* = 2.5 m, some positives have the pedestrian's box overlapping a MOVING
vehicle's box at the closest moving instant (box gap <= 0). This lists the
most severe ones (by encounter, so one long approach does not fill the list),
renders replay stills to look at, and reports how many positives the
`overlap_artifact` rule flags. Calibration split only.

    python -m eval.dlp_label_quality

Writes eval/results/dlp_label_overlaps.csv, dlp_label_artifact_counts.csv and
docs/figures/dlp_label_overlap_<n>.png.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .accuracy import write_csv
from .common import RESULTS_DIR
from .dlp_labels import (ARTIFACT_GAP_M, CO_MOVING_MPS, D_STAR_M, DLP_DIR, PED_HALF_M, cached_labels,
                         dense_tracks, overlap_artifact, splits)
from .dlp_figures import FIG_DIR, INK, MUTED, SURFACE, _style

HIGHLIGHT_V, HIGHLIGHT_P, OTHER_V, OTHER_P = "#eb6834", "#4a3aa7", "#c9c7bf", "#b9b3e0"


def encounters(rows, gap_s: float = 1.0):
    """Group (rec, veh, ped, t, ...) rows into runs with gaps <= gap_s."""
    by = defaultdict(list)
    for r in rows:
        by[(r["recording"], r["veh"], r["ped"])].append(r)
    out = []
    for rs in by.values():
        rs.sort(key=lambda r: r["t_s"])
        cur = [rs[0]]
        for r in rs[1:]:
            if r["t_s"] - cur[-1]["t_s"] > gap_s + 1e-6:
                out.append(cur)
                cur = [r]
            else:
                cur.append(r)
        out.append(cur)
    return out


def box(x, y, h, length, width):
    c, s = np.cos(h), np.sin(h)
    local = np.array([[0.5, 0.5], [0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]) * [length, width]
    return local @ np.array([[c, s], [-s, c]]) + [x, y]


def render(rec: str, veh: int, ped: int, t: float, out: str, title: str) -> None:
    """Four snapshots around t in one fixed view centred on the vehicle at t:
    every agent within 14 m, the pair highlighted, 2 s trails, speeds labelled."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    tr, _ = dense_tracks(rec)
    with open(os.path.join(DLP_DIR, f"{rec}.meta.json")) as f:
        meta = json.load(f)
    sizes = {int(i): a["size_m"] for i, a in meta["agents"].items()}
    vid, vx, vy, vs, vh = tr["vehicle"]
    pid, px, py, ps, _ = tr["pedestrian"]
    vi, pi = list(vid).index(veh), list(pid).index(ped)
    k0 = int(round(t * 60))
    offsets = (-2.0, -1.0, 0.0, 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.7))
    cx, cy = vx[vi, k0], vy[vi, k0]           # one fixed view, centred on the event, so motion shows
    for ax, dt in zip(axes, offsets):
        k = k0 + int(dt * 60)
        trail = slice(max(0, k - 120), k + 1)
        for o in meta["obstacles"]:
            b = box(o["x_m"], o["y_m"], o["heading_rad"], o["length_m"], o["width_m"])
            if np.hypot(o["x_m"] - cx, o["y_m"] - cy) < 15:
                ax.fill(b[:, 0], b[:, 1], color="#ecebe6", ec="#d6d4cc", lw=0.6)
        for j, i in enumerate(vid):
            if np.isnan(vx[j, k]) or np.hypot(vx[j, k] - cx, vy[j, k] - cy) > 14:
                continue
            col = HIGHLIGHT_V if i == veh else OTHER_V
            b = box(vx[j, k], vy[j, k], vh[j, k], *sizes[int(i)])
            ax.fill(b[:, 0], b[:, 1], color=col, alpha=0.35 if i == veh else 0.5, ec=col, lw=1.2)
            ax.plot(vx[j, trail], vy[j, trail], color=col, lw=1)
            if i == veh:
                ax.text(vx[j, k], vy[j, k], f"V{i}\n{vs[j, k]:.1f} m/s", ha="center", va="center", fontsize=7,
                        color=INK)
        for j, i in enumerate(pid):
            if np.isnan(px[j, k]) or np.hypot(px[j, k] - cx, py[j, k] - cy) > 14:
                continue
            col = HIGHLIGHT_P if i == ped else OTHER_P
            b = box(px[j, k], py[j, k], 0.0, 2 * PED_HALF_M, 2 * PED_HALF_M)
            ax.fill(b[:, 0], b[:, 1], color=col, alpha=0.6 if i == ped else 0.4, ec=col, lw=1)
            ax.plot(px[j, trail], py[j, trail], color=col, lw=0.9)
            if i == ped:
                ax.annotate(f"P{i} {ps[j, k]:.1f} m/s", (px[j, k], py[j, k]), xytext=(6, 8),
                            textcoords="offset points", fontsize=7, color=INK)
        ax.set_xlim(cx - 9, cx + 9)
        ax.set_ylim(cy - 9, cy + 9)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t + dt:.1f} s", loc="left", fontsize=8.5)
        ax.tick_params(labelsize=7)
    fig.suptitle(title, x=0.01, ha="left", fontsize=9.5, color=INK)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    names = splits()["calibration"]
    rows, counts = [], []
    for n in names:
        L = cached_labels(n)
        pos = L["d_min_moving"] < D_STAR_M
        flag = overlap_artifact(L)
        counts.append({"recording": n, "positives": int(pos.sum()), "overlap_gap_le_0": int(
            (pos & (L["box_gap_at_min_moving"] <= 0)).sum()), "flagged_artifact": int((pos & flag).sum()),
            "co_moving_any_gap": int((pos & (L["rel_speed_at_min_moving"] < CO_MOVING_MPS)).sum()),
            "deep_overlap_any_speed": int((pos & (L["box_gap_at_min_moving"] < ARTIFACT_GAP_M)).sum())})
        sel = np.flatnonzero(pos & (L["box_gap_at_min_moving"] <= 0))
        for i in sel:
            rows.append({"recording": n, "veh": int(L["veh"][i]), "ped": int(L["ped"][i]), "t_s": float(L["t_s"][i]),
                         "box_gap_m": float(L["box_gap_at_min_moving"][i]), "lon_m": float(L["lon_at_min_moving"][i]),
                         "lat_m": float(L["lat_at_min_moving"][i]),
                         "rel_speed_mps": float(L["rel_speed_at_min_moving"][i]),
                         "veh_speed_mps": float(L["veh_speed_at_min_moving"][i]), "artifact": bool(flag[i])})
    tot = {k: sum(c[k] for c in counts) for k in counts[0] if k != "recording"}
    counts.append({"recording": "ALL", **tot})
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_artifact_counts.csv"), counts)

    enc = encounters(rows)
    worst = sorted(enc, key=lambda e: (min(r["box_gap_m"] for r in e), -len(e)))
    top = []
    for e in worst[:10]:
        w = min(e, key=lambda r: r["box_gap_m"])
        top.append({**w, "encounter_start_s": e[0]["t_s"], "encounter_end_s": e[-1]["t_s"], "updates": len(e)})
    write_csv(os.path.join(RESULTS_DIR, "dlp_label_overlaps.csv"), top)
    print(f"[label-quality] {len(rows)} overlap positives in {len(enc)} encounters; "
          f"flagged {tot['flagged_artifact']} of {tot['positives']} positives")

    picks = (("DJI_0004", 152, 139, 456.8, "co-moving: two pedestrians travel with a creeping car for ~100 s"),
             ("DJI_0003", 3, 22, 80.0, "fast pass: a 2.8 m/s car's side passes over a walking pedestrian"),
             ("DJI_0008", 191, 107, 404.4, "fast pass: a 3.7 m/s car passes over a pedestrian at its rear quarter"))
    for j, (rec, v, p, t, title) in enumerate(picks, 1):
        out = os.path.join(FIG_DIR, f"dlp_label_overlap_{j}.png")
        render(rec, v, p, t, out, f"{rec}  V{v} x P{p}  -  {title}")
        print(out)


if __name__ == "__main__":
    main()
