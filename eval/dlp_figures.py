"""Figures for the DLP evaluation, drawn only from eval/results/dlp_*.csv.

    python -m eval.dlp_figures
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .common import RESULTS_DIR, ROOT

FIG_DIR = os.path.join(ROOT, "docs", "figures")
MOVING, STATIONARY = "#eb6834", "#2a78d6"
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#fcfcfb"


def read_csv(name: str):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return list(csv.DictReader(f))


def _style(plt):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE})


def label_min_distance() -> str:
    """True minimum distance over the next 3 s, vehicle moving vs stationary
    at that minimum, with the physical contact distances marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    rows = read_csv("dlp_label_min_distance_hist.csv")
    sizes = {r["quantity"]: float(r["median"]) for r in read_csv("dlp_label_sizes.csv")}
    side = sizes["vehicle_half_width"] + sizes["pedestrian_half_width"]
    front = sizes["vehicle_half_length"] + sizes["pedestrian_half_width"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    near = {state: [x for x in rows if x["subset"] == "within_15m" and x["vehicle_at_min"] == state]
            for state in ("moving", "stationary")}
    for ax, xmax, log in ((axes[0], 15.0, True), (axes[1], 4.0, False)):
        for state, color in (("moving", MOVING), ("stationary", STATIONARY)):
            r = [x for x in near[state] if float(x["d_hi_m"]) <= xmax + 1e-9]
            lo = np.array([float(x["d_lo_m"]) for x in r])
            c = np.array([int(x["count"]) for x in r])
            total = sum(int(x["count"]) for x in near[state])
            ax.step(np.append(lo, lo[-1] + 0.25), np.append(c, c[-1]), where="post", color=color, lw=2,
                    label=f"vehicle {state} at the minimum ({total:,} pair-updates)")
        for x, name, ha, dx in ((side, "side contact", "right", -0.08), (front, "front contact", "left", 0.08)):
            ax.axvline(x, color=MUTED, lw=0.9, ls=(0, (4, 3)))
            if not log:  # labelled once, on the zoom, where there is room
                ax.text(x + dx, 0.97, f"{name}\n{x:.2f} m", transform=ax.get_xaxis_transform(), va="top", ha=ha,
                        fontsize=7.5, color=MUTED)
        ax.set_xlim(0, xmax)
        if log:
            ax.set_yscale("log")
            ax.set_ylim(0.8, None)
            ax.set_title("pairs within 15 m now (log scale)", loc="left")
            ax.set_ylabel("pair-updates (5 Hz) per 0.25 m bin")
        else:
            ax.set_ylim(0, None)
            ax.set_title("zoom: 0-4 m (linear)", loc="left")
        ax.set_xlabel("true minimum centre distance over the next 3 s (m)")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("DLP calibration split: how close vehicle-pedestrian pairs really get (true positions, no model)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, "dlp_label_min_distance.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


LEARNED, CV = "#4a3aa7", "#6b6a64"
FLOOR = 1e-4


def _rel_panel(fig, outer, rows, title):
    """Reliability on log-log axes with 95% bars, bin counts underneath.
    Bins whose mean score or observed frequency is exactly 0 are drawn at the
    axis floor with a hollow marker (log axes cannot show 0)."""
    g = outer.subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax, hx = fig.add_subplot(g[0]), fig.add_subplot(g[1])
    x = np.array([float(r["mean_score"]) for r in rows])
    y = np.array([float(r["observed"]) for r in rows])
    lo = np.array([float(r["observed_lo"]) for r in rows])
    hi = np.array([float(r["observed_hi"]) for r in rows])
    n = np.array([int(r["count"]) for r in rows])
    xp, yp = np.maximum(x, FLOOR), np.maximum(y, FLOOR)
    ax.plot([FLOOR, 1], [FLOOR, 1], color=MUTED, lw=0.9, ls=(0, (4, 3)))
    ax.vlines(xp, np.maximum(lo, FLOOR), np.maximum(hi, FLOOR), color=LEARNED, lw=1.2, alpha=0.6)
    zero = (x <= 0) | (y <= 0)
    ax.plot(xp[~zero], yp[~zero], "o", color=LEARNED, ms=5)
    ax.plot(xp[zero], yp[zero], "o", mfc=SURFACE, mec=LEARNED, ms=5)
    for a_ in (ax,):
        a_.set_xscale("log")
        a_.set_yscale("log")
        a_.set_xlim(FLOOR * 0.8, 1.1)
        a_.set_ylim(FLOOR * 0.8, 1.1)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_ylabel("observed frequency", fontsize=8)
    ax.tick_params(labelbottom=False, labelsize=7)
    hx.bar(np.arange(len(n)), n, color=MUTED, width=0.8)
    hx.set_yscale("log")
    hx.set_ylim(1, n.max() * 3)                 # from 1, so the small bins are visible next to the zero bin
    hx.set_xticks(np.arange(len(n)))
    hx.set_xticklabels([f"{v:.0e}" if v > 0 else "0" for v in x], rotation=60, fontsize=6)
    hx.set_ylabel("pairs", fontsize=7)
    hx.tick_params(labelsize=6)
    hx.set_xlabel("bin (mean predicted score, equal-count bins)", fontsize=7)
    for a_ in (ax, hx):
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
    return ax


def reliability(split: str = "test") -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    rows = [r for r in read_csv(f"dlp_reliability_{split}.csv") if r["models"] == "learned"]
    pick = lambda sc, sub, how: [r for r in rows if r["score"] == sc and r["subset"] == sub
                                 and r["recalibration"] == how]
    names = {"a_d2.5": "(a) original score", "b_d2.5": "(b) motion-aware score"}
    outs = []
    fig = plt.figure(figsize=(10, 8))
    outer = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.25)
    for i, sub in enumerate(("within_15m", "all")):
        for j, sc in enumerate(("a_d2.5", "b_d2.5")):
            _rel_panel(fig, outer[i, j], pick(sc, sub, "raw"),
                       f"{names[sc]}, {'pairs within 15 m' if sub != 'all' else 'all pairs'}")
    fig.suptitle(f"DLP {split} split, learned models, raw scores vs the event (d* = 2.5 m, vehicle moving)",
                 x=0.01, ha="left", fontsize=10, color=INK)
    out = os.path.join(FIG_DIR, "dlp_reliability_raw.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    outs.append(out)
    fig = plt.figure(figsize=(13, 8))
    outer = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.25)
    for i, sc in enumerate(("a_d2.5", "b_d2.5")):
        for j, how in enumerate(("raw", "platt", "isotonic")):
            _rel_panel(fig, outer[i, j], pick(sc, "within_15m", how), f"{names[sc]}: {how}")
    fig.suptitle(f"DLP {split} split, pairs within 15 m: recalibration fitted on the calibration recordings",
                 x=0.01, ha="left", fontsize=10, color=INK)
    out = os.path.join(FIG_DIR, "dlp_reliability_recal.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    outs.append(out)
    return outs


def accuracy(split: str = "test") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    rows = read_csv(f"dlp_accuracy_{split}.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [6, 3]})
    for ax, typ, states in ((axes[0], "vehicle", ("all", "stationary", "moving", "starting", "stopping", "reversing")),
                            (axes[1], "pedestrian", ("all", "moving", "standing"))):
        for k, (models, color, dy) in enumerate((("cv", CV, -0.12), ("learned", LEARNED, 0.12))):
            for i, st in enumerate(states):
                r = next((x for x in rows if x["type"] == typ and x["state"] == st and x["models"] == models), None)
                if r is None:
                    continue
                v, lo, hi = float(r["top1_ade"]), float(r["top1_ade_lo"]), float(r["top1_ade_hi"])
                ax.plot([lo, hi], [i + dy, i + dy], color=color, lw=2, alpha=0.5)
                ax.plot(v, i + dy, "o", color=color, ms=6,
                        label=("constant velocity" if models == "cv" else "learned") if i == 0 else None)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels([f"{s}\n(n={next((x['n'] for x in rows if x['type'] == typ and x['state'] == s and x['models'] == 'learned'), 0)})"
                            for s in states], fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("top-1 ADE (m), 95% cluster-bootstrap CI")
        ax.set_title("vehicles (ContextVAE)" if typ == "vehicle" else "pedestrians (SocialVAE)", loc="left")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle(f"DLP {split} split: accuracy under domain shift, by state", x=0.01, ha="left", fontsize=10)
    out = os.path.join(FIG_DIR, "dlp_accuracy.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def cluster_share(split: str = "test") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    rows = read_csv(f"dlp_cluster_share_reliability_{split}.csv")
    metrics = read_csv(f"dlp_cluster_share_{split}.csv")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (model, what) in zip(axes, [(m, w) for m in ("ContextVAE", "SocialVAE")
                                        for w in ("all candidates", "top-1 candidate")]):
        r = [x for x in rows if x["model"] == model and x["candidates"] == what]
        x = np.array([float(v["mean_share"]) for v in r])
        y = np.array([float(v["observed"]) for v in r])
        lo = np.array([float(v["observed_lo"]) for v in r])
        hi = np.array([float(v["observed_hi"]) for v in r])
        top = max(x.max(), hi.max()) * 1.1
        ax.plot([0, top], [0, top], color=MUTED, lw=0.9, ls=(0, (4, 3)))
        ax.vlines(x, lo, hi, color=LEARNED, lw=1.2, alpha=0.6)
        ax.plot(x, y, "o", color=LEARNED, ms=5)
        m = next(v for v in metrics if v["model"] == model and v["candidates"] == what)
        ax.set_title(f"{model}, {what}\nadaptive ECE {float(m['ece_adaptive']):.3f} "
                     f"[{float(m['ece_adaptive_lo']):.3f}, {float(m['ece_adaptive_hi']):.3f}]", loc="left", fontsize=8.5)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("share (predicted)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("was the nearest candidate (observed)")
    fig.suptitle(f"DLP {split} split: are candidate shares probabilities?", x=0.01, y=1.08, ha="left", fontsize=10)
    out = os.path.join(FIG_DIR, "dlp_cluster_share.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    import sys
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    print(label_min_distance())
    for fn in (reliability, accuracy, cluster_share):
        try:
            print(fn(split))
        except FileNotFoundError as e:
            print(f"skip {fn.__name__}: {e}")


if __name__ == "__main__":
    main()
