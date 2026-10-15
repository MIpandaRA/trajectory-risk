"""Figures for the model comparison, written to docs/figures/.

1. results.png                - top-1 error, constant velocity vs learned model,
                                straight vs turning, cars and pedestrians (numbers
                                from compare_vehicles.py and compare_pedestrians.py,
                                typed in below).
2. examples_cars.png,         - straight and turning examples on the T11 map, one
   examples_pedestrians.png     image per agent type, matching results.png. Not
                                cherry-picked: in each group, among all predictions
                                in the t11_crossing scene, the one whose learned
                                top-1 error is the median is shown.

    python -m eval.make_figures
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval.common import ade_fde, future_of, ground_truth
from trajectory_risk.maps import MapBundle
from trajectory_risk.predictors import ConstantVelocity
from trajectory_risk.predictors.contextvae import load_contextvae
from trajectory_risk.predictors.socialvae import SocialVAEPredictor
from trajectory_risk.runtime.checkpoints import CONTEXTVAE_CKPT, SOCIALVAE_CKPT
from trajectory_risk.sim import WorldConfig, generate
from trajectory_risk.tracking import TrackStore

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ebeae5"
CV_COLOR, LEARNED_COLOR = "#eb6834", "#2a78d6"   # same roles as the Day 5 toy figure
ROAD = "#e4e3de"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                     "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- figure 1
RESULTS = {  # top-1 ADE (m), best-of-k ADE (m)
    "Cars (ContextVAE, 3 s ahead)": {"Straight": (0.22, 0.65, 0.30), "Turning": (1.79, 1.37, 0.74)},
    "Pedestrians (SocialVAE, 4.8 s ahead)": {"Straight": (0.02, 0.17, 0.11), "Turning": (1.27, 1.37, 0.84)},
}


def results_figure():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    w = 0.34
    for ax, (title, groups) in zip(axes, RESULTS.items()):
        style(ax)
        for i, (motion, (cv, learned, best)) in enumerate(groups.items()):
            for dx, val, col in ((-w / 2 - 0.01, cv, CV_COLOR), (w / 2 + 0.01, learned, LEARNED_COLOR)):
                ax.bar(i + dx, val, w, color=col, zorder=2)
                ax.text(i + dx, val + 0.04, f"{val:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
            ax.plot([i + w / 2 + 0.01 - w / 2, i + w / 2 + 0.01 + w / 2], [best, best], color=INK, lw=2,
                    zorder=3, solid_capstyle="butt")
        ax.set_xticks(range(len(groups)), list(groups))
        ax.set_title(title, loc="left", fontsize=11)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Average error of top guess (m)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=CV_COLOR), plt.Rectangle((0, 0), 1, 1, color=LEARNED_COLOR),
               plt.Line2D([], [], color=INK, lw=2)]
    fig.legend(handles, ["Constant velocity", "Learned model (top guess)", "Learned model, best of its guesses"],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, "Simulated T11 car park, lower is better. Turning = direction changes by more than 20°.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(os.path.join(OUT, "results.png"), dpi=160)


# ---------------------------------------------------------------- figure 2
def heading_change_deg(future, now):
    pts = np.vstack([now, future])
    d0, d1 = pts[1] - pts[0], pts[-1] - pts[-2]
    if np.linalg.norm(d0) < 1e-6 or np.linalg.norm(d1) < 1e-6:
        return 0.0
    a = math.atan2(d1[1], d1[0]) - math.atan2(d0[1], d0[0])
    return abs(math.degrees((a + math.pi) % (2 * math.pi) - math.pi))


def median_cases(ticks, kind, step, horizon, learned, baseline):
    """Run tracking; for "straight" and "turning" requests separately, return the
    one with the median learned top-1 error, plus how many there were."""
    truth = ground_truth(ticks, step, kind)
    store = TrackStore(kind, step, learned.buffer_len, predict_every=2)
    cases = {"Straight": [], "Turning": []}
    for tick in ticks:
        store.ingest(tick)
        for req in store.due_requests():
            actual = future_of(truth, req.agent_id, req.sample_index, horizon)
            if actual is None:
                continue
            now = np.array(truth[req.agent_id][req.sample_index])
            motion = "Turning" if heading_change_deg(actual, now) > 20 else "Straight"
            cands = learned.predict(req)
            cases[motion].append((ade_fde(cands[0].trajectory, actual)[0], req, actual, cands,
                                  baseline.predict(req)[0].trajectory))
    out = {}
    for motion, cs in cases.items():
        cs.sort(key=lambda c: c[0])
        out[motion] = (cs[len(cs) // 2], len(cs))
    return out


def draw_case(ax, bundle, case, title, learned_name):
    _, req, actual, cands, cv_path = case
    hist = req.positions[2:]
    now = hist[-1]
    span = 14.0
    # Keep the view inside the map: outside it there is no road data, which looks like the road ends.
    cx = float(np.clip(now[0], bundle.x_min + span, bundle.x_min + bundle.width_m - span))
    cy = float(np.clip(now[1], bundle.y_min + span, bundle.y_min + bundle.height_m - span))
    xs = np.linspace(cx - span, cx + span, 281)
    ys = np.linspace(cy - span, cy + span, 281)
    gx, gy = np.meshgrid(xs, ys)
    road = bundle.is_drivable(np.stack([gx, gy], -1)).astype(float)
    ax.contourf(gx, gy, road, levels=[0.5, 1.5], colors=[ROAD], zorder=0)
    top_conf = cands[0].confidence
    for c in cands[1:]:
        ax.plot(*np.vstack([now, c.trajectory]).T, color=LEARNED_COLOR, alpha=0.35,
                lw=0.6 + 2.5 * c.confidence / top_conf, zorder=2)
    ax.plot(*np.vstack([now, cands[0].trajectory]).T, color=LEARNED_COLOR, lw=3, zorder=3)
    ax.plot(*np.vstack([now, cv_path]).T, color=CV_COLOR, lw=3, zorder=3)
    ax.plot(*np.vstack([now, actual]).T, color=INK, lw=2, ls=(0, (4, 3)), zorder=4)
    ax.plot(*hist.T, "-o", color=INK, ms=3.5, lw=2, zorder=5)
    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cy - span, cy + span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=11)


def examples_figure():
    bundle = MapBundle.load("T11")
    ticks = generate(WorldConfig(map_name="T11", num_pedestrians=10, crossing_bias=0.6, seed=2), 120.0, 10.0)
    cars = median_cases(ticks, "vehicle", 0.5, 6, load_contextvae(CONTEXTVAE_CKPT, bundle, seed=0),
                        ConstantVelocity(0.5, 5, 6))
    peds = median_cases(ticks, "pedestrian", 0.4, 12, SocialVAEPredictor(SOCIALVAE_CKPT, seed=0),
                        ConstantVelocity(0.4, 8, 12))
    handles = [plt.Line2D([], [], color=INK, marker="o", ms=3.5, lw=2),
               plt.Line2D([], [], color=INK, lw=2, ls=(0, (4, 3))),
               plt.Line2D([], [], color=CV_COLOR, lw=3),
               plt.Line2D([], [], color=LEARNED_COLOR, lw=3),
               plt.Line2D([], [], color=LEARNED_COLOR, lw=1.5, alpha=0.35),
               plt.Rectangle((0, 0), 1, 1, color=ROAD)]
    labels = ["Observed history", "What actually happened", "Constant velocity",
              "Learned model: top guess", "Other guesses (thicker = more confident)", "Road"]
    for label, model, horizon, cases, filename in (("Cars", "ContextVAE", "3 s", cars, "examples_cars.png"),
                                                   ("Pedestrians", "SocialVAE", "4.8 s", peds,
                                                    "examples_pedestrians.png")):
        fig, axes = plt.subplots(1, 2, figsize=(9, 5.6))
        for ax, motion in zip(axes, ("Straight", "Turning")):
            case, n = cases[motion]
            draw_case(ax, bundle, case, f"{motion} ({n} predictions)", model)
        fig.suptitle(f"{label}: constant velocity vs {model}, {horizon} ahead", x=0.01, ha="left", fontsize=12)
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
        fig.text(0.01, 0.905, "Representative cases, not best cases: each panel is the median-error prediction "
                 "of its kind in one simulated scene.\nTurning = the last step of the true future points more than "
                 "20° away from the first step.", fontsize=8.5, color=INK_2, va="top")
        fig.tight_layout(rect=(0, 0.1, 1, 0.83))
        fig.savefig(os.path.join(OUT, filename), dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    results_figure()
    examples_figure()
    print("wrote", OUT)
