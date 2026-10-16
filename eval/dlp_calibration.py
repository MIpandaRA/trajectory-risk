"""Is the DLP risk score a probability of the event? Calibration before and
after recalibration, with recording-level (cluster) bootstrap CIs.

Pre-declared before any learned score existed (docs/dlp_evaluation.md):

* event: true centre distance < d at some instant in the next 3 s while the
  vehicle moves > 0.5 m/s; primary d* = 2.5 m. Secondary: d = 2.0, 3.0 m
  (score buffer = d), and moving > 1.0 m/s (label and score b1).
* scores: (a) original pairwise_risk, (b) motion-aware variant.
* subsets: all pairs; pairs within 15 m now.
* bins: 10 equal-count, tie-aware (adaptive ECE, reliability diagrams) and
  10 equal-width (standard ECE), computed on the split being evaluated.
* recalibration: Platt (logistic on clipped logit) and isotonic, fitted on
  the CALIBRATION recordings only, one map per (model, score, label, subset);
  applied unchanged to test.
* variants: all labels (primary); positives flagged overlap_artifact
  excluded (secondary, primary configurations only).
* CIs: 1000 cluster-bootstrap replicates over recordings.

    python -m eval.dlp_calibration --split calibration          # development, in-sample
    python -m eval.dlp_calibration --split test                 # final, once
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

from trajectory_risk import calibration as cal

from .accuracy import write_csv
from .common import RESULTS_DIR
from .dlp_labels import NEAR_M, splits
from .dlp_risk import risk_path

N_BINS = 10
N_BOOT = 1000
PRIMARY = [("a_d2.5", "event_d2.5"), ("b_d2.5", "event_d2.5")]
SECONDARY = [("a_d2", "event_d2"), ("b_d2", "event_d2"), ("a_d3", "event_d3"), ("b_d3", "event_d3"),
             ("a_d2.5", "event_d2.5_moving1"), ("b1_d2.5", "event_d2.5_moving1")]
SUBSETS = ("all", f"within_{NEAR_M:g}m")
RECAL = ("raw", "platt", "isotonic")
MODELS = ("learned", "cv")


def label(D: dict, name: str) -> np.ndarray:
    d = float(name.split("_d")[1].split("_")[0])
    col = "d_min_moving_1" if name.endswith("moving1") else "d_min_moving"
    return D[col] < d


def load(split: str, models: str) -> dict:
    names = splits()[split]
    parts = defaultdict(list)
    for i, n in enumerate(names):
        z = np.load(risk_path(n, models))
        for k in z.files:
            if not k.startswith("_"):
                parts[k].append(z[k])
        parts["cluster"].append(np.full(len(z["t_s"]), i))
        parts["recording_index"].append(np.full(len(z["t_s"]), i))
    D = {k: np.concatenate(v) for k, v in parts.items()}
    D["names"] = names
    return D


def subset_mask(D: dict, subset: str) -> np.ndarray:
    return np.ones(len(D["t_s"]), bool) if subset == "all" else D["d_now"] < NEAR_M


def fit_maps(Dcal: dict) -> dict:
    """Recalibration maps fitted on calibration recordings only."""
    maps = {}
    for score, lab in PRIMARY:
        for subset in SUBSETS:
            m = subset_mask(Dcal, subset)
            s, y = Dcal[score][m], label(Dcal, lab)[m]
            maps[(score, lab, subset)] = {"platt": cal.fit_platt(s, y), "isotonic": cal.fit_isotonic(s, y)}
    return maps


def recalibrate(s: np.ndarray, maps: dict, key, how: str) -> np.ndarray:
    if how == "raw":
        return s
    if how == "platt":
        return cal.apply_platt(s, maps[key]["platt"])
    return maps[key]["isotonic"].predict(s)


def metrics(s, y, c, C, n_boot=N_BOOT, seed=0):
    edges = cal.equal_count_edges(s, N_BINS)
    pa = cal.prepare(s, y, c, edges)
    pw = cal.prepare(s, y, c, cal.equal_width_edges(N_BINS))
    nb = len(edges) - 1

    def stats(w):
        b = cal.brier(pa, w)
        r = cal.reliability(pa, w)
        out = {"ece_adaptive": cal.ece(pa, w), "ece_width": cal.ece(pw, w), "auroc": cal.auroc(pa, w),
               "pr_auc": cal.pr_auc(pa, w), **{k: b[k] for k in ("brier", "reliability", "resolution",
                                                                  "uncertainty", "base_rate")},
               "mean_score": float((pa.s.sum(0) @ w) / (pa.n.sum(0) @ w))}
        out.update({f"obs_{i}": r["observed"][i] for i in range(nb)})
        return out

    boot = cal.cluster_bootstrap_many(stats, C, n_boot, seed)
    rel = cal.reliability(pa)
    bins = [{"bin": i, "edge_lo": float(edges[i]), "edge_hi": float(edges[i + 1]), "count": int(rel["count"][i]),
             "mean_score": float(rel["mean_score"][i]), "observed": boot[f"obs_{i}"]["point"],
             "observed_lo": boot[f"obs_{i}"]["ci_lo"], "observed_hi": boot[f"obs_{i}"]["ci_hi"]} for i in range(nb)]
    return {k: v for k, v in boot.items() if not k.startswith("obs_")}, bins


def encounters(D, y, mask, gap_s=1.0) -> int:
    t, v, p, r = D["t_s"][mask & y], D["veh"][mask & y], D["ped"][mask & y], D["recording_index"][mask & y]
    if not len(t):
        return 0
    order = np.lexsort((t, p, v, r))
    t, v, p, r = t[order], v[order], p[order], r[order]
    new = np.r_[True, (np.diff(r) != 0) | (np.diff(v) != 0) | (np.diff(p) != 0) | (np.diff(t) > gap_s + 1e-6)]
    return int(new.sum())


def run(split: str, n_boot: int = N_BOOT, models_list=MODELS, tag: str = ""):
    Dcal = {m: load("calibration", m) for m in models_list}
    maps = {m: fit_maps(Dcal[m]) for m in models_list}
    rows, rel_rows, balance, params = [], [], [], []
    for models in models_list:
        D = Dcal[models] if split == "calibration" else load(split, models)
        C = len(D["names"])
        for subset in SUBSETS:
            sm = subset_mask(D, subset)
            for lab in sorted({l for _, l in PRIMARY + SECONDARY}):
                y = label(D, lab)
                if models == models_list[0]:
                    balance.append({"split": split, "subset": subset, "event": lab, "pair_updates": int(sm.sum()),
                                    "positives": int((y & sm).sum()), "negatives": int((~y & sm).sum()),
                                    "per_10k": (y & sm).sum() / max(1, sm.sum()) * 1e4,
                                    "encounters": encounters(D, y, sm),
                                    "positives_flagged_artifact": int((y & sm & D["artifact"]).sum()),
                                    "n_recordings": C})
            configs = [(sc, lab, how, var) for sc, lab in PRIMARY for how in RECAL
                       for var in ("all_labels", "artifact_excluded")]
            configs += [(sc, lab, "raw", "all_labels") for sc, lab in SECONDARY]
            for score, lab, how, var in configs:
                y = label(D, lab)
                keep = sm & ~(y & D["artifact"]) if var == "artifact_excluded" else sm
                s = recalibrate(D[score][keep], maps[models], (score, lab, subset), how)
                m, bins = metrics(s, y[keep].astype(float), D["cluster"][keep], C, n_boot)
                base = {"split": split, "models": models, "score": score, "event": lab, "subset": subset,
                        "recalibration": how, "labels": var, "n": int(keep.sum()), "positives": int(y[keep].sum()),
                        "n_recordings": C}
                row = dict(base)
                for k, v in m.items():
                    row[k], row[f"{k}_lo"], row[f"{k}_hi"] = v["point"], v["ci_lo"], v["ci_hi"]
                rows.append(row)
                if (score, lab) in PRIMARY and var == "all_labels":
                    rel_rows += [{**base, **b} for b in bins]
                print(f"[dlp_calibration] {split} {models} {score} {lab} {subset} {how} {var}: "
                      f"ECE_ad {m['ece_adaptive']['point']:.4f} AUROC {m['auroc']['point']:.3f}", flush=True)
    for models in models_list:
        for (score, lab, subset), mp in maps[models].items():
            params.append({"models": models, "score": score, "event": lab, "subset": subset,
                           "platt_a": mp["platt"][0], "platt_b": mp["platt"][1],
                           "isotonic_steps": int(len(mp["isotonic"].X_thresholds_)),
                           "fitted_on": "calibration recordings only"})
    write_csv(os.path.join(RESULTS_DIR, f"dlp_calibration_{split}{tag}.csv"), rows)
    write_csv(os.path.join(RESULTS_DIR, f"dlp_reliability_{split}{tag}.csv"), rel_rows)
    write_csv(os.path.join(RESULTS_DIR, f"dlp_class_balance_{split}{tag}.csv"), balance)
    write_csv(os.path.join(RESULTS_DIR, f"dlp_recalibration_params{tag}.csv"), params)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="calibration", choices=["calibration", "test"])
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tag", default="", help="suffix for the output CSVs (debug runs)")
    args = ap.parse_args(argv)
    run(args.split, args.n_boot, tuple(args.models), args.tag)


if __name__ == "__main__":
    main()
