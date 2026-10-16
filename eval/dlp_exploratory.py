"""EXPLORATORY, post-unblinding analyses of the DLP risk score (test split).

These were requested AFTER the pre-registered test results had been seen.
They change no pre-registered choice: same event (d* = 2.5 m, moving
> 0.5 m/s), same scores, same recalibration maps (fitted on the calibration
recordings), same 10 tie-aware equal-count bins, same recording-level
cluster bootstrap (1000 replicates). They add statistics, not decisions.

1. Cluster-bootstrap CIs for recalibrated ECE, Brier skill score
   (1 - Brier / (base rate x (1 - base rate)); raw and recalibrated), AUROC
   and PR-AUC, for learned (a), learned (b) and CV (b); and the paired
   differences learned (b) - CV (b) in AUROC and PR-AUC.
2. ECE / base rate, raw and recalibrated.
3. AUROC and PR-AUC after Platt and isotonic recalibration, and the number
   of distinct isotonic output values (did recalibration flatten resolution?).
4. Per-bin table of the recalibrated reliability diagram: pairs, positives,
   mean predicted, observed rate, recordings contributing.
5. Matched operating point: CV (b)'s single (recall, precision) point on
   learned (b)'s PR curve: learned (b)'s precision at CV's recall and its
   recall at CV's precision (interpolated: best value over thresholds that
   meet the constraint), paired within each bootstrap replicate.
7. ROC split at CV (b)'s false-positive rate: learned (b)'s true-positive
   rate there, and each score's ROC area below / above that FPR (point
   estimates): where does the learned score's AUROC advantage come from?
6. The CV blind spot: among positive pair-updates (within 15 m), the share
   whose vehicle is at rest (true speed < 0.5 m/s) at prediction time, for
   CV (b) score 0 vs score 1; per pair-update and per encounter.

    python -m eval.dlp_exploratory
"""
from __future__ import annotations

import os

import numpy as np

from trajectory_risk import calibration as cal

import csv


def write_csv(path, rows):
    """Like eval.accuracy.write_csv but with 6 decimals: recalibrated ECEs are ~1e-4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
from .common import RESULTS_DIR
from .dlp_calibration import N_BINS, fit_maps, label, load, recalibrate, subset_mask
from .dlp_risk import pair_key

N_BOOT = 1000
EVENT = "event_d2.5"
SCORES = (("learned", "a_d2.5", "learned (a)"), ("learned", "b_d2.5", "learned (b)"), ("cv", "b_d2.5", "CV (b)"))
RECAL = ("raw", "platt", "isotonic")
SUBSETS = ("within_15m", "all")


def stats_for(p: cal.Prepared):
    def stats(w):
        b = cal.brier(p, w)
        e = cal.ece(p, w)
        return {"ece_adaptive": e, "base_rate": b["base_rate"], "ece_over_base_rate": e / b["base_rate"],
                "brier": b["brier"], "brier_skill": 1 - b["brier"] / b["uncertainty"],
                "auroc": cal.auroc(p, w), "pr_auc": cal.pr_auc(p, w)}
    return stats


def main() -> None:
    test = {m: load("test", m) for m in ("learned", "cv")}
    maps = {m: fit_maps(load("calibration", m)) for m in ("learned", "cv")}
    # The two model runs score the same pairs (same schedule): check, then pair rows.
    kl = pair_key(test["learned"]["t_s"], test["learned"]["veh"], test["learned"]["ped"])
    kc = pair_key(test["cv"]["t_s"], test["cv"]["veh"], test["cv"]["ped"])
    rec_l, rec_c = test["learned"]["recording_index"], test["cv"]["recording_index"]
    assert np.array_equal(kl, kc) and np.array_equal(rec_l, rec_c), "learned and CV rows differ"
    C = len(test["learned"]["names"])
    metrics, paired, bins = [], [], []
    for subset in SUBSETS:
        prepared = {}
        for models, score, name in SCORES:
            D = test[models]
            m = subset_mask(D, subset)
            y = label(D, EVENT)[m].astype(float)
            c = D["cluster"][m]
            for how in RECAL:
                s = recalibrate(D[score][m], maps[models], (score, EVENT, subset), how)
                edges = cal.equal_count_edges(s, N_BINS)
                p = cal.prepare(s, y, c, edges)
                prepared[(name, how)] = p
                b = cal.cluster_bootstrap_many(stats_for(p), C, N_BOOT, seed=0)
                row = {"subset": subset, "score": name, "recalibration": how, "n": int(m.sum()),
                       "positives": int(y.sum()), "n_recordings": C,
                       "distinct_score_values": int(len(np.unique(s))), "bins_formed": len(edges) - 1}
                row["isotonic_map_steps"] = (int(len(maps[models][(score, EVENT, subset)]["isotonic"].X_thresholds_))
                                             if how == "isotonic" else "")
                for k, v in b.items():
                    row[k], row[f"{k}_lo"], row[f"{k}_hi"] = v["point"], v["ci_lo"], v["ci_hi"]
                metrics.append(row)
                if subset == "within_15m":
                    bi = cal.assign_bins(s, edges)
                    for j in range(len(edges) - 1):
                        sel = bi == j
                        recs = np.unique(c[sel])
                        bins.append({"subset": subset, "score": name, "recalibration": how, "bin": j,
                                     "edge_lo": float(edges[j]), "edge_hi": float(edges[j + 1]),
                                     "pairs": int(sel.sum()), "positives": int(y[sel].sum()),
                                     "mean_predicted": float(s[sel].mean()) if sel.any() else float("nan"),
                                     "observed_rate": float(y[sel].mean()) if sel.any() else float("nan"),
                                     "recordings_contributing": int(len(recs)),
                                     "recordings_with_a_positive": int(len(np.unique(c[sel & (y > 0)])))})
                print(f"[exploratory] {subset} {name} {how}: ECE {b['ece_adaptive']['point']:.4f} "
                      f"BSS {b['brier_skill']['point']:.3f} AUROC {b['auroc']['point']:.3f}", flush=True)
        for how in RECAL:
            pl, pc = prepared[("learned (b)", how)], prepared[("CV (b)", how)]
            diff = lambda w: {"auroc": cal.auroc(pl, w) - cal.auroc(pc, w), "pr_auc": cal.pr_auc(pl, w) - cal.pr_auc(pc, w)}
            b = cal.cluster_bootstrap_many(diff, C, N_BOOT, seed=0)
            row = {"subset": subset, "comparison": "learned (b) - CV (b)", "recalibration": how, "n_recordings": C}
            for k, v in b.items():
                row[f"{k}_diff"], row[f"{k}_diff_lo"], row[f"{k}_diff_hi"] = v["point"], v["ci_lo"], v["ci_hi"]
            paired.append(row)
    write_csv(os.path.join(RESULTS_DIR, "dlp_exploratory_metrics_test.csv"), metrics)
    write_csv(os.path.join(RESULTS_DIR, "dlp_exploratory_paired_test.csv"), paired)
    write_csv(os.path.join(RESULTS_DIR, "dlp_exploratory_bins_test.csv"), bins)


def pr_points(pos, neg):
    """Weighted per-level positives/negatives (ascending score) -> (recall,
    precision) at every threshold "score >= level", from the top down."""
    pos, neg = pos[::-1], neg[::-1]
    tp, fp = np.cumsum(pos), np.cumsum(neg)
    return tp / max(tp[-1], 1e-300), tp / np.maximum(tp + fp, 1e-300)


def operating_point(test, subset="within_15m"):
    D, Dc = test["learned"], test["cv"]
    m = subset_mask(D, subset)
    y = label(D, EVENT)[m].astype(float)
    c = D["cluster"][m]
    C = len(D["names"])
    edges = cal.equal_count_edges(D["b_d2.5"][m], N_BINS)
    pl = cal.prepare(D["b_d2.5"][m], y, c, edges)                 # raw: ranking is what matters
    pc = cal.prepare(Dc["b_d2.5"][m], y, c, edges)

    def stats(w):
        pos_c, neg_c = pc.pos @ w, pc.neg @ w                       # CV levels: 0 and 1
        top = pc.levels >= 1.0
        tp, fp, P = pos_c[top].sum(), neg_c[top].sum(), pos_c.sum()
        r_cv, p_cv = tp / P, tp / (tp + fp)
        rec, prec = pr_points(pl.pos @ w, pl.neg @ w)
        ok_r, ok_p = rec >= r_cv, prec >= p_cv
        p_at = prec[ok_r].max() if ok_r.any() else float("nan")
        r_at = rec[ok_p].max() if ok_p.any() else 0.0
        return {"cv_recall": r_cv, "cv_precision": p_cv, "learned_precision_at_cv_recall": p_at,
                "learned_recall_at_cv_precision": r_at, "precision_diff": p_at - p_cv, "recall_diff": r_at - r_cv}

    b = cal.cluster_bootstrap_many(stats, C, N_BOOT, seed=0)
    return [{"subset": subset, "quantity": k, "point": v["point"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
             "n_recordings": C, "nan_replicates": v["nan_replicates"]} for k, v in b.items()]


def blind_spot(test, subset="within_15m"):
    """Vehicle speed at prediction time for positives, CV (b) score 0 vs 1."""
    from .dlp_labels import TICK_HZ, dense_tracks
    D, Dc = test["learned"], test["cv"]
    m = subset_mask(D, subset) & label(D, EVENT)
    rest = np.zeros(m.sum(), bool)
    rec_idx = D["recording_index"][m]
    t, veh = D["t_s"][m], D["veh"][m]
    for i, name in enumerate(D["names"]):
        sel = rec_idx == i
        tr, _ = dense_tracks(name)
        ids, S = tr["vehicle"][0], tr["vehicle"][3]
        j = {int(v): k for k, v in enumerate(ids)}
        k = np.round(t[sel] * TICK_HZ).astype(int)
        rest[sel] = np.array([S[j[int(v)], kk] for v, kk in zip(veh[sel], k)]) < 0.5
    cv1 = Dc["b_d2.5"][m] >= 1.0
    C = len(D["names"])
    rows = []
    for name, g in (("CV (b) = 0", ~cv1), ("CV (b) = 1", cv1)):
        cnt = np.bincount(rec_idx[g], minlength=C).astype(float)
        rst = np.bincount(rec_idx[g], weights=rest[g].astype(float), minlength=C)
        b = cal.cluster_bootstrap(lambda w: float(rst @ w / (cnt @ w)) if cnt @ w > 0 else float("nan"), C, N_BOOT, 0)
        rows.append({"subset": subset, "group": name, "level": "pair-update", "positives": int(g.sum()),
                     "vehicle_at_rest": int(rest[g].sum()), "share_at_rest": b["point"], "ci_lo": b["ci_lo"],
                     "ci_hi": b["ci_hi"], "n_recordings": C})
    cnt0 = np.bincount(rec_idx[~cv1], minlength=C).astype(float)
    rst0 = np.bincount(rec_idx[~cv1], weights=rest[~cv1].astype(float), minlength=C)
    cnt1 = np.bincount(rec_idx[cv1], minlength=C).astype(float)
    rst1 = np.bincount(rec_idx[cv1], weights=rest[cv1].astype(float), minlength=C)
    d = cal.cluster_bootstrap(lambda w: float(rst0 @ w / (cnt0 @ w) - rst1 @ w / (cnt1 @ w)), C, N_BOOT, 0)
    rows.append({"subset": subset, "group": "difference (score 0 - score 1)", "level": "pair-update",
                 "positives": "", "vehicle_at_rest": "", "share_at_rest": d["point"], "ci_lo": d["ci_lo"],
                 "ci_hi": d["ci_hi"], "n_recordings": C})
    # Encounter level: an encounter = run of positive updates for one pair (gaps <= 1 s);
    # classify it by its FIRST update (the earliest moment a warning could come).
    order = np.lexsort((t, D["ped"][m], veh, rec_idx))
    tt, vv, pp, rr = t[order], veh[order], D["ped"][m][order], rec_idx[order]
    first = np.r_[True, (np.diff(rr) != 0) | (np.diff(vv) != 0) | (np.diff(pp) != 0) | (np.diff(tt) > 1.0 + 1e-6)]
    for name, g in (("CV (b) = 0", ~cv1[order]), ("CV (b) = 1", cv1[order])):
        sel = first & g
        rows.append({"subset": subset, "group": name, "level": "encounter (first update)",
                     "positives": int(sel.sum()), "vehicle_at_rest": int(rest[order][sel].sum()),
                     "share_at_rest": float(rest[order][sel].mean()) if sel.any() else float("nan"),
                     "ci_lo": "", "ci_hi": "", "n_recordings": C})
    return rows


def roc_split(test, subset="within_15m"):
    D, Dc = test["learned"], test["cv"]
    m = subset_mask(D, subset)
    y = label(D, EVENT)[m]
    sl, sc = D["b_d2.5"][m], Dc["b_d2.5"][m]
    P, N = y.sum(), (~y).sum()
    fpr_cv, tpr_cv = ((sc >= 1) & ~y).sum() / N, ((sc >= 1) & y).sum() / P
    o = np.argsort(-sl, kind="stable")
    ys, ss = y[o], sl[o]
    last = np.r_[np.diff(ss) != 0, True]                    # thresholds between distinct scores
    tpr, fpr = np.cumsum(ys)[last] / P, np.cumsum(~ys)[last] / N
    k = np.searchsorted(fpr, fpr_cv, side="right") - 1
    grid = lambda lo, hi: np.linspace(lo, hi, 20001)
    area = lambda f, t, lo, hi: float(np.trapz(np.interp(grid(lo, hi), f, t), grid(lo, hi)))
    fl, tl = np.r_[0, fpr], np.r_[0, tpr]
    fc, tc = np.array([0, fpr_cv, 1]), np.array([0, tpr_cv, 1])
    rows = [{"subset": subset, "quantity": "cv_fpr", "learned": "", "cv": float(fpr_cv)},
            {"subset": subset, "quantity": "tpr_at_cv_fpr", "learned": float(tpr[k]), "cv": float(tpr_cv)}]
    for name, lo, hi in (("roc_area_below_cv_fpr", 0.0, fpr_cv), ("roc_area_above_cv_fpr", fpr_cv, 1.0)):
        rows.append({"subset": subset, "quantity": name, "learned": area(fl, tl, lo, hi), "cv": area(fc, tc, lo, hi)})
    return rows


def main_extra() -> None:
    test = {m: load("test", m) for m in ("learned", "cv")}
    write_csv(os.path.join(RESULTS_DIR, "dlp_exploratory_operating_point_test.csv"), operating_point(test))
    write_csv(os.path.join(RESULTS_DIR, "dlp_exploratory_blind_spot_test.csv"), blind_spot(test))
    write_csv(os.path.join(RESULTS_DIR, "dlp_exploratory_roc_split_test.csv"), roc_split(test))


if __name__ == "__main__":
    import sys
    if "--extra" in sys.argv:
        main_extra()
        raise SystemExit

    main()
