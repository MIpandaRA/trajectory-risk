"""Calibration metrics on hand-computed toy cases."""
import numpy as np
import pytest

from trajectory_risk import calibration as cal


def prep(s, y, clusters=None, edges=None):
    s, y = np.asarray(s, float), np.asarray(y, float)
    c = np.zeros(len(s), int) if clusters is None else np.asarray(clusters)
    return cal.prepare(s, y, c, cal.equal_width_edges(10) if edges is None else edges)


def test_perfectly_calibrated_scores_have_ece_near_zero():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 200_000)
    y = rng.uniform(0, 1, s.size) < s
    assert cal.ece(prep(s, y)) < 0.005


def test_constant_miscalibration_gives_the_exact_ece():
    """Every score says 0.3; exactly 200 of 1000 happen: ECE = |0.2 - 0.3| = 0.1."""
    s = np.full(1000, 0.3)
    y = np.r_[np.ones(200), np.zeros(800)]
    assert cal.ece(prep(s, y)) == pytest.approx(0.1, abs=1e-12)
    assert cal.ece(prep(s, y, edges=cal.equal_count_edges(s, 10))) == pytest.approx(0.1, abs=1e-12)


def test_two_level_miscalibration_weights_bins_by_count():
    """600 at 0.1 with 30 events (obs 0.05), 400 at 0.8 with 360 (obs 0.9):
    ECE = 0.6 * 0.05 + 0.4 * 0.1 = 0.07."""
    s = np.r_[np.full(600, 0.1), np.full(400, 0.8)]
    y = np.r_[np.ones(30), np.zeros(570), np.ones(360), np.zeros(40)]
    assert cal.ece(prep(s, y)) == pytest.approx(0.07, abs=1e-12)


def test_tied_scores_share_one_bin_and_the_rest_are_split():
    """5000 exact zeros and 100 distinct positive scores, 10 equal-count bins:
    all zeros in bin 0 (never split), the 100 others over the remaining bins."""
    s = np.r_[np.zeros(5000), np.linspace(0.01, 0.99, 100)]
    edges = cal.equal_count_edges(s, 10)
    b = cal.assign_bins(s, edges)
    assert set(b[:5000]) == {0} and not np.any(b[5000:] == 0)
    counts = np.bincount(b[5000:])[1:]
    assert len(counts) == 9 and counts.max() - counts.min() <= 1


def test_ties_are_never_split_even_in_the_middle():
    s = np.r_[np.linspace(0, 0.4, 50), np.full(300, 0.5), np.linspace(0.6, 1, 50)]
    b = cal.assign_bins(s, cal.equal_count_edges(s, 10))
    assert len(set(b[50:350])) == 1


def test_brier_decomposition_sums_exactly():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 5000)
    y = rng.uniform(0, 1, s.size) < s ** 2
    d = cal.brier(prep(s, y))
    assert d["brier"] == pytest.approx(np.mean((s - y) ** 2), abs=1e-12)
    total = d["reliability"] - d["resolution"] + d["uncertainty"] + d["within_bin_variance"] - d["within_bin_covariance"]
    assert total == pytest.approx(d["brier"], abs=1e-12)


def test_brier_reduces_to_murphy_when_scores_are_constant_per_bin():
    s = np.r_[np.full(600, 0.15), np.full(400, 0.85)]
    y = np.r_[np.ones(30), np.zeros(570), np.ones(360), np.zeros(40)]
    d = cal.brier(prep(s, y))
    assert d["within_bin_variance"] == pytest.approx(0, abs=1e-12)
    assert d["within_bin_covariance"] == pytest.approx(0, abs=1e-12)
    assert d["reliability"] == pytest.approx(0.6 * 0.1 ** 2 + 0.4 * 0.05 ** 2)
    assert d["uncertainty"] == pytest.approx(0.39 * 0.61)
    assert d["reliability"] - d["resolution"] + d["uncertainty"] == pytest.approx(d["brier"], abs=1e-12)


def test_auroc_and_pr_auc_match_sklearn_with_ties():
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(2)
    s = np.round(rng.uniform(0, 1, 3000), 1)          # heavy ties
    y = rng.uniform(0, 1, s.size) < s
    p = prep(s, y)
    assert cal.auroc(p) == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    assert cal.pr_auc(p) == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_platt_recovers_known_parameters():
    """y ~ Bernoulli(sigmoid(a * logit(s) + b)) with a = 0.5, b = -2."""
    rng = np.random.default_rng(3)
    x = rng.normal(0, 2, 200_000)
    s = 1 / (1 + np.exp(-x))
    y = rng.uniform(0, 1, s.size) < 1 / (1 + np.exp(-(0.5 * x - 2.0)))
    a, b = cal.fit_platt(s, y)
    assert a == pytest.approx(0.5, abs=0.02) and b == pytest.approx(-2.0, abs=0.03)
    assert cal.ece(prep(cal.apply_platt(s, (a, b)), y)) < 0.005


def test_isotonic_is_monotone_and_fixes_a_monotone_distortion():
    rng = np.random.default_rng(4)
    s = rng.uniform(0, 1, 50_000)
    y = rng.uniform(0, 1, s.size) < s ** 3          # score much too high
    iso = cal.fit_isotonic(s, y)
    r = iso.predict(np.sort(s))
    assert np.all(np.diff(r) >= 0)
    assert cal.ece(prep(iso.predict(s), y)) < cal.ece(prep(s, y)) / 10


def test_cluster_weights_reproduce_pooled_metrics():
    """Weights (2, 0) must equal the metric on cluster 0 duplicated."""
    rng = np.random.default_rng(5)
    s = rng.uniform(0, 1, 2000)
    y = rng.uniform(0, 1, s.size) < s
    c = (np.arange(s.size) >= 1000).astype(int)
    p = prep(s, y, c)
    only0 = prep(np.r_[s[:1000], s[:1000]], np.r_[y[:1000], y[:1000]])
    assert cal.auroc(p, [2, 0]) == pytest.approx(cal.auroc(only0))
    assert cal.ece(p, [2, 0]) == pytest.approx(cal.ece(only0))
    assert cal.brier(p, [2, 0])["brier"] == pytest.approx(cal.brier(only0)["brier"])


def test_cluster_bootstrap_is_seeded_and_reports_cluster_count():
    rng = np.random.default_rng(6)
    s = rng.uniform(0, 1, 3000)
    y = rng.uniform(0, 1, s.size) < s
    c = rng.integers(0, 6, s.size)
    p = prep(s, y, c)
    r1 = cal.cluster_bootstrap(lambda w: cal.ece(p, w), 6, n_boot=200, seed=0)
    r2 = cal.cluster_bootstrap(lambda w: cal.ece(p, w), 6, n_boot=200, seed=0)
    assert r1 == r2 and r1["n_clusters"] == 6
    assert r1["ci_lo"] <= r1["ci_hi"] and r1["nan_replicates"] == 0


def test_bootstrap_many_matches_single_statistic():
    rng = np.random.default_rng(7)
    s = rng.uniform(0, 1, 3000)
    y = rng.uniform(0, 1, s.size) < s
    c = rng.integers(0, 5, s.size)
    p = prep(s, y, c)
    one = cal.cluster_bootstrap(lambda w: cal.ece(p, w), 5, n_boot=100, seed=1)
    many = cal.cluster_bootstrap_many(lambda w: {"ece": cal.ece(p, w), "auroc": cal.auroc(p, w)}, 5, n_boot=100, seed=1)
    assert many["ece"]["ci_lo"] == one["ci_lo"] and many["ece"]["ci_hi"] == one["ci_hi"]
