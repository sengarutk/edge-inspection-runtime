"""Statistical hypothesis testing and bootstrap CI validation tests."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.metrics.significance import apply_holm_bonferroni_correction, paired_significance_test
from src.metrics.stats import bootstrap_ci


def test_bootstrap_ci_standard_normal() -> None:
    """Verify bootstrap confidence intervals enclose true mean on standard normal samples."""
    rng = np.random.RandomState(2026)
    data = rng.normal(loc=10.0, scale=2.0, size=500)

    res = bootstrap_ci(data, stat_fn=np.mean, n_boot=1000, ci=0.95, seed=2026, unit="item")

    assert res["n_samples"] == 500
    assert not res["is_degenerate"]
    assert res["unit"] == "item"
    assert res["ci_level"] == 0.95
    # Mean of sample should be near 10.0
    assert 9.7 <= res["mean"] <= 10.3
    # 95% CI should enclose the true mean 10.0
    assert res["ci_lower"] <= 10.0 <= res["ci_upper"]
    assert res["ci_lower"] < res["mean"] < res["ci_upper"]


def test_bootstrap_ci_edge_cases() -> None:
    """Verify bootstrap CI behavior on empty, single-element, and constant arrays."""
    # Empty
    empty_res = bootstrap_ci([], n_boot=100)
    assert empty_res["is_degenerate"] is True
    assert empty_res["mean"] == 0.0

    # Single element
    single_res = bootstrap_ci([42.0], n_boot=100)
    assert single_res["is_degenerate"] is True
    assert single_res["mean"] == 42.0
    assert single_res["ci_lower"] == 42.0

    # Constant array
    const_res = bootstrap_ci([5.0, 5.0, 5.0, 5.0], n_boot=100)
    assert const_res["is_degenerate"] is True
    assert const_res["ci_lower"] == 5.0 == const_res["ci_upper"]


def test_paired_wilcoxon_significance() -> None:
    """Verify paired Wilcoxon signed-rank and rank-biserial correlation outputs."""
    # Group A consistently higher than Group B
    a = [12.0, 15.0, 14.0, 18.0, 20.0, 16.0, 19.0, 22.0]
    b = [5.0, 6.0, 4.0, 7.0, 8.0, 6.0, 5.0, 9.0]

    res = paired_significance_test(a, b, "test_metric", "POLICY_A", "POLICY_B")

    assert res["metric_name"] == "test_metric"
    assert res["policy_a"] == "POLICY_A"
    assert res["policy_b"] == "POLICY_B"
    assert res["mean_diff"] > 0
    assert res["median_diff"] > 0
    assert res["rank_biserial_r"] == 1.0  # All differences strictly positive
    assert res["p_value"] < 0.05
    assert res["is_significant"] is True

    # Identical samples
    res_ident = paired_significance_test([5.0, 5.0], [5.0, 5.0], "metric", "A", "B")
    assert res_ident["p_value"] == 1.0
    assert res_ident["rank_biserial_r"] == 0.0


def test_holm_bonferroni_adjustment() -> None:
    """Verify step-down Holm-Bonferroni correction enforces strict monotonicity of adjusted p-values."""
    raw_tests = [
        {"metric_name": "m1", "policy_a": "A", "policy_b": "B", "p_value": 0.001, "rank_biserial_r": 0.9, "median_diff": 5.0},
        {"metric_name": "m2", "policy_a": "A", "policy_b": "B", "p_value": 0.015, "rank_biserial_r": 0.7, "median_diff": 3.0},
        {"metric_name": "m3", "policy_a": "A", "policy_b": "B", "p_value": 0.040, "rank_biserial_r": 0.5, "median_diff": 2.0},
        {"metric_name": "m4", "policy_a": "A", "policy_b": "B", "p_value": 0.250, "rank_biserial_r": 0.1, "median_diff": 0.5},
    ]

    df = apply_holm_bonferroni_correction(raw_tests, alpha=0.05)

    assert len(df) == 4
    # Check monotonicity
    adj_p = df["adjusted_p_value"].tolist()
    assert all(adj_p[i] <= adj_p[i+1] for i in range(len(adj_p)-1))

    # m1: 0.001 * 4 = 0.004 <= 0.05
    assert bool(df.loc[df["metric_name"] == "m1", "significant_fwer"].values[0]) is True
    # m4: 0.250 * 1 = 0.250 > 0.05
    assert bool(df.loc[df["metric_name"] == "m4", "significant_fwer"].values[0]) is False

    # Test empty input
    df_empty = apply_holm_bonferroni_correction([])
    assert df_empty.empty
