"""Non-parametric paired hypothesis testing and step-down Holm-Bonferroni correction."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def paired_significance_test(
    sample_a: List[float],
    sample_b: List[float],
    metric_name: str,
    policy_a_name: str,
    policy_b_name: str,
    unit: str = "run",
) -> Dict[str, Any]:
    """Compute paired Wilcoxon signed-rank test, median difference, and rank-biserial effect size.

    Args:
        sample_a: Metric observations from Policy A.
        sample_b: Metric observations from Policy B.
        metric_name: Name of metric tested.
        policy_a_name: Identifier for Policy A.
        policy_b_name: Identifier for Policy B.
        unit: Observation unit ("run", "item", "stream").

    Returns:
        Dict containing statistic, p_value, mean_diff, median_diff, rank_biserial_r, is_significant.
    """
    arr_a = np.asarray(sample_a, dtype=np.float64)
    arr_b = np.asarray(sample_b, dtype=np.float64)
    n = min(len(arr_a), len(arr_b))
    arr_a = arr_a[:n]
    arr_b = arr_b[:n]

    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff)) if n > 0 else 0.0
    median_diff = float(np.median(diff)) if n > 0 else 0.0

    non_zero = diff[diff != 0]
    if len(non_zero) == 0:
        return {
            "metric_name": metric_name,
            "policy_a": policy_a_name,
            "policy_b": policy_b_name,
            "statistic": 0.0,
            "p_value": 1.0,
            "mean_diff": 0.0,
            "median_diff": 0.0,
            "rank_biserial_r": 0.0,
            "is_significant": False,
            "n_pairs": n,
            "unit": unit,
        }

    try:
        res = stats.wilcoxon(arr_a, arr_b, zero_method="pratt")
        statistic = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        statistic = 0.0
        p_val = 1.0

    ranks = stats.rankdata(np.abs(diff))
    pos_ranks = np.sum(ranks[diff > 0])
    neg_ranks = np.sum(ranks[diff < 0])
    total_ranks = pos_ranks + neg_ranks
    r_biserial = float((pos_ranks - neg_ranks) / total_ranks) if total_ranks > 0 else 0.0

    return {
        "metric_name": metric_name,
        "policy_a": policy_a_name,
        "policy_b": policy_b_name,
        "statistic": statistic,
        "p_value": p_val,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "rank_biserial_r": r_biserial,
        "is_significant": bool(p_val < 0.05),
        "n_pairs": n,
        "unit": unit,
    }


def apply_holm_bonferroni_correction(
    test_results: List[Dict[str, Any]], alpha: float = 0.05
) -> pd.DataFrame:
    """Apply step-down Holm-Bonferroni adjustment across a family of pairwise hypothesis tests.

    Args:
        test_results: List of test summary dictionaries containing 'p_value'.
        alpha: Target family-wise error rate (FWER).

    Returns:
        pd.DataFrame sorted by raw p-value with adjusted significance columns.
    """
    if not test_results:
        return pd.DataFrame()

    df = pd.DataFrame(test_results)
    df = df.sort_values(by="p_value", ascending=True).reset_index(drop=True)
    m = len(df)

    adj_p_values = []
    thresholds = []
    significant = []

    running_max = 0.0
    for i, row in df.iterrows():
        multiplier = m - i
        raw_p = float(row["p_value"])
        adjusted_p = min(1.0, raw_p * multiplier)
        adjusted_p = max(running_max, adjusted_p)
        running_max = adjusted_p
        adj_p_values.append(adjusted_p)

        thresh = alpha / (m - i)
        thresholds.append(thresh)
        significant.append(adjusted_p <= alpha)

    df["raw_p_value"] = df["p_value"]
    df["adjusted_p_value"] = adj_p_values
    df["rank"] = range(1, m + 1)
    df["threshold_alpha"] = thresholds
    df["significant_fwer"] = significant

    return df
