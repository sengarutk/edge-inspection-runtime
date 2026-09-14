"""Statistical analysis, hypothesis testing, and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 2026,
    unit: str = "run",
    n_bootstraps: Optional[int] = None,
    return_dict: Optional[bool] = None,
) -> Any:
    """
    Unified bootstrap confidence interval calculation.
    Supports both tuple return (mean, low, high) and dict return.
    """
    arr = np.asarray(data, dtype=np.float64)
    n_samples = len(arr)
    num_boot = n_bootstraps if n_bootstraps is not None else n_boot

    # If caller specifically passed n_bootstraps, or requested tuple:
    is_tuple_request = (n_bootstraps is not None) or (return_dict is False)

    if n_samples == 0:
        if is_tuple_request:
            return (0.0, 0.0, 0.0)
        return {
            "mean": 0.0,
            "median": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_level": ci,
            "n_samples": 0,
            "n_boot": num_boot,
            "unit": unit,
            "is_degenerate": True,
        }

    if n_samples == 1:
        val = float(arr[0])
        if is_tuple_request:
            return (val, val, val)
        return {
            "mean": val,
            "median": val,
            "ci_lower": val,
            "ci_upper": val,
            "ci_level": ci,
            "n_samples": 1,
            "n_boot": num_boot,
            "unit": unit,
            "is_degenerate": True,
        }

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n_samples, size=(num_boot, n_samples))
    boot_samples = arr[boot_indices]
    boot_stats = np.apply_along_axis(stat_fn, 1, boot_samples)

    alpha = 1.0 - ci
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_stats, lower_pct))
    ci_upper = float(np.percentile(boot_stats, upper_pct))
    mean_val = float(np.mean(arr))
    median_val = float(np.median(arr))
    is_degenerate = bool(np.isclose(ci_lower, ci_upper, atol=1e-12))

    if is_tuple_request:
        return (mean_val, ci_lower, ci_upper)

    return {
        "mean": mean_val,
        "median": median_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "n_samples": n_samples,
        "n_boot": num_boot,
        "unit": unit,
        "is_degenerate": is_degenerate,
    }


def validate_bootstrap_ci_coverage(
    n_simulations: int = 500,
    sample_size: int = 100,
    true_mean: float = 5.0,
    true_std: float = 2.0,
    ci: float = 0.95,
    n_bootstraps: int = 500,
    seed: int = 42
) -> float:
    """
    Monte Carlo empirical coverage rate validation for non-parametric bootstrap confidence intervals.
    """
    rng = np.random.RandomState(seed)
    covered_count = 0

    for i in range(n_simulations):
        data = rng.normal(loc=true_mean, scale=true_std, size=sample_size)
        sim_seed = seed + i + 1
        _, low, high = bootstrap_ci(data, n_bootstraps=n_bootstraps, ci=ci, seed=sim_seed)
        if low <= true_mean <= high:
            covered_count += 1

    return float(covered_count / n_simulations)


def hierarchical_bootstrap_ci(
    data_records: List[Dict[str, Any]],
    metric_fn: Callable[[List[Dict[str, Any]]], float],
    n_resamples: int = 2000,
    ci: float = 0.95,
    seed: int = 2026
) -> Dict[str, Any]:
    """
    Two-stage hierarchical bootstrap resampling separating run/seed uncertainty from item-level uncertainty.
    """
    if len(data_records) == 0:
        return {
            "estimate": 0.0,
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_lower": 0.0,
            "ci_high": 0.0,
            "ci_upper": 0.0,
            "std_error": 0.0,
            "bootstrap_unit": "hierarchical_item_run"
        }

    original_estimate = float(metric_fn(data_records))

    rng = np.random.RandomState(seed)
    n_runs = len(data_records)
    resample_estimates = []

    for _ in range(n_resamples):
        run_indices = rng.choice(n_runs, size=n_runs, replace=True)
        resampled_runs = []

        for r_idx in run_indices:
            rec = data_records[r_idx]
            n_items = len(rec["scores"]) if "scores" in rec else 0
            if n_items > 0:
                item_indices = rng.choice(n_items, size=n_items, replace=True)
                new_rec = dict(rec)
                if "scores" in rec:
                    new_rec["scores"] = rec["scores"][item_indices]
                if "labels" in rec:
                    new_rec["labels"] = rec["labels"][item_indices]
                resampled_runs.append(new_rec)
            else:
                resampled_runs.append(rec)

        boot_metric = metric_fn(resampled_runs)
        resample_estimates.append(boot_metric)

    resample_arr = np.array(resample_estimates, dtype=np.float64)
    alpha = 1.0 - ci
    ci_low = float(np.percentile(resample_arr, (alpha / 2.0) * 100.0))
    ci_high = float(np.percentile(resample_arr, (1.0 - alpha / 2.0) * 100.0))
    std_error = float(np.std(resample_arr))

    return {
        "estimate": original_estimate,
        "mean": original_estimate,
        "ci_low": ci_low,
        "ci_lower": ci_low,
        "ci_high": ci_high,
        "ci_upper": ci_high,
        "std_error": std_error,
        "bootstrap_unit": "hierarchical_item_run"
    }


def compute_paired_wilcoxon_analysis(
    method_a_metrics: np.ndarray,
    method_b_metrics: np.ndarray,
    alpha: float = 0.05,
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes rigorous paired non-parametric statistical analysis across independent experimental units.
    """
    diffs = np.asarray(method_a_metrics, dtype=np.float64) - np.asarray(method_b_metrics, dtype=np.float64)
    n = len(diffs)
    if n == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "hodges_lehmann": 0.0,
            "rank_biserial": 0.0,
            "mean_diff": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "n_pairs": 0
        }

    # Hodges-Lehmann Estimator: median of all pairwise Walsh averages
    walsh_averages = []
    for i in range(n):
        for j in range(i, n):
            walsh_averages.append((diffs[i] + diffs[j]) / 2.0)
    hl_estimator = float(np.median(walsh_averages))

    # Rank-Biserial Correlation
    nonzero_diffs = diffs[diffs != 0]
    if len(nonzero_diffs) == 0:
        stat = 0.0
        p_val = 1.0
        r_rb = 0.0
    else:
        try:
            res = stats.wilcoxon(method_a_metrics, method_b_metrics, alternative="two-sided")
            stat = float(res.statistic)
            p_val = float(res.pvalue)
        except Exception:
            stat = 0.0
            p_val = 1.0

        abs_diffs = np.abs(nonzero_diffs)
        ranks = stats.rankdata(abs_diffs)
        w_plus = float(np.sum(ranks[nonzero_diffs > 0]))
        w_minus = float(np.sum(ranks[nonzero_diffs < 0]))
        total_w = w_plus + w_minus
        r_rb = float((w_plus - w_minus) / total_w) if total_w > 0 else 0.0

    _, ci_low, ci_high = bootstrap_ci(diffs, n_bootstraps=n_bootstraps, ci=1.0 - alpha, seed=seed)

    return {
        "statistic": stat,
        "p_value": p_val,
        "hodges_lehmann": hl_estimator,
        "rank_biserial": r_rb,
        "mean_diff": float(np.mean(diffs)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_pairs": n
    }


def compute_wilcoxon_significance(
    method_a_scores: np.ndarray,
    method_b_scores: np.ndarray
) -> Dict[str, float]:
    """
    Wrapper for Wilcoxon signed-rank test.
    """
    analysis = compute_paired_wilcoxon_analysis(method_a_scores, method_b_scores)
    return {
        "statistic": analysis["statistic"],
        "p_value": analysis["p_value"],
        "significant_0_05": float(analysis["p_value"] < 0.05),
        "significant_0_01": float(analysis["p_value"] < 0.01)
    }


def apply_holm_bonferroni_correction(
    p_values: Union[Dict[str, float], List[float]],
    alpha: float = 0.05
) -> Any:
    """
    Applies Holm-Bonferroni step-down procedure to strictly control Family-Wise Error Rate (FWER).
    Supports both dict input (returns dict) and list input (returns tuple of lists).
    """
    if isinstance(p_values, list):
        m = len(p_values)
        if m == 0:
            return ([], [])
        indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
        rejected = [False] * m
        adjusted_p = [0.0] * m

        for rank, (orig_idx, p_val) in enumerate(indexed_p):
            k = rank + 1
            threshold = alpha / (m - k + 1)
            adj_p = min(1.0, p_val * (m - k + 1))
            adjusted_p[orig_idx] = adj_p

            if p_val <= threshold:
                rejected[orig_idx] = True
            else:
                break

        # Enforce monotonicity of adjusted p-values
        sorted_adj = [adjusted_p[orig_idx] for orig_idx, _ in indexed_p]
        for i in range(1, m):
            if sorted_adj[i] < sorted_adj[i - 1]:
                sorted_adj[i] = sorted_adj[i - 1]
        for (orig_idx, _), adj_val in zip(indexed_p, sorted_adj):
            adjusted_p[orig_idx] = adj_val

        return (rejected, adjusted_p)

    # Dict input branch (benchmark behavior)
    if not p_values:
        return {}

    sorted_tests = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_tests)

    results = {}
    cum_max_adj_p = 0.0
    rejected = True

    for k, (key, raw_p) in enumerate(sorted_tests):
        divisor = m - k
        alpha_k = alpha / float(divisor)
        raw_adj_p = raw_p * float(divisor)
        cum_max_adj_p = max(cum_max_adj_p, raw_adj_p)
        adj_p = min(1.0, cum_max_adj_p)

        if raw_p > alpha_k:
            rejected = False

        results[key] = {
            "raw_p": float(raw_p),
            "adjusted_p": float(adj_p),
            "alpha_k": float(alpha_k),
            "is_significant": bool(rejected and raw_p <= alpha_k),
            "rank": k + 1
        }

    return results
