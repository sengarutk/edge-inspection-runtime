"""Bootstrap confidence interval estimation and statistical sampling helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np


def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    stat_fn: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 2026,
    unit: str = "run",
) -> Dict[str, Any]:
    """Compute empirical percentile bootstrap confidence intervals with unit of analysis tracking.

    Args:
        data: Sequence or array of numerical observations.
        stat_fn: Function computing summary statistic across samples.
        n_boot: Number of bootstrap iterations.
        ci: Confidence interval coverage level (default: 0.95).
        seed: Deterministic random seed.
        unit: Granularity unit ("run", "item", "stream").

    Returns:
        Dict with keys: mean, median, ci_lower, ci_upper, ci_level,
        n_samples, n_boot, unit, is_degenerate.
    """
    arr = np.asarray(data, dtype=np.float64)
    n_samples = len(arr)
    if n_samples == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_level": ci,
            "n_samples": 0,
            "n_boot": n_boot,
            "unit": unit,
            "is_degenerate": True,
        }

    if n_samples == 1:
        val = float(arr[0])
        return {
            "mean": val,
            "median": val,
            "ci_lower": val,
            "ci_upper": val,
            "ci_level": ci,
            "n_samples": 1,
            "n_boot": n_boot,
            "unit": unit,
            "is_degenerate": True,
        }

    rng = np.random.RandomState(seed)
    boot_indices = rng.randint(0, n_samples, size=(n_boot, n_samples))
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

    return {
        "mean": mean_val,
        "median": median_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": ci,
        "n_samples": n_samples,
        "n_boot": n_boot,
        "unit": unit,
        "is_degenerate": is_degenerate,
    }
