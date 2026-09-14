from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.isotonic import IsotonicRegression


def compute_ece(
    arg1: np.ndarray,
    arg2: np.ndarray,
    num_bins: int = 10,
    n_bins: Optional[int] = None,
    **kwargs
) -> float:
    """
    Expected Calibration Error (ECE) for exploratory score-outcome reliability analysis.
    
    Calibration Note: Raw anomaly scores are not intrinsically calibrated class probabilities.
    In unsupervised visual anomaly detection (trained on nominal y=0 data only), fitting
    posterior probability calibration without an independent labeled validation split introduces
    test-label leakage. Reliability diagrams are provided strictly as exploratory score-outcome
    rank diagnostics, not as deployment-calibrated risk estimates.
    """
    bins = n_bins if n_bins is not None else num_bins
    arr1 = np.asarray(arg1)
    arr2 = np.asarray(arg2)

    unique1 = np.unique(arr1)
    unique2 = np.unique(arr2)

    if np.all(np.isin(unique1, [0, 1])) and not np.all(np.isin(unique2, [0, 1])):
        labels = arr1.astype(int)
        scores = arr2.astype(float)
    elif np.all(np.isin(unique2, [0, 1])) and not np.all(np.isin(unique1, [0, 1])):
        labels = arr2.astype(int)
        scores = arr1.astype(float)
    else:
        if np.issubdtype(arr1.dtype, np.integer) or np.all(np.isin(unique1, [0, 1])):
            labels = arr1.astype(int)
            scores = arr2.astype(float)
        else:
            scores = arr1.astype(float)
            labels = arr2.astype(int)

    s_min, s_max = np.min(scores), np.max(scores)
    if s_max > s_min:
        confidences = (scores - s_min) / (s_max - s_min)
    else:
        confidences = np.zeros_like(scores)

    bin_boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    N = float(len(labels))

    if N == 0:
        return 0.0

    for i in range(bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / N) * np.abs(bin_acc - bin_conf)

    return float(ece)


def fit_isotonic_calibration(
    nominal_scores: np.ndarray,
    val_scores: np.ndarray,
    val_labels: np.ndarray
) -> IsotonicRegression:
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(val_scores, val_labels)
    return iso_reg


def get_reliability_diagram_data(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    num_bins: int = 10,
    n_bins: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    bins = n_bins if n_bins is not None else num_bins
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_scores, dtype=float)

    s_min, s_max = np.min(scores), np.max(scores)
    if s_max > s_min:
        confidences = (scores - s_min) / (s_max - s_min)
    else:
        confidences = np.zeros_like(scores)

    bin_boundaries = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    bin_accuracies = np.zeros(bins)
    bin_confidences = np.zeros(bins)
    bin_counts = np.zeros(bins)

    for i in range(bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_size = np.sum(in_bin)
        bin_counts[i] = bin_size
        if bin_size > 0:
            bin_accuracies[i] = np.mean(labels[in_bin])
            bin_confidences[i] = np.mean(confidences[in_bin])
        else:
            bin_accuracies[i] = 0.0
            bin_confidences[i] = bin_centers[i]

    return {
        "bin_centers": bin_centers,
        "bin_accuracies": bin_accuracies,
        "bin_confidences": bin_confidences,
        "bin_counts": bin_counts
    }
