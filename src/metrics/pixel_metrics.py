import warnings
from typing import Dict, Any, Tuple
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score


def compute_pixel_auroc(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    gt_flat = ground_truth_masks.astype(int).ravel()
    amap_flat = anomaly_maps.ravel()

    unique_classes = np.unique(gt_flat)
    if len(unique_classes) < 2:
        warnings.warn("compute_pixel_auroc: Ground truth contains only one class. Returning 0.5.")
        return 0.5

    return float(roc_auc_score(gt_flat, amap_flat))


def compute_pixel_ap(ground_truth_masks: np.ndarray, anomaly_maps: np.ndarray) -> float:
    gt_flat = ground_truth_masks.astype(int).ravel()
    amap_flat = anomaly_maps.ravel()

    unique_classes = np.unique(gt_flat)
    if len(unique_classes) < 2:
        warnings.warn("compute_pixel_ap: Ground truth contains only one class. Returning 0.0.")
        return 0.0

    return float(average_precision_score(gt_flat, amap_flat))


def compute_aupro(
    ground_truth_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 200
) -> float:
    """
    Per-Region Overlap (AU-PRO) implementation following the MVTec-style evaluation protocol up to a maximum FPR of 0.30.
    Traverses the threshold descent trajectory and integrates area under PRO curve strictly over normal pixels up to max_fpr.
    """
    N = ground_truth_masks.shape[0]
    gt_masks = (ground_truth_masks > 0.5).astype(np.uint8)
    amaps = anomaly_maps.astype(np.float64)

    # 1. Connected components identification
    components = []
    for i in range(N):
        mask_i = gt_masks[i]
        if np.sum(mask_i) == 0:
            continue
        labeled_mask, num_features = label(mask_i)
        for comp_id in range(1, num_features + 1):
            comp_pixels = (labeled_mask == comp_id)
            comp_size = np.sum(comp_pixels)
            if comp_size > 0:
                components.append((i, comp_pixels, comp_size))

    if len(components) == 0:
        warnings.warn("compute_aupro: No defect components found in masks. Returning 0.0.")
        return 0.0

    total_normal_pixels = float(np.sum(1 - gt_masks))
    if total_normal_pixels == 0:
        warnings.warn("compute_aupro: No normal background pixels found. Returning 0.0.")
        return 0.0

    # 2. Extract adaptive thresholds in descending order
    unique_vals = np.unique(amaps)
    if len(unique_vals) <= num_thresholds:
        thresholds = np.sort(unique_vals)[::-1]
    else:
        percentiles = np.linspace(100, 0, num_thresholds)
        thresholds = np.percentile(amaps, percentiles)
        thresholds = np.unique(thresholds)[::-1]

    # Prepend threshold above max to anchor (FPR=0, PRO=0)
    thresholds = np.concatenate([[float(thresholds[0]) + 1e-5], thresholds])

    fpr_list = [0.0]
    pro_list = [0.0]

    for th in thresholds:
        binary_pred = (amaps >= th).astype(np.uint8)

        fp_pixels = float(np.sum(binary_pred * (1 - gt_masks)))
        fpr = fp_pixels / total_normal_pixels

        pro_vals = []
        for img_idx, comp_pixels, comp_size in components:
            inter = float(np.sum(binary_pred[img_idx] * comp_pixels))
            pro_vals.append(inter / comp_size)
        mean_pro = float(np.mean(pro_vals))

        fpr_list.append(fpr)
        pro_list.append(mean_pro)

    # 3. Process trajectory segments up to max_fpr
    clipped_fprs = [0.0]
    clipped_pros = [0.0]

    for i in range(1, len(fpr_list)):
        f_prev, p_prev = fpr_list[i - 1], pro_list[i - 1]
        f_curr, p_curr = fpr_list[i], pro_list[i]

        if f_curr <= max_fpr:
            clipped_fprs.append(f_curr)
            clipped_pros.append(p_curr)
        else:
            # Crosses max_fpr: interpolate exact intercept
            if f_curr > f_prev:
                p_interp = p_prev + (p_curr - p_prev) * (max_fpr - f_prev) / (f_curr - f_prev)
            else:
                p_interp = p_prev
            clipped_fprs.append(max_fpr)
            clipped_pros.append(p_interp)
            break

    if clipped_fprs[-1] < max_fpr:
        clipped_fprs.append(max_fpr)
        clipped_pros.append(clipped_pros[-1])

    # Trapezoidal integration normalized by max_fpr
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aupro_val = _trapz(clipped_pros, clipped_fprs) / max_fpr
    return float(np.clip(aupro_val, 0.0, 1.0))
