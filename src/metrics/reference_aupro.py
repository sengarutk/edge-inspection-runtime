from typing import List, Tuple
import numpy as np
from scipy.ndimage import label


def compute_aupro_reference(
    ground_truth_masks: np.ndarray,
    anomaly_maps: np.ndarray,
    max_fpr: float = 0.30,
    num_thresholds: int = 500
) -> float:
    """
    Independent reference implementation of Area Under the Per-Region Overlap curve (AU-PRO).
    Uses unvectorized loops and explicit connected component bounding evaluations to serve
    as an oracle for numerical parity testing.
    """
    gt_masks = (ground_truth_masks > 0.5).astype(np.uint8)
    amaps = anomaly_maps.astype(np.float64)
    N = gt_masks.shape[0]

    # Find connected components in each defect mask
    components: List[Tuple[int, np.ndarray, int]] = []
    for i in range(N):
        mask_i = gt_masks[i]
        if np.sum(mask_i) == 0:
            continue
        labeled, num_features = label(mask_i)
        for c_id in range(1, num_features + 1):
            comp_pixels = (labeled == c_id)
            comp_size = int(np.sum(comp_pixels))
            if comp_size > 0:
                components.append((i, comp_pixels, comp_size))

    if len(components) == 0:
        return 0.0

    total_normal_pixels = float(np.sum(1 - gt_masks))
    if total_normal_pixels == 0:
        return 0.0

    # Extract unique thresholds
    unique_vals = np.unique(amaps)
    if len(unique_vals) <= num_thresholds:
        thresholds = np.sort(unique_vals)[::-1]
    else:
        percentiles = np.linspace(100, 0, num_thresholds)
        thresholds = np.percentile(amaps, percentiles)
        thresholds = np.unique(thresholds)[::-1]

    # Anchor at FPR=0, PRO=0
    thresholds = np.concatenate([[float(thresholds[0]) + 1e-5], thresholds])

    fpr_curve = [0.0]
    pro_curve = [0.0]

    for th in thresholds:
        bin_map = (amaps >= th).astype(np.uint8)
        fp_pixels = float(np.sum(bin_map * (1 - gt_masks)))
        fpr = fp_pixels / total_normal_pixels

        pros = []
        for img_idx, comp_pixels, comp_size in components:
            overlap = float(np.sum(bin_map[img_idx] * comp_pixels))
            pros.append(overlap / float(comp_size))
        mean_pro = float(np.mean(pros))

        fpr_curve.append(fpr)
        pro_curve.append(mean_pro)

    # Clip to max_fpr
    clipped_fprs = [0.0]
    clipped_pros = [0.0]

    for i in range(1, len(fpr_curve)):
        f_prev, p_prev = fpr_curve[i - 1], pro_curve[i - 1]
        f_curr, p_curr = fpr_curve[i], pro_curve[i]

        if f_curr <= max_fpr:
            clipped_fprs.append(f_curr)
            clipped_pros.append(p_curr)
        else:
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

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    area = _trapz(clipped_pros, clipped_fprs) / max_fpr
    return float(np.clip(area, 0.0, 1.0))