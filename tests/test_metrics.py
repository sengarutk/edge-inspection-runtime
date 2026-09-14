import numpy as np
import pytest

from src.metrics.pixel_metrics import compute_pixel_auroc, compute_pixel_ap, compute_aupro
from src.metrics.image_metrics import compute_image_auroc, compute_image_ap, compute_optimal_f1, auroc
from src.metrics.calibration import compute_ece, get_reliability_diagram_data


def test_pixel_auroc_and_aupro_perfect():
    """
    Test that a perfect anomaly map produces AU-PRO > 0.95 and Pixel AUROC > 0.99.
    """
    masks = np.zeros((4, 256, 256), dtype=np.uint8)
    # Image 0: center circle
    y, x = np.ogrid[:256, :256]
    masks[0, (y - 128)**2 + (x - 128)**2 <= 30**2] = 1
    # Image 1: rectangle
    masks[1, 50:100, 50:150] = 1
    # Image 2: two separate blobs
    masks[2, 30:60, 30:60] = 1
    masks[2, 180:210, 180:210] = 1
    # Image 3: normal (no defects)

    # Perfect anomaly map: high scores on defect pixels, 0 on normal
    amaps = masks.astype(float) * 10.0 + np.random.uniform(0.0, 0.01, size=masks.shape)

    p_auroc = compute_pixel_auroc(masks, amaps)
    p_ap = compute_pixel_ap(masks, amaps)
    aupro = compute_aupro(masks, amaps, max_fpr=0.30)

    assert p_auroc > 0.99, f"Expected Pixel AUROC > 0.99, got {p_auroc}"
    assert p_ap > 0.90, f"Expected Pixel AP > 0.90, got {p_ap}"
    assert aupro > 0.95, f"Expected AU-PRO > 0.95, got {aupro}"


def test_pixel_aupro_inverted():
    """
    Test that an inverted anomaly map produces AU-PRO < 0.10.
    """
    masks = np.zeros((2, 256, 256), dtype=np.uint8)
    masks[0, 50:100, 50:100] = 1
    masks[1, 150:200, 150:200] = 1

    # Inverted: high score on normal pixels, low on defect pixels
    amaps = (1.0 - masks.astype(float)) * 10.0

    aupro = compute_aupro(masks, amaps, max_fpr=0.30)
    assert aupro < 0.10, f"Expected inverted AU-PRO < 0.10, got {aupro}"


def test_multiple_disjoint_components():
    """
    Test connected component isolation when an image contains multiple separate defect spots.
    """
    masks = np.zeros((1, 256, 256), dtype=np.uint8)
    # Component 1
    masks[0, 20:50, 20:50] = 1
    # Component 2
    masks[0, 100:130, 100:130] = 1
    # Component 3
    masks[0, 200:230, 200:230] = 1

    # Moderate predictions
    amaps = masks.astype(float) * 5.0 + 0.1

    aupro = compute_aupro(masks, amaps, max_fpr=0.30)
    assert aupro > 0.90


def test_pixel_metrics_edge_cases():
    """
    Test edge cases: all normal images, zero-defect masks, single class.
    """
    empty_masks = np.zeros((2, 256, 256), dtype=np.uint8)
    amaps = np.random.rand(2, 256, 256)

    # Should safely return fallback values without crashing
    assert compute_pixel_auroc(empty_masks, amaps) == 0.5
    assert compute_pixel_ap(empty_masks, amaps) == 0.0
    assert compute_aupro(empty_masks, amaps) == 0.0


def test_image_metrics():
    """
    Test image AUROC, AP, and optimal F1 computation.
    """
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.7, 0.8, 0.65, 0.9])

    auc = compute_image_auroc(labels, scores)
    ap = compute_image_ap(labels, scores)
    f1_res = compute_optimal_f1(labels, scores)

    assert auc == 1.0
    assert ap == 1.0
    assert auroc(labels, scores) == 1.0
    assert f1_res["max_f1"] == 1.0
    assert f1_res["precision_at_optimal"] == 1.0
    assert f1_res["recall_at_optimal"] == 1.0
    assert 0.3 <= f1_res["optimal_threshold"] <= 0.65


def test_calibration_ece():
    """
    Test Expected Calibration Error and reliability diagram data.
    """
    # Perfectly calibrated
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece_good = compute_ece(scores, labels, n_bins=10)
    assert ece_good <= 0.3

    # Inverted confidence
    scores_bad = np.array([0.9, 0.95, 0.05, 0.1])
    labels_bad = np.array([0, 0, 1, 1])
    ece_bad = compute_ece(scores_bad, labels_bad, n_bins=10)
    assert ece_bad > 0.6

    diag_data = get_reliability_diagram_data(scores, labels, n_bins=5)
    assert len(diag_data["bin_centers"]) == 5
    assert len(diag_data["bin_accuracies"]) == 5
    assert len(diag_data["bin_confidences"]) == 5
    assert len(diag_data["bin_counts"]) == 5
    assert sum(diag_data["bin_counts"]) == 4
