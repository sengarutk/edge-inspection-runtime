import numpy as np
import pytest

from src.metrics.pixel_metrics import compute_aupro
from src.metrics.reference_aupro import compute_aupro_reference


def test_aupro_parity_against_reference():
    """
    Exact numerical equivalence test: assert |AUPRO_fast - AUPRO_ref| < 1e-5
    across complex multi-component ground-truth masks.
    """
    rng = np.random.RandomState(42)
    N = 10
    H, W = 64, 64
    
    # Create masks with distinct connected components
    masks = np.zeros((N, H, W), dtype=np.uint8)
    for i in range(N):
        if i % 2 == 0:
            # Component 1
            masks[i, 10:20, 10:20] = 1
            # Component 2
            masks[i, 40:50, 40:50] = 1
        elif i % 3 == 0:
            masks[i, 25:35, 25:35] = 1
            
    # Synthetic anomaly maps
    amaps = rng.rand(N, H, W).astype(np.float64)
    # Give defect pixels higher scores
    amaps[masks == 1] += 0.8
    
    # Calculate with fast method
    aupro_fast = compute_aupro(masks, amaps, max_fpr=0.30, num_thresholds=200)
    
    # Calculate with unvectorized reference
    aupro_ref = compute_aupro_reference(masks, amaps, max_fpr=0.30, num_thresholds=200)
    
    assert abs(aupro_fast - aupro_ref) < 1e-5, f"Discrepancy detected: fast={aupro_fast:.6f}, ref={aupro_ref:.6f}"


def test_aupro_edge_cases():
    """
    Verifies reference and fast AU-PRO handle zero defect components gracefully.
    """
    masks = np.zeros((5, 32, 32), dtype=np.uint8)
    amaps = np.ones((5, 32, 32), dtype=np.float64)
    
    assert compute_aupro(masks, amaps) == 0.0
    assert compute_aupro_reference(masks, amaps) == 0.0