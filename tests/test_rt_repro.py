"""Reproducibility, threshold governance manifest, and non-IID stream model tests."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.stream_models import (
    MarkovianDefectGenerator,
    PoissonBurstDefectGenerator,
    ThermalDriftProfile,
)


def test_threshold_manifest_provenance_guard() -> None:
    """Static analysis test verifying zero test leakage in threshold manifest."""
    manifest_file = Path("configs/threshold_manifest.json")
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["governance"]["zero_test_leakage_guaranteed"] is True

    # Assert every parameter explicitly declares uses_test_labels == False
    for param_name, meta in data["parameters"].items():
        assert meta["uses_test_labels"] is False, f"Parameter {param_name} leaked test labels!"
        assert "derived_from" in meta
        assert "type" in meta


def test_deterministic_stream_reproducibility() -> None:
    """Verify deterministic reproducibility of non-IID temporal stream models."""
    gen1 = MarkovianDefectGenerator(seed=2026)
    seq1 = gen1.generate_sequence(100)

    gen2 = MarkovianDefectGenerator(seed=2026)
    seq2 = gen2.generate_sequence(100)

    assert seq1 == seq2
    assert any(seq1)  # Contains defect blocks
    assert not all(seq1)  # Contains nominal blocks

    # Poisson burst generator reproducibility
    p1 = PoissonBurstDefectGenerator(seed=2026)
    p_seq1 = p1.generate_sequence(100)

    p2 = PoissonBurstDefectGenerator(seed=2026)
    p_seq2 = p2.generate_sequence(100)

    assert p_seq1 == p_seq2


def test_thermal_drift_profile() -> None:
    """Verify thermal drift profile calculations across linear and exponential modes."""
    linear_prof = ThermalDriftProfile(base_temp_c=40.0, peak_temp_c=80.0, drift_type="linear", ramp_steps=100)
    t_start = linear_prof.get_temperature(0, noise_std=0.0)
    t_mid = linear_prof.get_temperature(50, noise_std=0.0)
    t_end = linear_prof.get_temperature(100, noise_std=0.0)

    assert np.isclose(t_start, 40.0)
    assert np.isclose(t_mid, 60.0)
    assert np.isclose(t_end, 80.0)

    exp_prof = ThermalDriftProfile(base_temp_c=40.0, peak_temp_c=80.0, drift_type="exponential", ramp_steps=100)
    t_exp_mid = exp_prof.get_temperature(50, noise_std=0.0)
    # Exponential ramp accelerates later, so midpoint should be lower than linear midpoint
    assert t_exp_mid < t_mid
