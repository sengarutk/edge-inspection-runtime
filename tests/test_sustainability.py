"""
Unit tests for ESG Sustainability modules (QCF Engine and SQI Calculator).
Verifies:
1. Monotonicity: higher defect escapes increase QCF and decrease SQI.
2. Boundary checks: zero escapes and zero false alarms produce optimal SQI.
3. Non-negativity and dimension consistency of all carbon factors.
4. Correct YAML loading and parameter mapping for metal_nut and tile.
"""

import pytest
from pathlib import Path
from src.sustainability.qcf_engine import SustainabilityParameters, QualityCarbonFootprintEngine
from src.sustainability.sqi_calculator import SustainabilityQualityIndexCalculator


CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "sustainability_parameters.yaml"


def test_yaml_config_loading():
    nut_params = SustainabilityParameters.from_yaml(str(CONFIG_PATH), "metal_nut")
    assert nut_params.m_part == 0.45
    assert nut_params.kappa_mat == 2.89
    assert nut_params.E_rework == 1.85
    assert nut_params.gamma_fatal == 0.35
    assert nut_params.theta_tier == 8.0
    assert nut_params.xi_grid == 0.230

    tile_params = SustainabilityParameters.from_yaml(str(CONFIG_PATH), "tile")
    assert tile_params.m_part == 1.20
    assert tile_params.kappa_mat == 0.78
    assert tile_params.E_rework == 3.20
    assert tile_params.gamma_fatal == 0.80
    assert tile_params.theta_tier == 4.5


def test_qcf_non_negativity_and_components():
    engine = QualityCarbonFootprintEngine()
    res = engine.compute_footprint(tp=10, tn=85, fp=3, fn=2, latency_sec=0.035)

    assert res["n_total"] == 100
    assert res["scrapped_mass_batch_kg"] > 0
    assert res["energy_batch_kwh"] > 0
    assert res["qcf_scrap_kgco2e"] > 0
    assert res["qcf_rework_kgco2e"] > 0
    assert res["qcf_escape_kgco2e"] > 0
    assert res["qcf_compute_kgco2e"] > 0
    assert res["qcf_annual_kgco2e"] > 0
    
    # Check sum of components matches batch total
    expected_batch = (
        res["qcf_scrap_kgco2e"]
        + res["qcf_rework_kgco2e"]
        + res["qcf_escape_kgco2e"]
        + res["qcf_compute_kgco2e"]
    )
    assert abs(res["qcf_batch_kgco2e"] - expected_batch) < 1e-6


def test_monotonicity_escapes_increase_qcf():
    engine = QualityCarbonFootprintEngine()
    base = engine.compute_footprint(tp=10, tn=85, fp=3, fn=2, latency_sec=0.04)
    higher_escape = engine.compute_footprint(tp=8, tn=85, fp=3, fn=4, latency_sec=0.04)

    assert higher_escape["qcf_escape_kgco2e"] > base["qcf_escape_kgco2e"]
    assert higher_escape["qcf_annual_kgco2e"] > base["qcf_annual_kgco2e"]


def test_monotonicity_escapes_decrease_sqi():
    calc = SustainabilityQualityIndexCalculator()
    baseline = {"tp": 0, "tn": 0, "fp": 0, "fn": 100}  # all escaped defects

    policy_good = {"tp": 20, "tn": 79, "fp": 1, "fn": 0}
    policy_worse = {"tp": 10, "tn": 79, "fp": 1, "fn": 10}

    res_good = calc.calculate_sqi(policy_good, baseline)
    res_worse = calc.calculate_sqi(policy_worse, baseline)

    assert res_good["sqi"] > res_worse["sqi"]
    assert res_good["csf"] >= res_worse["csf"]


def test_boundary_zero_escapes_zero_false_alarms():
    calc = SustainabilityQualityIndexCalculator()
    baseline = {"tp": 0, "tn": 0, "fp": 0, "fn": 100}
    optimal_policy = {"tp": 20, "tn": 80, "fp": 0, "fn": 0}

    res = calc.calculate_sqi(optimal_policy, baseline)
    assert 0.0 <= res["sqi"] <= 1.0
    assert res["cf"] == 1.0 or res["cf"] > 0.9  # near optimal circularity
    assert res["csf"] > 0.8  # major carbon savings over all-escape baseline


def test_sqi_clamping_bounds():
    calc = SustainabilityQualityIndexCalculator()
    # Pathological case where policy is worse than baseline
    policy_terrible = {"tp": 0, "tn": 0, "fp": 100, "fn": 100}
    baseline_clean = {"tp": 10, "tn": 90, "fp": 0, "fn": 0}

    res = calc.calculate_sqi(policy_terrible, baseline_clean)
    assert 0.0 <= res["sqi"] <= 1.0
    assert res["msf"] >= 0.0
    assert res["esf"] >= 0.0
    assert res["csf"] >= 0.0


def test_zero_total_parts_edge_case():
    engine = QualityCarbonFootprintEngine()
    res = engine.compute_footprint(0, 0, 0, 0)
    assert res["n_total"] == 0
    assert res["qcf_annual_kgco2e"] == 0.0

    calc = SustainabilityQualityIndexCalculator()
    empty_res = calc.calculate_sqi({"tp": 0, "tn": 0, "fp": 0, "fn": 0}, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
    assert 0.0 <= empty_res["sqi"] <= 1.0
