"""Unit and Integration Tests for Policy Ablation Modes & Experimental Framework."""

import json
from pathlib import Path
import pytest
import numpy as np

from src.audit_log import AuditLogDB
from src.config import (
    ConfirmationWindowConfig,
    CooldownConfig,
    InferenceConfig,
    PolicyConfig,
    PolicyMode,
    ScenarioConfig,
    SensorConfig,
    TemporalSmoothingConfig,
    ThresholdsConfig,
    load_scenario_config,
    load_sensor_config,
)
from src.inference_service import InferenceEngine, InferenceResult, OpticalHealthStatus
from src.metrics import (
    BenchmarkEvaluator,
    aggregate_ablation_results,
    generate_ablation_latex_table,
    generate_ablation_markdown_table,
)
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorReading, SensorSimulator


def create_dummy_inference_result(vision_score: float, is_valid: bool = True) -> InferenceResult:
    """Helper creating mock InferenceResult."""
    return InferenceResult(
        frame_id="frame-1234",
        timestamp_utc="2026-08-27T15:30:00.000Z",
        camera_id="line1_overhead_cam01",
        model_metadata={"model_name": "PatchCore-Mock", "backbone": "wide_resnet50_2"},
        vision_score=vision_score,
        is_blurred=not is_valid,
        is_occluded=False,
        optical_health=OpticalHealthStatus(
            is_valid=is_valid,
            laplacian_var=200.0 if is_valid else 50.0,
            mean_brightness=120.0,
            degradation_reason=None if is_valid else "OPTICAL_BLUR_DETECTED",
        ),
        heatmap=None,
        latency_ms=8.0,
        metadata={},
    )


def create_dummy_sensor_reading(
    sensor_score: float = 0.05,
    machine_state: MachineState = MachineState.RUNNING,
    is_degraded: bool = False,
) -> SensorReading:
    """Helper creating mock SensorReading."""
    return SensorReading(
        reading_id="sensor-1234",
        timestamp_utc="2026-08-27T15:30:00.000Z",
        machine_id="press_unit_04",
        machine_state=machine_state,
        vibration_rms=0.45,
        temperature_c=62.0,
        current_amps=12.0,
        missing_channels=["current"] if is_degraded else [],
        is_degraded=is_degraded,
        sensor_score=sensor_score,
        sensor_breakdown={"vibration_zscore": 0.1, "temperature_zscore": 0.1, "current_zscore": 0.1},
    )


def test_baseline_mode_instantaneous_trigger() -> None:
    """Verify that BASELINE mode triggers immediately on a single frame spike without history."""
    cfg = PolicyConfig(policy_mode=PolicyMode.BASELINE)
    engine = TemporalPolicyEngine(config=cfg)

    # Step 1: Single spike above tau_high (0.80)
    inf_spike = create_dummy_inference_result(0.85)
    sens_nom = create_dummy_sensor_reading(0.10)
    dec1 = engine.evaluate(inf_spike, sens_nom)

    assert dec1.risk_state == RiskState.HIGH_SEVERITY
    assert dec1.trigger_reason == TriggerReason.SUSTAINED_VISION_ANOMALY
    assert dec1.cooldown_remaining == 0

    # Step 2: Return to normal immediately drops to NORMAL
    inf_norm = create_dummy_inference_result(0.10)
    dec2 = engine.evaluate(inf_norm, sens_nom)
    assert dec2.risk_state == RiskState.NORMAL


def test_ema_only_mode_smoothing_without_window() -> None:
    """Verify that EMA_ONLY applies exponential smoothing without sliding window confirmation."""
    cfg = PolicyConfig(
        policy_mode=PolicyMode.EMA_ONLY,
        temporal_smoothing=TemporalSmoothingConfig(alpha_vision=0.5, alpha_sensor=0.5),
    )
    engine = TemporalPolicyEngine(config=cfg)

    # Step 1: Single spike 0.90 -> EMA = 0.90 -> HIGH_SEVERITY
    dec1 = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.10))
    assert dec1.risk_state == RiskState.HIGH_SEVERITY

    # Step 2: Drop to 0.0 -> EMA = 0.5 * 0.0 + 0.5 * 0.90 = 0.45 -> NORMAL
    dec2 = engine.evaluate(create_dummy_inference_result(0.0), create_dummy_sensor_reading(0.10))
    assert dec2.risk_state == RiskState.NORMAL
    assert dec2.smoothed_scores["vision_ema"] == pytest.approx(0.45, rel=1e-3)


def test_ema_kofn_mode_window_confirmation_suppresses_spikes() -> None:
    """Verify that EMA_KOFN mode suppresses isolated 1-2 frame spikes and requires k frames."""
    cfg = PolicyConfig(
        policy_mode=PolicyMode.EMA_KOFN,
        confirmation_window=ConfirmationWindowConfig(window_size_n=5, consecutive_k=3),
    )
    engine = TemporalPolicyEngine(config=cfg)

    # Step 1 & 2: 2 spikes (less than k=3)
    dec1 = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.10))
    dec2 = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.10))
    assert dec1.risk_state != RiskState.HIGH_SEVERITY
    assert dec2.risk_state != RiskState.HIGH_SEVERITY

    # Step 3: 3rd spike -> hits k=3 -> HIGH_SEVERITY
    dec3 = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.10))
    assert dec3.risk_state == RiskState.HIGH_SEVERITY
    assert dec3.cooldown_remaining == 0


def test_no_cooldown_mode_emits_repeated_escalations() -> None:
    """Verify that NO_COOLDOWN executes multi-modal cascade but emits continuous HIGH_SEVERITY without latching."""
    cfg = PolicyConfig(
        policy_mode=PolicyMode.NO_COOLDOWN,
        confirmation_window=ConfirmationWindowConfig(window_size_n=5, consecutive_k=2),
    )
    engine = TemporalPolicyEngine(config=cfg)

    # Prime confirmation
    engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.85))
    dec1 = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.85))
    assert dec1.risk_state == RiskState.HIGH_SEVERITY
    assert dec1.cooldown_remaining == 0

    # Next step: Still HIGH_SEVERITY (no cooldown suppression)
    dec2 = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.85))
    assert dec2.risk_state == RiskState.HIGH_SEVERITY
    assert dec2.cooldown_remaining == 0


def test_no_fusion_mode_ignores_physical_sensors() -> None:
    """Verify that NO_FUSION mode ignores cross-modal divergence and sensor degradation."""
    cfg = PolicyConfig(
        policy_mode=PolicyMode.NO_FUSION,
        confirmation_window=ConfirmationWindowConfig(window_size_n=5, consecutive_k=2),
    )
    engine = TemporalPolicyEngine(config=cfg)

    # Prime confirmation with high vision, nominal sensor
    engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.05))
    dec = engine.evaluate(create_dummy_inference_result(0.90), create_dummy_sensor_reading(0.05))

    # In FULL_POLICY, |0.90 - 0.05| >= 0.45 would trigger CROSS_MODAL_DISCREPANCY
    # In NO_FUSION, sensor is ignored, so it escalates straight to HIGH_SEVERITY
    assert dec.risk_state == RiskState.HIGH_SEVERITY
    assert dec.trigger_reason == TriggerReason.SUSTAINED_VISION_ANOMALY


def test_scenario_config_loading() -> None:
    """Verify loading all 6 standardized scenario YAML configurations."""
    scenarios_dir = Path("/home/sengar/edge-inspection-runtime/configs/scenarios")
    expected_names = [
        "nominal",
        "transient_glitches",
        "sustained_defects",
        "sensor_drift_dropout",
        "network_partitions",
        "distribution_shift",
    ]

    for name in expected_names:
        yaml_path = scenarios_dir / f"{name}.yaml"
        assert yaml_path.is_file(), f"Missing scenario file: {yaml_path}"
        sc = load_scenario_config(yaml_path)
        assert isinstance(sc, ScenarioConfig)
        assert sc.name == name
        assert sc.total_steps == 300


def test_operator_workload_metrics(tmp_path: Path) -> None:
    """Verify review_load_per_hour and operator_overload_fraction computation."""
    db_path = tmp_path / "test_metrics_audit.db"
    audit_db = AuditLogDB(db_path=str(db_path))

    cfg = PolicyConfig(policy_mode=PolicyMode.BASELINE)
    engine = TemporalPolicyEngine(config=cfg)

    for i in range(300):
        # 10 anomalous steps
        v_score = 0.85 if (i % 30 == 0) else 0.10
        dec = engine.evaluate(create_dummy_inference_result(v_score), create_dummy_sensor_reading(0.10))
        audit_db.insert_risk_event(dec)

    evaluator = BenchmarkEvaluator(audit_db=audit_db)
    metrics = evaluator.compute_metrics(ground_truth_defect_steps=[0, 30, 60])

    assert metrics["total_steps"] == 300
    assert metrics["policy_escalations"] == 10
    assert metrics["review_load_per_hour"] > 0.0
    assert 0.0 <= metrics["operator_overload_fraction"] <= 1.0
    assert 0.0 <= metrics["true_positive_rate"] <= 1.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0


def test_table_generation_functions(tmp_path: Path) -> None:
    """Verify aggregate_ablation_results and table generators produce valid LaTeX and Markdown."""
    res_dir = tmp_path / "ablation_results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Create mock result JSONs
    for sc in ["nominal", "transient_glitches"]:
        for mode in ["BASELINE", "FULL_POLICY"]:
            for seed in [42, 43]:
                file_p = res_dir / sc / f"{mode}_seed{seed}.json"
                file_p.parent.mkdir(parents=True, exist_ok=True)
                dummy_payload = {
                    "scenario": sc,
                    "policy_mode": mode,
                    "seed": seed,
                    "metrics": {
                        "alert_suppression_factor": 0.95 if mode == "FULL_POLICY" else 0.0,
                        "mean_detection_latency_frames": 3.0 if mode == "FULL_POLICY" else 0.0,
                        "review_load_per_hour": 10.0 if mode == "FULL_POLICY" else 100.0,
                        "operator_overload_fraction": 0.05 if mode == "FULL_POLICY" else 1.0,
                        "fatigue_delay_tradeoff_index": 0.237 if mode == "FULL_POLICY" else 0.0,
                        "true_positive_rate": 1.0,
                        "false_positive_rate": 0.01 if mode == "FULL_POLICY" else 0.20,
                        "cross_modal_discrepancy_rate": 0.0,
                        "hardware_degraded_rate": 0.0,
                    }
                }
                with open(file_p, "w", encoding="utf-8") as f:
                    json.dump(dummy_payload, f)

    aggregated = aggregate_ablation_results(res_dir)
    assert "nominal" in aggregated["scenarios"]
    assert "transient_glitches" in aggregated["scenarios"]

    latex_tbl = generate_ablation_latex_table(aggregated)
    assert r"\begin{table*}" in latex_tbl
    assert r"\end{table*}" in latex_tbl

    md_tbl = generate_ablation_markdown_table(aggregated)
    assert "# Comprehensive Ablation Study Master Results Table" in md_tbl
    assert "| **transient_glitches** | `FULL_POLICY` |" in md_tbl


def test_no_divergence_mode_ablation() -> None:
    """Verify that NO_DIVERGENCE mode executes cascade but bypasses cross-modal divergence trigger."""
    cfg = PolicyConfig(
        policy_mode=PolicyMode.NO_DIVERGENCE,
        confirmation_window=ConfirmationWindowConfig(window_size_n=5, consecutive_k=2),
    )
    engine = TemporalPolicyEngine(config=cfg)

    # Moderate vision anomaly with nominal sensor
    inf = create_dummy_inference_result(0.60)
    sens = create_dummy_sensor_reading(0.05)

    dec = engine.evaluate(inf, sens)
    assert dec.trigger_reason != TriggerReason.CROSS_MODAL_DISCREPANCY


def test_no_state_gating_mode_ablation() -> None:
    """Verify that NO_STATE_GATING mode bypasses operational state suppression."""
    cfg = PolicyConfig(
        policy_mode=PolicyMode.NO_STATE_GATING,
        confirmation_window=ConfirmationWindowConfig(window_size_n=5, consecutive_k=2),
    )
    engine = TemporalPolicyEngine(config=cfg)

    inf = create_dummy_inference_result(0.90)
    sens_idle = create_dummy_sensor_reading(0.10, machine_state=MachineState.IDLE)

    # 2 consecutive high vision readings
    engine.evaluate(inf, sens_idle)
    dec2 = engine.evaluate(inf, sens_idle)

    assert dec2.risk_state == RiskState.HIGH_SEVERITY
    assert dec2.trigger_reason == TriggerReason.SUSTAINED_VISION_ANOMALY
