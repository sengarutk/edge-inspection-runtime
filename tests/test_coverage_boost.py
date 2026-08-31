"""Extended Unit Tests for 100% Policy Branch & Metrics Coverage."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from streamlit.testing.v1 import AppTest

from src.audit_log import AuditLogDB
from src.config import (
    ConfirmationWindowConfig,
    CooldownConfig,
    InferenceConfig,
    MachineStateGatingConfig,
    PolicyConfig,
    PolicyMode,
    ScenarioConfig,
    SensorConfig,
    TemporalSmoothingConfig,
    ThresholdsConfig,
)
from src.dashboard.app import get_database, get_evidence_mgr, get_system_status
from src.evidence_manager import EvidenceManager
from src.fault_injector import ChaosFaultConfig, FaultInjector, FaultType
from src.inference_service import InferenceEngine, InferenceResult, OpticalHealthStatus
from src.metrics import BenchmarkEvaluator, aggregate_ablation_results
from src.mqtt_publisher import ResilientMQTTPublisher
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorReading, SensorSimulator
from src.spooler import DiskSpooler


def make_inf(vision_score: float, is_valid: bool = True, degradation_reason: str = None) -> InferenceResult:
    return InferenceResult(
        frame_id="frame-boost-1",
        timestamp_utc="2026-08-27T15:30:00.000Z",
        camera_id="line1_overhead_cam01",
        model_metadata={"model_name": "PatchCore-Mock"},
        vision_score=vision_score,
        is_blurred=not is_valid,
        is_occluded=False,
        optical_health=OpticalHealthStatus(
            is_valid=is_valid,
            laplacian_var=200.0 if is_valid else 50.0,
            mean_brightness=120.0,
            degradation_reason=degradation_reason,
        ),
        heatmap=None,
        latency_ms=7.5,
        metadata={},
    )


def make_sens(
    sensor_score: float = 0.05,
    machine_state: MachineState = MachineState.RUNNING,
    is_degraded: bool = False,
) -> SensorReading:
    return SensorReading(
        reading_id="sensor-boost-1",
        timestamp_utc="2026-08-27T15:30:00.000Z",
        machine_id="press_unit_04",
        machine_state=machine_state,
        vibration_rms=0.45,
        temperature_c=62.0,
        current_amps=12.0,
        missing_channels=["temperature"] if is_degraded else [],
        is_degraded=is_degraded,
        sensor_score=sensor_score,
        sensor_breakdown={"vibration_zscore": 0.1, "temperature_zscore": 0.1, "current_zscore": 0.1},
    )


def test_policy_branches_edge_cases() -> None:
    """Exercise comprehensive branch paths across all policy modes."""
    # 1. BASELINE medium score
    eng_base = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.BASELINE))
    d_base_med = eng_base.evaluate(make_inf(0.55), make_sens(0.10))
    assert d_base_med.risk_state == RiskState.REVIEW_REQUIRED

    # 2. EMA_ONLY medium score
    eng_ema = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.EMA_ONLY))
    d_ema_med = eng_ema.evaluate(make_inf(0.55), make_sens(0.10))
    assert d_ema_med.risk_state == RiskState.REVIEW_REQUIRED

    # 3. EMA_KOFN medium score
    eng_kofn = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.EMA_KOFN))
    d_kofn_med = eng_kofn.evaluate(make_inf(0.55), make_sens(0.10))
    assert d_kofn_med.risk_state == RiskState.REVIEW_REQUIRED

    # 4. NO_FUSION: Optical degraded, Machine state FAULT, IDLE gating, and medium exceedance
    eng_nf = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.NO_FUSION))
    
    # 4a. Optical degraded
    d_nf_opt = eng_nf.evaluate(make_inf(0.0, is_valid=False, degradation_reason="BLUR"), make_sens(0.10))
    assert d_nf_opt.risk_state == RiskState.REVIEW_REQUIRED
    assert d_nf_opt.trigger_reason == TriggerReason.OPTICAL_DEGRADATION_FALLBACK

    # 4b. Machine state FAULT
    d_nf_fault = eng_nf.evaluate(make_inf(0.10), make_sens(0.10, machine_state=MachineState.FAULT))
    assert d_nf_fault.risk_state == RiskState.HIGH_SEVERITY
    assert d_nf_fault.trigger_reason == TriggerReason.CRITICAL_MACHINE_FAULT

    # 4c. Machine state IDLE gating with high vision
    eng_nf.reset()
    eng_nf.evaluate(make_inf(0.90), make_sens(0.10, machine_state=MachineState.IDLE))
    d_nf_idle = eng_nf.evaluate(make_inf(0.90), make_sens(0.10, machine_state=MachineState.IDLE))
    assert d_nf_idle.risk_state == RiskState.REVIEW_REQUIRED
    assert d_nf_idle.trigger_reason == TriggerReason.STATE_GATED_SUPPRESSION

    # 4d. Machine state IDLE with nominal vision
    eng_nf.reset()
    d_nf_idle_nom = eng_nf.evaluate(make_inf(0.10), make_sens(0.10, machine_state=MachineState.IDLE))
    assert d_nf_idle_nom.risk_state == RiskState.NORMAL

    # 4e. Medium vision score in NO_FUSION
    eng_nf.reset()
    d_nf_med = eng_nf.evaluate(make_inf(0.55), make_sens(0.10))
    assert d_nf_med.risk_state == RiskState.REVIEW_REQUIRED

    # 5. FULL_POLICY & NO_COOLDOWN: Sensor degraded with high vision
    eng_full = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.FULL_POLICY))
    d_sens_deg = eng_full.evaluate(make_inf(0.85), make_sens(0.10, is_degraded=True))
    assert d_sens_deg.risk_state == RiskState.REVIEW_REQUIRED
    assert d_sens_deg.trigger_reason == TriggerReason.SENSOR_DEGRADATION_FALLBACK

    # 6. IDLE state gating in FULL_POLICY with anomaly
    eng_full.reset()
    d_full_idle = eng_full.evaluate(make_inf(0.85), make_sens(0.10, machine_state=MachineState.IDLE))
    assert d_full_idle.risk_state == RiskState.REVIEW_REQUIRED
    assert d_full_idle.trigger_reason == TriggerReason.STATE_GATED_SUPPRESSION

    # 7. MAINTENANCE state gating in FULL_POLICY nominal
    eng_full.reset()
    d_full_maint = eng_full.evaluate(make_inf(0.05), make_sens(0.05, machine_state=MachineState.MAINTENANCE))
    assert d_full_maint.risk_state == RiskState.NORMAL

    # 8. Cross-modal discrepancy triggering
    eng_full.reset()
    # High vision (0.60 > 0.50), low sensor (0.05), delta = 0.55 >= 0.45
    d_div = eng_full.evaluate(make_inf(0.60), make_sens(0.05))
    assert d_div.risk_state == RiskState.REVIEW_REQUIRED


def test_metrics_rolling_windows_and_defect_steps(tmp_path: Path) -> None:
    """Exercise rolling window calculations with more than 300 steps and ground truth defect steps."""
    db_path = tmp_path / "long_audit.db"
    db = AuditLogDB(db_path=str(db_path))

    cfg = PolicyConfig(policy_mode=PolicyMode.FULL_POLICY)
    eng = TemporalPolicyEngine(config=cfg)

    # Simulate 500 steps (exceeds rolling 5-minute window of ~270 frames at 33.33ms)
    gt_defect_steps = []
    for i in range(500):
        if 100 <= i < 150:
            gt_defect_steps.append(i)
            v_score = 0.95
            s_score = 0.95
        else:
            v_score = 0.05
            s_score = 0.05
        dec = eng.evaluate(make_inf(v_score), make_sens(s_score))
        db.insert_risk_event(dec)

    evaluator = BenchmarkEvaluator(audit_db=db)
    metrics = evaluator.compute_metrics(ground_truth_defect_steps=gt_defect_steps)

    assert metrics["total_steps"] == 500
    assert metrics["policy_escalations"] > 0
    assert 0.0 <= metrics["operator_overload_fraction"] <= 1.0
    assert metrics["true_positive_rate"] > 0.0
    assert 0.0 <= metrics["false_positive_rate"] <= 1.0
