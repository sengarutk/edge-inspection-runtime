"""Policy ablation mode verification tests for NO_DIVERGENCE and NO_STATE_GATING."""

from pathlib import Path
import pytest

from src.config import PolicyConfig, PolicyMode, ThresholdsConfig
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


def make_inf(vision_score: float) -> InferenceResult:
    return InferenceResult(
        frame_id="f-ablate",
        timestamp_utc="2026-08-27T15:00:00.000Z",
        camera_id="line1_overhead_cam01",
        model_metadata={},
        vision_score=vision_score,
        is_blurred=False,
        is_occluded=False,
        optical_health=OpticalHealthStatus(is_valid=True, laplacian_var=200.0, mean_brightness=120.0),
        heatmap=None,
        latency_ms=8.0,
    )


def make_sens(sensor_score: float = 0.05, machine_state: MachineState = MachineState.RUNNING) -> SensorReading:
    return SensorReading(
        reading_id="s-ablate",
        timestamp_utc="2026-08-27T15:00:00.000Z",
        machine_id="press_unit_04",
        machine_state=machine_state,
        vibration_rms=0.45,
        temperature_c=60.0,
        current_amps=12.0,
        missing_channels=[],
        is_degraded=False,
        sensor_score=sensor_score,
    )


def test_no_divergence_policy_mode() -> None:
    """Verify NO_DIVERGENCE bypasses cross-modal divergence trigger."""
    inf = make_inf(0.60)
    sens = make_sens(0.05)

    eng_full = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.FULL_POLICY))
    d_full = eng_full.evaluate(inf, sens)
    assert d_full.risk_state == RiskState.REVIEW_REQUIRED
    assert d_full.trigger_reason == TriggerReason.CROSS_MODAL_DISCREPANCY

    eng_nodiv = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.NO_DIVERGENCE))
    d_nodiv = eng_nodiv.evaluate(inf, sens)
    assert d_nodiv.trigger_reason != TriggerReason.CROSS_MODAL_DISCREPANCY


def test_no_state_gating_policy_mode() -> None:
    """Verify NO_STATE_GATING allows HIGH_SEVERITY alarms during IDLE/MAINTENANCE states."""
    inf = make_inf(0.95)
    sens_idle = make_sens(0.10, machine_state=MachineState.IDLE)

    eng_full = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.FULL_POLICY))
    d_full = [eng_full.evaluate(inf, sens_idle) for _ in range(4)][-1]
    assert d_full.risk_state == RiskState.REVIEW_REQUIRED
    assert d_full.trigger_reason == TriggerReason.STATE_GATED_SUPPRESSION

    eng_nogate = TemporalPolicyEngine(config=PolicyConfig(policy_mode=PolicyMode.NO_STATE_GATING))
    d_nogate = [eng_nogate.evaluate(inf, sens_idle) for _ in range(4)][-1]
    assert d_nogate.risk_state == RiskState.HIGH_SEVERITY
    assert d_nogate.trigger_reason == TriggerReason.SUSTAINED_VISION_ANOMALY
