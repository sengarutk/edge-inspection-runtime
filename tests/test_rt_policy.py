"""Unit tests for temporal decision policies and cross-modal risk engine."""

import uuid
from datetime import datetime, timezone
import pytest

from src.config import (
    ConfirmationWindowConfig,
    CooldownConfig,
    MachineStateGatingConfig,
    PolicyConfig,
    TemporalSmoothingConfig,
    ThresholdsConfig,
)
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


@pytest.fixture
def policy_config() -> PolicyConfig:
    """Fixture providing standard test policy configuration."""
    return PolicyConfig(
        temporal_smoothing=TemporalSmoothingConfig(alpha_vision=0.35, alpha_sensor=0.25),
        confirmation_window=ConfirmationWindowConfig(window_size_n=10, consecutive_k=4),
        cooldown=CooldownConfig(cooldown_steps=15),
        thresholds=ThresholdsConfig(
            vision_medium=0.50,
            vision_high=0.80,
            sensor_anomaly=0.70,
            cross_modal_divergence=0.45,
        ),
        machine_state_gating=MachineStateGatingConfig(
            suppress_high_severity_on_idle=True,
            suppress_high_severity_on_maintenance=True,
        ),
    )


@pytest.fixture
def engine(policy_config: PolicyConfig) -> TemporalPolicyEngine:
    """Fixture providing an initialized TemporalPolicyEngine."""
    return TemporalPolicyEngine(config=policy_config)


def make_inference_result(
    vision_score: float = 0.05,
    is_valid_optical: bool = True,
    degradation_reason: str | None = None,
) -> InferenceResult:
    """Helper to construct synthetic InferenceResult for testing."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return InferenceResult(
        frame_id=str(uuid.uuid4()),
        timestamp_utc=now_utc,
        camera_id="line1_overhead_cam01",
        model_metadata={"model_name": "patchcore_mock", "engine": "mock", "version": "v1.0.0"},
        vision_score=vision_score,
        is_blurred=degradation_reason == "OPTICAL_BLURRED",
        is_occluded=degradation_reason in ("OPTICAL_OCCLUDED_DARK", "OPTICAL_OCCLUDED_BRIGHT"),
        optical_health=OpticalHealthStatus(
            is_valid=is_valid_optical,
            laplacian_var=150.0 if is_valid_optical else 20.0,
            mean_brightness=128.0 if is_valid_optical else 5.0,
            degradation_reason=degradation_reason,
        ),
        heatmap=None,
        latency_ms=8.5,
        metadata={"optical_health_valid": is_valid_optical},
    )


def make_sensor_reading(
    sensor_score: float = 0.0,
    machine_state: MachineState = MachineState.RUNNING,
    is_degraded: bool = False,
    missing_channels: list[str] | None = None,
) -> SensorReading:
    """Helper to construct synthetic SensorReading for testing."""
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    channels = missing_channels or (["current"] if is_degraded else [])
    return SensorReading(
        reading_id=str(uuid.uuid4()),
        timestamp_utc=now_utc,
        machine_id="press_unit_04",
        machine_state=machine_state,
        vibration_rms=0.45 if machine_state != MachineState.FAULT else 2.8,
        temperature_c=62.0 if machine_state != MachineState.FAULT else 88.0,
        current_amps=12.8 if machine_state != MachineState.FAULT else 22.4,
        missing_channels=channels,
        is_degraded=is_degraded,
        sensor_score=sensor_score,
        sensor_breakdown={"vibration_zscore": 0.1, "temperature_zscore": 0.1, "current_zscore": 0.1},
    )


def test_transient_spike_suppression(engine: TemporalPolicyEngine) -> None:
    """Verify that a 1-to-2 frame vision score spike of 0.95 does not trigger HIGH_SEVERITY."""
    engine.reset()

    # Step 1: Nominal baseline
    d1 = engine.evaluate(make_inference_result(0.05), make_sensor_reading(0.0))
    assert d1.risk_state == RiskState.NORMAL

    # Step 2: Sudden 1-frame spike
    d2 = engine.evaluate(make_inference_result(0.95), make_sensor_reading(0.0))
    assert d2.risk_state in (RiskState.NORMAL, RiskState.REVIEW_REQUIRED)
    assert d2.risk_state != RiskState.HIGH_SEVERITY

    # Step 3: Second frame spike
    d3 = engine.evaluate(make_inference_result(0.95), make_sensor_reading(0.0))
    assert d3.risk_state != RiskState.HIGH_SEVERITY

    # Step 4: Drop back to normal
    d4 = engine.evaluate(make_inference_result(0.05), make_sensor_reading(0.0))
    assert d4.risk_state != RiskState.HIGH_SEVERITY


def test_sustained_defect_escalation(engine: TemporalPolicyEngine) -> None:
    """Verify that sustained high scores for >= k consecutive frames trigger HIGH_SEVERITY and activate cooldown."""
    engine.reset()

    decisions = []
    # Feed sustained high vision scores (0.95)
    for _ in range(8):
        decisions.append(engine.evaluate(make_inference_result(0.95), make_sensor_reading(0.0)))

    high_decisions = [d for d in decisions if d.risk_state == RiskState.HIGH_SEVERITY]
    assert len(high_decisions) > 0

    first_high = high_decisions[0]
    assert first_high.trigger_reason == TriggerReason.SUSTAINED_VISION_ANOMALY
    assert first_high.cooldown_remaining == 15


def test_anti_fatigue_cooldown_verification(engine: TemporalPolicyEngine) -> None:
    """Verify that subsequent high scores during cooldown are suppressed from emitting repeated HIGH_SEVERITY."""
    engine.reset()

    # Trigger initial high severity
    for _ in range(6):
        d = engine.evaluate(make_inference_result(0.95), make_sensor_reading(0.0))

    assert engine.cooldown_counter > 0

    # Next steps during cooldown with ongoing defect
    cooldown_decisions = []
    for _ in range(10):
        d = engine.evaluate(make_inference_result(0.95), make_sensor_reading(0.0))
        cooldown_decisions.append(d)
        assert d.risk_state != RiskState.HIGH_SEVERITY
        assert d.trigger_reason == TriggerReason.COOLDOWN_ACTIVE


def test_cross_modal_divergence_detection(engine: TemporalPolicyEngine) -> None:
    """Verify high vision score coupled with low sensor score triggers REVIEW_REQUIRED with CROSS_MODAL_DISCREPANCY."""
    engine.reset()

    # Warm up with baseline
    engine.evaluate(make_inference_result(0.05), make_sensor_reading(0.0))

    # Feed 2-3 frames of high vision creating divergence >= 0.45 before high severity k=4 is reached
    engine.evaluate(make_inference_result(0.85), make_sensor_reading(0.0))
    d = engine.evaluate(make_inference_result(0.85), make_sensor_reading(0.0))

    assert d.risk_state == RiskState.REVIEW_REQUIRED
    assert d.trigger_reason in (TriggerReason.CROSS_MODAL_DISCREPANCY, TriggerReason.SUSTAINED_VISION_ANOMALY)
    assert d.diagnostics["cross_modal_divergence"] >= 0.45


def test_sustained_sensor_only_anomaly(engine: TemporalPolicyEngine) -> None:
    """Verify that sustained sensor anomaly while vision is normal triggers REVIEW_REQUIRED with SUSTAINED_SENSOR_ANOMALY."""
    engine.reset()

    decisions = []
    for _ in range(5):
        decisions.append(engine.evaluate(make_inference_result(0.05), make_sensor_reading(0.95)))

    sensor_anom_decisions = [
        d for d in decisions if d.trigger_reason == TriggerReason.SUSTAINED_SENSOR_ANOMALY
    ]
    assert len(sensor_anom_decisions) > 0
    assert sensor_anom_decisions[0].risk_state == RiskState.REVIEW_REQUIRED


def test_machine_state_gating_idle(engine: TemporalPolicyEngine) -> None:
    """Verify that during IDLE state, high anomaly inputs are suppressed from HIGH_SEVERITY."""
    engine.reset()

    for _ in range(6):
        d = engine.evaluate(
            make_inference_result(0.95),
            make_sensor_reading(0.0, machine_state=MachineState.IDLE),
        )
        assert d.risk_state != RiskState.HIGH_SEVERITY
        assert d.trigger_reason == TriggerReason.STATE_GATED_SUPPRESSION


def test_machine_state_gating_maintenance(engine: TemporalPolicyEngine) -> None:
    """Verify that during MAINTENANCE state, high anomaly inputs are suppressed from HIGH_SEVERITY."""
    engine.reset()

    for _ in range(6):
        d = engine.evaluate(
            make_inference_result(0.95),
            make_sensor_reading(0.85, machine_state=MachineState.MAINTENANCE),
        )
        assert d.risk_state != RiskState.HIGH_SEVERITY
        assert d.trigger_reason == TriggerReason.STATE_GATED_SUPPRESSION


def test_machine_state_fault_escalates_immediately(engine: TemporalPolicyEngine) -> None:
    """Verify that FAULT machine state escalates to HIGH_SEVERITY with CRITICAL_MACHINE_FAULT."""
    engine.reset()

    d = engine.evaluate(
        make_inference_result(0.05),
        make_sensor_reading(0.90, machine_state=MachineState.FAULT),
    )

    assert d.risk_state == RiskState.HIGH_SEVERITY
    assert d.trigger_reason == TriggerReason.CRITICAL_MACHINE_FAULT


def test_optical_failure_fallback_nominal_sensor(engine: TemporalPolicyEngine) -> None:
    """Verify that when optical_health.is_valid=False, decision safely routes to OPTICAL_DEGRADATION_FALLBACK."""
    engine.reset()

    d = engine.evaluate(
        make_inference_result(0.0, is_valid_optical=False, degradation_reason="OPTICAL_BLURRED"),
        make_sensor_reading(0.0),
    )

    assert d.risk_state == RiskState.REVIEW_REQUIRED
    assert d.trigger_reason == TriggerReason.OPTICAL_DEGRADATION_FALLBACK
    assert d.is_degraded is True


def test_optical_failure_fallback_during_critical_fault(engine: TemporalPolicyEngine) -> None:
    """Verify that optical failure during critical physical fault escalates to CRITICAL_MACHINE_FAULT."""
    engine.reset()

    d = engine.evaluate(
        make_inference_result(0.0, is_valid_optical=False, degradation_reason="OPTICAL_OCCLUDED_DARK"),
        make_sensor_reading(0.99, machine_state=MachineState.FAULT),
    )

    assert d.risk_state == RiskState.HIGH_SEVERITY
    assert d.trigger_reason == TriggerReason.CRITICAL_MACHINE_FAULT


def test_sensor_degradation_fallback(engine: TemporalPolicyEngine) -> None:
    """Verify that sensor degradation with high vision score triggers SENSOR_DEGRADATION_FALLBACK."""
    engine.reset()

    d = engine.evaluate(
        make_inference_result(0.85),
        make_sensor_reading(0.0, is_degraded=True, missing_channels=["current"]),
    )

    assert d.risk_state == RiskState.REVIEW_REQUIRED
    assert d.trigger_reason == TriggerReason.SENSOR_DEGRADATION_FALLBACK
    assert d.is_degraded is True


def test_multi_modal_joint_fault_escalation(engine: TemporalPolicyEngine) -> None:
    """Verify concurrent physical sensor fault and visual defect trigger MULTI_MODAL_CONFIRMED_FAULT."""
    engine.reset()

    # Pre-condition with 4 sustained multi-modal high frames
    for _ in range(4):
        d = engine.evaluate(
            make_inference_result(0.95),
            make_sensor_reading(0.95, machine_state=MachineState.RUNNING),
        )

    assert d.risk_state == RiskState.HIGH_SEVERITY
    assert d.trigger_reason == TriggerReason.MULTI_MODAL_CONFIRMED_FAULT
    assert d.cooldown_remaining == 15


def test_engine_reset_behavior(engine: TemporalPolicyEngine) -> None:
    """Verify reset() clears histories, EMAs, counters, and statistics."""
    engine.reset()

    for _ in range(5):
        engine.evaluate(make_inference_result(0.90), make_sensor_reading(0.90))

    assert engine.total_evaluations == 5
    assert engine.vision_ema is not None
    assert len(engine.vision_history) > 0

    engine.reset()

    assert engine.total_evaluations == 0
    assert engine.total_escalations == 0
    assert engine.vision_ema is None
    assert engine.sensor_ema is None
    assert len(engine.vision_history) == 0
    assert len(engine.sensor_history) == 0
    assert engine.cooldown_counter == 0


def test_telemetry_stats_reporting(engine: TemporalPolicyEngine) -> None:
    """Verify get_telemetry_stats returns complete and accurate counters."""
    engine.reset()

    for _ in range(10):
        engine.evaluate(make_inference_result(0.05), make_sensor_reading(0.0))

    stats = engine.get_telemetry_stats()
    assert stats["total_evaluations"] == 10
    assert stats["total_escalations"] == 0
    assert 0.0 <= stats["suppression_rate"] <= 1.0