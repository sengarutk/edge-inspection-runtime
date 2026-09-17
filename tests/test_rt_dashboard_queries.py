"""Unit tests for dashboard-specific database query methods."""

from pathlib import Path
import pytest

from src.audit_log import AuditLogDB
from src.config import AuditConfig
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


@pytest.fixture
def audit_db(tmp_path: Path) -> AuditLogDB:
    """Fixture providing an isolated AuditLogDB."""
    db_file = tmp_path / "dashboard_test.db"
    db = AuditLogDB(config=AuditConfig(db_path=str(db_file)))
    yield db
    db.close()


def test_dashboard_queries(audit_db: AuditLogDB) -> None:
    """Test query_recent_telemetry, query_recent_health, and get_actionable_event_by_id."""
    # 1. Insert telemetry
    r = SensorReading(
        timestamp_utc="2026-08-27T15:00:00.000Z",
        machine_id="press_01",
        machine_state=MachineState.RUNNING,
        vibration_rms=0.45,
        temperature_c=62.0,
        current_amps=12.5,
        missing_channels=[],
        is_degraded=False,
        sensor_score=0.02,
        sensor_breakdown={},
    )
    inf = InferenceResult(
        timestamp_utc="2026-08-27T15:00:00.000Z",
        camera_id="cam_01",
        model_metadata={},
        vision_score=0.05,
        is_blurred=False,
        is_occluded=False,
        optical_health=OpticalHealthStatus(is_valid=True, laplacian_var=120.0, mean_brightness=120.0),
        heatmap=None,
        latency_ms=8.0,
    )
    audit_db.insert_telemetry(r, inf)

    telem_rows = audit_db.query_recent_telemetry(limit=10)
    assert len(telem_rows) == 1
    assert telem_rows[0]["vibration_rms"] == 0.45

    # 2. Insert health
    audit_db.insert_system_health("camera", "HEALTHY", "30 FPS")
    health_rows = audit_db.query_recent_health(limit=10)
    assert len(health_rows) == 1
    assert health_rows[0]["component"] == "camera"

    # 3. Insert and get event by ID
    d = PolicyDecision(
        timestamp_utc="2026-08-27T15:00:00.000Z",
        camera_id="cam_01",
        machine_id="press_01",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.HIGH_SEVERITY,
        trigger_reason=TriggerReason.SUSTAINED_VISION_ANOMALY,
        raw_scores={"vision_raw": 0.9, "sensor_raw": 0.05},
        smoothed_scores={"vision_ema": 0.85, "sensor_ema": 0.05},
        window_stats={},
        cooldown_remaining=15,
        is_degraded=False,
        evidence_uri="data/evidence/test.png",
    )
    audit_db.insert_risk_event(d)

    event = audit_db.get_actionable_event_by_id(d.decision_id)
    assert event is not None
    assert event["event_id"] == d.decision_id
    assert event["evidence_uri"] == "data/evidence/test.png"

    # Query nonexistent
    assert audit_db.get_actionable_event_by_id("nonexistent_id") is None