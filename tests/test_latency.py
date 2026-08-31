"""Latency percentile profiling and hard deadline SLA compliance tests."""

from pathlib import Path
import numpy as np
import pytest

from src.audit_log import AuditLogDB
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.metrics import BenchmarkEvaluator
from src.policy import PolicyDecision, RiskState, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


def test_deadline_miss_and_latency_percentiles(tmp_path: Path) -> None:
    """Verify BenchmarkEvaluator accurately profiles p50, p95, p99 and 30 FPS deadline misses."""
    db_path = tmp_path / "lat_audit.db"
    db = AuditLogDB(db_path=str(db_path))

    # Generate 100 cycles with controlled latencies:
    # 90 cycles at 8.0ms (compliant), 5 cycles at 20.0ms (compliant), 5 cycles at 45.0ms (deadline miss > 33.333ms)
    latencies = [8.0] * 90 + [20.0] * 5 + [45.0] * 5

    for idx, lat in enumerate(latencies):
        inf = InferenceResult(
            frame_id=f"f-{idx}",
            timestamp_utc="2026-08-27T15:00:00.000Z",
            camera_id="line1_overhead_cam01",
            model_metadata={"name": "mock"},
            vision_score=0.1,
            is_blurred=False,
            is_occluded=False,
            optical_health=OpticalHealthStatus(is_valid=True, laplacian_var=200.0, mean_brightness=120.0),
            heatmap=None,
            latency_ms=lat,
        )
        sens = SensorReading(
            reading_id=f"s-{idx}",
            timestamp_utc="2026-08-27T15:00:00.000Z",
            machine_id="press_unit_04",
            machine_state=MachineState.RUNNING,
            vibration_rms=0.4,
            temperature_c=55.0,
            current_amps=10.0,
            missing_channels=[],
            is_degraded=False,
            sensor_score=0.1,
        )
        dec = PolicyDecision(
            timestamp_utc="2026-08-27T15:00:00.000Z",
            camera_id="line1_overhead_cam01",
            machine_id="press_unit_04",
            machine_state=MachineState.RUNNING,
            risk_state=RiskState.NORMAL,
            trigger_reason=TriggerReason.NOMINAL_OPERATION,
            raw_scores={"vision_raw": 0.1, "sensor_raw": 0.1},
            smoothed_scores={"vision_ema": 0.1, "sensor_ema": 0.1},
            window_stats={},
            cooldown_remaining=0,
            is_degraded=False,
        )
        db.insert_telemetry(sens, inf)
        db.insert_risk_event(dec)

    evaluator = BenchmarkEvaluator(audit_db=db)
    metrics = evaluator.compute_metrics(frame_interval_ms=33.333)

    assert metrics["total_steps"] == 100
    assert metrics["deadline_miss_count"] == 5
    assert metrics["deadline_miss_rate"] == 0.05
    assert metrics["latency_p50_ms"] == 8.0
    assert metrics["latency_max_ms"] == 45.0
    assert metrics["latency_p95_ms"] >= 20.0
    assert metrics["latency_p99_ms"] >= 45.0

    db.close()
