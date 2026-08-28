"""Unit tests for statistical benchmark evaluator and reliability metrics."""

import json
from pathlib import Path
import pytest

from src.audit_log import AuditLogDB
from src.config import AuditConfig
from src.metrics import BenchmarkEvaluator
from src.policy import PolicyDecision, RiskState, TriggerReason
from src.sensor_simulator import MachineState


@pytest.fixture
def eval_db(tmp_path: Path) -> AuditLogDB:
    """Fixture providing populated AuditLogDB for metrics testing."""
    db_file = tmp_path / "metrics_audit.db"
    db = AuditLogDB(config=AuditConfig(db_path=str(db_file)))

    # Step 0: Nominal
    db.insert_risk_event(
        PolicyDecision(
            timestamp_utc="2026-08-27T15:00:00.000Z",
            camera_id="cam1",
            machine_id="press1",
            machine_state=MachineState.RUNNING,
            risk_state=RiskState.NORMAL,
            trigger_reason=TriggerReason.NOMINAL_OPERATION,
            raw_scores={"vision_raw": 0.05, "sensor_raw": 0.02},
            smoothed_scores={"vision_ema": 0.05, "sensor_ema": 0.02},
            window_stats={},
            cooldown_remaining=0,
            is_degraded=False,
        )
    )

    # Step 1: Transient spike (suppressed)
    db.insert_risk_event(
        PolicyDecision(
            timestamp_utc="2026-08-27T15:00:01.000Z",
            camera_id="cam1",
            machine_id="press1",
            machine_state=MachineState.RUNNING,
            risk_state=RiskState.REVIEW_REQUIRED,
            trigger_reason=TriggerReason.CROSS_MODAL_DISCREPANCY,
            raw_scores={"vision_raw": 0.95, "sensor_raw": 0.02},
            smoothed_scores={"vision_ema": 0.35, "sensor_ema": 0.02},
            window_stats={},
            cooldown_remaining=0,
            is_degraded=False,
        )
    )

    # Step 2: Optical degradation fallback
    db.insert_risk_event(
        PolicyDecision(
            timestamp_utc="2026-08-27T15:00:02.000Z",
            camera_id="cam1",
            machine_id="press1",
            machine_state=MachineState.RUNNING,
            risk_state=RiskState.REVIEW_REQUIRED,
            trigger_reason=TriggerReason.OPTICAL_DEGRADATION_FALLBACK,
            raw_scores={"vision_raw": 0.0, "sensor_raw": 0.02},
            smoothed_scores={"vision_ema": 0.0, "sensor_ema": 0.02},
            window_stats={},
            cooldown_remaining=0,
            is_degraded=True,
        )
    )

    # Step 3: Sustained defect escalation
    d_esc = PolicyDecision(
        timestamp_utc="2026-08-27T15:00:03.000Z",
        camera_id="cam1",
        machine_id="press1",
        machine_state=MachineState.RUNNING,
        risk_state=RiskState.HIGH_SEVERITY,
        trigger_reason=TriggerReason.SUSTAINED_VISION_ANOMALY,
        raw_scores={"vision_raw": 0.90, "sensor_raw": 0.05},
        smoothed_scores={"vision_ema": 0.85, "sensor_ema": 0.05},
        window_stats={},
        cooldown_remaining=15,
        is_degraded=False,
    )
    db.insert_risk_event(d_esc)

    # Record operator review
    db.record_operator_review(d_esc.decision_id, "CONFIRMED", "Verified crack")

    yield db
    db.close()


def test_empty_database_metrics(tmp_path: Path) -> None:
    """Test BenchmarkEvaluator handling of empty database."""
    empty_db_file = tmp_path / "empty.db"
    evaluator = BenchmarkEvaluator(db_path=str(empty_db_file))
    m = evaluator.compute_metrics()

    assert m["total_steps"] == 0
    assert m["alert_suppression_factor"] == 1.0
    assert m["operator_confirmation_precision"] == 0.0


def test_populated_database_metrics(eval_db: AuditLogDB) -> None:
    """Test metric computations on populated audit database."""
    evaluator = BenchmarkEvaluator(db_path=str(eval_db.db_path))
    m = evaluator.compute_metrics()

    assert m["total_steps"] == 4
    assert m["raw_threshold_crossings"] == 2
    assert m["policy_escalations"] == 1
    assert m["alert_suppression_factor"] == 0.50
    assert m["optical_degradations_handled"] == 1
    assert m["confirmed_defects"] == 1
    assert m["operator_confirmation_precision"] == 1.0


def test_markdown_and_latex_generation(eval_db: AuditLogDB, tmp_path: Path) -> None:
    """Test markdown and LaTeX table generation and JSON export."""
    evaluator = BenchmarkEvaluator(db_path=str(eval_db.db_path))
    md = evaluator.generate_markdown_summary()
    assert "# Industrial Edge Inspection Runtime" in md
    assert "Alert Fatigue Suppression" in md

    latex = evaluator.generate_latex_table()
    assert "\\begin{table}" in latex
    assert "\\label{tab:edge_inspection_reliability}" in latex

    json_path = tmp_path / "metrics.json"
    evaluator.export_json(json_path)
    assert json_path.is_file()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["total_steps"] == 4