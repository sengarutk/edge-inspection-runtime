"""Integration tests for Streamlit dashboard application functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from src.audit_log import AuditLogDB
from src.config import AuditConfig
from src.dashboard.app import get_database, get_evidence_mgr, get_system_status
from src.evidence_manager import EvidenceManager
from src.inference_service import InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TriggerReason
from src.sensor_simulator import MachineState, SensorReading


def test_get_system_status_helper() -> None:
    """Test get_system_status helper with various event configurations."""
    status, badge = get_system_status([])
    assert status == "OPERATIONAL"
    assert "healthy" in badge

    status, badge = get_system_status([{"risk_state": "HIGH_SEVERITY", "is_degraded": 0}])
    assert status == "CRITICAL LOCKOUT"
    assert "critical" in badge

    status, badge = get_system_status([{"risk_state": "REVIEW_REQUIRED", "is_degraded": 1}])
    assert status == "DEGRADED REVIEW"
    assert "warning" in badge

    status, badge = get_system_status([{"risk_state": "NORMAL", "is_degraded": 0}])
    assert status == "OPERATIONAL"
    assert "healthy" in badge


def test_factories_initialization(tmp_path: Path) -> None:
    """Test get_database and get_evidence_mgr factories."""
    db_file = tmp_path / "factory_audit.db"
    db = get_database(db_path=str(db_file))
    assert db is not None
    db.close()

    default_db = get_database()
    assert default_db is not None
    default_db.close()

    ev_mgr = get_evidence_mgr(storage_dir=str(tmp_path / "factory_ev"))
    assert ev_mgr is not None


def test_streamlit_app_rendering(tmp_path: Path) -> None:
    """Test Streamlit app rendering using AppTest with sufficient timeout."""
    app_file = Path(__file__).parent.parent / "src" / "dashboard" / "app.py"

    at = AppTest.from_file(str(app_file), default_timeout=15)
    at.run(timeout=15)
    assert not at.exception