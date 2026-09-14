"""Persistent SQLite Audit & Operator Review Subsystem.

Provides historical event journaling, continuous telemetry archival, and operator triage metrics.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from loguru import logger

from src.config import AuditConfig, load_mqtt_config
from src.inference_service import InferenceResult
from src.policy import PolicyDecision
from src.sensor_simulator import SensorReading


class AuditLogDB:
    """Thread-safe persistent SQLite storage for industrial audit logging and operator reviews."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        config: Optional[AuditConfig] = None,
    ) -> None:
        """Initialize SQLite audit database and schema.

        Args:
            db_path: Optional path to audit log SQLite file.
            config: Optional AuditConfig instance.
        """
        if config is not None:
            self.db_path = Path(config.db_path)
        else:
            mqtt_cfg = load_mqtt_config()
            self.db_path = Path(db_path or mqtt_cfg.audit.db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA busy_timeout = 30000;")
        self._conn.row_factory = sqlite3.Row
        self._init_db()

        logger.info(f"Initialized AuditLogDB at {self.db_path}")

    def _init_db(self) -> None:
        """Create audit, telemetry, review, and health tables if not existing."""
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    machine_id TEXT NOT NULL,
                    machine_state TEXT NOT NULL,
                    risk_state TEXT NOT NULL,
                    trigger_reason TEXT NOT NULL,
                    vision_raw REAL,
                    vision_ema REAL,
                    sensor_raw REAL,
                    sensor_ema REAL,
                    cooldown_remaining INTEGER,
                    is_degraded INTEGER DEFAULT 0,
                    frame_id TEXT,
                    reading_id TEXT,
                    evidence_uri TEXT,
                    raw_payload TEXT,
                    review_status TEXT DEFAULT 'PENDING',
                    operator_notes TEXT,
                    reviewed_at TEXT
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS telemetry_stream (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    vibration_rms REAL,
                    temperature_c REAL,
                    current_amps REAL,
                    vision_score REAL,
                    sensor_score REAL,
                    latency_ms REAL
                );
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp_utc TEXT NOT NULL,
                    component TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT
                );
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_time ON risk_events(timestamp_utc DESC);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry_stream(timestamp_utc DESC);"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_health_time ON system_health(timestamp_utc DESC);"
            )

            # Backward-compatible column migration
            try:
                cursor = self._conn.cursor()
                cursor.execute("PRAGMA table_info(telemetry_stream);")
                col_names = [col[1] for col in cursor.fetchall()]
                if "latency_ms" not in col_names:
                    cursor.execute("ALTER TABLE telemetry_stream ADD COLUMN latency_ms REAL;")
                    self._conn.commit()
            except Exception:
                pass

    def insert_risk_event(self, decision: Union[PolicyDecision, Dict[str, Any]]) -> str:
        """Insert or update a policy risk decision into the audit log.

        Args:
            decision: PolicyDecision model instance or parsed JSON dictionary.

        Returns:
            The unique event_id inserted.
        """
        if isinstance(decision, PolicyDecision):
            data = decision.to_mqtt_payload()
        else:
            data = decision

        event_id = str(data.get("event_id") or data.get("decision_id"))
        timestamp_utc = str(data.get("timestamp_utc"))
        camera_id = str(data.get("camera_id", "line1_overhead_cam01"))
        machine_id = str(data.get("machine_id", "press_unit_04"))
        machine_state = str(data.get("machine_state", "RUNNING"))
        risk_state = str(data.get("risk_state", "NORMAL"))
        trigger_reason = str(data.get("trigger_reason", "NOMINAL_OPERATION"))

        raw_scores = data.get("raw_scores", {})
        smoothed_scores = data.get("smoothed_scores", {})

        vision_raw = float(raw_scores.get("vision_raw", 0.0))
        sensor_raw = float(raw_scores.get("sensor_raw", 0.0))
        vision_ema = float(smoothed_scores.get("vision_ema", 0.0))
        sensor_ema = float(smoothed_scores.get("sensor_ema", 0.0))

        cooldown_remaining = int(data.get("cooldown_remaining", 0))
        is_degraded = 1 if data.get("is_degraded") else 0
        frame_id = data.get("frame_id")
        reading_id = data.get("reading_id")
        evidence_uri = data.get("evidence_uri")
        raw_payload = json.dumps(data)

        initial_review_status = str(data.get("review_status") or ("NOMINAL" if risk_state == "NORMAL" else "PENDING"))

        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO risk_events (
                    event_id, timestamp_utc, camera_id, machine_id, machine_state,
                    risk_state, trigger_reason, vision_raw, vision_ema, sensor_raw, sensor_ema,
                    cooldown_remaining, is_degraded, frame_id, reading_id, evidence_uri,
                    raw_payload, review_status, operator_notes, reviewed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL);
                """,
                (
                    event_id, timestamp_utc, camera_id, machine_id, machine_state,
                    risk_state, trigger_reason, vision_raw, vision_ema, sensor_raw, sensor_ema,
                    cooldown_remaining, is_degraded, frame_id, reading_id, evidence_uri,
                    raw_payload, initial_review_status,
                ),
            )
        return event_id

    def insert_telemetry(self, reading: SensorReading, inf_result: InferenceResult) -> int:
        """Record high-frequency sensor and vision telemetry into archival stream.

        Args:
            reading: SensorReading instance.
            inf_result: InferenceResult instance.

        Returns:
            Row ID inserted.
        """
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        lat_ms = float(inf_result.latency_ms) if inf_result.latency_ms is not None else 0.0
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO telemetry_stream (
                    timestamp_utc, vibration_rms, temperature_c, current_amps,
                    vision_score, sensor_score, latency_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    now_utc,
                    reading.vibration_rms,
                    reading.temperature_c,
                    reading.current_amps,
                    inf_result.vision_score,
                    reading.sensor_score,
                    lat_ms,
                ),
            )
            return cursor.lastrowid or 0

    def insert_system_health(self, component: str, status: str, details: Optional[str] = None) -> int:
        """Record system health and component heartbeat logs."""
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO system_health (timestamp_utc, component, status, details)
                VALUES (?, ?, ?, ?);
                """,
                (now_utc, component, status, details or ""),
            )
            return cursor.lastrowid or 0

    def record_operator_review(self, event_id: str, action: Optional[str] = None, notes: Optional[str] = None, review_status: Optional[str] = None) -> bool:
        """Record human operator audit review (CONFIRMED or REJECTED).

        Args:
            event_id: The unique decision/event ID.
            action: Review status action ('CONFIRMED' or 'REJECTED').
            notes: Optional operator remarks.

        Returns:
            True if row was updated.
        """
        valid_actions = {"CONFIRMED", "REJECTED", "PENDING"}
        normalized_action = action.upper()
        if normalized_action not in valid_actions:
            raise ValueError(f"Invalid review action: {action}. Must be one of {valid_actions}")

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with self._lock, self._conn:
            cursor = self._conn.execute(
                """
                UPDATE risk_events
                SET review_status = ?, operator_notes = ?, reviewed_at = ?
                WHERE event_id = ?;
                """,
                (normalized_action, notes, now_utc, event_id),
            )
            return cursor.rowcount > 0

    def query_recent_events(
        self, limit: int = 50, risk_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query recent logged risk events with optional risk state filtering.

        Args:
            limit: Maximum records to return.
            risk_filter: Optional filter on risk_state (e.g., 'HIGH_SEVERITY').

        Returns:
            List of dictionaries representing matched event rows.
        """
        with self._lock:
            cursor = self._conn.cursor()
            if risk_filter:
                cursor.execute(
                    """
                    SELECT * FROM risk_events
                    WHERE risk_state = ?
                    ORDER BY timestamp_utc DESC
                    LIMIT ?;
                    """,
                    (risk_filter, limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM risk_events
                    ORDER BY timestamp_utc DESC
                    LIMIT ?;
                    """,
                    (limit,),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def query_recent_telemetry(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Query recent high-frequency telemetry rows ordered by most recent first.

        Args:
            limit: Maximum records to return.

        Returns:
            List of dictionaries containing telemetry points.
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp_utc, vibration_rms, temperature_c, current_amps, vision_score, sensor_score, latency_ms
                FROM telemetry_stream
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp_utc": r[1],
                    "vibration_rms": r[2],
                    "temperature_c": r[3],
                    "current_amps": r[4],
                    "vision_score": r[5],
                    "sensor_score": r[6],
                    "latency_ms": r[7] if len(r) > 7 and r[7] is not None else 8.0,
                }
                for r in rows
            ]

    def query_recent_health(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Query recent component health logs.

        Args:
            limit: Maximum records to return.

        Returns:
            List of dictionaries containing component health logs.
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp_utc, component, status, details
                FROM system_health
                ORDER BY id DESC
                LIMIT ?;
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_actionable_event_by_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Fetch complete details of a specific risk event by event_id.

        Args:
            event_id: The unique decision/event ID.

        Returns:
            Dictionary of event attributes or None if not found.
        """
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT * FROM risk_events
                WHERE event_id = ?;
                """,
                (event_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_operator_metrics(self) -> Dict[str, Any]:
        """Compute operational triage metrics and operator confirmation rate."""
        with self._lock:
            cursor = self._conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_events,
                    SUM(CASE WHEN review_status = 'PENDING' THEN 1 ELSE 0 END) as pending_count,
                    SUM(CASE WHEN review_status = 'CONFIRMED' THEN 1 ELSE 0 END) as confirmed_count,
                    SUM(CASE WHEN review_status = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count
                FROM risk_events
                WHERE risk_state IN ('HIGH_SEVERITY', 'REVIEW_REQUIRED');
                """
            )
            row = cursor.fetchone()
            total = int(row["total_events"] or 0)
            pending = int(row["pending_count"] or 0)
            confirmed = int(row["confirmed_count"] or 0)
            rejected = int(row["rejected_count"] or 0)

            reviewed_total = confirmed + rejected
            confirm_rate = (confirmed / max(1, reviewed_total)) if reviewed_total > 0 else 0.0

            return {
                "total_actionable_events": total,
                "pending_reviews": pending,
                "confirmed_defects": confirmed,
                "rejected_false_positives": rejected,
                "confirmation_rate": confirm_rate,
                "reviewed_total": reviewed_total,
            }

    def query_events(self, limit: int = 50, risk_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Alias for query_recent_events."""
        return self.query_recent_events(limit=limit, risk_filter=risk_filter)

    def record_decision(self, decision: Union[PolicyDecision, Dict[str, Any]]) -> str:
        """Alias for insert_risk_event."""
        return self.insert_risk_event(decision)

    def close(self) -> None:
        """Close database connection."""
        with self._lock:
            self._conn.close()
            logger.info("AuditLogDB connection closed.")