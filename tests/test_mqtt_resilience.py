"""Unit tests for resilient MQTT publisher, subscriber, and offline spooling."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.audit_log import AuditLogDB
from src.config import AuditConfig, MQTTConfig, SpoolerConfig
from src.mqtt_publisher import ResilientMQTTPublisher
from src.mqtt_subscriber import MQTTEventSubscriber
from src.spooler import DiskSpooler


@pytest.fixture
def temp_spooler(tmp_path: Path) -> DiskSpooler:
    """Fixture providing an isolated DiskSpooler."""
    db_file = tmp_path / "mqtt_spool.db"
    sp = DiskSpooler(config=SpoolerConfig(db_path=str(db_file), max_spool_records=100))
    yield sp
    sp.close()


@pytest.fixture
def temp_audit_db(tmp_path: Path) -> AuditLogDB:
    """Fixture providing an isolated AuditLogDB."""
    db_file = tmp_path / "audit_sub.db"
    db = AuditLogDB(config=AuditConfig(db_path=str(db_file)))
    yield db
    db.close()


def test_publisher_disconnected_fallback_to_spooler(temp_spooler: DiskSpooler) -> None:
    """Test that publisher safely commits to DiskSpooler when offline without losing events."""
    pub = ResilientMQTTPublisher(spooler=temp_spooler)
    assert pub.is_connected is False

    pub.publish_event("inspection/line1/risk", {"event_id": "e1", "risk": "HIGH_SEVERITY"})
    pub.publish_event("inspection/line1/risk", {"event_id": "e2", "risk": "NORMAL"})
    pub.publish_heartbeat({"status": "OFFLINE_TEST"})

    assert temp_spooler.get_queue_depth() == 3
    batch = temp_spooler.peek_batch(limit=10)
    assert len(batch) == 3
    assert "e1" in batch[0][2]
    assert "e2" in batch[1][2]


def test_publisher_connected_direct_publish(temp_spooler: DiskSpooler) -> None:
    """Test that publisher directly sends message to broker when connected."""
    pub = ResilientMQTTPublisher(spooler=temp_spooler)

    mock_publish_info = MagicMock()
    mock_publish_info.rc = 0
    pub._client.publish = MagicMock(return_value=mock_publish_info)
    pub._is_connected = True

    success = pub.publish_event("inspection/line1/risk", {"event_id": "e1", "risk": "NORMAL"})
    assert success is True
    assert pub._client.publish.called
    assert temp_spooler.get_queue_depth() == 0


def test_publisher_publish_failure_routes_to_spooler(temp_spooler: DiskSpooler) -> None:
    """Test that if publish throws exception or returns error code, it spools to disk."""
    pub = ResilientMQTTPublisher(spooler=temp_spooler)
    pub._is_connected = True

    # Case 1: Error code returned
    mock_err_info = MagicMock()
    mock_err_info.rc = 1
    pub._client.publish = MagicMock(return_value=mock_err_info)
    pub.publish_event("inspection/line1/risk", {"event_id": "err1"})
    assert temp_spooler.get_queue_depth() == 1

    # Case 2: Exception raised
    pub._client.publish = MagicMock(side_effect=RuntimeError("Socket error"))
    pub.publish_event("inspection/line1/risk", {"event_id": "err2"})
    assert temp_spooler.get_queue_depth() == 2


def test_publisher_callbacks_and_lifecycle(temp_spooler: DiskSpooler) -> None:
    """Test publisher connect/disconnect callback triggers and start/stop lifecycle."""
    pub = ResilientMQTTPublisher(spooler=temp_spooler)

    # Connect callback success
    pub._client.on_connect(pub._client, None, None, 0)
    assert pub.is_connected is True

    # Connect callback failure
    pub._client.on_connect(pub._client, None, None, 5)
    assert pub.is_connected is False

    # Disconnect callback
    pub._client.on_disconnect(pub._client, None, None, 0)
    assert pub.is_connected is False

    # Lifecycle start exception handling
    with patch.object(pub._client, "connect_async", side_effect=Exception("Connection refused")):
        pub.start()
        assert pub._running is True
        pub.start()  # Idempotent

    # Stop exception handling
    with patch.object(pub._client, "disconnect", side_effect=Exception("Disconnect error")):
        pub.stop()
        assert pub._running is False
        pub.stop()  # Idempotent


def test_publisher_background_drain_flushes_spooler(temp_spooler: DiskSpooler) -> None:
    """Test that drain worker automatically flushes all queued records upon broker reconnection."""
    for i in range(5):
        temp_spooler.enqueue("inspection/line1/risk", json.dumps({"event_id": f"spool_{i}"}), qos=1)

    assert temp_spooler.get_queue_depth() == 5

    pub = ResilientMQTTPublisher(spooler=temp_spooler)

    mock_info = MagicMock()
    mock_info.rc = 0
    pub._client.publish = MagicMock(return_value=mock_info)

    pub.start()
    with pub._state_lock:
        pub._is_connected = True

    import time
    time.sleep(0.8)

    assert temp_spooler.get_queue_depth() == 0
    assert pub._client.publish.call_count >= 5
    pub.stop()


def test_publisher_drain_worker_publish_error_handling(temp_spooler: DiskSpooler) -> None:
    """Test that drain worker safely handles exceptions when publishing a spooled record."""
    temp_spooler.enqueue("inspection/line1/risk", json.dumps({"event_id": "fail_record"}), qos=1)

    pub = ResilientMQTTPublisher(spooler=temp_spooler)
    pub._client.publish = MagicMock(side_effect=Exception("Broker pipe broken"))

    pub.start()
    with pub._state_lock:
        pub._is_connected = True

    import time
    time.sleep(0.6)

    # Record should remain in spooler since publish failed
    assert temp_spooler.get_queue_depth() == 1
    pub.stop()


def test_subscriber_message_handling_and_audit_ingestion(temp_audit_db: AuditLogDB) -> None:
    """Test subscriber dispatching incoming MQTT messages to AuditLogDB and callback."""
    received_events = []

    def custom_callback(topic: str, data: dict) -> None:
        received_events.append((topic, data))

    sub = MQTTEventSubscriber(
        audit_db=temp_audit_db,
        on_event_callback=custom_callback,
    )

    # 1. Risk event message
    mock_risk_msg = MagicMock()
    mock_risk_msg.topic = "inspection/line1/risk"
    mock_risk_msg.payload = json.dumps({
        "event_id": "sub_event_01",
        "timestamp_utc": "2026-08-27T15:00:00.000Z",
        "camera_id": "line1_overhead_cam01",
        "machine_id": "press_unit_04",
        "machine_state": "RUNNING",
        "risk_state": "HIGH_SEVERITY",
        "trigger_reason": "SUSTAINED_VISION_ANOMALY",
        "raw_scores": {"vision_raw": 0.92, "sensor_raw": 0.05},
        "smoothed_scores": {"vision_ema": 0.88, "sensor_ema": 0.05},
        "cooldown_remaining": 15,
        "is_degraded": False,
    }).encode("utf-8")

    sub._client.on_message(sub._client, None, mock_risk_msg)

    # 2. Health event message
    mock_health_msg = MagicMock()
    mock_health_msg.topic = "inspection/line1/health"
    mock_health_msg.payload = json.dumps({
        "component": "camera_01",
        "status": "HEALTHY",
    }).encode("utf-8")

    sub._client.on_message(sub._client, None, mock_health_msg)

    # 3. Telemetry event message
    mock_telem_msg = MagicMock()
    mock_telem_msg.topic = "inspection/line1/telemetry"
    mock_telem_msg.payload = json.dumps({"vib": 0.45}).encode("utf-8")

    sub._client.on_message(sub._client, None, mock_telem_msg)

    # 4. Invalid JSON payload handling
    bad_msg = MagicMock()
    bad_msg.topic = "inspection/line1/risk"
    bad_msg.payload = b"not-a-json"
    sub._client.on_message(sub._client, None, bad_msg)

    assert len(received_events) == 3

    events = temp_audit_db.query_recent_events(limit=10)
    assert len(events) == 1
    assert events[0]["event_id"] == "sub_event_01"


def test_subscriber_error_handling_in_callbacks(temp_audit_db: AuditLogDB) -> None:
    """Test subscriber error resilience when audit DB or on_event callback raises."""
    mock_audit = MagicMock()
    mock_audit.insert_risk_event.side_effect = Exception("DB Disk Full")

    def failing_callback(topic: str, data: dict) -> None:
        raise RuntimeError("Callback crashed")

    sub = MQTTEventSubscriber(
        audit_db=mock_audit,
        on_event_callback=failing_callback,
    )

    mock_msg = MagicMock()
    mock_msg.topic = "inspection/line1/risk"
    mock_msg.payload = json.dumps({"event_id": "test_err"}).encode("utf-8")

    # Should log errors without crashing
    sub._client.on_message(sub._client, None, mock_msg)


def test_subscriber_callbacks_and_lifecycle(temp_audit_db: AuditLogDB) -> None:
    """Test subscriber connect/disconnect callbacks and start/stop methods with error handling."""
    sub = MQTTEventSubscriber(audit_db=temp_audit_db)

    # Connect success
    sub._client.on_connect(sub._client, None, None, 0)
    assert sub.is_connected is True

    # Connect failure
    sub._client.on_connect(sub._client, None, None, 5)
    assert sub.is_connected is False

    # Disconnect
    sub._client.on_disconnect(sub._client, None, None, 0)
    assert sub.is_connected is False

    # Start exception handling
    with patch.object(sub._client, "connect_async", side_effect=Exception("Broker unreachable")):
        sub.start()
        assert sub._running is True
        sub.start()  # Idempotent

    # Stop exception handling
    with patch.object(sub._client, "disconnect", side_effect=Exception("Disconnect socket error")):
        sub.stop()
        assert sub._running is False
        sub.stop()  # Idempotent