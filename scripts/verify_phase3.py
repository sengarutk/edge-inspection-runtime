#!/usr/bin/env python3
"""Verification script for Phase 3: Resilient MQTT Messaging, Disk Spooling & Audit Subsystem.

Executes a 100-step continuous multi-modal simulation testing:
1. Online MQTT publishing and automatic subscriber audit logging (steps 0-30).
2. Simulated broker outage / network partition with local SQLite disk spooling (steps 31-60).
3. Broker reconnection and automatic background spool draining (steps 61-100).
4. Zero data loss verification in SQLite audit log database.
5. Simulated human operator triage and triage KPI metrics calculation.
"""

import json
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from loguru import logger

from src.audit_log import AuditLogDB
from src.config import load_mqtt_config, load_policy_config, load_sensor_config, load_system_config
from src.inference_service import InferenceEngine
from src.mqtt_publisher import ResilientMQTTPublisher
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine
from src.sensor_simulator import MachineState, SensorSimulator
from src.spooler import DiskSpooler


def create_synthetic_test_frame(step: int, state: MachineState) -> np.ndarray:
    """Generate synthetic test camera frame."""
    h, w = 224, 224
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (184, 184), (220, 220, 220), 2)
    cv2.circle(frame, (112, 112), 35, (180, 180, 180), 2)
    offset = int(5 * np.sin(step * 0.2))
    cv2.circle(frame, (112 + offset, 112), 8, (255, 255, 255), -1)
    return frame


def run_phase3_verification(num_steps: int = 100) -> bool:
    """Execute end-to-end continuous Phase 3 resilience verification.

    Args:
        num_steps: Total simulation steps (default 100).

    Returns:
        True if all resilience and zero-loss assertions pass.
    """
    logger.info("Initializing Phase 3 Resilient Messaging & Audit Verification...")

    # Load configs
    system_config = load_system_config()
    sensor_config = load_sensor_config()
    policy_config = load_policy_config()
    mqtt_config = load_mqtt_config()

    # Use dedicated test database paths
    spool_db_path = "data/verify_spooler.db"
    audit_db_path = "data/verify_audit.db"

    # Clean up previous test artifacts if present
    for p in (spool_db_path, audit_db_path):
        f = Path(p)
        if f.exists():
            f.unlink()

    spooler = DiskSpooler(db_path=spool_db_path)
    audit_db = AuditLogDB(db_path=audit_db_path)
    publisher = ResilientMQTTPublisher(config=mqtt_config, spooler=spooler)
    policy = TemporalPolicyEngine(config=policy_config, camera_id=system_config.camera_id, machine_id=system_config.machine_id)
    engine = InferenceEngine(config=system_config, seed=42)
    simulator = SensorSimulator(config=sensor_config, machine_id=system_config.machine_id, seed=42)

    simulator.reset()
    policy.reset()

    # Setup simulated in-process broker dispatch for zero-loss test
    def direct_broker_publish(topic: str, payload_str: str, qos: int = 1) -> MagicMock:
        # Ingest directly to audit DB simulating subscriber reception
        data = json.loads(payload_str)
        if topic == mqtt_config.topics.risk_events:
            audit_db.insert_risk_event(data)
        info = MagicMock()
        info.rc = 0
        return info

    publisher._client.publish = direct_broker_publish
    publisher.start()

    header = (
        f"{'Step':>4} | {'State':<11} | {'BrokerState':<11} | {'SpoolDepth':>10} | "
        f"{'AuditCount':>10} | {'RiskState':<15} | {'TriggerReason':<28}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" INDUSTRIAL EDGE INSPECTION RUNTIME - PHASE 3 RESILIENT MESSAGING VERIFICATION")
    print("=" * len(header))
    print(header)
    print(separator)

    for step in range(num_steps):
        # 1. Broker connectivity state schedule
        if step < 30:
            # Phase 3A: Broker ONLINE (steps 0-29)
            broker_online = True
            broker_state_str = "CONNECTED"
            state = MachineState.RUNNING
            inject_physical = False
            inject_vision = (step in (15, 16, 17, 18))  # Trigger an early defect
        elif step < 60:
            # Phase 3B: Network Partition / Outage (steps 30-59)
            broker_online = False
            broker_state_str = "DISCONNECTED"
            state = MachineState.FAULT if step >= 45 else MachineState.RUNNING
            inject_physical = (step >= 45)
            inject_vision = (step >= 45)
        else:
            # Phase 3C: Connection RESTORED (steps 60-99)
            broker_online = True
            broker_state_str = "RESTORED"
            state = MachineState.MAINTENANCE if step < 80 else MachineState.RUNNING
            inject_physical = False
            inject_vision = False

        # Set publisher connection flag simulating physical network link
        with publisher._state_lock:
            publisher._is_connected = broker_online

        # 2. Pipeline processing
        frame = create_synthetic_test_frame(step, state)
        inf_res = engine.run_inference(frame, inject_anomaly=inject_vision)
        sensor_read = simulator.step(machine_state=state, inject_fault=inject_physical)
        decision = policy.evaluate(inf_res, sensor_read)

        # 3. Publish to resilient MQTT publisher
        publisher.publish_event(
            topic=mqtt_config.topics.risk_events,
            payload=decision.to_mqtt_payload(),
            qos=mqtt_config.qos.risk_events,
        )

        # Record telemetry archival entry
        audit_db.insert_telemetry(sensor_read, inf_res)

        # 4. Telemetry printout
        spool_depth = spooler.get_queue_depth()
        recent_audit_count = len(audit_db.query_recent_events(limit=1000))

        is_event = (
            step in (0, 15, 18, 29, 30, 45, 59, 60, 61, 65, 75, 80, 99)
            or (step % 10 == 0)
        )
        if is_event:
            print(
                f"{step:>4} | {state.value:<11} | {broker_state_str:<11} | {spool_depth:>10} | "
                f"{recent_audit_count:>10} | {decision.risk_state.value:<15} | {decision.trigger_reason.value:<28}"
            )

    # Allow background drain worker time to flush remaining spooled records
    time.sleep(0.8)
    publisher.stop()

    final_spool_depth = spooler.get_queue_depth()
    all_audit_events = audit_db.query_recent_events(limit=200)

    print(separator)
    print("\n--- PHASE 3 ZERO-DATA-LOSS AUDIT VERIFICATION REPORT ---")
    print(f"Total Steps Executed:                  {num_steps}")
    print(f"Final Spool Queue Depth:               {final_spool_depth} (expected: 0)")
    print(f"Total Audit Events Persisted:          {len(all_audit_events)} (expected: {num_steps})")

    # Assert 0% data loss
    assert final_spool_depth == 0, f"Spooler queue must be completely drained, got {final_spool_depth} remaining."
    assert len(all_audit_events) == num_steps, f"Audit DB must contain all {num_steps} events, got {len(all_audit_events)}."

    # 5. Simulate human operator triage review
    high_events = audit_db.query_recent_events(limit=100, risk_filter="HIGH_SEVERITY")
    for i, ev in enumerate(high_events):
        action = "CONFIRMED" if i % 4 != 0 else "REJECTED"
        notes = "Verified critical failure" if action == "CONFIRMED" else "Operator verified transient test"
        audit_db.record_operator_review(ev["event_id"], action, notes)

    triage_metrics = audit_db.get_operator_metrics()
    print("\n--- OPERATOR TRIAGE METRICS ---")
    print(f"Total Actionable Events:               {triage_metrics['total_actionable_events']}")
    print(f"Confirmed Defects:                     {triage_metrics['confirmed_defects']}")
    print(f"Rejected False Positives:              {triage_metrics['rejected_false_positives']}")
    print(f"Operator Confirmation Rate:            {triage_metrics['confirmation_rate']:.1%}")

    assert triage_metrics["confirmed_defects"] > 0, "Must have confirmed defect events."

    # Cleanup test databases
    spooler.close()
    audit_db.close()
    for p in (spool_db_path, audit_db_path):
        f = Path(p)
        if f.exists():
            f.unlink()

    logger.info("Phase 3 Resilient Messaging & Audit Verification completed successfully.")
    return True


if __name__ == "__main__":
    success = run_phase3_verification()
    sys.exit(0 if success else 1)