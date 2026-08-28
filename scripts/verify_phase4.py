#!/usr/bin/env python3
"""End-to-End Verification Script for Phase 4: Chaos Engineering, Triage & Evidence Subsystem.

Executes a 120-step continuous industrial simulation testing:
1. Nominal baseline (steps 0-19)
2. Optical blur injection & degraded fallback (steps 20-29)
3. Network partition & offline disk spooling during defect (steps 30-44)
4. Network recovery & sensor dropout handling (steps 45-59)
5. Sustained multi-modal machine FAULT with evidence generation (steps 60-84)
6. Programmatic human-in-the-loop operator review triage (steps 85-104)
7. Return to nominal steady-state production (steps 105-119)
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
from src.evidence_manager import EvidenceManager
from src.fault_injector import ChaosFaultConfig, FaultInjector, FaultType
from src.inference_service import InferenceEngine
from src.mqtt_publisher import ResilientMQTTPublisher
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorSimulator
from src.spooler import DiskSpooler


def create_synthetic_test_frame(step: int) -> np.ndarray:
    """Generate high-contrast synthetic camera frame."""
    h, w = 224, 224
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (184, 184), (220, 220, 220), 2)
    cv2.circle(frame, (112, 112), 35, (180, 180, 180), 2)
    offset = int(5 * np.sin(step * 0.2))
    cv2.circle(frame, (112 + offset, 112), 8, (255, 255, 255), -1)
    return frame


def run_phase4_verification(num_steps: int = 120) -> bool:
    """Execute end-to-end continuous Phase 4 chaos & triage verification.

    Args:
        num_steps: Total simulation steps (default 120).

    Returns:
        True if all verification assertions pass.
    """
    logger.info("Initializing Phase 4 Chaos Engineering & Human-in-the-Loop Triage Verification...")

    # Load configs
    system_config = load_system_config()
    sensor_config = load_sensor_config()
    policy_config = load_policy_config()
    mqtt_config = load_mqtt_config()

    spool_db_path = "data/verify_p4_spooler.db"
    audit_db_path = "data/verify_p4_audit.db"
    evidence_dir = "data/verify_p4_evidence"

    # Clean up test artifacts
    for p in (spool_db_path, audit_db_path):
        f = Path(p)
        if f.exists():
            f.unlink()

    evidence_mgr = EvidenceManager(storage_dir=evidence_dir)
    spooler = DiskSpooler(db_path=spool_db_path)
    audit_db = AuditLogDB(db_path=audit_db_path)
    publisher = ResilientMQTTPublisher(config=mqtt_config, spooler=spooler)
    policy = TemporalPolicyEngine(config=policy_config, camera_id=system_config.camera_id, machine_id=system_config.machine_id)
    engine = InferenceEngine(config=system_config, seed=42)
    simulator = SensorSimulator(config=sensor_config, machine_id=system_config.machine_id, seed=42)
    fault_injector = FaultInjector()

    simulator.reset()
    policy.reset()

    # Schedule chaos faults
    fault_injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.OPTICAL_BLUR, start_step=20, duration_steps=10, intensity=1.2)
    )
    fault_injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.NETWORK_PARTITION, start_step=30, duration_steps=15)
    )
    fault_injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.MODEL_DISTRIBUTION_SHIFT, start_step=32, duration_steps=10)
    )
    fault_injector.add_fault_schedule(
        ChaosFaultConfig(fault_type=FaultType.SENSOR_DROPOUT, start_step=45, duration_steps=15, target_channels=["current"])
    )

    # Setup direct broker simulation dispatch
    def direct_broker_publish(topic: str, payload_str: str, qos: int = 1) -> MagicMock:
        data = json.loads(payload_str)
        if topic == mqtt_config.topics.risk_events:
            audit_db.insert_risk_event(data)
        info = MagicMock()
        info.rc = 0
        return info

    publisher._client.publish = direct_broker_publish
    publisher.start()

    header = (
        f"{'Step':>4} | {'State':<11} | {'ActiveFaults':<22} | {'VisRaw':>6} | "
        f"{'SensEMA':>7} | {'RiskState':<15} | {'TriggerReason':<28} | {'Spool':>5}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" INDUSTRIAL EDGE INSPECTION RUNTIME - PHASE 4 CHAOS & TRIAGE VERIFICATION")
    print("=" * len(header))
    print(header)
    print(separator)

    evidence_uris_generated: list[str] = []
    optical_fallback_count = 0
    spooled_record_count = 0

    for step in range(num_steps):
        # 1. Determine machine state
        if step < 60:
            machine_state = MachineState.RUNNING
        elif step < 85:
            machine_state = MachineState.FAULT
        elif step < 105:
            machine_state = MachineState.MAINTENANCE
        else:
            machine_state = MachineState.RUNNING

        # 2. Chaos transformations
        raw_frame = create_synthetic_test_frame(step)
        chaos_frame = fault_injector.apply_optical_fault(raw_frame, step)
        inject_physical_fault, dropouts = fault_injector.apply_sensor_fault(simulator, step)
        is_vision_shifted = fault_injector.apply_vision_shift(False, step)
        broker_online = fault_injector.apply_network_fault(publisher, step)

        if machine_state == MachineState.FAULT:
            inject_physical_fault = True
            is_vision_shifted = True

        # 3. Vision & Sensor execution
        inf_res = engine.run_inference(chaos_frame, inject_anomaly=is_vision_shifted)
        sensor_read = simulator.step(
            machine_state=machine_state,
            inject_fault=inject_physical_fault,
            simulate_dropout=dropouts,
        )

        # 4. Save evidence for actionable alerts (HIGH_SEVERITY or REVIEW_REQUIRED with anomaly)
        evidence_uri = None
        if inf_res.vision_score >= 0.50 or machine_state == MachineState.FAULT or not inf_res.optical_health.is_valid:
            evidence_uri = evidence_mgr.save_evidence(
                frame=chaos_frame,
                heatmap=inf_res.heatmap,
                frame_id=inf_res.frame_id,
            )
            evidence_uris_generated.append(evidence_uri)

        # 5. Evaluate policy decision
        decision = policy.evaluate(inf_res, sensor_read, evidence_uri=evidence_uri)

        if decision.trigger_reason == TriggerReason.OPTICAL_DEGRADATION_FALLBACK:
            optical_fallback_count += 1

        # 6. Resilient publishing
        publisher.publish_event(
            topic=mqtt_config.topics.risk_events,
            payload=decision.to_mqtt_payload(),
            qos=mqtt_config.qos.risk_events,
        )
        audit_db.insert_telemetry(sensor_read, inf_res)

        active_faults = fault_injector.get_active_faults(step)
        faults_str = ",".join(f.fault_type.value for f in active_faults) or "NONE"
        spool_depth = spooler.get_queue_depth()
        if spool_depth > 0:
            spooled_record_count += 1

        is_event = (
            step in (0, 19, 20, 25, 29, 30, 35, 44, 45, 50, 59, 60, 65, 84, 85, 104, 105, 119)
            or (decision.risk_state == RiskState.HIGH_SEVERITY)
            or (step % 10 == 0)
        )

        if is_event:
            sens_ema_str = f"{decision.smoothed_scores['sensor_ema']:.3f}"
            print(
                f"{step:>4} | {machine_state.value:<11} | {faults_str:<22} | {inf_res.vision_score:>6.3f} | "
                f"{sens_ema_str:>7} | {decision.risk_state.value:<15} | {decision.trigger_reason.value:<28} | "
                f"{spool_depth:>5}"
            )

    # Allow background drain worker to flush remaining spooled records
    time.sleep(1.0)
    publisher.stop()

    print(separator)

    # 7. Perform Programmatic Operator Triage Reviews
    actionable_events = audit_db.query_recent_events(limit=200, risk_filter="HIGH_SEVERITY")
    for i, ev in enumerate(actionable_events):
        # 80% confirmed, 20% rejected
        action = "CONFIRMED" if i % 5 != 0 else "REJECTED"
        notes = "Verified critical machine fault in press line" if action == "CONFIRMED" else "Operator verified test alert"
        audit_db.record_operator_review(ev["event_id"], action, notes)

    triage_metrics = audit_db.get_operator_metrics()
    final_spool_depth = spooler.get_queue_depth()
    all_audit_events = audit_db.query_recent_events(limit=300)

    print("\n--- PHASE 4 CHAOS & TRIAGE KPI SUMMARY REPORT ---")
    print(f"Total Steps Simulated:                 {num_steps}")
    print(f"Optical Fallbacks Triggered:           {optical_fallback_count}")
    print(f"Evidence Artifacts Generated:          {len(evidence_uris_generated)}")
    print(f"Final Spool Queue Depth:               {final_spool_depth} (expected: 0)")
    print(f"Total Audit Events Ingested:           {len(all_audit_events)} (expected: {num_steps})")
    print(f"Total Actionable Events Triaged:       {triage_metrics['total_actionable_events']}")
    print(f"Confirmed Defects:                     {triage_metrics['confirmed_defects']}")
    print(f"Rejected False Positives:              {triage_metrics['rejected_false_positives']}")
    print(f"Operator Confirmation Rate:            {triage_metrics['confirmation_rate']:.1%}")

    # Assertions for Phase 4 Acceptance
    assert optical_fallback_count >= 5, "Optical blur must trigger optical degradation fallbacks."
    assert len(evidence_uris_generated) > 0, "Must have generated optical evidence artifacts."
    for uri in evidence_uris_generated[:5]:
        assert Path(uri).is_file(), f"Evidence image {uri} must exist on disk."
    assert final_spool_depth == 0, "Disk spooler must be completely drained after network restoration."
    assert len(all_audit_events) == num_steps, f"Zero data loss expected. Found {len(all_audit_events)} of {num_steps}."
    assert triage_metrics["confirmed_defects"] > 0, "Operator triage must confirm defect incidents."

    # Clean up test artifacts
    spooler.close()
    audit_db.close()
    for p in (spool_db_path, audit_db_path):
        f = Path(p)
        if f.exists():
            f.unlink()

    import shutil
    if Path(evidence_dir).exists():
        shutil.rmtree(evidence_dir)

    logger.info("Phase 4 Chaos Engineering & Human-in-the-Loop Triage Verification PASSED 100%.")
    return True


if __name__ == "__main__":
    success = run_phase4_verification()
    sys.exit(0 if success else 1)