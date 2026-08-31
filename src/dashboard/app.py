from __future__ import annotations

import sys
from pathlib import Path

# Guarantee project root is in sys.path when executed via `streamlit run`
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

"""Industrial Edge Inspection Runtime - Operator Reliability & Triage Console.

Production-grade Streamlit application providing real-time telemetry streaming,
human-in-the-loop triage queue, active chaos engineering, and dynamic FMEA diagnostics.
"""

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.audit_log import AuditLogDB
from src.config import (
    AuditConfig,
    MQTTConfig,
    PolicyConfig,
    SensorConfig,
    SpoolerConfig,
    SystemConfig,
    load_mqtt_config,
    load_policy_config,
    load_sensor_config,
    load_system_config,
)
from src.evidence_manager import EvidenceManager
from src.inference_service import InferenceEngine, InferenceResult, OpticalHealthStatus
from src.policy import PolicyDecision, RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorReading, SensorSimulator
from src.spooler import DiskSpooler


# =============================================================================
# Streamlit App Configuration & Styling
# =============================================================================
st.set_page_config(
    page_title="Edge Inspection Runtime | Operator Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .metric-card {
        background-color: #1e2130;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2e344e;
        margin-bottom: 10px;
    }
    .status-badge-healthy {
        background-color: #0e4429;
        color: #3fb950;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-warning {
        background-color: #4d2d00;
        color: #d29922;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
    .status-badge-critical {
        background-color: #4c1114;
        color: #f85149;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
        font-size: 0.85rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =============================================================================
# Cached Singletons & Database Access
# =============================================================================
@st.cache_resource
def get_database(db_path: Optional[str] = None) -> AuditLogDB:
    """Retrieve thread-safe AuditLogDB singleton connection."""
    if db_path:
        return AuditLogDB(db_path=db_path)
    return AuditLogDB()


@st.cache_resource
def get_evidence_mgr(storage_dir: Optional[str] = None) -> EvidenceManager:
    """Retrieve EvidenceManager singleton."""
    return EvidenceManager(storage_dir=storage_dir or "data/evidence")


def get_system_status(recent_events: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Derive global operational health status and styling badge."""
    if not recent_events:
        return "OPERATIONAL", "status-badge-healthy"

    latest = recent_events[0]
    if latest.get("risk_state") == "HIGH_SEVERITY":
        return "CRITICAL LOCKOUT", "status-badge-critical"
    if latest.get("risk_state") == "REVIEW_REQUIRED" or latest.get("is_degraded"):
        return "DEGRADED REVIEW", "status-badge-warning"

    return "OPERATIONAL", "status-badge-healthy"


# =============================================================================
# Interactive Demo Seeder & Chaos Simulation Helpers
# =============================================================================
def seed_demo_simulation(
    n_steps: int = 100,
    db: Optional[AuditLogDB] = None,
    evidence_mgr: Optional[EvidenceManager] = None,
) -> int:
    """Execute multi-step simulated edge inspection cycles to seed telemetry and risk events."""
    audit_db = db or get_database()
    ev_mgr = evidence_mgr or get_evidence_mgr()

    sensor_sim = SensorSimulator(seed=42)
    inf_engine = InferenceEngine(seed=42)
    policy_engine = TemporalPolicyEngine()

    rng = np.random.RandomState(42)
    base_frame = rng.randint(90, 160, (224, 224, 3), dtype=np.uint8)

    events_generated = 0
    for step in range(n_steps):
        # Determine step conditions
        is_defect = (25 <= step <= 40) or (70 <= step <= 80)
        is_sensor_fault = (50 <= step <= 65)
        is_optical_fault = (step == 88)

        # 1. Optical processing
        frame = base_frame.copy()
        if is_optical_fault:
            frame = cv2.GaussianBlur(frame, (31, 31), 0)

        inf_result = inf_engine.run_inference(frame, inject_anomaly=is_defect)

        # 2. Physical sensor simulation
        dropouts = ["current_amps"] if (15 <= step <= 18) else []
        sensor_reading = sensor_sim.step(
            machine_state=MachineState.RUNNING,
            inject_fault=is_sensor_fault,
            simulate_dropout=dropouts,
        )

        # 3. Evidence generation for significant anomalies
        ev_uri = None
        if is_defect or is_optical_fault or is_sensor_fault:
            heatmap = (
                inf_result.heatmap
                if inf_result.heatmap is not None
                else np.zeros((224, 224), dtype=np.float32)
            )
            ev_uri = ev_mgr.save_evidence(frame, heatmap, f"sim_frame_{step:04d}")

        # 4. Temporal Policy Evaluation
        decision = policy_engine.evaluate(inf_result, sensor_reading, evidence_uri=ev_uri)

        # 5. Database commits
        audit_db.insert_telemetry(sensor_reading, inf_result)
        audit_db.insert_risk_event(decision)
        events_generated += 1

    return events_generated


def inject_chaos_fault_event(
    fault_type: str,
    db: AuditLogDB,
    evidence_mgr: EvidenceManager,
) -> None:
    """Inject an immediate single-cycle anomalous condition and update audit records."""
    sensor_sim = SensorSimulator()
    inf_engine = InferenceEngine()
    policy_engine = TemporalPolicyEngine()

    rng = np.random.RandomState(int(time.time() * 1000) % 100000)
    frame = rng.randint(90, 160, (224, 224, 3), dtype=np.uint8)

    if fault_type == "OPTICAL_BLUR":
        frame = cv2.GaussianBlur(frame, (45, 45), 0)
        inf_result = inf_engine.run_inference(frame, inject_anomaly=False)
        sensor_reading = sensor_sim.step(machine_state=MachineState.RUNNING)
        ev_uri = evidence_mgr.save_evidence(
            frame, np.zeros((224, 224), dtype=np.float32), f"chaos_blur_{int(time.time())}"
        )
        decision = policy_engine.evaluate(inf_result, sensor_reading, evidence_uri=ev_uri)
        db.insert_telemetry(sensor_reading, inf_result)
        db.insert_risk_event(decision)
        db.insert_system_health("camera_optics", "DEGRADED", "OPTICAL_BLUR_DETECTED: Laplacian Var < 100.0")

    elif fault_type == "NETWORK_PARTITION":
        spooler = DiskSpooler()
        spooler.enqueue("inspection/line1/risk", json.dumps({"chaos_event": "NETWORK_PARTITION"}), qos=1)
        spooler.close()
        db.insert_system_health("mqtt_broker", "OFFLINE", "NETWORK_PARTITION: Broker Disconnected, Local Spool Active")

    elif fault_type == "SENSOR_DRIFT":
        inf_result = inf_engine.run_inference(frame, inject_anomaly=False)
        sensor_reading = sensor_sim.step(machine_state=MachineState.RUNNING, inject_fault=True)
        decision = policy_engine.evaluate(inf_result, sensor_reading)
        db.insert_telemetry(sensor_reading, inf_result)
        db.insert_risk_event(decision)
        db.insert_system_health("physical_sensors", "WARNING", "Thermal Drift Detected (+35C elevation)")

    elif fault_type == "SENSOR_DROPOUT":
        inf_result = inf_engine.run_inference(frame, inject_anomaly=False)
        sensor_reading = sensor_sim.step(
            machine_state=MachineState.RUNNING, simulate_dropout=["current_amps"]
        )
        decision = policy_engine.evaluate(inf_result, sensor_reading)
        db.insert_telemetry(sensor_reading, inf_result)
        db.insert_risk_event(decision)
        db.insert_system_health("current_sensor", "DEGRADED", "Channel Dropout: Imputed ZOH")


def compute_dynamic_fmea(
    audit_db: AuditLogDB,
    recent_events: List[Dict[str, Any]],
    recent_telemetry: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Derive real-time dynamic FMEA subsystem health rows based on recent telemetry and events."""
    try:
        recent_health = audit_db.query_recent_health(limit=20)
    except Exception:
        recent_health = []
    health_by_comp = {h["component"]: h for h in recent_health}

    # 1. Camera Optics
    has_optical_degradation = any(
        e.get("is_degraded") or e.get("trigger_reason") == "OPTICAL_DEGRADATION_FALLBACK"
        for e in recent_events[:10]
    ) or health_by_comp.get("camera_optics", {}).get("status") == "DEGRADED"

    camera_row = {
        "Subsystem": "Camera Optics (Line 1 Overhead)",
        "Health Status": "DEGRADED (Blur Detected)" if has_optical_degradation else "HEALTHY",
        "Degradation Mode": "Laplacian Var < 100.0" if has_optical_degradation else "Laplacian Var >= 100.0",
        "Action / Fallback": "Fallback to Sensor Telemetry" if has_optical_degradation else "Nominal",
    }

    # 2. PatchCore Model
    avg_latency = 8.0
    if recent_telemetry:
        latencies = [t.get("latency_ms", 8.0) for t in recent_telemetry[:20] if t.get("latency_ms")]
        if latencies:
            avg_latency = float(np.mean(latencies))

    model_row = {
        "Subsystem": "PatchCore Vision Inference Model",
        "Health Status": "WARNING (High Latency)" if avg_latency > 25.0 else "HEALTHY",
        "Degradation Mode": f"Mean Latency: {avg_latency:.1f}ms",
        "Action / Fallback": "Throttle Frame Rate" if avg_latency > 25.0 else "Nominal",
    }

    # 3. Physical Sensors
    has_dropout = any(
        t.get("vibration_rms", 0.0) == 0.0 or t.get("current_amps", 0.0) == 0.0
        for t in recent_telemetry[:10]
    ) or health_by_comp.get("current_sensor", {}).get("status") == "DEGRADED"

    has_sensor_spike = any(
        t.get("sensor_score", 0.0) > 0.65 or t.get("temperature_c", 0.0) > 85.0
        for t in recent_telemetry[:10]
    ) or health_by_comp.get("physical_sensors", {}).get("status") == "WARNING"

    if has_dropout:
        sensor_status = "DEGRADED (Imputed ZOH)"
        sensor_mode = "Channel Dropout / Signal Detached"
        sensor_action = "Zero-Order Hold (ZOH) Fallback"
    elif has_sensor_spike:
        sensor_status = "WARNING (Elevated Load/Heat)"
        sensor_mode = "Excess Z-Score > 3.0"
        sensor_action = "Cross-Modal Anomaly Verification"
    else:
        sensor_status = "HEALTHY"
        sensor_mode = "Continuous Sampling @ 30Hz"
        sensor_action = "Nominal"

    sensor_row = {
        "Subsystem": "3-Axis Vibration & Thermal Sensors",
        "Health Status": sensor_status,
        "Degradation Mode": sensor_mode,
        "Action / Fallback": sensor_action,
    }

    # 4. MQTT Broker
    is_broker_offline = health_by_comp.get("mqtt_broker", {}).get("status") == "OFFLINE"
    mqtt_row = {
        "Subsystem": "Mosquitto MQTT Broker (localhost:1883)",
        "Health Status": "DISCONNECTED (Offline)" if is_broker_offline else "CONNECTED",
        "Degradation Mode": "Broker Unreachable / Partitions" if is_broker_offline else "QoS 1 Ack Latency <= 2ms",
        "Action / Fallback": "Disk Spooler Active" if is_broker_offline else "Nominal",
    }

    # 5. Disk Spooler
    spooler = DiskSpooler()
    queue_depth = spooler.get_queue_depth()
    spooler.close()

    spooler_row = {
        "Subsystem": "SQLite Local Disk Spooler",
        "Health Status": f"BUFFERING (Active: {queue_depth} spooled)" if queue_depth > 0 else "HEALTHY (Idle)",
        "Degradation Mode": f"Capacity: {queue_depth}/50000 records",
        "Action / Fallback": "Zero Data Loss Local Buffer",
    }

    return [camera_row, model_row, sensor_row, mqtt_row, spooler_row]


# =============================================================================
# Main Application Flow
# =============================================================================
def main() -> None:
    """Render operator reliability dashboard UI components."""
    audit_db = get_database()
    evidence_mgr = get_evidence_mgr()

    # Query operational data
    recent_events = audit_db.query_recent_events(limit=50)
    recent_telemetry = audit_db.query_recent_telemetry(limit=60)
    operator_metrics = audit_db.get_operator_metrics()

    sys_status, status_class = get_system_status(recent_events)

    # -------------------------------------------------------------------------
    # Header Section
    # -------------------------------------------------------------------------
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.title("Industrial Edge Inspection Runtime")
        st.caption("Operator Reliability Console, Temporal Anomaly Fusion & Human-in-the-Loop Triage")
    with h_col2:
        st.markdown(
            f"<div style='text-align: right; padding-top: 15px;'>"
            f"System State: <span class='{status_class}'>{sys_status}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # Sidebar Controls & Seed Engine
    # -------------------------------------------------------------------------
    st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.sidebar.title("Runtime Controls")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Live Simulation Engine")
    if st.sidebar.button("⚡ Run 100-Cycle Live Simulation", use_container_width=True):
        with st.spinner("Executing 100-cycle edge inspection simulation..."):
            count = seed_demo_simulation(100, audit_db, evidence_mgr)
        st.sidebar.success(f"Generated {count} telemetry cycles & events!")
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Metadata")
    st.sidebar.info(
        "**Line:** Automotive Stamping #1\n\n"
        "**Camera:** Line1 Overhead 4K\n\n"
        "**Sampling:** 30 FPS / 30 Hz\n\n"
        "**Storage:** SQLite WAL & Local Spooler"
    )

    # -------------------------------------------------------------------------
    # Main Tabs
    # -------------------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📊 Live Telemetry & Risk Stream",
        "🔍 Operator Triage Queue",
        "💥 Chaos Engineering & FMEA Diagnostics",
    ])

    # =========================================================================
    # TAB 1: Live Telemetry & Risk Stream
    # =========================================================================
    with tab1:
        col1, col2, col3, col4 = st.columns(4)

        latest_risk = recent_events[0]["risk_state"] if recent_events else "NOMINAL"
        latest_state = recent_events[0].get("machine_state", "RUNNING") if recent_events else "RUNNING"
        confirm_rate = operator_metrics.get("confirmation_rate", 0.0)
        confirm_rate_pct = f"{confirm_rate * 100:.1f}%" if confirm_rate is not None else "N/A"

        with col1:
            st.metric("Operational Risk", latest_risk)
        with col2:
            st.metric("Machine State", latest_state)
        with col3:
            st.metric("Pending Triage Queue", str(operator_metrics["pending_reviews"]))
        with col4:
            st.metric("Operator Confirmation Rate", confirm_rate_pct)

        st.markdown("---")

        # Time Series Charts & Empty State Handling
        if recent_telemetry:
            df_telem = pd.DataFrame(recent_telemetry).iloc[::-1]
            df_telem["timestamp"] = pd.to_datetime(df_telem["timestamp_utc"])

            st.subheader("Inspection Risk Trajectory & Anomaly Scores")
            chart_data = df_telem.set_index("timestamp")[["vision_score", "sensor_score"]]
            st.line_chart(chart_data)

            st.subheader("Physical Telemetry Multi-Sensor Strip")
            sensor_data = df_telem.set_index("timestamp")[
                ["vibration_rms", "temperature_c", "current_amps"]
            ]
            st.line_chart(sensor_data)

            st.subheader("Recent Inspection Telemetry (Last 15 Cycles)")
            st.dataframe(
                df_telem.tail(15)[
                    [
                        "timestamp_utc",
                        "vision_score",
                        "sensor_score",
                        "vibration_rms",
                        "temperature_c",
                        "current_amps",
                    ]
                ],
                use_container_width=True,
            )
        else:
            st.info("No continuous telemetry records ingested yet.")
            st.markdown(
                "Click below to generate realistic cyber-physical telemetry, thermal dynamics, "
                "and optical defect heatmaps:"
            )
            if st.button("🚀 Seed Demo Telemetry Data", use_container_width=True):
                with st.spinner("Seeding initial live telemetry stream..."):
                    seed_demo_simulation(100, audit_db, evidence_mgr)
                st.rerun()

    # =========================================================================
    # TAB 2: Human-in-the-Loop Triage Review Queue
    # =========================================================================
    with tab2:
        st.subheader("Actionable Incident Triage Queue")

        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            status_filter = st.selectbox(
                "Review Status Filter", ["ALL", "PENDING", "CONFIRMED", "REJECTED"], index=1
            )
        with f_col2:
            risk_filter = st.selectbox(
                "Risk Level Filter", ["ALL", "HIGH_SEVERITY", "REVIEW_REQUIRED"], index=0
            )

        filtered_events = recent_events
        if status_filter != "ALL":
            filtered_events = [e for e in filtered_events if e.get("review_status") == status_filter]
        if risk_filter != "ALL":
            filtered_events = [e for e in filtered_events if e.get("risk_state") == risk_filter]

        if filtered_events:
            event_options = [
                f"{e['event_id'][:8]}... | {e['timestamp_utc'][11:19]} | {e['risk_state']} | {e['trigger_reason']} | Status: {e['review_status']}"
                for e in filtered_events
            ]
            selected_idx = st.selectbox(
                "Select Incident for Review",
                range(len(event_options)),
                format_func=lambda i: event_options[i],
            )
            selected_event = filtered_events[selected_idx]

            st.markdown("### Incident Investigation & Optical Evidence")
            col_img, col_meta, col_action = st.columns([2, 2, 2])

            with col_img:
                st.markdown("**Optical Defect Heatmap Overlay**")
                evidence_uri = selected_event.get("evidence_uri")
                if evidence_uri:
                    img = evidence_mgr.load_evidence(evidence_uri)
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(
                            img_rgb,
                            caption=f"Evidence: {Path(evidence_uri).name}",
                            use_container_width=True,
                        )
                    else:
                        st.warning(f"Evidence file not found on disk at: {evidence_uri}")
                else:
                    st.info("No optical evidence image attached to this nominal event.")

            with col_meta:
                st.markdown("**Telemetry & Correlation Attributes**")
                meta_table = {
                    "Event ID": selected_event["event_id"],
                    "Timestamp (UTC)": selected_event["timestamp_utc"],
                    "Risk Classification": selected_event["risk_state"],
                    "Trigger Reason": selected_event["trigger_reason"],
                    "Vision Score (Raw / EMA)": f"{selected_event['vision_raw']:.3f} / {selected_event['vision_ema']:.3f}",
                    "Sensor Score (Raw / EMA)": f"{selected_event['sensor_raw']:.3f} / {selected_event['sensor_ema']:.3f}",
                    "Cooldown Remaining": str(selected_event["cooldown_remaining"]),
                    "Hardware Degraded": "YES" if selected_event["is_degraded"] else "NO",
                    "Correlated Frame ID": str(selected_event["frame_id"]),
                    "Correlated Sensor ID": str(selected_event["reading_id"]),
                }
                st.table(pd.DataFrame(list(meta_table.items()), columns=["Field", "Value"]))

            with col_action:
                st.markdown("**Operator Triage & Action Panel**")
                current_status = selected_event.get("review_status", "PENDING")
                st.info(f"Current Review Status: **{current_status}**")

                notes = st.text_area(
                    "Operator Remarks / Root Cause Notes",
                    value=selected_event.get("operator_notes") or "",
                    placeholder="e.g. Confirmed surface fracture on weld seam...",
                )

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Confirm Defect", use_container_width=True):
                        audit_db.record_operator_review(
                            selected_event["event_id"], action="CONFIRMED", notes=notes
                        )
                        st.toast("Defect CONFIRMED and escalated to maintenance!", icon="✅")
                        st.rerun()

                with btn_col2:
                    if st.button("❌ Reject Alarm", use_container_width=True):
                        audit_db.record_operator_review(
                            selected_event["event_id"], action="REJECTED", notes=notes
                        )
                        st.toast("Alert flagged as FALSE POSITIVE.", icon="❌")
                        st.rerun()
        else:
            st.info("No incidents found matching the selected filters.")

    # =========================================================================
    # TAB 3: Chaos Engineering & Dynamic FMEA Diagnostics
    # =========================================================================
    with tab3:
        st.subheader("Interactive Chaos Engineering & Fault Simulation")

        st.markdown("Inject programmatic hardware and network faults into the live edge runtime loop:")
        c_col1, c_col2, c_col3, c_col4 = st.columns(4)

        with c_col1:
            if st.button("📷 Inject Camera Blur", use_container_width=True):
                inject_chaos_fault_event("OPTICAL_BLUR", audit_db, evidence_mgr)
                st.toast("Injected OPTICAL_BLUR chaos fault!", icon="📷")
                st.rerun()
        with c_col2:
            if st.button("🔌 Disconnect MQTT Broker", use_container_width=True):
                inject_chaos_fault_event("NETWORK_PARTITION", audit_db, evidence_mgr)
                st.toast("Injected NETWORK_PARTITION chaos fault! Disk spool active.", icon="🔌")
                st.rerun()
        with c_col3:
            if st.button("🌡️ Trigger Thermal Drift", use_container_width=True):
                inject_chaos_fault_event("SENSOR_DRIFT", audit_db, evidence_mgr)
                st.toast("Injected SENSOR_DRIFT chaos fault!", icon="🌡️")
                st.rerun()
        with c_col4:
            if st.button("⚡ Current Sensor Dropout", use_container_width=True):
                inject_chaos_fault_event("SENSOR_DROPOUT", audit_db, evidence_mgr)
                st.toast("Injected SENSOR_DROPOUT on current channel!", icon="⚡")
                st.rerun()

        st.markdown("---")
        st.subheader("Dynamic Subsystem FMEA Diagnostic Health Matrix")

        fmea_rows = compute_dynamic_fmea(audit_db, recent_events, recent_telemetry)
        st.dataframe(pd.DataFrame(fmea_rows), use_container_width=True)


if __name__ == "__main__":
    main()
