#!/usr/bin/env python3
"""End-to-End Verification Script for Phase 2: Temporal Decision Policies & Cross-Modal Risk Engine.

Runs a continuous 150-step multi-modal industrial simulation across:
1. IDLE baseline (steps 0-24)
2. Normal operational RUNNING (steps 25-49)
3. Transient 2-frame optical glitch (steps 50-52)
4. Sustained surface visual defect with anti-fatigue cooldown (steps 53-74)
5. Optical blur degradation fallback (steps 75-84)
6. Critical physical machine FAULT (steps 85-109)
7. MAINTENANCE state with testing & lockout gating (steps 110-129)
8. Return to steady-state nominal production (steps 130-149)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from loguru import logger

from src.config import load_policy_config, load_sensor_config, load_system_config
from src.inference_service import InferenceEngine
from src.policy import RiskState, TemporalPolicyEngine, TriggerReason
from src.sensor_simulator import MachineState, SensorSimulator


def create_synthetic_test_frame(
    step: int,
    state: MachineState,
    is_blurred: bool = False,
    is_occluded: bool = False,
) -> np.ndarray:
    """Generate synthetic camera frame simulating pristine, blurred, or occluded conditions."""
    h, w = 224, 224

    if is_occluded:
        return np.zeros((h, w, 3), dtype=np.uint8)

    if is_blurred:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.circle(img, (w // 2, h // 2), 60, (220, 220, 220), -1)
        return cv2.GaussianBlur(img, (35, 35), sigmaX=20.0)

    # Sharp industrial part frame with high contrast edges
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (184, 184), (220, 220, 220), 2)
    cv2.circle(frame, (112, 112), 35, (180, 180, 180), 2)
    cv2.line(frame, (50, 112), (174, 112), (150, 150, 150), 1)

    offset = int(5 * np.sin(step * 0.2))
    cv2.circle(frame, (112 + offset, 112), 8, (255, 255, 255), -1)
    return frame


def run_phase2_verification(num_steps: int = 150) -> bool:
    """Execute end-to-end continuous Phase 2 verification loop.

    Args:
        num_steps: Total simulation steps (default 150).

    Returns:
        True if all verification assertions pass.
    """
    logger.info("Initializing Phase 2 Temporal Decision Policy Verification...")

    system_config = load_system_config()
    sensor_config = load_sensor_config()
    policy_config = load_policy_config()

    engine = InferenceEngine(config=system_config, seed=42)
    simulator = SensorSimulator(config=sensor_config, machine_id=system_config.machine_id, seed=42)
    policy = TemporalPolicyEngine(config=policy_config, camera_id=system_config.camera_id, machine_id=system_config.machine_id)

    simulator.reset()
    policy.reset()

    header = (
        f"{'Step':>4} | {'State':<11} | {'VisRaw':>6} | {'VisEMA':>6} | "
        f"{'SensEMA':>7} | {'RiskState':<15} | {'TriggerReason':<28} | "
        f"{'Cool':>4} | {'AlertRed%':>9}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" INDUSTRIAL EDGE INSPECTION RUNTIME - PHASE 2 TEMPORAL POLICY SIMULATION")
    print("=" * len(header))
    print(header)
    print(separator)

    raw_threshold_crossings = 0
    policy_escalations = 0
    transient_high_severity_count = 0
    sustained_defect_onset_step: int | None = None
    first_escalation_step: int | None = None

    for step in range(num_steps):
        # 1. Operational phase scheduling
        if step < 25:
            state = MachineState.IDLE
            inject_physical_fault = False
            inject_vision_anomaly = False
            is_blurred = False
            is_occluded = False
        elif step < 50:
            state = MachineState.RUNNING
            inject_physical_fault = False
            inject_vision_anomaly = False
            is_blurred = False
            is_occluded = False
        elif step < 53:  # Transient 2-frame glitch (steps 50, 51)
            state = MachineState.RUNNING
            inject_physical_fault = False
            inject_vision_anomaly = (step in (50, 51))
            is_blurred = False
            is_occluded = False
        elif step < 75:  # Sustained surface defect (steps 53-74)
            state = MachineState.RUNNING
            inject_physical_fault = False
            inject_vision_anomaly = True
            is_blurred = False
            is_occluded = False
            if sustained_defect_onset_step is None:
                sustained_defect_onset_step = step
        elif step < 85:  # Optical blur degradation (steps 75-84)
            state = MachineState.RUNNING
            inject_physical_fault = False
            inject_vision_anomaly = False
            is_blurred = True
            is_occluded = False
        elif step < 110:  # Physical machine FAULT (steps 85-109)
            state = MachineState.FAULT
            inject_physical_fault = True
            inject_vision_anomaly = True
            is_blurred = False
            is_occluded = False
        elif step < 130:  # MAINTENANCE state testing (steps 110-129)
            state = MachineState.MAINTENANCE
            inject_physical_fault = True  # High noisy signals during motor test
            inject_vision_anomaly = True
            is_blurred = False
            is_occluded = False
        else:  # Steady-state nominal RUNNING (steps 130-149)
            state = MachineState.RUNNING
            inject_physical_fault = False
            inject_vision_anomaly = False
            is_blurred = False
            is_occluded = False

        # 2. Multi-modal simulation & inference execution
        frame = create_synthetic_test_frame(step, state, is_blurred=is_blurred, is_occluded=is_occluded)
        inf_res = engine.run_inference(frame, inject_anomaly=inject_vision_anomaly)
        sensor_read = simulator.step(machine_state=state, inject_fault=inject_physical_fault)

        # 3. Policy evaluation
        decision = policy.evaluate(inf_res, sensor_read)

        # 4. KPI Tracking
        raw_is_high = (
            inf_res.vision_score >= policy_config.thresholds.vision_high
            or sensor_read.sensor_score >= policy_config.thresholds.sensor_anomaly
        )
        if raw_is_high:
            raw_threshold_crossings += 1

        if decision.risk_state == RiskState.HIGH_SEVERITY:
            policy_escalations += 1
            if step in (50, 51, 52):
                transient_high_severity_count += 1
            if sustained_defect_onset_step is not None and first_escalation_step is None and step >= sustained_defect_onset_step:
                first_escalation_step = step

        # Cumulative alert reduction percentage
        if raw_threshold_crossings > 0:
            reduction_pct = max(0.0, 100.0 * (1.0 - (policy_escalations / raw_threshold_crossings)))
        else:
            reduction_pct = 100.0

        # Print representative telemetry rows
        is_event = (
            step in (0, 24, 25, 49, 50, 51, 52, 53, 54, 55, 56, 57, 60, 74, 75, 84, 85, 86, 109, 110, 129, 130, 149)
            or (decision.risk_state == RiskState.HIGH_SEVERITY)
            or (step % 10 == 0)
        )

        if is_event:
            vis_ema_str = f"{decision.smoothed_scores['vision_ema']:.3f}"
            sens_ema_str = f"{decision.smoothed_scores['sensor_ema']:.3f}"
            print(
                f"{step:>4} | {state.value:<11} | {inf_res.vision_score:>6.3f} | {vis_ema_str:>6} | "
                f"{sens_ema_str:>7} | {decision.risk_state.value:<15} | {decision.trigger_reason.value:<28} | "
                f"{decision.cooldown_remaining:>4} | {reduction_pct:>8.1f}%"
            )

    print(separator)

    # 5. Calculate final summary KPIs
    suppression_factor = 1.0 - (policy_escalations / max(1, raw_threshold_crossings))
    detection_latency = (
        (first_escalation_step - sustained_defect_onset_step)
        if (first_escalation_step is not None and sustained_defect_onset_step is not None)
        else 0
    )

    print("\n--- PHASE 2 OPERATIONAL KPI SUMMARY REPORT ---")
    print(f"Total Steps Simulated:                 {num_steps}")
    print(f"Total Raw Single-Frame Crossings:      {raw_threshold_crossings}")
    print(f"Total Policy Escalations:              {policy_escalations}")
    print(f"Alert Fatigue Suppression Factor:      {suppression_factor:.2%} (rho_supp)")
    print(f"Transient Glitch Escalation Count:     {transient_high_severity_count}")
    print(f"Mean Defect Detection Latency:         {detection_latency} frames")

    # Assertions for Phase 2 Acceptance
    assert transient_high_severity_count == 0, "Transient glitches must never trigger HIGH_SEVERITY."
    assert suppression_factor >= 0.60, f"Suppression factor ({suppression_factor:.2%}) must be >= 60%."
    assert policy_escalations >= 2, "Real sustained defects and faults must trigger escalations."
    logger.info("Phase 2 temporal policy verification passed with 100% KPI compliance.")
    return True


if __name__ == "__main__":
    success = run_phase2_verification()
    sys.exit(0 if success else 1)