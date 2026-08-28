#!/usr/bin/env python3
"""Verification script for Phase 1 of Edge Inspection Runtime.

Runs a continuous 100-step multi-modal simulation exercising:
1. State transitions: IDLE -> RUNNING -> FAULT -> MAINTENANCE
2. Optical health inspection & visual anomaly detection
3. Physical sensor simulation, thermal convergence, and composite anomaly scoring
4. Formatted console telemetry reporting with UUIDv4 and ISO-8601 UTC validation
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from loguru import logger

from src.config import load_sensor_config, load_system_config
from src.inference_service import InferenceEngine
from src.sensor_simulator import MachineState, SensorSimulator


def create_synthetic_test_frame(step: int, state: MachineState, inject_optical_fault: bool) -> np.ndarray:
    """Create synthetic test camera frame simulating normal or degraded optical conditions."""
    h, w = 224, 224
    if inject_optical_fault:
        if step % 2 == 0:
            # Simulate heavy defocus/blur
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.circle(img, (w // 2, h // 2), 60, (200, 200, 200), -1)
            return cv2.GaussianBlur(img, (35, 35), sigmaX=20.0)
        else:
            # Simulate dark occlusion (e.g., lens blocked)
            return np.zeros((h, w, 3), dtype=np.uint8)

    # Sharp industrial part frame with high contrast edges
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (40, 40), (184, 184), (220, 220, 220), 2)
    cv2.circle(frame, (112, 112), 35, (180, 180, 180), 2)
    cv2.line(frame, (50, 112), (174, 112), (150, 150, 150), 1)

    offset = int(5 * np.sin(step * 0.2))
    cv2.circle(frame, (112 + offset, 112), 8, (255, 255, 255), -1)
    return frame


def run_verification(num_steps: int = 100) -> bool:
    """Execute end-to-end continuous verification loop.

    Args:
        num_steps: Total simulation steps (default 100).

    Returns:
        True if all telemetry and state assertions pass.
    """
    logger.info("Initializing Phase 1 Edge Inspection Runtime Verification...")

    system_config = load_system_config()
    sensor_config = load_sensor_config()

    engine = InferenceEngine(config=system_config, seed=42)
    simulator = SensorSimulator(config=sensor_config, machine_id=system_config.machine_id, seed=42)
    simulator.reset()

    header = (
        f"{'Step':>4} | {'State':<11} | {'Temp(C)':>7} | {'Vib(g)':>6} | "
        f"{'Curr(A)':>7} | {'SensScore':>9} | {'VisScore':>8} | "
        f"{'OptValid':<8} | {'Degr':<5} | {'Lat(ms)':>7} | {'Status Notes':<22}"
    )
    separator = "-" * len(header)

    print("\n" + "=" * len(header))
    print(" INDUSTRIAL EDGE INSPECTION RUNTIME - 100-STEP PHASE 1 SIMULATION")
    print("=" * len(header))
    print(header)
    print(separator)

    fault_detected_count = 0
    optical_degraded_count = 0

    for step in range(num_steps):
        # State progression schedule
        if step < 25:
            state = MachineState.IDLE
            inject_physical_fault = False
            inject_optical_fault = False
            inject_vision_anomaly = False
            dropout_channels = None
            notes = "Cold idle state"
        elif step < 60:
            state = MachineState.RUNNING
            inject_physical_fault = False
            inject_optical_fault = (step == 45 or step == 46)
            inject_vision_anomaly = (step == 50 or step == 51)
            dropout_channels = ["current"] if step == 40 else None
            notes = "Normal operational run"
            if inject_optical_fault:
                notes = "Optical blur/occlusion"
            elif inject_vision_anomaly:
                notes = "Visual surface defect"
            elif dropout_channels:
                notes = "Current sensor dropout"
        elif step < 80:
            state = MachineState.FAULT
            inject_physical_fault = True
            inject_optical_fault = False
            inject_vision_anomaly = True
            dropout_channels = None
            notes = "CRITICAL MACHINE FAULT"
        else:
            state = MachineState.MAINTENANCE
            inject_physical_fault = False
            inject_optical_fault = False
            inject_vision_anomaly = False
            dropout_channels = None
            notes = "Scheduled maintenance"

        # 1. Generate multi-modal signals & execute inference
        frame = create_synthetic_test_frame(step, state, inject_optical_fault)
        reading = simulator.step(
            machine_state=state,
            inject_fault=inject_physical_fault,
            simulate_dropout=dropout_channels,
        )
        inf_res = engine.run_inference(frame, inject_anomaly=inject_vision_anomaly)

        # 2. Strict contract validation
        assert len(reading.reading_id) == 36, "reading_id must be valid UUIDv4"
        assert len(inf_res.frame_id) == 36, "frame_id must be valid UUIDv4"
        assert reading.timestamp_utc.endswith("Z"), "timestamp_utc must be ISO-8601 UTC"
        assert inf_res.timestamp_utc.endswith("Z"), "timestamp_utc must be ISO-8601 UTC"
        assert reading.machine_id == "press_unit_04"
        assert inf_res.camera_id == "line1_overhead_cam01"

        # 3. Track counters
        if reading.sensor_score > 0.70 or inf_res.vision_score > 0.70:
            fault_detected_count += 1
        if not inf_res.optical_health.is_valid:
            optical_degraded_count += 1

        # 4. Print periodic telemetry rows
        is_event = (
            inject_physical_fault
            or inject_optical_fault
            or inject_vision_anomaly
            or (dropout_channels is not None)
            or (step % 5 == 0)
            or (step == num_steps - 1)
        )
        if is_event:
            opt_valid_str = "VALID" if inf_res.optical_health.is_valid else "DEGRADED"
            degr_str = "YES" if reading.is_degraded else "NO"
            print(
                f"{step:>4} | {state.value:<11} | {reading.temperature_c:>7.2f} | "
                f"{reading.vibration_rms:>6.3f} | {reading.current_amps:>7.2f} | "
                f"{reading.sensor_score:>9.4f} | {inf_res.vision_score:>8.4f} | "
                f"{opt_valid_str:<8} | {degr_str:<5} | {inf_res.latency_ms:>7.2f} | {notes:<22}"
            )

    print(separator)
    print("\n--- VERIFICATION SUMMARY REPORT ---")
    print(f"Total Steps Simulated:          {num_steps}")
    print(f"Final Internal Temperature:     {simulator.internal_temp_c:.2f} C")
    print(f"Optical Degradations Handled:   {optical_degraded_count}")
    print(f"Total High Anomaly Steps:       {fault_detected_count}")

    # Assertions for Phase 1 Success
    assert optical_degraded_count >= 2, "Optical degradations should be captured."
    assert fault_detected_count >= 20, "Fault state steps should produce high anomaly scores."
    logger.info("Phase 1 verification completed successfully with 100% telemetry validation.")
    return True


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)