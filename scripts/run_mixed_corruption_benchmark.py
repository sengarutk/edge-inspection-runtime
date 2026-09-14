"""Mixed-Corruption Industrial Stream Benchmark Suite.

Evaluates BASELINE vs FULL_POLICY under stochastic mixed optical & RF corruptions:
- Gaussian sensor noise spikes (sigma = 20.0)
- Optical motion / defocus blur (kernel = 7)
- Lossy JPEG compression artifacts (quality = 30)
- Empirical corruption injection probability p = 0.20
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PolicyConfig, PolicyMode
from src.inference_service import InferenceEngine
from src.policy import RiskState, TemporalPolicyEngine
from src.sensor_simulator import MachineState, SensorReading, SensorSimulator
from src.stream_models import MixedCorruptionStream


def run_single_scenario_benchmark(
    scenario_name: str,
    policy_mode: PolicyMode,
    n_cycles: int = 300,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute a single mixed-corruption simulation run."""
    rng = np.random.RandomState(seed)
    corruption_stream = MixedCorruptionStream(
        p_corrupt=0.20,
        noise_sigma=20.0,
        blur_kernel=7,
        jpeg_quality=30,
        seed=seed,
    )

    inference_engine = InferenceEngine(seed=seed)
    policy_config = PolicyConfig(policy_mode=policy_mode)
    policy_engine = TemporalPolicyEngine(config=policy_config)
    sensor_sim = SensorSimulator(seed=seed)

    escalations = 0
    false_alarms = 0
    true_positives = 0
    total_ground_truth_defects = 0
    latencies_ms: List[float] = []

    # Scenario setup
    is_transient = scenario_name == "transient_glitches"
    is_sustained = scenario_name == "sustained_defects"

    for cycle in range(n_cycles):
        t0 = time.perf_counter()

        # 1. Ground truth label
        is_true_defect = False
        if is_sustained and (100 <= cycle < 160):
            is_true_defect = True
            total_ground_truth_defects += 1

        # 2. Synthetic base image frame
        base_frame = rng.randint(40, 200, (224, 224, 3), dtype=np.uint8)
        if is_true_defect:
            # Draw synthetic scratch
            base_frame[100:110, 50:180] = 255
        elif is_transient and (cycle in (50, 51, 150, 151, 250)):
            # 1-2 frame camera flare
            base_frame[:, :] = 245

        # 3. Apply stochastic mixed corruption
        corrupted_frame, applied_corruptions = corruption_stream.corrupt_frame(base_frame, step=cycle)

        # 4. Inference & optical health evaluation
        inf_result = inference_engine.infer(corrupted_frame, inject_anomaly=is_true_defect)

        # 5. Sensor telemetry
        reading = sensor_sim.step(
            machine_state=MachineState.RUNNING,
            inject_fault=is_true_defect,
        )

        # 6. Policy decision
        decision = policy_engine.evaluate(
            inference_result=inf_result,
            sensor_reading=reading,
        )

        latency = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency)

        is_escalated = decision.risk_state in (RiskState.REVIEW_REQUIRED, RiskState.HIGH_SEVERITY)
        if is_escalated:
            escalations += 1
            if is_true_defect:
                true_positives += 1
            else:
                false_alarms += 1

    duration_hours = n_cycles / (30.0 * 3600.0)
    fa_rate = round(false_alarms / max(duration_hours, 1e-4), 2)
    tpr = round(true_positives / max(total_ground_truth_defects, 1), 4) if is_sustained else 1.0

    return {
        "scenario": scenario_name,
        "policy_mode": policy_mode.value,
        "total_cycles": n_cycles,
        "total_escalations": escalations,
        "false_alarms_per_hour": fa_rate,
        "true_positive_rate": tpr,
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "deadline_miss_rate": round(float(np.mean([lat > 33.333 for lat in latencies_ms])), 4),
    }


def run_mixed_corruption_benchmark_suite(
    output_json: str = "results/mixed_corruption_summary.json",
) -> Dict[str, Any]:
    """Run full benchmark across nominal, transient_glitches, and sustained_defects."""
    scenarios = ["nominal", "transient_glitches", "sustained_defects"]
    policies = [PolicyMode.BASELINE, PolicyMode.FULL_POLICY]

    results: Dict[str, Any] = {}

    for scen in scenarios:
        results[scen] = {}
        for pol in policies:
            results[scen][pol.value] = run_single_scenario_benchmark(
                scenario_name=scen,
                policy_mode=pol,
                n_cycles=300,
                seed=42,
            )

    # Compute aggregate comparative summary
    baseline_fa = float(np.mean([results[s]["BASELINE"]["false_alarms_per_hour"] for s in scenarios]))
    full_fa = float(np.mean([results[s]["FULL_POLICY"]["false_alarms_per_hour"] for s in scenarios]))
    if baseline_fa > 0:
        suppression_ratio = round(float(max(0.0, 1.0 - (full_fa / baseline_fa))), 4)
    else:
        suppression_ratio = 1.0 if full_fa == 0 else 0.0

    summary_payload = {
        "corruption_parameters": {
            "p_corrupt": 0.20,
            "noise_sigma": 20.0,
            "blur_kernel": 7,
            "jpeg_quality": 30,
        },
        "aggregate_suppression_ratio": suppression_ratio,
        "scenarios": results,
    }

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    logger.info(f"Mixed corruption benchmark completed -> {output_json} (Suppression: {suppression_ratio*100:.1f}%)")
    return summary_payload


if __name__ == "__main__":
    run_mixed_corruption_benchmark_suite()
