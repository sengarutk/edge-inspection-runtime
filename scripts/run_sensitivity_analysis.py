"""Hyperparameter sensitivity analysis benchmarking module."""

from __future__ import annotations

import concurrent.futures
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from loguru import logger
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_log import AuditLogDB
from src.config import ConfirmationWindowConfig, CooldownConfig, PolicyConfig, PolicyMode, ThresholdsConfig
from src.inference_service import InferenceEngine
from src.metrics import BenchmarkEvaluator
from src.policy import RiskState, TemporalPolicyEngine
from src.sensor_simulator import MachineState, SensorSimulator


def evaluate_single_grid_point(params: Tuple[float, int, float, int, int]) -> Dict[str, Any]:
    """Evaluate a single parameter combination on synthetic defect burst stream."""
    tau_h, k_val, tau_d, c_val, seed = params
    np.random.seed(seed)

    temp_dir = tempfile.mkdtemp(prefix=f"sens_{tau_h}_{k_val}_{tau_d}_{c_val}_")
    db_path = os.path.join(temp_dir, "sens.db")

    try:
        audit_db = AuditLogDB(db_path=db_path)
        with audit_db._lock:
            audit_db._conn.execute("PRAGMA synchronous = OFF;")
            audit_db._conn.execute("PRAGMA journal_mode = MEMORY;")

        p_cfg = PolicyConfig(
            policy_mode=PolicyMode.FULL_POLICY,
            thresholds=ThresholdsConfig(vision_high=tau_h, cross_modal_divergence=tau_d),
            confirmation_window=ConfirmationWindowConfig(window_size_n=10, consecutive_k=k_val),
            cooldown=CooldownConfig(cooldown_steps=c_val),
        )
        policy_engine = TemporalPolicyEngine(config=p_cfg)
        sensor_sim = SensorSimulator(seed=seed)
        inf_engine = InferenceEngine(seed=seed)

        # 120 steps total: 20 nominal, 30 defect, 70 recovery
        gt_defects = list(range(20, 50))
        rng_frame = np.random.RandomState(seed)
        frame_nom = rng_frame.randint(90, 160, (224, 224, 3), dtype=np.uint8)

        for step in range(120):
            is_defect = step in gt_defects
            v_score = inf_engine.run_inference(frame_nom, inject_anomaly=is_defect)
            s_reading = sensor_sim.step(machine_state=MachineState.RUNNING, inject_fault=is_defect)
            dec = policy_engine.evaluate(v_score, s_reading)
            audit_db.insert_risk_event(dec)

        evaluator = BenchmarkEvaluator(audit_db=audit_db)
        metrics = evaluator.compute_metrics(ground_truth_defect_steps=gt_defects)

        audit_db.close()
        return {
            "tau_high": tau_h,
            "consecutive_k": k_val,
            "tau_divergence": tau_d,
            "cooldown_steps": c_val,
            "false_alarms_per_hour": metrics["false_alarms_per_hour"],
            "suppression_ratio": metrics["suppression_ratio"],
            "mean_detection_delay_frames": metrics["mean_detection_delay_frames"],
            "operator_overload_fraction": metrics["operator_overload_fraction"],
            "true_positive_rate": metrics["true_positive_rate"],
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def run_sensitivity_sweep(
    output_file: str = "results/sensitivity/sensitivity_summary.json",
    workers: int = 8,
) -> Dict[str, Any]:
    """Execute parallel parameter sensitivity sweeps across grid points."""
    logger.remove()
    logger.add(sys.stderr, level="ERROR")
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tau_high_values = [0.70, 0.75, 0.80, 0.85]
    k_values = [3, 4, 5]
    tau_div_values = [0.35, 0.45, 0.55]
    cooldown_values = [0, 5, 10, 15, 20, 30]

    tasks = [
        (th, k, td, c, 42)
        for th in tau_high_values
        for k in k_values
        for td in tau_div_values
        for c in cooldown_values
    ]

    print(f"[INFO] Running {len(tasks)} sensitivity grid evaluations with ProcessPoolExecutor...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        sweep_results = list(executor.map(evaluate_single_grid_point, tasks))

    summary_data = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_evaluations": len(sweep_results),
        "parameter_ranges": {
            "tau_high": tau_high_values,
            "consecutive_k": k_values,
            "tau_divergence": tau_div_values,
            "cooldown_steps": cooldown_values,
        },
        "results": sweep_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[SUCCESS] Sensitivity sweep completed ({len(sweep_results)} configurations evaluated) -> {out_path}")
    return summary_data


if __name__ == "__main__":
    run_sensitivity_sweep()
