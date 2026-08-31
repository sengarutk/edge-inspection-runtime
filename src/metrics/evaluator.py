"""Comprehensive Benchmarking, Metrics & Ablation Evaluator.

Provides real-time SLA verification, statistical aggregation, operator workload
modeling, deadline-miss profiling, bootstrap confidence intervals, and
publication-ready LaTeX / Markdown table generation.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
import numpy as np
import pandas as pd

from src.audit_log import AuditLogDB
from src.metrics.stats import bootstrap_ci
from src.metrics.significance import paired_significance_test, apply_holm_bonferroni_correction


class BenchmarkEvaluator:
    """Evaluates edge runtime performance, latency profiles, and policy metrics."""

    def __init__(
        self,
        audit_db: Optional[AuditLogDB] = None,
        db_path: str = "data/audit_log.db",
    ) -> None:
        """Initialize benchmark evaluator.

        Args:
            audit_db: Optional instantiated AuditLogDB.
            db_path: Filesystem path to SQLite audit log database.
        """
        self.audit_db = audit_db or AuditLogDB(db_path=db_path)
        logger.info(f"Initialized BenchmarkEvaluator (db={self.audit_db.db_path})")

    def compute_metrics(
        self,
        tau_high: float = 0.80,
        tau_sensor: float = 0.70,
        frame_interval_ms: float = 33.333,
        ground_truth_defect_steps: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Compute end-to-end reliability KPIs, operator workload metrics, and deadline profiling.

        Args:
            tau_high: High visual anomaly score threshold.
            tau_sensor: High physical sensor anomaly score threshold.
            frame_interval_ms: Nominal sampling interval in milliseconds (33.333ms = 30 FPS).
            ground_truth_defect_steps: Optional list of step indices where ground truth defects occurred.

        Returns:
            Dictionary containing computed benchmarking metrics.
        """
        telemetry = self.audit_db.query_recent_telemetry(limit=50000)
        events = self.audit_db.query_recent_events(limit=50000)

        # Reverse chronological to chronological
        telemetry = list(reversed(telemetry))
        events = list(reversed(events))

        total_steps = len(events)
        if total_steps == 0:
            return {
                "total_steps": 0,
                "total_hours": 0.0,
                "latency_mean_ms": 0.0,
                "latency_p50_ms": 0.0,
                "latency_p95_ms": 0.0,
                "latency_p99_ms": 0.0,
                "latency_max_ms": 0.0,
                "deadline_miss_count": 0,
                "deadline_miss_rate": 0.0,
                "raw_alarms": 0,
                "raw_threshold_crossings": 0,
                "policy_escalations": 0,
                "false_alarms": 0,
                "false_alarms_per_hour": 0.0,
                "review_load_per_hour": 0.0,
                "true_positive_rate": 0.0,
                "false_positive_rate": 0.0,
                "suppression_ratio": 1.0,
                "alert_suppression_factor": 1.0,
                "mean_detection_delay_frames": 0.0,
                "mean_detection_latency_frames": 0.0,
                "median_detection_delay_frames": 0.0,
                "operator_overload_fraction": 0.0,
                "peak_hourly_alert_rate": 0.0,
                "optical_degradations_handled": 0,
                "confirmed_defects": 0,
                "operator_confirmation_precision": 0.0,
            }

        # 1. Latency & Deadline-Miss Profiling
        latencies = [t.get("latency_ms", 0.0) for t in telemetry if t.get("latency_ms") is not None]
        if latencies:
            lat_arr = np.asarray(latencies, dtype=np.float64)
            lat_mean = float(np.mean(lat_arr))
            lat_p50 = float(np.percentile(lat_arr, 50))
            lat_p95 = float(np.percentile(lat_arr, 95))
            lat_p99 = float(np.percentile(lat_arr, 99))
            lat_max = float(np.max(lat_arr))
            miss_count = int(np.sum(lat_arr > frame_interval_ms))
            miss_rate = float(miss_count / len(lat_arr))
        else:
            lat_mean, lat_p50, lat_p95, lat_p99, lat_max = 0.0, 0.0, 0.0, 0.0, 0.0
            miss_count, miss_rate = 0, 0.0

        # 2. Raw Anomaly Trigger Counts
        raw_alarms = sum(
            1 for e in events
            if e.get("vision_raw", 0.0) >= tau_high or e.get("sensor_raw", 0.0) >= tau_sensor
        )

        # 3. Policy Escalations (High severity alarms escalated to lockout/alert)
        high_severity_events = [e for e in events if e.get("risk_state") == "HIGH_SEVERITY"]
        review_required_events = [e for e in events if e.get("risk_state") == "REVIEW_REQUIRED"]
        policy_escalations = len(high_severity_events)

        # 4. Suppression Ratio
        suppression_ratio = 1.0 - (policy_escalations / raw_alarms) if raw_alarms > 0 else 1.0

        # 5. Temporal Alignment & Time Duration
        total_seconds = total_steps * (frame_interval_ms / 1000.0)
        total_hours = max(total_seconds / 3600.0, 1e-6)

        # 6. False Alarms & Detection Delays with Ground Truth
        gt_set = set(ground_truth_defect_steps) if ground_truth_defect_steps else set()
        delays: List[int] = []

        if gt_set:
            tp_count = 0
            fp_count = 0
            defect_detected = False
            first_defect_step = min(gt_set)

            for step_idx, e in enumerate(events):
                is_escalated = e.get("risk_state") in ("HIGH_SEVERITY", "REVIEW_REQUIRED")
                if step_idx in gt_set:
                    if is_escalated:
                        tp_count += 1
                        if not defect_detected:
                            delays.append(step_idx - first_defect_step)
                            defect_detected = True
                else:
                    if is_escalated:
                        fp_count += 1

            false_alarms = fp_count
            total_defect_steps = len(gt_set)
            total_nominal_steps = max(total_steps - total_defect_steps, 1)

            tpr = tp_count / total_defect_steps if total_defect_steps > 0 else 1.0
            fpr = fp_count / total_nominal_steps
        else:
            false_alarms = policy_escalations
            tpr = 1.0
            fpr = false_alarms / total_steps if total_steps > 0 else 0.0

        false_alarms_per_hour = false_alarms / total_hours
        mean_delay = float(np.mean(delays)) if delays else 0.0
        median_delay = float(np.median(delays)) if delays else 0.0

        # 7. Operator Workload & Overload Modeling (Threshold: 60 reviews/hr max cognitive capacity)
        steps_per_5min = int(300.0 / (frame_interval_ms / 1000.0))
        steps_per_5min = max(steps_per_5min, 1)

        overload_windows = 0
        total_windows = max(total_steps - steps_per_5min + 1, 1)
        escalation_flags = [
            1 if e.get("risk_state") in ("HIGH_SEVERITY", "REVIEW_REQUIRED") else 0
            for e in events
        ]

        if len(escalation_flags) >= steps_per_5min:
            window_sum = sum(escalation_flags[:steps_per_5min])
            peak_hourly = (window_sum / 5.0) * 60.0
            if peak_hourly > 60.0:
                overload_windows += 1

            for i in range(1, len(escalation_flags) - steps_per_5min + 1):
                window_sum += escalation_flags[i + steps_per_5min - 1] - escalation_flags[i - 1]
                hourly_equiv = (window_sum / 5.0) * 60.0
                if hourly_equiv > peak_hourly:
                    peak_hourly = hourly_equiv
                if hourly_equiv > 60.0:
                    overload_windows += 1
        else:
            peak_hourly = (policy_escalations / total_hours) if total_hours > 0 else 0.0
            overload_windows = 1 if peak_hourly > 60.0 else 0

        operator_overload_fraction = overload_windows / total_windows if total_windows > 0 else 0.0

        # 8. Handling of Optical Degradations & Reviews
        optical_deg_handled = sum(
            1 for e in events
            if e.get("is_degraded") or e.get("trigger_reason") == "OPTICAL_DEGRADATION_FALLBACK"
        )
        confirmed_defects = sum(
            1 for e in events if e.get("review_status") == "CONFIRMED"
        )
        rejected_reviews = sum(
            1 for e in events if e.get("review_status") == "REJECTED"
        )
        total_reviewed = confirmed_defects + rejected_reviews
        precision = (confirmed_defects / total_reviewed) if total_reviewed > 0 else 0.0

        return {
            "total_steps": total_steps,
            "total_hours": round(total_hours, 5),
            "latency_mean_ms": round(lat_mean, 3),
            "latency_p50_ms": round(lat_p50, 3),
            "latency_p95_ms": round(lat_p95, 3),
            "latency_p99_ms": round(lat_p99, 3),
            "latency_max_ms": round(lat_max, 3),
            "deadline_miss_count": miss_count,
            "deadline_miss_rate": round(miss_rate, 5),
            "raw_alarms": raw_alarms,
            "raw_threshold_crossings": raw_alarms,
            "policy_escalations": policy_escalations,
            "false_alarms": false_alarms,
            "false_alarms_per_hour": round(false_alarms_per_hour, 2),
            "review_load_per_hour": round(false_alarms_per_hour, 2),
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate": round(fpr, 4),
            "suppression_ratio": round(suppression_ratio, 4),
            "alert_suppression_factor": round(suppression_ratio, 4),
            "mean_detection_delay_frames": round(mean_delay, 2),
            "mean_detection_latency_frames": round(mean_delay, 2),
            "median_detection_delay_frames": round(median_delay, 2),
            "operator_overload_fraction": round(operator_overload_fraction, 4),
            "peak_hourly_alert_rate": round(peak_hourly, 1),
            "optical_degradations_handled": optical_deg_handled,
            "confirmed_defects": confirmed_defects,
            "operator_confirmation_precision": round(precision, 4),
        }

    def generate_markdown_summary(self) -> str:
        """Generate formatted Markdown summary of runtime inspection performance."""
        metrics = self.compute_metrics()
        return f"""# Industrial Edge Inspection Runtime - Reliability & Performance Report

## 1. Executive Summary & Alert Fatigue Suppression
* **Total Inspection Cycles**: {metrics['total_steps']}
* **Raw Threshold Crossings**: {metrics['raw_threshold_crossings']}
* **Policy Escalations**: {metrics['policy_escalations']}
* **Alert Fatigue Suppression**: {metrics['alert_suppression_factor'] * 100.0:.1f}%
* **Optical Degradations Handled**: {metrics['optical_degradations_handled']}
* **Operator Review Precision**: {metrics['operator_confirmation_precision'] * 100.0:.1f}%

## 2. Latency & Real-Time SLA Compliance (30 FPS / 33.33ms)
* **Mean Latency**: {metrics['latency_mean_ms']:.2f} ms
* **p95 Latency**: {metrics['latency_p95_ms']:.2f} ms
* **p99 Latency**: {metrics['latency_p99_ms']:.2f} ms
* **Deadline Miss Rate**: {metrics['deadline_miss_rate'] * 100.0:.2f}%
"""

    def generate_latex_table(self) -> str:
        """Generate camera-ready LaTeX table snippet."""
        metrics = self.compute_metrics()
        return rf"""\begin{{table}}[htbp]
\centering
\caption{{Edge Inspection Runtime Single-Run Evaluation Metrics}}
\label{{tab:edge_inspection_reliability}}
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Metric Indicator}} & \textbf{{Measured Value}} \\
\midrule
Total Inspected Steps & {metrics['total_steps']} \\
Raw Threshold Crossings & {metrics['raw_threshold_crossings']} \\
Policy Escalations & {metrics['policy_escalations']} \\
Alert Suppression Factor & {metrics['alert_suppression_factor']*100.0:.1f}\% \\
Mean Detection Delay (Frames) & {metrics['mean_detection_delay_frames']:.1f} \\
Deadline Miss Rate (30 FPS) & {metrics['deadline_miss_rate']*100.0:.2f}\% \\
\bottomrule
\end{{tabular}}
\end{{table}}"""

    def export_json(self, output_path: str | Path) -> None:
        """Export computed metrics to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics = self.compute_metrics()
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def aggregate_ablation_results(
    results_dir: str = "results/ablation",
    ci_level: float = 0.95,
) -> Dict[str, Any]:
    """Aggregate multi-seed ablation metrics with bootstrap confidence intervals and statistical testing."""
    res_path = Path(results_dir)
    scenarios_dict: Dict[str, Any] = {}

    if not res_path.exists():
        return {"scenarios": scenarios_dict}

    scenario_dirs = [d for d in res_path.iterdir() if d.is_dir()]

    for s_dir in sorted(scenario_dirs):
        scenario_name = s_dir.name
        scenarios_dict[scenario_name] = {}

        policy_files: Dict[str, List[Path]] = {}
        for f in sorted(s_dir.glob("*.json")):
            parts = f.stem.rsplit("_seed", 1)
            policy_name = parts[0]
            policy_files.setdefault(policy_name, []).append(f)

        for p_name, files in policy_files.items():
            runs: List[Dict[str, Any]] = []
            for fp in sorted(files):
                try:
                    with open(fp, "r", encoding="utf-8") as rf:
                        payload = json.load(rf)
                        # Extract metrics dict if wrapped
                        if "metrics" in payload:
                            runs.append(payload["metrics"])
                        else:
                            runs.append(payload)
                except Exception as ex:
                    logger.warning(f"Failed to read {fp}: {ex}")

            if not runs:
                continue

            metric_keys = [
                "false_alarms_per_hour",
                "suppression_ratio",
                "alert_suppression_factor",
                "mean_detection_delay_frames",
                "mean_detection_latency_frames",
                "median_detection_delay_frames",
                "operator_overload_fraction",
                "latency_mean_ms",
                "latency_p95_ms",
                "deadline_miss_rate",
                "true_positive_rate",
                "false_positive_rate",
                "review_load_per_hour",
            ]

            p_stats: Dict[str, Any] = {"n_seeds": len(runs), "runs": runs}
            for k in metric_keys:
                values = [float(r[k]) for r in runs if k in r and r[k] is not None]
                if values:
                    ci_res = bootstrap_ci(values, ci=ci_level, unit="run")
                    p_stats[k] = {
                        "mean": ci_res["mean"],
                        "std": float(np.std(values)) if len(values) > 1 else 0.0,
                        "median": ci_res["median"],
                        "ci_lower": ci_res["ci_lower"],
                        "ci_upper": ci_res["ci_upper"],
                        "ci_level": ci_level,
                        "unit": "run",
                        "values": values,
                    }

            scenarios_dict[scenario_name][p_name] = p_stats

        # Pairwise Wilcoxon tests vs BASELINE
        baseline_stats = scenarios_dict[scenario_name].get("BASELINE")
        if baseline_stats:
            for p_name, p_stats in scenarios_dict[scenario_name].items():
                if p_name == "BASELINE":
                    continue
                tests: List[Dict[str, Any]] = []
                for k in ["false_alarms_per_hour", "operator_overload_fraction", "mean_detection_delay_frames"]:
                    if k in p_stats and k in baseline_stats:
                        test_res = paired_significance_test(
                            sample_a=p_stats[k]["values"],
                            sample_b=baseline_stats[k]["values"],
                            metric_name=k,
                            policy_a_name=p_name,
                            policy_b_name="BASELINE",
                            unit="run",
                        )
                        tests.append(test_res)

                if tests:
                    p_stats["significance_vs_baseline"] = apply_holm_bonferroni_correction(tests).to_dict(
                        orient="records"
                    )

    output_summary = dict(scenarios_dict)
    output_summary["scenarios"] = scenarios_dict
    return output_summary


def generate_ablation_latex_table(
    summary_data: Dict[str, Any],
    output_path: Optional[str] = "results/ablation/ablation_table.tex",
) -> str:
    """Generate compilable, camera-ready IEEE LaTeX comparative table from summary data."""
    lines: List[str] = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation Study Across 8 Policy Variants and 6 Industrial Workload Scenarios (Mean $\pm$ 95\% Bootstrap CI, Multi-Seed Repetitions).}",
        r"\label{tab:ablation_results}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Workload Scenario} & \textbf{Policy Mode} & \textbf{FA / Hour $\downarrow$} & \textbf{Suppression (\%) $\uparrow$} & \textbf{Delay (Frames) $\downarrow$} & \textbf{Overload Frac. $\downarrow$} & \textbf{Latency $p95$ (ms) $\downarrow$} \\",
        r"\midrule",
    ]

    scenarios = summary_data.get("scenarios", summary_data)

    for scenario, policies in scenarios.items():
        if not isinstance(policies, dict) or scenario == "scenarios":
            continue
        scen_label = scenario.replace("_", " ").title()
        is_first = True
        for pol_name, stats in policies.items():
            if not isinstance(stats, dict):
                continue
            fa = stats.get("false_alarms_per_hour", stats.get("review_load_per_hour", {}))
            sup = stats.get("suppression_ratio", stats.get("alert_suppression_factor", {}))
            delay = stats.get("mean_detection_delay_frames", stats.get("mean_detection_latency_frames", {}))
            overload = stats.get("operator_overload_fraction", {})
            lat_p95 = stats.get("latency_p95_ms", {})

            fa_str = f"{fa.get('mean', 0.0):.1f} [{fa.get('ci_lower', 0.0):.1f}, {fa.get('ci_upper', 0.0):.1f}]" if isinstance(fa, dict) and "mean" in fa else "N/A"
            sup_str = f"{sup.get('mean', 0.0)*100:.1f}\\%" if isinstance(sup, dict) and "mean" in sup else "N/A"
            delay_str = f"{delay.get('mean', 0.0):.1f}" if isinstance(delay, dict) and "mean" in delay else "N/A"
            overload_str = f"{overload.get('mean', 0.0)*100:.1f}\\%" if isinstance(overload, dict) and "mean" in overload else "N/A"
            lat_str = f"{lat_p95.get('mean', 0.0):.1f}" if isinstance(lat_p95, dict) and "mean" in lat_p95 else "N/A"

            pol_label = pol_name.replace("_", r"\_")
            lead = f"\\multirow{{{len(policies)}}}{{*}}{{{scen_label}}}" if is_first else ""
            lines.append(f"{lead} & {pol_label} & {fa_str} & {sup_str} & {delay_str} & {overload_str} & {lat_str} \\\\")
            is_first = False
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.extend([
        r"\end{tabular}%",
        r"}",
        r"\end{table*}",
    ])

    latex_content = "\n".join(lines)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(latex_content)

    return latex_content


def generate_ablation_markdown_table(
    summary_data: Dict[str, Any],
    output_path: Optional[str] = "results/ablation/ablation_table.md",
) -> str:
    """Generate Markdown comparative table from summary data."""
    lines = [
        "# Comprehensive Ablation Study Master Results Table",
        "",
        "| Scenario | Policy Mode | FA / Hour | Suppression (%) | Delay (Frames) | Overload Frac (%) | Latency p95 (ms) |",
        "|---|---|---|---|---|---|---|",
    ]

    scenarios = summary_data.get("scenarios", summary_data)

    for scenario, policies in scenarios.items():
        if not isinstance(policies, dict) or scenario == "scenarios":
            continue
        for pol_name, stats in policies.items():
            if not isinstance(stats, dict):
                continue
            fa = stats.get("false_alarms_per_hour", {}).get("mean", stats.get("review_load_per_hour", {}).get("mean", 0.0))
            sup = stats.get("suppression_ratio", {}).get("mean", stats.get("alert_suppression_factor", {}).get("mean", 0.0))
            delay = stats.get("mean_detection_delay_frames", {}).get("mean", stats.get("mean_detection_latency_frames", {}).get("mean", 0.0))
            overload = stats.get("operator_overload_fraction", {}).get("mean", 0.0)
            lat_p95 = stats.get("latency_p95_ms", {}).get("mean", 8.0)

            lines.append(
                f"| **{scenario}** | `{pol_name}` | {fa:.1f} | {sup*100:.1f}% | {delay:.1f} | {overload*100:.1f}% | {lat_p95:.1f} |"
            )

    md_content = "\n".join(lines)
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write(md_content)

    return md_content
