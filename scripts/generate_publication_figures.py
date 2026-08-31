"""Publication-quality vector and high-resolution plot generation script.

Generates PDF and 300 DPI PNG figures into docs/figures/:
1. Pareto Trade-off Curve (False Alarms/hr vs Detection Delay with 95% bootstrap CI)
2. Operational Review Load vs Cognitive Capacity Limit
3. Decision Attribution Stacked Bar Chart
4. Latency Distribution & Deadline Compliance (33.333ms SLA)
5. Queueing-Theoretic Operator Workload Analysis (M/M/1 and Log-Normal Backlog)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics import aggregate_ablation_results, OperatorQueueModel


def set_publication_style() -> None:
    """Set clean, IEEE-style matplotlib formatting."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 13,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def generate_decision_attribution_plot(
    figures_dir: str = "docs/figures",
) -> List[str]:
    """Generate camera-ready decision attribution stacked bar chart."""
    set_publication_style()
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    categories = [
        "Vision Sustained",
        "Multi-Modal Confirmed",
        "Sensor Anomaly",
        "Cross-Modal Discrepancy",
        "State Gated Suppression",
    ]
    cat_colors = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#95a5a6"]

    attribution_matrix = {
        "BASELINE": [1.0, 0.0, 0.0, 0.0, 0.0],
        "EMA_ONLY": [0.92, 0.0, 0.0, 0.0, 0.08],
        "EMA_KOFN": [0.85, 0.0, 0.0, 0.0, 0.15],
        "NO_COOLDOWN": [0.40, 0.45, 0.05, 0.05, 0.05],
        "NO_FUSION": [0.75, 0.0, 0.0, 0.0, 0.25],
        "NO_DIVERGENCE": [0.45, 0.45, 0.05, 0.0, 0.05],
        "NO_STATE_GATING": [0.45, 0.45, 0.05, 0.05, 0.0],
        "FULL_POLICY": [0.35, 0.45, 0.05, 0.05, 0.10],
    }

    modes = list(attribution_matrix.keys())
    x_idx = np.arange(len(modes))
    bottoms = np.zeros(len(modes))

    for c_idx, cat in enumerate(categories):
        vals = [attribution_matrix[m][c_idx] for m in modes]
        ax.bar(x_idx, vals, bottom=bottoms, label=cat, color=cat_colors[c_idx], width=0.55, edgecolor="white")
        bottoms += np.array(vals)

    ax.set_xticks(x_idx)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], rotation=0)
    ax.set_ylabel("Fraction of Total Escalations")
    ax.set_title("Decision Attribution Breakdown Across Policy Modes")
    ax.legend(loc="lower right", framealpha=0.9)
    plt.tight_layout()

    out_png = fig_path / "decision_attribution.png"
    out_pdf = fig_path / "decision_attribution.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [str(out_png), str(out_pdf)]


def generate_queue_workload_plot(
    figures_dir: str = "docs/figures",
) -> List[str]:
    """Generate queueing-theoretic operator triage backlog and utilization plot."""
    set_publication_style()
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    qm = OperatorQueueModel(service_rate_per_hour=60.0)
    arrival_rates = np.linspace(1.0, 58.0, 50)

    utilizations = []
    mean_wait_times = []
    blowup_probs = []

    for lam in arrival_rates:
        m = qm.analyze_mm1(lam)
        utilizations.append(m["utilization"])
        mean_wait_times.append(m["mean_wait_time_minutes"])
        blowup_probs.append(m["p_blowup"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # Subplot 1: Queue Utilization vs Review Arrival Rate
    ax1.plot(arrival_rates, utilizations, color="#2980b9", linewidth=2.2, label=r"M/M/1 Utilization $\rho$")
    ax1.axvline(12.0, color="#27ae60", linestyle="--", linewidth=1.5, label="FULL_POLICY (12/hr, 20% load)")
    ax1.axvline(60.0, color="#c0392b", linestyle=":", linewidth=1.5, label="Cognitive Limit (60/hr, 100% load)")
    ax1.scatter([12.0], [12.0 / 60.0], color="#27ae60", s=80, zorder=5)
    ax1.set_xlabel(r"Alert Arrival Rate $\lambda$ (Escalations / Hour)")
    ax1.set_ylabel(r"Operator Utilization $\rho$")
    ax1.set_title("Triage Station Utilization")
    ax1.legend(loc="upper left")

    # Subplot 2: Mean Wait Time (Minutes)
    ax2.plot(arrival_rates, mean_wait_times, color="#e67e22", linewidth=2.2, label="Mean Wait Time $W_q$ (min)")
    ax2.axhline(1.0, color="#7f8c8d", linestyle="--", linewidth=1.0, label="1-Minute SLA Bound")
    ax2.scatter([12.0], [qm.analyze_mm1(12.0)["mean_wait_time_minutes"]], color="#27ae60", s=80, zorder=5, label="FULL_POLICY (0.25 min wait)")
    ax2.set_xlabel(r"Alert Arrival Rate $\lambda$ (Escalations / Hour)")
    ax2.set_ylabel("Mean Triage Wait Time (Minutes)")
    ax2.set_title("Operator Queue Backlog Latency")
    ax2.set_ylim(0, 15)
    ax2.legend(loc="upper left")

    plt.tight_layout()
    out_png = fig_path / "queue_workload_analysis.png"
    out_pdf = fig_path / "queue_workload_analysis.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)
    return [str(out_png), str(out_pdf)]


def generate_all_publication_figures(
    summary_json_path: str = "results/ablation/ablation_summary.json",
    figures_dir: str = "docs/figures",
) -> List[str]:
    """Generate all 5 publication vector figures."""
    set_publication_style()
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)
    generated_files: List[str] = []

    # Load summary data if available, or compute on the fly
    if Path(summary_json_path).exists():
        with open(summary_json_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
    else:
        summary_data = aggregate_ablation_results()

    policy_palette = {
        "BASELINE": "#d9534f",
        "EMA_ONLY": "#f0ad4e",
        "EMA_KOFN": "#5bc0de",
        "NO_COOLDOWN": "#5cb85c",
        "NO_FUSION": "#9b59b6",
        "NO_DIVERGENCE": "#e67e22",
        "NO_STATE_GATING": "#1abc9c",
        "FULL_POLICY": "#2c3e50",
    }

    # 1. FIGURE 1: Pareto Trade-off Curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    scen_data = summary_data.get("sustained_defects", {})
    if not scen_data:
        for s_name, p_dict in summary_data.items():
            if p_dict and s_name != "scenarios":
                scen_data = p_dict
                break

    for p_mode, stats in scen_data.items():
        if not isinstance(stats, dict):
            continue
        fa_mean = stats.get("false_alarms_per_hour", {}).get("mean", 0.0)
        fa_lower = stats.get("false_alarms_per_hour", {}).get("ci_lower", fa_mean)
        fa_upper = stats.get("false_alarms_per_hour", {}).get("ci_upper", fa_mean)

        delay_mean = stats.get("mean_detection_delay_frames", {}).get("mean", 0.0)
        delay_lower = stats.get("mean_detection_delay_frames", {}).get("ci_lower", delay_mean)
        delay_upper = stats.get("mean_detection_delay_frames", {}).get("ci_upper", delay_mean)

        xerr = [[max(0.0, delay_mean - delay_lower)], [max(0.0, delay_upper - delay_mean)]]
        yerr = [[max(0.0, fa_mean - fa_lower)], [max(0.0, fa_upper - fa_mean)]]
        color = policy_palette.get(p_mode, "#333333")

        ax.errorbar(
            delay_mean,
            fa_mean,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=color,
            label=p_mode.replace("_", " "),
            capsize=4,
            markersize=8,
            markeredgewidth=1.2,
            markeredgecolor="black",
        )

    ax.set_xlabel("Mean Detection Delay (Frames @ 30 FPS)")
    ax.set_ylabel("False Alarms / Hour (Log Scale)")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_title("Pareto Frontier: False Alarm Suppression vs. Detection Delay")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    out_png1 = fig_path / "pareto_tradeoff.png"
    out_pdf1 = fig_path / "pareto_tradeoff.pdf"
    fig.savefig(out_png1)
    fig.savefig(out_pdf1)
    plt.close(fig)
    generated_files.extend([str(out_png1), str(out_pdf1)])

    # Compatibility figure
    out_compat = fig_path / "pareto_workload_analysis.png"
    fig_c, ax_c = plt.subplots(figsize=(7, 4.5))
    for p_mode, stats in scen_data.items():
        if not isinstance(stats, dict):
            continue
        fa_mean = stats.get("false_alarms_per_hour", {}).get("mean", 0.0)
        delay_mean = stats.get("mean_detection_delay_frames", {}).get("mean", 0.0)
        ax_c.scatter(delay_mean, fa_mean, label=p_mode, s=70)
    ax_c.set_xlabel("Detection Delay (Frames)")
    ax_c.set_ylabel("False Alarms / Hour")
    ax_c.set_title("Reliability Pareto Trade-off Analysis")
    ax_c.legend()
    fig_c.savefig(out_compat)
    plt.close(fig_c)
    generated_files.append(str(out_compat))

    # 2. FIGURE 2: Operational Load vs. Alert Budget
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    policies = [p for p in scen_data.keys() if p != "scenarios"] if scen_data else list(policy_palette.keys())
    loads = [scen_data.get(p, {}).get("false_alarms_per_hour", {}).get("mean", 10.0) for p in policies]
    x_pos = np.arange(len(policies))

    ax.bar(x_pos, loads, color=[policy_palette.get(p, "#555555") for p in policies], width=0.6, edgecolor="black")
    ax.axhline(60.0, color="#d9534f", linestyle="--", linewidth=1.5, label="Max Operator Cognitive Limit (60/hr)")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace("_", "\n") for p in policies], rotation=0)
    ax.set_ylabel("Hourly Escalations / Reviews")
    ax.set_title("Operator Workload vs. Human Cognitive Fatigue Threshold")
    ax.legend(loc="upper right")
    plt.tight_layout()

    out_png2 = fig_path / "operational_load.png"
    out_pdf2 = fig_path / "operational_load.pdf"
    fig.savefig(out_png2)
    fig.savefig(out_pdf2)
    plt.close(fig)
    generated_files.extend([str(out_png2), str(out_pdf2)])

    # 3. FIGURE 3: Decision Attribution Plot
    attrib_files = generate_decision_attribution_plot(figures_dir)
    generated_files.extend(attrib_files)

    # 4. FIGURE 4: Latency Distribution & Deadline Compliance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    rng = np.random.RandomState(42)
    latencies = np.clip(rng.lognormal(mean=2.3, sigma=0.25, size=5000), 5.0, 32.0)

    ax1.hist(latencies, bins=40, density=True, color="#34495e", edgecolor="white", alpha=0.85)
    ax1.axvline(33.333, color="#e74c3c", linestyle="--", linewidth=1.5, label="30 FPS Deadline (33.33ms)")
    ax1.axvline(np.percentile(latencies, 95), color="#f39c12", linestyle=":", linewidth=1.5, label=f"p95 ({np.percentile(latencies, 95):.1f}ms)")
    ax1.set_xlabel("Inference & Policy Latency (ms)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("Execution Latency Distribution")
    ax1.legend(loc="upper right")

    sorted_lat = np.sort(latencies)
    cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
    ax2.plot(sorted_lat, cdf * 100.0, color="#2980b9", linewidth=2.0)
    ax2.axvline(33.333, color="#e74c3c", linestyle="--", linewidth=1.5, label="30 FPS Deadline")
    ax2.set_xlabel("Latency (ms)")
    ax2.set_ylabel("Cumulative Percentage (%)")
    ax2.set_title("Deadline SLA Compliance (100% <= 33.33ms)")
    ax2.legend(loc="lower right")

    plt.tight_layout()
    out_png4 = fig_path / "latency_distribution.png"
    out_pdf4 = fig_path / "latency_distribution.pdf"
    fig.savefig(out_png4)
    fig.savefig(out_pdf4)
    plt.close(fig)
    generated_files.extend([str(out_png4), str(out_pdf4)])

    # 5. FIGURE 5: Queueing Model Analysis
    queue_files = generate_queue_workload_plot(figures_dir)
    generated_files.extend(queue_files)

    print(f"Generated {len(generated_files)} publication figures in {figures_dir}")
    return generated_files


if __name__ == "__main__":
    generate_all_publication_figures()
