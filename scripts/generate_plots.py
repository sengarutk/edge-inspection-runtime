#!/usr/bin/env python3
"""Generate publication-ready ablation figures and Pareto curves."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_pareto_and_workload_plots(
    summary_json_path: Path = PROJECT_ROOT / "results" / "ablation" / "ablation_summary.json",
    output_png_path: Path = PROJECT_ROOT / "docs" / "figures" / "pareto_workload_analysis.png",
) -> None:
    """Generate high-resolution multi-panel ablation and operator workload analysis figure."""
    if not summary_json_path.exists():
        print(f"[ERROR] Summary JSON not found at {summary_json_path}. Run ablation study first.")
        sys.exit(1)

    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios_data = data.get("scenarios", {})
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    # Set publication styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 14,
    })

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=300)

    policy_colors = {
        "BASELINE": "#d9534f",      # Red
        "EMA_ONLY": "#f0ad4e",      # Orange
        "EMA_KOFN": "#5bc0de",      # Light Blue
        "NO_COOLDOWN": "#9b59b6",  # Purple
        "NO_FUSION": "#34495e",    # Dark Slate
        "FULL_POLICY": "#2ecc71",  # Emerald Green
    }
    policy_markers = {
        "BASELINE": "o",
        "EMA_ONLY": "s",
        "EMA_KOFN": "^",
        "NO_COOLDOWN": "v",
        "NO_FUSION": "D",
        "FULL_POLICY": "*",
    }

    # -------------------------------------------------------------
    # Plot 1: Operator Fatigue vs Delay Pareto Curve (sustained_defects & transient_glitches)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.set_title("(a) Alert Suppression vs Detection Latency", fontweight="bold")
    ax1.set_xlabel("Mean Detection Latency (Frames)")
    ax1.set_ylabel("Alert Suppression Factor (rho_supp)")

    for sc_name in ["sustained_defects", "transient_glitches"]:
        sc = scenarios_data.get(sc_name, {})
        for p_name, p_data in sc.items():
            latency = p_data.get("mean_detection_latency_frames", {}).get("mean", 0.0)
            suppression = p_data.get("alert_suppression_factor", {}).get("mean", 0.0)
            alpha_val = 0.9 if sc_name == "sustained_defects" else 0.5
            lbl = f"{p_name} ({sc_name[:4]})" if sc_name == "sustained_defects" else None

            ax1.scatter(
                latency,
                suppression,
                color=policy_colors.get(p_name, "black"),
                marker=policy_markers.get(p_name, "o"),
                s=120 if p_name == "FULL_POLICY" else 75,
                alpha=alpha_val,
                label=lbl,
                edgecolors="black",
                linewidth=0.8,
            )

    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 2: Operator Review Hourly Rate across Scenarios
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.set_title("(b) Operator Review Load Rate Comparison", fontweight="bold")
    ax2.set_xlabel("Industrial Workload Scenario")
    ax2.set_ylabel("Review Load (Reviews / Hour)")

    sc_list = ["nominal", "transient_glitches", "sustained_defects", "sensor_drift_dropout", "distribution_shift"]
    x = np.arange(len(sc_list))
    width = 0.13

    for idx, (p_name, col) in enumerate(policy_colors.items()):
        means = []
        for sc_name in sc_list:
            sc = scenarios_data.get(sc_name, {})
            val = sc.get(p_name, {}).get("review_load_per_hour", {}).get("mean", 0.0)
            means.append(val)
        ax2.bar(x + (idx - 2.5) * width, means, width, label=p_name, color=col, alpha=0.85, edgecolor="black", linewidth=0.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["Nominal", "Glitches", "Sustained", "Drift/Drop", "Shift"], rotation=15)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 3: False Positive Rate vs True Positive Rate (Sensitivity-Specificity)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.set_title("(c) Classification Performance (TPR vs FPR)", fontweight="bold")
    ax3.set_xlabel("False Positive Rate (FPR)")
    ax3.set_ylabel("True Positive Rate (TPR)")

    for sc_name in ["sustained_defects", "distribution_shift", "transient_glitches"]:
        sc = scenarios_data.get(sc_name, {})
        for p_name, p_data in sc.items():
            fpr = p_data.get("false_positive_rate", {}).get("mean", 0.0)
            tpr = p_data.get("true_positive_rate", {}).get("mean", 0.0)
            lbl = p_name if sc_name == "sustained_defects" else None

            ax3.scatter(
                fpr,
                tpr,
                color=policy_colors.get(p_name, "black"),
                marker=policy_markers.get(p_name, "o"),
                s=140 if p_name == "FULL_POLICY" else 80,
                label=lbl,
                edgecolors="black",
                linewidth=0.8,
            )

    ax3.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random Guess")
    ax3.set_xlim(-0.05, 1.05)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.6)

    # -------------------------------------------------------------
    # Plot 4: Fatigue-Delay Tradeoff Index (Phi) by Policy Mode
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.set_title("(d) Fatigue-Delay Tradeoff Index (Phi)", fontweight="bold")
    ax4.set_xlabel("Policy Evaluation Mode")
    ax4.set_ylabel("Tradeoff Index Phi = rho_supp / (Delta t + 1)")

    p_modes = list(policy_colors.keys())
    overall_phis = []
    overall_errs = []

    for p_name in p_modes:
        phis = []
        for sc_name, sc in scenarios_data.items():
            p_val = sc.get(p_name, {}).get("fatigue_delay_tradeoff_index", {}).get("mean", 0.0)
            phis.append(p_val)
        overall_phis.append(np.mean(phis) if phis else 0.0)
        overall_errs.append(np.std(phis) if phis else 0.0)

    bars = ax4.bar(
        p_modes,
        overall_phis,
        yerr=overall_errs,
        capsize=4,
        color=[policy_colors[p] for p in p_modes],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.8,
    )
    ax4.set_xticks(range(len(p_modes)))
    ax4.set_xticklabels(p_modes, rotation=20)
    ax4.grid(True, linestyle="--", alpha=0.6)

    # Add numeric labels on bars
    for bar, val in zip(bars, overall_phis):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.suptitle("Empirical Ablation Analysis & Operator Reliability Tradeoffs", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_png_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"[SUCCESS] High-resolution Pareto & workload analysis plot saved to {output_png_path}")


if __name__ == "__main__":
    generate_pareto_and_workload_plots()
