import os
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.visualization.publication_plots import (
    plot_pareto_latency_vs_aupro,
    plot_robustness_heatmap,
    plot_calibration_diagram,
    plot_robust_training_ablation
)
from scripts.generate_report import (
    generate_main_results_table,
    generate_deployment_table,
    generate_robustness_table
)


def test_plotting_functions(tmp_path):
    out_dir = str(tmp_path / "figures")
    os.makedirs(out_dir, exist_ok=True)

    summary_data = {
        "category": ["bottle", "bottle", "cable", "cable"],
        "method": ["patchcore", "padim", "patchcore", "padim"],
        "aupro_mean": [0.95, 0.90, 0.94, 0.88],
        "aupro_std": [0.01, 0.02, 0.01, 0.03],
        "p50_model_ms": [10.0, 5.0, 12.0, 6.0],
        "p95_model_ms": [12.0, 7.0, 15.0, 8.0],
        "fps_model": [100.0, 200.0, 83.3, 166.7],
        "p50_e2e_ms": [30.0, 25.0, 35.0, 28.0],
        "fps_e2e": [33.3, 40.0, 28.5, 35.7],
        "peak_vram_mb": [200.0, 300.0, 220.0, 310.0],
        "image_auroc_mean": [0.99, 0.95, 0.98, 0.92],
        "image_auroc_std": [0.005, 0.01, 0.008, 0.015],
        "pixel_auroc_mean": [0.97, 0.94, 0.96, 0.93],
        "pixel_auroc_std": [0.005, 0.01, 0.006, 0.01],
        "mrd_mean": [0.05, 0.12, 0.06, 0.15],
        "mrd_std": [0.01, 0.02, 0.01, 0.03]
    }
    summary_df = pd.DataFrame(summary_data)

    p1 = os.path.join(out_dir, "pareto.png")
    plot_pareto_latency_vs_aupro(summary_df, p1)
    assert os.path.exists(p1)

    heatmap_data = {
        "corruption_type": ["gaussian_blur", "gaussian_blur", "motion_blur", "motion_blur"],
        "severity": [1, 2, 1, 2],
        "delta_image_auroc": [0.02, 0.05, 0.03, 0.07],
        "delta_aupro": [0.01, 0.04, 0.02, 0.06]
    }
    heatmap_df = pd.DataFrame(heatmap_data)
    p2 = os.path.join(out_dir, "heatmap.png")
    plot_robustness_heatmap(heatmap_df, p2)
    assert os.path.exists(p2)

    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.4, 0.85])
    p3 = os.path.join(out_dir, "calibration.png")
    plot_calibration_diagram(y_true, y_scores, p3)
    assert os.path.exists(p3)

    ablation_data = {
        "category": ["bottle", "cable"],
        "nominal_clean_auroc": [0.99, 0.98],
        "nominal_corrupted_auroc": [0.92, 0.89],
        "robust_clean_auroc": [0.985, 0.975],
        "robust_corrupted_auroc": [0.96, 0.94]
    }
    ablation_df = pd.DataFrame(ablation_data)
    p4 = os.path.join(out_dir, "ablation.png")
    plot_robust_training_ablation(ablation_df, p4)
    assert os.path.exists(p4)


def test_report_generation(tmp_path):
    tables_dir = str(tmp_path / "tables")
    os.makedirs(tables_dir, exist_ok=True)

    summary_data = {
        "category": ["bottle", "cable"],
        "method": ["patchcore", "padim"],
        "image_auroc_mean": [0.99, 0.95],
        "image_auroc_std": [0.01, 0.02],
        "pixel_auroc_mean": [0.97, 0.94],
        "pixel_auroc_std": [0.005, 0.01],
        "aupro_mean": [0.95, 0.90],
        "aupro_std": [0.01, 0.02],
        "mrd_mean": [0.05, 0.12],
        "mrd_std": [0.01, 0.02],
        "p50_model_ms": [10.0, 5.0],
        "fps_model": [100.0, 200.0],
        "p50_e2e_ms": [30.0, 25.0],
        "peak_vram_mb": [200.0, 300.0]
    }
    summary_df = pd.DataFrame(summary_data)

    runs_data = {
        "category": ["bottle", "bottle", "cable", "cable"],
        "method": ["patchcore", "padim", "patchcore", "padim"],
        "image_auroc": [0.99, 0.95, 0.98, 0.92],
        "mrd_image_auroc": [0.05, 0.12, 0.06, 0.15],
        "mrd_aupro": [0.03, 0.08, 0.04, 0.10],
        "mean_performance_change_auroc": [0.05, 0.12, 0.06, 0.15]
    }
    runs_df = pd.DataFrame(runs_data)

    out_main = os.path.join(tables_dir, "main_results.tex")
    out_deploy = os.path.join(tables_dir, "deployment_profiling.tex")
    out_robustness = os.path.join(tables_dir, "robustness_mrd_mpc.tex")

    generate_main_results_table(summary_df, out_main)
    generate_deployment_table(summary_df, out_deploy)
    generate_robustness_table(runs_df, out_robustness)

    assert os.path.exists(out_main)
    assert os.path.exists(out_deploy)
    assert os.path.exists(out_robustness)
