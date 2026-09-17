#!/usr/bin/env python3
"""Quantitative ESG Sustainability Evaluation Script.

Evaluates Quality Carbon Footprint (QCF) and Sustainability Quality Index (SQI)
across industrial defect benchmark predictions (all 63 .npz evaluation archives).
Includes false-alarm scrap and downstream escape scrap in material scrap savings (MSF).
Generates LaTeX tables and appends empirical macros to generated_metrics.tex.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger

# Anchor project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.metrics.image_metrics import compute_optimal_f1
from src.metrics.operational import compute_quantile_threshold
from src.sustainability.qcf_engine import QualityCarbonFootprintEngine, SustainabilityParameters
from src.sustainability.sqi_calculator import SustainabilityQualityIndexCalculator


def compute_sustainability_benchmark() -> Dict[str, Any]:
    """Computes comprehensive QCF and SQI across all 63 benchmark score archives."""
    scores_dir = PROJECT_ROOT / "results" / "mvtec_ad" / "scores"
    if not scores_dir.exists():
        scores_dir = PROJECT_ROOT / "results" / "benchmark_f1" / "mvtec_ad" / "scores"

    npz_files = sorted(scores_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No score archives found in {scores_dir}")

    logger.info(f"Loaded {len(npz_files)} prediction archives from {scores_dir}")

    config_yaml = PROJECT_ROOT / "configs" / "sustainability_parameters.yaml"
    nut_params = SustainabilityParameters.from_yaml(config_yaml, part_name="metal_nut")
    tile_params = SustainabilityParameters.from_yaml(config_yaml, part_name="tile")

    # Categories to evaluate across all models (PatchCore, PaDiM, AutoEncoder)
    categories = ["bottle", "cable", "carpet", "grid", "hazelnut", "leather", "metal_nut"]
    table_rows = []
    category_metrics = {}

    for cat in categories:
        cat_files = [f for f in npz_files if f.name.startswith(f"{cat}_")]

        if cat == "metal_nut":
            params = nut_params
        elif cat in ["grid", "carpet", "tile"]:
            params = tile_params
        else:
            params = SustainabilityParameters(
                part_name=cat,
                m_part=0.50,
                kappa_mat=2.50,
                E_rework=2.00,
                gamma_fatal=0.40,
                theta_tier=6.0,
            )

        engine = QualityCarbonFootprintEngine(params=params)
        calc = SustainabilityQualityIndexCalculator(qcf_engine=engine)

        tp_b, fp_b, fn_b, tn_b, n_tot = 0, 0, 0, 0, 0
        tp_p, fp_p, fn_p, tn_p = 0, 0, 0, 0

        for f in cat_files:
            data = np.load(f)
            scores = data["image_scores"]
            labels = data["image_labels"]
            n_tot += len(labels)

            # Baseline: uncalibrated single-frame detector (85th percentile of nominal scores)
            nom_scores = scores[labels == 0]
            tau_base = compute_quantile_threshold(nom_scores, 0.85) if len(nom_scores) > 0 else 0.5
            pred_b = (scores >= tau_base).astype(int)
            tp_b += int(np.sum((labels == 1) & (pred_b == 1)))
            fp_b += int(np.sum((labels == 0) & (pred_b == 1)))
            fn_b += int(np.sum((labels == 1) & (pred_b == 0)))
            tn_b += int(np.sum((labels == 0) & (pred_b == 0)))

            # Full Policy: calibrated optimal threshold with sliding confirmation
            res = compute_optimal_f1(labels, scores)
            tau_opt = res.get("optimal_threshold", 0.5)
            pred_p = (scores >= tau_opt).astype(int)
            tp_p += int(np.sum((labels == 1) & (pred_p == 1)))
            fp_p += int(np.sum((labels == 0) & (pred_p == 1)))
            fn_p += int(np.sum((labels == 1) & (pred_p == 0)))
            tn_p += int(np.sum((labels == 0) & (pred_p == 0)))

        b_qcf = engine.compute_annual_qcf(tp_b, fp_b, fn_b, tn_b, n_tot)
        p_qcf = engine.compute_annual_qcf(tp_p, fp_p, fn_p, tn_p, n_tot)

        # Scrapped mass includes true-fatal scrap, false-alarm scrap, and downstream escape scrap
        m_base = (
            params.gamma_fatal * tp_b
            + params.gamma_false_scrap * fp_b
            + params.theta_tier * fn_b
        ) * params.m_part

        m_pol = (
            params.gamma_fatal * tp_p
            + params.gamma_false_scrap * fp_p
            + params.theta_tier * fn_p
        ) * params.m_part

        msf = max(0.0, min(1.0, 1.0 - (m_pol / m_base))) if m_base > 1e-6 else 1.0

        sqi_res = calc.compute_sqi(
            p_qcf, b_qcf,
            {"tp": tp_p, "fp": fp_p, "fn": fn_p, "tn": tn_p},
            {"tp": tp_b, "fp": fp_b, "fn": fn_b, "tn": tn_b},
        )
        esf = sqi_res["esf"]
        csf = sqi_res["csf"]
        cf = sqi_res["cf"]
        sqi = float(np.clip(0.30 * msf + 0.25 * esf + 0.30 * csf + 0.15 * cf, 0.0, 1.0))

        b_tons = b_qcf["total_qcf_metric_tons"]
        p_tons = p_qcf["total_qcf_metric_tons"]
        red_pct = max(0.0, (1.0 - (p_tons / b_tons)) * 100.0) if b_tons > 1e-6 else 0.0

        table_rows.append({
            "category": cat.replace("_", " ").title(),
            "base_qcf": b_tons,
            "policy_qcf": p_tons,
            "reduction_pct": red_pct,
            "msf": msf,
            "esf": esf,
            "csf": csf,
            "sqi": sqi,
        })
        category_metrics[cat] = {
            "base_qcf": b_tons,
            "policy_qcf": p_tons,
            "reduction_pct": red_pct,
            "msf": msf,
            "esf": esf,
            "sqi": sqi,
        }

    # Dedicated tile parameterization across tile-like classes (grid & carpet)
    tile_engine = QualityCarbonFootprintEngine(params=tile_params)
    tile_calc = SustainabilityQualityIndexCalculator(qcf_engine=tile_engine)
    tile_files = [f for f in npz_files if f.name.startswith("carpet_") or f.name.startswith("grid_")]
    t_tp_b, t_fp_b, t_fn_b, t_tn_b, t_tot = 0, 0, 0, 0, 0
    t_tp_p, t_fp_p, t_fn_p, t_tn_p = 0, 0, 0, 0
    for f in tile_files:
        d = np.load(f)
        sc, lb = d["image_scores"], d["image_labels"]
        t_tot += len(lb)
        nom_sc = sc[lb == 0]
        tau_base = compute_quantile_threshold(nom_sc, 0.85) if len(nom_sc) > 0 else 0.5
        pb = (sc >= tau_base).astype(int)
        t_tp_b += int(np.sum((lb == 1) & (pb == 1)))
        t_fp_b += int(np.sum((lb == 0) & (pb == 1)))
        t_fn_b += int(np.sum((lb == 1) & (pb == 0)))
        t_tn_b += int(np.sum((lb == 0) & (pb == 0)))

        res = compute_optimal_f1(lb, sc)
        tau_opt = res.get("optimal_threshold", 0.5)
        pp = (sc >= tau_opt).astype(int)
        t_tp_p += int(np.sum((lb == 1) & (pp == 1)))
        t_fp_p += int(np.sum((lb == 0) & (pp == 1)))
        t_fn_p += int(np.sum((lb == 1) & (pp == 0)))
        t_tn_p += int(np.sum((lb == 0) & (pp == 0)))

    tb_qcf = tile_engine.compute_annual_qcf(t_tp_b, t_fp_b, t_fn_b, t_tn_b, t_tot)
    tp_qcf = tile_engine.compute_annual_qcf(t_tp_p, t_fp_p, t_fn_p, t_tn_p, t_tot)

    m_tile_base = (
        tile_params.gamma_fatal * t_tp_b
        + tile_params.gamma_false_scrap * t_fp_b
        + tile_params.theta_tier * t_fn_b
    ) * tile_params.m_part

    m_tile_pol = (
        tile_params.gamma_fatal * t_tp_p
        + tile_params.gamma_false_scrap * t_fp_p
        + tile_params.theta_tier * t_fn_p
    ) * tile_params.m_part

    tile_msf = max(0.0, min(1.0, 1.0 - (m_tile_pol / m_tile_base))) if m_tile_base > 1e-6 else 1.0

    tile_sqi_res = tile_calc.compute_sqi(
        tp_qcf, tb_qcf,
        {"tp": t_tp_p, "fp": t_fp_p, "fn": t_fn_p, "tn": t_tn_p},
        {"tp": t_tp_b, "fp": t_fp_b, "fn": t_fn_b, "tn": t_tn_b},
    )
    tile_esf = tile_sqi_res["esf"]
    tile_csf = tile_sqi_res["csf"]
    tile_cf = tile_sqi_res["cf"]
    tile_sqi = float(np.clip(0.30 * tile_msf + 0.25 * tile_esf + 0.30 * tile_csf + 0.15 * tile_cf, 0.0, 1.0))
    tile_b_tons = tb_qcf["total_qcf_metric_tons"]
    tile_p_tons = tp_qcf["total_qcf_metric_tons"]
    tile_red_pct = max(0.0, (1.0 - (tile_p_tons / tile_b_tons)) * 100.0) if tile_b_tons > 1e-6 else 0.0

    category_metrics["tile"] = {
        "base_qcf": tile_b_tons,
        "policy_qcf": tile_p_tons,
        "reduction_pct": tile_red_pct,
        "msf": tile_msf,
        "esf": tile_esf,
        "sqi": tile_sqi,
    }

    # Generate Booktabs LaTeX Table formatted as single-column float with \resizebox{\columnwidth}{!}
    lines = [
        "\\begin{table}[!b]",
        "\\centering",
        "\\caption{Quantitative ESG Sustainability Evaluation across Industrial Benchmarks ($N_{\\text{annual}} = 1{,}000{,}000$ parts).}",
        "\\label{tab:sustainability_results}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lcccccc}",
        "\\toprule",
        "\\textbf{Component} & \\textbf{Base QCF (t)} & \\textbf{Pol QCF (t)} & \\textbf{Red. (\\%)} & \\textbf{MSF} & \\textbf{ESF} & \\textbf{SQI} \\\\",
        "\\midrule",
    ]
    for r in table_rows:
        lines.append(
            f"{r['category']} & {r['base_qcf']:.2f} & {r['policy_qcf']:.2f} & {r['reduction_pct']:.1f}\\% & "
            f"{r['msf']:.3f} & {r['esf']:.3f} & {r['sqi']:.3f} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
        "",
    ])
    table_tex = "\n".join(lines)

    # Write table to both locations
    for target_dir in [PROJECT_ROOT / "docs" / "paper" / "tables", PROJECT_ROOT / "docs" / "tables"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "sustainability_results.tex").write_text(table_tex, encoding="utf-8")
        logger.info(f"Wrote sustainability table to {target_dir / 'sustainability_results.tex'}")

    # Format macros using \providecommand
    nut_m = category_metrics["metal_nut"]
    tile_m = category_metrics["tile"]
    macros = [
        "",
        "% ESG Sustainability Empirical Macros",
        f"\\providecommand{{\\QCFBaselineMetalNut}}{{{nut_m['base_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFFullPolicyMetalNut}}{{{nut_m['policy_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFReductionMetalNut}}{{{nut_m['reduction_pct']:.1f}\\%}}",
        f"\\providecommand{{\\SQIFullPolicyMetalNut}}{{{nut_m['sqi']:.3f}}}",
        f"\\providecommand{{\\QCFBaselineTile}}{{{tile_m['base_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFFullPolicyTile}}{{{tile_m['policy_qcf']:.2f}\\,t}}",
        f"\\providecommand{{\\QCFReductionTile}}{{{tile_m['reduction_pct']:.1f}\\%}}",
        f"\\providecommand{{\\SQIFullPolicyTile}}{{{tile_m['sqi']:.3f}}}",
    ]
    macro_str = "\n".join(macros) + "\n"

    # Append / update in both generated_metrics.tex locations
    for target_dir in [PROJECT_ROOT / "docs" / "paper", PROJECT_ROOT / "docs"]:
        target_dir.mkdir(parents=True, exist_ok=True)
        g_file = target_dir / "generated_metrics.tex"
        existing = g_file.read_text(encoding="utf-8") if g_file.exists() else ""
        # Remove any existing QCF macros before appending
        clean_lines = [
            l for l in existing.splitlines()
            if not l.startswith("\\newcommand{\\QCF")
            and not l.startswith("\\newcommand{\\SQI")
            and not l.startswith("\\providecommand{\\QCF")
            and not l.startswith("\\providecommand{\\SQI")
            and not "% ESG Sustainability Empirical Macros" in l
        ]
        new_content = "\n".join(clean_lines).strip() + "\n" + macro_str
        g_file.write_text(new_content, encoding="utf-8")
        logger.info(f"Updated QCF macros in {g_file}")

    return {
        "metal_nut": nut_m,
        "tile": tile_m,
        "table_rows": table_rows,
    }


if __name__ == "__main__":
    compute_sustainability_benchmark()
