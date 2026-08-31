"""Metrics, statistical inference, and queueing models package."""

from __future__ import annotations

from src.metrics.stats import bootstrap_ci
from src.metrics.significance import paired_significance_test, apply_holm_bonferroni_correction
from src.metrics.queue_model import OperatorQueueModel
from src.metrics.evaluator import (
    BenchmarkEvaluator,
    aggregate_ablation_results,
    generate_ablation_latex_table,
    generate_ablation_markdown_table,
)

__all__ = [
    "bootstrap_ci",
    "paired_significance_test",
    "apply_holm_bonferroni_correction",
    "OperatorQueueModel",
    "BenchmarkEvaluator",
    "aggregate_ablation_results",
    "generate_ablation_latex_table",
    "generate_ablation_markdown_table",
]
