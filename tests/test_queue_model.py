"""Unit tests for queueing-theoretic operator triage models."""

import numpy as np
import pytest

from src.metrics.queue_model import OperatorQueueModel


def test_mm1_analytical_boundaries() -> None:
    """Verify M/M/1 analytical formulas at boundary conditions."""
    qm = OperatorQueueModel(service_rate_per_hour=60.0)

    # 1. Zero arrival rate
    m_zero = qm.analyze_mm1(0.0)
    assert m_zero["utilization"] == 0.0
    assert m_zero["mean_queue_length"] == 0.0
    assert m_zero["mean_wait_time_minutes"] == 0.0
    assert m_zero["p_blowup"] == 0.0

    # 2. Low utilization (12 reviews/hr -> rho = 0.20)
    m_low = qm.analyze_mm1(12.0)
    assert m_low["utilization"] == 0.20
    # L_q = 0.20^2 / (1 - 0.20) = 0.04 / 0.80 = 0.05
    assert np.isclose(m_low["mean_queue_length"], 0.05)
    # W_q = 0.20 / (60 * 0.80) hours = 0.20 / 48 hours = 0.25 minutes = 15 seconds
    assert np.isclose(m_low["mean_wait_time_minutes"], 0.25)
    # P(N >= 10) = 0.20^10 = 1.024e-7
    assert m_low["p_blowup"] < 1e-4

    # 3. Saturation / blowup boundary (rho >= 1.0)
    m_sat = qm.analyze_mm1(60.0)
    assert m_sat["utilization"] == 1.0
    assert m_sat["mean_queue_length"] == float("inf")
    assert m_sat["p_blowup"] == 1.0

    m_over = qm.analyze_mm1(75.0)
    assert m_over["utilization"] == 1.25
    assert m_over["mean_queue_length"] == float("inf")


def test_md1_vs_mm1_variance_reduction() -> None:
    """Verify Pollaczek-Khinchine formula: M/D/1 backlog is exactly half of M/M/1 backlog."""
    qm = OperatorQueueModel(service_rate_per_hour=60.0)
    lam = 30.0  # rho = 0.50

    mm1 = qm.analyze_mm1(lam)
    md1 = qm.analyze_md1(lam)

    assert mm1["utilization"] == md1["utilization"] == 0.50
    # M/D/1 mean queue length must be exactly 0.5 * M/M/1
    assert np.isclose(md1["mean_queue_length"], 0.5 * mm1["mean_queue_length"])
    assert np.isclose(md1["mean_wait_time_minutes"], 0.5 * mm1["mean_wait_time_minutes"])

    # Zero and saturation in MD1
    assert qm.analyze_md1(0.0)["utilization"] == 0.0
    assert qm.analyze_md1(65.0)["mean_queue_length"] == float("inf")


def test_variable_service_monte_carlo() -> None:
    """Verify log-normal event-driven Monte Carlo simulation of operator queue."""
    qm = OperatorQueueModel(service_rate_per_hour=60.0)

    # 1. Zero arrivals
    sim_zero = qm.simulate_variable_service(arrival_rate_per_hour=0.0)
    assert sim_zero["mean_queue_length"] == 0.0
    assert sim_zero["mean_wait_minutes"] == 0.0

    # 2. Moderate traffic (15 reviews/hr over 8 hour shift)
    sim = qm.simulate_variable_service(
        arrival_rate_per_hour=15.0,
        duration_hours=8.0,
        service_sigma=0.30,
        seed=2026,
    )

    assert sim["mean_wait_minutes"] >= 0.0
    assert sim["p95_wait_minutes"] >= sim["mean_wait_minutes"]
    assert sim["max_queue_length"] >= sim["mean_queue_length"]
