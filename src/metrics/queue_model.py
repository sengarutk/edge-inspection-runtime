"""Queueing-theoretic operator cognitive load and triage backlog modeling."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np


class OperatorQueueModel:
    """Analytical and Monte Carlo queueing models (M/M/1, M/D/1, Log-Normal) for triage queues."""

    def __init__(self, service_rate_per_hour: float = 60.0) -> None:
        """Initialize queue model with operator service capacity (default: 60 reviews/hr)."""
        self.mu = float(service_rate_per_hour)

    def analyze_mm1(self, arrival_rate_per_hour: float) -> Dict[str, float]:
        """Compute analytical M/M/1 queue metrics."""
        lam = float(arrival_rate_per_hour)
        if lam <= 0.0:
            return {"utilization": 0.0, "mean_queue_length": 0.0, "mean_wait_time_minutes": 0.0, "p_blowup": 0.0}

        rho = lam / self.mu
        if rho >= 1.0:
            return {"utilization": rho, "mean_queue_length": float("inf"), "mean_wait_time_minutes": float("inf"), "p_blowup": 1.0}

        l_q = (rho ** 2) / (1.0 - rho)
        w_q_hours = rho / (self.mu * (1.0 - rho))
        w_q_min = w_q_hours * 60.0
        p_over_10 = rho ** 10  # Probability queue exceeds 10 items

        return {
            "utilization": round(rho, 4),
            "mean_queue_length": round(l_q, 3),
            "mean_wait_time_minutes": round(w_q_min, 3),
            "p_blowup": round(p_over_10, 4),
        }

    def analyze_md1(self, arrival_rate_per_hour: float) -> Dict[str, float]:
        """Compute analytical M/D/1 queue metrics (deterministic service times)."""
        lam = float(arrival_rate_per_hour)
        if lam <= 0.0:
            return {"utilization": 0.0, "mean_queue_length": 0.0, "mean_wait_time_minutes": 0.0, "p_blowup": 0.0}

        rho = lam / self.mu
        if rho >= 1.0:
            return {"utilization": rho, "mean_queue_length": float("inf"), "mean_wait_time_minutes": float("inf"), "p_blowup": 1.0}

        l_q = (rho ** 2) / (2.0 * (1.0 - rho))
        w_q_hours = rho / (2.0 * self.mu * (1.0 - rho))
        w_q_min = w_q_hours * 60.0

        return {
            "utilization": round(rho, 4),
            "mean_queue_length": round(l_q, 3),
            "mean_wait_time_minutes": round(w_q_min, 3),
            "p_blowup": round(rho ** 10, 4),
        }

    def simulate_variable_service(
        self,
        arrival_rate_per_hour: float,
        duration_hours: float = 8.0,
        service_sigma: float = 0.35,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Event-driven simulation of operator triage queue with log-normal review times."""
        lam = float(arrival_rate_per_hour)
        if lam <= 0.0:
            return {"mean_queue_length": 0.0, "max_queue_length": 0.0, "mean_wait_minutes": 0.0, "p95_wait_minutes": 0.0}

        rng = np.random.RandomState(seed)
        n_arrivals = rng.poisson(lam * duration_hours)
        if n_arrivals == 0:
            return {"mean_queue_length": 0.0, "max_queue_length": 0.0, "mean_wait_minutes": 0.0, "p95_wait_minutes": 0.0}

        arrival_times = np.sort(rng.uniform(0.0, duration_hours * 60.0, size=n_arrivals))
        mean_service_min = (60.0 / self.mu)
        mu_log = np.log(mean_service_min) - (0.5 * service_sigma ** 2)
        service_durations = rng.lognormal(mean=mu_log, sigma=service_sigma, size=n_arrivals)

        current_time = 0.0
        wait_times_min = []

        for arr_t, s_dur in zip(arrival_times, service_durations):
            if arr_t > current_time:
                wait_t = 0.0
                current_time = arr_t + s_dur
            else:
                wait_t = current_time - arr_t
                current_time += s_dur
            wait_times_min.append(wait_t)

        return {
            "mean_queue_length": round(float(np.mean(wait_times_min) / mean_service_min), 2),
            "max_queue_length": round(float(np.max(wait_times_min) / mean_service_min), 2),
            "mean_wait_minutes": round(float(np.mean(wait_times_min)), 2),
            "p95_wait_minutes": round(float(np.percentile(wait_times_min, 95)), 2),
        }

    def sweep_service_variability(
        self,
        arrival_rates: List[float],
        sigmas: List[float] = [0.2, 0.4, 0.6],
        duration_hours: float = 8.0,
        n_trials: int = 5,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Sweep arrival rates lambda across multiple log-normal service-time variance levels sigma.

        Returns a dictionary mapping sigma to arrays of L_q, W_q, and p95 wait times.
        """
        results: Dict[str, Any] = {}

        for sigma in sigmas:
            sigma_key = f"sigma_{sigma:.1f}"
            results[sigma_key] = {
                "arrival_rates": arrival_rates,
                "mean_queue_lengths": [],
                "mean_wait_times_min": [],
                "p95_wait_times_min": [],
            }

            for lam in arrival_rates:
                trial_lq: List[float] = []
                trial_wq: List[float] = []
                trial_p95: List[float] = []

                for trial_idx in range(n_trials):
                    sim = self.simulate_variable_service(
                        arrival_rate_per_hour=lam,
                        duration_hours=duration_hours,
                        service_sigma=sigma,
                        seed=seed + trial_idx * 100,
                    )
                    trial_lq.append(sim["mean_queue_length"])
                    trial_wq.append(sim["mean_wait_minutes"])
                    trial_p95.append(sim["p95_wait_minutes"])

                results[sigma_key]["mean_queue_lengths"].append(round(float(np.mean(trial_lq)), 3))
                results[sigma_key]["mean_wait_times_min"].append(round(float(np.mean(trial_wq)), 3))
                results[sigma_key]["p95_wait_times_min"].append(round(float(np.mean(trial_p95)), 3))

        return results
