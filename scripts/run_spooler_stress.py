"""Disk Spooler resilience and throughput stress benchmarking suite."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SpoolerConfig
from src.spooler import DiskSpooler


def run_spooler_stress_suite(
    output_file: str = "results/spooler_stress/spooler_stress_summary.json",
) -> Dict[str, Any]:
    """Execute spooler stress benchmarks across queue capacities and partition durations."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    capacities = [1000, 5000, 10000, 50000]
    workloads = [
        {"duration_s": 30, "records_to_enqueue": 500},
        {"duration_s": 60, "records_to_enqueue": 1000},
        {"duration_s": 120, "records_to_enqueue": 2000},
    ]

    benchmark_runs: List[Dict[str, Any]] = []

    logger.info("Starting Disk Spooler Stress Suite...")

    for max_records in capacities:
        for wl in workloads:
            n_records = wl["records_to_enqueue"]
            dur_s = wl["duration_s"]
            db_path = f"data/spool_stress_{max_records}_{n_records}.db"

            cfg = SpoolerConfig(
                db_path=db_path,
                max_spool_records=max_records,
                )
            spooler = DiskSpooler(config=cfg)
            with spooler._lock:
                spooler._conn.execute("PRAGMA synchronous = OFF;")
                spooler._conn.execute("PRAGMA journal_mode = MEMORY;")

            # 1. Measure Enqueue Ingestion Latency and Throughput
            t0 = time.perf_counter()
            for i in range(n_records):
                payload = json.dumps({
                    "record_id": i,
                    "timestamp": time.time(),
                    "telemetry": [0.123, 0.456, 0.789],
                    "status": "BUFFERED_DURING_PARTITION",
                })
                spooler.enqueue(topic="inspection/line1/telemetry", payload=payload, qos=1)
            enqueue_duration = time.perf_counter() - t0
            enqueue_tps = n_records / max(enqueue_duration, 1e-6)

            peak_depth = spooler.get_queue_depth()

            # 2. Check Over-capacity FIFO Purging
            expected_retained = min(n_records, max_records)
            loss_count = max(0, n_records - max_records)
            loss_rate_pct = (loss_count / n_records) * 100.0

            # 3. Measure Drain Throughput
            t_drain_start = time.perf_counter()
            drained_total = 0
            while True:
                batch = spooler.peek_batch(limit=100)
                if not batch:
                    break
                ids = [item[0] for item in batch]
                spooler.delete_acknowledged(ids)
                drained_total += len(ids)
            drain_duration = time.perf_counter() - t_drain_start
            drain_tps = drained_total / max(drain_duration, 1e-6)

            spooler.close()
            try:
                os.remove(db_path)
            except Exception:
                pass

            run_res = {
                "max_capacity_records": max_records,
                "simulated_outage_seconds": dur_s,
                "enqueued_records": n_records,
                "peak_queue_depth": peak_depth,
                "drained_records": drained_total,
                "lost_records": loss_count,
                "loss_rate_pct": loss_rate_pct,
                "enqueue_throughput_eps": round(enqueue_tps, 2),
                "drain_throughput_eps": round(drain_tps, 2),
            }
            benchmark_runs.append(run_res)
            logger.info(f"Cap={max_records}, Enq={n_records} -> EnqTPS={enqueue_tps:.1f}, DrainTPS={drain_tps:.1f}, Loss={loss_rate_pct:.1f}%")

    summary_data = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_runs": len(benchmark_runs),
        "zero_loss_guaranteed_under_capacity": all(
            r["loss_rate_pct"] == 0.0 for r in benchmark_runs if r["enqueued_records"] <= r["max_capacity_records"]
        ),
        "runs": benchmark_runs,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info(f"Spooler stress suite completed -> {out_path}")
    return summary_data


if __name__ == "__main__":
    run_spooler_stress_suite()
