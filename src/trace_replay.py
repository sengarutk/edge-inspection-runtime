"""Offline Physical Sensor Trace Replay Adapter and Generator."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from loguru import logger
import numpy as np
import pandas as pd

from src.sensor_simulator import MachineState, SensorReading


class RealSensorTraceReplay:
    """Replays historical industrial multi-modal sensor traces with calibration."""

    def __init__(
        self,
        trace_path: str | Path,
        machine_id: str = "press_unit_04",
        calibration_window_steps: int = 50,
        z_threshold: float = 3.0,
    ) -> None:
        """Initialize trace replay loader and calibrate nominal baseline envelopes.

        Args:
            trace_path: Path to CSV or Parquet file containing telemetry traces.
            machine_id: Target machine identifier string.
            calibration_window_steps: Number of initial steps used for baseline calibration.
            z_threshold: Z-score cutoff for normalizing anomaly scores.
        """
        self.trace_path = Path(trace_path)
        self.machine_id = machine_id
        self.calib_steps = calibration_window_steps
        self.z_threshold = z_threshold
        self._current_index = 0

        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {self.trace_path}")

        if self.trace_path.suffix.lower() == ".parquet":
            self.df = pd.read_parquet(self.trace_path)
        else:
            self.df = pd.read_csv(self.trace_path)

        required_cols = {"vibration_rms", "temperature_c", "current_amps"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Trace data missing required columns: {missing}")

        self._calibrate()
        logger.info(
            f"Initialized RealSensorTraceReplay (rows={len(self.df)}, calib_steps={self.calib_steps})"
        )

    def _calibrate(self) -> None:
        """Derive channel means and standard deviations from calibration window."""
        calib_df = self.df.iloc[: max(self.calib_steps, 5)]
        self.means = {
            "vibration_rms": float(calib_df["vibration_rms"].mean()),
            "temperature_c": float(calib_df["temperature_c"].mean()),
            "current_amps": float(calib_df["current_amps"].mean()),
        }
        self.stds = {
            "vibration_rms": max(float(calib_df["vibration_rms"].std()), 1e-6),
            "temperature_c": max(float(calib_df["temperature_c"].std()), 1e-6),
            "current_amps": max(float(calib_df["current_amps"].std()), 1e-6),
        }

    def compute_sensor_score(self, vib: float, temp: float, curr: float) -> float:
        """Compute composite normalized sensor anomaly score via z-score deviations."""
        z_vib = max(0.0, (vib - self.means["vibration_rms"]) / self.stds["vibration_rms"])
        z_temp = max(0.0, (temp - self.means["temperature_c"]) / self.stds["temperature_c"])
        z_curr = max(0.0, (curr - self.means["current_amps"]) / self.stds["current_amps"])

        mean_z = (z_vib + z_temp + z_curr) / 3.0
        normalized = min(1.0, mean_z / self.z_threshold)
        return float(np.clip(normalized, 0.0, 1.0))

    def step(self) -> SensorReading:
        """Yield the next sequential SensorReading from the trace."""
        if self._current_index >= len(self.df):
            self._current_index = 0  # Loop stream

        row = self.df.iloc[self._current_index]
        self._current_index += 1

        vib = float(row["vibration_rms"])
        temp = float(row["temperature_c"])
        curr = float(row["current_amps"])

        # Check for NaN / dropout in trace
        missing_channels: List[str] = []
        if np.isnan(vib):
            missing_channels.append("vibration_rms")
            vib = self.means["vibration_rms"]
        if np.isnan(temp):
            missing_channels.append("temperature_c")
            temp = self.means["temperature_c"]
        if np.isnan(curr):
            missing_channels.append("current_amps")
            curr = self.means["current_amps"]

        sensor_score = self.compute_sensor_score(vib, temp, curr)
        machine_state_str = str(row.get("machine_state", "RUNNING"))
        try:
            m_state = MachineState(machine_state_str)
        except ValueError:
            m_state = MachineState.RUNNING

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        return SensorReading(
            reading_id=f"replay_{self._current_index:06d}",
            timestamp_utc=now_utc,
            machine_id=self.machine_id,
            machine_state=m_state,
            vibration_rms=round(vib, 4),
            temperature_c=round(temp, 2),
            current_amps=round(curr, 2),
            missing_channels=missing_channels,
            is_degraded=len(missing_channels) > 0,
            sensor_score=round(sensor_score, 4),
        )

    def __iter__(self) -> Iterator[SensorReading]:
        """Iterate over all rows in the trace as SensorReading objects."""
        self.reset()
        while self._current_index < len(self.df):
            yield self.step()

    def reset(self) -> None:
        """Reset replay pointer to beginning of trace."""
        self._current_index = 0

    def total_steps(self) -> int:
        """Return total number of rows in trace."""
        return len(self.df)


def generate_sample_physical_trace(
    csv_path: str | Path = "data/traces/sample_industrial_trace.csv",
    n_steps: int = 500,
    seed: int = 42,
) -> Path:
    """Generate a realistic synthetic multi-channel industrial physical trace."""
    out_file = Path(csv_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    time_series = []

    p1 = int(n_steps * 0.3)
    p2 = int(n_steps * 0.6)

    # 1. Warmup / Nominal Phase
    for i in range(p1):
        vib = 0.40 + rng.normal(0.0, 0.03)
        temp = 55.0 + rng.normal(0.0, 0.5)
        curr = 12.0 + rng.normal(0.0, 0.4)
        time_series.append({
            "step": i,
            "vibration_rms": max(0.01, vib),
            "temperature_c": temp,
            "current_amps": curr,
            "machine_state": "RUNNING",
        })

    # 2. Bearing Wear / Friction Anomaly Phase
    for i in range(p1, p2):
        progress = (i - p1) / max(p2 - p1, 1)
        vib = 0.40 + (0.55 * progress) + rng.normal(0.0, 0.05)
        temp = 55.0 + (18.0 * progress) + rng.normal(0.0, 0.8)
        curr = 12.0 + (6.0 * progress) + rng.normal(0.0, 0.6)
        time_series.append({
            "step": i,
            "vibration_rms": vib,
            "temperature_c": temp,
            "current_amps": curr,
            "machine_state": "RUNNING",
        })

    # 3. Intermittent Sensor Dropout & Recovery Phase
    dropout_start = p2 + int((n_steps - p2) * 0.25)
    dropout_end = p2 + int((n_steps - p2) * 0.45)
    maint_start = p2 + int((n_steps - p2) * 0.75)

    for i in range(p2, n_steps):
        is_dropout = dropout_start <= i <= dropout_end
        vib = np.nan if is_dropout else (0.42 + rng.normal(0.0, 0.04))
        temp = 58.0 + rng.normal(0.0, 0.6)
        curr = 12.5 + rng.normal(0.0, 0.4)
        m_state = "MAINTENANCE" if i >= maint_start else "RUNNING"
        time_series.append({
            "step": i,
            "vibration_rms": vib,
            "temperature_c": temp,
            "current_amps": curr,
            "machine_state": m_state,
        })

    df = pd.DataFrame(time_series)
    df.to_csv(out_file, index=False)
    logger.info(f"Generated sample physical sensor trace ({len(df)} rows) -> {out_file}")
    return out_file
