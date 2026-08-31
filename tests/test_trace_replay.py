"""Unit tests for physical sensor trace replay and mixed corruption streams."""

from pathlib import Path
import numpy as np
import pytest

from src.sensor_simulator import MachineState, SensorReading
from src.stream_models import MixedCorruptionStream
from src.trace_replay import RealSensorTraceReplay, generate_sample_physical_trace


def test_trace_generation_and_replay_lifecycle(tmp_path: Path) -> None:
    """Verify trace generation, baseline calibration, and sequential reading playback."""
    csv_file = tmp_path / "test_trace.csv"
    generated_path = generate_sample_physical_trace(csv_path=csv_file, n_steps=200, seed=42)

    assert generated_path.exists()
    assert generated_path.stat().st_size > 0

    replay = RealSensorTraceReplay(
        trace_path=csv_file,
        calibration_window_steps=50,
        z_threshold=3.0,
    )

    assert replay.total_steps() == 200
    assert replay.means["vibration_rms"] > 0.0
    assert replay.stds["vibration_rms"] > 0.0

    # Read steps
    readings = [replay.step() for _ in range(100)]
    assert len(readings) == 100
    assert all(isinstance(r, SensorReading) for r in readings)
    assert all(r.machine_state == MachineState.RUNNING for r in readings[:50])

    # Test reset and generator iteration
    replay.reset()
    all_readings = list(replay)
    assert len(all_readings) == 200

    # Test error handling on missing file
    with pytest.raises(FileNotFoundError):
        RealSensorTraceReplay("non_existent_trace.csv")


def test_mixed_corruption_stream() -> None:
    """Verify stochastic mixed corruption stream applies noise, blur, and compression."""
    rng = np.random.RandomState(42)
    clean_frame = rng.randint(50, 200, (128, 128, 3), dtype=np.uint8)

    # 1. 0% corruption probability (clean pass-through)
    stream_zero = MixedCorruptionStream(p_corrupt=0.0)
    out_clean, applied_zero = stream_zero.corrupt_frame(clean_frame)
    assert np.array_equal(clean_frame, out_clean)
    assert applied_zero == []

    # 2. 100% corruption probability
    stream_full = MixedCorruptionStream(
        p_corrupt=1.0,
        noise_sigma=30.0,
        blur_kernel=9,
        jpeg_quality=20,
        seed=42,
    )
    corrupted, applied = stream_full.corrupt_frame(clean_frame, step=1)

    assert corrupted.shape == clean_frame.shape
    assert corrupted.dtype == np.uint8
    assert len(applied) > 0
    # Must differ from original due to noise / blur / compression
    assert not np.array_equal(clean_frame, corrupted)
