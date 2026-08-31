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


def test_nasa_trace_generators_and_benchmark(tmp_path: Path) -> None:
    """Verify NASA IMS bearing and C-MAPSS turbofan trace generation and evaluation."""
    from src.trace_replay import generate_cmapss_turbofan_trace, generate_ims_bearing_trace
    from scripts.run_real_trace_benchmark import evaluate_trace_on_policy, run_real_trace_benchmark_suite
    from src.config import PolicyMode

    ims_path = generate_ims_bearing_trace(tmp_path / "ims_trace.csv", n_steps=100, seed=42)
    cmapss_path = generate_cmapss_turbofan_trace(tmp_path / "cmapss_trace.csv", n_steps=100, seed=42)

    assert ims_path.exists() and ims_path.stat().st_size > 0
    assert cmapss_path.exists() and cmapss_path.stat().st_size > 0

    # Evaluate single trace under BASELINE and FULL_POLICY
    res_base = evaluate_trace_on_policy(ims_path, PolicyMode.BASELINE, defect_start_step=70, total_steps=100)
    res_full = evaluate_trace_on_policy(ims_path, PolicyMode.FULL_POLICY, defect_start_step=70, total_steps=100)

    assert "false_alarms_per_hour" in res_base
    assert "true_positive_rate" in res_full
    assert res_full["queue_utilization"] >= 0.0

    # Test full suite execution to tmp output
    out_summary = tmp_path / "real_trace_summary.json"
    summary = run_real_trace_benchmark_suite(output_json=str(out_summary))
    assert "nasa_ims_bearing" in summary
    assert "nasa_cmapss_turbofan" in summary
    assert out_summary.exists()


def test_mixed_corruption_benchmark(tmp_path: Path) -> None:
    """Verify mixed-corruption benchmark pipeline executes cleanly."""
    from scripts.run_mixed_corruption_benchmark import run_single_scenario_benchmark, run_mixed_corruption_benchmark_suite
    from src.config import PolicyMode

    res = run_single_scenario_benchmark(
        scenario_name="nominal",
        policy_mode=PolicyMode.FULL_POLICY,
        n_cycles=50,
        seed=42,
    )
    assert res["total_cycles"] == 50
    assert res["mean_latency_ms"] > 0.0
    assert res["deadline_miss_rate"] == 0.0

    out_json = tmp_path / "mixed_summary.json"
    summary = run_mixed_corruption_benchmark_suite(output_json=str(out_json))
    assert "aggregate_suppression_ratio" in summary
    assert "nominal" in summary["scenarios"]
