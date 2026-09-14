import torch
import torch.nn as nn
import pytest

from src.benchmarking.profiler import CUDAPerformanceProfiler


def test_cuda_performance_profiler_cpu():
    """
    Tests CUDAPerformanceProfiler on CPU with a dummy Conv2d + ReLU model.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU()
    )
    model.eval()

    sample_input = torch.randn(1, 3, 64, 64)
    profiler = CUDAPerformanceProfiler(warmup_runs=5, active_runs=20)
    results = profiler.profile(model, sample_input)

    expected_keys = [
        "p50_ms", "p90_ms", "p95_ms", "p99_ms",
        "mean_ms", "std_ms", "fps", "peak_vram_mb"
    ]
    for k in expected_keys:
        assert k in results, f"Missing key {k} in profiler results"

    assert results["p50_ms"] > 0.0
    assert results["mean_ms"] > 0.0
    assert results["fps"] > 0.0
    assert results["p99_ms"] >= results["p50_ms"]
    assert results["peak_vram_mb"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_performance_profiler_cuda():
    """
    Tests CUDAPerformanceProfiler on CUDA if a GPU is available.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU()
    ).cuda()
    model.eval()

    sample_input = torch.randn(1, 3, 64, 64, device="cuda")
    profiler = CUDAPerformanceProfiler(warmup_runs=5, active_runs=20)
    results = profiler.profile(model, sample_input)

    assert results["p50_ms"] > 0.0
    assert results["fps"] > 0.0
    assert results["peak_vram_mb"] >= 0.0
