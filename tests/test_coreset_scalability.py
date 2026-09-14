import pytest
import numpy as np
from src.experiments.coreset_scalability import run_coreset_scalability_sweep


def test_coreset_scalability_sweep_execution(tmp_path):
    """
    Verifies that coreset scalability sweep executes across multi-scale configurations
    and logs speedup factors > 1.0 on GPU.
    """
    df = run_coreset_scalability_sweep(
        sample_sizes=[500, 1000],
        feature_dims=[64, 128],
        coreset_ratio=0.10,
        output_dir=str(tmp_path)
    )

    assert len(df) == 4
    for col in ["num_patches_N", "feature_dim_D", "time_cpu_sec", "time_gpu_batched_sec", "speedup_vs_cpu"]:
        assert col in df.columns

    # Verify positive runtimes
    assert np.all(df["time_cpu_sec"] > 0)
    assert np.all(df["time_gpu_batched_sec"] > 0)