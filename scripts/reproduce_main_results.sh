#!/usr/bin/env bash
set -e

# Prioritize active Conda or vmunet environment
if [ -f "/home/sengar/miniconda3/envs/vmunet/bin/python" ]; then
    PYTHON_BIN="/home/sengar/miniconda3/envs/vmunet/bin/python"
elif [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
else
    PYTHON_BIN="python3"
fi

PYTEST_BIN="$PYTHON_BIN -m pytest"

echo "=== 1. Verifying Hardware and Environment Metadata ==="
$PYTHON_BIN -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo "=== 2. Running Comprehensive Pytest Suite with Coverage ==="
$PYTEST_BIN --cov=src --cov-report=term-missing tests/ -v

echo "=== 3. Executing Operational Stream Evaluations (IID, Burst, Drift) ==="
$PYTHON_BIN scripts/run_operational_eval.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad

echo "=== 4. Running Cost-Calibrated Thresholding (CCT) & Systems Scalability Ablations ==="
$PYTHON_BIN scripts/run_cct_experiments.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad
$PYTHON_BIN scripts/run_ablations.py --scores-dir results/mvtec_ad/scores --output-dir results/mvtec_ad

echo "=== 5. Compiling Publication LaTeX Tables and Figures ==="
$PYTHON_BIN scripts/generate_plots.py --tables-dir results/mvtec_ad/tables --output-dir results/mvtec_ad/figures
$PYTHON_BIN scripts/generate_operational_plots.py --scores-dir results/mvtec_ad/scores --tables-dir results/mvtec_ad/tables --output-dir results/mvtec_ad/figures/operational
$PYTHON_BIN scripts/generate_report.py --tables-dir results/mvtec_ad/tables --docs-dir docs

echo "=== ✅ Master Benchmark Reproduction Successfully Completed ==="