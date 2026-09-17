#!/usr/bin/env bash
# ==============================================================================
# Master Execution & Benchmark Pipeline Runner
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "========================================================================"
echo "  Flagship 4: Industrial Edge Inspection Runtime & Reliability System   "
echo "========================================================================"

# Activate virtual environment if present
if [ -d ".venv" ]; then
    echo "[INFO] Activating virtual environment (.venv)..."
    source .venv/bin/activate
fi

# Export PYTHONPATH and install package in editable mode
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
pip install -e . --no-deps --quiet 2>/dev/null || true

# Ensure storage directories exist
mkdir -p data/evidence data/logs configs

echo ""
echo "[STEP 1/3] Executing Full Unit & Integration Test Suite with Coverage..."
python -m pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=90

echo ""
echo "[STEP 2/3] Executing Automated 300-Step Reliability Benchmark Suite..."
python scripts/benchmark_suite.py

echo ""
echo "[STEP 3/3] Verification Complete! Ready to Launch Operator Console."
echo "------------------------------------------------------------------------"
echo "To launch the Streamlit Operator Review Console:"
echo "  $ streamlit run src/dashboard/app.py --server.port 8501"
echo "  (If port 8501 is busy, specify an alternate port: --server.port 8502)"
echo "========================================================================"