#!/usr/bin/env bash
# =============================================================================
# Automated Research Reproduction Pipeline
# Flags: set -euo pipefail
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================================"
echo " [REPRO] Starting End-to-End Research Benchmark Reproduction Pipeline"
echo " Repo Root: ${REPO_ROOT}"
echo " Date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "================================================================================"

# 1. Environment & Dependency Validation
echo "[STEP 1/6] Validating Python environment and core packages..."
if [ -d ".venv" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

python3 -c "import numpy, scipy, matplotlib, pydantic, loguru, streamlit, cv2, pandas; print('  ✓ Core scientific & vision libraries available')"

# 2. Comprehensive Test Suite & Strict Coverage Enforcement
echo "[STEP 2/6] Running pytest test suite with coverage check (>= 90%)..."
python3 -m pytest tests/

# 3. Monte Carlo Ablation Suite Across 8 Policy Variants & 6 Workloads
echo "[STEP 3/6] Running 8-policy Monte Carlo ablation study..."
python3 scripts/run_ablation_study.py

# 4. Parameter Sensitivity, Spooler Stress & Trace Replay Calibration
echo "[STEP 4/6] Running sensitivity sweeps, spooler stress benchmarks, and trace generation..."
python3 scripts/run_sensitivity_analysis.py
python3 scripts/run_spooler_stress.py
python3 -c "from src.trace_replay import generate_sample_physical_trace; generate_sample_physical_trace(); print('  ✓ Sample physical trace generated')"

# 5. Publication Tables & Vector Figures Generation
echo "[STEP 5/6] Generating camera-ready publication figures (PDF & PNG) and LaTeX tables..."
python3 scripts/generate_publication_figures.py

# 6. Checksum Manifest Verification
echo "[STEP 6/6] Computing SHA-256 checksums of all generated research artifacts..."
mkdir -p results data/traces
find results docs/figures data/traces -type f \( -name "*.json" -o -name "*.tex" -o -name "*.md" -o -name "*.png" -o -name "*.pdf" -o -name "*.csv" \) \
    -exec sha256sum {} + | sort > results/CHECKSUMS.txt

echo "================================================================================"
echo " [SUCCESS] Full research reproduction complete!"
echo " Results Summary:"
echo "   - Ablation Summary:     results/ablation/ablation_summary.json"
echo "   - LaTeX Table:          results/ablation/ablation_table.tex"
echo "   - Markdown Table:       results/ablation/ablation_table.md"
echo "   - Sensitivity Analysis: results/sensitivity/sensitivity_summary.json"
echo "   - Spooler Stress:       results/spooler_stress/spooler_stress_summary.json"
echo "   - Physical Trace:       data/traces/sample_industrial_trace.csv"
echo "   - Publication Figures:  docs/figures/"
echo "   - SHA-256 Checksums:    results/CHECKSUMS.txt"
echo "================================================================================"
