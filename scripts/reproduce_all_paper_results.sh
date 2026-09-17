#!/usr/bin/env bash
# ==============================================================================
# reproduce_all_paper_results.sh
# End-to-end scientific and engineering reproducibility pipeline for Flagship 4:
# "Reducing Alert Fatigue in Industrial Edge Inspection Through Multi-Modal
#  Temporal Policy Gating and Durable Event Spooling"
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"

echo "======================================================================"
echo "[REPRODUCE] Starting End-to-End Publication Hardening Reproduction"
echo "  Project Root: ${PROJECT_ROOT}"
echo "  Python Interpreter: ${PYTHON}"
echo "======================================================================"

cd "${PROJECT_ROOT}"

# 1. Evaluate Multimodal Truth Table Diagnostic Suite (Phase 3)
echo ""
echo "--- [1/6] Running Multimodal Truth Table Diagnostic Suite ---"
"${PYTHON}" scripts/evaluate_multimodal_truth_table.py

# 2. Benchmark Pipeline Latency Decomposition (Phase 4)
echo ""
echo "--- [2/6] Running Latency Profiling (N=5000 cycles) ---"
"${PYTHON}" scripts/benchmark_latency.py

# 3. Benchmark Spooler Resilience Under Broker Partition (Phase 2)
echo ""
echo "--- [3/6] Running Idempotent Spooler Resilience Stress Benchmark ---"
"${PYTHON}" scripts/benchmark_spooler_resilience.py

# 4. Run Real Trace Replay Protocol (Phase 1 & 6)
echo ""
echo "--- [4/6] Running Offline Trace Replay Protocol (IMS & C-MAPSS) ---"
"${PYTHON}" scripts/run_real_trace_benchmark.py

# 5. Validate Paper Claims & Regenerate Single Source of Truth LaTeX Macros (Phase 5)
echo ""
echo "--- [5/6] Validating Paper Claims & Generating LaTeX Macros ---"
"${PYTHON}" scripts/validate_paper_claims.py

# 6. Build Camera-Ready 4-Page Manuscript via pdflatex + bibtex (Phase 7)
echo ""
echo "--- [6/6] Building Camera-Ready Manuscript with pdflatex & bibtex ---"
cd "${PROJECT_ROOT}/docs/paper"
rm -f *.aux *.bbl *.blg *.log *.out *.pdf

pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

# Assert exit code and exact 4-page ceiling
if grep -q "Output written on main.pdf (4 pages" main.log; then
    echo "  [SUCCESS] Manuscript compiled successfully: EXACTLY 4 PAGES."
else
    echo "  [ERROR] Manuscript compilation did not yield exactly 4 pages!"
    grep "Output written on main.pdf" main.log || true
    exit 1
fi

# Verify no undefined references
if grep -i "undefined" main.log | grep -v "0 undefined"; then
    echo "  [ERROR] Found undefined citations or references in main.log!"
    grep -i "undefined" main.log
    exit 1
fi

echo ""
echo "======================================================================"
echo "[SUCCESS] All experimental benchmarks, validations, and PDF builds passed!"
echo "  Artifact: ${PROJECT_ROOT}/docs/paper/main.pdf"
echo "======================================================================"
