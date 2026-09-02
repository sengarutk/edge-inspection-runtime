#!/usr/bin/env bash
# =============================================================================
# Package LaTeX Manuscript & Figures for Overleaf / Camera-Ready Archiving
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

echo "================================================================================"
echo " [OVERLEAF] Packaging Paper Assets for Overleaf Submission"
echo " Repo Root: ${REPO_ROOT}"
echo "================================================================================"

# 1. Regenerate Figures and Metrics
echo "[1/4] Regenerating metrics and publication vector figures..."
if [ -d ".venv" ]; then
    export PATH="${REPO_ROOT}/.venv/bin:${PATH}"
fi

python3 scripts/build_paper_assets.py
python3 scripts/generate_publication_figures.py

# 2. Stage Assets in Temporary Build Directory
STAGE_DIR="/tmp/overleaf_paper"
echo "[2/4] Staging paper files into ${STAGE_DIR}..."
rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/figures"

cp docs/paper.tex "${STAGE_DIR}/main.tex"
cp docs/generated_metrics.tex "${STAGE_DIR}/generated_metrics.tex"
cp docs/references.bib "${STAGE_DIR}/references.bib"
cp -r docs/figures/* "${STAGE_DIR}/figures/"

# 3. Fetch Official IEEEtran.bst if needed
echo "[3/4] Ensuring IEEEtran.bst bibliography style is bundled..."
if [ -f "docs/IEEEtran.bst" ]; then
    cp docs/IEEEtran.bst "${STAGE_DIR}/IEEEtran.bst"
elif [ ! -f "${STAGE_DIR}/IEEEtran.bst" ]; then
    curl -fsSL https://mirrors.ctan.org/macros/latex/contrib/IEEEtran/bibtex/IEEEtran.bst -o "${STAGE_DIR}/IEEEtran.bst" || true
fi

# 4. Create ZIP Archive
echo "[4/4] Creating paper_overleaf.zip archive..."
rm -f "${REPO_ROOT}/paper_overleaf.zip"
(cd "${STAGE_DIR}" && zip -r "${REPO_ROOT}/paper_overleaf.zip" .)

echo "================================================================================"
echo " [SUCCESS] Overleaf archive packaged: ${REPO_ROOT}/paper_overleaf.zip"
echo " Archive Contents:"
unzip -l "${REPO_ROOT}/paper_overleaf.zip"
echo "================================================================================"
