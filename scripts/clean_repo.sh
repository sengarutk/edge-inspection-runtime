#!/usr/bin/env bash
# ==============================================================================
# Repository Artifact Cleanup Utility
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "[INFO] Cleaning temporary databases, evidence artifacts, and test caches..."

# Remove temporary SQLite database files
rm -f data/*.db data/*.db-journal data/*.db-wal data/*.db-shm

# Remove temporary verification evidence PNGs (keep .gitkeep)
find data/evidence/ -type f ! -name '.gitkeep' -delete 2>/dev/null || true
find data/logs/ -type f ! -name '.gitkeep' -delete 2>/dev/null || true

# Remove Python and pytest caches
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
rm -f .coverage coverage.xml

echo "[INFO] Repository clean complete."