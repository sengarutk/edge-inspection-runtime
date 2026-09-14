#!/usr/bin/env bash
set -e

DATA_DIR=${1:-"data/mvtec_ad"}
CATEGORIES=${2:-"bottle cable hazelnut metal_nut"}

echo "=== MVTec AD Dataset Setup ==="
python scripts/download_dataset.py --data-root "$DATA_DIR" --categories $CATEGORIES
