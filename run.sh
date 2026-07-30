#!/usr/bin/env bash
# Run one RSMeM experiment from the repository root.
#
# Usage:
#   bash run.sh
#   bash run.sh configs/qwen3_8b_baseline.yaml
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CONFIG_PATH="${1:-configs/deepseek_v32_rsmem.yaml}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON:-python}"
fi

exec "$PYTHON_BIN" -m rsmem run --config "$CONFIG_PATH"

