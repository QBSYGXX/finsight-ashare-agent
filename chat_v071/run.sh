#!/usr/bin/env bash
set -euo pipefail
ADDON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$ADDON_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
export FINSIGHT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/src"
unset PYTHONHOME
PROJECT_PYTHON="$PROJECT_ROOT/.runtime/envs/finsight/bin/python"
if [[ ! -x "$PROJECT_PYTHON" || ! -f "$PROJECT_ROOT/.runtime/envs/finsight/.finsight-ready.json" ]]; then
    echo 'Existing portable FinSight environment is required.' >&2
    exit 2
fi
export CONDA_PREFIX="$PROJECT_ROOT/.runtime/envs/finsight"
export PATH="$(dirname -- "$PROJECT_PYTHON"):$PATH"
if [[ "${1:-}" == "chat" ]]; then shift; fi
exec "$PROJECT_PYTHON" -B -u "$ADDON_ROOT/chat.py" "$@"
