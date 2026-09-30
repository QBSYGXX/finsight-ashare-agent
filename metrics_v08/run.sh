#!/usr/bin/env bash
set -euo pipefail
ADDON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$ADDON_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
export FINSIGHT_ROOT="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PROJECT_ROOT/src"
unset PYTHONHOME
PROJECT_PYTHON="$PROJECT_ROOT/.runtime/envs/finsight/bin/python"
if [[ ! -x "$PROJECT_PYTHON" ]]; then
    echo 'Existing portable Python environment is required.' >&2
    exit 2
fi
export CONDA_PREFIX="$PROJECT_ROOT/.runtime/envs/finsight"
export PATH="$(dirname -- "$PROJECT_PYTHON"):$PATH"
if [[ "${1:-}" == "chat" ]]; then shift; fi
RUN_DIR="$PROJECT_ROOT/runs/$(date +%Y%m%d-%H%M%S)-metrics-v085-$$"
mkdir -p "$RUN_DIR"
echo "LOG=$RUN_DIR/run.log"
set +e
"$PROJECT_PYTHON" -B -u "$ADDON_ROOT/chat.py" "$@" 2>&1 | tee "$RUN_DIR/run.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
echo "EXIT=$RUN_EXIT" | tee -a "$RUN_DIR/run.log"
exit "$RUN_EXIT"
