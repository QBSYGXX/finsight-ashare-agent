#!/usr/bin/env bash
set -euo pipefail
ADDON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$ADDON_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
if [[ "${1:-doctor}" != doctor || "$#" -gt 1 ]]; then
    echo 'Usage: bash vllm_probe_v011/run.sh doctor'; exit 2
fi
PROJECT_PYTHON="$PROJECT_ROOT/.runtime/envs/finsight/bin/python"
[[ -x "$PROJECT_PYTHON" ]] || { echo 'Existing portable Python is required'; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
unset PYTHONHOME
RUN_DIR="$PROJECT_ROOT/runs/$(date +%Y%m%d-%H%M%S)-vllm-probe-v011-$$"
mkdir -p "$RUN_DIR"
echo "LOG=$RUN_DIR/run.log"
set +e
"$PROJECT_PYTHON" -B -u "$ADDON_ROOT/doctor.py" --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
exit "$RUN_EXIT"
