#!/usr/bin/env bash
set -euo pipefail
ADDON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$ADDON_ROOT/.." && pwd)"
cd "$PROJECT_ROOT"
PROJECT_PYTHON="$PROJECT_ROOT/.runtime/envs/finsight/bin/python"
[[ -x "$PROJECT_PYTHON" ]] || { echo 'Existing portable Python required'; exit 2; }
export PYTHONUNBUFFERED=1 PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONHOME
RUN_DIR="$PROJECT_ROOT/runs/$(date +%Y%m%d-%H%M%S)-native-probe-v012-$$"
mkdir -p "$RUN_DIR"
echo "LOG=$RUN_DIR/run.log"
set +e
"$PROJECT_PYTHON" -B -u "$ADDON_ROOT/probe.py" "$@" --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
RUN_EXIT=${PIPESTATUS[0]}
set -e
echo "EXIT=$RUN_EXIT" | tee -a "$RUN_DIR/run.log"
exit "$RUN_EXIT"
