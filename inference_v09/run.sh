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
[[ -x "$PROJECT_PYTHON" ]] || { echo 'Existing portable environment required'; exit 2; }
export CONDA_PREFIX="$PROJECT_ROOT/.runtime/envs/finsight"
export PATH="$(dirname -- "$PROJECT_PYTHON"):$PATH"
RUN_DIR="$PROJECT_ROOT/runs/$(date +%Y%m%d-%H%M%S)-inference-v091-$$"
mkdir -p "$RUN_DIR"
BACKGROUND=0
ARGS=()
for argument in "$@"; do
    if [[ "$argument" == '--background' ]]; then BACKGROUND=1; else ARGS+=("$argument"); fi
done
echo "LOG=$RUN_DIR/run.log"
if [[ "$BACKGROUND" == 1 ]]; then
    if [[ "${ARGS[0]:-}" == chat ]]; then echo 'Interactive chat cannot run in background'; exit 2; fi
    nohup "$PROJECT_PYTHON" -B -u "$ADDON_ROOT/driver.py" "${ARGS[@]}" --run-dir "$RUN_DIR" > "$RUN_DIR/run.log" 2>&1 < /dev/null &
    RUN_PID=$!
    echo "$RUN_PID" > "$RUN_DIR/pid"
    echo "PID=$RUN_PID"
    echo "TAIL=tail -n 60 -f $RUN_DIR/run.log"
else
    set +e
    "$PROJECT_PYTHON" -B -u "$ADDON_ROOT/driver.py" "${ARGS[@]}" --run-dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/run.log"
    RUN_EXIT=${PIPESTATUS[0]}
    set -e
    echo "EXIT=$RUN_EXIT" | tee -a "$RUN_DIR/run.log"
    exit "$RUN_EXIT"
fi
