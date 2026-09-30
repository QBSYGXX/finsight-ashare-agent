#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export PYTHONNOUSERSITE=1
unset PYTHONHOME
export PYTHONPATH="$PROJECT_ROOT/src"
PORTABLE_PREFIX="$PROJECT_ROOT/.runtime/envs/finsight"
if [[ -d "$PORTABLE_PREFIX" ]]; then
    if [[ ! -x "$PORTABLE_PREFIX/bin/python" || ! -f "$PORTABLE_PREFIX/.finsight-ready.json" ]]; then
        echo 'Portable environment installation is incomplete. Run bash server/install_env.sh ARCHIVE first.' >&2
        exit 2
    fi
    PROJECT_PYTHON="$PORTABLE_PREFIX/bin/python"
    export CONDA_PREFIX="$PORTABLE_PREFIX"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PROJECT_PYTHON="$PROJECT_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PROJECT_PYTHON="$(command -v python3)"
else
    echo 'Install the portable environment with bash server/install_env.sh ARCHIVE; no system Python or Conda is needed.' >&2
    exit 2
fi
export PATH="$(dirname -- "$PROJECT_PYTHON"):$PATH"
exec "$PROJECT_PYTHON" -u "$PROJECT_ROOT/server/runner.py" "$@"
