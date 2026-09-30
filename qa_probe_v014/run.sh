#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export FINSIGHT_ROOT="$PWD"
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$PWD/.runtime/envs/finsight/lib"
exec .runtime/envs/finsight/bin/python qa_probe_v014/run.py "$@"
