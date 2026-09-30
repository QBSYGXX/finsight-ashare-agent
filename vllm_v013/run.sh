#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$PWD/.runtime/envs/finsight/lib"
exec .runtime/envs/finsight/bin/python vllm_v013/launcher.py "$@"
