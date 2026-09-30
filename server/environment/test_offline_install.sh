#!/usr/bin/env bash
set -euo pipefail
[[ ! -e /build/finsight-env ]] || { echo 'Original build environment must be inaccessible.'; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
mkdir -p /verification
TEST_ROOT="$(mktemp -d /verification/finsight-install.XXXXXX)"
cp -R /source/server /source/configs /source/src "$TEST_ROOT/"
ARCHIVE=/exports/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz
echo "STAGE=offline_fresh_install TEST_ROOT=$TEST_ROOT"
bash "$TEST_ROOT/server/install_env.sh" "$ARCHIVE"
echo 'STAGE=repeat_install_receipt_check'
bash "$TEST_ROOT/server/install_env.sh" "$ARCHIVE"
echo 'STAGE=project_and_training_entrypoints'
bash "$TEST_ROOT/server/run.sh" status
export PATH="$TEST_ROOT/.runtime/envs/finsight/bin:$PATH"
"$TEST_ROOT/.runtime/envs/finsight/bin/llamafactory-cli" version
cp "$TEST_ROOT/outputs/environment/server-env-check.json" /exports/finsight-offline-install-check.json
echo "OFFLINE_INSTALL_PASSED=$TEST_ROOT"
