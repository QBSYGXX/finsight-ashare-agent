#!/usr/bin/env bash
set -euo pipefail
export MAMBA_ROOT_PREFIX=/build/mamba
export PIP_CACHE_DIR=/build/cache/pip
export TMPDIR=/build/tmp
mkdir -p /build/tmp /build/cache/pip /exports
ENV_PREFIX=/build/finsight-env
ENV_PYTHON="$ENV_PREFIX/bin/python"
ARCHIVE=/exports/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz
echo 'STAGE=conda_python'
if [[ ! -x "$ENV_PYTHON" ]]; then
    micromamba create --yes --prefix "$ENV_PREFIX" --channel conda-forge --strict-channel-priority python=3.11 pip setuptools=80.9.0
fi
# conda-pack 0.8.1 imports pkg_resources, removed in recent setuptools.
micromamba install --yes --prefix "$ENV_PREFIX" --channel conda-forge --strict-channel-priority setuptools=80.9.0
echo 'STAGE=training_dependencies'
"$ENV_PYTHON" /project/server/environment/install_dependencies.py /project/configs/server-profile.json
echo 'STAGE=linux_import_and_small_model_checks'
"$ENV_PYTHON" /project/server/environment/verify_environment.py --output "$ENV_PREFIX/finsight-env-check.json"
"$ENV_PYTHON" -m pip check
"$ENV_PYTHON" -m pytest -q -p no:cacheprovider --basetemp /build/finsight-pytest-build /project/tests
"$ENV_PYTHON" -m pip freeze > "$ENV_PREFIX/finsight-requirements.lock.txt"
micromamba list --prefix "$ENV_PREFIX" --explicit > "$ENV_PREFIX/finsight-conda-explicit.txt"
cp /project/configs/server-profile.json "$ENV_PREFIX/finsight-server-profile.json"
echo 'STAGE=conda_pack'
"$ENV_PREFIX/bin/conda-pack" --prefix "$ENV_PREFIX" --output "${ARCHIVE}.partial" --format tar.gz --compress-level 3 --n-threads 4 --force
echo 'STAGE=relocation_check'
CHECK_PREFIX="$(mktemp -d /build/finsight-relocation.XXXXXX)"
tar -xzf "${ARCHIVE}.partial" -C "$CHECK_PREFIX"
"$CHECK_PREFIX/bin/python" "$CHECK_PREFIX/bin/conda-unpack"
"$CHECK_PREFIX/bin/python" /project/server/environment/verify_environment.py --output /exports/finsight-conda-environment-check.json
"$CHECK_PREFIX/bin/python" -m pip check
export PATH="$CHECK_PREFIX/bin:$PATH"
"$CHECK_PREFIX/bin/llamafactory-cli" version
cp "$ENV_PREFIX/finsight-requirements.lock.txt" /exports/finsight-conda-requirements.lock.txt
cp "$ENV_PREFIX/finsight-conda-explicit.txt" /exports/finsight-conda-explicit.txt
mv "${ARCHIVE}.partial" "$ARCHIVE"
cd /exports
sha256sum "$(basename "$ARCHIVE")" > "${ARCHIVE}.sha256"
echo "STAGE=complete ARCHIVE=$ARCHIVE"
echo "RELOCATED_ENV=$CHECK_PREFIX"
