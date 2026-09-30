#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-}"
if [[ -z "$ARCHIVE" || "$#" != 1 || ! -f "$ARCHIVE" ]]; then
    echo 'Usage: bash server/install_env.sh /path/to/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz' >&2
    exit 2
fi
for TOOL in uname getconf awk tar sha256sum tee flock df; do
    command -v "$TOOL" >/dev/null || { echo "Missing system utility: $TOOL" >&2; exit 2; }
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo 'This archive requires Linux x86_64.' >&2
    exit 2
fi
GLIBC="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
if ! awk -v v="$GLIBC" 'BEGIN {split(v,a,"."); exit !(a[1]>2 || (a[1]==2 && a[2]>=28))}'; then
    echo "glibc >= 2.28 required; found $GLIBC. Send this output back before installing." >&2
    exit 2
fi
CHECKSUM="$PROJECT_ROOT/server/environment/finsight-conda-linux-x86_64-cu126-v0.1.tar.gz.sha256"
if [[ ! -f "$CHECKSUM" ]]; then
    echo 'The verified checksum is missing from this source release; environment build is incomplete.' >&2
    exit 2
fi
EXPECTED="$(awk 'NR==1 {print $1}' "$CHECKSUM")"
[[ "$EXPECTED" =~ ^[a-f0-9]{64}$ ]] || { echo 'Invalid release checksum.' >&2; exit 2; }
mkdir -p "$PROJECT_ROOT/.runtime/envs" "$PROJECT_ROOT/runs"
exec 9>"$PROJECT_ROOT/.runtime/install-env.lock"
flock -n 9 || { echo 'Another environment installation is running.' >&2; exit 2; }
LOG="$PROJECT_ROOT/runs/install-env-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'RESULT=$?; echo "INSTALL_EXIT=$RESULT LOG=$LOG"' EXIT
echo "STAGE=archive_integrity GLIBC=$GLIBC"
ACTUAL="$(sha256sum -- "$ARCHIVE" | awk '{print $1}')"
[[ "$ACTUAL" == "$EXPECTED" ]] || { echo 'SHA256 mismatch. Upload the matching environment archive again.'; exit 2; }
PREFIX="$PROJECT_ROOT/.runtime/envs/finsight"
export PYTHONNOUSERSITE=1 PYTHONUTF8=1 PYTHONUNBUFFERED=1
unset PYTHONHOME PYTHONPATH
if [[ -e "$PREFIX/.finsight-ready.json" ]]; then
    "$PREFIX/bin/python" "$PROJECT_ROOT/server/environment/finish_install.py" --root "$PROJECT_ROOT" --sha256 "$EXPECTED" --check
    echo "ALREADY_INSTALLED=$PREFIX"
    exit 0
fi
if [[ -e "$PREFIX" ]]; then
    if [[ ! -f "$PREFIX/.finsight-installing" || "$(< "$PREFIX/.finsight-installing")" != "$EXPECTED" ]]; then
        echo "Refusing to overwrite an unknown environment: $PREFIX"
        exit 2
    fi
    echo 'Resuming this previously interrupted extraction at the same final prefix.'
else
    mkdir "$PREFIX"
    printf '%s\n' "$EXPECTED" > "$PREFIX/.finsight-installing"
fi
FREE_KIB="$(df -Pk "$PREFIX" | awk 'END {print $4}')"
if (( FREE_KIB < 30 * 1024 * 1024 )); then
    echo 'At least 30 GiB of free disk is required during installation.'
    exit 2
fi
echo "STAGE=extract PREFIX=$PREFIX"
# Relocate only at the final prefix; moving it after conda-unpack breaks shebangs.
tar -xzf "$ARCHIVE" -C "$PREFIX" --checkpoint=20000 --checkpoint-action=echo='extracting checkpoint %u'
echo 'STAGE=conda_unpack'
"$PREFIX/bin/python" "$PREFIX/bin/conda-unpack"
export PATH="$PREFIX/bin:$PATH"
export CONDA_PREFIX="$PREFIX"
echo 'STAGE=verify_imports_and_lora'
"$PREFIX/bin/python" "$PROJECT_ROOT/server/environment/verify_environment.py" --output "$PROJECT_ROOT/outputs/environment/server-env-check.json"
"$PREFIX/bin/python" -m pip check
"$PREFIX/bin/python" "$PROJECT_ROOT/server/environment/finish_install.py" --root "$PROJECT_ROOT" --sha256 "$EXPECTED"
echo "INSTALL_COMPLETE=$PREFIX"
echo 'Next: bash server/run.sh doctor'
