"""Install wheels into an owned venv, reusing only the existing Python runtime."""
import json
import subprocess
import sys
from runtime import ADDON, BASE, ENV, ROOT, process_env, save, verify_package, sha256


# 模块：独立安装；作用：创建可重试的专用环境，绝不向基线环境装包；核心业务：否。
def main(gpu):
    print("STAGE=verify_all_offline_wheel_hashes", flush=True)
    manifest = verify_package(include_wheels=True)
    lock_hash = sha256(ADDON / "requirements.lock.txt")
    owner_path = ENV.parent / "finsight-vllm-v013.owner.json"
    expected = {"path": str(ENV.resolve()), "lock_sha256": lock_hash, "base_python": str(BASE / "bin/python")}
    if ENV.exists() or owner_path.exists():
        if not owner_path.exists() or json.loads(owner_path.read_text()) != expected:
            raise RuntimeError("Existing environment is not owned by this pinned installer; no files changed")
    else:
        if ENV.resolve() == BASE.resolve():
            raise RuntimeError("Refusing to install into baseline")
        ENV.parent.mkdir(parents=True, exist_ok=True)
        save(owner_path, expected)
    env = process_env(gpu)
    print("STAGE=create_owned_venv PATH=" + str(ENV), flush=True)
    subprocess.run([str(BASE / "bin/python"), "-m", "venv", "--copies", "--without-pip", str(ENV)], env=env, check=True)
    python = ENV / "bin/python"
    bootstrap = ADDON / "wheelhouse/pip-25.3-py3-none-any.whl"
    if not bootstrap.is_file():
        raise RuntimeError("Pinned pip bootstrap wheel missing")
    install_env = {**env, "PYTHONPATH": str(bootstrap)}
    print("STAGE=install_pinned_wheels; no network, no source builds", flush=True)
    subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--only-binary=:all:", "--no-compile",
        "--find-links", str(ADDON / "wheelhouse"), "--require-hashes", "-r", str(ADDON / "requirements.lock.txt")],
        env=install_env, check=True)
    print("STAGE=pip_check", flush=True)
    subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    save(ENV / "finsight-install.json", {**expected, "package_commit": manifest["commit"],
        "status": "dependencies_installed", "note": "GPU/import checks are a separate required stage"})


if __name__ == "__main__":
    main(int(sys.argv[1]))
