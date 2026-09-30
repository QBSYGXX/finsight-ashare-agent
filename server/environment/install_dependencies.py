"""Install the exact candidate profile into a Linux Conda prefix."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(args):
    print("RUN " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


def main():
    profile = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    constraints = Path(os.environ.get("TMPDIR", "/tmp")) / "finsight-constraints.txt"
    constraints.write_text("\n".join(f"{name}=={profile[name]}" for name in ("torch", "torchvision", "torchaudio", "transformers", "peft")) + "\n", encoding="utf-8")
    pip = [sys.executable, "-m", "pip", "install", "--timeout", "90", "--retries", "5"]
    run([*pip, "--index-url", profile["torch_index"], *[f"{name}=={profile[name]}" for name in ("torch", "torchvision", "torchaudio")]])
    run([*pip, "-c", constraints,
         "https://github.com/hiyouga/LlamaFactory/archive/" + profile["llamafactory_commit"] + ".zip",
         "sentence-transformers>=3,<6", "PyMuPDF>=1.25,<2", "PyYAML>=6,<7", "conda-pack==0.8.1", "setuptools==80.9.0", "pytest>=8,<10"])


if __name__ == "__main__":
    main()
