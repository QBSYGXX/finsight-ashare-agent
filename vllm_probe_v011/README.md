# vLLM offline runtime selection probe v0.11

This package inventories Linux/kernel/glibc, GPU/driver/free memory, installed
Python packages, CUDA compiler and build tools, CUDA compatibility library
locations, container executables, Docker daemon access, module-system candidates,
filesystem/free space, and Qwen3.5-9B MTP metadata and shard presence.

It runs no network downloads, installation, model generation or training. Existing
Python/environment/weights are not modified. Only runs/, outputs/ and exports/
receive diagnostic artifacts. Commands have a 12-second timeout, retain stdout
and stderr, and the process prints a heartbeat every20seconds. Missing optional
tools are recorded as unavailable rather than treated as a deployment failure.
Container --version output is not proof that GPU containers can run.

Upload the ZIP and SHA256 to /calc/users/geyuboao/workspace/FinSight:

```bash
cd /calc/users/geyuboao/workspace/FinSight &&
sha256sum -c finsight-vllm-probe-v0.11.zip.sha256 &&
unzip -n finsight-vllm-probe-v0.11.zip &&
bash vllm_probe_v011/run.sh doctor
```

Download the ZIP shown by RESULTS and its .sha256 to the local _release_work
folder. This is the environment-selection step, not a vLLM installation package.
A successful exit means diagnostics were collected, not that vLLM is usable.

After reviewing this report, prepare a separately pinned offline environment or
compatible GPU container. Then compare single-GPU BF16 Transformers vs vLLM,
followed by vLLM with native MTP (draft depth1). Retain thinking and the same
prompt/sampling settings. Separate decode token/s from startup and prefill;
check complete financial plans and sources before switching the normal entrypoint.

CUDA13/new vLLM cannot be assumed compatible merely because the current Torch
cu126 works. Existing CUDA compatibility libraries, container availability, and
host kernel matter. This probe deliberately does not upgrade the driver or select
an unverified vLLM release automatically.
