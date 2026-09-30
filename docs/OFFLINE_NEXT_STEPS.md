# Offline first model run

Target: `/calc/users/geyuboao/workspace/FinSight` on `Cisri-M-lgpu2`.
The installed portable environment is reused. No Miniconda repair, pip install,
local SSH, or server network access is needed for these steps.

Upload these four files into the target directory:

- `finsight-offline-patch-v0.3.zip`
- `finsight-offline-patch-v0.3.zip.sha256`
- `finsight-assets-v0.3.zip`
- `finsight-assets-v0.3.zip.sha256`

The patch is flat: unzip inside FinSight, with no extra project folder.
It updates only listed source/test/doc files, keeping the environment profile
unchanged so the portable environment receipt stays valid.
The assets ZIP contains Qwen3.5-9B, bge-m3, a pinned SHA256 manifest, and the
audited Maotai 2024 PDF. It does not include all 36 reports or training results.
Install into the current initial deployment before adding your own reports:
the assets ZIP includes the initial `data/manifest.json`.

```bash
cd /calc/users/geyuboao/workspace/FinSight
sha256sum -c finsight-offline-patch-v0.3.zip.sha256
sha256sum -c finsight-assets-v0.3.zip.sha256
unzip -o finsight-offline-patch-v0.3.zip
unzip -o finsight-assets-v0.3.zip
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
bash server/run.sh doctor --offline
```

Continue only after `EXIT=0`:

```bash
bash server/run.sh setup --offline
```

Continue only after `EXIT=0`. First verify the financial tools without an LLM:

```bash
bash server/run.sh smoke --replay
```

Then run the real model and BGE retrieval:

```bash
bash server/run.sh smoke
```

`SMOKE_OUTPUT=...` gives the HTML report and full tool trace.
`LLM_EXECUTED=True` distinguishes model execution from scripted replay.
Completion also requires `EXIT=0`; a launched process is not a passing case.
Logs and exit codes are stored under `runs/`. Send the final console output or
the latest `run.log` back for the next step.

Driver 535 is conditionally accepted under CUDA 12 minor-version compatibility
only after a fresh four-GPU BF16 forward/backward probe. This is not evidence
that Qwen kernels, multi-GPU NCCL training, or the 20-step LoRA probe passed.
Do not start full training before those checks and dataset validation.

Local package production:

```text
python scripts/prepare_offline_assets.py --output ../_release_work/finsight-assets-v0.3.zip
python scripts/build_offline_patch.py --output ../_release_work/finsight-offline-patch-v0.3.zip
```

The downloader uses standard HTTPS and accepts proxy environment variables.
It pins model commits and checks large files against upstream LFS SHA256 hashes.
Do not embed credentials or machine-specific proxy addresses in source control.

For unreliable long downloads, `--transport curl` keeps resumable `.partial`
files. `--transport modelscope` uses Qwen's official ModelScope weight files
only after their published SHA256 hashes match the pinned Hugging Face commit;
configuration/tokenizer files still come from the pinned Hugging Face snapshot.
Set `FINSIGHT_ARIA2` to a local aria2 executable to enable eight connections per
Qwen shard with SHA256 validation. Interrupted aria2 downloads need both the
`.partial` file and its `.aria2` control file for safe continuation. Apparent
file size during a segmented download is not evidence of completion.
