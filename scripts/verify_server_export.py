"""Verify a private server export before importing into an isolated directory."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import sys
import zipfile
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from finsight_ashare.common import digest, file_hash, write_json
from finsight_ashare.evaluation import aggregate, score


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_provenance(provenance, receipt, base_config, runtime_config):
    experiment = {k: v for k, v in provenance.items() if k not in {"experiment_digest", "status", "created"}}
    expected = receipt["experiment_digest"]
    require(expected == provenance["experiment_digest"], "Experiment receipt digests differ")
    if digest(experiment) == expected:
        return "exact"
    # v0.3.3 kept a mutable reference to the config before adding runtime overrides.
    require(experiment["training"] == runtime_config, "Unrecognized provenance mismatch")
    restored = {**experiment, "training": base_config}
    require(digest(restored) == expected, "Original config cannot reproduce experiment digest")
    return "legacy_v0.3.3_mutated_config_reconstructed_from_archived_base_config"


def verify_and_import(archive, expected_hash, destination):
    require(file_hash(archive) == expected_hash.lower(), "Export SHA256 differs from server")
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        names = package.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP member")
        for item in package.infolist():
            path = PurePosixPath(item.filename)
            require(not path.is_absolute() and not PureWindowsPath(item.filename).drive
                    and ".." not in path.parts and "\\" not in item.filename,
                    "Unsafe archive member: " + item.filename)
            require(not stat.S_ISLNK(item.external_attr >> 16), "Archive contains a symbolic link")
        require(package.testzip() is None, "ZIP CRC check failed")
        manifest = json.loads(package.read("EXPORT_MANIFEST.json"))
        require(set(names) == set(manifest["files"]) | {"EXPORT_MANIFEST.json", "run_summaries.json"},
                "Manifest inventory differs from ZIP")
        for name, expected in manifest["files"].items():
            with package.open(name) as stream:
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            require(h.hexdigest() == expected, "Manifest hash mismatch: " + name)
        print(f"ARCHIVE_VERIFIED files={len(names)}", flush=True)
        read = lambda name: json.loads(package.read(name))
        receipt = read("outputs/training/pilot-probe.json")
        folder = receipt["adapter"]
        require(receipt["status"] == "completed" and receipt["pilot"] is True
                and receipt["probe"] is True and receipt["global_step"] == 20, "Pilot receipt incomplete")
        require(receipt == read(folder + "/completion.json"), "Completion receipts differ")
        for name, expected in receipt["adapter_hashes"].items():
            require(manifest["files"][folder + "/" + name] == expected, "Adapter hash differs from receipt")
        frozen = read("data/pilot/manifest.json")
        require(digest(frozen) == receipt["dataset_digest"], "Dataset digest differs from training receipt")
        require(digest(read("configs/project.json")) == frozen["config_digest"], "Frozen config mismatch")
        require(digest(read("data/verified/facts.json")) == frozen["facts_digest"], "Frozen fact mismatch")
        for name, expected in frozen["artifacts"].items():
            require(digest(read("data/pilot/" + name)) == expected, "Frozen artifact mismatch: " + name)
        provenance = read(folder + "/provenance.json")
        provenance_check = verify_provenance(provenance, receipt, yaml.safe_load(package.read("configs/lora.yaml")),
                                            yaml.safe_load(package.read(folder + "/train.yaml")))
        state = read(folder + "/trainer_state.json")
        losses = [float(x["loss"]) for x in state["log_history"] if "loss" in x]
        require(state["global_step"] == 20 and losses and all(math.isfinite(x) for x in losses),
                "Incomplete steps or invalid training losses")
        require(losses[-1] == receipt["last_loss"], "Last loss differs from receipt")
        reload = read(folder + "/reload_check.json")
        require(bool(reload["response"].strip()), "Adapter reload response is empty")
        lengths = read(folder + "/token_lengths.json")
        train_rows = read("data/pilot/train.json")
        require(len(lengths) == len(train_rows) == 10, "Unexpected pilot count")
        require({x["id"] for x in lengths} == {x["id"] for x in train_rows}, "Token-check IDs differ")
        require(all(0 < x["tokens"] <= 4096 for x in lengths), "Token limit exceeded")
        statuses = [(n, read(n)) for n in names if n.startswith("runs/") and n.endswith("/status.json")]
        runs = [(n, s) for n, s in statuses if s.get("command") == ["train", "--probe", "--pilot"]
                and s.get("status") == "completed" and s.get("exit_code") == 0]
        require(len(runs) == 1, "Expected one completed pilot run; select explicitly if multiple exist")
        run_name, run_status = runs[0]
        gpu_name = run_name.removesuffix("status.json") + "gpu.csv"
        gpu = {}
        for _, index, memory, util in csv.reader(io.StringIO(package.read(gpu_name).decode("utf-8"))):
            item = gpu.setdefault(index.strip(), {"sampled_max_memory_mib": 0, "sampled_max_utilization_pct": 0})
            item["sampled_max_memory_mib"] = max(item["sampled_max_memory_mib"], int(memory))
            item["sampled_max_utilization_pct"] = max(item["sampled_max_utilization_pct"], int(util))
        require(len(gpu) == 4 and all(x["sampled_max_utilization_pct"] > 0 for x in gpu.values()),
                "Four active GPUs not observed")
        cases_paths = [n for n in names if n.startswith("outputs/smoke/") and n.endswith("-llm/cases.json")
                       and len(read(n)) == 10]
        require(len(cases_paths) == 1, "Expected one 10-question baseline")
        cases = read(cases_paths[0])
        tasks = {t["id"]: t for t in read("data/pilot/train_tasks.json")}
        rescored = []
        for case in cases:
            task = tasks[case["id"]]
            result = score(task, case["trace"])
            require(all(result[k] == case["score"][k] for k in result), "Baseline score mismatch")
            rescored.append({"expected": task["expected"], "score": result})
        report = {"archive": str(archive.resolve()), "archive_sha256": expected_hash.lower(),
                  "member_count": len(names), "manifest_hashes_verified": len(manifest["files"]),
                  "destination": str(destination), "receipt": receipt, "run_status": run_status,
                  "provenance_check": provenance_check,
                  "training_metrics": read(folder + "/train_results.json"),
                  "max_pilot_tokens": max(x["tokens"] for x in lengths), "gpu_samples": gpu,
                  "baseline": aggregate(rescored), "baseline_cases": cases_paths[0],
                  "reload_response": reload["response"],
                  "limits": ["No local GPU reload; verified recorded server reload and weight hashes.",
                             "GPU samples are device-wide at ~30s intervals, not exact process peaks.",
                             "Structured baseline score does not validate every prose statement.",
                             "Raw PDFs, full run logs and source snapshot are not included in this export.",
                             "No formal training or held-out evaluation results."]}
        # Check existing destinations before writing any file; never overlay current project data.
        for name in names:
            target = destination / name
            require(target.resolve().is_relative_to(destination), "Destination escapes import directory")
            if target.exists():
                require(target.is_file() and file_hash(target) == hashlib.sha256(package.read(name)).hexdigest(),
                        "Existing import differs: " + name)
        for name in names:
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(name) as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
        write_json(destination / "LOCAL_VERIFICATION.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    result = verify_and_import(args.archive, args.sha256, args.destination)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
