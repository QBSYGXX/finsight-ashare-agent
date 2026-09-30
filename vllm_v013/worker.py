"""One process per engine: native gate, full financial smoke, or fixed-token benchmark."""
import argparse
import ctypes
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
from runtime import ADDON, ROOT, save, verify_package

QUESTION = "中兴通讯2024年毛利率、资产负债率"
EXPECTED = {(2024, "gross_margin", "value"), (2024, "debt_ratio", "value")}


def versions():
    names = ("torch", "transformers", "vllm", "triton", "llguidance")
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def doctor(output):
    import torch
    import vllm
    import triton
    from llguidance import LLMatcher
    ver = versions()
    required = {"torch": "2.11.0+cu129", "vllm": "0.23.0+cu129", "transformers": "5.8.0", "llguidance": "1.7.6"}
    for name, expected in required.items():
        if ver[name] != expected:
            raise RuntimeError(f"Unexpected {name}: {ver[name]}, expected {expected}")
    grammar = LLMatcher.grammar_from_json_schema(json.dumps({"type": "object"}))
    error = LLMatcher.validate_grammar(grammar)
    if error:
        raise RuntimeError("Grammar compile failed: " + str(error))
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU must be visible")
    if torch.version.cuda != "12.9":
        raise RuntimeError("Unexpected CUDA runtime")
    x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    torch.cuda.synchronize()
    if not torch.isfinite(y).all().item():
        raise RuntimeError("Non-finite BF16 matmul")
    lib = ctypes.CDLL("libcuda.so.1")
    driver = ctypes.c_int()
    if lib.cuDriverGetVersion(ctypes.byref(driver)) != 0 or driver.value < 12090:
        raise RuntimeError("Compatibility CUDA driver API not active")
    maps = Path("/proc/self/maps").read_text()
    relevant = sorted({line.split()[-1] for line in maps.splitlines() if any(n in line for n in ("libcuda.so", "libnvidia-ptxjitcompiler", "libnvJitLink"))})
    if not any("native_probe_v012/payload/compat/libcuda" in p for p in relevant):
        raise RuntimeError("Wrong libcuda loaded")
    result = {"status": "doctor_passed", "versions": ver, "cuda": torch.version.cuda,
        "driver_api": driver.value, "gpu": torch.cuda.get_device_name(0), "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "loaded_libraries": relevant, "note": "Imports, grammar and BF16 passed; full vLLM model execution is still required."}
    save(output / "doctor.json", result)
    print("DOCTOR=" + json.dumps(result), flush=True)


def capture_prompt(question, engine, resolver):
    from planner import model_plan, request_context
    class Captured(Exception):
        pass
    class Capture:
        messages = None
        def generate(self, messages):
            self.messages = messages
            raise Captured()
    capture = Capture()
    try:
        model_plan(request_context(question, resolver, engine.available_years), engine.registry, capture)
    except Captured:
        return capture.messages
    raise RuntimeError("Could not capture the exact financial planner prompt")


def financial_matches(result, expected):
    if result.get("status") != "completed" or expected.get("status") != "completed":
        return False
    def canonical(r):
        return sorted((row["year"], row["metric"], row["transform"], row["status"], str(row.get("value")),
                       row.get("unit"), tuple(row.get("citations", []))) for row in r["results"])
    return canonical(result) == canonical(expected)


def inference(args, output):
    os.environ["FINSIGHT_ROOT"] = str(ROOT)
    sys.path[:0] = [str(ADDON / "metrics"), str(ROOT / "src"), str(ROOT / "evaluation_v07"), str(ROOT / "chat_v071")]
    from chat import verify_inputs, execute_question, write_report
    from engine import MetricEngine
    from resolver import CompanyResolver
    from adapter import Engine
    facts, _ = verify_inputs()
    engine = MetricEngine(facts)
    resolver = CompanyResolver.load(ROOT, ROOT / "evaluation_v07")
    messages = capture_prompt(QUESTION, engine, resolver)
    save(output / "prompt.json", messages)
    baseline = execute_question(QUESTION, engine, resolver, "rules")
    model = Engine(args.backend)
    save(output / "runtime.json", {"versions": versions(), "backend": args.backend,
        "load_seconds": model.load_seconds, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
    rows = []
    if args.mode == "bench":
        for i in range(2):
            print(f"STAGE=warmup INDEX={i+1}/2", flush=True)
            _, record = model.run(messages, 64, fixed=True)
            save(output / f"warmup-{i+1}.json", record)
        for i in range(3):
            print(f"STAGE=fixed_token_benchmark INDEX={i+1}/3 TOKENS=512", flush=True)
            _, record = model.run(messages, 512, fixed=True)
            rows.append(record)
            save(output / f"sample-{i+1}.json", record)
        save(output / "benchmark.json", {"samples": rows,
            "median_tokens_per_second": statistics.median(r["tokens_per_generation_second"] for r in rows),
            "note": "Single-request prefill plus 512 generated tokens, 2 warmups. EOS suppressed equally. Truncated outputs are NOT financial answers. Same seed does not guarantee same text across engines."})
    print("STAGE=financial_smoke THINKING=on MAX_NEW_TOKENS=8192", flush=True)
    result = execute_question(QUESTION, engine, resolver, "model", model)
    result["matches_deterministic_reference"] = financial_matches(result, baseline)
    save(output / "trace.json", result)
    write_report(output / "report.html", result)
    print("RESULT=" + json.dumps({k: result.get(k) for k in ("status", "message", "results", "matches_deterministic_reference")}, ensure_ascii=False), flush=True)
    if hasattr(model, "llm"):
        try:
            save(output / "engine_metrics.json", [str(m) for m in model.llm.get_metrics()])
        except Exception as error:
            save(output / "engine_metrics.json", {"unavailable": repr(error)})
    if not result["matches_deterministic_reference"]:
        raise RuntimeError("Financial smoke failed or disagreed with the pinned deterministic reference")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["doctor", "smoke", "bench"])
    parser.add_argument("--backend", choices=["hf", "vllm", "vllm-mtp"], default="vllm")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    try:
        verify_package()
        if args.mode == "doctor":
            doctor(output)
        else:
            inference(args, output)
    except Exception as error:
        save(output / "error.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
