"""Fresh-process baseline/optimized measurements, using one identical pinned planner."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ADDON = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("FINSIGHT_ROOT", ADDON.parent)).resolve()
sys.path[:0] = [str(ADDON / "metrics"), str(ROOT / "src"), str(ROOT / "evaluation_v07"), str(ROOT / "chat_v071")]
from runtime_check import save


def signature(result):
    return [{k: row.get(k) for k in ("year", "metric", "transform", "status", "value", "unit", "citations")}
            for row in result.get("results", [])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["benchmark", "chat"])
    parser.add_argument("--backend", choices=["baseline", "fast"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--thinking", choices=["on", "off"], default="on")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--question", default="中兴通讯2024年毛利率、资产负债率")
    args = parser.parse_args()
    import torch
    from chat import execute_question, verify_inputs, write_report
    from engine import MetricEngine
    from planner import model_plan, request_context
    from planner_model import PlannerModel
    from resolver import CompanyResolver
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    facts, _ = verify_inputs()
    engine = MetricEngine(facts)
    resolver = CompanyResolver.load(ROOT, ROOT / "evaluation_v07")
    cls = PlannerModel
    extra = {}
    if args.backend == "fast":
        from fast_model import FastPlannerModel
        cls, extra = FastPlannerModel, {"gpu": args.gpu}
    print("STAGE=load_model BACKEND=" + args.backend, flush=True)
    started = time.perf_counter()
    model = cls(ROOT, thinking=args.thinking == "on", max_new_tokens=args.max_new_tokens,
                context_length=args.context_length, decoding="sample", seed=42, **extra)
    devices = sorted({p.device.index for p in model.model.parameters() if p.device.type == "cuda"})

    def sync():
        for device in devices:
            torch.cuda.synchronize(device)

    sync()
    runtime = {"backend": args.backend, "load_seconds": time.perf_counter() - started,
               "fast_path_available": bool(module.is_fast_path_available),
               "hf_device_map": {k: str(v) for k, v in getattr(model.model, "hf_device_map", {}).items()},
               "parameter_devices": sorted({str(p.device) for p in model.model.parameters()}),
               "gpu_peak_memory_bytes": {str(d): torch.cuda.max_memory_allocated(d) for d in devices}}
    save(output / "runtime.json", runtime)
    print("RUNTIME=" + json.dumps(runtime), flush=True)
    if args.mode == "chat":
        while True:
            try:
                question = input("财务指标> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in {"exit", "quit"}:
                break
            if not question:
                continue
            result = execute_question(question, engine, resolver, "model", model)
            result.update(inference_runtime=runtime, inference_protocol="inference-v0.9.1")
            dest = output / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            save(dest / "trace.json", result)
            write_report(dest / "report.html", result)
            print(json.dumps({k: result.get(k) for k in ("status", "message", "results")}, ensure_ascii=False, indent=2), flush=True)
            print("TRACE=" + str(dest / "trace.json"), flush=True)
        return

    # Obtain exactly the actual planner prompt without running inference or financial calculations.
    class Capture:
        messages = None

        def generate(self, messages):
            self.messages = messages
            return '{"requests":[],"clarification":"prompt capture only"}'

    capture = Capture()
    from engine import MetricError
    try:
        model_plan(request_context(args.question, resolver, engine.available_years), engine.registry, capture)
    except MetricError as error:
        if error.code != "needs_clarification" or capture.messages is None:
            raise
    prompt = model.tokenizer.apply_chat_template(capture.messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=args.thinking == "on")
    batch = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.model.device)
    input_tokens = batch["input_ids"].shape[1]
    eos = model.tokenizer.convert_tokens_to_ids("<|im_end|>")
    pad = model.tokenizer.pad_token_id
    if pad is None:
        pad = model.tokenizer.eos_token_id
    print("STAGE=warmup; excludes first kernel compilation from measured runs", flush=True)
    with torch.inference_mode():
        model.model.generate(**batch, max_new_tokens=8, min_new_tokens=8, do_sample=False,
                             eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
    sync()
    timings = []
    for repeat in range(args.repeat):
        print(f"STAGE=fixed_work repeat={repeat + 1}/{args.repeat} tokens=128", flush=True)
        sync()
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.model.generate(**batch, max_new_tokens=128, min_new_tokens=128,
                do_sample=False, eos_token_id=eos, pad_token_id=pad, return_dict_in_generate=False)
        sync()
        elapsed = time.perf_counter() - started
        count = generated.shape[1] - input_tokens
        if count != 128:
            raise RuntimeError("Fixed-work run did not generate 128 tokens")
        timings.append({"tokens": count, "seconds": elapsed, "tokens_per_second": count / elapsed})
        print("FIXED_WORK=" + json.dumps(timings[-1]), flush=True)
    fixed = {"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_tokens": input_tokens,
             "samples": timings, "median_seconds": statistics.median(x["seconds"] for x in timings),
             "note": "128 output tokens including prefill; not steady-state decode-only throughput"}
    save(output / "fixed_work.json", fixed)
    print("STAGE=real_question THINKING=" + args.thinking, flush=True)
    sync()
    started = time.perf_counter()
    result = execute_question(args.question, engine, resolver, "model", model)
    sync()
    result.update(inference_protocol="inference-v0.9.1", inference_runtime=runtime,
                  synchronized_question_seconds=time.perf_counter() - started)
    save(output / "trace.json", result)
    write_report(output / "report.html", result)
    print("QUESTION_STATUS=" + result["status"], flush=True)
    print("TRACE=" + str(output / "trace.json"), flush=True)


if __name__ == "__main__":
    main()
