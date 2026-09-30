"""Fresh subprocess probe; no saved result can bypass a failed CUDA check."""
import json
import sys


def main():
    import torch
    report = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
              "gpu_count": torch.cuda.device_count(), "devices": [], "passed": False}
    try:
        if not torch.cuda.is_available() or report["gpu_count"] != 4:
            raise RuntimeError("Exactly four visible CUDA GPUs are required")
        for index in range(report["gpu_count"]):
            print(f"BF16 GPU {index}: {torch.cuda.get_device_name(index)}", file=sys.stderr, flush=True)
            with torch.cuda.device(index):
                x = torch.randn(512, 512, device=f"cuda:{index}", dtype=torch.bfloat16, requires_grad=True)
                loss = (x @ x.T).float().square().mean()
                loss.backward()
                torch.cuda.synchronize()
                if not torch.isfinite(loss).item() or not torch.isfinite(x.grad).all().item():
                    raise RuntimeError("Non-finite BF16 forward/backward result")
                report["devices"].append({"index": index, "name": torch.cuda.get_device_name(index), "passed": True})
                del x, loss
                torch.cuda.empty_cache()
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
    print(json.dumps(report), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
