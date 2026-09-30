"""CPU validation of local model assets; does not claim 9B GPU inference."""
import os
from pathlib import Path
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from transformers import AutoConfig, AutoTokenizer
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from finsight_ashare.assets import verify_snapshots
    from finsight_ashare.common import now, write_json
    verify_snapshots(ROOT)
    results = {}
    for name in ("Qwen3.5-9B", "bge-m3"):
        path = str(ROOT / "models" / name)
        configuration = AutoConfig.from_pretrained(path, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        tokens = tokenizer("Financial report revenue growth", return_tensors="pt")
        if tokens["input_ids"].numel() < 1:
            raise ValueError("Empty tokenization")
        results[name] = {"model_type": configuration.model_type, "tokenizer": "passed"}
        print("CONFIG_TOKENIZER_PASS=" + name, flush=True)
    print("Loading bge-m3 on CPU for an actual embedding...", flush=True)
    model = SentenceTransformer(str(ROOT / "models/bge-m3"), device="cpu", local_files_only=True)
    vectors = model.encode(["Revenue increased", "Revenue fell"], normalize_embeddings=True)
    if vectors.shape != (2, 1024) or not np.isfinite(vectors).all():
        raise ValueError("Invalid BGE embeddings")
    results["bge-m3"]["embedding_shape"] = list(vectors.shape)
    write_json(ROOT / "exports/offline-model-check.json", {"created": now(), "results": results,
        "network": "offline", "qwen_gpu_inference_tested": False})
    print("OFFLINE_MODEL_ASSETS_PASS; Qwen GPU inference remains a server check", flush=True)


if __name__ == "__main__":
    main()
