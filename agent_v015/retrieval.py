"""模块：财报 RAG；作用：持久化分块、BM25/向量融合与页码溯源；核心业务：是。"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .storage import ROOT, read, save, sha


def terms(text):
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", text.lower())
    return [t for word in words for t in
            ([word] if not re.search(r"[\u4e00-\u9fff]", word) or len(word) == 1
             else [word[i:i+2] for i in range(len(word)-1)])]


class Encoder:
    """BGE-M3 dense CLS pooling; matches the local 1_Pooling configuration."""
    def __init__(self, root, device="cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.device = torch, device
        path = root / "models/bge-m3"
        pooling = read(path / "1_Pooling/config.json")
        if not pooling.get("pooling_mode_cls_token") or pooling.get("pooling_mode_mean_tokens"):
            raise ValueError("Unsupported BGE pooling config")
        print("EMBEDDING_LOAD device=" + device, flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(device).eval()

    def encode(self, texts):
        batch = self.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        if batch["input_ids"].shape[1] > 1024:
            raise ValueError("Embedding input exceeds 1024 tokens; not silently truncated")
        with self.torch.inference_mode():
            output = self.model(**{k: v.to(self.device) for k, v in batch.items()})
            vectors = self.torch.nn.functional.normalize(output.last_hidden_state[:, 0].float(), dim=-1)
        return vectors.cpu().numpy()


def embedding_fingerprint(root):
    path = root / "models/bge-m3"
    names = ["config.json", "tokenizer.json", "1_Pooling/config.json", "pytorch_model.bin"]
    return {name: sha(path / name) for name in names}


def build(root=ROOT, *, split="validation", backend="hybrid", device="cpu", batch_size=16):
    """Write metadata last: interrupted builds never look complete."""
    import numpy as np
    if split not in {"validation", "development"} or backend not in {"bm25", "hybrid"}:
        raise ValueError("Invalid index configuration; held-out test split is reserved")
    config = read(root / "configs/project.json")
    allowed = {c["code"] for c in config["companies"]
               if c["split"] == "validation" or (split == "development" and c["split"] == "train")}
    reports = {r["id"]: r for r in read(root / "data/manifest.json") if r["company"] in allowed}
    for report in reports.values():
        if sha(root / "data/raw" / (report["id"] + ".pdf")) != report["sha256"]:
            raise ValueError("PDF hash mismatch: " + report["id"])
    chunks, pages, seen = [], [], set()
    with (root / "data/processed/pages.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            if page["company"] not in allowed:
                continue
            report = reports[page["report_id"]]
            if (page["company"], page["report_year"], page["pdf_sha256"]) != (
                    report["company"], report["report_year"], report["sha256"]):
                raise ValueError("Page provenance mismatch")
            key = (page["report_id"], page["page"])
            if key in seen or type(page["page"]) is not int or page["page"] < 1:
                raise ValueError("Duplicate/invalid PDF page")
            seen.add(key)
            page = {**page, "source_url": report["url"]}
            pages.append(page)
            # Preserve exact text and page boundary. Overlap prevents losing boundary sentences.
            for start in range(0, len(page["text"]), 432):
                text = page["text"][start:start+512]
                if not text.strip():
                    continue
                chunks.append({**page, "id": f"{page['report_id']}:p{page['page']}:c{start}",
                               "start": start, "end": start+len(text), "text": text})
    if not chunks:
        raise ValueError("No pages for requested split")
    dest = root / "outputs/agent_v015/index" / f"{split}-{backend}"
    if (dest / "metadata.json").exists():
        index = Index(root, dest)
        print(f"INDEX_CACHE_VERIFIED chunks={len(index.chunks)} path={dest}", flush=True)
        return dest
    save(dest / "chunks.json", chunks)
    save(dest / "pages.json", pages)
    fingerprint = None
    if backend == "hybrid":
        fingerprint = embedding_fingerprint(root)
        encoder = Encoder(root, device)
        output = np.lib.format.open_memmap(dest / "vectors.partial.npy", mode="w+", dtype="float32", shape=(len(chunks), 1024))
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start+batch_size]
            vectors=encoder.encode([c["text"] for c in batch])
            if not np.isfinite(vectors).all():
                raise ValueError("Embedding produced non-finite values; index not finalized")
            output[start:start+len(batch)] = vectors
            print(f"EMBEDDING_PROGRESS={min(start+batch_size,len(chunks))}/{len(chunks)}", flush=True)
        output.flush()
        del output
        (dest / "vectors.partial.npy").replace(dest / "vectors.npy")
    names = ["chunks.json", "pages.json"] + (["vectors.npy"] if backend == "hybrid" else [])
    metadata = {"protocol": "agent-v015-index-1", "split": split, "backend": backend,
                "companies": sorted(allowed), "reports": len(reports), "pages": len(pages), "chunks": len(chunks),
                "chunk_chars": 512, "overlap_chars": 80, "embedding": fingerprint,
                "inputs": {n: sha(root/n) for n in ["data/processed/pages.jsonl", "data/manifest.json", "configs/project.json"]},
                "files": {n: sha(dest/n) for n in names}}
    save(dest / "metadata.json", metadata)
    print(f"INDEX_READY={dest} chunks={len(chunks)}", flush=True)
    return dest


class Index:
    def __init__(self, root, path):
        self.root, self.path = root, Path(path)
        self.meta = read(self.path / "metadata.json")
        for name, expected in self.meta["inputs"].items():
            if sha(root/name) != expected:
                raise ValueError("Index input changed: " + name)
        for name, expected in self.meta["files"].items():
            if sha(self.path/name) != expected:
                raise ValueError("Index artifact changed: " + name)
        self.chunks = read(self.path / "chunks.json")
        self.pages = {(p["report_id"],p["page"]): p for p in read(self.path/"pages.json")}
        self.counts = [Counter(terms(c["text"])) for c in self.chunks]
        self.encoder, self.vectors = None, None
        if self.meta["backend"] == "hybrid":
            import numpy as np
            if embedding_fingerprint(root) != self.meta["embedding"]:
                raise ValueError("Embedding snapshot changed; rebuild in a new index directory")
            self.vectors = np.load(self.path/"vectors.npy", mmap_mode="r", allow_pickle=False)
            if self.vectors.shape != (len(self.chunks), 1024):
                raise ValueError("Invalid vector shape")

    def search(self, query, company, years, top_k=4):
        if not isinstance(query,str) or not query.strip() or len(query)>300 or type(top_k) is not int or not 1<=top_k<=6:
            raise ValueError("Invalid search query/top_k")
        candidates = [i for i,c in enumerate(self.chunks) if c["company"]==company and c["report_year"] in years]
        if not candidates:
            return []
        tokens = set(terms(query))
        avg = sum(sum(self.counts[i].values()) for i in candidates)/len(candidates) or 1
        frequencies = {t: sum(t in self.counts[i] for i in candidates) for t in tokens}
        scores = {}
        for i in candidates:
            counts = self.counts[i]
            scores[i] = sum(math.log(1+(len(candidates)-frequencies[t]+0.5)/(frequencies[t]+0.5))*
                            counts[t]*2.2/(counts[t]+1.2*(0.25+0.75*sum(counts.values())/avg))
                            for t in tokens if counts[t])
        lexical = sorted((i for i in candidates if scores[i]>0), key=lambda i:(-scores[i],i))[:50]
        rankings = [lexical]
        if self.vectors is not None:
            if self.encoder is None:
                self.encoder = Encoder(self.root, "cpu")
            vector = self.encoder.encode([query])[0]
            dense = self.vectors[candidates] @ vector
            rankings.append([candidates[j] for j in sorted(range(len(candidates)), key=lambda j:-float(dense[j]))[:50]])
        fused = Counter()
        for ranking in rankings:
            for rank, i in enumerate(ranking,1):
                fused[i] += 1/(60+rank)
        return [{**self.chunks[i], "rrf_score":score, "evidence_kind":"report_excerpt_not_audited_fact"}
                for i,score in fused.most_common(top_k)]
