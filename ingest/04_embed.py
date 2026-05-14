"""Generate BGE-small-en-v1.5 embeddings for every chunk.

Outputs corpus/chunks/<paper_id>.emb.npy  (float32, shape [n_chunks, 384]) and
amends each chunk JSON with its row index so 05_build_index.py can pair them.

Model: BAAI/bge-small-en-v1.5  — 384 dims, ~30 MB ONNX, CPU-friendly.
Convention: BGE expects 'passage: ' prefix for documents, 'query: ' for queries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import orjson
from fastembed import TextEmbedding
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import CHUNK_DIR  # noqa: E402

MODEL_NAME = "BAAI/bge-small-en-v1.5"
BATCH = 64


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("rb") as fh:
        for line in fh:
            if line.strip():
                rows.append(orjson.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as fh:
        for r in rows:
            fh.write(orjson.dumps(r))
            fh.write(b"\n")


def main() -> None:
    print(f"Loading {MODEL_NAME}...")
    model = TextEmbedding(MODEL_NAME)

    chunk_files = sorted(CHUNK_DIR.glob("*.jsonl"))
    for cf in tqdm(chunk_files, desc="embed"):
        emb_path = cf.with_suffix(".emb.npy")
        rows = load_jsonl(cf)
        if not rows:
            continue
        if emb_path.exists() and emb_path.stat().st_size > 0:
            # Quick sanity: shape matches row count.
            arr = np.load(emb_path)
            if arr.shape[0] == len(rows):
                continue
        # Build embedding input: include title + section as context so the
        # chunk embedding reflects paper-level signal as well as the body.
        def _embed_text(r: dict) -> str:
            title = r.get("title", "") or ""
            section = (r.get("section", "") or "").strip()
            ctx = title if not section else f"{title} — {section}"
            return f"passage: {ctx}\n\n{r['text']}" if ctx else f"passage: {r['text']}"

        texts = [_embed_text(r) for r in rows]
        # fastembed yields per-doc np.ndarray
        vecs: list[np.ndarray] = []
        for v in model.embed(texts, batch_size=BATCH):
            vecs.append(np.asarray(v, dtype=np.float32))
        arr = np.stack(vecs).astype(np.float32, copy=False)
        # L2-normalise so cosine == dot product downstream.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        np.save(emb_path, arr)

        for i, r in enumerate(rows):
            r["row"] = i
        save_jsonl(cf, rows)

    print(f"DONE: embedded {len(chunk_files)} papers")


if __name__ == "__main__":
    main()
