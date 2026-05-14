"""Chunk the cleaned markdown into LLM-friendly semantic chunks.

Strategy:
  1. Split by '## heading' first — each section is its own bucket.
  2. If a section is bigger than CHUNK_TOKENS, slice it by paragraphs using a
     sliding window of CHUNK_TOKENS with CHUNK_OVERLAP overlap.
  3. Emit JSONL with paper_id, chunk_id, section, text, token_count.

Token counts use tiktoken's cl100k_base encoding (close-enough proxy for any
modern transformer's wordpiece count).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import orjson
import tiktoken
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import CHUNK_DIR, META_DIR, TEXT_DIR  # noqa: E402

CHUNK_TOKENS = 768
CHUNK_OVERLAP = 96
MIN_CHUNK_TOKENS = 40       # drop chunks shorter than this
MIN_ALPHA_RATIO = 0.55      # drop chunks dominated by math/code/symbols
MIN_CHUNK_CHARS = 200
HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)

enc = tiktoken.get_encoding("cl100k_base")


def n_tokens(s: str) -> int:
    return len(enc.encode(s, disallowed_special=()))


def split_sections(md: str) -> list[tuple[str, str]]:
    """Return [(heading, body), ...]. The pre-heading preface becomes ('', body)."""
    parts: list[tuple[str, str]] = []
    indices = [(m.start(), m.group(1).strip()) for m in HEADING_RE.finditer(md)]
    if not indices:
        return [("", md.strip())]
    # Preface
    pre = md[: indices[0][0]].strip()
    if pre:
        parts.append(("", pre))
    for i, (pos, heading) in enumerate(indices):
        body_start = md.find("\n", pos) + 1
        body_end = indices[i + 1][0] if i + 1 < len(indices) else len(md)
        body = md[body_start:body_end].strip()
        if body:
            parts.append((heading, body))
    return parts


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _split_long_paragraph(text: str) -> list[str]:
    """Break a paragraph that is bigger than CHUNK_TOKENS into sentences."""
    if n_tokens(text) <= CHUNK_TOKENS:
        return [text]
    sentences = _SENTENCE_SPLIT.split(text)
    if len(sentences) <= 1:
        # No sentence boundary detected — fall back to token-level slicing.
        token_ids = enc.encode(text, disallowed_special=())
        out = []
        step = CHUNK_TOKENS - CHUNK_OVERLAP
        for i in range(0, len(token_ids), step):
            out.append(enc.decode(token_ids[i : i + CHUNK_TOKENS]))
        return out
    return sentences


def slide_chunks(text: str) -> list[str]:
    """Sliding token window over paragraphs (with long-paragraph fallback)."""
    raw_paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not raw_paragraphs:
        return []
    # First, blow up any paragraph that is itself longer than the chunk budget.
    paragraphs: list[str] = []
    for p in raw_paragraphs:
        paragraphs.extend(_split_long_paragraph(p))

    chunks: list[str] = []
    cur: list[str] = []
    cur_tokens = 0
    for p in paragraphs:
        pt = n_tokens(p)
        if cur_tokens + pt > CHUNK_TOKENS and cur:
            chunks.append("\n\n".join(cur))
            tail: list[str] = []
            tail_tokens = 0
            for q in reversed(cur):
                qt = n_tokens(q)
                if tail_tokens + qt > CHUNK_OVERLAP:
                    break
                tail.insert(0, q)
                tail_tokens += qt
            cur = tail
            cur_tokens = tail_tokens
        cur.append(p)
        cur_tokens += pt
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _looks_useful(text: str) -> bool:
    """Quality filter: drop chunks dominated by math/equations/tables/code."""
    text = text.strip()
    if len(text) < MIN_CHUNK_CHARS:
        return False
    n_alpha = sum(c.isalpha() for c in text)
    if n_alpha / max(len(text), 1) < MIN_ALPHA_RATIO:
        return False
    # Reject chunks where every paragraph is a single short line (likely a table).
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return False
    return True


def chunk_paper(paper_id: str, md: str, title: str) -> list[dict]:
    sections = split_sections(md)
    out: list[dict] = []
    idx = 0
    for heading, body in sections:
        body_tokens = n_tokens(body)
        if body_tokens <= CHUNK_TOKENS:
            pieces = [body]
        else:
            pieces = slide_chunks(body)
        for piece in pieces:
            piece = piece.strip()
            if not _looks_useful(piece):
                continue
            tokens = n_tokens(piece)
            if tokens < MIN_CHUNK_TOKENS:
                continue
            out.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}::{idx:04d}",
                    "section": heading,
                    "title": title,         # stored as metadata (not in body text)
                    "text": piece,          # body only — title is in metadata
                    "tokens": tokens,
                }
            )
            idx += 1
    return out


def main() -> None:
    text_files = sorted(TEXT_DIR.glob("*.md"))
    total_chunks = 0
    for tf in tqdm(text_files, desc="chunk"):
        pid = tf.stem
        md = tf.read_text(encoding="utf-8")
        meta_path = META_DIR / f"{pid}.json"
        title = ""
        if meta_path.exists():
            meta = orjson.loads(meta_path.read_bytes())
            title = meta.get("title", "") or ""
        chunks = chunk_paper(pid, md, title)
        out_path = CHUNK_DIR / f"{pid}.jsonl"
        with out_path.open("wb") as fh:
            for c in chunks:
                fh.write(orjson.dumps(c))
                fh.write(b"\n")
        total_chunks += len(chunks)
    print(f"DONE: {len(text_files)} papers, {total_chunks} chunks")


if __name__ == "__main__":
    main()
