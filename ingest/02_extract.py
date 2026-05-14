"""Extract text from downloaded PDFs into clean Markdown.

For each successful PDF in manifest.json:
  - Open with PyMuPDF (fitz).
  - Detect and strip recurring headers/footers.
  - Reconstruct paragraphs (dehyphenation, line joining).
  - Detect section headings via font-size heuristics; emit ## headings.
  - Save corpus/text/<id>.md and corpus/meta/<id>.json.

Marks PDFs with very low text density as `needs_ocr` in the manifest
so the downstream pipeline can skip them gracefully.
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # pymupdf
import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import (  # noqa: E402
    MANIFEST_PATH,
    META_DIR,
    PDF_DIR,
    TEXT_DIR,
    read_csv,
)

MIN_CHARS_PER_PAGE = 100  # below this -> needs_ocr
HEADING_RATIO = 1.30  # heading font must be >= body_size * ratio
HEADING_MAX_WORDS = 12


@dataclass
class Block:
    text: str
    font_size: float
    bold: bool
    page: int
    y_top: float


def _iter_blocks(doc) -> Iterable[Block]:
    for page_idx, page in enumerate(doc, start=1):
        d = page.get_text("dict")
        for blk in d.get("blocks", []):
            if blk.get("type") != 0:
                continue
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s["text"] for s in spans).strip()
                if not text:
                    continue
                sizes = [s["size"] for s in spans]
                bold = any("Bold" in (s.get("font") or "") or (s.get("flags", 0) & 16) for s in spans)
                yield Block(
                    text=text,
                    font_size=max(sizes),
                    bold=bold,
                    page=page_idx,
                    y_top=float(line["bbox"][1]),
                )


def _detect_recurring(lines: list[Block], n_pages: int) -> set[str]:
    """Find short lines (likely headers/footers) that repeat across many pages."""
    if n_pages < 4:
        return set()
    candidates: Counter[str] = Counter()
    page_seen: dict[str, set[int]] = {}
    for b in lines:
        t = b.text.strip()
        if 3 <= len(t) <= 120 and not t.endswith("."):
            page_seen.setdefault(t, set()).add(b.page)
    threshold = max(3, int(n_pages * 0.4))
    return {t for t, pages in page_seen.items() if len(pages) >= threshold}


_DEHYPHEN = re.compile(r"(\w)-\s*\n\s*(\w)")
_MULTISPACE = re.compile(r"[ \t]+")
_MULTI_NL = re.compile(r"\n{3,}")


def _clean_paragraph(text: str) -> str:
    text = _DEHYPHEN.sub(r"\1\2", text)
    text = text.replace("­", "")  # soft hyphen
    text = _MULTISPACE.sub(" ", text)
    text = _MULTI_NL.sub("\n\n", text)
    return text.strip()


_NUM_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+\S")
_ARXIV_BANNER = re.compile(r"^\s*arxiv\s*:\s*\d", re.IGNORECASE)
_MATH_NOISE = re.compile(r"[=≈≤≥∈∉∧∨∑∏∫√±·×÷⊆⊇⊂⊃∀∃∞→←↔⇒⇐⇔]")
_BAREMATH = re.compile(r"^[\d\W]+$")  # only digits/symbols, no letters
_URL_OR_DOI = re.compile(r"https?://|doi\.org|github\.com|@\S+\.\S+", re.IGNORECASE)


def _is_heading(blk: Block, body_size: float) -> bool:
    text = blk.text.strip()
    if not text:
        return False

    # ----- Hard rejects (regardless of font) ----------------------------
    if _ARXIV_BANNER.match(text):
        return False
    if _URL_OR_DOI.search(text):
        return False
    if _BAREMATH.match(text):
        return False
    if _MATH_NOISE.search(text):
        return False
    # Need at least 3 letters to be a meaningful heading.
    if sum(c.isalpha() for c in text) < 3:
        return False
    # Reject lines that are mostly non-letter content (>40% non-alpha-non-space).
    non_text = sum(1 for c in text if not (c.isalpha() or c.isspace()))
    if non_text / max(len(text), 1) > 0.40:
        return False

    is_numbered = bool(_NUM_HEADING.match(text))

    # For numbered headings, require ≥3 letters after the numbering prefix.
    if is_numbered:
        after = re.sub(r"^\d+(\.\d+)*\.?\s+", "", text)
        if sum(c.isalpha() for c in after) < 3:
            return False

    # ----- Visual / structural criteria --------------------------------
    bigger = blk.font_size >= body_size * HEADING_RATIO
    if not is_numbered and not bigger:
        return False

    words = text.split()
    if not words or len(words) > HEADING_MAX_WORDS:
        return False

    # Mid-sentence / wrap indicators.
    if text.endswith(("-", ",", ";", ":")) and not is_numbered:
        return False
    if text.endswith(".") and not is_numbered:
        return False
    last_word = words[-1].lower()
    if last_word in {"and", "or", "the", "a", "an", "of", "to", "in", "for", "with", "by", "from", "on", "at"}:
        return False

    if len(text) > 90:
        return False
    if len(text) < 4:
        return False
    return True


def extract_pdf(pdf_path: Path) -> tuple[str, dict]:
    doc = fitz.open(pdf_path)
    n_pages = doc.page_count
    blocks = list(_iter_blocks(doc))
    sizes = [b.font_size for b in blocks]
    body_size = statistics.median(sizes) if sizes else 10.0

    recurring = _detect_recurring(blocks, n_pages)

    # Group blocks into "paragraph buckets" by page + nearby y.
    md_lines: list[str] = []
    current_paragraph: list[str] = []

    def flush_paragraph():
        if current_paragraph:
            p = _clean_paragraph(" ".join(current_paragraph))
            if p:
                md_lines.append(p)
                md_lines.append("")
            current_paragraph.clear()

    abstract_text: str | None = None
    abstract_collecting = False

    for blk in blocks:
        if blk.text in recurring:
            continue
        if _is_heading(blk, body_size):
            flush_paragraph()
            md_lines.append(f"## {blk.text}")
            md_lines.append("")
            # Detect abstract section to capture as metadata.
            if re.match(r"^\s*abstract\b", blk.text, re.IGNORECASE):
                abstract_collecting = True
                abstract_text = ""
            else:
                abstract_collecting = False
            continue
        current_paragraph.append(blk.text)
        if abstract_collecting:
            abstract_text = ((abstract_text or "") + " " + blk.text).strip()
            if len(abstract_text) > 2500:
                abstract_collecting = False
    flush_paragraph()

    body_md = "\n".join(md_lines).strip() + "\n"
    n_chars = sum(len(b.text) for b in blocks)
    meta = {
        "n_pages": n_pages,
        "n_chars": n_chars,
        "chars_per_page": n_chars / max(n_pages, 1),
        "abstract": _clean_paragraph(abstract_text or "")[:2000],
        "needs_ocr": (n_chars / max(n_pages, 1)) < MIN_CHARS_PER_PAGE,
    }
    doc.close()
    return body_md, meta


def main() -> None:
    manifest = orjson.loads(MANIFEST_PATH.read_bytes())
    rows = {r.id: r for r in read_csv()}

    extracted = 0
    skipped_ocr = 0
    errors = 0

    for pid, entry in tqdm(list(manifest.items()), desc="extract"):
        if entry.get("status") != "ok":
            continue
        pdf_path = PDF_DIR / f"{pid}.pdf"
        text_path = TEXT_DIR / f"{pid}.md"
        meta_path = META_DIR / f"{pid}.json"
        if not pdf_path.exists():
            continue
        if text_path.exists() and meta_path.exists():
            extracted += 1
            continue

        try:
            md, meta = extract_pdf(pdf_path)
        except Exception as e:
            entry["extract_error"] = f"{type(e).__name__}: {e}"
            errors += 1
            continue

        row = rows.get(pid)
        full_meta = {
            "id": pid,
            "title": row.title if row else "",
            "authors": row.authors if row else "",
            "topics": row.topics if row else [],
            "release_date": row.release_date if row else "",
            "pdf_url": entry.get("url", ""),
            **meta,
        }

        if meta["needs_ocr"]:
            entry["needs_ocr"] = True
            skipped_ocr += 1

        text_path.write_text(md, encoding="utf-8")
        meta_path.write_bytes(orjson.dumps(full_meta, option=orjson.OPT_INDENT_2))
        entry["text_chars"] = len(md)
        extracted += 1

    MANIFEST_PATH.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"DONE: extracted={extracted} skipped_ocr_flagged={skipped_ocr} errors={errors}")


if __name__ == "__main__":
    main()
