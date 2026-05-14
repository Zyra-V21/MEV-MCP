"""Import manually-downloaded SSRN PDFs into the corpus.

Walks corpus/pdfs/manual/, identifies each PDF's title via:
  1. The PDF document's `/Title` metadata (set by SSRN), OR
  2. The first ~600 chars of extracted text (as a fallback), OR
  3. The filename stem (last resort, useful if you renamed to <paper_id>.pdf already).

Matches each title to a `paper_id` in `manifest.json` whose status is `failed`
by computing the same slug used at CSV ingestion time and picking the closest
match by simple normalised-string overlap.

For every matched PDF:
  - Validates it's a real PDF (%PDF magic bytes).
  - Copies into corpus/pdfs/<paper_id>.pdf.
  - Updates manifest entry to status=ok / source=manual.
  - Removes the file from manual/ once copied.

After this, just re-run 02_extract.py … 07_citation_graph.py.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import fitz
import orjson
from slugify import slugify

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import MANIFEST_PATH, PDF_DIR, is_pdf_bytes  # noqa: E402

MANUAL_DIR = PDF_DIR / "manual"


def normalize(s: str) -> set[str]:
    s = re.sub(r"\W+", " ", s.lower())
    return {w for w in s.split() if len(w) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / max(len(a | b), 1)


def pdf_title(path: Path) -> str:
    try:
        doc = fitz.open(path)
        meta = doc.metadata or {}
        t = (meta.get("title") or "").strip()
        if len(t) >= 8:
            doc.close()
            return t
        # Fallback: first page text head.
        head = doc.load_page(0).get_text("text")[:800]
        doc.close()
        # Heuristic: title is usually the first non-empty line.
        for line in head.splitlines():
            line = line.strip()
            if 10 <= len(line) <= 200:
                return line
        return ""
    except Exception:
        return ""


def main():
    if not MANUAL_DIR.exists() or not any(MANUAL_DIR.glob("*.pdf")):
        print(f"No PDFs found in {MANUAL_DIR}.")
        print("Run `uv run python ingest/ssrn_manual_helper.py` first.")
        return
    manifest = orjson.loads(MANIFEST_PATH.read_bytes())
    candidates = {
        pid: (e.get("title", "") or "")
        for pid, e in manifest.items()
        if e.get("status") == "failed"
    }
    if not candidates:
        print("Manifest has no `failed` entries left to match.")
        return

    # Build a fast abstract_id → paper_id map from the manifest URLs.
    abs_re = re.compile(r"abstract(?:_)?id=(\d+)|SSRN_ID(\d+)", re.IGNORECASE)
    abstract_to_pid: dict[str, str] = {}
    for pid, e in manifest.items():
        if e.get("status") != "failed":
            continue
        url = e.get("url", "") or ""
        m = abs_re.search(url)
        if m:
            aid = m.group(1) or m.group(2)
            abstract_to_pid[aid] = pid

    print(f"Manual PDFs to import: {sum(1 for _ in MANUAL_DIR.glob('*.pdf'))}")
    print(f"Failed manifest entries to match against: {len(candidates)}")
    print(f"Abstract-id index built: {len(abstract_to_pid)} entries")

    imported = 0
    skipped = []
    for src in sorted(MANUAL_DIR.glob("*.pdf")):
        data = src.read_bytes()
        if len(data) < 8000 or not is_pdf_bytes(data):
            print(f"  ! {src.name}: not a real PDF — skipping")
            continue
        stem = src.stem
        pid = None
        score = 0.0
        # 1. Filename is already a paper_id?
        if stem in candidates:
            pid = stem
            score = 1.0
        # 2. Filename matches an SSRN abstract_id pattern (e.g. ssrn-4377561.pdf)?
        if pid is None:
            m = re.search(r"(\d{6,8})", stem)
            if m and m.group(1) in abstract_to_pid:
                pid = abstract_to_pid[m.group(1)]
                score = 0.99
        # 3. Fall back to PDF title matching.
        if pid is None:
            t = pdf_title(src)
            stem_slug = slugify(t or stem, max_length=80)
            t_norm = normalize(t or stem)
            best_pid, best_score = None, 0.0
            for cpid, title in candidates.items():
                s = jaccard(t_norm, normalize(title))
                slug_b = slugify(title, max_length=80)
                if stem_slug.startswith(slug_b[:30]) or slug_b.startswith(stem_slug[:30]):
                    s += 0.2
                if s > best_score:
                    best_score, best_pid = s, cpid
            pid, score = best_pid, best_score
            if score < 0.35:
                print(f"  ? {src.name}: no confident match (best={best_pid} score={score:.2f}); skipping")
                skipped.append((src.name, t or stem))
                continue
        target = PDF_DIR / f"{pid}.pdf"
        target.write_bytes(data)
        manifest[pid].update(
            status="ok",
            bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            source="manual",
        )
        manifest[pid].pop("error", None)
        src.unlink()
        print(f"  ✓ {src.name} → {pid}.pdf  (score={score:.2f})")
        imported += 1

    MANIFEST_PATH.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"\nImported {imported} papers.")
    if skipped:
        print(f"{len(skipped)} could not be matched — list:")
        for name, t in skipped:
            print(f"  - {name}: title-detected='{t[:60]}'")
        print("\nFix by renaming each to <paper_id>.pdf and re-running this script.")
    if imported:
        print("\nNow re-run:")
        print("  uv run python ingest/02_extract.py")
        print("  uv run python ingest/02b_arxiv_enrich.py")
        print("  uv run python ingest/03_clean_chunk.py")
        print("  uv run python ingest/04_embed.py")
        print("  uv run python ingest/05_build_index.py")
        print("  uv run python ingest/06_enrich_topics.py")
        print("  uv run python ingest/07_citation_graph.py")


if __name__ == "__main__":
    main()
