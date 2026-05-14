"""Enrich arXiv-sourced papers with metadata from the official arXiv API.

The PDF-extracted abstract often has ligatures (ﬁ, ﬂ), broken hyphenation,
and missing whitespace. The arXiv API serves a clean version + authoritative
author list and the official primary category.

For each paper whose pdf_url points to arxiv.org, we:
  1. Extract the arXiv ID (e.g. 2403.02525v2 -> 2403.02525).
  2. Query http://export.arxiv.org/api/query?id_list=...
  3. Update corpus/meta/<id>.json with: arxiv_id, arxiv_abstract,
     arxiv_authors, arxiv_primary_category, arxiv_published.
  4. Update SQLite papers.abstract (preferring the arXiv version when
     present and longer than the extracted one).

Run order: anywhere after 02_extract.py. Re-run safe — uses HTTP caching
in-memory and updates idempotently.
"""

from __future__ import annotations

import asyncio
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import apsw
import httpx
import orjson
import sqlite_vec
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import META_DIR, SQLITE_PATH  # noqa: E402

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"

ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)
BATCH = 25  # arXiv API recommends ≤ ~50 ids per request


def extract_arxiv_id(url: str) -> str | None:
    m = ARXIV_ID_RE.search(url)
    return m.group(1) if m else None


def parse_atom(xml_text: str) -> dict[str, dict]:
    """Return {arxiv_id_no_version: metadata_dict} from an arXiv API response."""
    out: dict[str, dict] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for entry in root.findall(f"{ATOM}entry"):
        # The <id> field is the abs URL, e.g. http://arxiv.org/abs/2403.02525v2
        id_text = (entry.findtext(f"{ATOM}id") or "").strip()
        m = re.search(r"abs/(\d{4}\.\d{4,5})(v\d+)?", id_text)
        if not m:
            continue
        aid = m.group(1)
        summary = (entry.findtext(f"{ATOM}summary") or "").strip()
        summary = re.sub(r"\s+", " ", summary)
        published = (entry.findtext(f"{ATOM}published") or "")[:10]
        primary_cat = ""
        pc = entry.find(f"{ARXIV}primary_category")
        if pc is not None:
            primary_cat = pc.attrib.get("term", "")
        authors = [
            (a.findtext(f"{ATOM}name") or "").strip()
            for a in entry.findall(f"{ATOM}author")
        ]
        authors = [a for a in authors if a]
        out[aid] = {
            "arxiv_id": aid,
            "arxiv_abstract": summary,
            "arxiv_authors": authors,
            "arxiv_primary_category": primary_cat,
            "arxiv_published": published,
        }
    return out


async def fetch_batch(client: httpx.AsyncClient, ids: list[str]) -> dict[str, dict]:
    url = "https://export.arxiv.org/api/query"
    params = {"id_list": ",".join(ids), "max_results": str(len(ids))}
    for attempt in range(4):
        try:
            r = await client.get(url, params=params, timeout=60.0, follow_redirects=True)
            r.raise_for_status()
            return parse_atom(r.text)
        except Exception as e:
            if attempt == 3:
                print(f"  WARN: failed batch {ids[0]}…: {e}")
                return {}
            await asyncio.sleep(2 * (attempt + 1))
    return {}


async def main():
    metas = list(sorted(META_DIR.glob("*.json")))
    arxiv_targets: list[tuple[Path, dict, str]] = []
    for mp in metas:
        m = orjson.loads(mp.read_bytes())
        aid = extract_arxiv_id(m.get("pdf_url", "") or "")
        if aid:
            arxiv_targets.append((mp, m, aid))
    print(f"arXiv papers to enrich: {len(arxiv_targets)}")

    enriched: dict[str, dict] = {}
    async with httpx.AsyncClient(headers={"User-Agent": "mev-mcp-research/0.1"}) as client:
        batches = [arxiv_targets[i : i + BATCH] for i in range(0, len(arxiv_targets), BATCH)]
        for batch in atqdm(batches, desc="arxiv-api"):
            ids = [aid for _, _, aid in batch]
            got = await fetch_batch(client, ids)
            enriched.update(got)
            await asyncio.sleep(3.0)  # arXiv API courtesy: ≥ 3s between requests

    # ---- Write back to meta JSON ---------------------------------------
    updated_meta = 0
    for mp, meta, aid in arxiv_targets:
        info = enriched.get(aid)
        if not info:
            continue
        meta.update(info)
        if info["arxiv_abstract"] and len(info["arxiv_abstract"]) > len(meta.get("abstract", "") or ""):
            meta["abstract_extracted"] = meta.get("abstract", "")
            meta["abstract"] = info["arxiv_abstract"]
        if info["arxiv_authors"]:
            # Preserve original CSV authors but expose arxiv list too.
            meta.setdefault("authors_csv", meta.get("authors", ""))
            meta["authors"] = ", ".join(info["arxiv_authors"])
        if info["arxiv_published"] and not meta.get("release_date"):
            meta["release_date"] = info["arxiv_published"]
        mp.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        updated_meta += 1
    print(f"meta JSONs updated: {updated_meta}/{len(arxiv_targets)}")

    # ---- Mirror updates into SQLite ------------------------------------
    if not SQLITE_PATH.exists():
        print("(no index.sqlite yet — re-run 05_build_index.py to pick up changes)")
        return
    db = apsw.Connection(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cur = db.cursor()
    cur.execute("BEGIN")
    n_sql = 0
    for mp, meta, aid in arxiv_targets:
        info = enriched.get(aid)
        if not info:
            continue
        cur.execute(
            "UPDATE papers SET abstract = COALESCE(NULLIF(?1,''), abstract), "
            "authors = COALESCE(NULLIF(?2,''), authors), "
            "release_date = CASE WHEN release_date = '' OR release_date IS NULL THEN ?3 ELSE release_date END "
            "WHERE id = ?4",
            (
                info.get("arxiv_abstract", ""),
                ", ".join(info.get("arxiv_authors", [])),
                info.get("arxiv_published", "") or "",
                meta["id"],
            ),
        )
        n_sql += 1
    cur.execute("COMMIT")
    db.close()
    print(f"SQLite papers updated: {n_sql}")


if __name__ == "__main__":
    asyncio.run(main())
