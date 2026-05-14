"""Shared helpers: paths, CSV parsing, paper-id generation, URL classification."""

from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

from slugify import slugify

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "MEV.fyi Research Hub - Papers.csv"
CORPUS = ROOT / "corpus"
PDF_DIR = CORPUS / "pdfs"
TEXT_DIR = CORPUS / "text"
META_DIR = CORPUS / "meta"
CHUNK_DIR = CORPUS / "chunks"
TANTIVY_DIR = CORPUS / "tantivy"
SQLITE_PATH = CORPUS / "index.sqlite"
MANIFEST_PATH = CORPUS / "manifest.json"

for d in (PDF_DIR, TEXT_DIR, META_DIR, CHUNK_DIR, TANTIVY_DIR):
    d.mkdir(parents=True, exist_ok=True)


DIRECT_HOSTS = {
    "arxiv.org",
    "eprint.iacr.org",
    "angeris.github.io",
    "cms.nil.foundation",
    "github.com",
    "raw.githubusercontent.com",
    "uniswap.org",
    "xenophonlabs.com",
    "people.eecs.berkeley.edu",
    "moallemi.com",
    "timroughgarden.org",
    "assets-global.website-files.com",
    "res.cloudinary.com",
}

PLAYWRIGHT_HOSTS = {
    "papers.ssrn.com",
    "www.sec.gov",
    "www.researchgate.net",
    "dl.acm.org",
    "www.sciencedirect.com",
    "www.nature.com",
    "core.ac.uk",
}


def classify_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if not host:
        return "unknown"
    if "arxiv.org" in host:
        return "arxiv"
    if "eprint.iacr.org" in host:
        return "iacr"
    if "ssrn.com" in host:
        return "ssrn"
    if "sec.gov" in host:
        return "sec"
    if "researchgate.net" in host:
        return "researchgate"
    if host in PLAYWRIGHT_HOSTS:
        return "gated"
    if host in DIRECT_HOSTS:
        return "direct"
    return "direct"  # default optimistic


def normalize_arxiv(url: str) -> str:
    """Ensure arXiv URL points to .pdf endpoint."""
    if "arxiv.org" not in url:
        return url
    # http://arxiv.org/pdf/2405.01329v2  -> keep as is
    # http://arxiv.org/abs/2405.01329    -> rewrite to /pdf/ ... .pdf
    m = re.search(r"arxiv\.org/(?:abs|pdf)/([\w.\-/]+?)(v\d+)?(?:\.pdf)?$", url)
    if not m:
        return url
    paper = m.group(1)
    version = m.group(2) or ""
    return f"https://arxiv.org/pdf/{paper}{version}.pdf"


def normalize_iacr(url: str) -> str:
    """https://eprint.iacr.org/2024/442 -> .../2024/442.pdf"""
    if "eprint.iacr.org" not in url:
        return url
    if url.endswith(".pdf"):
        return url
    return url.rstrip("/") + ".pdf"


def normalize_url(url: str) -> str:
    url = url.strip()
    url = normalize_arxiv(url)
    url = normalize_iacr(url)
    return url


def paper_id(title: str, pdf_url: str) -> str:
    slug = slugify(title, max_length=60, word_boundary=True)
    if not slug:
        slug = "untitled"
    h = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{h}"


@dataclass
class PaperRow:
    id: str
    title: str
    authors: str
    pdf_url: str
    topics: list[str] = field(default_factory=list)
    release_date: str = ""
    referrer: str = ""
    url_class: str = ""


_ARXIV_V = re.compile(r"v\d+(?=\.pdf)")


def _content_key(title: str, url: str) -> str:
    """Deduplication key that collapses arXiv v1/v2/etc. of the same paper."""
    slug = slugify(title.lower(), max_length=80, word_boundary=True)
    # For arXiv we drop the version suffix; for other URLs keep the full URL.
    if "arxiv.org" in url:
        canon = _ARXIV_V.sub("", url)
    else:
        canon = url
    return f"{slug}|{canon}"


def read_csv() -> Iterator[PaperRow]:
    """Yield unique PaperRows. arXiv v1/v2 of the same paper collapse to the
    latest version (last seen wins, since CSV is sorted newest-first)."""
    with CSV_PATH.open() as f:
        reader = csv.DictReader(f)
        rows_by_key: dict[str, PaperRow] = {}
        for row in reader:
            title = (row.get("Title") or "").strip()
            raw_url = (row.get("PDF link") or "").strip()
            if not title or not raw_url:
                continue
            url = normalize_url(raw_url)
            pid = paper_id(title, url)
            topics = [t.strip() for t in (row.get("Topics") or "").split(",") if t.strip()]
            pr = PaperRow(
                id=pid,
                title=title,
                authors=(row.get("Authors") or "").strip(),
                pdf_url=url,
                topics=topics,
                release_date=(row.get("Release date") or "").strip(),
                referrer=(row.get("Referrer") or "").strip(),
                url_class=classify_url(url),
            )
            key = _content_key(title, url)
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = pr
            else:
                # Prefer the row whose URL has the higher arXiv version, else newer date.
                def _ver(u: str) -> int:
                    m = re.search(r"v(\d+)(?=\.pdf)", u)
                    return int(m.group(1)) if m else 0
                if _ver(pr.pdf_url) > _ver(existing.pdf_url):
                    rows_by_key[key] = pr
                elif pr.release_date and pr.release_date > existing.release_date:
                    rows_by_key[key] = pr
        for pr in rows_by_key.values():
            yield pr


def is_pdf_bytes(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
