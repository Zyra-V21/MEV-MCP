"""Build a citation graph by scanning markdown for arXiv IDs and DOIs.

For each paper we read corpus/text/<paper_id>.md and:
  1. Collect every arXiv ID and DOI mentioned in the body.
  2. Map each match to a paper in our own corpus when possible.
  3. Insert an edge (src_paper_id, dst_paper_id, kind) into SQLite.

Two new tables go into corpus/index.sqlite:
  citations(src_paper, dst_paper, kind, src_count)
  paper_refs(paper_id, arxiv_id, doi)   -- raw mentions, including external

Self-citations (a paper's own arXiv ID showing up in its own banner or
running header) are excluded.

Run after 05_build_index.py. Re-running is idempotent — the tables are
dropped and rebuilt from scratch each time.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import apsw
import sqlite_vec
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import SQLITE_PATH, TEXT_DIR  # noqa: E402

# arXiv ID patterns:
#   2403.02525        new style (post-Apr 2007)
#   2105.02784v3      with version
#   math/0309136      old style
ARXIV_NEW = re.compile(r"(?<!\d)(\d{4}\.\d{4,5})(v\d+)?(?!\d)")
ARXIV_OLD = re.compile(r"(?:^|[\s\(\[/])([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?")
DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")

# Strip common trailing punctuation that the regex over-captures.
_DOI_TRIM = re.compile(r"[.,;:\)\]]+$")


def _norm_arxiv(s: str) -> str:
    """Return the arXiv ID stripped of any version suffix and lowercased."""
    return re.sub(r"v\d+$", "", s).lower()


def _norm_doi(s: str) -> str:
    return _DOI_TRIM.sub("", s).lower()


def load_paper_arxiv_index(cur) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Return (arxiv_id -> paper_id, doi -> paper_id, paper_id -> own_arxiv_id)."""
    arxiv_to_pid: dict[str, str] = {}
    doi_to_pid: dict[str, str] = {}
    pid_to_own: dict[str, str] = {}
    for pid, url in cur.execute("SELECT id, pdf_url FROM papers"):
        url = url or ""
        m = ARXIV_NEW.search(url) or ARXIV_OLD.search(url)
        if m:
            aid = _norm_arxiv(m.group(1))
            arxiv_to_pid[aid] = pid
            pid_to_own[pid] = aid
        m = DOI_RE.search(url)
        if m:
            doi = _norm_doi(m.group(1))
            doi_to_pid[doi] = pid
    return arxiv_to_pid, doi_to_pid, pid_to_own


def extract_mentions(md: str) -> tuple[set[str], set[str]]:
    arxiv_ids = set()
    for m in ARXIV_NEW.finditer(md):
        arxiv_ids.add(_norm_arxiv(m.group(1)))
    for m in ARXIV_OLD.finditer(md):
        arxiv_ids.add(_norm_arxiv(m.group(1)))
    dois = set()
    for m in DOI_RE.finditer(md):
        dois.add(_norm_doi(m.group(1)))
    return arxiv_ids, dois


def main() -> None:
    if not SQLITE_PATH.exists():
        sys.exit("corpus/index.sqlite missing — run 05_build_index.py first.")
    db = apsw.Connection(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cur = db.cursor()

    cur.execute("DROP TABLE IF EXISTS citations")
    cur.execute("DROP TABLE IF EXISTS paper_refs")
    cur.execute(
        """
        CREATE TABLE citations(
            src_paper TEXT NOT NULL,
            dst_paper TEXT NOT NULL,
            kind TEXT NOT NULL,        -- 'arxiv' | 'doi'
            src_count INTEGER NOT NULL,
            PRIMARY KEY(src_paper, dst_paper, kind),
            FOREIGN KEY(src_paper) REFERENCES papers(id),
            FOREIGN KEY(dst_paper) REFERENCES papers(id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE paper_refs(
            paper_id TEXT NOT NULL,
            arxiv_id TEXT,
            doi TEXT
        )
        """
    )
    cur.execute("CREATE INDEX idx_cites_src ON citations(src_paper)")
    cur.execute("CREATE INDEX idx_cites_dst ON citations(dst_paper)")

    arxiv_to_pid, doi_to_pid, pid_to_own = load_paper_arxiv_index(cur)

    edges: dict[tuple[str, str, str], int] = defaultdict(int)
    refs_rows: list[tuple[str, str | None, str | None]] = []
    n_md = 0

    md_files = sorted(TEXT_DIR.glob("*.md"))
    for mp in tqdm(md_files, desc="cite-scan"):
        pid = mp.stem
        md = mp.read_text(encoding="utf-8", errors="ignore")
        arxiv_ids, dois = extract_mentions(md)
        n_md += 1
        own = pid_to_own.get(pid, "")
        # Count occurrences for src_count
        arxiv_counts: Counter[str] = Counter(_norm_arxiv(m) for m in ARXIV_NEW.findall_unique(md) if False) if False else Counter()
        # Cheaper: count occurrences using finditer
        ac: Counter[str] = Counter()
        for m in ARXIV_NEW.finditer(md):
            ac[_norm_arxiv(m.group(1))] += 1
        for m in ARXIV_OLD.finditer(md):
            ac[_norm_arxiv(m.group(1))] += 1
        dc: Counter[str] = Counter(_norm_doi(m.group(1)) for m in DOI_RE.finditer(md))
        for aid in arxiv_ids:
            if aid == own:
                continue
            refs_rows.append((pid, aid, None))
            tgt = arxiv_to_pid.get(aid)
            if tgt and tgt != pid:
                edges[(pid, tgt, "arxiv")] += ac[aid]
        for doi in dois:
            refs_rows.append((pid, None, doi))
            tgt = doi_to_pid.get(doi)
            if tgt and tgt != pid:
                edges[(pid, tgt, "doi")] += dc[doi]

    cur.execute("BEGIN")
    cur.executemany("INSERT INTO paper_refs(paper_id, arxiv_id, doi) VALUES(?,?,?)", refs_rows)
    cur.executemany(
        "INSERT INTO citations(src_paper, dst_paper, kind, src_count) VALUES(?,?,?,?)",
        [(s, d, k, n) for (s, d, k), n in edges.items()],
    )
    cur.execute("COMMIT")
    db.close()

    print(f"scanned {n_md} markdown files")
    print(f"edges (internal paper↔paper): {len(edges)}")
    print(f"raw mentions captured:       {len(refs_rows)}")
    # Top cited inside the corpus.
    db2 = apsw.Connection(str(SQLITE_PATH))
    db2.enable_load_extension(True); sqlite_vec.load(db2); db2.enable_load_extension(False)
    top = list(db2.execute(
        "SELECT p.title, COUNT(*) AS n FROM citations c "
        "JOIN papers p ON p.id = c.dst_paper GROUP BY c.dst_paper "
        "ORDER BY n DESC LIMIT 10"
    ))
    print("\nMost-cited papers within the corpus:")
    for title, n in top:
        print(f"  {n:3d}  {title[:80]}")
    db2.close()


if __name__ == "__main__":
    main()
