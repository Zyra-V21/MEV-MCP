"""Build the dual index used by the MCP server.

  - SQLite (with sqlite-vec) at corpus/index.sqlite:
      papers(id, title, authors, topics_json, release_date, pdf_url, sha256,
             n_pages, abstract)
      chunks(rowid, paper_id, section, text, tokens)
      chunks_vec virtual table over chunks.rowid (cosine, 384 dims)

  - Tantivy at corpus/tantivy/ (index_format v7):
      schema: id (STRING|STORED), paper_id (STRING|FAST|STORED),
              section (TEXT|STORED), text (TEXT|STORED), topics (TEXT|STORED),
              date (TEXT|FAST|STORED), tokens (INT|STORED)
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import apsw
import numpy as np
import orjson
import sqlite_vec
import tantivy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import (  # noqa: E402
    CHUNK_DIR,
    META_DIR,
    SQLITE_PATH,
    TANTIVY_DIR,
    read_csv,
)

EMB_DIM = 384


def vec_blob(arr: np.ndarray) -> bytes:
    """Pack a 1-D float32 vector for sqlite-vec."""
    return struct.pack(f"{arr.shape[0]}f", *arr.tolist())


def build_sqlite() -> None:
    if SQLITE_PATH.exists():
        SQLITE_PATH.unlink()
    db = apsw.Connection(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    cur = db.cursor()
    cur.execute("PRAGMA journal_mode = WAL")
    cur.execute("PRAGMA synchronous = NORMAL")
    cur.execute(
        """
        CREATE TABLE papers(
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            topics_json TEXT,
            release_date TEXT,
            pdf_url TEXT,
            sha256 TEXT,
            n_pages INTEGER,
            abstract TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE chunks(
            rowid INTEGER PRIMARY KEY,
            paper_id TEXT NOT NULL,
            chunk_id TEXT NOT NULL UNIQUE,
            section TEXT,
            text TEXT NOT NULL,
            tokens INTEGER,
            FOREIGN KEY(paper_id) REFERENCES papers(id)
        )
        """
    )
    cur.execute("CREATE INDEX idx_chunks_paper ON chunks(paper_id)")
    cur.execute(
        f"CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{EMB_DIM}])"
    )
    # Per-paper centroid embeddings — used by find_related.
    cur.execute(
        f"CREATE VIRTUAL TABLE papers_vec USING vec0(paper_rowid integer primary key, embedding float[{EMB_DIM}])"
    )
    cur.execute(
        "CREATE TABLE paper_rowid_map(paper_id TEXT PRIMARY KEY, paper_rowid INTEGER UNIQUE NOT NULL)"
    )

    # ---- Populate papers from manifest + metadata ---------------------
    rows = {r.id: r for r in read_csv()}
    papers_inserted = 0
    for meta_path in tqdm(sorted(META_DIR.glob("*.json")), desc="papers"):
        meta = orjson.loads(meta_path.read_bytes())
        row = rows.get(meta["id"])
        cur.execute(
            """
            INSERT INTO papers(id, title, authors, topics_json, release_date,
                               pdf_url, sha256, n_pages, abstract)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                meta["id"],
                meta.get("title", ""),
                meta.get("authors", "") or (row.authors if row else ""),
                orjson.dumps(meta.get("topics", []) or (row.topics if row else [])).decode(),
                meta.get("release_date", "") or (row.release_date if row else ""),
                meta.get("pdf_url", ""),
                meta.get("sha256", ""),
                meta.get("n_pages", 0),
                meta.get("abstract", ""),
            ),
        )
        papers_inserted += 1

    # ---- Populate chunks + vectors ------------------------------------
    chunk_rowid = 0
    paper_centroids: dict[str, np.ndarray] = {}
    paper_counts: dict[str, int] = {}
    cur.execute("BEGIN")
    for jsonl in tqdm(sorted(CHUNK_DIR.glob("*.jsonl")), desc="chunks"):
        emb_path = jsonl.with_suffix(".emb.npy")
        if not emb_path.exists():
            continue
        emb = np.load(emb_path)
        chunks = []
        with jsonl.open("rb") as fh:
            for line in fh:
                if line.strip():
                    chunks.append(orjson.loads(line))
        if emb.shape[0] != len(chunks):
            print(f"  WARN: row count mismatch for {jsonl.name}")
            continue
        for i, c in enumerate(chunks):
            chunk_rowid += 1
            cur.execute(
                "INSERT INTO chunks(rowid, paper_id, chunk_id, section, text, tokens) VALUES(?,?,?,?,?,?)",
                (chunk_rowid, c["paper_id"], c["chunk_id"], c.get("section", ""), c["text"], c.get("tokens", 0)),
            )
            cur.execute(
                "INSERT INTO chunks_vec(rowid, embedding) VALUES(?, ?)",
                (chunk_rowid, vec_blob(emb[i])),
            )
            pid = c["paper_id"]
            if pid not in paper_centroids:
                paper_centroids[pid] = emb[i].astype(np.float64).copy()
                paper_counts[pid] = 1
            else:
                paper_centroids[pid] += emb[i].astype(np.float64)
                paper_counts[pid] += 1
    cur.execute("COMMIT")

    # ---- Per-paper centroid vectors (L2-normalised mean of chunks) -----
    cur.execute("BEGIN")
    for paper_rowid, (pid, vec) in enumerate(paper_centroids.items(), start=1):
        n = paper_counts[pid]
        v = vec / max(n, 1)
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
        v = v.astype(np.float32)
        cur.execute(
            "INSERT INTO paper_rowid_map(paper_id, paper_rowid) VALUES(?, ?)",
            (pid, paper_rowid),
        )
        cur.execute(
            "INSERT INTO papers_vec(paper_rowid, embedding) VALUES(?, ?)",
            (paper_rowid, vec_blob(v)),
        )
    cur.execute("COMMIT")
    cur.execute("VACUUM")
    db.close()
    print(f"sqlite: {papers_inserted} papers, {chunk_rowid} chunks  ->  {SQLITE_PATH}")


def build_tantivy() -> None:
    if TANTIVY_DIR.exists():
        shutil.rmtree(TANTIVY_DIR)
    TANTIVY_DIR.mkdir(parents=True, exist_ok=True)

    sb = tantivy.SchemaBuilder()
    sb.add_text_field("id", stored=True, tokenizer_name="raw")
    sb.add_text_field("paper_id", stored=True, fast=True, tokenizer_name="raw")
    sb.add_text_field("section", stored=True)
    sb.add_text_field("text", stored=True, tokenizer_name="en_stem")
    sb.add_text_field("topics", stored=True, fast=True)
    sb.add_text_field("date", stored=True, fast=True, tokenizer_name="raw")
    sb.add_text_field("title", stored=True)
    sb.add_integer_field("tokens", stored=True, indexed=True)
    schema = sb.build()

    index = tantivy.Index(schema, path=str(TANTIVY_DIR))
    writer = index.writer(heap_size=128 * 1024 * 1024, num_threads=2)

    # Load paper metadata into memory for quick lookups.
    papers_meta: dict[str, dict] = {}
    for mp in META_DIR.glob("*.json"):
        m = orjson.loads(mp.read_bytes())
        papers_meta[m["id"]] = m

    n_chunks = 0
    for jsonl in tqdm(sorted(CHUNK_DIR.glob("*.jsonl")), desc="tantivy"):
        with jsonl.open("rb") as fh:
            for line in fh:
                if not line.strip():
                    continue
                c = orjson.loads(line)
                pid = c["paper_id"]
                meta = papers_meta.get(pid, {})
                topics = " ".join(meta.get("topics", []) or [])
                doc = tantivy.Document()
                doc.add_text("id", c["chunk_id"])
                doc.add_text("paper_id", pid)
                doc.add_text("section", c.get("section", "") or "")
                doc.add_text("text", c["text"])
                doc.add_text("topics", topics)
                doc.add_text("date", meta.get("release_date", "") or "")
                doc.add_text("title", meta.get("title", "") or "")
                doc.add_integer("tokens", int(c.get("tokens", 0)))
                writer.add_document(doc)
                n_chunks += 1
    writer.commit()
    writer.wait_merging_threads()
    print(f"tantivy: {n_chunks} chunks  ->  {TANTIVY_DIR}")


def main() -> None:
    build_sqlite()
    build_tantivy()


if __name__ == "__main__":
    main()
