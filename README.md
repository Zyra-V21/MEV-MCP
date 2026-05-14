# MEV-MCP

> A local **Model Context Protocol** server that turns the entire
> [MEV.fyi Research Hub](https://mev.fyi/) into a hybrid-search,
> reranker-backed, citation-aware research base any Claude (or other
> MCP-compatible) agent can query on-demand.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

---

## What it does

Drop the [MEV.fyi Papers CSV](MEV.fyi%20Research%20Hub%20-%20Papers.csv) into the
repo, run six Python scripts, build one Rust binary — and you have an MCP
server that exposes **8 research tools** over a fully offline, sub-50 ms
hybrid (BM25 + semantic) index of every paper in MEV.fyi's curated
collection.

| Tool | What it does |
|---|---|
| `search` | Hybrid BM25 + cosine over BGE-small embeddings, fused with RRF. Optional cross-encoder rerank. MEV-alias query expansion. |
| `get_paper` | Full clean markdown + metadata for a paper, optionally section-filtered. |
| `list_papers` | Filter the catalogue by curated MEV topic or release date. |
| `list_topics` | The curated MEV vocabulary with paper counts. |
| `cite` | BibTeX entry for a paper. |
| `summarize_paper` | Structured TL;DR: title + abstract + key excerpts from intro/conclusion/discussion. |
| `find_related` | kNN over per-paper centroid embeddings — "more like this". |
| `citations` | Internal citation graph (cites / cited_by) built from arXiv-id and DOI regex matching across the corpus. |

## Why this stack

Hybrid Python + Rust, each used where it shines:

| Stage | Language | Why |
|---|---|---|
| Download + Playwright | Python | `playwright-python` + `playwright-stealth` are the most mature toolchain for browser-driven scrapes. |
| PDF extraction | Python | `pymupdf` binds to MuPDF (C) — already C-fast. |
| Embeddings (one-shot) | Python | `fastembed` is ONNX; identical speed to Rust at batch. |
| **MCP server** | **Rust** | Single static binary, **arrancs <50 ms**, sub-ms queries with `tantivy` + `sqlite-vec`, no Python in the hot path. |

## Architecture

```
CSV -> 01_download -> 02_extract -> 02b_arxiv_enrich -> 03_chunk -> 04_embed
                                                                       |
                                            05_build_index <-----------+
                                                  |
                                                  +--> 06_enrich_topics
                                                  +--> 07_citation_graph
                                                  |
                                                  v
                                  corpus/index.sqlite + corpus/tantivy/
                                                  |
                                                  v
                                           mev-mcp (Rust)
                                                  |
                                                  v
                                        Claude Code / any MCP client
```

## Install

Prerequisites: [`uv`](https://docs.astral.sh/uv/) (Python),
[`cargo`](https://www.rust-lang.org/tools/install) (Rust ≥ 1.80), Chromium runtime libs.

```bash
git clone https://github.com/<your-user>/MEV-MCP.git
cd MEV-MCP

# Python deps (uv creates and locks the venv)
uv sync
uv run playwright install chromium

# Rust binary (~30 MB stripped release)
cargo build --manifest-path mcp-server/Cargo.toml --release
```

## Build the corpus

Each step is **idempotent** — re-run after dropping a new paper to the CSV
or to `corpus/pdfs/` and only the new work happens.

```bash
uv run python ingest/01_download.py         # httpx + Playwright -> corpus/pdfs/
uv run python ingest/01b_retry_failed.py    # smart per-host retry for failures
uv run python ingest/01c_download_authed.py # optional, see docs/AUTH_COOKIES.md
uv run python ingest/02_extract.py          # pymupdf -> corpus/text/<id>.md + meta
uv run python ingest/02b_arxiv_enrich.py    # canonical abstracts/authors via arXiv API
uv run python ingest/03_clean_chunk.py      # semantic chunks -> corpus/chunks/<id>.jsonl
uv run python ingest/04_embed.py            # BGE-small embeddings (CPU, ~5 min)
uv run python ingest/05_build_index.py      # SQLite + sqlite-vec + Tantivy + paper centroids
uv run python ingest/06_enrich_topics.py    # curated MEV taxonomy on top of arXiv tags
uv run python ingest/07_citation_graph.py   # citation edges via arXiv-id / DOI regex
```

Expected output on a fresh run with the bundled MEV.fyi CSV:

```
Papers OK:       ~236 / 256  (gated SSRN/RG/Nature/ACM the rest)
Chunks:          ~8,500 (avg ~36/paper, p50 ~720 tokens)
Topics:          ~82 curated MEV labels
Citation edges:  ~340 internal paper↔paper
Corpus size:     ~330 MB total (mostly PDFs)
Binary:          ~30 MB stripped
```

## Register the MCP

### Project-scope (recommended)

Drop a `.mcp.json` at the repo root (paths must be absolute):

```json
{
  "mcpServers": {
    "mev-mcp": {
      "command": "/absolute/path/to/MEV-MCP/mcp-server/target/release/mev-mcp",
      "args": ["--corpus", "/absolute/path/to/MEV-MCP/corpus"]
    }
  }
}
```

A template `.mcp.json.example` is included.

### User-scope (all your projects can use it)

```bash
claude mcp add mev-mcp --scope user -- \
  /absolute/path/to/MEV-MCP/mcp-server/target/release/mev-mcp \
  --corpus /absolute/path/to/MEV-MCP/corpus
```

Verify:

```bash
claude mcp list
# mev-mcp: /path/to/mev-mcp --corpus /path/to/corpus - ✓ Connected
```

## Use it

In any Claude Code session where the MCP is registered:

> *Show me what the corpus says about LVR mitigation strategies for AMM LPs.*

The agent automatically picks `search`, optionally chains `summarize_paper`
or `find_related`, and cites paper IDs. Use the tool names directly for
explicit calls: `mcp__mev-mcp__search`, `mcp__mev-mcp__find_related`, etc.

Smoke test from the shell:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search","arguments":{"query":"sandwich attack mitigation","k":3}}}' | \
  mcp-server/target/release/mev-mcp --corpus corpus
```

## Auth for gated papers (SSRN, ResearchGate, …)

Some papers sit behind Cloudflare-protected logins. See
[docs/AUTH_COOKIES.md](docs/AUTH_COOKIES.md) for the cookie-based recovery
flow. For SSRN specifically Cloudflare blocks *all* automated paths even
with valid cookies — the repo bundles a manual-download helper:

```bash
uv run python ingest/ssrn_manual_helper.py   # generates corpus/auth/ssrn-manual.html
# open the HTML in your browser, click each title, save PDFs to corpus/pdfs/manual/
uv run python ingest/ssrn_manual_import.py   # auto-matches by abstract_id / title
```

## Stack

- **Python 3.11+** ingestion: `httpx`, `playwright`, `playwright-stealth`,
  `curl-cffi`, `pymupdf`, `tiktoken`, `fastembed`, `apsw`, `sqlite-vec`,
  `tantivy-py`, `orjson`, `slugify`.
- **Rust ≥ 1.80** server: `tokio`, `rusqlite` (bundled), `sqlite-vec`,
  `tantivy`, `fastembed`, `clap`, `serde`, `anyhow`.
- **Models** (auto-downloaded, ONNX, CPU): `BAAI/bge-small-en-v1.5` (384d,
  ~30 MB) for embeddings; `BAAI/bge-reranker-base` (~280 MB) for optional
  cross-encoder rerank.

## License

[MIT](LICENSE).

The pipeline imports [PyMuPDF](https://github.com/pymupdf/PyMuPDF), which is
licensed under AGPL-3.0. When you run the ingestion scripts that
`import fitz`, the combined work is subject to AGPL terms. If you need a
non-AGPL combined work, swap the extractor for `pypdfium2` (Apache-2.0) —
it requires <30 minutes of refactoring in `ingest/02_extract.py`.

Source data: paper titles and URLs come from the
[MEV.fyi Research Hub](https://mev.fyi/). All PDFs are downloaded from
their original publishers and remain subject to the rights of their
respective copyright holders.
