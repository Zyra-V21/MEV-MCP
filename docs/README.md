# MEV-MCP — detailed reference

Self-hosted, offline research base built from the **MEV.fyi Research Hub**
papers list. A Python ingestion pipeline downloads every PDF, normalises it
into LLM-friendly Markdown, chunks and embeds it, and a small Rust **MCP
server** exposes the indexed corpus over stdio with hybrid search,
cross-encoder rerank, citation graph and per-paper centroid kNN.

```
csv -> 01_download -> 02_extract -> 02b_arxiv -> 03_chunk -> 04_embed -> 05_index -> 06_topics -> 07_citations
                                                                                  |
                                                                                  v
                                                              corpus/index.sqlite + tantivy/
                                                                                  |
                                                                                  v
                                                                       mev-mcp server (Rust, stdio)
                                                                                  |
                                                                                  v
                                                                       Claude Code / any MCP client
```

## Layout

```
.
├── MEV.fyi Research Hub - Papers.csv   # source data
├── pyproject.toml                       # uv-managed Python deps
├── ingest/                              # offline pipeline (Python)
├── mcp-server/                          # MCP daemon (Rust)
├── qa/                                  # QA probes + audits
└── corpus/                              # produced artefacts (gitignored)
    ├── pdfs/                              # raw PDFs
    ├── text/<id>.md                       # clean markdown
    ├── meta/<id>.json                     # per-paper metadata
    ├── chunks/<id>.jsonl                  # semantic chunks
    ├── chunks/<id>.emb.npy                # BGE-small embeddings (L2-normalised)
    ├── index.sqlite                       # papers + chunks + sqlite-vec + citations
    ├── tantivy/                           # BM25 index
    └── manifest.json                      # download status per paper
```

## Quick start

Prerequisites: `uv` (Python), `cargo` (Rust ≥ 1.80), Chromium runtime libs.

```bash
# 1. Install dependencies
uv sync
uv run playwright install chromium

# 2. Run ingestion (one-time, ~10–25 min depending on network + CPU)
uv run python ingest/01_download.py         # httpx + Playwright -> corpus/pdfs/
uv run python ingest/01b_retry_failed.py    # smart per-host retry
uv run python ingest/01c_download_authed.py # optional: SSRN/RG with session cookies
uv run python ingest/02_extract.py          # pymupdf -> corpus/text/*.md + meta
uv run python ingest/02b_arxiv_enrich.py    # pull clean abstract/authors from arXiv API
uv run python ingest/03_clean_chunk.py      # semantic chunks -> corpus/chunks/*.jsonl
uv run python ingest/04_embed.py            # BGE-small embeddings (CPU, ~5 min)
uv run python ingest/05_build_index.py      # SQLite + sqlite-vec + Tantivy + paper centroids
uv run python ingest/06_enrich_topics.py    # curated MEV taxonomy on top of arXiv tags
uv run python ingest/07_citation_graph.py   # paper -> paper edges via arXiv-id / DOI matching

# 3. Build the MCP server
cargo build --manifest-path mcp-server/Cargo.toml --release

# 4. Smoke-test
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
  mcp-server/target/release/mev-mcp --corpus corpus
```

## Hybrid search + rerank

The `search` tool fuses BM25 (Tantivy) and semantic similarity
(BGE-small-en-v1.5 over sqlite-vec) using Reciprocal Rank Fusion. Toggle
`mode` between `"lex"`, `"sem"`, `"hybrid"` (default). Set `rerank=true`
to apply a BGE cross-encoder rerank on the top-25 fused hits (+3–6s on CPU,
significant precision boost on ambiguous queries).

Query expansion (`expand=true` default) injects MEV alias synonyms:
`MEV ↔ maximal extractable value`, `LVR ↔ loss-versus-rebalancing`,
`PBS ↔ proposer-builder separation`, `AMM ↔ CFMM ↔ constant function market
maker`, `intent ↔ solver`, …

## MCP tools

| Tool | Use |
|---|---|
| `search` | `{ query, k?, mode?, per_paper_cap?, rerank?, expand? }` → ranked chunks with match-centered snippet, paper context, BM25 + cosine scores. |
| `get_paper` | `{ paper_id, sections? }` → full markdown plus metadata. |
| `list_papers` | `{ topic?, since?, limit? }` → filtered catalogue. |
| `list_topics` | Topic facets and counts. |
| `cite` | BibTeX entry from `paper_id`. |
| `summarize_paper` | `{ paper_id }` → structured TL;DR: metadata + curated excerpts from intro/conclusion/discussion. |
| `find_related` | `{ paper_id, k? }` → semantically nearest papers via per-paper centroid embeddings. |
| `citations` | `{ paper_id, direction? }` → in-corpus citation edges (cites / cited_by / both). |

## Registering with Claude Code

### Project-scope (only when cwd is this repo or a subdirectory)

Create `.mcp.json` at the repo root:

```json
{
  "mcpServers": {
    "mev-mcp": {
      "command": "/absolute/path/to/MEV-MCP/mcp-server/target/release/mev-mcp",
      "args": [
        "--corpus",
        "/absolute/path/to/MEV-MCP/corpus"
      ]
    }
  }
}
```

### User-scope (available from any project directory)

```bash
claude mcp add mev-mcp --scope user -- \
  /absolute/path/to/MEV-MCP/mcp-server/target/release/mev-mcp \
  --corpus /absolute/path/to/MEV-MCP/corpus
```

Verify: `claude mcp list` should show `mev-mcp: ... ✓ Connected`.

## Topic taxonomy

The CSV ships raw arXiv categories (`cs.GT`, `q-fin.TR`, …). Step 06
enriches each paper with a curated MEV-focused vocabulary derived from
arXiv-tag mapping plus keyword/regex scanning of title + abstract + body
head. Labels include: `mev`, `sandwich-attacks`, `front-running`,
`back-running`, `arbitrage`, `flash-loans`, `amm`, `concentrated-liquidity`,
`lvr-impermanent-loss`, `pbs`, `block-building`, `flashbots-mev-boost`,
`private-mempools`, `intents`, `order-flow`, `l2-rollups`, `cross-chain`,
`zk-proofs`, `zk-vm`, `mechanism-design`, `auctions`, `stablecoins`,
`lending`, `governance`, `tfm`, `eip-1559`, `encrypted-mempool`,
`multi-block-mev`, `reorgs`, `censorship-resistance`, `ml-forecasting`,
`privacy`, and the underlying research domain (`game-theory`,
`cryptography`, `economic-theory`, …). The original arXiv tags are
preserved under `topics_original` in each meta JSON for traceability.

## Notes

- `corpus/manifest.json` records the status (`ok`, `failed`, `needs_ocr`, …)
  of every paper plus the error reason for skipped ones. SSRN / SEC.gov /
  ResearchGate links that require login show up as `failed` — see
  `AUTH_COOKIES.md` for the cookie-based recovery workflow.
- Embeddings are CPU-only; the BGE-small ONNX model (~30 MB) and the
  BGE-reranker-base model (~280 MB) are auto-downloaded by both Python
  (`fastembed`) and Rust (`fastembed`) on first use.
- Latency budget: ~50 ms / query default (hybrid + RRF), ~3–6 s / query
  with `rerank=true`. Warm Tantivy + mmap'd sqlite-vec.
- Pipeline is idempotent: re-run any step after dropping new PDFs to
  incorporate them. SHA256-based change detection in `manifest.json`.
