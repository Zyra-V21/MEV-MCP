"""Enrich raw arXiv-style topics with a curated MEV-focused taxonomy.

The CSV ships tags such as `cs.GT`, `q-fin.TR`, `arXiv` that are useful as
domain hints but not MEV-specific. This script computes per-paper topic
labels from a controlled vocabulary using:

  1. arXiv category mapping (cs.GT -> game-theory, q-fin.TR -> trading-microstructure, …)
  2. Keyword/regex hits on the title + abstract + first 4 KB of the markdown body.

Result is written back to:
  - per-paper meta JSON (`topics_curated` field, original `topics` preserved)
  - SQLite `papers.topics_json`   (overwritten with curated list)
  - Tantivy index is rebuilt to refresh the `topics` facet.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import apsw
import orjson
import sqlite_vec
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import META_DIR, SQLITE_PATH, TEXT_DIR  # noqa: E402

# ---------------------------------------------------------------------------
#  Curated taxonomy
# ---------------------------------------------------------------------------

ARXIV_MAP: dict[str, str] = {
    "cs.GT": "game-theory",
    "cs.CR": "cryptography",
    "cs.DC": "distributed-systems",
    "cs.LG": "machine-learning",
    "cs.AI": "ai-agents",
    "cs.DS": "algorithms",
    "cs.MA": "multi-agent",
    "cs.NI": "networking",
    "cs.CY": "ethics-policy",
    "cs.LO": "logic-formal",
    "cs.SE": "software-engineering",
    "cs.PL": "programming-languages",
    "cs.SI": "social-networks",
    "cs.CE": "comp-engineering",
    "cs.FL": "formal-languages",
    "q-fin.TR": "trading-microstructure",
    "q-fin.MF": "mathematical-finance",
    "q-fin.PM": "portfolio-management",
    "q-fin.PR": "asset-pricing",
    "q-fin.RM": "risk-management",
    "q-fin.GN": "general-finance",
    "q-fin.CP": "computational-finance",
    "q-fin.ST": "stat-finance",
    "q-fin.EC": "economics-finance",
    "econ.TH": "economic-theory",
    "econ.GN": "general-economics",
    "math.OC": "optimization",
    "math.DS": "dynamical-systems",
    "math.PR": "probability",
    "math.FA": "functional-analysis",
    "stat.ML": "stat-learning",
    "iacr": "cryptography",
    "Self-host": "self-hosted",
    "SSRN": "ssrn",
}

# Keyword → curated topic. Patterns matched case-insensitively against
# title + abstract + first 4 KB of markdown body. Multiple matches are kept.
KEYWORD_TOPICS: list[tuple[str, str]] = [
    # MEV core
    (r"\bmev\b|maximal extractable value|miner extractable value", "mev"),
    (r"sandwich attack|sandwiching|sandwich.*amm", "sandwich-attacks"),
    (r"front[\s-]?running|frontrun", "front-running"),
    (r"back[\s-]?running|backrun", "back-running"),
    (r"arbitrag", "arbitrage"),
    (r"liquidat", "liquidations"),
    (r"flash[\s-]?loan", "flash-loans"),
    (r"toxic flow|adverse selection", "adverse-selection"),
    # AMMs & DEX
    (r"\bamm\b|automated market maker|constant function market maker|cfmm", "amm"),
    (r"uniswap\s*v?3|concentrated liquidity", "concentrated-liquidity"),
    (r"impermanent loss|loss[-\s]versus[-\s]rebalancing|lvr", "lvr-impermanent-loss"),
    (r"\bdex\b|decentralized exchange", "dex"),
    (r"order ?book|clob", "orderbook"),
    (r"limit order|market microstructure", "market-microstructure"),
    # Auctions & mechanism design
    (r"auction\b|first-price|second-price|english auction|dutch auction", "auctions"),
    (r"mechanism design|incentive compat", "mechanism-design"),
    (r"vickrey|vcg", "vcg-auctions"),
    # Block building / PBS / builder market
    (r"proposer[-\s]builder separation|\bpbs\b", "pbs"),
    (r"builder market|block builder|block building", "block-building"),
    (r"flashbots|mev[-\s]boost|relay\b", "flashbots-mev-boost"),
    (r"private mempool|private order flow|dark pool", "private-mempools"),
    (r"transaction fee mechanism|\btfm\b|fee market design", "tfm"),
    (r"eip[-\s]?1559|base fee", "eip-1559"),
    (r"censorship[-\s]resist|ofac|sanctioned", "censorship-resistance"),
    (r"bundle|bundling", "bundles"),
    (r"multi[-\s]block mev|cross[-\s]block mev", "multi-block-mev"),
    (r"threshold[-\s]?encrypt|encrypted mempool|sealed[-\s]bid", "encrypted-mempool"),
    (r"reorg|reorganization", "reorgs"),
    # Intent / order flow
    (r"intent[-\s]based|intents?\b|solver", "intents"),
    (r"order flow|order routing", "order-flow"),
    # Layer 2 / scaling
    (r"\bl2\b|layer[-\s]?2|rollup|optimistic|zk[-\s]?rollup|validium|plasma", "l2-rollups"),
    (r"data availability|\bda\b layer", "data-availability"),
    (r"cross[-\s]chain|bridge|interoperab", "cross-chain"),
    # Consensus & networking
    (r"consensus|fork choice|reorg|stale block", "consensus"),
    (r"proof of stake|\bpos\b|proof[-\s]of[-\s]work|\bpow\b", "consensus"),
    (r"validator|stak(ing|er)", "staking-validators"),
    # ZK / proofs
    (r"zero[-\s]knowledge|\bzk\b|snark|stark|plonk|halo|groth16|kzg", "zk-proofs"),
    (r"zk[-\s]vm|zkrollup|zkevm|zero knowledge virtual machine", "zk-vm"),
    (r"folding|recursive proof|ivc|nova\b", "recursive-proofs"),
    # Smart contracts / security
    (r"smart contract|solidity|evm bytecode", "smart-contracts"),
    (r"vulnerab|exploit|attack vector|security audit", "security"),
    # Stablecoins / DeFi
    (r"stablecoin|peg(ged)?|usdc|dai\b|usdt", "stablecoins"),
    (r"lending|borrow.*protocol|aave|compound", "lending"),
    (r"governance|voting|dao\b", "governance"),
    # Empirical / measurement
    (r"empirical study|measurement|on[-\s]chain analysis|dataset", "empirical"),
    (r"forecast|deep learning|neural net", "ml-forecasting"),
    # Privacy
    (r"\btornado\b|mix(er|ing)|privacy[-\s]preserving|anonymous", "privacy"),
    # Specific protocols
    (r"ethereum", "ethereum"),
    (r"bitcoin", "bitcoin"),
    (r"solana", "solana"),
    (r"cosmos|tendermint|ibc\b", "cosmos"),
]


def normalize_curated(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pat, label in KEYWORD_TOPICS:
        if re.search(pat, text, re.IGNORECASE):
            if label not in seen:
                seen.add(label)
                found.append(label)
    return found


def map_arxiv_tags(raw_topics: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in raw_topics:
        t = t.strip()
        if not t:
            continue
        label = ARXIV_MAP.get(t)
        if not label:
            # Pass through if already curated-looking (no dot, lowercase tokens)
            if re.fullmatch(r"[a-z][a-z0-9-]+", t):
                label = t
            else:
                continue
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def enrich_paper(pid: str, meta: dict) -> list[str]:
    title = meta.get("title", "") or ""
    abstract = meta.get("abstract", "") or ""
    md_path = TEXT_DIR / f"{pid}.md"
    body_head = ""
    if md_path.exists():
        body_head = md_path.read_text(encoding="utf-8", errors="ignore")[:4096]
    blob = "\n".join([title, abstract, body_head])
    curated = normalize_curated(blob)
    arxiv_mapped = map_arxiv_tags(meta.get("topics", []) or [])
    # Merge preserving order, dedupe.
    merged: list[str] = []
    for t in curated + arxiv_mapped:
        if t not in merged:
            merged.append(t)
    if not merged:
        merged = ["uncategorized"]
    return merged


def main() -> None:
    # 1. Compute curated topics per paper from meta + markdown.
    meta_files = sorted(META_DIR.glob("*.json"))
    enriched: dict[str, list[str]] = {}
    for mp in tqdm(meta_files, desc="enrich"):
        meta = orjson.loads(mp.read_bytes())
        topics = enrich_paper(meta["id"], meta)
        meta["topics_original"] = meta.get("topics", [])
        meta["topics"] = topics
        mp.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        enriched[meta["id"]] = topics

    # 2. Update SQLite: overwrite papers.topics_json with curated lists.
    db = apsw.Connection(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cur = db.cursor()
    cur.execute("BEGIN")
    for pid, topics in enriched.items():
        cur.execute(
            "UPDATE papers SET topics_json = ?1 WHERE id = ?2",
            (orjson.dumps(topics).decode(), pid),
        )
    cur.execute("COMMIT")
    db.close()
    print(f"SQLite updated: {len(enriched)} papers")

    # 3. Tantivy: rebuild from scratch with refreshed topics.
    import subprocess

    rebuild = subprocess.run(
        ["uv", "run", "python", "-c", _REBUILD_TANTIVY_SNIPPET],
        cwd=str(Path(__file__).resolve().parents[1]),
        check=True,
    )
    print("Tantivy rebuild rc:", rebuild.returncode)


_REBUILD_TANTIVY_SNIPPET = """
import sys
from pathlib import Path
sys.path.insert(0, '.')
from ingest.common import TANTIVY_DIR
import shutil

# Reuse 05_build_index.build_tantivy()
import importlib.util
spec = importlib.util.spec_from_file_location('bi', 'ingest/05_build_index.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.build_tantivy()
"""


if __name__ == "__main__":
    main()
