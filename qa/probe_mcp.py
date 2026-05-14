"""QA: fire a battery of research queries at the MCP server and dump results."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "mcp-server/target/release/mev-mcp"
CORPUS = ROOT / "corpus"

QUERIES = [
    "sandwich attack detection and mitigation in AMM swaps",
    "proposer builder separation mev-boost auction",
    "loss versus rebalancing LVR for liquidity providers",
    "intent-based markets and solvers",
    "cross-chain MEV and bridge security",
    "concentrated liquidity Uniswap v3 fee tier optimal",
    "ZK-rollup data availability and validity proofs",
    "redistribution of MEV to users dynamic mechanism",
    "private mempool order flow auctions",
    "stale block rate fork incentives selfish mining MEV",
    "automated market maker constant function curves analysis",
    "flash loan exploit and security analysis",
]


def call(method: str, params: dict, req_id: int):
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})


def run_probes():
    requests = [call("initialize", {}, 0)]
    for i, q in enumerate(QUERIES, 1):
        requests.append(
            call(
                "tools/call",
                {"name": "search", "arguments": {"query": q, "k": 5, "mode": "hybrid"}},
                i,
            )
        )
    payload = "\n".join(requests) + "\n"

    t0 = time.time()
    p = subprocess.run(
        [str(BINARY), "--corpus", str(CORPUS)],
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.time() - t0

    results: dict[int, dict] = {}
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "id" in d and d.get("result") and d["id"] != 0:
            content = d["result"].get("content", [])
            if content and content[0].get("type") == "text":
                try:
                    results[d["id"]] = json.loads(content[0]["text"])
                except json.JSONDecodeError:
                    pass

    print(f"### Probe run ({len(QUERIES)} queries, {elapsed:.2f}s total)\n")
    for i, q in enumerate(QUERIES, 1):
        res = results.get(i, {})
        hits = res.get("hits", [])
        print(f"## Q{i}. {q}")
        if not hits:
            print("  (no hits)\n")
            continue
        for j, h in enumerate(hits, 1):
            bm = h.get("bm25")
            cs = h.get("cosine")
            bm_s = f"{bm:.1f}" if bm is not None else "-"
            cs_s = f"{cs:.3f}" if cs is not None else "-"
            print(
                f"  {j}. score={h['score']:.4f} bm25={bm_s} cos={cs_s}  {h['title'][:78]}"
            )
            section = (h.get("section") or "").strip()[:70]
            print(f"     § {section}")
            snippet = (h.get("snippet") or "").replace("\n", " ")[:240]
            print(f"     ¬ {snippet}")
        print()


if __name__ == "__main__":
    run_probes()
