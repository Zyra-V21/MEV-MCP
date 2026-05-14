"""Enrich each paper with a MEV-faithful taxonomy.

This taxonomy was designed by reading the abstracts of all 236 papers in the
corpus and grouping concepts that actually appear in MEV/DeFi research. It is
NOT a generic blockchain taxonomy — labels are kept narrow so a researcher can
filter on the actual MEV phenomena, vectors, infrastructure, countermeasures
and adjacent disciplines that the corpus covers.

Per paper we produce three orthogonal classifications stored in the meta JSON:
  - `topics`     : MEV-relevant categorical labels (the only ones shown by the
                   MCP's `list_topics` / `list_papers` tools).
  - `domains`    : academic discipline labels derived from arXiv categories
                   (game-theory, economic-theory, …).
  - `platforms`  : blockchain platforms referenced (ethereum, bitcoin, …).

Topic assignment rules per regex:
  * Match in `title` or `abstract`                    → label assigned.
  * Match only in body head (first 8 KB) AND ≥ 2 hits → label assigned.
  * Otherwise: not assigned (suppresses incidental mentions).

If nothing applies, the paper gets the `uncategorized` topic so it is still
visible to facet queries.
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
from ingest.common import META_DIR, SQLITE_PATH, TEXT_DIR, read_csv  # noqa: E402

# ===========================================================================
#  KEYWORD_TOPICS — content-grounded MEV taxonomy.
# ===========================================================================

KEYWORD_TOPICS: list[tuple[str, str]] = [
    # ----- MEV core ------------------------------------------------------
    (r"\bmev\b|maximal[\s-]extractable[\s-]value|miner[\s-]extractable[\s-]value|blockchain[\s-]extractable[\s-]value|\bbev\b|extractable value", "mev"),
    (r"multi[-\s]block mev|multi-slot mev|cross[-\s]block mev|consecutive blocks?.{0,40}mev|momentum.{0,30}mev|secur(?:ing|e).{0,20}k[-\s]consecutive", "multi-block-mev"),
    (r"cross[-\s]domain mev|cross[-\s]chain mev|cross[-\s]layer mev|cross[-\s]rollup mev|multi[-\s]domain mev", "cross-domain-mev"),
    (r"\bsearcher\b|mev[\s-]searcher|searchers compete|searcher relay", "searcher"),
    (r"builder[-\s]searcher|integrated builder|vertical(?:ly)? integrat.{0,30}build", "builder-searcher-integration"),
    (r"\bbev\b|blockchain extractable value", "blockchain-extractable-value"),
    (r"intrinsic[-\s]extractable.value|time[-\s]extractable.value", "time-extractable-value"),

    # ----- MEV attack vectors --------------------------------------------
    (r"sandwich attack|sandwiching|sandwich.{0,30}(amm|cfmm|dex|swap|trade)", "sandwich-attacks"),
    (r"front[\s-]?running|frontrun", "front-running"),
    (r"back[\s-]?running|backrun", "back-running"),
    (r"\barbitrag", "arbitrage"),
    (r"non[-\s]atomic arbitrage|statistical arbitrage|cyclic arbitrage|cross[-\s]exchange arbitrage|cex.{0,20}dex.{0,20}arbitrag", "non-atomic-arbitrage"),
    (r"liquidation", "liquidations"),
    (r"flash[\s-]?loan.{0,40}(attack|exploit|vulnerab)|flash[\s-]?loan.based attack", "flash-loan-attacks"),
    (r"\bflash[\s-]?loan", "flash-loans"),
    (r"oracle manipulation|twap.{0,20}(attack|manipulation|exploit)|oracle attack|oracle exploit|price oracle.{0,20}(manipulat|exploit)", "oracle-manipulation"),
    (r"\bjit\b liquidity|just[-\s]in[-\s]time liquidity|jit attack", "jit-liquidity"),
    (r"nft.{0,20}(mint|drop)|mint sniping|sniping bot|nft mev|nft.{0,20}front[-\s]?run", "nft-mev"),
    (r"\bvampire attack\b|liquidity migration attack|protocol migration attack", "vampire-attacks"),
    (r"time[-\s]bandit|history[-\s]revision|undercut(?:ting)?.{0,15}(attack|block)|reorg.{0,15}attack|reorg.{0,15}mev", "time-bandit"),
    (r"double[-\s]spending", "double-spending"),
    (r"speculative.{0,15}denial[-\s]of[-\s]service|speculative.{0,15}dos|\bdos attack", "dos-attacks"),
    (r"rug[-\s]pull|scam token|fraudulent token|exit scam", "rug-pull-detection"),
    (r"bribe|bribery", "bribery"),

    # ----- MEV markets / infrastructure ----------------------------------
    (r"\bauction\b", "auctions"),
    (r"frequent batch auction|\bfba\b|batch auction|batched.{0,20}exchange", "frequent-batch-auctions"),
    (r"dutch auction", "dutch-auctions"),
    (r"sealed[-\s]bid|sealed bid auction", "sealed-bid-auctions"),
    (r"vickrey|\bvcg\b auction|second[-\s]price auction", "vcg-auctions"),
    (r"proposer[-\s]builder[\s-]separation|\bpbs\b", "pbs"),
    (r"flashbots|mev[-\s]boost|mev[-\s]share|\bsuave\b", "flashbots-mev-boost"),
    (r"\brelay\b|relayer|mev relay|trusted relay", "relays"),
    (r"priority gas auction|\bpga\b|priority fee auction", "priority-gas-auction"),
    (r"order[\s-]flow auction|\bofa\b|order[-\s]flow.{0,15}auction", "order-flow-auctions"),
    (r"order[\s-]flow\b|private order flow|exclusive order flow", "order-flow"),
    (r"private mempool|private transaction|private pool|dark pool|stealth mempool", "private-mempools"),
    (r"\bbundle\b|transaction bundle|bundle of transactions|bundling", "bundles"),
    (r"intent[-\s]based|\bintents?\b|\bsolver\b|cowswap|cow swap|uniswap.?x|uniswap.x", "intents"),
    (r"block build|block builder|builder market|builder centralization|builder concentration", "block-building"),
    (r"execution tickets?|execution payload right", "execution-tickets"),
    (r"timeboost|time[-\s]boost.{0,20}(auction|ordering)", "timeboost"),

    # ----- MEV countermeasures -------------------------------------------
    (r"threshold encrypt|encrypted mempool|encrypted transaction|ferveo|time[-\s]lock encrypt|delayed decrypt", "encrypted-mempool"),
    (r"commit[-\s]reveal|commit and reveal|two[-\s]phase commit|time capsule|hidden bid|hidden commitment", "commit-reveal"),
    (r"fair ordering|order[-\s]fair|order fairness|\baequitas\b|\bthemis\b|\bwendy\b|fair sequencing", "fair-ordering"),
    (r"bullshark|narwhal|autobahn|motorway|hotstuff|dag[-\s]bft|dag[-\s]based bft|byzantine atomic broadcast", "bft-ordering-protocols"),
    (r"verifiable sequencing|fair sequencing service|\bfss\b|sequencing rule", "verifiable-sequencing-rules"),
    (r"mev redistribution|mev sharing|kickback|redistribut.{0,15}(mev|extractable)|mev rebate", "mev-redistribution"),
    (r"censorship[-\s]resist|\bofac\b|sanctioned address|censorship attack", "censorship-resistance"),
    (r"\bpacc\b|private.{0,15}collateralizable|anonymous commitment", "private-commitments"),
    (r"differential privacy.{0,20}(cfmm|amm|dex)", "differential-privacy-defi"),

    # ----- DeFi venues where MEV happens ---------------------------------
    (r"\bamm\b|automated market maker", "amm"),
    (r"\bcfmm\b|constant[\s-]function[\s-]market[\s-]maker|constant[\s-]product[\s-]market", "cfmm"),
    (r"concentrated liquidity|uniswap.?v.?3|uniswap v3|tick[-\s]based liquidity|active liquidity", "concentrated-liquidity"),
    (r"\bdex\b|decentralized exchange|on[-\s]chain exchange", "dex"),
    (r"order[\s-]?book|\bclob\b|limit order book", "orderbook"),
    (r"prediction market|cost[-\s]function.{0,20}market", "prediction-markets"),
    (r"impermanent loss|loss[-\s]versus[-\s]rebalancing|\blvr\b|predictable loss", "lvr-impermanent-loss"),
    (r"liquidity provision|liquidity provider\b|\bLP\b strategy|\blp\b return|\blp\b profit|\blps\b ", "liquidity-provision"),
    (r"lending protocol|aave|compound\b|maker[ -]?dao|defi lending|borrow.{0,15}protocol|collateralized loan", "defi-lending"),
    (r"stablecoin|\bpeg\b|\busdc\b|\busdt\b|\bdai\b|\bfrax\b|depeg|maker.{0,10}stablecoin", "stablecoins"),
    (r"perpetual future|\bperp\b|perpetual contract|perpetual swap", "perpetuals"),
    (r"\boption\b pricing|option contract|derivative pricing", "derivatives"),
    (r"prograde routing|\bcfmm\b.{0,30}routing|swap routing|trade routing.{0,30}(dex|amm)", "swap-routing"),

    # ----- Cross-domain / scaling MEV ------------------------------------
    (r"\bl2\b|layer[-\s]?2\b|rollup|optimistic rollup|zk[-\s]?rollup|validium|plasma\b", "l2-rollups"),
    (r"fraud proof|validity proof|interactive dispute", "fraud-validity-proofs"),
    (r"cross[-\s]chain|\bbridge\b|interoperab|atomic swap|cross[-\s]chain.{0,20}(transfer|swap)", "cross-chain-bridges"),
    (r"data availability|\bDA\b layer|data availability sampling|\beip[-\s]?4844\b|\bblob\b posting|blob space|proto[-\s]?danksharding", "data-availability"),
    (r"shared sequencer|decentralized sequencer|sequencer market|rollup sequencer", "shared-sequencer"),

    # ----- Empirical / measurement studies --------------------------------
    (r"empirical (?:analysis|study|investigation)|measurement study|on[-\s]chain analysis|dark forest|quantif.{0,15}(mev|bev|extractable)|forensic", "empirical-mev"),
    (r"scam.{0,20}detect|rug.{0,20}detect|fraud.{0,20}detect", "rug-pull-detection"),

    # ----- Economics & fee markets ---------------------------------------
    (r"transaction fee mechanism|\btfm\b|fee market design|fee mechanism", "tfm"),
    (r"eip[-\s]?1559|base fee", "eip-1559"),
    (r"multi[-\s]dimensional fee|multi[-\s]resource fee|non[-\s]fungible resource", "multi-dimensional-fees"),
    (r"mechanism design|incentive compatib|strategyproof", "mechanism-design"),
    (r"stackelberg", "stackelberg-games"),

    # ----- Consensus-level MEV effects ------------------------------------
    (r"\breorg|reorganization attack|chain fork|forking attack|stale block", "reorgs"),
    (r"selfish mining|undetect.{0,20}mining|fork[-\s]aware mining|withholding attack", "selfish-mining"),
    (r"block reward|incentive scheme.{0,30}mining|miner reward|allocation rule.{0,20}mining", "block-rewards-economics"),
    (r"single[-\s]slot finalit|fast finalit", "single-slot-finality"),
    (r"permissionless consensus|sleepy.{0,20}consensus|ghost|casper", "consensus-protocols"),
    (r"proof[-\s]of[-\s]stake|\bpos\b consensus|stake[-\s]based consensus", "proof-of-stake"),
    (r"\bpow\b|proof[-\s]of[-\s]work", "proof-of-work"),

    # ----- Staking & validator strategy ----------------------------------
    (r"liquid staking|\blst\b|liquid stake|staking derivative", "liquid-staking"),
    (r"\bstaking\b|\bstaker\b|stake pool|staked ether|validator pool", "staking"),
    (r"validator strategy|proposer strategy|timing game|waiting game|delayed proposal|withhold.{0,20}block", "validator-timing-games"),

    # ----- Cryptography (MEV-relevant primitives) ------------------------
    (r"\bsnark\b|\bstark\b|plonk|groth16|\bkzg\b|polynomial commitment|zk[-\s]?proof|zero[-\s]knowledge proof", "zk-proofs"),
    (r"zk[-\s]vm|zk[-\s]evm|zero[-\s]knowledge virtual machine", "zk-vm"),
    (r"recursive proof|folding scheme|nova\b|incrementally verifiable", "recursive-proofs"),
    (r"verifiable delay function|\bvdf\b", "verifiable-delay-functions"),
    (r"threshold crypt|threshold signature|threshold encryption", "threshold-cryptography"),
    (r"witness encryption|laconic.{0,10}\bot\b", "witness-encryption"),
    (r"distributed randomness|randomness beacon|drand|bicorn", "randomness-beacons"),

    # ----- Privacy --------------------------------------------------------
    (r"\btornado\b|\bmix(er|ing)\b|privacy[-\s]preserving (?:transaction|protocol|exchange)|anonymous transaction|dark[-\s]?dao", "privacy-defi"),

    # ----- Governance & DAOs ---------------------------------------------
    (r"governance|\bdao\b|protocol voting|on[-\s]chain voting|delegated voting|voting bloc", "dao-governance"),

    # ----- Traditional market microstructure (cross-pollination) ---------
    (r"market microstructure|tick size|spread.{0,20}market|limit order placement|order placement", "market-microstructure"),
    (r"high[-\s]frequency trading|\bhft\b|latency arbitrage|latency racing|latency competition", "high-frequency-trading"),
    (r"equity market structure|regulation nms|market fragment|reg nms|sec rule|sec disclosure|sec filing", "regulation-market-structure"),
    (r"market making|market maker$|fx dealer|electronic trading|dealer.{0,15}quote", "electronic-trading"),

    # ----- Cooperative AI / multi-agent (MEV-adjacent) -------------------
    (r"cooperative ai|multi[-\s]agent reinforcement|program equilibri|commitment device|formal contract.{0,30}equilibrium|prisoner.s dilemma", "cooperative-ai"),

    # ----- Tooling / formal methods --------------------------------------
    (r"formal verification|model checking|symbolic execution|smt solver|theorem prove|neural network.{0,15}verif", "formal-verification"),
    (r"economic security.{0,30}smart contract|adaptive learning.{0,20}security|economic security analysis", "economic-security-analysis"),

    # ----- Foundational / MEV-adjacent research (catch-all for uncategorized) --
    (r"geospatial|geographic.{0,15}(distribution|concentration|centralization)|validator.{0,15}geographic|node distribution", "validator-decentralization"),
    (r"federated learning|incentivized federated|distributed ml|distributed machine learning", "decentralized-ml"),
    (r"mean field game|mfg\b|mean[-\s]field model", "mean-field-finance"),
    (r"continuous benchmarking|cryptography (?:rust )?library|cryptanalysis tool|crypto.{0,10}benchmark", "crypto-engineering"),
    (r"gas reservation|gas hedge|hedge.{0,15}gas|gas market|gas pricing", "gas-markets"),
    (r"critical infrastructure|illicit finance|defi.{0,15}polic|crypto policy|regulatory framework.{0,30}defi", "defi-policy"),
    (r"link prediction|graph neural network|gnn\b|bellman[-\s]ford", "graph-learning"),
    (r"tax policy|taxation|tax design|redistributive tax", "policy-mechanism-design"),
    (r"tail protection|trend convexity|tail risk|long.{0,10}investor", "quant-investing"),
    (r"price impact|impact model|market impact", "price-impact"),
    (r"logical clock|lamport.{0,10}clock|happened[-\s]before|ordering.{0,15}distributed|partial order.{0,15}event", "distributed-clocks"),
    (r"zk[-\s]?snark|\bsnark\b|zero[-\s]knowledge for ml|verifiable.{0,15}(ml|machine learning)|verifiable evaluation.{0,15}model", "zk-proofs"),
    (r"strong.{0,10}mediated equilibrium|mediated equilibrium", "mediated-equilibrium"),
]

# ===========================================================================
#  Academic disciplines — derived from arXiv categories, stored separately.
# ===========================================================================

DOMAIN_FROM_ARXIV: dict[str, str] = {
    "cs.GT": "game-theory",
    "cs.CR": "cryptography",
    "cs.DC": "distributed-systems",
    "cs.LG": "machine-learning",
    "cs.AI": "ai",
    "cs.DS": "algorithms",
    "cs.MA": "multi-agent-systems",
    "cs.NI": "networking",
    "cs.CY": "ethics-policy",
    "cs.LO": "logic",
    "cs.SE": "software-engineering",
    "cs.PL": "programming-languages",
    "cs.SI": "social-networks",
    "cs.CE": "computational-engineering",
    "cs.FL": "formal-languages",
    "q-fin.TR": "trading-microstructure",
    "q-fin.MF": "mathematical-finance",
    "q-fin.PM": "portfolio-management",
    "q-fin.PR": "asset-pricing",
    "q-fin.RM": "risk-management",
    "q-fin.GN": "general-finance",
    "q-fin.CP": "computational-finance",
    "q-fin.ST": "statistical-finance",
    "q-fin.EC": "economics-finance",
    "econ.TH": "economic-theory",
    "econ.GN": "general-economics",
    "math.OC": "optimization",
    "math.DS": "dynamical-systems",
    "math.PR": "probability",
    "math.FA": "functional-analysis",
    "stat.ML": "statistical-learning",
}

# ===========================================================================
#  Blockchain platforms — stored separately.
# ===========================================================================

PLATFORM_KEYWORDS: list[tuple[str, str]] = [
    (r"\bethereum\b|\bevm\b|eth mainnet|ethereum blockchain|ether\b", "ethereum"),
    (r"\bbitcoin\b|\bbtc\b blockchain|bitcoin network|bitcoin protocol", "bitcoin"),
    (r"\bsolana\b", "solana"),
    (r"\bcosmos\b|tendermint|cosmos sdk|\bibc\b", "cosmos"),
    (r"\bavalanche\b|\bavax\b", "avalanche"),
    (r"\bpolkadot\b|substrate.{0,10}runtime|parachain", "polkadot"),
    (r"\bnear\b protocol|near blockchain", "near"),
    (r"\bstarknet\b|starkware|stark net", "starknet"),
    (r"arbitrum", "arbitrum"),
    (r"\boptimism\b", "optimism"),
    (r"\bzksync\b", "zksync"),
    (r"\bpolygon\b", "polygon"),
    (r"\balgorand\b", "algorand"),
]

BODY_HEAD_BYTES = 8 * 1024
BODY_HIT_THRESHOLD = 2


def _hits_in(blob: str, pattern: str) -> int:
    return sum(1 for _ in re.finditer(pattern, blob, flags=re.IGNORECASE))


def derive_topics(title: str, abstract: str, body_head: str) -> list[str]:
    """Title|abstract match = strong (1 hit assigns). Body-only match needs ≥2 hits."""
    head_blob = f"{title}\n\n{abstract}"
    found: list[str] = []
    seen: set[str] = set()
    for pat, label in KEYWORD_TOPICS:
        if label in seen:
            continue
        if re.search(pat, head_blob, re.IGNORECASE):
            seen.add(label)
            found.append(label)
            continue
        if _hits_in(body_head, pat) >= BODY_HIT_THRESHOLD:
            seen.add(label)
            found.append(label)
    return found


def derive_platforms(blob: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for pat, label in PLATFORM_KEYWORDS:
        if re.search(pat, blob, re.IGNORECASE) and label not in seen:
            seen.add(label)
            out.append(label)
    return out


def derive_domains(raw_topics: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in raw_topics:
        d = DOMAIN_FROM_ARXIV.get((t or "").strip())
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def enrich_paper(pid: str, meta: dict, raw_csv_topics: list[str]) -> tuple[list[str], list[str], list[str]]:
    title = meta.get("title", "") or ""
    abstract = meta.get("abstract", "") or ""
    md_path = TEXT_DIR / f"{pid}.md"
    body_head = ""
    if md_path.exists():
        body_head = md_path.read_text(encoding="utf-8", errors="ignore")[:BODY_HEAD_BYTES]

    topics = derive_topics(title, abstract, body_head)
    domains = derive_domains(raw_csv_topics)
    platforms = derive_platforms(f"{title}\n{abstract}\n{body_head}")
    if not topics:
        topics = ["uncategorized"]
    return topics, domains, platforms


def main() -> None:
    # Map paper_id → raw arXiv-style topics from the source CSV (authoritative).
    csv_topics: dict[str, list[str]] = {row.id: row.topics for row in read_csv()}

    meta_files = sorted(META_DIR.glob("*.json"))
    enriched_topics: dict[str, list[str]] = {}
    for mp in tqdm(meta_files, desc="enrich"):
        meta = orjson.loads(mp.read_bytes())
        pid = meta["id"]
        raw = csv_topics.get(pid, [])
        meta["topics_original"] = raw  # always re-stamp from the CSV (authoritative)

        topics, domains, platforms = enrich_paper(pid, meta, raw)
        meta["topics"] = topics
        meta["domains"] = domains
        meta["platforms"] = platforms
        mp.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        enriched_topics[pid] = topics

    # ---- Mirror MEV topics into SQLite -------------------------------------
    db = apsw.Connection(str(SQLITE_PATH))
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cur = db.cursor()
    cur.execute("BEGIN")
    for pid, topics in enriched_topics.items():
        cur.execute(
            "UPDATE papers SET topics_json = ?1 WHERE id = ?2",
            (orjson.dumps(topics).decode(), pid),
        )
    cur.execute("COMMIT")
    db.close()
    print(f"SQLite updated: {len(enriched_topics)} papers")

    # ---- Rebuild Tantivy so its `topics` facet matches ----------------------
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
import importlib.util
spec = importlib.util.spec_from_file_location('bi', 'ingest/05_build_index.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.build_tantivy()
"""


if __name__ == "__main__":
    main()
