"""QA: audit the curated-topic enrichment for false positives and coverage gaps."""

import json
import sys
from collections import Counter
from pathlib import Path

import apsw
import sqlite_vec

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "corpus/index.sqlite"


def main():
    db = apsw.Connection(str(SQL), apsw.SQLITE_OPEN_READONLY)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    cur = db.cursor()
    rows = list(cur.execute("SELECT id, title, topics_json, abstract FROM papers"))

    counts: Counter = Counter()
    for _id, _title, tjson, _abs in rows:
        for t in json.loads(tjson):
            counts[t] += 1

    print(f"### {len(rows)} papers, {len(counts)} distinct topics\n")
    print("Top 25 topics:")
    for t, c in counts.most_common(25):
        print(f"  {c:4d}  {t}")

    # ----- Spot-check: papers tagged 'bitcoin' (often incidental mention) ---
    print("\n--- 'bitcoin' tagged papers (likely false positives if not about BTC) ---")
    for _id, title, tjson, _abs in rows:
        if "bitcoin" in json.loads(tjson):
            print(f"  - {title[:80]}")

    # ----- Uncategorized papers ---
    print("\n--- 'uncategorized' papers ---")
    for _id, title, tjson, _abs in rows:
        if "uncategorized" in json.loads(tjson):
            print(f"  - {title[:80]}")

    # ----- Coverage gap: papers with title mentioning a topic but not tagged ---
    print("\n--- Coverage check: titles mentioning core MEV terms but not tagged ---")
    GAPS = [
        ("mev", "mev"),
        ("amm", "amm"),
        ("intent", "intents"),
        ("rollup", "l2-rollups"),
        ("bundle", "bundles"),
        ("LVR", "lvr-impermanent-loss"),
        ("solver", "intents"),
        ("sandwich", "sandwich-attacks"),
        ("flash loan", "flash-loans"),
    ]
    for key, expected_topic in GAPS:
        missing = []
        for _id, title, tjson, _abs in rows:
            if key.lower() in title.lower():
                topics = json.loads(tjson)
                if expected_topic not in topics:
                    missing.append(title[:80])
        if missing:
            print(f"\n  title has '{key}' but no '{expected_topic}' tag:")
            for t in missing[:8]:
                print(f"    - {t}")


if __name__ == "__main__":
    main()
