"""Build a clickable HTML helper for the SSRN papers blocked by Cloudflare.

Produces:
  corpus/auth/ssrn-manual.html  — open this in your browser. Each row links to
                                  the SSRN abstract page and shows the exact
                                  filename to save the downloaded PDF as.
  corpus/pdfs/manual/           — drop the PDFs here (or rename them yourself).

After you've finished downloading, run:
    uv run python ingest/ssrn_manual_import.py
to copy them into corpus/pdfs/<paper_id>.pdf and re-run the pipeline.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import orjson

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import MANIFEST_PATH, PDF_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MANUAL_DIR = PDF_DIR / "manual"
HTML_PATH = ROOT / "corpus" / "auth" / "ssrn-manual.html"

SSRN_ABS_FROM_URL = re.compile(r"abstractid=(\d+)|abstract_id=(\d+)", re.IGNORECASE)


def abstract_url(url: str) -> str:
    m = SSRN_ABS_FROM_URL.search(url)
    aid = m.group(1) or m.group(2) if m else ""
    return (
        f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={aid}"
        if aid
        else url
    )


def main():
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)
    manifest = orjson.loads(MANIFEST_PATH.read_bytes())
    todo = [
        (pid, e)
        for pid, e in manifest.items()
        if e.get("status") == "failed" and "ssrn.com" in (e.get("url", "") or "")
    ]
    if not todo:
        print("No SSRN failed papers left.")
        return
    print(f"SSRN papers to download manually: {len(todo)}")

    rows = []
    for i, (pid, e) in enumerate(sorted(todo, key=lambda x: x[1].get("title", "")), 1):
        url = abstract_url(e["url"])
        title = e.get("title", "").replace("<", "&lt;")
        target_name = f"{pid}.pdf"
        rows.append(
            f"""
<tr>
  <td>{i}</td>
  <td><a href="{url}" target="_blank" rel="noopener">{title}</a></td>
  <td><code>{target_name}</code></td>
</tr>"""
        )

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>SSRN manual download helper</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ font-size: 1.4em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ text-align: left; padding: 0.45em 0.6em; border-bottom: 1px solid #eee; vertical-align: top; }}
  th {{ background: #f7f7f9; font-weight: 600; }}
  td code {{ background: #f3f3f5; padding: 0.05em 0.4em; border-radius: 3px; font-size: 0.85em; }}
  a {{ color: #1a5fb4; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .hint {{ background: #fff8e0; border: 1px solid #ffd066; padding: 0.8em 1em; border-radius: 6px; margin: 1em 0; }}
</style>
</head><body>
<h1>SSRN papers to download manually ({len(todo)})</h1>
<div class="hint">
<b>Why this exists:</b> SSRN's Cloudflare protection blocks automated downloads, even with valid session cookies.
<br><b>What to do:</b> click each title to open the abstract in a new tab, then click the "Download" button inside SSRN.
Save the downloaded PDF to <code>corpus/pdfs/manual/</code> using <b>any name</b> — the importer matches by the
title metadata embedded in the PDF.
<br><b>Then run:</b> <code>uv run python ingest/ssrn_manual_import.py</code>
</div>
<table>
<thead><tr><th>#</th><th>Title</th><th>Save as (recommended)</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>"""
    HTML_PATH.write_text(html)
    print(f"\nWrote {HTML_PATH}")
    print(f"Drop the PDFs into: {MANUAL_DIR}")
    print("Open the HTML in your browser:")
    print(f"  xdg-open {HTML_PATH}")
    print("\nWhen done, run:")
    print("  uv run python ingest/ssrn_manual_import.py")


if __name__ == "__main__":
    main()
