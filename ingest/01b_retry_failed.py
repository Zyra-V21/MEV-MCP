"""Targeted retry for papers that failed in 01_download.py.

Per-host strategies:
  - arxiv:        plain retry — most arxiv failures are transient timeouts.
  - github.com/blob: rewrite to raw.githubusercontent.com.
  - SEC.gov:      retry with browser UA and Referer.
  - SSRN:         load the abstract page first to seed cookies, then go to the
                  delivery URL with proper Referer.
  - direct (other): retry with browser UA + Referer = scheme://host/.

For gated hosts (ACM, Nature, ScienceDirect, ResearchGate) we record a
clear reason and skip — these require institutional access.
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
import orjson
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import (  # noqa: E402
    MANIFEST_PATH,
    PDF_DIR,
    is_pdf_bytes,
    sha256_hex,
)

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
MIN_PDF_BYTES = 10_000


def rewrite_url(url: str) -> str:
    # github.com/<u>/<r>/blob/<branch>/<path>  ->  raw.githubusercontent.com/<u>/<r>/<branch>/<path>
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/blob/(.+)$", url)
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return url


def referer_for(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}/"


async def try_direct(client: httpx.AsyncClient, url: str, referer: str | None = None) -> bytes | None:
    headers = {
        "User-Agent": UA,
        "Accept": "application/pdf,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if referer:
        headers["Referer"] = referer
    try:
        r = await client.get(url, headers=headers, follow_redirects=True, timeout=60.0)
        if r.status_code >= 400:
            return None
        data = r.content
        if len(data) >= MIN_PDF_BYTES and is_pdf_bytes(data):
            return data
        return None
    except (httpx.RequestError, httpx.HTTPStatusError):
        return None


async def try_ssrn(client: httpx.AsyncClient, url: str) -> bytes | None:
    """Load the abstract page first to set cookies, then fetch the delivery URL."""
    # Extract abstractid
    m = re.search(r"abstractid=(\d+)", url)
    if not m:
        return await try_direct(client, url, referer="https://papers.ssrn.com/")
    abs_id = m.group(1)
    abstract_url = f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abs_id}"
    headers = {
        "User-Agent": UA,
        "Accept": "text/html,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        await client.get(abstract_url, headers=headers, follow_redirects=True, timeout=45.0)
    except Exception:
        pass
    return await try_direct(client, url, referer=abstract_url)


GATED_HOSTS = {
    "dl.acm.org",
    "www.nature.com",
    "www.sciencedirect.com",
    "www.researchgate.net",
    "core.ac.uk",
}


def reason_for_gate(url: str) -> str | None:
    host = urlparse(url).netloc.lower()
    if host in GATED_HOSTS:
        return f"auth_required: {host} requires institutional access"
    return None


async def retry_one(client: httpx.AsyncClient, pid: str, entry: dict) -> tuple[bool, str | None]:
    url = entry["url"]
    rewritten = rewrite_url(url)
    if rewritten != url:
        entry["retry_url"] = rewritten
        url = rewritten

    if "ssrn.com" in url:
        data = await try_ssrn(client, url)
    else:
        data = await try_direct(client, url, referer=referer_for(url))

    if data is None:
        # Light second attempt with a different UA hint and ?download=true / direct .pdf
        if not url.endswith(".pdf"):
            data = await try_direct(client, url + (".pdf" if "?" not in url else ""), referer=referer_for(url))

    if data is None:
        gate = reason_for_gate(url)
        return False, gate or "retry failed"

    p = PDF_DIR / f"{pid}.pdf"
    p.write_bytes(data)
    entry.update(
        status="ok",
        bytes=len(data),
        sha256=sha256_hex(data),
        source="retry",
    )
    entry.pop("error", None)
    return True, None


async def main():
    manifest = orjson.loads(MANIFEST_PATH.read_bytes())
    todo = [(pid, e) for pid, e in manifest.items() if e.get("status") == "failed"]
    print(f"failed papers to retry: {len(todo)}")
    sem = asyncio.Semaphore(4)
    ok_n = 0
    skipped_gated = 0
    async with httpx.AsyncClient(timeout=60.0) as client:

        async def runner(pid, entry):
            nonlocal ok_n, skipped_gated
            async with sem:
                ok, reason = await retry_one(client, pid, entry)
                if ok:
                    ok_n += 1
                else:
                    entry["status"] = "failed"
                    entry["error"] = reason or entry.get("error", "")
                    if reason and reason.startswith("auth_required"):
                        skipped_gated += 1

        await atqdm.gather(*[runner(pid, e) for pid, e in todo], desc="retry")

    MANIFEST_PATH.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"retry recovered: {ok_n}/{len(todo)}  (gated/skipped: {skipped_gated})")


if __name__ == "__main__":
    asyncio.run(main())
