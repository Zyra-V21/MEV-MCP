"""Download all PDFs from the CSV.

Two routes:
  - Direct via httpx (arxiv/iacr/direct hosts) with retries and backoff.
  - Playwright Chromium fallback for gated hosts (ssrn/sec/researchgate/etc.)
    and as a last resort if direct fails with 403/406.

Output:
  - corpus/pdfs/<paper_id>.pdf
  - corpus/manifest.json   (status per paper)

Idempotent: skips papers whose PDF already exists and matches expected magic bytes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import httpx
import orjson
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import (  # noqa: E402
    MANIFEST_PATH,
    PDF_DIR,
    PaperRow,
    is_pdf_bytes,
    read_csv,
    sha256_hex,
)

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
DIRECT_HEADERS = {
    "User-Agent": UA,
    "Accept": "application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DIRECT_CONCURRENCY = 6
PLAYWRIGHT_CONCURRENCY = 2
DIRECT_TIMEOUT = 60.0
MIN_PDF_BYTES = 10_000


def manifest_load() -> dict:
    if MANIFEST_PATH.exists():
        return orjson.loads(MANIFEST_PATH.read_bytes())
    return {}


def manifest_save(m: dict) -> None:
    MANIFEST_PATH.write_bytes(orjson.dumps(m, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def pdf_path_for(paper_id: str) -> Path:
    return PDF_DIR / f"{paper_id}.pdf"


def already_ok(paper_id: str) -> bool:
    p = pdf_path_for(paper_id)
    if not p.exists() or p.stat().st_size < MIN_PDF_BYTES:
        return False
    with p.open("rb") as fh:
        return is_pdf_bytes(fh.read(5))


async def fetch_direct(client: httpx.AsyncClient, url: str) -> bytes:
    """Single direct attempt; raises on non-PDF or HTTP error."""
    r = await client.get(url, headers=DIRECT_HEADERS, follow_redirects=True, timeout=DIRECT_TIMEOUT)
    r.raise_for_status()
    data = r.content
    if len(data) < MIN_PDF_BYTES:
        raise ValueError(f"response too small ({len(data)} bytes)")
    if not is_pdf_bytes(data):
        ct = r.headers.get("content-type", "")
        raise ValueError(f"not a PDF (content-type={ct})")
    return data


async def download_direct(client: httpx.AsyncClient, row: PaperRow) -> tuple[Optional[bytes], Optional[str]]:
    """Direct download with simple retry. Returns (data, error)."""
    last_err: Optional[str] = None
    delays = [0.5, 1.5, 4.0]
    for attempt, delay in enumerate([0.0] + delays):
        if delay:
            await asyncio.sleep(delay)
        try:
            data = await fetch_direct(client, row.pdf_url)
            return data, None
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code}"
            # 403/404/410 unlikely to recover by retry alone -> escalate
            if e.response.status_code in (403, 404, 410, 451):
                break
        except (httpx.RequestError, ValueError) as e:
            last_err = f"{type(e).__name__}: {e}"
    return None, last_err


async def download_via_playwright(row: PaperRow, browser) -> tuple[Optional[bytes], Optional[str]]:
    """Use a real browser to navigate, follow JS-driven download, capture bytes."""
    from playwright.async_api import TimeoutError as PWTimeout

    context = await browser.new_context(user_agent=UA, accept_downloads=True)
    try:
        page = await context.new_page()
        # If the URL already points to a PDF, intercept the response.
        url = row.pdf_url

        # Strategy: try direct navigation; if the page renders the PDF, fetch via response.
        captured: list[bytes] = []

        async def _on_response(resp):
            ct = resp.headers.get("content-type", "")
            if "application/pdf" in ct.lower():
                try:
                    body = await resp.body()
                    if is_pdf_bytes(body) and len(body) >= MIN_PDF_BYTES:
                        captured.append(body)
                except Exception:
                    pass

        page.on("response", _on_response)

        try:
            # Look for an explicit download event too (SSRN delivery links).
            async with page.expect_download(timeout=10_000) as dl_info:
                await page.goto(url, wait_until="load", timeout=45_000)
            download = await dl_info.value
            path = await download.path()
            if path:
                data = Path(path).read_bytes()
                if is_pdf_bytes(data):
                    return data, None
        except PWTimeout:
            # No explicit download — rely on captured PDF response above.
            pass
        except Exception:
            pass

        # Allow late responses to land.
        await asyncio.sleep(1.5)
        if captured:
            return captured[0], None
        return None, "playwright: no pdf response captured"
    finally:
        await context.close()


async def run() -> None:
    rows = list(read_csv())
    manifest = manifest_load()

    todo_direct: list[PaperRow] = []
    todo_playwright: list[PaperRow] = []

    for row in rows:
        if already_ok(row.id):
            manifest[row.id] = {
                **manifest.get(row.id, {}),
                "id": row.id,
                "title": row.title,
                "url": row.pdf_url,
                "url_class": row.url_class,
                "status": "ok",
                "bytes": pdf_path_for(row.id).stat().st_size,
                "source": manifest.get(row.id, {}).get("source", "cached"),
            }
            continue
        if row.url_class in ("ssrn", "sec", "researchgate", "gated"):
            todo_playwright.append(row)
        else:
            todo_direct.append(row)

    print(f"Total: {len(rows)} | direct: {len(todo_direct)} | playwright: {len(todo_playwright)} | cached: {sum(1 for r in rows if already_ok(r.id))}")

    # ---- Direct phase --------------------------------------------------
    sem_direct = asyncio.Semaphore(DIRECT_CONCURRENCY)
    async with httpx.AsyncClient(timeout=DIRECT_TIMEOUT) as client:

        async def one_direct(row: PaperRow):
            async with sem_direct:
                data, err = await download_direct(client, row)
                if data is None:
                    manifest[row.id] = {
                        "id": row.id,
                        "title": row.title,
                        "url": row.pdf_url,
                        "url_class": row.url_class,
                        "status": "needs_playwright",
                        "error": err,
                    }
                    return False
                p = pdf_path_for(row.id)
                p.write_bytes(data)
                manifest[row.id] = {
                    "id": row.id,
                    "title": row.title,
                    "url": row.pdf_url,
                    "url_class": row.url_class,
                    "status": "ok",
                    "bytes": len(data),
                    "sha256": sha256_hex(data),
                    "source": "direct",
                }
                return True

        if todo_direct:
            await atqdm.gather(*[one_direct(r) for r in todo_direct], desc="direct")

    manifest_save(manifest)

    # ---- Playwright phase ---------------------------------------------
    fallback = [r for r in todo_direct if manifest.get(r.id, {}).get("status") == "needs_playwright"]
    playwright_queue = todo_playwright + fallback
    print(f"Playwright queue: {len(playwright_queue)} (incl. {len(fallback)} direct fallbacks)")

    if playwright_queue:
        from playwright.async_api import async_playwright

        sem_pw = asyncio.Semaphore(PLAYWRIGHT_CONCURRENCY)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:

                async def one_pw(row: PaperRow):
                    async with sem_pw:
                        try:
                            data, err = await download_via_playwright(row, browser)
                        except Exception as e:
                            data, err = None, f"playwright crash: {type(e).__name__}: {e}"
                        if data is None:
                            manifest[row.id] = {
                                "id": row.id,
                                "title": row.title,
                                "url": row.pdf_url,
                                "url_class": row.url_class,
                                "status": "failed",
                                "error": err,
                            }
                            return False
                        p = pdf_path_for(row.id)
                        p.write_bytes(data)
                        manifest[row.id] = {
                            "id": row.id,
                            "title": row.title,
                            "url": row.pdf_url,
                            "url_class": row.url_class,
                            "status": "ok",
                            "bytes": len(data),
                            "sha256": sha256_hex(data),
                            "source": "playwright",
                        }
                        return True

                await atqdm.gather(*[one_pw(r) for r in playwright_queue], desc="playwright")
            finally:
                await browser.close()

    manifest_save(manifest)

    # ---- Summary -------------------------------------------------------
    ok = sum(1 for v in manifest.values() if v.get("status") == "ok")
    failed = sum(1 for v in manifest.values() if v.get("status") == "failed")
    print(f"DONE: ok={ok} failed={failed} total={len(manifest)}")


if __name__ == "__main__":
    asyncio.run(run())
