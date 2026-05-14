"""Re-attempt gated downloads using a real browser with session cookies.

SSRN, ResearchGate and similar sites run Akamai / Cloudflare bot detection
that 403s any httpx-style client even with valid login cookies. Solution:
launch Playwright Chromium, inject the user's session cookies, and navigate
to the abstract / delivery URLs like a normal user would.

The user exports Netscape-format cookies.txt from a logged-in browser into
corpus/auth/{ssrn,rg,nature,…}-cookies.txt. We pick the right cookies per
host and drive the browser to download the PDF.

See docs/AUTH_COOKIES.md for the export workflow.
"""

from __future__ import annotations

import asyncio
import re
import sys
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import urlparse

import orjson
from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ingest.common import (  # noqa: E402
    MANIFEST_PATH,
    PDF_DIR,
    is_pdf_bytes,
    sha256_hex,
)

ROOT = Path(__file__).resolve().parents[1]
AUTH_DIR = ROOT / "corpus" / "auth"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
MIN_PDF_BYTES = 10_000


def load_cookie_files() -> dict[str, list[dict]]:
    """Return {host_suffix: [{name,value,domain,path,expires,...}, ...]} for each .txt in auth/."""
    if not AUTH_DIR.exists():
        return {}
    out: dict[str, list[dict]] = {}
    for path in sorted(AUTH_DIR.glob("*.txt")):
        mz = MozillaCookieJar(str(path))
        try:
            mz.load(ignore_discard=True, ignore_expires=True)
        except Exception as e:
            print(f"  WARN: could not load {path.name}: {e}")
            continue
        cookies = []
        domains: set[str] = set()
        for c in mz:
            cookies.append(
                {
                    "name": c.name,
                    "value": c.value,
                    "domain": c.domain,
                    "path": c.path or "/",
                    "secure": bool(c.secure),
                    "httpOnly": False,
                    "expires": int(c.expires) if c.expires else -1,
                }
            )
            domains.add(c.domain.lstrip("."))
        for d in domains:
            out.setdefault(d, []).extend(cookies)
        print(f"  loaded {path.name}: {len(cookies)} cookies covering {sorted(domains)}")
    return out


def matching_cookies(cookies_by_host: dict[str, list[dict]], url: str) -> list[dict] | None:
    host = urlparse(url).netloc.lower()
    if host in cookies_by_host:
        return cookies_by_host[host]
    for d, jar in cookies_by_host.items():
        if host == d or host.endswith("." + d):
            return jar
    return None


SSRN_DELIVERY_RE = re.compile(r"abstractid=(\d+)", re.IGNORECASE)
MIN_PDF_BYTES_PW = 8_000


async def try_ssrn(context, url: str) -> bytes | None:
    """Drive a real Chromium tab past Akamai. Navigate the abstract page first
    so the user's authenticated cookies are validated and the bot challenge
    completes, then trigger the PDF download from the Delivery link."""
    m = SSRN_DELIVERY_RE.search(url)
    abs_id = m.group(1) if m else None
    abstract_url = (
        f"https://papers.ssrn.com/sol3/papers.cfm?abstract_id={abs_id}"
        if abs_id
        else "https://papers.ssrn.com/"
    )
    page = await context.new_page()
    captured: list[bytes] = []

    async def on_response(resp):
        try:
            ct = resp.headers.get("content-type", "")
            if "pdf" in ct.lower():
                body = await resp.body()
                if len(body) >= MIN_PDF_BYTES_PW and is_pdf_bytes(body):
                    captured.append(body)
        except Exception:
            pass

    page.on("response", on_response)
    try:
        await page.goto(abstract_url, wait_until="domcontentloaded", timeout=45_000)
        # Give Akamai's JS challenge time to settle, then move to the PDF URL.
        await page.wait_for_timeout(2500)
        try:
            async with page.expect_download(timeout=20_000) as dl_info:
                await page.goto(url, wait_until="load", timeout=45_000)
            download = await dl_info.value
            local = await download.path()
            if local:
                data = Path(local).read_bytes()
                if is_pdf_bytes(data):
                    return data
        except Exception:
            pass
        # Allow late responses + retry the delivery URL via fetch within the page.
        await page.wait_for_timeout(2000)
        if captured:
            return captured[0]
        # Last resort: download by issuing fetch() from the page (carries cookies).
        body = await page.evaluate(
            """async (u) => {
                const r = await fetch(u, { credentials: 'include' });
                if (!r.ok) return null;
                const buf = await r.arrayBuffer();
                return Array.from(new Uint8Array(buf));
            }""",
            url,
        )
        if body:
            data = bytes(body)
            if len(data) >= MIN_PDF_BYTES_PW and is_pdf_bytes(data):
                return data
    finally:
        await page.close()
    return None


async def try_researchgate(context, url: str) -> bytes | None:
    page = await context.new_page()
    captured: list[bytes] = []

    async def on_response(resp):
        try:
            ct = resp.headers.get("content-type", "")
            if "pdf" in ct.lower():
                body = await resp.body()
                if len(body) >= MIN_PDF_BYTES_PW and is_pdf_bytes(body):
                    captured.append(body)
        except Exception:
            pass

    page.on("response", on_response)
    try:
        await page.goto(url, wait_until="load", timeout=60_000)
        await page.wait_for_timeout(2500)
        # Try clicking the "Download" / "Download full-text PDF" button.
        try:
            await page.locator("text=Download full-text PDF").first.click(timeout=5000)
            await page.wait_for_timeout(4000)
        except Exception:
            pass
        if captured:
            return captured[0]
        # Look for direct PDF href in the rendered DOM.
        href = await page.evaluate(
            "() => { const a = Array.from(document.querySelectorAll('a[href]')).find(x => x.href.includes('.pdf')); return a ? a.href : null; }"
        )
        if href:
            body = await page.evaluate(
                """async (u) => {
                    const r = await fetch(u, { credentials: 'include' });
                    if (!r.ok) return null;
                    const buf = await r.arrayBuffer();
                    return Array.from(new Uint8Array(buf));
                }""",
                href,
            )
            if body:
                data = bytes(body)
                if len(data) >= MIN_PDF_BYTES_PW and is_pdf_bytes(data):
                    return data
    finally:
        await page.close()
    return None


async def try_generic(context, url: str) -> bytes | None:
    page = await context.new_page()
    captured: list[bytes] = []

    async def on_response(resp):
        try:
            ct = resp.headers.get("content-type", "")
            if "pdf" in ct.lower():
                body = await resp.body()
                if len(body) >= MIN_PDF_BYTES_PW and is_pdf_bytes(body):
                    captured.append(body)
        except Exception:
            pass

    page.on("response", on_response)
    try:
        try:
            await page.goto(url, wait_until="load", timeout=60_000)
        except Exception:
            pass
        await page.wait_for_timeout(2000)
    finally:
        await page.close()
    return captured[0] if captured else None


async def attempt(context, pid: str, entry: dict, cookies_by_host: dict[str, list[dict]]) -> bool:
    url = entry["url"]
    if not matching_cookies(cookies_by_host, url):
        return False
    host = urlparse(url).netloc.lower()
    try:
        if "ssrn" in host:
            data = await try_ssrn(context, url)
        elif "researchgate" in host:
            data = await try_researchgate(context, url)
        else:
            data = await try_generic(context, url)
    except Exception as e:
        entry["authed_error"] = f"{type(e).__name__}: {e}"
        return False
    if data is None:
        return False
    (PDF_DIR / f"{pid}.pdf").write_bytes(data)
    entry.update(
        status="ok",
        bytes=len(data),
        sha256=sha256_hex(data),
        source="authed_playwright",
    )
    entry.pop("error", None)
    entry.pop("authed_error", None)
    return True


async def main() -> None:
    cookies_by_host = load_cookie_files()
    if not cookies_by_host:
        print("No cookie files in corpus/auth/ — see docs/AUTH_COOKIES.md.")
        return
    print(f"Cookies cover hosts: {sorted(cookies_by_host.keys())}")

    manifest = orjson.loads(MANIFEST_PATH.read_bytes())
    todo: list[tuple[str, dict]] = [
        (pid, e) for pid, e in manifest.items() if e.get("status") == "failed"
        and matching_cookies(cookies_by_host, e.get("url", "") or "") is not None
    ]
    if not todo:
        print("Nothing to retry — manifest has no matching failures for the supplied cookies.")
        return
    print(f"Matched candidates: {len(todo)}")

    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    ok_n = 0
    async with Stealth().use_async(async_playwright()) as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-features=IsolateOrigins,site-per-process",
                "--no-sandbox",
            ],
        )
        try:
            context = await browser.new_context(
                user_agent=UA,
                accept_downloads=True,
                viewport={"width": 1366, "height": 900},
                java_script_enabled=True,
                locale="en-US",
            )
            # Inject cookies for every covered host.
            all_cookies: list[dict] = []
            seen_keys: set[tuple[str, str, str]] = set()
            for jar in cookies_by_host.values():
                for c in jar:
                    key = (c["name"], c["domain"], c["path"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_cookies.append(c)
            await context.add_cookies(all_cookies)
            print(f"injected {len(all_cookies)} unique cookies into Chromium context.")

            sem = asyncio.Semaphore(2)  # SSRN rate-limits, keep low

            async def runner(pid: str, entry: dict):
                nonlocal ok_n
                async with sem:
                    if await attempt(context, pid, entry, cookies_by_host):
                        ok_n += 1

            await atqdm.gather(*[runner(pid, e) for pid, e in todo], desc="authed")
            await context.close()
        finally:
            await browser.close()

    MANIFEST_PATH.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))
    print(f"authed retry recovered: {ok_n}/{len(todo)}")
    if ok_n:
        print("Run the rest of the pipeline (02_extract.py … 07_citation_graph.py) to incorporate.")


if __name__ == "__main__":
    asyncio.run(main())
