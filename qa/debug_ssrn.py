"""Debug: drive Playwright manually on ONE SSRN abstract page and dump:
- Final URL
- Status
- Whether body looks like login wall / Akamai challenge / real abstract
- All cookies present in the context after navigation"""

import asyncio
import sys
from http.cookiejar import MozillaCookieJar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)

COOKIES_PATH = Path(__file__).resolve().parents[1] / "corpus/auth/ssrn-cookies.txt"


def load_cookies():
    mz = MozillaCookieJar(str(COOKIES_PATH))
    mz.load(ignore_discard=True, ignore_expires=True)
    out = []
    for c in mz:
        out.append({
            "name": c.name,
            "value": c.value,
            "domain": c.domain,
            "path": c.path or "/",
            "secure": bool(c.secure),
            "httpOnly": False,
            "expires": int(c.expires) if c.expires else -1,
        })
    return out


async def main():
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    cookies = load_cookies()
    print(f"Loaded {len(cookies)} cookies, domains: {sorted(set(c['domain'] for c in cookies))}")
    print("Cookie names:", [c['name'] for c in cookies])

    async with Stealth().use_async(async_playwright()) as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = await browser.new_context(
            user_agent=UA,
            viewport={"width": 1366, "height": 900},
            locale="en-US",
            accept_downloads=True,
        )
        await context.add_cookies(cookies)
        page = await context.new_page()
        url = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4377561"
        print(f"\nNavigating to: {url}")
        try:
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            print(f"Initial response status: {resp.status if resp else 'no response'}")
            print(f"Final URL after load: {page.url}")
            # Cloudflare Turnstile challenge needs more time + DOM stabilization.
            for i in range(8):
                await page.wait_for_timeout(3000)
                title = await page.title()
                if title != "Just a moment...":
                    print(f"  challenge cleared after ~{(i+1)*3}s, title={title!r}")
                    break
                print(f"  waiting... ({(i+1)*3}s) title still={title!r}")
            html = await page.content()
            title = await page.title()
            print(f"Page title: {title!r}")
            print(f"HTML length: {len(html)}")
            # Indicators
            lc = html.lower()
            indicators = {
                "akamai_challenge": "_abck" in html or "akamai" in lc,
                "login_wall": "sign in" in lc or "login" in lc and "ssrn-login" in lc,
                "abstract_visible": "abstract" in lc and ("download" in lc or "free download" in lc),
                "captcha": "captcha" in lc,
                "blocked": "access denied" in lc or "403 forbidden" in lc,
            }
            print("Indicators:", indicators)
            # Save HTML for inspection
            out = Path("/tmp/ssrn_debug.html")
            out.write_text(html)
            print(f"\nFull HTML saved to: {out}")

            # Try clicking Download
            try:
                btn = page.locator("a:has-text('Download')").first
                count = await page.locator("a:has-text('Download')").count()
                print(f"\n'Download' links found: {count}")
                if count > 0:
                    href = await btn.get_attribute("href")
                    print(f"First 'Download' href: {href}")
            except Exception as e:
                print(f"download locator error: {e}")

            # Check cookies in context after nav
            ctx_cookies = await context.cookies()
            print(f"\nContext now has {len(ctx_cookies)} cookies (initial: {len(cookies)})")
            for c in ctx_cookies:
                if c['name'] in ('SSRNUserID', 'SSRNSession', '_abck', 'bm_sz', 'ASP.NET_SessionId'):
                    print(f"  {c['name']}={c['value'][:30]}...  domain={c['domain']}")
        finally:
            await page.close()
            await context.close()
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
