# How to provide auth cookies for gated downloads

A handful of papers (mostly SSRN and ResearchGate) sit behind a login wall.
If you have a free account on each, you can export your **session cookies**
and the ingestion pipeline will use them to fetch those PDFs automatically.

> Cookies are as sensitive as your password. Do **not** paste them into chat —
> drop them as files into `corpus/auth/` (the directory is `chmod 700` and
> excluded from git).

## 1. Get the cookies

Pick one workflow:

### Option A — Browser extension (easiest)

Install one of:
- **Chrome**: [Get cookies.txt LOCALLY](https://chrome.google.com/webstore/detail/get-cookiestxt-locally/cclelndahbckbenkjhflpdbgdldlbecc)
- **Firefox**: [cookies.txt](https://addons.mozilla.org/firefox/addon/cookies-txt/)

Then:

1. Log into <https://www.ssrn.com/index.cfm/en/> with your SSRN account.
2. Click the extension → **Export** → save as `corpus/auth/ssrn-cookies.txt`.
3. Log into <https://www.researchgate.net> with your RG account.
4. Repeat → save as `corpus/auth/rg-cookies.txt`.

Format must be **Netscape** (the default for these extensions).

### Option B — From the browser devtools

`Application` → `Storage` → `Cookies` → select the domain → copy each
cookie as `name<TAB>value`. Build a Netscape-formatted text file manually.
Tedious; only do this if extensions aren't an option.

## 2. Drop the files and run

```bash
ls corpus/auth/
# ssrn-cookies.txt   rg-cookies.txt

uv run python ingest/01c_download_authed.py
```

The script will:
1. Read every cookies.txt in `corpus/auth/`.
2. Pick the matching gated rows from `manifest.json` (status=`failed`).
3. Re-attempt the downloads with those cookies + a realistic browser
   User-Agent + Referer.
4. Update `manifest.json` in place with success/failure per paper.
5. Print a summary.

Then re-run the rest of the pipeline:

```bash
uv run python ingest/02_extract.py
uv run python ingest/03_clean_chunk.py
uv run python ingest/04_embed.py
uv run python ingest/05_build_index.py
uv run python ingest/06_enrich_topics.py
uv run python ingest/07_citation_graph.py   # builds citations table
```

## 3. What about the rest (Nature / ScienceDirect / ACM)?

Those require **institutional access** (university VPN or a paid Elsevier/
Springer/ACM subscription). If you have institutional access, the same
cookies workflow works — just export cookies from those domains too and
save them as `nature-cookies.txt`, `acm-cookies.txt`, `sd-cookies.txt`.

If you don't, those papers stay marked `auth_required` in the manifest.
You can always download a PDF manually and drop it as
`corpus/pdfs/<paper_id>.pdf` (find the `paper_id` in the manifest); the
next `02_extract.py` run will pick it up.

## Cookie hygiene

- The `corpus/auth/` directory is `chmod 700` and listed in `.gitignore`.
- The downloader **never logs cookie values**, only the cookie *count* per
  domain.
- Cookies typically expire in 1–14 days; if a future run fails, re-export.
- If your account uses 2FA, cookies still work — they're issued post-2FA.
