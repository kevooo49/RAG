#!/usr/bin/env python3
"""
HTML -> Markdown Crawler for RAG (AGH University).

Usage:
    python scrape_html.py --list              # sites defined in config/sites.yaml
    python scrape_html.py bg                  # crawl one site ("bg" = key under `sites:` in the YAML)
    python scrape_html.py bg --max-pages 50   # test on 50 pages
    python scrape_html.py bg --reset          # full re-crawl from scratch
    python scrape_html.py bg --config other.yaml

Single crawler logic + one entry per site in config/sites.yaml (selectors, exclusions).
New site = new entry in the YAML, not a new script and not a code change.
"""
import argparse
import hashlib
import html
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from playwright.sync_api import sync_playwright
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
import urllib3
from urllib3.util.retry import Retry

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from sites import load_or_exit # type: ignore

# ==========================================
# GLOBAL SETTINGS
# ==========================================

USER_AGENT = "AGH-RAG-Crawler/1.0 (+contact: kevinst@student.agh.edu.pl)"
DELAY = 0.3
TIMEOUT = (5, 20)
MAX_BYTES = 5 * 1024 * 1024
MAX_QUERY_VARIANTS = 15
SAVE_STATE_EVERY = 25
MAX_JUNK_CHARS = 3000

PARAMS_TO_REMOVE = {"fbclid", "gclid", "chash", "no_cache", "phpsessid", "sid", "jsessionid"}
PREFIXES_TO_REMOVE = ("utm_", "mc_", "pk_")
PARAMS_SKIP_URL = {"print", "sort", "order", "orderby", "sortby", "format", "output", "ical"}

PDF_EXTENSIONS = re.compile(r"\.pdf$", re.I)
BINARY_EXTENSIONS = re.compile(
    r"\.(docx?|xlsx?|pptx?|odt|ods|zip|rar|7z|gz|jpe?g|png|gif|svg|webp|bmp|ico|mp[34]|avi|mov|wav|"
    r"css|js|woff2?|ttf|eot|xml|json|rss|ics|epub|csv|txt|exe|dmg)$",
    re.I,
)

JUNK_UNCONDITIONAL = ["script", "style", "noscript", "template", "svg", "iframe", "canvas",
                       "object", "embed", "link", "meta"]
JUNK_CAUTIOUS = [
    "nav", "form", "button", "select", "textarea", "input",
    "[role=navigation]", "[role=search]", "[role=banner]",
    "[class*=breadcrumb]", "[class*=cookie]", "[id*=cookie]",
    "[class*=share]", "[class*=social]", "[class*=pagination]",
    "[class*=skip]", "[class*=carousel-control]",
]

JUNK_STRUCTURE = ["header", "footer", "aside"]

# ==========================================
# SITE PROFILES - live in config/sites.yaml (not in code)
# ==========================================
_SITES = None


def get_sites(config_path=None):
    """Load (once) the shared sites config. `config_path` overrides config/sites.yaml."""
    global _SITES
    if _SITES is None or config_path:
        _SITES = load_or_exit(config_path)
    return _SITES


# ==========================================
# HTTP SESSION
# ==========================================
def create_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "pl,en;q=0.5"})
    retry = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=("GET", "HEAD"),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


session = create_session()
log = logging.getLogger("crawler")


# ==========================================
# URLS
# ==========================================
def netloc_without_www(url):
    n = urlsplit(url).netloc.lower()
    n = re.sub(r":(80|443)$", "", n)
    return n[4:] if n.startswith("www.") else n


def is_junk_param(name):
    n = name.lower()
    return n in PARAMS_TO_REMOVE or n.startswith(PREFIXES_TO_REMOVE)


def normalize_url(url):
    """Standardizes URL: https, no #anchors, no tracking params, sorted query,
    no trailing '/' and no /index.html. Raises ValueError for invalid addresses."""
    p = urlsplit(url.strip())
    scheme = "https" if p.scheme.lower() in ("http", "https") else p.scheme.lower()
    netloc = re.sub(r":(80|443)$", "", p.netloc.lower())

    path = re.sub(r"/{2,}", "/", p.path)
    path = re.sub(r"/index\.(html?|php)$", "", path, flags=re.I)
    path = path.rstrip("/") or "/"

    pairs = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not is_junk_param(k)]
    pairs.sort()
    return urlunsplit((scheme, netloc, path, urlencode(pairs), ""))


def belongs_to_domain(url, main_domain):
    """Exact host comparison PLUS exceptions for WIMiR CDN and COK."""
    netloc = netloc_without_www(url)
    if netloc in ["cdn.imir.agh.edu.pl", "cok.agh.edu.pl", "www.cok.agh.edu.pl"]:
        return True
    return netloc == netloc_without_www(main_domain)


def skip_foreign_language(url, cfg):
    p = urlsplit(url)
    path = p.path.lower()
    for lang in cfg["foreign_language_prefixes"]:
        if path == lang or path.startswith(lang + "/"):
            return True
    q = p.query.lower()
    return "lang=" in q and "lang=pl" not in q


def is_excluded_path(url, cfg):
    path = urlsplit(url).path.lower().rstrip("/")
    for w in cfg["excluded_paths"]:
        w = w.lower().rstrip("/")
        if path == w or path.startswith(w + "/") or (w.endswith(":") and path.startswith(w)):
            return True
    return False


def evaluate_url(url, cfg):
    """'ok' | 'pdf' | rejection reason."""
    p = urlsplit(url)
    if p.scheme not in ("http", "https"):
        return "scheme"
    if not belongs_to_domain(url, cfg["start_url"]):
        return "foreign_domain"
    if skip_foreign_language(url, cfg):
        return "language"
    if is_excluded_path(url, cfg):
        return "excluded"
    if PDF_EXTENSIONS.search(p.path):
        return "pdf"
    if BINARY_EXTENSIONS.search(p.path):
        return "file"
    if {k.lower() for k, _ in parse_qsl(p.query)} & PARAMS_SKIP_URL:
        return "view_parameter"
    return "ok"


def generate_filename(url):
    p = urlsplit(url)
    slug = p.path.strip("/").replace("/", "__") or "_index"
    slug = re.sub(r"[^\w\-.]+", "_", slug)[:80].strip("_.") or "_index"
    return f"{slug}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}.md"


# ==========================================
# CLEANING HTML -> MARKDOWN
# ==========================================
def remove_elements(root, selectors, cautious):
    for sel in selectors:
        for el in root.select(sel):
            if getattr(el, "decomposed", False):
                continue
            if cautious and len(el.get_text(" ", strip=True)) > MAX_JUNK_CHARS:
                continue
            el.decompose()


def select_content(soup, cfg):
    """Returns (element, is_fallback_to_body)."""
    for sel in cfg["content_selectors"]:
        el = soup.select_one(sel)
        if el is not None and el.get_text(strip=True):
            return el, False
    return soup.body, True


def fix_links(root, base_url):
    """Relative -> absolute links (in a RAG chunk '/contact' is useless).
    Empty, #anchors and javascript: are replaced by text only; mailto:/tel: remain."""
    for a in root.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:")):
            a.unwrap()
        elif not href.startswith(("mailto:", "tel:")):
            a["href"] = urljoin(base_url, href)


def clean_text(t):
    t = t.replace("\u00ad", "")
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    result, previous = [], None
    for line in t.split("\n"):
        if line.strip() and line.strip() == previous:
            continue
        result.append(line)
        if line.strip():
            previous = line.strip()
    t = "\n".join(result)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def html_to_markdown(root):
    return clean_text(md(str(root), heading_style="ATX", strip=["img"], bullets="-"))


def extract_links(soup, base_url):
    """Links are gathered from the FULL page (including menu) BEFORE cleaning – the menu leads to subpages."""
    return [urljoin(base_url, a["href"].strip()) for a in soup.find_all("a", href=True)]


def extract_footer(soup, cfg, base_url):
    """Footer (address, phone, e-mail) is on every page – we save it ONCE, not in every document."""
    for sel in cfg["footer_selectors"]:
        el = soup.select_one(sel)
        if el is not None and el.get_text(strip=True):
            copy_soup = BeautifulSoup(str(el), "html.parser")
            remove_elements(copy_soup, JUNK_UNCONDITIONAL, cautious=False)
            fix_links(copy_soup, base_url)
            return html_to_markdown(copy_soup)
    return ""


def process_page(soup, cfg, url):
    """Returns dict: title, language, noindex, canonical, markdown. Modifies soup!"""
    html_tag = soup.find("html")
    language = ((html_tag.get("lang") or "").lower().split("-")[0]) if html_tag else ""
    robots_meta = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    noindex = bool(robots_meta and "noindex" in (robots_meta.get("content") or "").lower())
    canon_tag = soup.find("link", rel="canonical")
    canonical = canon_tag.get("href") if canon_tag else None
    title_tag = soup.title.get_text(strip=True) if soup.title else ""

    root, fallback = select_content(soup, cfg)
    if root is None:
        return {"markdown": "", "title": title_tag, "language": language, "noindex": noindex, "canonical": canonical}

    remove_elements(root, JUNK_UNCONDITIONAL, cautious=False)
    if fallback:
        remove_elements(root, JUNK_STRUCTURE, cautious=False)
    remove_elements(root, JUNK_CAUTIOUS + cfg["additional_junk_selectors"], cautious=True)
    fix_links(root, url)

    h1 = root.find("h1")
    title = (h1.get_text(" ", strip=True) if h1 else "") or title_tag or "No title"
    return {
        "markdown": html_to_markdown(root),
        "title": clean_text(title),
        "language": language,
        "noindex": noindex,
        "canonical": canonical,
    }


def build_frontmatter(meta):
    lines = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()] + ["---", "", ""]
    return "\n".join(lines)


def save_atomically(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ==========================================
# CRAWLER STATE
# ==========================================
class State:
    def __init__(self):
        self.visited = set()
        self.queue = deque()
        self.known = set()
        self.hashes = {}
        self.variants = defaultdict(set)
        self.pdfs = set()
        self.stats = Counter()

    def save(self, path, pdf_path):
        data = {
            "visited": sorted(self.visited),
            "queue": list(self.queue),
            "hashes": self.hashes,
            "variants": {k: sorted(v) for k, v in self.variants.items()},
            "pdfs": sorted(self.pdfs),
            "stats": dict(self.stats),
        }
        save_atomically(path, json.dumps(data, ensure_ascii=False))
        save_atomically(pdf_path, "\n".join(sorted(self.pdfs)) + "\n")

    @classmethod
    def load(cls, path):
        s = cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        s.visited = set(d["visited"])
        s.queue = deque((u, g) for u, g in d["queue"])
        s.hashes = d["hashes"]
        s.variants = defaultdict(set, {k: set(v) for k, v in d["variants"].items()})
        s.pdfs = set(d["pdfs"])
        s.stats = Counter(d["stats"])
        s.known = s.visited | {u for u, _ in s.queue}
        return s


def add_to_queue(state, link, depth, cfg, robots):
    try:
        link = normalize_url(link)
    except ValueError:
        return
    if link in state.known:
        return
    state.known.add(link)

    result = evaluate_url(link, cfg)
    if result == "pdf":
        state.pdfs.add(link)
        return
    if result != "ok":
        state.stats["skipped_" + result] += 1
        return
    if depth > cfg["max_depth"]:
        state.stats["skipped_depth"] += 1
        return
    if not robots.can_fetch(USER_AGENT, link):
        state.stats["skipped_robots"] += 1
        return
    p = urlsplit(link)
    if p.query:
        variants = state.variants[p.path]
        if p.query not in variants and len(variants) >= MAX_QUERY_VARIANTS:
            state.stats["skipped_query_trap"] += 1
            return
        variants.add(p.query)
    state.queue.append((link, depth))


# ==========================================
# ROBOTS.TXT AND SITEMAPS
# ==========================================
def load_robots(cfg):
    rp = RobotFileParser()
    try:
        r = session.get(urljoin(cfg["start_url"], "/robots.txt"), timeout=TIMEOUT)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            if r.status_code in (401, 403):
                log.warning("robots.txt returned %s – treating as no restrictions", r.status_code)
            rp.allow_all = True
    except requests.RequestException as e:
        log.warning("Failed to fetch robots.txt (%s) – no restrictions", e)
        rp.allow_all = True
    return rp


def load_sitemaps(urls, depth=0):
    result = []
    for url in urls:
        try:
            r = session.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        loc = [html.unescape(x) for x in re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text, flags=re.S)]
        if "<sitemapindex" in r.text and depth < 2:
            result += load_sitemaps(loc, depth + 1)
        else:
            result += loc
    return result


# ==========================================
# DOWNLOADING
# ==========================================

class DummyResponse:
    """Mock requests.Response object for Playwright."""
    def __init__(self, url, status_code=200):
        self.url = url
        self.status_code = status_code
        self.headers = {}

def download(url, requires_js=False):
    """Returns (res, bytes, None) or (None, None, reason)."""
    if requires_js:
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.set_extra_http_headers({"User-Agent": USER_AGENT, "Accept-Language": "pl,en;q=0.5"})
                
                response = page.goto(url, wait_until="networkidle", timeout=30000)
                
                if response and response.status >= 400:
                    browser.close()
                    return None, None, f"HTTP {response.status}"

                ct = response.headers.get("content-type", "").lower() if response else "text/html"
                if "html" not in ct:
                    browser.close()
                    return None, None, f"not HTML ({ct})"
                
                final_url = page.url
                html_content = page.content().encode("utf-8")
                browser.close()
                
                if len(html_content) > MAX_BYTES:
                    return None, None, "file too large"
                    
                return DummyResponse(final_url), html_content, None
        except Exception as e:
            return None, None, f"Playwright error: {e}"

    try:
        res = session.get(url, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        return None, None, f"network error: {e}"
    try:
        if res.status_code >= 400:
            return None, None, f"HTTP {res.status_code}"
        ct = res.headers.get("Content-Type", "").lower()
        if "html" not in ct:
            return None, None, f"not HTML ({ct or 'missing Content-Type'})"
        if int(res.headers.get("Content-Length") or 0) > MAX_BYTES:
            return None, None, "file too large"
        chunks, size = [], 0
        for chunk in res.iter_content(65536):
            size += len(chunk)
            if size > MAX_BYTES:
                return None, None, "file too large"
            chunks.append(chunk)
        return res, b"".join(chunks), None
    except requests.RequestException as e:
        return None, None, f"network error: {e}"
    finally:
        res.close()


def parse_html(content):
    try:
        return BeautifulSoup(content, "lxml")
    except Exception:
        return BeautifulSoup(content, "html.parser")


# ==========================================
# MAIN CRAWL PROCESS (BFS)
# ==========================================
def crawl(cfg, reset=False, max_pages=None):
    folder = cfg["folder"]
    meta_folder = folder + "_meta"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)
    state_path = os.path.join(meta_folder, "stan.json")
    pdf_path = os.path.join(meta_folder, "pdf_linki.txt")
    manifest_path = os.path.join(meta_folder, "manifest.jsonl")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(meta_folder, "crawler.log"), encoding="utf-8")],
    )

    robots = load_robots(cfg)
    delay = max(DELAY, float(robots.crawl_delay(USER_AGENT) or 0)) if robots.last_checked else DELAY

    if os.path.exists(state_path) and not reset:
        state = State.load(state_path)
        log.info("Resuming: %d visited, %d in queue", len(state.visited), len(state.queue))
    else:
        state = State()
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        start = normalize_url(cfg["start_url"])
        state.known.add(start)
        state.queue.append((start, 0))
        try:
            maps = robots.site_maps() or []
        except AttributeError:
            maps = []
        maps = maps or [urljoin(cfg["start_url"], "/sitemap.xml")]
        for u in load_sitemaps(maps):
            add_to_queue(state, u, 1, cfg, robots)
        log.info("Start: %d addresses in queue (after sitemaps)", len(state.queue))

    footer_saved = os.path.exists(os.path.join(folder, "_stopka.md")) and not reset
    processed_in_session = 0
    current = None
    now = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")

    log.info("🕷 Crawler starting: %s", cfg["start_url"])
    try:
        while state.queue:
            if max_pages and processed_in_session >= max_pages:
                log.info("Reached --max-pages=%d limit", max_pages)
                break
            url, depth = state.queue.popleft()
            current = (url, depth)
            if url in state.visited:
                continue
            state.visited.add(url)
            processed_in_session += 1

            res, content, reason = download(url, requires_js=cfg.get("requires_js", False))
            time.sleep(delay)
            if res is None:
                state.stats["errors_or_not_html"] += 1
                log.info("⏩ %s – %s", url, reason)
                continue

            try:
                final = normalize_url(res.url)
            except ValueError:
                continue
            if final != url:
                if final in state.visited or evaluate_url(final, cfg) != "ok":
                    state.stats["skipped_redirect"] += 1
                    continue
                state.visited.add(final)
                state.known.add(final)
                url = final

            soup = parse_html(content)

            if "historia" in url:
                anniversaries = soup.find(id="Rocznice")
                if anniversaries:
                    single = anniversaries.parent if anniversaries.parent and "mw-heading" in (anniversaries.parent.get("class") or []) else anniversaries
                    if single and single.parent and single.parent.name == "div":
                        single.parent.decompose()
                for hdr in soup.find_all(id=re.compile("^(Bibliografia|Źródła)", re.I)):
                    single = hdr.parent if hdr.parent and "mw-heading" in (hdr.parent.get("class") or []) else hdr
                    if single:
                        for sibling in single.find_next_siblings():
                            for a in sibling.find_all("a"):
                                a.unwrap()

            if "imir" in url and url.rstrip("/") == cfg["start_url"].rstrip("/"):
                hidden_zones = [
                    "https://imir.agh.edu.pl/pl/studenci/blankiety_dokumenty/blankiety",
                    "https://imir.agh.edu.pl/pl/dzialalnosc_naukowa/komunikaty_prodziekana_ds_nauki/",
                    "https://imir.agh.edu.pl/pl/studia/stacjonarne_i-st/",
                    "https://imir.agh.edu.pl/pl/wydzial/o_wydziale/",
                    "https://imir.agh.edu.pl/pl/studenci/blankiety_dokumenty/dokumenty/",
                    "https://imir.agh.edu.pl/pl/studenci/stypendia/stypendium_ministra/",
                    "https://imir.agh.edu.pl/pl/dzialalnosc_naukowa/doktoranci/",
                    "https://imir.agh.edu.pl/pl/dzialalnosc_naukowa/program_doktorat_wdrozeniowy/",
                    "https://imir.agh.edu.pl/pl/dzialalnosc_naukowa/kategoryzacja_mnisw_2017/",
                    "https://imir.agh.edu.pl/pl/dzialalnosc_naukowa/oferta_naukowo_badawcza/",
                    "https://imir.agh.edu.pl/pl/dzialalnosc_naukowa/projekty_wspolfinansowane_z_funduszy_ue/"
                ]
                for zone in hidden_zones:
                    log.info("Injecting WIMiR hidden zone: %s", zone)
                    add_to_queue(state, zone, depth + 1, cfg, robots)

            for link in extract_links(soup, url):
                add_to_queue(state, link, depth + 1, cfg, robots)

            if not footer_saved:
                footer = extract_footer(soup, cfg, url)
                if footer:
                    meta = {"url": url, "title": "Site footer (address, phone, e-mail)", "document_type": "footer",
                            "download_date": now()}
                    save_atomically(os.path.join(folder, "_stopka.md"), build_frontmatter(meta) + footer)
                    log.info("Site footer saved")
                footer_saved = True

            doc = process_page(soup, cfg, url)
            lang_ok = cfg["allowed_languages"] is None or not doc["language"] or doc["language"] in cfg["allowed_languages"]
            if not lang_ok:
                state.stats["skipped_html_language"] += 1
                log.info("⏩ %s – page language: %s", url, doc["language"])
                continue
            if doc["noindex"]:
                state.stats["skipped_noindex"] += 1
                continue
            body = doc["markdown"]
            if len(body) < cfg["min_chars"]:
                state.stats["skipped_too_short"] += 1
                log.info("⏩ %s – not enough content (%d char.)", url, len(body))
                continue

            text_for_hash = re.sub(r"\]\([^)]*\)", "]", body).lower()
            hash_val = hashlib.sha1(re.sub(r"\W+", " ", text_for_hash).encode("utf-8")).hexdigest()
            if hash_val in state.hashes and state.hashes[hash_val] != url:
                state.stats["skipped_duplicates"] += 1
                log.info("⏩ %s – content duplicate %s", url, state.hashes[hash_val])
                continue
            state.hashes[hash_val] = url

            filename = generate_filename(url)
            path_parts = [s for s in urlsplit(url).path.split("/") if s]
            meta = {
                "url": url,
                "title": doc["title"],
                "document_type": "html",
                "language": doc["language"] or "pl",
                "url_path": path_parts,
                "modification_date": res.headers.get("Last-Modified", ""),
                "download_date": now(),
                "char_count": len(body),
            }
            save_atomically(os.path.join(folder, filename), build_frontmatter(meta) + body)
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"url": url, "file": filename, "title": doc["title"], "chars": len(body),
                                     "hash": hash_val, "date": meta["download_date"]}, ensure_ascii=False) + "\n")
            state.stats["saved"] += 1
            log.info("🔍 Saved [%d]: %s (%d chars.) | queue: %d",
                     state.stats["saved"], url, len(body), len(state.queue))
            current = None

            if processed_in_session % SAVE_STATE_EVERY == 0:
                state.save(state_path, pdf_path)

    except KeyboardInterrupt:
        log.info("Interrupted (Ctrl+C) – saving state, can be resumed with the same command.")
        if current and current[0] in state.visited and current[0] not in {u for u, _ in state.queue}:
            state.visited.discard(current[0])
            state.queue.appendleft(current)
    finally:
        state.save(state_path, pdf_path)

    s = state.stats
    print("\n" + "=" * 50)
    print("FINAL SCRAPER REPORT")
    print("=" * 50)
    print(f"Visited subpages:          {len(state.visited)}")
    print(f"Saved .md files:           {s['saved']}")
    print(f"Found PDF links:           {len(state.pdfs)}  (-> {pdf_path})")
    print(f"Too short:                 {s['skipped_too_short']}")
    print(f"Content duplicates:        {s['skipped_duplicates']}")
    print(f"Errors / non-HTML:         {s['errors_or_not_html']}")
    print(f"Excluded paths (links):    {s['skipped_excluded']}")
    print(f"Query traps/pagination:    {s['skipped_query_trap']}")
    print(f"Remaining in queue:        {len(state.queue)}")
    print("=" * 50)
    return state


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HTML -> Markdown Crawler for RAG")
    ap.add_argument("profile", nargs="?", help="site key from config/sites.yaml (see --list)")
    ap.add_argument("--list", action="store_true", help="list crawlable sites and exit")
    ap.add_argument("--config", help="alternative sites.yaml (default: config/sites.yaml or $RAG_SITES_FILE)")
    ap.add_argument("--reset", action="store_true", help="ignore saved state and start from scratch")
    ap.add_argument("--max-pages", dest="max_pages", type=int, default=None, help="page limit for this session (for testing)")
    a = ap.parse_args()

    sites = get_sites(a.config)
    if a.list or not a.profile:
        print("Crawlable sites:")
        for key in sites.crawl_sites():
            print(f"  {key:<14} {sites.sites[key]['start_url']}")
        raise SystemExit(0 if a.list else "\nGive a site key, e.g.:  python scrape_html.py <site>")
    if a.profile not in sites.crawl_sites():
        raise SystemExit(f"Unknown site {a.profile!r}. Available: {', '.join(sites.crawl_sites())}")
    crawl(sites.crawl_profile(a.profile), reset=a.reset, max_pages=a.max_pages)
