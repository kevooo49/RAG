"""Finds sites whose scraped pages still contain the site menu / footer (boilerplate repeated on every page).

Works on the files already saved in <prefix>_Strony folders - no network, no HTML needed. A line counts as
boilerplate when it appears on at least --min-share of a site's pages (menus and footers are the same on every
page, real content is not). For each site it reports how much of the average page is such repeated text, so you
know which sites to re-scrape after a fix in scrape_html.py / sites.yaml.

Usage (from the repository root or scrapers/):
    python scrapers/check_boilerplate.py                  # all sites with pages
    python scrapers/check_boilerplate.py --sites zarz,wilgz --examples 8
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from sites import load_or_exit  # noqa: E402  # type: ignore

_FRONTMATTER = re.compile(r"\A﻿?---[ \t]*\r?\n.*?\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_URL_LINE = re.compile(r'^url:\s*"?([^"\n]+)"?\s*$', re.M)


def host(url: str) -> str:
    h = urlsplit(url.strip()).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def foreign_pages(files, site: dict) -> list:
    """Pages whose frontmatter url is on another domain than the site (and its extra_domains)."""
    own = {host(site.get("start_url", ""))} | {d.lower().removeprefix("www.")
                                               for d in (site.get("crawl") or {}).get("extra_domains", [])}
    out = []
    for p in files:
        head = p.read_text(encoding="utf-8", errors="replace")[:1500]
        m = _URL_LINE.search(head)
        if m and host(m.group(1)) not in own:
            out.append((p, host(m.group(1))))
    return out


def page_lines(path: Path) -> list:
    text = _FRONTMATTER.sub("", path.read_text(encoding="utf-8", errors="replace"), count=1)
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= 12 and not ln.startswith("#"):        # headings are page-specific titles
            out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", default="", help="comma-separated site keys from sites.yaml (default: all)")
    ap.add_argument("--min-share", type=float, default=0.5, help="line on >= this share of pages = boilerplate")
    ap.add_argument("--min-pages", type=int, default=10, help="skip sites with fewer pages")
    ap.add_argument("--examples", type=int, default=3, help="most common boilerplate lines to show per site")
    ap.add_argument("--list-foreign", action="store_true",
                    help="list pages from other domains (e.g. COK pages saved in a faculty folder)")
    ap.add_argument("--move-foreign", action="store_true",
                    help="move pages from other domains to <folder>_foreign/ (they belong to another site's crawl)")
    a = ap.parse_args()

    cfg = load_or_exit()
    keys = [k for k in a.sites.split(",") if k] or [k for k, site in cfg.sites.items() if site.get("pages")]
    rows = []
    for key in keys:
        try:
            folder = Path(cfg.folder(key, "pages"))
        except Exception:
            continue
        files = sorted(p for p in folder.glob("*.md") if not p.name.startswith("_")) if folder.is_dir() else []
        if len(files) < a.min_pages:
            continue
        pages = [page_lines(p) for p in files]
        df = Counter(ln for lines in pages for ln in set(lines))
        boiler = {ln for ln, n in df.items() if n >= a.min_share * len(pages)}
        shares, chars = [], []
        for lines in pages:
            total = sum(len(ln) for ln in lines) or 1
            b = sum(len(ln) for ln in lines if ln in boiler)
            shares.append(b / total)
            chars.append(b)
        foreign = foreign_pages(files, cfg.sites[key])
        rows.append((key, len(files), sum(shares) / len(shares), sum(chars) / len(chars),
                     [ln for ln, _ in df.most_common() if ln in boiler][:a.examples], foreign))
        if foreign and a.move_foreign:
            dest = folder.with_name(folder.name + "_foreign")
            dest.mkdir(exist_ok=True)
            for p, _h in foreign:
                p.rename(dest / p.name)

    rows.sort(key=lambda r: -r[2])
    print(f"{'site':<18}{'pages':>7}{'boilerplate share':>19}{'avg chars/page':>16}{'foreign':>9}   most repeated lines")
    print("-" * 118)
    for key, n, share, ch, ex, foreign in rows:
        flag = "  <-- re-scrape" if share >= 0.3 else ""
        hosts = Counter(h for _p, h in foreign)
        print(f"{key:<18}{n:>7}{share:>18.0%}{ch:>16.0f}{len(foreign):>9}{flag}"
              + (f"  foreign: {dict(hosts.most_common(3))}" if foreign else ""))
        for ln in ex:
            print(f"{'':<52}{ln[:90]}")
        if a.list_foreign:
            for p, h in foreign:
                print(f"{'':<52}[{h}] {p.name}")
    moved = sum(len(r[5]) for r in rows) if a.move_foreign else 0
    if moved:
        print(f"\nMoved {moved} pages from other domains to <folder>_foreign/ (delete them once checked).")
    print("foreign = pages whose frontmatter url is on another domain (saved by mistake from another site's crawl).")
    print("\nboilerplate share = average fraction of a page's text made of lines repeated on "
          f">= {a.min_share:.0%} of the site's pages (menu, footer, cookie banners).")


if __name__ == "__main__":
    main()