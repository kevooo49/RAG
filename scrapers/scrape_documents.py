#!/usr/bin/env python3
"""
Downloading documents (PDF / Word / ...) detected by the HTML crawler (scraper_html.py).

DOES NOT crawl the site again: it reads the <folder>_meta/pdf_linki.txt list,
which the HTML crawler saves alongside (url + link texts + source pages).

Order of execution:
    python scraper_html.py bg                  # 1) HTML pages (creates document list)
    python download_documents.py bg            # 2) documents from that list

Other options:
    python download_documents.py css --max 20  # test on 20 files
    python download_documents.py bg --reset    # ignore manifest, download everything again

Output:
    <documents_folder>/                        files (readable_name_<hash>.pdf)
    <folder>_meta/documents_manifest.jsonl     metadata of each file: working title (link text),
                                               source page, sha256, size, status...
Both scripts must be in the same folder (import from scraper_html.py).
"""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import unquote, urlsplit

import requests

import scrape_html as S

# ==========================================
# SETTINGS
# ==========================================
DOWNLOAD_EXTENSIONS = {".pdf", ".docx", ".doc", ".odt"}
MAX_SIZE = 50 * 1024 * 1024
DELAY = 0.5
TIMEOUT = (5, 30)

SIGNATURES = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",), ".xlsx": (b"PK\x03\x04",), ".pptx": (b"PK\x03\x04",),
    ".odt": (b"PK\x03\x04",), ".ods": (b"PK\x03\x04",),
    ".doc": (b"\xd0\xcf\x11\xe0",), ".xls": (b"\xd0\xcf\x11\xe0",), ".ppt": (b"\xd0\xcf\x11\xe0",),
}

FINAL_STATUSES = {"ok", "duplicate", "missing", "rejected", "robots"}

log = logging.getLogger("documents")


def is_signature_ok(extension, header):
    sign = SIGNATURES.get(extension)
    if not sign:
        return True
    if extension == ".pdf":
        return b"%PDF" in header[:1024]
    return any(header.startswith(x) for x in sign)


def generate_filename(url):
    """Readable name from the original (after decoding %20 etc.) + URL hash (different folders, same name)."""
    original = unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "document"
    base, ext = os.path.splitext(original)
    base = re.sub(r"[^\w\-.]+", "_", base)[:80].strip("_.") or "document"
    return f"{base}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}{ext.lower()}"


def download_file(url, destination, extension):
    """Returns (status, info). Saved via .part file -> os.replace (never a truncated file in the folder)."""
    tmp = destination + ".part"
    try:
        res = S.session.get(url, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        return "error", {"reason": f"network error: {e}"}
    try:
        if res.status_code in (404, 410):
            return "missing", {"reason": f"HTTP {res.status_code}"}
        if res.status_code >= 400:
            return "error", {"reason": f"HTTP {res.status_code}"}
        ct = res.headers.get("Content-Type", "").lower()
        if "text/html" in ct:
            return "rejected", {"reason": "server returned an HTML page instead of a document"}
        if int(res.headers.get("Content-Length") or 0) > MAX_SIZE:
            return "rejected", {"reason": "file too large"}

        sha, size, is_first = hashlib.sha256(), 0, True
        with open(tmp, "wb") as f:
            for chunk in res.iter_content(65536):
                if is_first:
                    is_first = False
                    if not is_signature_ok(extension, chunk[:1024]):
                        return "rejected", {"reason": f"bad file signature (this is not a {extension})"}
                size += len(chunk)
                if size > MAX_SIZE:
                    return "rejected", {"reason": "file too large"}
                sha.update(chunk)
                f.write(chunk)
        if size == 0:
            return "rejected", {"reason": "empty file"}
        return "ok", {"sha256": sha.hexdigest(), "size": size, "content_type": ct,
                      "last_modified": res.headers.get("Last-Modified", "")}
    except requests.RequestException as e:
        return "error", {"reason": f"network error: {e}"}
    finally:
        res.close()


def choose_title(texts):
    """Working title = longest link text (specific description beats generic 'BIP', 'Download', 'more')."""
    return max(texts, key=len) if texts else ""


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def load_links_txt(path):
    """Reads links from a txt file from the HTML crawler."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [{"url": line.strip()} for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser(description="Download documents from a list detected by the HTML crawler")
    ap.add_argument("profile", nargs="?", default="bg", choices=sorted(S.PROFILES))
    ap.add_argument("--max", type=int, default=None, help="file limit for this session (for testing)")
    ap.add_argument("--reset", action="store_true", help="ignore the manifest and download everything again")
    a = ap.parse_args()

    cfg = {**S.DEFAULT_CONFIG, **S.PROFILES[a.profile]}
    meta = cfg["folder"] + "_meta"
    input_file = os.path.join(meta, "pdf_linki.txt")
    if not os.path.exists(input_file):
        sys.exit(f"Missing {input_file}\nFirst run the HTML crawler:  python scraper_html.py {a.profile}")
    folder = cfg.get("documents_folder") or (cfg["folder"] + "_Raw_Documents")
    os.makedirs(folder, exist_ok=True)
    manifest_path = os.path.join(meta, "documents_manifest.jsonl")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(meta, "documents.log"), encoding="utf-8")])

    if a.reset and os.path.exists(manifest_path):
        os.remove(manifest_path)

    last_state = {r["url"]: r for r in load_jsonl(manifest_path)}
    hashes = {r["sha256"]: r["file"] for r in last_state.values() if r.get("status") == "ok" and r.get("sha256")}

    links = load_links_txt(input_file)
    to_do = []
    for record in links:
        ext = os.path.splitext(urlsplit(record["url"]).path)[1].lower()
        if ext not in DOWNLOAD_EXTENSIONS:
            continue
        if last_state.get(record["url"], {}).get("status") in FINAL_STATUSES:
            continue
        to_do.append((record, ext))
    log.info("Documents on list: %d | to download now: %d", len(links), len(to_do))

    robots = {}
    def robots_for(url):
        host = urlsplit(url).netloc.lower()
        if host not in robots:
            robots[host] = S.load_robots({"start_url": f"https://{host}/"})
        return robots[host]

    stat = Counter()
    try:
        for i, (record, ext) in enumerate(to_do, 1):
            if a.max and i > a.max:
                break
            url = record["url"]
            filename = generate_filename(url)
            destination = os.path.join(folder, filename)
            entry = {"url": url, "file": filename, "extension": ext,
                     "working_title": choose_title(record.get("texts", [])),
                     "link_texts": record.get("texts", []), "source_pages": record.get("sources", []),
                     "download_date": datetime.now(timezone.utc).isoformat(timespec="seconds")}

            rp = robots_for(url)
            if not rp.can_fetch(S.USER_AGENT, url):
                status, info = "robots", {"reason": "blocked in robots.txt"}
            else:
                status, info = download_file(url, destination, ext)
                time.sleep(max(DELAY, float((rp.crawl_delay(S.USER_AGENT) if rp.last_checked else 0) or 0)))

            tmp = destination + ".part"
            if status == "ok":
                if info["sha256"] in hashes:
                    os.remove(tmp)
                    status, info = "duplicate", {**info, "duplicate_of": hashes[info["sha256"]]}
                else:
                    os.replace(tmp, destination)
                    hashes[info["sha256"]] = filename
            elif os.path.exists(tmp):
                os.remove(tmp)

            stat[status] += 1
            entry.update(info)
            entry["status"] = status
            with open(manifest_path, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
            icon = {"ok": "✅", "duplicate": "♻️", "missing": "🚫", "rejected": "⚠️", "robots": "🤖", "error": "❌"}[status]
            log.info("%s [%d/%d] %s %s", icon, i, len(to_do), os.path.basename(urlsplit(url).path),
                     info.get("reason", "") or (f"{info.get('size', 0) // 1024} KB" if status == "ok" else ""))
    except KeyboardInterrupt:
        log.info("Interrupted (Ctrl+C) – manifest saved, you can resume with the same command.")
    finally:
        for f in os.listdir(folder):
            if f.endswith(".part"):
                os.remove(os.path.join(folder, f))

    print("\n" + "=" * 50)
    print(" DOCUMENT DOWNLOAD REPORT")
    print("=" * 50)
    print(f" Downloaded:                         {stat['ok']}")
    print(f" Duplicates (same content):          {stat['duplicate']}")
    print(f" Do not exist (404/410):             {stat['missing']}")
    print(f" Rejected (HTML/signature/size):     {stat['rejected']}")
    print(f" Blocked by robots:                  {stat['robots']}")
    print(f" Temporary errors (retried later):   {stat['error']}")
    print(f" Folder: {folder}")
    print(f" Manifest: {manifest_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()