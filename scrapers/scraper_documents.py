#!/usr/bin/env python3
"""
Pobieranie dokumentów (PDF / Word / ...) wykrytych przez crawler HTML (scraper_html.py).

NIE crawluje serwisu drugi raz: czyta listę  <folder>_meta/dokumenty_linki.jsonl,
którą crawler HTML zapisuje przy okazji (url + teksty linków + strony źródłowe).

Kolejność:
    python scraper_html.py bg                  # 1) strony HTML  (tworzy listę dokumentów)
    python pobierz_dokumenty.py bg            # 2) dokumenty z tej listy

Inne opcje:
    python pobierz_dokumenty.py css --max 20  # test na 20 plikach
    python pobierz_dokumenty.py bg --od-nowa  # ignoruj manifest, pobierz wszystko ponownie

Wynik:
    <folder_dokumentow>/            pliki (nazwa_czytelna_<hash>.pdf)
    <folder>_meta/dokumenty_manifest.jsonl   metadane każdego pliku: tytuł roboczy (tekst linku),
                                             strona źródłowa, sha256, rozmiar, status...
Oba skrypty muszą leżeć w jednym folderze (import z scraper_html.py).
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

import scraper_html as S

# ==========================================
# USTAWIENIA
# ==========================================
POBIERAJ = {".pdf", ".docx", ".doc", ".odt"}
MAX_ROZMIAR = 50 * 1024 * 1024
OPOZNIENIE = 0.5
TIMEOUT = (5, 30)

SYGNATURY = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",), ".xlsx": (b"PK\x03\x04",), ".pptx": (b"PK\x03\x04",),
    ".odt": (b"PK\x03\x04",), ".ods": (b"PK\x03\x04",),
    ".doc": (b"\xd0\xcf\x11\xe0",), ".xls": (b"\xd0\xcf\x11\xe0",), ".ppt": (b"\xd0\xcf\x11\xe0",),
}

OSTATECZNE = {"ok", "duplikat", "brak", "odrzucony", "robots"}

log = logging.getLogger("dokumenty")


def czy_sygnatura_ok(rozszerzenie, naglowek):
    sygn = SYGNATURY.get(rozszerzenie)
    if not sygn:
        return True
    if rozszerzenie == ".pdf":
        return b"%PDF" in naglowek[:1024]
    return any(naglowek.startswith(x) for x in sygn)


def nazwa_pliku(url):
    """Czytelna nazwa z oryginału (po odkodowaniu %20 itp.) + hash URL-a (różne katalogi, ta sama nazwa)."""
    oryg = unquote(urlsplit(url).path.rsplit("/", 1)[-1]) or "dokument"
    baza, ext = os.path.splitext(oryg)
    baza = re.sub(r"[^\w\-.]+", "_", baza)[:80].strip("_.") or "dokument"
    return f"{baza}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}{ext.lower()}"


def pobierz_plik(url, cel, rozszerzenie):
    """Zwraca (status, info). Zapis przez plik .part -> os.replace (nigdy uciętego pliku w folderze)."""
    tmp = cel + ".part"
    try:
        res = S.session.get(url, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        return "blad", {"powod": f"błąd sieci: {e}"}
    try:
        if res.status_code in (404, 410):
            return "brak", {"powod": f"HTTP {res.status_code}"}
        if res.status_code >= 400:
            return "blad", {"powod": f"HTTP {res.status_code}"}
        ct = res.headers.get("Content-Type", "").lower()
        if "text/html" in ct:
            return "odrzucony", {"powod": "serwer zwrócił stronę HTML zamiast dokumentu"}
        if int(res.headers.get("Content-Length") or 0) > MAX_ROZMIAR:
            return "odrzucony", {"powod": "za duży plik"}

        sha, rozmiar, pierwszy = hashlib.sha256(), 0, True
        with open(tmp, "wb") as f:
            for kawalek in res.iter_content(65536):
                if pierwszy:
                    pierwszy = False
                    if not czy_sygnatura_ok(rozszerzenie, kawalek[:1024]):
                        return "odrzucony", {"powod": f"zła sygnatura pliku (to nie jest {rozszerzenie})"}
                rozmiar += len(kawalek)
                if rozmiar > MAX_ROZMIAR:
                    return "odrzucony", {"powod": "za duży plik"}
                sha.update(kawalek)
                f.write(kawalek)
        if rozmiar == 0:
            return "odrzucony", {"powod": "pusty plik"}
        return "ok", {"sha256": sha.hexdigest(), "rozmiar": rozmiar, "content_type": ct,
                      "last_modified": res.headers.get("Last-Modified", "")}
    except requests.RequestException as e:
        return "blad", {"powod": f"błąd sieci: {e}"}
    finally:
        res.close()


def wybierz_tytul(teksty):
    """Tytuł roboczy = najdłuższy tekst linku (specyficzny opis bije ogólne 'BIP', 'Pobierz', 'więcej')."""
    return max(teksty, key=len) if teksty else ""


def wczytaj_jsonl(sciezka):
    if not os.path.exists(sciezka):
        return []
    with open(sciezka, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def wczytaj_linki_txt(sciezka):
    """Odczytuje linki z pliku txt z crawlera HTML."""
    if not os.path.exists(sciezka):
        return []
    with open(sciezka, "r", encoding="utf-8") as f:
        return [{"url": linia.strip()} for linia in f if linia.strip()]


def main():
    ap = argparse.ArgumentParser(description="Pobieranie dokumentów z listy wykrytej przez crawler HTML")
    ap.add_argument("profil", nargs="?", default="bg", choices=sorted(S.PROFILE))
    ap.add_argument("--max", type=int, default=None, help="limit plików w tej sesji (do testów)")
    ap.add_argument("--od-nowa", action="store_true", help="ignoruj manifest i pobierz wszystko ponownie")
    a = ap.parse_args()

    cfg = {**S.KONFIG_DOMYSLNY, **S.PROFILE[a.profil]}
    meta = cfg["folder"] + "_meta"
    wejscie = os.path.join(meta, "pdf_linki.txt")
    if not os.path.exists(wejscie):
        sys.exit(f"Brak {wejscie}\nNajpierw uruchom crawler HTML:  python scraper_html.py {a.profil}")
    folder = cfg.get("folder_dokumentow") or (cfg["folder"] + "_Dokumenty_Surowe")
    os.makedirs(folder, exist_ok=True)
    sciezka_manifestu = os.path.join(meta, "dokumenty_manifest.jsonl")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(meta, "dokumenty.log"), encoding="utf-8")])

    if a.od_nowa and os.path.exists(sciezka_manifestu):
        os.remove(sciezka_manifestu)

    ostatni = {r["url"]: r for r in wczytaj_jsonl(sciezka_manifestu)}
    hashe = {r["sha256"]: r["plik"] for r in ostatni.values() if r.get("status") == "ok" and r.get("sha256")}

    linki = wczytaj_linki_txt(wejscie)
    do_zrobienia = []
    for rek in linki:
        ext = os.path.splitext(urlsplit(rek["url"]).path)[1].lower()
        if ext not in POBIERAJ:
            continue
        if ostatni.get(rek["url"], {}).get("status") in OSTATECZNE:
            continue
        do_zrobienia.append((rek, ext))
    log.info("Dokumentów na liście: %d | do pobrania teraz: %d", len(linki), len(do_zrobienia))

    roboty = {}
    def robots_dla(url):
        host = urlsplit(url).netloc.lower()
        if host not in roboty:
            roboty[host] = S.zaladuj_robots({"start_url": f"https://{host}/"})
        return roboty[host]

    stat = Counter()
    try:
        for i, (rek, ext) in enumerate(do_zrobienia, 1):
            if a.max and i > a.max:
                break
            url = rek["url"]
            plik = nazwa_pliku(url)
            cel = os.path.join(folder, plik)
            zapis = {"url": url, "plik": plik, "rozszerzenie": ext,
                     "tytul_roboczy": wybierz_tytul(rek.get("teksty", [])),
                     "teksty_linkow": rek.get("teksty", []), "strony_zrodlowe": rek.get("zrodla", []),
                     "data_pobrania": datetime.now(timezone.utc).isoformat(timespec="seconds")}

            rp = robots_dla(url)
            if not rp.can_fetch(S.USER_AGENT, url):
                status, info = "robots", {"powod": "zablokowane w robots.txt"}
            else:
                status, info = pobierz_plik(url, cel, ext)
                time.sleep(max(OPOZNIENIE, float((rp.crawl_delay(S.USER_AGENT) if rp.last_checked else 0) or 0)))

            tmp = cel + ".part"
            if status == "ok":
                if info["sha256"] in hashe:
                    os.remove(tmp)
                    status, info = "duplikat", {**info, "duplikat_z": hashe[info["sha256"]]}
                else:
                    os.replace(tmp, cel)
                    hashe[info["sha256"]] = plik
            elif os.path.exists(tmp):
                os.remove(tmp)

            stat[status] += 1
            zapis.update(info)
            zapis["status"] = status
            with open(sciezka_manifestu, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(zapis, ensure_ascii=False) + "\n")
            ikona = {"ok": "✅", "duplikat": "♻️", "brak": "🚫", "odrzucony": "⚠️", "robots": "🤖", "blad": "❌"}[status]
            log.info("%s [%d/%d] %s %s", ikona, i, len(do_zrobienia), os.path.basename(urlsplit(url).path),
                     info.get("powod", "") or (f"{info.get('rozmiar', 0) // 1024} KB" if status == "ok" else ""))
    except KeyboardInterrupt:
        log.info("Przerwano (Ctrl+C) – manifest zapisany, można wznowić tym samym poleceniem.")
    finally:
        for f in os.listdir(folder):
            if f.endswith(".part"):
                os.remove(os.path.join(folder, f))

    print("\n" + "=" * 50)
    print("📊 RAPORT POBIERANIA DOKUMENTÓW")
    print("=" * 50)
    print(f"✅ Pobrane:                {stat['ok']}")
    print(f"♻️  Duplikaty (ta sama treść): {stat['duplikat']}")
    print(f"🚫 Nie istnieją (404/410):  {stat['brak']}")
    print(f"⚠️  Odrzucone (HTML/sygnatura/rozmiar): {stat['odrzucony']}")
    print(f"🤖 Zablokowane przez robots: {stat['robots']}")
    print(f"❌ Błędy tymczasowe (ponowią się przy kolejnym uruchomieniu): {stat['blad']}")
    print(f"📁 Folder: {folder}")
    print(f"🧾 Manifest: {sciezka_manifestu}")
    print("=" * 50)


if __name__ == "__main__":
    main()