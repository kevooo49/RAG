#!/usr/bin/env python3
"""
Crawler HTML -> Markdown pod RAG (Uczelnia AGH).

Uruchomienie:
    python scraper_agh.py                 # profil "bg" (Biblioteka Główna)
    python scraper_agh.py bg --max-stron 50   # test na 50 stronach
    python scraper_agh.py bg --reset      # pełny re-crawl od zera

Jedna logika crawlera + osobny PROFIL na każdy serwis (selektory, wykluczenia).
Nowy serwis = nowy wpis w PROFILE, a nie nowy skrypt.
"""
import argparse
import hashlib
import html
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==========================================
# USTAWIENIA GLOBALNE
# ==========================================
# Przedstaw się uczciwie i podaj kontakt – admin serwisu będzie wiedział, kto go odwiedza.
USER_AGENT = "AGH-RAG-Crawler/1.0 (+kontakt: kevinst@student.agh.edu.pl)"
OPOZNIENIE = 0.3               # sekundy między requestami (robots.txt Crawl-delay ma pierwszeństwo, jeśli większy)
TIMEOUT = (5, 20)              # (połączenie, odczyt)
MAX_BAJTOW = 5 * 1024 * 1024   # strony HTML większe niż 5 MB pomijamy
MAX_WARIANTOW_QUERY = 15       # ile wariantów ?query na jedną ścieżkę (ochrona przed pułapkami paginacji)
ZAPIS_STANU_CO = 25            # co ile stron zapisywać stan (wznawianie po przerwaniu)
MAX_SMIEC_ZNAKOW = 3000        # element "śmieciowy" dłuższy niż to uznajemy za kontener treści i NIE usuwamy

# Parametry URL, które tylko śledzą / sesjonują – wycinamy je z adresu
PARAMETRY_DO_USUNIECIA = {"fbclid", "gclid", "chash", "no_cache", "phpsessid", "sid", "jsessionid"}
PREFIKSY_DO_USUNIECIA = ("utm_", "mc_", "pk_")
# Parametry oznaczające inny "widok" tej samej treści – takich URL-i w ogóle nie odwiedzamy
PARAMETRY_POMIJAJ_URL = {"print", "sort", "order", "orderby", "sortby", "format", "output", "ical"}

ROZSZERZENIA_PDF = re.compile(r"\.pdf$", re.I)
ROZSZERZENIA_BINARNE = re.compile(
    r"\.(docx?|xlsx?|pptx?|odt|ods|zip|rar|7z|gz|jpe?g|png|gif|svg|webp|bmp|ico|mp[34]|avi|mov|wav|"
    r"css|js|woff2?|ttf|eot|xml|json|rss|ics|epub|csv|txt|exe|dmg)$",
    re.I,
)

# Elementy usuwane zawsze
SMIECI_BEZWARUNKOWO = ["script", "style", "noscript", "template", "svg", "iframe", "canvas",
                       "object", "embed", "link", "meta"]
# Elementy usuwane, o ile są "małe" (zabezpieczenie przed wycięciem kontenera z treścią)
SMIECI_OSTROZNIE = [
    "nav", "form", "button", "select", "textarea", "input",
    "[role=navigation]", "[role=search]", "[role=banner]",
    "[class*=breadcrumb]", "[class*=cookie]", "[id*=cookie]",
    "[class*=share]", "[class*=social]", "[class*=pagination]",
    "[class*=skip]", "[class*=carousel-control]",
]
# Dodatkowo usuwane TYLKO gdy nie znaleziono kontenera treści i bierzemy całe <body>
SMIECI_STRUKTURA = ["header", "footer", "aside"]

# ==========================================
# PROFILE SERWISÓW
# ==========================================
KONFIG_DOMYSLNY = {
    "selektory_tresci": ["#mainContent", "main", "[role=main]", "article", "#main", "#content"],
    "selektory_stopki": ["#pageFooter", "footer"],
    "selektory_smieci_dodatkowe": [],
    "wykluczone_sciezki": [],
    "jezyki_obce_prefiksy": ["/en", "/de", "/ru", "/uk", "/fr", "/es"],
    "jezyki_dozwolone": ("pl",),   # None = nie filtruj po <html lang>
    "min_znakow": 300,             # krótsze strony nie trafiają do bazy (linki z nich nadal śledzimy)
    "max_glebokosc": 12,
}

PROFILE = {
    "bg": {
        "start_url": "https://bg.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "Biblioteka_Strony"),
        
        # Pusta lista wymusza wczytanie całego tagu <body> zamiast fałszywego #mainContent
        "selektory_tresci": [], 
        
        # Wskazujemy precyzyjnie, co skrypt ma wyciąć (menu desktopowe, menu mobilne, stopka i linki "skip")
        "selektory_smieci_dodatkowe": ["#pageFooter", "#unitMenu", "#mainMenuNav", ".skip-to-content"],
        
        "wykluczone_sciezki": [
            "/wydarzenia", "/aktualnosci", "/galeria", "/galerie",
            "/konferencje", "/archiwum", "/blog",
            "/webinaria", "/noc-bibliotek-2026", 
            "/mapa-strony", "/intranet-bg",
            "/typo3", "/typo3conf", 
            "/szukaj", "/wyszukiwarka", "/login",
        ],
    },
    "css": {
        "start_url": "https://css.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "CSS_Strony"),
        "selektory_tresci": [],
        "selektory_stopki": ["#pageFooter"],
        "selektory_smieci_dodatkowe": ["#mainMenuNav", "#pageFooter", ".skip-to-content"],
        "wykluczone_sciezki": [
            "/swiadczenia/archiwum-dokumentow",
        ],
    },
    "ms": {
        "start_url": "https://www.miasteczko.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "Miasteczko_Strony"),
        
        "selektory_tresci": ["#maincontent"],
        
        "selektory_stopki": ["#footer"],
        
        "selektory_smieci_dodatkowe": ["#header", ".content-right", "#search", ".facebook-like"],
        
        "wykluczone_sciezki": [
            "/galeria-zdjec.html",
            "/pl/horizontal/pjm.html",
        ],
    },
    "ck": {
        "start_url": "https://www.ck.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "Centrum_Karier_Strony"),
        
        "selektory_tresci": [],
        
        "selektory_stopki": ["#pageFooter"],
        
        "selektory_smieci_dodatkowe": [
            "#mainMenuNav", "#unitMenu", ".fw-block", 
            ".skip-to-content", ".news__container_stud",
            "header"
        ],
        
        "wykluczone_sciezki": [
            "/aktualnosci",
            "/szukaj", "/login"
        ],
    },
    "rekrutacja": {
        "start_url": "https://rekrutacja.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "Rekrutacja_Strony"),
        
        "selektory_tresci": ["#tabs-content-wrapper", ".menu-content-result__option", ".site-main", "#content"],
        
        "selektory_stopki": ["#colophon", "#pageFooter"],
        
        "selektory_smieci_dodatkowe": [
            ".cr-counter-wrapper",
            ".navbar-row", ".menu-vertical",
            ".socialmedia-logos", ".social-section",
            ".dontforget-wrapper",
            ".menuslider-wrapper",
            ".faq-wrapper",
            ".sticky-icons",
            "#cai-user-widget",
            "iframe", "script", "style", "noscript",
            ".owl-carousel", ".owl-stage"
        ],
        
        "wykluczone_sciezki": [
            "/aktualnosci", 
            "/kalkulator", 
            "/rejestracja"
        ],
    },
    "csm": {
        "start_url": "https://csm.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "Centrum_Spraw_Miedzynarodowych_Strony"),
        "selektory_tresci": [], 
        
        "selektory_stopki": ["#pageFooter"],
        "selektory_smieci_dodatkowe": [
            "#unitMenu", 
            "#mainMenuNav", 
            ".skip-to-content", 
            "#cookieconsent", 
            "#scroller",
            ".carousel-indicators",
            ".carousel-control"
        ],
        "wykluczone_sciezki": [
            "/aktualnosci",
            "/szukaj", 
            "/login"
        ],
    },
    "cyfronet": {
        "start_url": "https://www.cyfronet.pl/pl",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "Cyfronet_Strony"),
        "selektory_tresci": ["main"],
        "selektory_stopki": ["footer"],
        "selektory_smieci_dodatkowe": [
            "header",
            ".searchbox-wrapper",
            "#skip-link",
            "#sitemap",
            "#return-to-top",
            "noscript", "script", "style"
        ],
        "wykluczone_sciezki": [
            "/search/query",
            "/login",
            "/newsletter"
        ],
    },
    "swfis": {
        "start_url": "https://swfis.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "SWFiS_Strony"),
        "selektory_tresci": [], 
        "selektory_stopki": ["#pageFooter"],
        "selektory_smieci_dodatkowe": [
            "#unitMenu",
            "#mainMenuNav",
            ".skip-to-content",
            "#cookieconsent",
            "#scroller",
            ".carousel-indicators",
            ".carousel-control"
        ],
        "wykluczone_sciezki": [
            "/szukaj", 
            "/login"
        ],
    },
    "sjo": {
        "start_url": "https://www.sjo.agh.edu.pl/",
        "folder": os.path.join("..", "data", "Baza_Wiedzy_AGH", "SJO_Strony"),
        "selektory_tresci": ["#main"],
        "selektory_stopki": ["footer"],
        "selektory_smieci_dodatkowe": [
            "header",
            ".header",
            "#cookieconsent",
            ".mobile-menu-head",
            "script", "style", "noscript"
        ],
        "wykluczone_sciezki": [
            "/aktualnosci",
            "/category/"
        ],
    },
}

# ==========================================
# SESJA HTTP
# ==========================================
def utworz_sesje():
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


session = utworz_sesje()
log = logging.getLogger("crawler")


# ==========================================
# URL-E
# ==========================================
def netloc_bez_www(url):
    n = urlsplit(url).netloc.lower()
    n = re.sub(r":(80|443)$", "", n)
    return n[4:] if n.startswith("www.") else n


def czy_param_smieciowy(nazwa):
    n = nazwa.lower()
    return n in PARAMETRY_DO_USUNIECIA or n.startswith(PREFIKSY_DO_USUNIECIA)


def normalizuj_url(url):
    """Ujednolica URL: https, bez #kotwicy, bez śledzących parametrów, posortowane query,
    bez końcowego '/' i bez /index.html. Rzuca ValueError dla niepoprawnych adresów."""
    p = urlsplit(url.strip())
    scheme = "https" if p.scheme.lower() in ("http", "https") else p.scheme.lower()
    netloc = re.sub(r":(80|443)$", "", p.netloc.lower())

    path = re.sub(r"/{2,}", "/", p.path)
    path = re.sub(r"/index\.(html?|php)$", "", path, flags=re.I)
    path = path.rstrip("/") or "/"

    pary = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not czy_param_smieciowy(k)]
    pary.sort()
    return urlunsplit((scheme, netloc, path, urlencode(pary), ""))


def czy_nalezy_do_domeny(url, domena_glowna):
    """Dokładne porównanie hosta (chroni przed agh.edu.pl.evil.com); 'www.' ignorowane."""
    return netloc_bez_www(url) == netloc_bez_www(domena_glowna)


def czy_pominac_obcy_jezyk(url, cfg):
    p = urlsplit(url)
    path = p.path.lower()
    for lang in cfg["jezyki_obce_prefiksy"]:
        if path == lang or path.startswith(lang + "/"):
            return True
    q = p.query.lower()
    return "lang=" in q and "lang=pl" not in q


def czy_wykluczona_sciezka(url, cfg):
    path = urlsplit(url).path.lower().rstrip("/")
    for w in cfg["wykluczone_sciezki"]:
        w = w.lower().rstrip("/")
        if path == w or path.startswith(w + "/"):
            return True
    return False


def ocen_url(url, cfg):
    """'ok' | 'pdf' | powód odrzucenia."""
    p = urlsplit(url)
    if p.scheme not in ("http", "https"):
        return "schemat"
    if not czy_nalezy_do_domeny(url, cfg["start_url"]):
        return "obca_domena"
    if czy_pominac_obcy_jezyk(url, cfg):
        return "jezyk"
    if czy_wykluczona_sciezka(url, cfg):
        return "wykluczona"
    if ROZSZERZENIA_PDF.search(p.path):
        return "pdf"
    if ROZSZERZENIA_BINARNE.search(p.path):
        return "plik"
    if {k.lower() for k, _ in parse_qsl(p.query)} & PARAMETRY_POMIJAJ_URL:
        return "parametr_widoku"
    return "ok"


def generuj_nazwe_pliku(url):
    p = urlsplit(url)
    slug = p.path.strip("/").replace("/", "__") or "_index"
    slug = re.sub(r"[^\w\-.]+", "_", slug)[:80].strip("_.") or "_index"
    return f"{slug}_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:8]}.md"


# ==========================================
# CZYSZCZENIE HTML -> MARKDOWN
# ==========================================
def usun_elementy(root, selektory, ostroznie):
    for sel in selektory:
        for el in root.select(sel):
            if getattr(el, "decomposed", False):
                continue
            if ostroznie and len(el.get_text(" ", strip=True)) > MAX_SMIEC_ZNAKOW:
                continue
            el.decompose()


def wybierz_tresc(soup, cfg):
    """Zwraca (element, czy_fallback_do_body)."""
    for sel in cfg["selektory_tresci"]:
        el = soup.select_one(sel)
        if el is not None and el.get_text(strip=True):
            return el, False
    return soup.body, True


def napraw_linki(root, baza):
    """Linki względne -> bezwzględne (w chunku RAG '/kontakt' jest bezużyteczne).
    Puste, #kotwice i javascript: zamieniamy na sam tekst; mailto:/tel: zostają."""
    for a in root.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "javascript:")):
            a.unwrap()
        elif not href.startswith(("mailto:", "tel:")):
            a["href"] = urljoin(baza, href)


def czysc_tekst(t):
    t = t.replace("\u00ad", "")                    # miękkie dzielenie wyrazów ("biblio­teka") psuje tokenizację
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", t)  # znaki zerowej szerokości
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    # usuń bezpośrednio powtórzone linie (zduplikowane karuzele, menu mobilne/desktop)
    wynik, poprzednia = [], None
    for linia in t.split("\n"):
        if linia.strip() and linia.strip() == poprzednia:
            continue
        wynik.append(linia)
        if linia.strip():
            poprzednia = linia.strip()
    t = "\n".join(wynik)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def html_do_markdown(root):
    return czysc_tekst(md(str(root), heading_style="ATX", strip=["img"], bullets="-"))


def wyciagnij_linki(soup, baza):
    """Linki zbieramy z PEŁNEJ strony (razem z menu) PRZED czyszczeniem – to menu prowadzi do podstron."""
    return [urljoin(baza, a["href"].strip()) for a in soup.find_all("a", href=True)]


def wyciagnij_stopke(soup, cfg, baza):
    """Stopka (adres, telefon, e-mail) jest na każdej stronie – zapisujemy ją RAZ, nie w każdym dokumencie."""
    for sel in cfg["selektory_stopki"]:
        el = soup.select_one(sel)
        if el is not None and el.get_text(strip=True):
            kopia = BeautifulSoup(str(el), "html.parser")
            usun_elementy(kopia, SMIECI_BEZWARUNKOWO, ostroznie=False)
            napraw_linki(kopia, baza)
            return html_do_markdown(kopia)
    return ""


def przetworz_strone(soup, cfg, url):
    """Zwraca dict: tytul, jezyk, noindex, canonical, markdown. Modyfikuje soup!"""
    html_tag = soup.find("html")
    jezyk = ((html_tag.get("lang") or "").lower().split("-")[0]) if html_tag else ""
    robots_meta = soup.find("meta", attrs={"name": re.compile("^robots$", re.I)})
    noindex = bool(robots_meta and "noindex" in (robots_meta.get("content") or "").lower())
    canon_tag = soup.find("link", rel="canonical")
    canonical = canon_tag.get("href") if canon_tag else None
    tytul_tag = soup.title.get_text(strip=True) if soup.title else ""

    root, fallback = wybierz_tresc(soup, cfg)
    if root is None:
        return {"markdown": "", "tytul": tytul_tag, "jezyk": jezyk, "noindex": noindex, "canonical": canonical}

    usun_elementy(root, SMIECI_BEZWARUNKOWO, ostroznie=False)
    if fallback:
        usun_elementy(root, SMIECI_STRUKTURA, ostroznie=False)
    usun_elementy(root, SMIECI_OSTROZNIE + cfg["selektory_smieci_dodatkowe"], ostroznie=True)
    napraw_linki(root, url)

    h1 = root.find("h1")
    tytul = (h1.get_text(" ", strip=True) if h1 else "") or tytul_tag or "Brak tytułu"
    return {
        "markdown": html_do_markdown(root),
        "tytul": czysc_tekst(tytul),
        "jezyk": jezyk,
        "noindex": noindex,
        "canonical": canonical,
    }


def zbuduj_frontmatter(meta):
    # json.dumps daje poprawne stringi YAML (cudzysłowy, ukośniki, polskie znaki)
    linie = ["---"] + [f"{k}: {json.dumps(v, ensure_ascii=False)}" for k, v in meta.items()] + ["---", "", ""]
    return "\n".join(linie)


def zapisz_atomowo(sciezka, tekst):
    tmp = sciezka + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(tekst)
    os.replace(tmp, sciezka)  # przerwanie skryptu nie zostawi uciętego pliku


# ==========================================
# STAN CRAWLERA (wznawianie bez ponownego pobierania)
# ==========================================
class Stan:
    def __init__(self):
        self.odwiedzone = set()
        self.kolejka = deque()          # (url, głębokość)
        self.znane = set()              # wszystko, co już widzieliśmy (kolejka + odwiedzone + odrzucone)
        self.hashe = {}                 # hash treści -> url (wykrywanie duplikatów)
        self.warianty = defaultdict(set)  # ścieżka -> zestaw query
        self.pdf = set()
        self.stat = Counter()

    def zapisz(self, sciezka, sciezka_pdf):
        dane = {
            "odwiedzone": sorted(self.odwiedzone),
            "kolejka": list(self.kolejka),
            "hashe": self.hashe,
            "warianty": {k: sorted(v) for k, v in self.warianty.items()},
            "pdf": sorted(self.pdf),
            "stat": dict(self.stat),
        }
        zapisz_atomowo(sciezka, json.dumps(dane, ensure_ascii=False))
        zapisz_atomowo(sciezka_pdf, "\n".join(sorted(self.pdf)) + "\n")

    @classmethod
    def wczytaj(cls, sciezka):
        s = cls()
        with open(sciezka, encoding="utf-8") as f:
            d = json.load(f)
        s.odwiedzone = set(d["odwiedzone"])
        s.kolejka = deque((u, g) for u, g in d["kolejka"])
        s.hashe = d["hashe"]
        s.warianty = defaultdict(set, {k: set(v) for k, v in d["warianty"].items()})
        s.pdf = set(d["pdf"])
        s.stat = Counter(d["stat"])
        s.znane = s.odwiedzone | {u for u, _ in s.kolejka}
        return s


def dodaj_do_kolejki(stan, link, glebokosc, cfg, robots):
    try:
        link = normalizuj_url(link)
    except ValueError:
        return
    if link in stan.znane:
        return
    stan.znane.add(link)

    wynik = ocen_url(link, cfg)
    if wynik == "pdf":
        stan.pdf.add(link)
        return
    if wynik != "ok":
        stan.stat["pominiete_" + wynik] += 1
        return
    if glebokosc > cfg["max_glebokosc"]:
        stan.stat["pominiete_glebokosc"] += 1
        return
    if not robots.can_fetch(USER_AGENT, link):
        stan.stat["pominiete_robots"] += 1
        return
    p = urlsplit(link)
    if p.query:
        warianty = stan.warianty[p.path]
        if p.query not in warianty and len(warianty) >= MAX_WARIANTOW_QUERY:
            stan.stat["pominiete_pulapka_query"] += 1
            return
        warianty.add(p.query)
    stan.kolejka.append((link, glebokosc))


# ==========================================
# ROBOTS.TXT I SITEMAPY
# ==========================================
def zaladuj_robots(cfg):
    rp = RobotFileParser()
    try:
        r = session.get(urljoin(cfg["start_url"], "/robots.txt"), timeout=TIMEOUT)
        if r.status_code == 200:
            rp.parse(r.text.splitlines())
        else:
            if r.status_code in (401, 403):
                log.warning("robots.txt zwrócił %s – traktuję jak brak ograniczeń", r.status_code)
            rp.allow_all = True
    except requests.RequestException as e:
        log.warning("Nie udało się pobrać robots.txt (%s) – brak ograniczeń", e)
        rp.allow_all = True
    return rp


def wczytaj_sitemapy(adresy, glebokosc=0):
    wynik = []
    for adres in adresy:
        try:
            r = session.get(adres, timeout=TIMEOUT)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        loc = [html.unescape(x) for x in re.findall(r"<loc>\s*(.*?)\s*</loc>", r.text, flags=re.S)]
        if "<sitemapindex" in r.text and glebokosc < 2:
            wynik += wczytaj_sitemapy(loc, glebokosc + 1)
        else:
            wynik += loc
    return wynik


# ==========================================
# POBIERANIE
# ==========================================
def pobierz(url):
    """Zwraca (res, bytes, None) albo (None, None, powód). Nie ściąga plików, które nie są HTML-em."""
    try:
        res = session.get(url, timeout=TIMEOUT, stream=True)
    except requests.RequestException as e:
        return None, None, f"błąd sieci: {e}"
    try:
        if res.status_code >= 400:
            return None, None, f"HTTP {res.status_code}"
        ct = res.headers.get("Content-Type", "").lower()
        if "html" not in ct:
            return None, None, f"nie HTML ({ct or 'brak Content-Type'})"
        if int(res.headers.get("Content-Length") or 0) > MAX_BAJTOW:
            return None, None, "za duży plik"
        czesci, rozmiar = [], 0
        for kawalek in res.iter_content(65536):
            rozmiar += len(kawalek)
            if rozmiar > MAX_BAJTOW:
                return None, None, "za duży plik"
            czesci.append(kawalek)
        return res, b"".join(czesci), None
    except requests.RequestException as e:
        return None, None, f"błąd sieci: {e}"
    finally:
        res.close()


def parsuj_html(content):
    # Przekazujemy BAJTY, nie res.text: requests bez 'charset' w nagłówku zgaduje ISO-8859-1
    # i psuje polskie znaki. BeautifulSoup sam wykrywa kodowanie (meta charset / UTF-8).
    try:
        return BeautifulSoup(content, "lxml")
    except Exception:
        return BeautifulSoup(content, "html.parser")


# ==========================================
# GŁÓWNY PROCES (BFS)
# ==========================================
def crawl(cfg, reset=False, max_stron=None):
    folder = cfg["folder"]
    meta_folder = folder + "_meta"   # stan/manifest poza folderem z .md, żeby nie trafiły do indeksowania
    os.makedirs(folder, exist_ok=True)
    os.makedirs(meta_folder, exist_ok=True)
    sciezka_stanu = os.path.join(meta_folder, "stan.json")
    sciezka_pdf = os.path.join(meta_folder, "pdf_linki.txt")
    sciezka_manifestu = os.path.join(meta_folder, "manifest.jsonl")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(meta_folder, "crawler.log"), encoding="utf-8")],
    )

    robots = zaladuj_robots(cfg)
    opoznienie = max(OPOZNIENIE, float(robots.crawl_delay(USER_AGENT) or 0)) if robots.last_checked else OPOZNIENIE

    if os.path.exists(sciezka_stanu) and not reset:
        stan = Stan.wczytaj(sciezka_stanu)
        log.info("Wznawiam: %d odwiedzonych, %d w kolejce", len(stan.odwiedzone), len(stan.kolejka))
    else:
        stan = Stan()
        if os.path.exists(sciezka_manifestu):
            os.remove(sciezka_manifestu)
        start = normalizuj_url(cfg["start_url"])
        stan.znane.add(start)
        stan.kolejka.append((start, 0))
        # sitemapy: pewniejsze pokrycie niż samo klikanie po linkach
        try:
            mapy = robots.site_maps() or []
        except AttributeError:
            mapy = []
        mapy = mapy or [urljoin(cfg["start_url"], "/sitemap.xml")]
        for u in wczytaj_sitemapy(mapy):
            dodaj_do_kolejki(stan, u, 1, cfg, robots)
        log.info("Start: %d adresów w kolejce (po sitemapach)", len(stan.kolejka))

    stopka_zapisana = os.path.exists(os.path.join(folder, "_stopka.md")) and not reset
    przetworzone_w_sesji = 0
    biezacy = None
    teraz = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")

    log.info("🕷 Crawler startuje: %s", cfg["start_url"])
    try:
        while stan.kolejka:
            if max_stron and przetworzone_w_sesji >= max_stron:
                log.info("Osiągnięto limit --max-stron=%d", max_stron)
                break
            url, glebokosc = stan.kolejka.popleft()
            biezacy = (url, glebokosc)
            if url in stan.odwiedzone:
                continue
            stan.odwiedzone.add(url)
            przetworzone_w_sesji += 1

            res, content, powod = pobierz(url)
            time.sleep(opoznienie)
            if res is None:
                stan.stat["bledy_lub_nie_html"] += 1
                log.info("⏩ %s – %s", url, powod)
                continue

            # Przekierowania: dalej pracujemy na adresie końcowym (to względem niego rozwiązujemy linki)
            try:
                final = normalizuj_url(res.url)
            except ValueError:
                continue
            if final != url:
                if final in stan.odwiedzone or ocen_url(final, cfg) != "ok":
                    stan.stat["pominiete_przekierowanie"] += 1
                    continue
                stan.odwiedzone.add(final)
                stan.znane.add(final)
                url = final

            soup = parsuj_html(content)

            # 1) linki z pełnej strony (zanim wytniemy menu)
            for link in wyciagnij_linki(soup, url):
                dodaj_do_kolejki(stan, link, glebokosc + 1, cfg, robots)

            # 2) stopka – raz
            if not stopka_zapisana:
                stopka = wyciagnij_stopke(soup, cfg, url)
                if stopka:
                    meta = {"url": url, "tytul": "Stopka serwisu (adres, telefon, e-mail)", "typ_dokumentu": "stopka",
                            "data_pobrania": teraz()}
                    zapisz_atomowo(os.path.join(folder, "_stopka.md"), zbuduj_frontmatter(meta) + stopka)
                    log.info("📌 Zapisano stopkę serwisu")
                stopka_zapisana = True

            # 3) treść właściwa
            dok = przetworz_strone(soup, cfg, url)
            jez_ok = cfg["jezyki_dozwolone"] is None or not dok["jezyk"] or dok["jezyk"] in cfg["jezyki_dozwolone"]
            if not jez_ok:
                stan.stat["pominiete_jezyk_html"] += 1
                log.info("⏩ %s – język strony: %s", url, dok["jezyk"])
                continue
            if dok["noindex"]:
                stan.stat["pominiete_noindex"] += 1
                continue
            body = dok["markdown"]
            if len(body) < cfg["min_znakow"]:
                stan.stat["pominiete_za_krotkie"] += 1
                log.info("⏩ %s – za mało treści (%d zn.)", url, len(body))
                continue

            # hash liczymy z samego tekstu (bez adresów linków), żeby ta sama treść pod innym URL-em była duplikatem
            tekst_do_hasha = re.sub(r"\]\([^)]*\)", "]", body).lower()
            skrot = hashlib.sha1(re.sub(r"\W+", " ", tekst_do_hasha).encode("utf-8")).hexdigest()
            if skrot in stan.hashe and stan.hashe[skrot] != url:
                stan.stat["pominiete_duplikaty"] += 1
                log.info("⏩ %s – duplikat treści %s", url, stan.hashe[skrot])
                continue
            stan.hashe[skrot] = url

            nazwa = generuj_nazwe_pliku(url)
            czesci_sciezki = [s for s in urlsplit(url).path.split("/") if s]
            meta = {
                "url": url,
                "tytul": dok["tytul"],
                "typ_dokumentu": "html",
                "jezyk": dok["jezyk"] or "pl",
                "sciezka_url": czesci_sciezki,
                "data_modyfikacji": res.headers.get("Last-Modified", ""),
                "data_pobrania": teraz(),
                "liczba_znakow": len(body),
            }
            zapisz_atomowo(os.path.join(folder, nazwa), zbuduj_frontmatter(meta) + body)
            with open(sciezka_manifestu, "a", encoding="utf-8") as mf:
                mf.write(json.dumps({"url": url, "plik": nazwa, "tytul": dok["tytul"], "znaki": len(body),
                                     "hash": skrot, "data": meta["data_pobrania"]}, ensure_ascii=False) + "\n")
            stan.stat["zapisane"] += 1
            log.info("🔍 Zapisano [%d]: %s (%d zn.) | kolejka: %d",
                     stan.stat["zapisane"], url, len(body), len(stan.kolejka))
            biezacy = None

            if przetworzone_w_sesji % ZAPIS_STANU_CO == 0:
                stan.zapisz(sciezka_stanu, sciezka_pdf)

    except KeyboardInterrupt:
        log.info("Przerwano (Ctrl+C) – zapisuję stan, można wznowić tym samym poleceniem.")
        if biezacy and biezacy[0] in stan.odwiedzone and biezacy[0] not in {u for u, _ in stan.kolejka}:
            stan.odwiedzone.discard(biezacy[0])
            stan.kolejka.appendleft(biezacy)
    finally:
        stan.zapisz(sciezka_stanu, sciezka_pdf)

    s = stan.stat
    print("\n" + "=" * 50)
    print("📊 RAPORT KOŃCOWY SCRAPERA")
    print("=" * 50)
    print(f"🌐 Odwiedzone podstrony:        {len(stan.odwiedzone)}")
    print(f"📝 Zapisane pliki .md:          {s['zapisane']}")
    print(f"📄 Znalezione linki do PDF:     {len(stan.pdf)}  (-> {sciezka_pdf})")
    print(f"⏩ Za krótkie:                  {s['pominiete_za_krotkie']}")
    print(f"⏩ Duplikaty treści:            {s['pominiete_duplikaty']}")
    print(f"⏩ Błędy / nie-HTML:            {s['bledy_lub_nie_html']}")
    print(f"⏩ Wykluczone ścieżki (linki):  {s['pominiete_wykluczona']}")
    print(f"⏩ Pułapki query/paginacja:     {s['pominiete_pulapka_query']}")
    print(f"⏳ Pozostało w kolejce:         {len(stan.kolejka)}")
    print("=" * 50)
    return stan


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Crawler HTML -> Markdown dla RAG")
    ap.add_argument("profil", nargs="?", default="bg", choices=sorted(PROFILE))
    ap.add_argument("--reset", action="store_true", help="zignoruj zapisany stan i zacznij od nowa")
    ap.add_argument("--max-stron", type=int, default=None, help="limit stron w tej sesji (do testów)")
    a = ap.parse_args()
    crawl({**KONFIG_DOMYSLNY, **PROFILE[a.profil]}, reset=a.reset, max_stron=a.max_stron)