import os
import time
import requests
from bs4 import BeautifulSoup
import re
import hashlib
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode
from collections import deque
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from markdownify import markdownify as md

# ==========================================
# USTAWIENIA SKRYPTU
# ==========================================
DOMENA = "https://bg.agh.edu.pl/"
FOLDER_ZAPISU = os.path.join("Baza_Wiedzy_AGH", "Biblioteka_Strony")
os.makedirs(FOLDER_ZAPISU, exist_ok=True)

# ==========================================
# KONFIGURACJA SESJI I RETRY
# ==========================================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
session = requests.Session()
session.headers.update(HEADERS)

WYKLUCZONE_SCIEZKI = [
    "/wydarzenia",
    "/aktualnosci",
    "/galeria",
    "/galerie",
    "/konferencje",
    "/archiwum",
    "/blog",
]

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def normalizuj_url(url):
    """Usuwa fragmenty (#), ujednolica linki i czyści parametry śledzące."""
    parts = urlsplit(url)
    
    # Usuwanie parametrów śledzących (utm_*, fbclid), zostawiamy resztę
    query_params = parse_qsl(parts.query)
    czyste_query = [(k, v) for k, v in query_params if not k.startswith('utm_') and k != 'fbclid']
    nowe_query = urlencode(czyste_query)
    
    return urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path.rstrip('/') or '/',
        nowe_query,
        ''  # Usuwamy kotwice
    ))

def czy_nalezy_do_domeny(sprawdzany_url, domena_glowna):
    """Bezpieczne sprawdzanie przynależności do domeny (chroni przed np. agh.edu.pl.evil.com)"""
    return urlsplit(sprawdzany_url).netloc.lower() == urlsplit(domena_glowna).netloc.lower()

def czysc_nazwe_z_url(url):
    nazwa = url.replace("https://", "").replace("http://", "")
    nazwa = re.sub(r'[\\/*?:"<>|]', "_", nazwa)
    
    MAX_DLUGOSC = 80
    if len(nazwa) > MAX_DLUGOSC:
        nazwa = nazwa[:MAX_DLUGOSC].strip()
        
    return nazwa

def czy_pominac_obcy_jezyk(url):
    parts = urlsplit(url)
    path = parts.path.lower()
    query = parts.query.lower()
    
    jezyki = ['/en', '/de', '/ru', '/uk', '/fr', '/es']
    for lang in jezyki:
        if path.startswith(lang + '/') or path == lang:
            return True
            
    if 'lang=' in query and 'lang=pl' not in query:
        return True
        
    return False

def czy_wykluczona_sciezka(url):
    path = urlsplit(url).path.lower().rstrip("/")

    for wykluczona in WYKLUCZONE_SCIEZKI:
        wykluczona = wykluczona.rstrip("/")

        if path == wykluczona or path.startswith(wykluczona + "/"):
            return True

    return False

def generuj_nazwe_pliku(url):
    czysta_nazwa = czysc_nazwe_z_url(url)
    skrot = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{czysta_nazwa}_{skrot}.md"

def wyciagnij_tytul(soup):
    """Próbuje wyciągnąć tytuł z tagu <title> lub <h1>."""
    if soup.title and soup.title.string:
        return soup.title.string.strip().replace('"', '\\"')
    h1 = soup.find('h1')
    if h1 and h1.text:
        return h1.text.strip().replace('"', '\\"')
    return "Brak tytułu"

def wyciagnij_czysty_markdown(soup, url):
    """Usuwa nawigację/stopki, konwertuje na Markdown i dodaje Frontmatter."""
    tytul_strony = wyciagnij_tytul(soup)
    
    for tag in soup(["script", "style", "nav", "footer", "aside", "noscript", "meta"]):
        tag.decompose()
        
    main_content = (
        soup.find('main') or 
        soup.find('article') or 
        soup.find(id=re.compile(r'^(main|content)$', re.I)) or 
        soup.body
    )
    
    if not main_content:
        return ""
        
    czysty_markdown = md(str(main_content), heading_style="ATX", strip=['img']) 
    czysty_markdown = re.sub(r'\n{3,}', '\n\n', czysty_markdown).strip()
    
    if not czysty_markdown:
        return ""
        
    frontmatter = f"""---
url: {url}
domena: {DOMENA}
tytul: "{tytul_strony}"
typ_dokumentu: html
zrodlo: https://bg.agh.edu.pl/
---

"""
    return frontmatter + czysty_markdown

# ==========================================
# GŁÓWNY PROCES CRAWLERA (BFS)
# ==========================================
odwiedzone = set()
do_odwiedzenia = deque([DOMENA])
zapisane_pliki = 0
stat_pominiete = 0

print("🕷 Uruchamiam zoptymalizowany Crawler HTML (Final RAG-Ready)...")

while do_odwiedzenia:
    url = do_odwiedzenia.popleft()
    url = normalizuj_url(url)
    
    if url in odwiedzone:
        continue

    if czy_wykluczona_sciezka(url):
        continue
        
    odwiedzone.add(url)
    sciezka_pliku = os.path.join(FOLDER_ZAPISU, generuj_nazwe_pliku(url))
    
    try:
        res = session.get(url, timeout=10)
        res.raise_for_status()
        
        content_type = res.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            print(f"⏩ Pomijam (to nie HTML): {url}")
            continue
            
    except Exception as e:
        print(f"  ⚠️ Błąd sieciowy na {url}: {e}")
        continue

    soup = BeautifulSoup(res.text, 'html.parser')
    
    if os.path.exists(sciezka_pliku):
        stat_pominiete += 1
        print(f"⏩ Pomijam (już istnieje): {url}")
    else:
        print(f"🔍 Skanuję i zapisuję: {url}")
        tekst_md = wyciagnij_czysty_markdown(soup, url)
        
        if tekst_md:
            with open(sciezka_pliku, "w", encoding="utf-8") as f:
                f.write(tekst_md)
            zapisane_pliki += 1
            
    # Bezpieczne łączenie linków (urljoin rozróżnia ścieżki bezwzględne i względne)
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        nowy_link = urljoin(url, href)
        
        # Filtrujemy tylko linki należące do głównej domeny AGH
        if czy_nalezy_do_domeny(nowy_link, DOMENA):
            nowy_link = normalizuj_url(nowy_link)
            if czy_pominac_obcy_jezyk(nowy_link):
                continue
            if czy_wykluczona_sciezka(nowy_link):
                continue
            if nowy_link not in odwiedzone and nowy_link not in do_odwiedzenia:
                do_odwiedzenia.append(nowy_link)
                
    time.sleep(0.3)

print("\n" + "="*50)
print("📊 RAPORT KOŃCOWY SCRAPERA")
print("="*50)
print(f"🌐 Odwiedzone podstrony: {len(odwiedzone)}")
print(f"📝 Zapisane pliki .md (z metadanymi): {zapisane_pliki}")
print(f"⏩ Pominięte (wznowione): {stat_pominiete}")
print("="*50)