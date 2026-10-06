import os
import sys
import time
import requests
from bs4 import BeautifulSoup
import re
from datetime import timedelta
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from markdownify import markdownify as md

# ==========================================
# USTAWIENIA SKRYPTU
# ==========================================
LIMIT_TESTOWY = False
MAX_SYLABUSOW = 5

BAZA_FOLDER = "Baza_Wiedzy_AGH"
DOMAIN = "https://sylabusy.agh.edu.pl"
START_URL = "https://sylabusy.agh.edu.pl/pl/1/2/22/0/0"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}

# ==========================================
# KONFIGURACJA SESJI I RETRY
# ==========================================
session = requests.Session()
session.headers.update(HEADERS)

retry_strategy = Retry(
    total=3,
    connect=3,
    read=3,
    status=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
    respect_retry_after_header=True,
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

# ==========================================
# STATYSTYKI
# ==========================================
stat_pobrane = 0
stat_pominiete = 0
stat_bledy = 0
stat_rozmiar_bajty = 0

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def czysc_nazwe(tekst):
    """Zabezpiecza nazwy plików pod rygorystyczne zasady systemu Windows."""
    nazwa = re.sub(r'[\\/*?:"<>|]', "_", tekst.strip())
    nazwa = nazwa.rstrip(" .")
    
    MAX_DLUGOSC = 120
    if len(nazwa) > MAX_DLUGOSC:
        nazwa = nazwa[:MAX_DLUGOSC].strip() + "..."

    zastrzezone = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "LPT1", "LPT2", "LPT3"}
    if nazwa.upper() in zastrzezone:
        nazwa += "_plik"
        
    return nazwa if nazwa else "nieznana_nazwa"

def pobierz_soup(url):
    """Pobiera stronę i zwraca obiekt BeautifulSoup, obsługując wyjątki."""
    try:
        res = session.get(url, timeout=10)
        res.raise_for_status()
        return BeautifulSoup(res.text, 'html.parser')
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Błąd połączenia z {url}: {e}")
        return None

# ==========================================
# GŁÓWNY PROCES CRAWLERA
# ==========================================
print("Uruchamiam Crawlera Sylabusów\n")
czas_start = time.time()

soup_main = pobierz_soup(START_URL)
if not soup_main:
    print("Nie udało się pobrać strony głównej. Zatrzymuję skrypt.")
    sys.exit(1)

roczniki = {}
rok_header = soup_main.find(string=lambda t: t and "Rozpoczęcie studiów" in t)
if rok_header:
    filter_box = rok_header.find_parent('div', class_='syl-filter-box')
    for li in filter_box.find_all('li', class_='list-group-item'):
        if 'active' in li.get('class', []):
            nazwa = czysc_nazwe(li.text)
            roczniki[nazwa] = START_URL
        else:
            a_tag = li.find('a')
            if a_tag:
                nazwa = czysc_nazwe(a_tag.text)
                roczniki[nazwa] = urljoin(DOMAIN, a_tag.get('href'))

koniec_pracy = False

for rocznik, url_rocznika in roczniki.items():
    if koniec_pracy: break
    print(f"📚 Przetwarzam rocznik: {rocznik}")
    
    soup_rocznik = pobierz_soup(url_rocznika)
    if not soup_rocznik: continue
    
    wydzialy_tagi = soup_rocznik.find_all('div', class_='syl-content-box-header')
    
    for wydzial_tag in wydzialy_tagi:
        if koniec_pracy: break
        nazwa_wydzialu = czysc_nazwe(wydzial_tag.text)
        a_wydzial = wydzial_tag.find_parent('a')
        if not a_wydzial: continue
        
        url_wydzialu = urljoin(DOMAIN, a_wydzial.get('href'))
        print(f"  🏛 Wydział: {nazwa_wydzialu}")
        
        soup_wydzial = pobierz_soup(url_wydzialu)
        if not soup_wydzial: continue
        
        kierunki_naglowki = soup_wydzial.find_all('div', class_='student-view-content-list-heading')
        
        for naglowek in kierunki_naglowki:
            if koniec_pracy: break
            nazwa_kierunku = czysc_nazwe(naglowek.text)
            lista_wariantow = naglowek.find_next_sibling('ul', class_='student-view-content-list')
            
            if not lista_wariantow: continue
            
            for a_tryb in lista_wariantow.find_all('a'):
                if koniec_pracy: break
                typ_studiow = czysc_nazwe(a_tryb.text.replace(', ', '_'))
                url_kierunku = urljoin(DOMAIN, a_tryb.get('href'))
                
                sciezka_zapisu = os.path.join(BAZA_FOLDER, "Sylabusy_Markdown", rocznik, nazwa_wydzialu, nazwa_kierunku, typ_studiow)
                os.makedirs(sciezka_zapisu, exist_ok=True)
                
                soup_przedmioty = pobierz_soup(url_kierunku)
                if not soup_przedmioty: continue
                
                przyciski = soup_przedmioty.find_all('button', attrs={'data-syllabus-id': True})
                
                przedmioty_dict = {}
                for btn in przyciski:
                    syl_id = btn.get('data-syllabus-id')
                    nazwa_przedmiotu = czysc_nazwe(btn.text)
                    if nazwa_przedmiotu and syl_id not in przedmioty_dict:
                        przedmioty_dict[syl_id] = nazwa_przedmiotu
                
                for syl_id, nazwa_przed in przedmioty_dict.items():
                    if koniec_pracy: break
                    
                    sciezka_pliku = os.path.join(sciezka_zapisu, f"{nazwa_przed}.md")
                    
                    if os.path.exists(sciezka_pliku):
                        stat_pominiete += 1
                        continue
                        
                    api_url = urljoin(DOMAIN, f"/pl/document/{syl_id}.html?partialType=modal&forceEpd=false")
                    soup_api = pobierz_soup(api_url)
                    
                    if soup_api:
                        # Pobieramy tylko właściwą zawartość modala, odrzucając zbędny HTML strony (jeśli istnieje)
                        surowy_html = str(soup_api)
                        czysty_markdown = md(surowy_html, heading_style="ATX", strip=['script', 'style'])
                        czysty_markdown = re.sub(r'\n{3,}', '\n\n', czysty_markdown)
                        
                        with open(sciezka_pliku, "w", encoding="utf-8") as f:
                            f.write(czysty_markdown.strip())
                        
                        stat_pobrane += 1
                        stat_rozmiar_bajty += len(czysty_markdown.encode('utf-8'))
                        print(f"    📥 Zapisano MD: {nazwa_przed}")
                        
                        if LIMIT_TESTOWY and stat_pobrane >= MAX_SYLABUSOW:
                            print("\n⚠️ Osiągnięto limit testowy. Zatrzymywanie...")
                            koniec_pracy = True
                    else:
                        stat_bledy += 1
                    
                    time.sleep(0.2)

# ==========================================
# RAPORT KOŃCOWY
# ==========================================
czas_koniec = time.time()
czas_trwania = str(timedelta(seconds=int(czas_koniec - czas_start)))
rozmiar_mb = stat_rozmiar_bajty / (1024 * 1024)
lacznie_przetworzono = stat_pobrane + stat_pominiete + stat_bledy

sredni_czas = (czas_koniec - czas_start) / max(stat_pobrane, 1)

raport = f"""==================================================
RAPORT Z POZYSKIWANIA DANYCH (Do Pracy Inżynierskiej)
==================================================
Data wygenerowania: {time.strftime("%Y-%m-%d %H:%M:%S")}
Czas całkowity wykonania: {czas_trwania} (HH:MM:SS)

Przetworzone pliki łącznie: {lacznie_przetworzono}
  - Pobrane i zapisane: {stat_pobrane}
  - Pominięte (już istniały): {stat_pominiete}
  - Błędy sieciowe/pobierania: {stat_bledy}

Rozmiar pobranych danych (tylko tekst): {rozmiar_mb:.2f} MB
Średni czas całkowitego cyklu na 1 plik: {sredni_czas:.2f} s
==================================================
"""

print(raport)

with open("sylabusy_statystyki.txt", "w", encoding="utf-8") as f:
    f.write(raport)