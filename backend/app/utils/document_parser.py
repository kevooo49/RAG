import os
import fitz  # PyMuPDF
import nest_asyncio
from llama_cloud_services import LlamaParse
import requests
import json
import time
from app.core.config import settings
from app.core.logging import logger

# Próba importu konwertera DOCX -> PDF
try:
    from docx2pdf import convert
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logger.warning("⚠️ Brak docx2pdf. Pliki .docx zostaną pominięte. (Zainstaluj: pip install docx2pdf)")

# ==========================================
# USTAWIENIA SKRYPTU
# ==========================================
nest_asyncio.apply()

# Inicjalizacja parsera używając klucza z naszego bezpiecznego configu
parser = LlamaParse(
    api_key=settings.LLAMA_CLOUD_API_KEY,
    result_type="markdown", 
    verbose=False, 
    language="pl"
)

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def czy_wymaga_zaawansowanego_parsowania(sciezka_pdf: str) -> bool:
    try:
        dokument = fitz.open(sciezka_pdf)
        for strona in dokument:
            tekst = strona.get_text("text").strip()
            if len(tekst) < 50:
                return True
            if len(strona.get_images()) > 0:
                return True
            tabele = strona.find_tables()
            if tabele and len(tabele.tables) > 0:
                return True
        return False
    except Exception as e:
        logger.warning(f"⚠️ Błąd analizy lokalnej {sciezka_pdf}: {e}. Wymuszam LlamaParse.")
        return True

def wyciagnij_tekst_lokalnie(sciezka_pdf: str) -> str:
    dokument = fitz.open(sciezka_pdf)
    tekst_calosciowy = ""
    for strona in dokument:
        tekst_calosciowy += strona.get_text("text") + "\n\n"
    return tekst_calosciowy

def dodaj_frontmatter(tekst: str, nazwa_oryginalna: str, metoda: str, rozszerzenie: str="pdf", zrodlo: str="") -> str:
    frontmatter = f"""---
plik: "{nazwa_oryginalna}"
typ_dokumentu: {rozszerzenie}
zrodlo: {zrodlo}
metoda_ekstrakcji: {metoda}
---

"""
    return frontmatter + tekst

def wykonaj_llamaparse_v2(sciezka_do_pliku: str, klucz_api: str) -> str:
    url_upload = "https://api.cloud.llamaindex.ai/api/v2/parse/upload"
    headers = {"Authorization": f"Bearer {klucz_api}"}
    
    # 1. Upload pliku i utworzenie zadania
    with open(sciezka_do_pliku, "rb") as f:
        config = {"tier": "agentic", "version": "latest"}
        response = requests.post(
            url_upload, 
            headers=headers, 
            files={"file": f},
            data={"configuration": json.dumps(config)}
        )
        
    if not response.ok:
        raise Exception(f"Błąd połączenia z API v2: {response.text}")
        
    job_id = response.json()["id"]
    
    # 2. Odpytywanie o status i pobranie pełnego Markdowna (Polling)
    # Parametr expand=markdown_full zwraca od razu połączony tekst ze wszystkich stron
    url_status = f"https://api.cloud.llamaindex.ai/api/v2/parse/{job_id}?expand=markdown_full"
    
    while True:
        res = requests.get(url_status, headers=headers)
        if not res.ok:
            raise Exception(f"Błąd sprawdzania statusu: {res.text}")
            
        dane = res.json()
        status = dane["job"]["status"]
        
        if status == "COMPLETED":
            return dane.get("markdown_full", "")
        elif status in ["FAILED", "CANCELLED"]:
            error_msg = dane["job"].get("error_message", "Nieznany błąd")
            raise Exception(f"Zadanie odrzucone przez serwer: {error_msg}")
            
        time.sleep(2) # Czekamy 2 sekundy przed ponownym sprawdzeniem statusu

# ==========================================
# GŁÓWNY PROCES HYBRYDOWY
# ==========================================
def przetworz_katalog(folder_surowe: str, folder_przetworzone: str, folder_tymczasowy: str, domena_zrodlowa: str):
    """
    Skanuje folder_surowe, konwertuje DOCX, ocenia stopień skomplikowania PDF
    i ekstrahuje tekst do Markdown przy użyciu PyMuPDF lub LlamaParse.
    """
    os.makedirs(folder_przetworzone, exist_ok=True)
    os.makedirs(folder_tymczasowy, exist_ok=True)

    logger.info("⚙️ Uruchamiam Zaawansowany Parser (PyMuPDF + LlamaParse) z izolacją DOCX...")

    licznik_lokalny = 0
    licznik_chmurowy = 0
    licznik_docx = 0

    for nazwa_pliku in os.listdir(folder_surowe):
        sciezka_surowa = os.path.join(folder_surowe, nazwa_pliku)
        
        # Omijanie podkatalogów, w razie gdyby takie były
        if not os.path.isfile(sciezka_surowa):
            continue
            
        nazwa_bez_rozszerzenia, obecne_rozszerzenie = os.path.splitext(nazwa_pliku)
        obecne_rozszerzenie = obecne_rozszerzenie.lower().replace(".", "")
        
        sciezka_wyniku = os.path.join(folder_przetworzone, f"{nazwa_bez_rozszerzenia}.md")
        
        if os.path.exists(sciezka_wyniku):
            if os.path.getsize(sciezka_wyniku) > 150:
                continue

        sciezka_do_analizy = sciezka_surowa
        czy_plik_tymczasowy = False

        # 1. NORMALIZACJA: DOCX -> PDF w osobnym folderze
        if obecne_rozszerzenie == "docx" and DOCX_SUPPORT:
            sciezka_tymczasowa = os.path.join(folder_tymczasowy, f"{nazwa_bez_rozszerzenia}.pdf")
            logger.info(f"🔄 Normalizuję: {nazwa_pliku} -> PDF (Tymczasowy)")
            try:
                convert(sciezka_surowa, sciezka_tymczasowa)
                sciezka_do_analizy = sciezka_tymczasowa
                czy_plik_tymczasowy = True
                licznik_docx += 1
            except Exception as e:
                logger.error(f"❌ Błąd konwersji Worda: {e}")
                continue

        # 2. PRZETWARZANIE PDF
        if sciezka_do_analizy.lower().endswith(".pdf"):
            logger.info(f"📄 Analizuję: {os.path.basename(sciezka_surowa)}...")
            
            if czy_wymaga_zaawansowanego_parsowania(sciezka_do_analizy):
                logger.info("   ☁️ Wykryto złożony layout. Wysyłam do LlamaParse...")
                try:
                    tekst_surowy = wykonaj_llamaparse_v2(sciezka_do_analizy, settings.LLAMA_CLOUD_API_KEY)
                    tekst_koncowy = dodaj_frontmatter(tekst_surowy, nazwa_pliku, "llamaparse", obecne_rozszerzenie, domena_zrodlowa)
                    licznik_chmurowy += 1
                except Exception as e:
                    logger.error(f"   ❌ Błąd LlamaParse: {e}")
                    if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy): 
                        os.remove(sciezka_do_analizy)
                    continue
            else:
                logger.info("   💻 Czysty tekst. Przetwarzam lokalnie (PyMuPDF)...")
                try:
                    tekst_surowy = wyciagnij_tekst_lokalnie(sciezka_do_analizy)
                    tekst_koncowy = dodaj_frontmatter(tekst_surowy, nazwa_pliku, "pymupdf", obecne_rozszerzenie, domena_zrodlowa)
                    licznik_lokalny += 1
                except Exception as e:
                    logger.error(f"   ❌ Błąd ekstrakcji lokalnej: {e}")
                    if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy): 
                        os.remove(sciezka_do_analizy)
                    continue
            
            # 3. Zapis gotowego Markdowna
            with open(sciezka_wyniku, "w", encoding="utf-8") as f:
                f.write(tekst_koncowy)
                
            # 4. Sprzątanie plików tymczasowych
            if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy):
                os.remove(sciezka_do_analizy)

    logger.info("="*50)
    logger.info("📊 RAPORT Z POTOKU EKSTRAKCJI DOKUMENTÓW")
    logger.info("="*50)
    logger.info(f"Znormalizowane pliki DOCX: {licznik_docx}")
    logger.info(f"Ekstrakcja darmowa (PyMuPDF): {licznik_lokalny} plików")
    logger.info(f"Ekstrakcja chmurowa AI (LlamaParse): {licznik_chmurowy} plików")
    logger.info("="*50)


if __name__ == "__main__":
    przetworz_katalog(
        folder_surowe="C:\\Users\\micro\\Desktop\\Bachelors\\RAG\\data\\Baza_Wiedzy_AGH\\SJO_Dokumenty_Surowe",
        folder_przetworzone="C:\\Users\\micro\\Desktop\\Bachelors\\RAG\\data\\Baza_Wiedzy_AGH\\SJO_Dokumenty_Przetworzone",
        folder_tymczasowy="C:\\Users\\micro\\Desktop\\Bachelors\\RAG\\data\\Baza_Wiedzy_AGH\\SJO_Dokumenty_Tymczasowe",
        domena_zrodlowa="https://sjo.agh.edu.pl/"
    )