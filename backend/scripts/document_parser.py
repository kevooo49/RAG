import os
import json
import time
import requests
import fitz
from dotenv import load_dotenv

try:
    from docx2pdf import convert
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️ Brak docx2pdf. Pliki .docx zostaną pominięte. (Zainstaluj: pip install docx2pdf)")

load_dotenv()
API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

def czy_wymaga_zaawansowanego_parsowania(sciezka_pdf: str) -> bool:
    try:
        with fitz.open(sciezka_pdf) as dokument:
            for strona in dokument:
                tekst = strona.get_text("text").strip()
                if len(tekst) < 50:
                    return True
      
                obrazy = strona.get_images()
                if len(obrazy) > 2:
                    return True
                
                tabele = strona.find_tables()
                if tabele and len(tabele.tables) > 0:
                    return True
        return False
    except Exception as e:
        print(f"⚠️ Błąd analizy lokalnej {sciezka_pdf}: {e}. Wymuszam LlamaParse.")
        return True

def wyciagnij_tekst_lokalnie(sciezka_pdf: str) -> str:
    tekst_calosciowy = ""
    with fitz.open(sciezka_pdf) as dokument:
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
    if not klucz_api:
         raise Exception("Brak klucza LLAMA_CLOUD_API_KEY")
    url_upload = "https://api.cloud.llamaindex.ai/api/v2/parse/upload"
    headers = {"Authorization": f"Bearer {klucz_api}"}
    
    with open(sciezka_do_pliku, "rb") as f:
        config = {"tier": "agentic", "version": "latest"}
        response = requests.post(
            url_upload, 
            headers=headers, 
            files={"file": f},
            data={"configuration": json.dumps(config)},
            timeout=30
        )
        
    if not response.ok:
        raise Exception(f"Błąd połączenia z API v2: {response.text}")
        
    job_id = response.json()["id"]
    url_status = f"https://api.cloud.llamaindex.ai/api/v2/parse/{job_id}?expand=markdown_full"
    
    max_prob = 60 
    for _ in range(max_prob):
        res = requests.get(url_status, headers=headers, timeout=20)
        if not res.ok:
            raise Exception(f"Błąd sprawdzania statusu: {res.text}")
            
        dane = res.json()
        status = dane["job"]["status"]
        
        if status == "COMPLETED":
            return dane.get("markdown_full", "")
        elif status in ["FAILED", "CANCELLED"]:
            error_msg = dane["job"].get("error_message", "Nieznany błąd")
            raise Exception(f"Zadanie odrzucone przez serwer: {error_msg}")
            
        time.sleep(2)
        
    raise Exception("Timeout zadania LlamaParse: proces trwał zbyt długo.")

def przetworz_katalog(folder_surowe: str, folder_przetworzone: str, folder_tymczasowy: str, domena_zrodlowa: str):
    os.makedirs(folder_przetworzone, exist_ok=True)
    os.makedirs(folder_tymczasowy, exist_ok=True)

    print("⚙️ Uruchamiam Zaawansowany Parser (PyMuPDF + LlamaParse) z izolacją DOCX...")
    licznik_lokalny, licznik_chmurowy, licznik_docx = 0, 0, 0

    for nazwa_pliku in os.listdir(folder_surowe):
        sciezka_surowa = os.path.join(folder_surowe, nazwa_pliku)
        if not os.path.isfile(sciezka_surowa):
            continue
            
        nazwa_bez_rozszerzenia, obecne_rozszerzenie = os.path.splitext(nazwa_pliku)
        obecne_rozszerzenie = obecne_rozszerzenie.lower().replace(".", "")
        sciezka_wyniku = os.path.join(folder_przetworzone, f"{nazwa_bez_rozszerzenia}.md")
        
        if obecne_rozszerzenie not in ["pdf", "docx"]:
            print(f"Pominięto {nazwa_pliku} - nieobsługiwany format.")
            continue
        
        if os.path.exists(sciezka_wyniku) and os.path.getsize(sciezka_wyniku) > 150:
            continue

        sciezka_do_analizy = sciezka_surowa
        czy_plik_tymczasowy = False

        if obecne_rozszerzenie == "docx" and DOCX_SUPPORT:
            sciezka_tymczasowa = os.path.join(folder_tymczasowy, f"{nazwa_bez_rozszerzenia}.pdf")
            print(f"🔄 Normalizuję: {nazwa_pliku} -> PDF")
            try:
                convert(sciezka_surowa, sciezka_tymczasowa)
                sciezka_do_analizy = sciezka_tymczasowa
                czy_plik_tymczasowy = True
                licznik_docx += 1
            except Exception as e:
                print(f"❌ Błąd konwersji Worda: {e}")
                continue

        if sciezka_do_analizy.lower().endswith(".pdf"):
            print(f"📄 Analizuję: {os.path.basename(sciezka_surowa)}...")
            
            if czy_wymaga_zaawansowanego_parsowania(sciezka_do_analizy):
                print("   ☁️ Wysyłam do LlamaParse...")
                try:
                    tekst_surowy = wykonaj_llamaparse_v2(sciezka_do_analizy, API_KEY)
                    tekst_koncowy = dodaj_frontmatter(tekst_surowy, nazwa_pliku, "llamaparse", obecne_rozszerzenie, domena_zrodlowa)
                    licznik_chmurowy += 1
                except Exception as e:
                    print(f"   ❌ Błąd LlamaParse: {e}")
                    if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy): 
                        os.remove(sciezka_do_analizy)
                    continue
            else:
                print("   💻 Przetwarzam lokalnie...")
                try:
                    tekst_surowy = wyciagnij_tekst_lokalnie(sciezka_do_analizy)
                    tekst_koncowy = dodaj_frontmatter(tekst_surowy, nazwa_pliku, "pymupdf", obecne_rozszerzenie, domena_zrodlowa)
                    licznik_lokalny += 1
                except Exception as e:
                    print(f"   ❌ Błąd ekstrakcji lokalnej: {e}")
                    if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy): 
                        os.remove(sciezka_do_analizy)
                    continue
            
            with open(sciezka_wyniku, "w", encoding="utf-8") as f:
                f.write(tekst_koncowy)
                
            if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy):
                os.remove(sciezka_do_analizy)

    print("\n📊 RAPORT:")
    print(f"DOCX->PDF: {licznik_docx} | Lokalne (Darmowe): {licznik_lokalny} | Chmura AI: {licznik_chmurowy}")

if __name__ == "__main__":
    przetworz_katalog(
        folder_surowe="C:\\Users\\micro\\Desktop\\Bachelors\\RAG\\data\\Baza_Wiedzy_AGH\\COK_Dokumenty_Surowe",
        folder_przetworzone="C:\\Users\\micro\\Desktop\\Bachelors\\RAG\\data\\Baza_Wiedzy_AGH\\COK_Dokumenty_Przetworzone",
        folder_tymczasowy="C:\\Users\\micro\\Desktop\\Bachelors\\RAG\\data\\Baza_Wiedzy_AGH\\COK_Dokumenty_Tymczasowe",
        domena_zrodlowa="https://www.cok.agh.edu.pl/"
    )