import os
import fitz  # PyMuPDF
import nest_asyncio
from llama_parse import LlamaParse
from dotenv import load_dotenv

# Próba importu konwertera DOCX -> PDF
try:
    from docx2pdf import convert
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    print("⚠️ Brak docx2pdf. Pliki .docx zostaną pominięte. (Zainstaluj: pip install docx2pdf)")

# ==========================================
# USTAWIENIA SKRYPTU
# ==========================================
nest_asyncio.apply()
load_dotenv()
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMAKEY")

# Dostosuj te zmienne do aktualnie przetwarzanego wydziału/strony
DOMENA_ZRODLOWA = "ck.agh.edu.pl" 
FOLDER_SUROWE = os.path.join("Baza_Wiedzy_AGH", "Centrum_Karier_Strony_dokumenty_surowe")
FOLDER_PRZETWORZONE = os.path.join("Baza_Wiedzy_AGH", "Centrum_Karier_dokumenty_przetworzone")
FOLDER_TYMCZASOWY = os.path.join("Baza_Wiedzy_AGH", "Centrum_Karier_dokumenty_tymczasowe")

os.makedirs(FOLDER_PRZETWORZONE, exist_ok=True)
os.makedirs(FOLDER_TYMCZASOWY, exist_ok=True)

parser = LlamaParse(
    result_type="markdown", 
    verbose=False, 
    language="pl"
)

# ==========================================
# FUNKCJE POMOCNICZE
# ==========================================
def czy_wymaga_zaawansowanego_parsowania(sciezka_pdf):
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
        print(f"⚠️ Błąd analizy lokalnej {sciezka_pdf}: {e}. Wymuszam LlamaParse.")
        return True

def wyciagnij_tekst_lokalnie(sciezka_pdf):
    dokument = fitz.open(sciezka_pdf)
    tekst_calosciowy = ""
    for strona in dokument:
        tekst_calosciowy += strona.get_text("text") + "\n\n"
    return tekst_calosciowy

def dodaj_frontmatter(tekst, nazwa_oryginalna, metoda, rozszerzenie="pdf", zrodlo=""):
    frontmatter = f"""---
plik: "{nazwa_oryginalna}"
typ_dokumentu: {rozszerzenie}
zrodlo: {zrodlo}
metoda_ekstrakcji: {metoda}
---

"""
    return frontmatter + tekst

# ==========================================
# GŁÓWNY PROCES HYBRYDOWY
# ==========================================
print("⚙️ Uruchamiam Zaawansowany Parser (PyMuPDF + LlamaParse) z izolacją DOCX...")

licznik_lokalny = 0
licznik_chmurowy = 0
licznik_docx = 0

for nazwa_pliku in os.listdir(FOLDER_SUROWE):
    sciezka_surowa = os.path.join(FOLDER_SUROWE, nazwa_pliku)
    nazwa_bez_rozszerzenia, obecne_rozszerzenie = os.path.splitext(nazwa_pliku)
    obecne_rozszerzenie = obecne_rozszerzenie.lower().replace(".", "")
    
    sciezka_wyniku = os.path.join(FOLDER_PRZETWORZONE, f"{nazwa_bez_rozszerzenia}.md")
    
    if os.path.exists(sciezka_wyniku):
        continue

    sciezka_do_analizy = sciezka_surowa
    czy_plik_tymczasowy = False

    # 1. NORMALIZACJA: DOCX -> PDF w osobnym folderze
    if obecne_rozszerzenie == "docx" and DOCX_SUPPORT:
        sciezka_tymczasowa = os.path.join(FOLDER_TYMCZASOWY, f"{nazwa_bez_rozszerzenia}.pdf")
        print(f"🔄 Normalizuję: {nazwa_pliku} -> PDF (Tymczasowy)")
        try:
            convert(sciezka_surowa, sciezka_tymczasowa)
            sciezka_do_analizy = sciezka_tymczasowa
            czy_plik_tymczasowy = True
            licznik_docx += 1
        except Exception as e:
            print(f"❌ Błąd konwersji Worda: {e}")
            continue

    # 2. PRZETWARZANIE PDF
    if sciezka_do_analizy.lower().endswith(".pdf"):
        print(f"📄 Analizuję: {os.path.basename(sciezka_surowa)}...")
        
        if czy_wymaga_zaawansowanego_parsowania(sciezka_do_analizy):
            print("   ☁️ Wykryto złożony layout. Wysyłam do LlamaParse...")
            try:
                dokumenty = parser.load_data(sciezka_do_analizy)
                tekst_surowy = "\n\n".join([doc.text for doc in dokumenty])
                tekst_koncowy = dodaj_frontmatter(tekst_surowy, nazwa_pliku, "llamaparse", obecne_rozszerzenie, DOMENA_ZRODLOWA)
                licznik_chmurowy += 1
            except Exception as e:
                print(f"   ❌ Błąd LlamaParse: {e}")
                if czy_plik_tymczasowy: os.remove(sciezka_do_analizy)
                continue
        else:
            print("   💻 Czysty tekst. Przetwarzam lokalnie (PyMuPDF)...")
            try:
                tekst_surowy = wyciagnij_tekst_lokalnie(sciezka_do_analizy)
                tekst_koncowy = dodaj_frontmatter(tekst_surowy, nazwa_pliku, "pymupdf", obecne_rozszerzenie, DOMENA_ZRODLOWA)
                licznik_lokalny += 1
            except Exception as e:
                print(f"   ❌ Błąd ekstrakcji lokalnej: {e}")
                if czy_plik_tymczasowy: os.remove(sciezka_do_analizy)
                continue
        
        # 3. Zapis gotowego Markdowna
        with open(sciezka_wyniku, "w", encoding="utf-8") as f:
            f.write(tekst_koncowy)
            
        # 4. Sprzątanie plików tymczasowych
        if czy_plik_tymczasowy and os.path.exists(sciezka_do_analizy):
            os.remove(sciezka_do_analizy)

print("\n" + "="*50)
print("📊 RAPORT Z POTOKU EKSTRAKCJI DOKUMENTÓW")
print("="*50)
print(f"Znormalizowane pliki DOCX: {licznik_docx}")
print(f"Ekstrakcja darmowa (PyMuPDF): {licznik_lokalny} plików")
print(f"Ekstrakcja chmurowa AI (LlamaParse): {licznik_chmurowy} plików")
print("="*50)