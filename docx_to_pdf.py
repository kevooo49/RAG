import os
from docx2pdf import convert

FOLDER_SUROWE = os.path.join("Baza_Wiedzy_AGH", "Biblioteka_Dokumenty_Surowe")

print("🔄 Rozpoczynam normalizację: Konwersja plików DOCX do PDF...")

przekonwertowane = 0
for nazwa_pliku in os.listdir(FOLDER_SUROWE):
    if nazwa_pliku.lower().endswith('.docx'):
        sciezka_docx = os.path.join(FOLDER_SUROWE, nazwa_pliku)
        sciezka_pdf = os.path.join(FOLDER_SUROWE, nazwa_pliku.replace('.docx', '.pdf').replace('.DOCX', '.pdf'))
        
        # Konwertujemy tylko, jeśli PDF jeszcze nie istnieje
        if not os.path.exists(sciezka_pdf):
            try:
                print(f"📄 Konwertuję: {nazwa_pliku}")
                convert(sciezka_docx, sciezka_pdf)
                przekonwertowane += 1
                
                # Opcjonalnie: usunięcie oryginalnego DOCX po sukcesie
                os.remove(sciezka_docx) 
            except Exception as e:
                print(f"❌ Błąd konwersji {nazwa_pliku}: {e}")

print(f"\n✅ Zakończono! Przekonwertowano {przekonwertowane} plików DOCX na PDF.")