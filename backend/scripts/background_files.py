import os
import random
import shutil

KATALOG_BAZY_GLOWNEJ = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Baza_Wiedzy_AGH"
KATALOG_EWALUACJI = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"

LACZNA_LICZBA_PLIKOW_TLA = 170  

def losuj_pliki_tla_zbalansowane():
    os.makedirs(KATALOG_EWALUACJI, exist_ok=True)
    obecne_pliki = set(os.listdir(KATALOG_EWALUACJI))
    
    # Słownik: kategoria (nazwa folderu) -> lista plików .md
    kategorie_plikow = {}
    
    print("🔍 Skanowanie kategorii w bazie...")
    for nazwa_dzialu in os.listdir(KATALOG_BAZY_GLOWNEJ):
        sciezka_dzialu = os.path.join(KATALOG_BAZY_GLOWNEJ, nazwa_dzialu)
        
        if os.path.isdir(sciezka_dzialu):
            kategorie_plikow[nazwa_dzialu] = []
            for root, dirs, files in os.walk(sciezka_dzialu):
                for plik in files:
                    if plik.endswith('.md') and plik not in obecne_pliki:
                        kategorie_plikow[nazwa_dzialu].append(os.path.join(root, plik))
                        
    # Odrzucamy puste katalogi
    kategorie_plikow = {k: v for k, v in kategorie_plikow.items() if v}
    
    if not kategorie_plikow:
        print("⚠️ Brak nowych plików do pobrania.")
        return

    liczba_dzialow = len(kategorie_plikow)
    limit_na_dzial = LACZNA_LICZBA_PLIKOW_TLA // liczba_dzialow
    reszta = LACZNA_LICZBA_PLIKOW_TLA % liczba_dzialow
    
    print(f"📊 Wykryto {liczba_dzialow} głównych działów. Pobieram ok. {limit_na_dzial} plików z każdego.")
    
    wylosowane_sciezki = []
    for dzial, pliki in kategorie_plikow.items():
        # Dodajemy jeden dodatkowy plik do pierwszych kilku działów, żeby rozdać resztę z dzielenia
        ile_pobrac = limit_na_dzial + (1 if reszta > 0 else 0)
        reszta -= 1
        
        pobrane = random.sample(pliki, min(ile_pobrac, len(pliki)))
        wylosowane_sciezki.extend(pobrane)
        print(f" -> {dzial}: wylosowano {len(pobrane)} plików")

    print(f"\n🚀 Kopiowanie łącznie {len(wylosowane_sciezki)} plików...")
    for sciezka_src in wylosowane_sciezki:
        sciezka_dst = os.path.join(KATALOG_EWALUACJI, os.path.basename(sciezka_src))
        if not os.path.exists(sciezka_dst):
            shutil.copy2(sciezka_src, sciezka_dst)

if __name__ == "__main__":
    losuj_pliki_tla_zbalansowane()
    print("✅ Zakończono zbalansowane pobieranie plików tła.")