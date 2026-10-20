import os
import json
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

KATALOG_EWALUACJI = r"C:\Users\micro\Desktop\Bachelors\RAG\data\Ewaluacja_Probka"
PLIK_WYNIKOWY = "test_dataset_balanced.json"

BAZOWE_PLIKI = [
    "Audyt energetyczny budynku i przedsiębiorstwa.md",
    "badania-lekarskie_3f0aa992.md",
    "dla-studentow__konsultacje-zawodowe_79452be3.md",
    "dla-studentow__praktyki_3c93d955.md",
    "dokumenty__numer-konta-do-wnoszenia-oplat_c339f9de.md",
    "Ekonomia.md",
    "erasmus-doktorant__zasady-dotyczace-wyjazdow-doktorantow-agh-na-studia-i-praktyk_f0c12458.md",
    "jak-korzystac__regulaminy-i-cennik__szafki-skrytkowe_84be3288.md",
    "jak-korzystac__zapis-do-biblioteki_01cf6b4c.md",
    "kontakt__godziny-otwarcia_12fa9fe4.md",
    "kredyty-studenckie_391165f1.md",
    "Krystalizacja.md",
    "lektoraty__wydawanie-certyfikatow-jezykowych-agh_5b0f2052.md",
    "Muzealnictwo i wystawiennictwo.md",
    "obiekty-i-wynajem__cennik-i-wynajem_101e666d.md",
    "olimpijczycy_b2005775.md",
    "organizacje-studenckie__zakupy-za-granica_275f11ec.md",
    "pl__offer__how-to-use-the-resources_bd24b31e.md",
    "Programowanie zaawansowane 1.md",
    "REGULAMIN-SCIANY-WSPINACZKOWEJ_1_c7ed5a1a.md",
    "REGULAMIN_ERASMUS_2026_e687872f.md",
    "Regulamin_swiadczen_dla_studentow_AGH_2026.md",
    "Rejestracja_do_grup_z_Wychowania_Fizycznego_w_semestrze_zimowym_26-27_c4b53c96.md",
    "swiadczenia__stypendium-z-wlasnego-funduszu-na-stypendia_898cd999.md",
    "sylabus-francuski-b2-1_3_032889bc.md",
    "uchwala_2026_2027_warunki_rekrutacji.md",
    "warunki-rekrutacji_3fc3d539.md",
    "www.miasteczko.agh.edu.pl_faq.html_26e7b95c.md",
    "www.miasteczko.agh.edu.pl_strefa-mieszkanca_cennik.html_15c5ed3e.md",
    "zarz_rek_zasady_zaliczeni_WF43_2025_181b6db0.md"
]

def generuj_pytania():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    if not client.api_key:
        print("❌ Brak klucza OPENAI_API_KEY.")
        return

    zbior_testowy = []
    
    print(f"🚀 Rozpoczynam generowanie pytań testowych dla {len(BAZOWE_PLIKI)} plików...\n")

    for idx, nazwa_pliku in enumerate(BAZOWE_PLIKI, 1):
        sciezka = os.path.join(KATALOG_EWALUACJI, nazwa_pliku)
        
        if not os.path.exists(sciezka):
            print(f"⚠️ Nie znaleziono pliku {nazwa_pliku}, pomijam...")
            continue
            
        with open(sciezka, "r", encoding="utf-8") as f:
            tresc = f.read()
            tresc_skrocona = tresc[:5000] 

        prompt = f"""
Jesteś inżynierem AI budującym system ewaluacyjny RAG dla uczelni AGH. Twój cel to wygenerowanie 3 pytań testowych na podstawie poniższego fragmentu dokumentu.
Pytania muszą celowo faworyzować różne algorytmy wyszukiwania, aby zbiór był zbalansowany.

Wygeneruj dokładnie 3 pytania, po jednym dla każdej strategii:
1. "lexical" (Precyzyjne): Użyj dokładnych, unikalnych terminów, sygnatur, kodów, kwot lub nazw własnych z tekstu. (Ma na celu faworyzowanie Elasticsearch).
2. "semantic" (Opisowe): Zapytaj o dany proces, warunek lub cel, ale CELOWO NIE UŻYWAJ głównych słów kluczowych ani nazw własnych z tekstu. Użyj silnych parafraz i omówień. (Ma na celu faworyzowanie ChromaDB).
3. "colloquial" (Potoczne): Zapytaj o konkretne informacje, używając slangu studenckiego, potocznych słów, skrótów i naturalnego, niedbałego języka (np. stypa, wuef, warunek, dziekanat, hajs, akademik, oblać). (Ma na celu faworyzowanie RRF/Hybrydy).

Zwróć wynik WYŁĄCZNIE jako poprawny obiekt JSON zawierający tablicę "pytania" w następującym formacie:
{{
  "pytania": [
    {{"query": "pytanie leksykalne...", "expected_file": "{nazwa_pliku}", "type": "lexical"}},
    {{"query": "pytanie semantyczne...", "expected_file": "{nazwa_pliku}", "type": "semantic"}},
    {{"query": "pytanie potoczne...", "expected_file": "{nazwa_pliku}", "type": "colloquial"}}
  ]
}}

Tekst źródłowy:
{tresc_skrocona}
"""
        
        print(f"[{idx}/{len(BAZOWE_PLIKI)}] Generowanie dla: {nazwa_pliku}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",  # Zmiana modelu na potężniejszy do lepszej generacji złożeń semantycznych
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8, # Lekko podwyższona temperatura dla lepszych parafraz i słownictwa potocznego
                response_format={"type": "json_object"}
            )
            
            odp_tekst = response.choices[0].message.content.strip()
            wygenerowane_pytania = json.loads(odp_tekst)
            
            if isinstance(wygenerowane_pytania, dict) and "pytania" in wygenerowane_pytania:
                wygenerowane_pytania = wygenerowane_pytania["pytania"]
            elif isinstance(wygenerowane_pytania, dict): 
                klucz = list(wygenerowane_pytania.keys())[0]
                wygenerowane_pytania = wygenerowane_pytania[klucz]

            zbior_testowy.extend(wygenerowane_pytania)
            
        except Exception as e:
            print(f"❌ Błąd przy pliku {nazwa_pliku}: {e}")

    with open(PLIK_WYNIKOWY, "w", encoding="utf-8") as f:
        json.dump(zbior_testowy, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Sukces! Wygenerowano {len(zbior_testowy)} zbalansowanych pytań i zapisano w {PLIK_WYNIKOWY}.")

if __name__ == "__main__":
    generuj_pytania()