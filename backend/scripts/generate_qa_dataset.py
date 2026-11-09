
import os
import json
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
from sites import load_or_exit  # noqa: E402  # type: ignore
EVALUATION_DIRECTORY = str(load_or_exit().eval_dir)
OUTPUT_FILE = "test_dataset_balanced.json"

BASE_FILES = [
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


def generate_questions():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not client.api_key:
        print("Missing OPENAI_API_KEY.")
        return

    test_dataset = []

    print(f"Starting test question generation for {len(BASE_FILES)} files...\n")

    for idx, file_name in enumerate(BASE_FILES, 1):
        file_path = os.path.join(EVALUATION_DIRECTORY, file_name)

        if not os.path.exists(file_path):
            print(f"⚠️ File {file_name} not found, skipping...")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            truncated_content = content[:5000]

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
    {{"query": "pytanie leksykalne...", "expected_file": "{file_name}", "type": "lexical"}},
    {{"query": "pytanie semantyczne...", "expected_file": "{file_name}", "type": "semantic"}},
    {{"query": "pytanie potoczne...", "expected_file": "{file_name}", "type": "colloquial"}}
  ]
}}

Tekst źródłowy:
{truncated_content}
"""

        print(f"[{idx}/{len(BASE_FILES)}] Generating questions for: {file_name}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                response_format={"type": "json_object"}
            )

            response_text = response.choices[0].message.content.strip()
            generated_questions = json.loads(response_text)

            if isinstance(generated_questions, dict) and "pytania" in generated_questions:
                generated_questions = generated_questions["pytania"]
            elif isinstance(generated_questions, dict):
                key = list(generated_questions.keys())[0]
                generated_questions = generated_questions[key]

            test_dataset.extend(generated_questions)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    print(f"\nGenerated {len(test_dataset)} balanced questions and saved them to {OUTPUT_FILE}.")


if __name__ == "__main__":
    generate_questions()
