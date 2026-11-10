# Zbiory ewaluacyjne

Wszystkie testy jakości (`evaluate_*.py`) opierają się na tych samych zbiorach. Ta strona opisuje, jak powstają, co zawierają i — co równie ważne — czego **nie** mierzą.

## 1. Korpus testowy (`eval_dir`)

Katalog `data/Ewaluacja_Probka` (ścieżka z `sites.yaml`) zawiera ok. **200 plików Markdown**, płaskich (bez podfolderów):

* **30 plików bazowych** — lista `BASE_FILES` jest wpisana na sztywno w `generate_qa_dataset.py` (regulaminy, kontakty, cenniki, sylabusy, uchwała rekrutacyjna, FAQ miasteczka);
* **170 plików tła** — losowo dobranych przez `background_files.py` (`TOTAL_BACKGROUND_FILE_COUNT = 170`).

### `background_files.py`

1. Skanuje `data_root`; **kategorią jest każdy folder najwyższego poziomu** (np. `Biblioteka_Strony`, `Biblioteka_Dokumenty_Przetworzone`, `Sylabusy_Markdown` — ten ostatni liczy się jako *jedna* kategoria, mimo podfolderów roczników).
2. Rozdziela 170 plików równo między kategorie (`170 // n`), a resztę po jednym dodatkowym pliku dla pierwszych kategorii.
3. Losuje bez ustawionego ziarna (`random.sample`) — **każde uruchomienie daje inny zestaw**, więc wyników z różnych próbek nie wolno bezpośrednio porównywać.
4. Kopiuje tylko pliki `.md` do `eval_dir` pod **nazwą bazową**; plik, którego nazwa już tam jest, jest pomijany (kolizje nazw z różnych folderów, np. sylabusy o tej samej nazwie, są więc gubione).

Tło ma imitować prawdziwą bazę: dokument docelowy musi „wygrać” z setkami podobnie wyglądających.

## 2. Zbiór główny — `generate_qa_dataset.py` → `test_dataset_balanced.json`

Dla każdego z 30 plików bazowych `gpt-4o` (`temperature=0.8`, `response_format=json_object`) generuje **trzy pytania**, na podstawie **pierwszych 5000 znaków** dokumentu:

| `type` | Instrukcja w prompcie | Intencja |
|---|---|---|
| `lexical` | użyj dokładnych, unikalnych terminów, sygnatur, kodów, kwot, nazw własnych | faworyzować Elasticsearch |
| `semantic` | zapytaj o proces/warunek/cel, **celowo bez** głównych słów kluczowych i nazw własnych (parafrazy) | faworyzować ChromaDB |
| `colloquial` | slang studencki, skróty i niedbały język (stypa, wuef, hajs, oblać…) | faworyzować hybrydę RRF |

Etykiety `type` oznaczają więc **zamierzony charakter pytania**, a nie zmierzoną trudność. Zbiór jest celowo zbilansowany pod kątem różnych algorytmów, co ułatwia ich porównanie, ale powoduje, że wyniki per typ nie są próbą „naturalnych” pytań użytkowników.

Wynik: **90 pytań** (30 × 3), pola `query`, `expected_file`, `type`. Plik jest zapisywany do bieżącego katalogu roboczego (`OUTPUT_FILE` jest ścieżką względną), a repozytorium trzyma go w `backend/scripts/`. Poprawna odpowiedź = **nazwa pliku** (ocena na poziomie dokumentu, nie fragmentu).

Zastrzeżenia: pytania pisze model, który widział dokument, więc „semantyczne” mogą mimo instrukcji zachować jego terminologię; liczność (30 dokumentów, po 3 pytania) sprawia, że przedziały ufności w raportach są szerokie, a pytania z tego samego dokumentu nie są niezależne (dlatego raporty stosują bootstrap po dokumentach).

## 3. Zbiór odpowiedzi wzorcowych — `generate_llm_eva_dataset.py`

Dla każdego z 90 pytań model **`o3`** (model rozumujący, **nie przyjmuje parametru `temperature`**; jego wyniki nie są przez to deterministyczne) odpowiada na podstawie **pełnej treści** pliku i w co najwyżej 3–4 zdaniach. Rekord wyjściowy: `query`, `type`, `expected_file`, `context` (cały plik), `golden_answer`.

* Wznawianie: istniejące rekordy (po tekście pytania) są pomijane; pierwszy błąd API przerywa pętlę; przerwa 1 s między wywołaniami.
* **Rozbieżność nazw plików:** skrypt zapisuje `llm_golden_dataset.json`, a `evaluate_models.py` (i repozytorium) używa `llm_eva_dataset.json` (90 rekordów). Po wygenerowaniu plik trzeba ręcznie zmienić nazwę lub poprawić stałą.
* Pytania powstały z początku dokumentu (5000 zn.), a wzorzec z całości — odpowiedź wzorcowa może więc zawierać informacje, których pytanie nie zakładało.

## 4. Zbiór pełny — `generate_full_dataset.py` → `full_db_dataset.json`

W odróżnieniu od §2 próbkuje pliki z **prawdziwego indeksu Elasticsearch**, równo po źródłach (`zrodlo`), po `--per-source` (domyślnie 3) plików na źródło, pomijając pliki krótsze niż `--min-chars` (800). Argumenty: `--seed` (42), `--model` (`gpt-4o`), `--out`, `--yes`, `--refresh-inventory` oraz parametry połączenia. Przed wywołaniami LLM wypisuje ich liczbę i pyta o zgodę.

Zbiór w repozytorium: **366 pytań, 122 pliki, 41 źródeł**; rekord ma dodatkowo `expected_plik` (pełna ścieżka z indeksu, rozstrzyga kolizje nazw) i `source`. Próbkowanie jest równe na źródło, **nie proporcjonalne** do rozmiaru (sylabusy są niedoreprezentowane w stosunku do udziału w indeksie), dlatego raporty należy czytać per źródło, nie tylko po średniej globalnej.

> **Brak pliku w repozytorium:** skrypt i jego docstring odwołują się do `evaluate_production_full.py`, którego nie ma w archiwum. Zbiór jest więc gotowy, ale narzędzie, które miało go zużywać, nie jest dołączone.

## 5. Który zbiór do czego

| Zbiór | Używany przez |
|---|---|
| `test_dataset_balanced.json` (90) + `eval_dir` | `evaluate_chunking`, `evaluate_embeddings`, `evaluate_metadata`, `evaluate_search`, `evaluate_production` |
| `llm_eva_dataset.json` (90) | `evaluate_models` |
| `full_db_dataset.json` (366) | *brak dołączonego konsumenta* (patrz §4) |

## 6. Czego te zbiory nie mierzą

* Pytań wielokrokowych i z historią rozmowy.
* Pytań, na które baza nie zawiera odpowiedzi (zachowanie odmowy).
* Pytań o treść w środku długich dokumentów (pytania powstały z pierwszych 5000 znaków).
* Jakości cytowań i poprawności linków.