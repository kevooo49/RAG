# Dokumentacja Architektoniczna: Tworzenie Zbioru Ewaluacyjnego (Data Collection & Ground Truth)

## 1. Wprowadzenie i Rola Środowiska Testowego

Aby zapewnić miarodajną i powtarzalną ewaluację modeli językowych oraz wektorowych (odpowiedzialnych za retrieval w architekturze RAG), konieczne było stworzenie statycznego, zablokowanego wycinka danych korpusu. Trzy moduły analityczne: `background_files.py`, `generate_qa_dataset.py` oraz `generate_llm_eva_dataset.py` odpowiadają za pełny cykl preparacji danych testowych – od wyselekcjonowania reprezentatywnych dokumentów, przez wygenerowanie zbalansowanych pytań o zróżnicowanym stopniu trudności lingwistycznej, aż po wygenerowanie referencyjnych odpowiedzi modelu bazowego.

## 2. Losowanie Korpusu Testowego (`background_files.py`)

Skrypt ten gwarantuje, że do testów trafi zbalansowana próbka dokumentów z różnych wydziałów i domen uczelni.

* **Integracja Konfiguracji**: Skrypt bezpośrednio korzysta z konfiguracji `sites.yaml`, mapując główną bazę danych (`MAIN_DATABASE_DIRECTORY`) oraz ścieżkę folderu ewaluacyjnego (`EVALUATION_DIRECTORY`).
* **Zrównoważone Próbkowanie (Stratified Sampling)**: Algorytm najpierw skanuje wszystkie katalogi głównej bazy wiedzy, traktując je jako rozdzielne kategorie (straty). Dzieli pulę 170 plików docelowych (`TOTAL_BACKGROUND_FILE_COUNT`) równomiernie przez liczbę znalezionych kategorii, zapewniając, że do próbki trafią w równym stopniu regulaminy ze strefy Studenta, jak i techniczne sylabusy wydziałowe.
* **Idempotentność Działania**: Moduł weryfikuje już istniejące na dysku pliki za pomocą zrzutu struktury folderów (`set(os.listdir)`), dobudowując braki. Kopiowanie odbywa się przy użyciu `shutil.copy2`, aby zachować oryginalne metadane plików dla poprawności statystyk systemowych.

## 3. Generowanie Zapytań Opartych na Heurystykach Wyszukiwarki (`generate_qa_dataset.py`)

Skrypt odpowiedzialny za wytworzenie ustrukturyzowanego pliku JSON z zapytaniami używanymi do rygorystycznych testów porównawczych silników wektorowych i wyszukiwarki słownikowej.

* **Wykorzystanie Modelu Instrukcyjnego**: Pytania generowane są automatycznie przez model GPT-4o (`gpt-4o`), z precyzyjnie zaaplikowaną inżynierią promptu, symulującą zachowanie studentów. Narzucony został ścisły format odpowiedzi wymuszony przez opcję `response_format={"type": "json_object"}`.
* **Trójdrożny Podział Zapytań**: Każdy dokument (reprezentowany przez wycinek pierwszych 5000 znaków) służy do wygenerowania dokładnie 3 pytań, zaprojektowanych w celu faworyzowania (obciążania) konkretnych algorytmów systemu RAG:
1. *Lexical (Precyzyjne)* – faworyzujące klasyczny Elasticsearch. Wymaga wyłuskania kodów, numerów dzienników czy kwot.
2. *Semantic (Opisowe)* – faworyzujące wyszukiwanie gęste (ChromaDB / Dense). Zadaniem modelu jest unikanie kluczowych terminów, stosując parafrazę.
3. *Colloquial (Potoczne)* – faworyzujące wyszukiwanie hybrydowe (RRF). Zapytania korzystają ze slangu i skrótów uczelnianych (np. "wuef", "warunek", "dziekanka").


* **Ustrukturyzowany Eksport**: Skrypt buduje plik `test_dataset_balanced.json`, zagnieżdżając w nim słowniki wiążące treść pytania, przypisaną strategię algorytmu oraz referencyjną nazwę pliku, będącą kluczem (Ground Truth) w ewaluacji metryki *Hit@k*.

## 4. Wyznaczanie Standardu Odpowiedzi / Ground Truth (`generate_llm_eva_dataset.py`)

Końcowym etapem jest przygotowanie zestawu idealnych odpowiedzi systemu, stanowiących referencję przy mierzeniu ewentualnych halucynacji w głównym RAG-u. Wykorzystano model klasy analitycznej *o3* (O-series reasoning).

* **Model Rozumujący i Detachment (o3)**: Ze względu na to, że model referencyjny nie ma prawa wnosić własnej wiedzy o świecie (tzw. "out-of-domain knowledge"), użyto o3 – nie obsługuje on parametru temperatury (zablokowany na zero / powtarzalność absolutna).
* **Architektura Strict Prompting**: Prompt wymusza na modelu zachowanie w formie klasycznego, rzetelnego asystenta. Zasady narzucają kategoryczny zakaz dedukcji informacji spoza dostarczonego `<dokumentu>` i wymuszają profesjonalny język (nawet przy zapytaniu typu *Colloquial*).
* **Pamięć Postępu i Przewidywalność (Resuming)**: Operacja ta generuje duże obciążenie dla drogich i limitowanych punktów dostępowych (endpoints) modeli serii O. Skrypt implementuje kontrolę błędów; przed wywołaniem API wczytuje dotychczasowy dorobek z pliku `llm_golden_dataset.json` (przy użyciu `load_progress`) i w razie błędu sieci przerywa działanie, umożliwiając łagodne wznowienie procedury od ostatnio przetworzonego zapytania.