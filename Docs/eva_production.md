# Ewaluacja zintegrowanego stosu (`evaluate_production.py`)

W odróżnieniu od testów pojedynczych komponentów ten skrypt sprawdza **cały potok wyszukiwania naraz**: asymetryczne indeksowanie, hybrydę RRF i rozszerzanie zapytań. Zbiory: [`eva_datasets.md`](eva_datasets.md). Wyniki: `production_eva_stats/` — trzy przebiegi (patrz §3).

## 1. Konfiguracja skryptu (stan kodu)

| Element | Wartość |
|---|---|
| Embeddingi | `text-embedding-3-small` (stała `EMBED_MODEL`) |
| Rozszerzanie zapytań | `gpt-4o-mini`, własny prompt skryptu (zachowaj oryginał + dopisz 3–5 słów kluczowych) |
| Indeksy | Chroma `eval_production` (cosine), ES `eval_production_idx` (analizator `polish`) |
| Pula kandydatów / RRF | 40 chunków na ścieżkę, `k = 60` (jak w produkcji) |
| Statystyka | cluster bootstrap po plikach, 2000 prób, ziarno 42 |
| Cache | `.eval_cache_production/` (odcisk SHA-256 korpusu, kodu i wersji formatu) |

Wyniki trzech przebiegów powstały przy **różnych wartościach `EMBED_MODEL`/formatu chunków** (zmienianych między przebiegami w kodzie), więc ich nagłówki różnią się opisem architektury.

## 2. „Asymetria” — co jest w kodzie, a co w produkcji

* **W produkcji** (`build_database.py`, `FORMAT_VERSION = "asym-v2"`): embedding liczony jest z `text_emb` (breadcrumb `Tytuł › Sekcja` + treść, bez URL-i i bez etykiety źródła), a do Elasticsearch i do LLM trafia `text` z nagłówkiem `[Źródło] Tytuł › Sekcja` i linkami. Wersje się więc różnią. Szczegóły: [`build_database.md`](build_database.md), §7.
* **W skrypcie ewaluacyjnym** funkcja `chunk_file_asymmetric` buduje wersję wektorową jako `Tytuł › Sekcja` + treść, ale wersję dla ES też bierze z `text_emb`; w efekcie **oba indeksy mają praktycznie ten sam tekst**, bez etykiety `[Źródło]` i bez linków. Docstring funkcji twierdzi, że ES dostaje „pełny nagłówek”, co sugeruje, że skrypt pisano dla starszej wersji `build_database`, w której `text_emb` go zawierał (to wniosek z lektury kodu). Ewaluacja nie odwzorowuje więc w pełni leksykalnej strony obecnej produkcji. Dodatkowo `chunk_file_asymmetric` przekazuje konfigurację w starym formacie kluczy (`sciezka`, `zrodlo`…) — patrz [`eva_chunking.md`](eva_chunking.md), §5.
* Prompt rozszerzania w skrypcie (z przykładami „stypa → stypendium…”, „wuef → wychowanie fizyczne…”) różni się od promptu w `rag_service.py` (słownik skrótów AGH, obsługa historii). Przykłady slangu pochodzą z tej samej puli słów, której użyto do generowania pytań `colloquial` — zysk z rozszerzania w tej ewaluacji może być zatem lekko optymistyczny.
* Skrypt nie testuje historii rozmowy, progu podobieństwa ani cache'u.

## 3. Wyniki trzech przebiegów (90 zapytań)

| Przebieg | Konfiguracja | Hit@1 | Hit@5 | MRR@10 [95% CI] | Mediana wyszukiwania | LLM (śr.) |
|---|---|---|---|---|---|---|
| `run_…204829` (**odpowiada obecnemu kodowi**) | `3-small` + kontekst asymetryczny | 0.667 | 0.856 | 0.752 [0.67–0.83] | 278,7 ms (p95 390,7) | 1209 ms |
| `run_…212623` | `3-large` + kontekst asymetryczny | 0.711 | 0.922 | 0.801 [0.72–0.87] | 258,4 ms (p95 348,4) | 976 ms |
| `run_…141303` | `3-large` + „Enriched Context” | 0.678 | 0.944 | 0.795 [0.72–0.86] | 268,9 ms (p95 330,6) | 996 ms |

Hit@1 / MRR@10 według typu zapytania:

| Przebieg | colloquial | lexical | semantic |
|---|---|---|---|
| `3-small`, asymetryczny | 0.67 / 0.727 | 0.77 / 0.873 | 0.57 / 0.655 |
| `3-large`, asymetryczny | 0.70 / 0.800 | 0.83 / 0.908 | 0.60 / 0.696 |
| `3-large`, enriched | 0.70 / 0.796 | 0.77 / 0.878 | 0.57 / 0.713 |

## 4. Interpretacja

* **Jakość:** konfiguracja zgodna z obecnym kodem (`3-small`) trafia właściwy plik na pierwsze miejsce w 2/3 przypadków (Hit@1 0.667) i w top-5 w 85,6% (Hit@5 0.856), czyli w ok. **14% zapytań właściwego dokumentu nie ma w pierwszej piątce**. Nie przekłada się to na „pełne nasycenie” kontekstu poprawnymi źródłami.
* **Porównanie z pojedynczymi komponentami:** MRR 0.752 wobec 0.705 dla samej hybrydy bez rozszerzania i 0.779 dla `Hybrid + Expansion` z [`eva_search.md`](eva_search.md) — wyniki w granicach błędu, mimo różnic w chunkowaniu i prompcie.
* **Embeddingi:** `3-large` daje w pełnym stosie MRR ok. 0.80 wobec 0.75 (`3-small`) przy praktycznie takim samym opóźnieniu wyszukiwania (opóźnienie zapytania zależy od API OpenAI). Różnica ≈ 0.05 mieści się w nakładających się CI. Wybór `3-small` w produkcji jest więc decyzją kosztową; jakościowo `3-large` pozostaje nieco lepszy (patrz [`eva_embeddings.md`](eva_embeddings.md), §5).
* **Opóźnienia:** wyszukiwanie w bazach + RRF to ułamek sekundy (mediana ≈ 0,26–0,28 s). Rozszerzenie zapytania przez LLM dodaje średnio ok. 1,0–1,2 s, więc łączny narzut przed rozpoczęciem generowania wynosi **ok. 1,3–1,5 s w średniej** (mediana wyszukiwania + średnia LLM); dla p95 wyszukiwania i wolniejszych wywołań LLM będzie wyższy. W produkcji powtarzające się zapytania omijają LLM dzięki cache'owi (`backend.md`, §5.2).
* **Typy zapytań:** najlepiej `lexical`, najsłabiej `semantic` — zgodnie z projektem zbioru (n = 30 na typ).

## 5. Analiza błędów

Raport wypisuje zapytania, dla których oczekiwany plik nie znalazł się w top-5. Powtarzają się w trzech przebiegach m.in. pytanie o roczną pomoc finansową (oczekiwane: `kredyty-studenckie`; zwracane są regulaminy świadczeń i Erasmus) oraz pytanie o nagrodę dla studenta pierwszego roku. W przebiegu `3-small` pojawia się też zapytanie potoczne o „kolos z krychy” (oczekiwana `Krystalizacja.md`), na które system zwrócił nieistotne sylabusy i uchwałę rekrutacyjną. To dobre kandydaty do: (a) sprawdzenia, czy etykieta pytania jest jednoznaczna (kilka regulaminów dotyczy podobnych świadczeń), (b) rozbudowy słownika skrótów w prompcie rozszerzania, (c) ewentualnego strojenia embeddingów. Twierdzenie, że takie błędy „udowadniają granice możliwości systemu wektorowego”, byłoby za mocne — przy 90 pytaniach to pojedyncze przypadki.