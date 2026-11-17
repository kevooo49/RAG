# Zbiory ewaluacyjne

Testy jakości wyszukiwania (`evaluate_search.py`, `evaluate_embeddings.py`, `evaluate_chunking.py`, `evaluate_metadata.py`) korzystają z tych samych dwóch artefaktów: **próbki dokumentów** (`build_eval_sample.py`) i **zbioru pytań** (`generate_eval_questions.py`). Ta strona opisuje, jak powstają, co zawierają i czego nie mierzą. Wspólny kod ewaluacji (chunkowanie, metryki, statystyka, wykresy) opisuje [`eva_framework.md`](eva_framework.md).

## 1. Założenia

| Założenie | Dlaczego |
|---|---|
| ok. 300 dokumentów docelowych i ok. 900 pytań | przy kilkudziesięciu dokumentach przedziały ufności są tak szerokie, że większość różnic między wariantami jest nierozstrzygalna; przy 299 dokumentach CI MRR@10 ma szerokość ok. ±0,025 |
| pytania z fragmentów losowanych z całego dokumentu | odpowiedź często leży w środku lub na końcu długiego regulaminu; pytania tylko z początku dokumentów zawyżałyby wyniki |
| wiele poprawnych dokumentów na pytanie (`relevant_ids`) | ta sama treść występuje w kilku miejscach (sylabus w kolejnych latach, regulamin na kilku stronach); trafienie w kopię jest poprawną odpowiedzią |
| tło proporcjonalne do składu bazy + trudne dystraktory | dokument docelowy musi wygrać z dokumentami, które naprawdę z nim konkurują |
| próbka odwzorowuje strukturę folderów bazy, identyfikator = pole `plik` z indeksu | brak kolizji nazw; ten sam zbiór pytań działa na próbce i na pełnym indeksie |
| stałe ziarno losowania | wyniki są powtarzalne, a przebiegi porównywalne |

## 2. Próbka dokumentów — `build_eval_sample.py`

```bash
python build_eval_sample.py --dry-run                                  # plan i statystyki, nic nie kopiuje
python build_eval_sample.py --size 2000 --targets 300 --seed 42 --out ../../data/Ewaluacja_Probka
```

1. **Populacja = to, co indeksuje produkcja.** Kolekcje z `sites.yaml` (`collections()`), pliki `*.md`, treść ≥ 80 znaków (`MIN_DOC_CHARS` z `build_database.py`). Foldery surowe, tymczasowe i `_meta` nie są brane pod uwagę.
2. **Grupy duplikatów (cała baza).** Duplikaty dokładne (SHA-1 znormalizowanej treści) i bliskie: MinHash (128 permutacji) na 5-gramach słów, kandydaci z LSH (16 pasm × 8 wierszy), scalenie przy szacowanym podobieństwie Jaccarda ≥ 0,85 (`--dup-threshold`). Wynik zapisywany jest do `dup_groups_full.json`.
3. **Dokumenty docelowe (targets)** — po jednym na grupę duplikatów, losowane warstwowo po (źródło, typ) z alokacją proporcjonalną do √(rozmiar warstwy), min. 1 na warstwę, maks. 12 na warstwę (sylabusy są z limitu zwolnione). Odrzucane są:
   * strony szablonowe (`_stopka`, `mapa-serwisu`, `mapa-strony`, `deklaracja-dostepnosci`, `polityka-prywatnosci`, `cookies`, `index`…) — `boilerplate`,
   * strony, w których ponad połowa niepustych linii to same linki (menu, mapy serwisu) — `link_list`,
   * dokumenty z mniej niż 800 znakami „prawdziwego” tekstu po usunięciu linków i znaczników — `too_little_text`.

   Odrzucone pliki mogą nadal trafić do tła.
4. **Kopie** dokumentów docelowych (z ich grupy duplikatów, maks. 10) — `duplicate`.
5. **Trudne dystraktory** — do 2 dokumentów o tym samym znormalizowanym tytule, ale innej treści (ten sam przedmiot w innym roku lub na innym wydziale, regulamin o tej samej nazwie w innej jednostce) — `neighbor`.
6. **Tło** — reszta próbki losowana proporcjonalnie do rzeczywistego składu bazy (źródło × typ) — `background`.

Próbka zachowuje układ folderów `data_root`, a każdy plik ma identyfikator równy polu `plik` z indeksu produkcyjnego (np. `../../data/Baza_Wiedzy_AGH/CSS_Strony/…md`). Dzięki temu ten sam zbiór pytań działa na próbce i na pełnym indeksie. Skan bazy jest buforowany w `.eval_cache/inventory.*` (unieważniany przy zmianie rozmiaru lub daty pliku).

**Wyjście:** `<eval_dir>/…/*.md` (kopie), `sample_manifest.json` (parametry, statystyki, rekord na plik: `id`, `role`, `source`, `type`, `dup_group`, `target`…), `dup_groups_full.json`.

### Próbka użyta w ewaluacji (seed 42)

| | |
|---|---|
| Populacja | 107 245 dokumentów: 95 210 sylabusów, 10 146 stron, 1 889 dokumentów |
| Grupy duplikatów (> 1 członek) | 18 631 grup obejmujących **88 413 dokumentów** (prawie wyłącznie sylabusy — ten sam przedmiot w kolejnych latach i na różnych wydziałach) |
| Próbka | 2000 dokumentów: 300 docelowych, 185 kopii, 241 dystraktorów, 1274 tła; 1574 sylabusy, 300 stron, 126 dokumentów |
| Odrzuceni kandydaci na dokumenty docelowe | 574 `too_little_text`, 411 `link_list`, 20 `boilerplate` |
| Po pocięciu chunkerem produkcyjnym | 27 674 chunki (23 768 unikalnych tekstów) |

Spostrzeżenie uboczne: strony Wydziału Zarządzania (0 dokumentów docelowych z 312) i WILGZ (1 ze 162) prawie w całości nie przeszły filtra jakości, bo zostały zescrapowane razem z menu nawigacyjnym (`content_selectors: []` w `sites.yaml`). To problem scrapera, który dotyczy także produkcji.

## 3. Zbiór pytań — `generate_eval_questions.py` → `eval_questions.json`

```bash
python generate_eval_questions.py --dry-run          # fragmenty i szacunek kosztu, bez API
python generate_eval_questions.py --review 100       # generowanie + arkusz do ręcznego przeglądu
```

1. Każdy dokument docelowy jest dzielony na **fragmenty** wyrównane do akapitów (600–1800 znaków; długie akapity, typowe dla PDF, są dzielone na granicach linii/zdań; fragmenty złożone głównie z linków są pomijane).
2. Dla każdego dokumentu losowane są **3 rozłączne fragmenty**, w miarę możliwości po jednym z każdej tercji dokumentu, i każdy dostaje jeden styl pytania: `lexical`, `semantic`, `colloquial`.
3. **Jedno wywołanie `gpt-4.1`** (temperatura 0,7, structured output) na dokument zwraca dla każdego fragmentu: `usable`, `question`, `answer`, `evidence` — dosłowny cytat (40–300 znaków) zawierający odpowiedź. Model generujący pytania jest inny niż modele systemu (`gpt-4o-mini` / `gpt-4o`).
4. **Weryfikacja automatyczna** — pytanie jest odrzucane, gdy:
   * cytat nie występuje w fragmencie (po normalizacji: wielkość liter, białe znaki, znaczniki Markdown, cudzysłowy; dopuszczalne ≥ 80% zgodności najdłuższego wspólnego fragmentu) — `evidence_not_found`,
   * cytat jest krótszy niż 30 znaków lub zawiera znaczniki HTML — `html_evidence`,
   * pytanie kończy się odwołaniem wskazującym bez nazwy („…z tego przedmiotu?”, „…na tym kierunku?”) — `deictic_reference`,
   * model uznał fragment za bezużyteczny — `not_usable`.

   Odrzucone pytania są ponawiane na innych fragmentach (do 3 wywołań na dokument). Dokument, który nie dał żadnego pytania, jest zastępowany dokumentem tła z tej samej warstwy, który przechodzi ten sam filtr jakości.
5. **Wiele poprawnych dokumentów.** `relevant_ids` to wszystkie dokumenty próbki, których znormalizowana treść zawiera cytat. Pytania z cytatem obecnym w > 5 dokumentach dostają flagę `generic` (prawdopodobnie niejednoznaczne). Raporty pokazują wyniki z tymi pytaniami i bez nich.
6. Pole `answer` (odpowiedź z fragmentu) służy jako odpowiedź wzorcowa przy ocenie generowania.

Rekord: `qid, query, type, answer, evidence, expected_id, relevant_ids, n_relevant, flags, source, doc_type, title, passage, passage_start, passage_end, position, doc_chars, dup_group, replacement_for, model`. Postęp zapisywany jest w `eval_questions.json.progress.jsonl`. Przerwane uruchomienie kontynuuje pracę, a po zmianie reguł walidacji ponowne uruchomienie odrzuca niezgodne pytania i uzupełnia tylko brakujące style.

### Zbiór użyty w ewaluacji

| | |
|---|---|
| Pierwszy przebieg | 864 pytania z 299 dokumentów (271 kompletnych, 28 częściowych, 1 nieudany); typy 291 / 282 / 291; 511 wywołań, 742 tys. tokenów wejścia i 199 tys. wyjścia (ok. 3 USD) |
| Odrzucenia w pierwszym przebiegu | 315 `evidence_not_found`, 35 `not_usable` |
| Pozycja w dokumencie | początek 388, środek 266, koniec 210 |
| Liczba dokumentów zawierających cytat | 1: 590 pytań, 2–5: 178, > 5 (`generic`): 96 |
| Po zaostrzeniu reguł i uzupełnieniu (§3.1) — **zbiór końcowy** | **860 pytań o 299 dokumentów**: 290 `lexical`, 279 `semantic`, 291 `colloquial`; 209 z dokumentów, 386 ze stron, 265 z sylabusów; 95 `generic` |

Dokładne liczniki po uzupełnieniu zapisane są w `meta` pliku `eval_questions.json`.

### 3.1 Ręczny przegląd jakości

Losowa próba 100 pytań z pierwszego przebiegu została przejrzana ręcznie (`eval_questions_review.csv`): **87 poprawnych**, 13 wadliwych — 9 niejednoznacznych (pytanie wskazuje „ten przedmiot / ten kierunek / ten projekt” bez nazwy albo pyta o osobę bez nazwiska), 3 nienaturalne (np. pytanie o kolumny tabeli formularza), 1 z cytatem w postaci surowego HTML. Obawa, że brakuje pytań o treść tabel, się nie potwierdziła (ok. 12 pytań w próbie dotyczy tabel: cenniki, godziny, opłaty, ECTS).

Na tej podstawie dodano regułę `deictic_reference`, odrzucanie cytatów z HTML i doprecyzowano prompt (nazwy przedmiotów, kierunków i osób obowiązkowe także w stylu `semantic`; zakaz pytań o układ tabel, pola formularzy i cel pisma). Niejednoznacznych pytań o osoby, które nie kończą się odwołaniem wskazującym, reguła nie wykrywa — mogły pozostać w zbiorze.

## 4. Czego te zbiory nie mierzą

* **Pytań naturalnych** — pytania pisze LLM na podstawie fragmentu. Styl `colloquial` jest jego wyobrażeniem języka studentów. Pytania pisane ręcznie przez autora powstaną dopiero na etapie walidacji końcowej.
* Pytań wielokrokowych, z historią rozmowy i takich, na które baza nie zawiera odpowiedzi (zachowanie odmowy).
* **Równoważnych odpowiedzi w innym brzmieniu.** Dokument liczy się jako trafny tylko wtedy, gdy zawiera *dosłowny* cytat. Strona, która podaje ten sam fakt innymi słowami, jest liczona jako pudło, więc wyniki są **niedoszacowane** — w jednakowym stopniu dla wszystkich porównywanych wariantów.
* Pełnej bazy: próbka ma 2000 dokumentów, a produkcja 107 tys. Bezwzględne wartości metryk na pełnym indeksie będą niższe (więcej dystraktorów), natomiast porównania wariantów pozostają uczciwe.

## 5. Pliki

| Plik | Zawartość |
|---|---|
| `data/Ewaluacja_Probka/` (`paths.eval_dir`) | kopie dokumentów próbki, `sample_manifest.json`, `dup_groups_full.json` |
| `backend/scripts/eval_questions.json` | zbiór pytań z `meta` (parametry i liczniki generowania) |
| `backend/scripts/eval_questions_review.csv` | arkusz ręcznego przeglądu (§3.1) |