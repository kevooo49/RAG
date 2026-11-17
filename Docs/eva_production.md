# Ewaluacja konfiguracji produkcyjnej (`evaluate_production.py`)

Skrypt sprawdza **cały łańcuch RAG** — wyszukiwanie i odpowiedź — w dwóch konfiguracjach: dzisiejszej konfiguracji produkcji i konfiguracji zaproponowanej na podstawie testów składowych. Testy składowe zmieniały jeden element naraz; ten test sprawdza, czy zmiany działają **razem** i czy poprawa wyszukiwania przekłada się na lepsze odpowiedzi. Dane: [`eva_datasets.md`](eva_datasets.md); metryki i statystyka: [`eva_framework.md`](eva_framework.md); sędzia odpowiedzi: [`eva_models.md`](eva_models.md).

Wynik: `production_eva_stats/run_20261118_001827/`.

## 1. Jak to działa

### Konfiguracje

Testy składowe zaproponowały dwie zmiany:
* **wyłączenie rozszerzania zapytań** dla samodzielnych pytań ([`eva_search.md`](eva_search.md): rozszerzanie przez `gpt-4o-mini` obniżało MRR@10),
* **nagłówek `[Źródło] Tytuł` w tekście do embeddingu** zamiast `Tytuł › Sekcja › Podsekcja` ([`eva_metadata.md`](eva_metadata.md): +0.020 MRR@10).

Skrypt testuje obie zmiany razem i każdą osobno (układ 2 × 2):

| System | Nagłówek w tekście do embeddingu | Rozszerzanie zapytań | Rola |
|---|---|---|---|
| `current` | `Tytuł › Sekcja › Podsekcja` | tak | **produkcja dziś** (system bazowy) |
| `current_noexp` | `Tytuł › Sekcja › Podsekcja` | nie | efekt samego wyłączenia rozszerzania |
| `planned_exp` | `[Źródło] Tytuł` | tak | efekt samej zmiany nagłówka |
| `planned` | `[Źródło] Tytuł` | nie | **konfiguracja proponowana** |

Wszystko poza tymi dwoma elementami jest identyczne z produkcją: chunki z `build_database.chunk_file` (450 tokenów), tekst BM25 `[Źródło] Tytuł › Sekcja` w Elasticsearch z analizatorem `polish` (stempel), embeddingi `text-embedding-3-small`, RRF (k=60) z top 40 wektora i top 40 BM25, 5 chunków w kontekście. Przy rozszerzaniu rozszerzone zapytanie trafia do **obu** ścieżek wyszukiwania, jak w `rag_service.search_by_query`; prompt i post-processing rozszerzania są importowane z `rag_service`.

### Część 1 — wyszukiwanie

Wszystkie pytania z `eval_questions.json`, cztery systemy. Metryki jak w pozostałych testach wyszukiwania: Hit@k i MRR@10 na poziomie plików, CtxHit@5 (chunk z cytatem-dowodem wśród 5 chunków przekazanych do modelu). Kolumna „LLM ms” podaje średnie opóźnienie wywołania rozszerzającego (pierwsze, niezbuforowane wywołanie).

### Część 2 — odpowiedzi

Ten sam podzbiór 300 pytań co w [`eva_models.md`](eva_models.md) (po 100 z każdego typu, ziarno 42), systemy `current` i `planned`. Kontekst (5 chunków danego systemu) trafia do modelu `gpt-4o-mini` — domyślnego modelu produkcji — z promptem systemowym i blokami `<source>` z `llm_service.py`. Odpowiedzi ocenia ten sam sędzia (`gpt-4.1`) według tych samych kryteriów: correctness, faithfulness, citations, odmowy i halucynacje.

Raport podaje dla obu systemów średnie z przedziałami ufności, różnicę `planned − current` (parowany bootstrap klastrowy po dokumentach, korekta Holma po metrykach) oraz zestawienie pytanie po pytaniu: w ilu pytaniach odpowiedź `planned` była lepsza, taka sama i gorsza niż `current`.

### Koszt

Część 1 jest praktycznie darmowa: embeddingi obu nagłówków i rozszerzenia zapytań są już w pamięci podręcznej z wcześniejszych testów. W części 2 odpowiedzi `planned` modelu `gpt-4o-mini` są już zapisane z [`evaluate_models.py`](eva_models.md) (ta sama pamięć `model_eval.sqlite`), więc płatne jest tylko 300 odpowiedzi `current` i ich ocena: ok. 2 USD.

```bash
python evaluate_production.py --dry-run          # część 1 + szacunek kosztu części 2, bez wywołań LLM
python evaluate_production.py                    # całość
python evaluate_production.py --skip-answers     # tylko wyszukiwanie
```

Pliki wynikowe: `report.txt`, `results.json`, `per_query.csv` (wyszukiwanie), `per_answer.csv` (odpowiedzi obu systemów obok siebie, z ocenami i uzasadnieniem sędziego), standardowe wykresy wyszukiwania i `answers.png`.

## 2. Wyniki

### Część 1 — wyszukiwanie (860 pytań, 299 dokumentów)

| System | Hit@1 | Hit@5 | MRR@10 [95% CI] | ΔMRR vs current | CtxHit@5 [95% CI] | ΔCtxHit@5 | Rozszerzanie |
|---|---|---|---|---|---|---|---|
| **`planned`** | **0.742** | **0.914** | **0.815** [0.79–0.84] | **+0.034*** [+0.02, +0.05] | **0.766** [0.73–0.80] | +0.009 | — |
| `current_noexp` | 0.717 | 0.900 | 0.797 [0.77–0.82] | **+0.016*** [+0.00, +0.03] | 0.748 [0.72–0.78] | −0.009 | — |
| `planned_exp` | 0.717 | 0.905 | 0.796 [0.77–0.82] | **+0.015*** [+0.00, +0.03] | 0.749 [0.72–0.78] | −0.008 | 1142 ms |
| `current` | 0.699 | 0.886 | 0.781 [0.75–0.81] | (bazowy) | 0.757 [0.73–0.79] | (bazowy) | 1142 ms |

\* p_holm < 0,05 (korekta po trzech porównaniach). Różnice CtxHit@5 są nieistotne (p=0.99). „Rozszerzanie” = średni czas wywołania rozszerzającego zapytanie, doliczany przed każdym wyszukiwaniem.

| System | potoczne (291) | dosłowne (290) | semantyczne (279) | dokument (209) | strona (386) | sylabus (265) |
|---|---|---|---|---|---|---|
| `planned` | **0.793** | **0.873** | **0.778** | 0.826 | **0.870** | **0.726** |
| `current_noexp` | 0.777 | 0.859 | 0.753 | **0.837** | 0.849 | 0.690 |
| `planned_exp` | 0.762 | 0.858 | 0.767 | 0.813 | 0.856 | 0.696 |
| `current` | 0.750 | 0.839 | 0.752 | 0.814 | 0.849 | 0.656 |

(MRR@10 według typu pytania i typu dokumentu.)

### Część 2 — odpowiedzi (300 pytań, `gpt-4o-mini`, sędzia `gpt-4.1`)

| System | CtxHit@5 | Correctness [95% CI] | Fully correct | Faithfulness [95% CI] | Citations | Hallucination |
|---|---|---|---|---|---|---|
| `current` | 0.770 | 0.817 [0.78–0.85] | 0.733 | 0.948 [0.93–0.97] | 0.950 | 0.017 |
| **`planned`** | **0.797** | **0.850** [0.81–0.88] | **0.787** | **0.958** [0.94–0.98] | **0.966** | 0.017 |

Różnica `planned − current` (parowany bootstrap klastrowy, korekta Holma po sześciu metrykach):

| Metryka | Różnica [95% CI] | p (Holm) |
|---|---|---|
| CtxHit@5 | +0.027 [−0.013, +0.071] | 0.75 |
| Correctness | +0.033 [−0.002, +0.068] | 0.26 |
| Fully correct | +0.053 [+0.010, +0.097] | 0.07 |
| Faithfulness | +0.010 [−0.010, +0.030] | 0.75 |
| Citations | +0.019 [+0.002, +0.036] | 0.13 |
| Hallucination | 0.000 [−0.016, +0.016] | 1.00 |

Pytanie po pytaniu (correctness 0–2): `planned` lepszy w 30 pytaniach, taki sam w 252, gorszy w 18.

| System | Odmowy | Right refusal | False refusal | potoczne | dosłowne | semantyczne | dokument | strona | sylabus |
|---|---|---|---|---|---|---|---|---|---|
| `current` | 10% | 0.95 (n=19) | 0.01 (n=245) | 0.765 | 0.860 | 0.825 | 0.775 | 0.855 | 0.790 |
| `planned` | 8% | 0.88 (n=16) | 0.01 (n=255) | **0.795** | **0.915** | **0.840** | **0.790** | **0.888** | **0.839** |

(Ostatnie sześć kolumn: correctness według typu pytania i typu dokumentu.)

Czasy generowania (mediana 1.5 s dla `current`, 2.4 s dla `planned`) nie są porównywalne: odpowiedzi `planned` powstały wcześniej, w przebiegu `evaluate_models.py`, przy innym obciążeniu API. Porównywalna jest różnica, którą wprowadza sama konfiguracja: `current` wykonuje przed wyszukiwaniem dodatkowe wywołanie `gpt-4o-mini` (średnio 1.1 s), `planned` nie.

## 3. Wnioski — co wynika z danych

1. **Obie zmiany działają i ich efekty się sumują.** Każda osobno daje ok. +0.015 MRR@10 (zmiana nagłówka: `planned_exp` +0.015; wyłączenie rozszerzania: `current_noexp` +0.016), razem +0.034 — tyle, ile suma obu. Wszystkie trzy różnice są istotne po korekcie Holma.
2. **Zysk jest widoczny w każdej podgrupie.** `planned` jest najlepszy dla wszystkich typów pytań, dla stron i sylabusów (sylabusy: +0.070 MRR względem `current`), dla pytań z początku i końca dokumentu oraz dla pytań ogólnych i szczegółowych. Dwa wyjątki są minimalne: w dokumentach (PDF) `current_noexp` ma 0.837 wobec 0.826 dla `planned` — ten sam wzorzec co w [`eva_metadata.md`](eva_metadata.md) — a dla pytań ze środka dokumentu `planned_exp` ma 0.776 wobec 0.774.
3. **Wyszukiwanie poprawia ranking plików bardziej niż zawartość kontekstu.** CtxHit@5 rośnie tylko o 0.009 (nieistotne): właściwy dokument częściej jest na pierwszym miejscu, ale to, czy chunk z dowodem trafi do 5 chunków kontekstu, zmienia się niewiele.
4. **Odpowiedzi są lepsze, ale na 300 pytaniach różnica nie przechodzi korekty.** Correctness +0.033, odpowiedzi w pełni poprawne +0.053 (p=0.07), cytowania +0.019; poprawa występuje we wszystkich typach pytań i dokumentów, a w zestawieniu pytanie po pytaniu `planned` wygrywa 30:18. Żadna metryka nie spada, odsetek halucynacji jest taki sam.
5. **Konfiguracja proponowana jest też szybsza i tańsza.** Bez rozszerzania odpada jedno wywołanie LLM przed każdym pierwszym pytaniem: ok. 1.1 s mniej czasu oczekiwania i brak kosztu tego wywołania.

## 4. Decyzja

**Produkcja przechodzi na konfigurację `planned`:**
* tekst do embeddingu chunków z nagłówkiem `[Źródło] Tytuł` (zmiana `text_emb` i `FORMAT_VERSION` w `build_database.py`, ponowna wektoryzacja całej bazy; indeks Elasticsearch bez zmian),
* bez rozszerzania zapytań dla pierwszego pytania w rozmowie (bez historii). Przeformułowanie pytań z historią rozmowy zostaje.

Uzasadnienie: istotna poprawa wyszukiwania (+0.034 MRR@10), poprawa odpowiedzi we wszystkich podgrupach bez pogorszenia żadnej metryki, krótszy czas odpowiedzi i niższy koszt.

## 5. Granice

* Próbka 2000 dokumentów, pytania generowane przez LLM, jednoturowe. Przeformułowanie pytania w rozmowie (z historią) działa w produkcji dalej i nie jest tu testowane — wyłączenie rozszerzania dotyczy tylko pierwszego pytania bez historii.
* Sędzia to model językowy; jego zgodność z oceną człowieka opisuje [`eva_models.md`](eva_models.md).
* Ostateczne sprawdzenie na pełnej bazie (po ponownym zbudowaniu indeksów) z pytaniami pisanymi ręcznie: walidacja końcowa.