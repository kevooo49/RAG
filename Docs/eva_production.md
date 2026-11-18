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

Piąty system, `planned_dedup`, to `planned` z **filtrem bliskich duplikatów w kontekście**. Baza zawiera wiele niemal identycznych tekstów w osobnych plikach — ten sam sylabus w kolejnych latach akademickich, ten sam regulamin na stronach kilku wydziałów — i kilka takich kopii potrafi zająć miejsca w top 5, wypychając inne informacje. Filtr przechodzi przez pierwsze 25 kandydatów w kolejności RRF (tyle, ile produkcja pobiera: `top_k × 5`) i pomija chunk, którego treść (bez nagłówka) dzieli co najmniej 85% słów (miara Jaccarda na zbiorach słów) z chunkiem już wybranym. Z dwóch bliskich duplikatów zostaje ten z nowszego roku akademickiego (rok odczytany ze ścieżki pliku, np. `2025_2026`), na miejscu lepiej ocenionego. Pominięte chunki trafiają na koniec rankingu: nie wchodzą do kontekstu, ale metryki na poziomie plików nadal je widzą. Próg można zmienić opcją `--dedup-threshold`. Filtr jest na razie tylko w ewaluacji; do produkcji (`rag_service.search_by_query`) trafi, jeśli wyniki to uzasadnią.

Wszystko poza tymi elementami jest identyczne z produkcją: chunki z `build_database.chunk_file` (450 tokenów), tekst BM25 `[Źródło] Tytuł › Sekcja` w Elasticsearch z analizatorem `polish` (stempel), embeddingi `text-embedding-3-small`, RRF (k=60) z top 40 wektora i top 40 BM25, 5 chunków w kontekście. Przy rozszerzaniu rozszerzone zapytanie trafia do **obu** ścieżek wyszukiwania, jak w `rag_service.search_by_query`; prompt i post-processing rozszerzania są importowane z `rag_service`.

### Część 1 — wyszukiwanie

Wszystkie pytania z `eval_questions.json`, pięć systemów. Dla filtra duplikatów raport podaje osobno: odsetek pytań, w których top 5 systemu `planned` zawierało bliskie duplikaty, oraz porównanie `planned_dedup − planned` (MRR@10, CtxHit@5, CtxPrec@5; parowany bootstrap, korekta Holma), także tylko na pytaniach z duplikatami. Metryki jak w pozostałych testach wyszukiwania: Hit@k i MRR@10 na poziomie plików, CtxHit@5 (chunk z cytatem-dowodem wśród 5 chunków przekazanych do modelu). Kolumna „LLM ms” podaje średnie opóźnienie wywołania rozszerzającego (pierwsze, niezbuforowane wywołanie).

### Część 2 — odpowiedzi

Ten sam podzbiór 300 pytań co w [`eva_models.md`](eva_models.md) (po 100 z każdego typu, ziarno 42), systemy `current`, `planned` i `planned_dedup`. Kontekst (5 chunków danego systemu) trafia do modelu `gpt-4o-mini` — domyślnego modelu produkcji — z promptem systemowym i blokami `<source>` z `llm_service.py`. Odpowiedzi ocenia ten sam sędzia (`gpt-4.1`) według tych samych kryteriów: correctness, faithfulness, citations, odmowy i halucynacje.

Raport podaje dla każdego systemu średnie z przedziałami ufności, różnicę `planned − current` (parowany bootstrap klastrowy po dokumentach, korekta Holma po metrykach) oraz zestawienie pytanie po pytaniu (lepsza / taka sama / gorsza odpowiedź) dla dwóch par: `planned − current` i `planned_dedup − planned`.

### Koszt

Część 1 jest praktycznie darmowa: embeddingi obu nagłówków i rozszerzenia zapytań są już w pamięci podręcznej z wcześniejszych testów. W części 2 odpowiedzi `planned` modelu `gpt-4o-mini` są już zapisane z [`evaluate_models.py`](eva_models.md) (ta sama pamięć `model_eval.sqlite`), więc płatne jest tylko 300 odpowiedzi `current` i ich ocena: ok. 2 USD.

```bash
python evaluate_production.py --dry-run          # część 1 + szacunek kosztu części 2, bez wywołań LLM
python evaluate_production.py                    # całość
python evaluate_production.py --skip-answers     # tylko wyszukiwanie
```

Pliki wynikowe: `report.txt`, `results.json`, `per_query.csv` (wyszukiwanie), `per_answer.csv` (odpowiedzi wszystkich systemów obok siebie, z ocenami i uzasadnieniem sędziego), standardowe wykresy wyszukiwania i `answers.png`.

## 2. Wyniki

Wynik opisany poniżej: `production_eva_stats/run_20261118_102944/`. Poprzedni przebieg (`run_20261118_001827`, bez `planned_dedup`) dał te same wnioski; liczby różnią się o 0.001–0.005, bo między przebiegami indeks BM25 próbki został zbudowany od nowa (inna kolejność wyników o równych ocenach), a część odpowiedzi wygenerowano ponownie.

### Część 1 — wyszukiwanie (860 pytań, 299 dokumentów)

| System | Hit@1 | Hit@5 | MRR@10 [95% CI] | ΔMRR vs current | CtxHit@5 [95% CI] | ΔCtxHit@5 | Rozszerzanie |
|---|---|---|---|---|---|---|---|
| **`planned`** | **0.740** | **0.915** | **0.814** [0.79–0.84] | **+0.032*** [+0.01, +0.05] | 0.766 [0.73–0.80] | +0.010 | — |
| `planned_dedup` | 0.730 | 0.903 | 0.805 [0.78–0.83] | +0.024 (p=0.06) | **0.770** [0.74–0.80] | +0.014 | — |
| `planned_exp` | 0.720 | 0.905 | 0.797 [0.77–0.82] | +0.015 (p=0.06) | 0.750 [0.72–0.78] | −0.006 | 1142 ms |
| `current_noexp` | 0.709 | 0.900 | 0.793 [0.77–0.82] | +0.011 (p=0.16) | 0.749 [0.72–0.78] | −0.007 | — |
| `current` | 0.702 | 0.884 | 0.782 [0.75–0.81] | (bazowy) | 0.756 [0.72–0.79] | (bazowy) | 1142 ms |

\* p_holm < 0,05 (korekta po czterech porównaniach). Różnice CtxHit@5 są nieistotne (p=0.96). „Rozszerzanie” = średni czas wywołania rozszerzającego zapytanie, doliczany przed każdym wyszukiwaniem.

| System | potoczne (291) | dosłowne (290) | semantyczne (279) | dokument (209) | strona (386) | sylabus (265) |
|---|---|---|---|---|---|---|
| `planned` | **0.793** | **0.868** | **0.778** | 0.823 | 0.868 | **0.726** |
| `planned_dedup` | 0.790 | 0.853 | 0.772 | 0.826 | **0.871** | 0.694 |
| `planned_exp` | 0.765 | 0.860 | 0.765 | 0.814 | 0.857 | 0.697 |
| `current_noexp` | 0.776 | 0.855 | 0.747 | **0.833** | 0.848 | 0.680 |
| `current` | 0.749 | 0.843 | 0.752 | 0.814 | 0.848 | 0.659 |

(MRR@10 według typu pytania i typu dokumentu.)

#### Filtr bliskich duplikatów (`planned_dedup` − `planned`)

Bliskie duplikaty (≥ 85% wspólnych słów) były w top 5 systemu `planned` dla **34,9% pytań** (300 z 860); w takich pytaniach filtr pomijał średnio 4,7 chunka, zanim kontekst się zapełnił.

| Metryka | `planned` | `planned_dedup` | Różnica [95% CI] | p (Holm) |
|---|---|---|---|---|
| MRR@10 | 0.814 | 0.805 | −0.008 [−0.018, +0.000] | 0.12 |
| CtxHit@5 | 0.766 | 0.770 | +0.003 [−0.013, +0.020] | 0.70 |
| CtxPrec@5 | 0.514 | 0.474 | **−0.040*** [−0.052, −0.030] | 0.002 |
| CtxHit@5, tylko 300 pytań z duplikatami | 0.700 | 0.713 | +0.013 | — |

### Część 2 — odpowiedzi (300 pytań, `gpt-4o-mini`, sędzia `gpt-4.1`)

| System | CtxHit@5 | Correctness [95% CI] | Fully correct | Faithfulness [95% CI] | Citations | Hallucination |
|---|---|---|---|---|---|---|
| `current` | 0.770 | 0.817 [0.78–0.85] | 0.737 | 0.948 [0.93–0.97] | 0.948 | 0.013 |
| **`planned`** | 0.797 | **0.850** [0.81–0.88] | **0.787** | **0.963** [0.94–0.98] | **0.968** | 0.013 |
| `planned_dedup` | **0.800** | 0.843 [0.81–0.88] | 0.773 | 0.950 [0.93–0.97] | 0.955 | 0.017 |

Różnice (parowany bootstrap klastrowy, korekta Holma po sześciu metrykach w każdej parze):

| Metryka | `planned − current` | p (Holm) | `planned_dedup − planned` | p (Holm) |
|---|---|---|---|---|
| CtxHit@5 | +0.027 [−0.013, +0.068] | 0.43 | +0.003 [−0.022, +0.028] | 1.00 |
| Correctness | +0.033 [+0.002, +0.066] | 0.15 | −0.007 [−0.026, +0.012] | 1.00 |
| Fully correct | **+0.050** [+0.014, +0.089] | **0.012*** | −0.013 [−0.037, +0.010] | 1.00 |
| Faithfulness | +0.015 [−0.003, +0.034] | 0.41 | **−0.013** [−0.025, −0.005] | **0.018*** |
| Citations | **+0.023** [+0.007, +0.039] | **0.020*** | **−0.011** [−0.020, −0.004] | **0.030*** |
| Hallucination | 0.000 [−0.014, +0.013] | 1.00 | +0.003 [+0.000, +0.010] | 1.00 |
| Pytanie po pytaniu (lepsza / taka sama / gorsza) | 29 / 256 / 15 | | 9 / 279 / 12 | |

| System | Odmowy | Right refusal | False refusal | potoczne | dosłowne | semantyczne | dokument | strona | sylabus |
|---|---|---|---|---|---|---|---|---|---|
| `current` | 10% | 0.94 (n=18) | 0.02 (n=248) | 0.755 | 0.875 | 0.820 | 0.768 | 0.866 | 0.780 |
| `planned` | 8% | 0.92 (n=13) | 0.02 (n=258) | 0.805 | **0.905** | 0.840 | 0.804 | 0.884 | **0.833** |
| `planned_dedup` | 8% | 0.85 (n=13) | 0.02 (n=259) | **0.815** | 0.870 | **0.845** | **0.812** | **0.888** | 0.801 |

(Ostatnie sześć kolumn: correctness według typu pytania i typu dokumentu.)

Czasy generowania nie są porównywalne między systemami: część odpowiedzi powstała we wcześniejszych przebiegach (pamięć podręczna), przy innym obciążeniu API. Porównywalna jest różnica, którą wprowadza sama konfiguracja: `current` wykonuje przed wyszukiwaniem dodatkowe wywołanie `gpt-4o-mini` (średnio 1.1 s), `planned` nie.

## 3. Wnioski — co wynika z danych

1. **Obie zmiany pomagają, a razem dają istotną poprawę.** Zmiana nagłówka (`planned_exp`) daje +0.015 MRR@10, wyłączenie rozszerzania (`current_noexp`) +0.011; osobno żadna nie przechodzi korekty po czterech porównaniach (w poprzednim przebiegu, po trzech, obie przechodziły), razem +0.032 (p=0.008) — mniej więcej suma obu efektów.
2. **Zysk jest widoczny niemal w każdej podgrupie.** `planned` jest najlepszy dla wszystkich typów pytań i dla sylabusów (+0.067 MRR względem `current`). W dokumentach (PDF) najlepszy jest `current_noexp` (0.833 wobec 0.823) — ten sam wzorzec co w [`eva_metadata.md`](eva_metadata.md).
3. **Wyszukiwanie poprawia ranking plików bardziej niż zawartość kontekstu.** CtxHit@5 rośnie tylko o 0.010 (nieistotne): właściwy dokument częściej jest na pierwszym miejscu, ale to, czy chunk z dowodem trafi do 5 chunków kontekstu, zmienia się niewiele.
4. **Odpowiedzi są lepsze.** Odsetek odpowiedzi w pełni poprawnych rośnie z 73,7% do 78,7% (istotne), cytowania poprawiają się istotnie (+0.023), correctness +0.033 (p=0.15 po korekcie). Poprawa występuje we wszystkich typach pytań i dokumentów, pytanie po pytaniu `planned` wygrywa 29:15. Żadna metryka nie spada, odsetek halucynacji jest taki sam.
5. **Konfiguracja proponowana jest też szybsza i tańsza.** Bez rozszerzania odpada jedno wywołanie LLM przed każdym pierwszym pytaniem: ok. 1.1 s mniej czasu oczekiwania i brak kosztu tego wywołania.
6. **Duplikaty w kontekście są częste, ale nie szkodzą — a ich usuwanie trochę szkodzi.**
   * Co trzecie pytanie ma w top 5 bliskie kopie, a mimo to filtr prawie nie zwiększa szansy, że chunk z dowodem trafi do kontekstu (+0.003 ogółem, +0.013 na pytaniach z duplikatami). Jeśli kopia zawiera dowód, to i bez filtra dowód jest w kontekście; filtr zastępuje ją innym, zwykle mniej trafnym chunkiem.
   * Filtr myli różne dokumenty o wspólnym szablonie z kopiami. Sylabusy różnych przedmiotów mają prawie identyczne fragmenty (tabela „Nakład pracy studenta”, opisy metod weryfikacji) — przy progu 85% filtr usuwa właściwy sylabus jako „kopię” innego. Stąd spadek MRR@10 dla sylabusów (0.726 → 0.694) i dla pytań z początku dokumentu.
   * Model odpowiada nieco gorzej: faithfulness −0.013 i cytowania −0.011 (oba istotne, choć małe). Bardziej zróżnicowany kontekst wnosi więcej materiału niezwiązanego z pytaniem, a powtórzony fragment nie przeszkadza modelowi — jest po prostu potwierdzeniem.
   * Spadek CtxPrec@5 (−0.040) jest mechaniczny: kopie zawierające dowód liczą się jako trafne chunki, więc ich usunięcie obniża precyzję i nie świadczy o gorszym kontekście.

## 4. Decyzja

**Produkcja przechodzi na konfigurację `planned`:**
* tekst do embeddingu chunków z nagłówkiem `[Źródło] Tytuł` (zmiana `text_emb` i `FORMAT_VERSION` w `build_database.py`, ponowna wektoryzacja całej bazy; indeks Elasticsearch bez zmian),
* bez rozszerzania zapytań dla pierwszego pytania w rozmowie (bez historii; `QUERY_EXPANSION = False`). Przeformułowanie pytań z historią rozmowy zostaje.

Uzasadnienie: istotna poprawa wyszukiwania (+0.032 MRR@10) i odpowiedzi (odpowiedzi w pełni poprawne +5 p.p., cytowania), bez pogorszenia żadnej metryki, krótszy czas odpowiedzi i niższy koszt.

**Filtr bliskich duplikatów nie trafia do produkcji.** Nie zwiększa obecności dowodu w kontekście, obniża MRR@10 dla sylabusów i nieznacznie pogarsza wierność i cytowania odpowiedzi. Bardziej konserwatywny wariant (np. tylko kopie o tym samym tytule) mógłby uniknąć pomyłek na szablonach sylabusów, ale górna granica zysku jest mała: nawet idealny filtr poprawiłby CtxHit@5 najwyżej w części z 30% pytań, w których dowodu w kontekście brakuje.

## 5. Granice

* Próbka 2000 dokumentów, pytania generowane przez LLM, jednoturowe. Przeformułowanie pytania w rozmowie (z historią) działa w produkcji dalej i nie jest tu testowane — wyłączenie rozszerzania dotyczy tylko pierwszego pytania bez historii.
* Sędzia to model językowy; jego zgodność z oceną człowieka opisuje [`eva_models.md`](eva_models.md).
* Ostateczne sprawdzenie na pełnej bazie (po ponownym zbudowaniu indeksów) z pytaniami pisanymi ręcznie: walidacja końcowa.