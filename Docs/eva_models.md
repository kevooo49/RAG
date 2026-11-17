# Ewaluacja modeli generujących (`evaluate_models.py`)

Skrypt porównuje modele odpowiadające na pytania w warunkach, w jakich spotyka je użytkownik: z **promptem systemowym produkcji**, **kontekstem z wyszukiwarki** (5 chunków, które system faktycznie znajduje, a nie idealny dokument) i klientami modeli z `llm_service.py`. Każdy model odpowiada na **te same 300 pytań** z tym samym kontekstem, więc różnice wynikają wyłącznie z modelu. Dane: [`eva_datasets.md`](eva_datasets.md); statystyka: [`eva_framework.md`](eva_framework.md).

Wynik: `models_eva_stats/run_20261117_234121/`.

## 1. Jak to działa

### Pytania

Podzbiór 300 pytań z `eval_questions.json`, **po 100 z każdego typu** (dosłowne, semantyczne, potoczne), losowany ze stałym ziarnem 42 (`pick_subset`). Ten sam podzbiór dostaje każdy model i każde uruchomienie (także [`evaluate_production.py`](eva_production.md)). Pytania dotyczą 205 dokumentów docelowych.

### Kontekst

Dla każdego pytania wyszukiwanie hybrydowe na próbce (RRF z top 40 wektora i top 40 BM25) zwraca 5 chunków, które trafiają do modelu w takiej postaci jak w produkcji: bloki `<source>` z tytułem, źródłem, sekcją i adresem (`llm_service._build_context`), przycięte do limitu znaków (`_trim_context`, `RAG_MAX_CONTEXT_CHARS`).

| Opcja `--retrieval` | Nagłówek w tekście do embeddingu | Rozszerzanie zapytań |
|---|---|---|
| `planned` (domyślna, użyta w wynikach) | `[Źródło] Tytuł` | nie |
| `current` | `Tytuł › Sekcja › Podsekcja` (produkcja dziś) | tak |

`planned` to konfiguracja wybrana w [`eva_search.md`](eva_search.md) i [`eva_metadata.md`](eva_metadata.md). Chunk z cytatem-dowodem znalazł się w kontekście dla 239 z 300 pytań (79,7%).

### Generowanie

| Model | Klient | Uwagi |
|---|---|---|
| `gpt-4o-mini` | OpenAI | **model bazowy** porównań (domyślny w produkcji) |
| `gpt-4o` | OpenAI | |
| `bielik` | Ollama, `SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M` | lokalnie, jedno zapytanie naraz |

Wszystkie z `temperature=0.1` (jak w produkcji), z tym samym promptem systemowym (`_get_system_prompt`: odpowiadaj tylko na podstawie źródeł, cytuj `[n]`, a gdy brak informacji — użyj zdania odmowy `NO_ANSWER_FALLBACK`). Odpowiedzi są strumieniowane jak w aplikacji, co pozwala zmierzyć **czas do pierwszego tokenu** (TTFT) i czas całej odpowiedzi. Pytania są jednoturowe, bez historii rozmowy.

### Sędzia

Każdą odpowiedź ocenia `gpt-4.1` (`temperature=0`, odpowiedź w ustalonym schemacie JSON). Sędzia widzi pytanie, odpowiedź wzorcową i cytat-dowód z generatora pytań, **ten sam kontekst, który dostał model**, oraz odpowiedź modelu. Zwraca:

| Pole | Skala | Znaczenie |
|---|---|---|
| `context_sufficient` | yes / partial / no | czy kontekst w ogóle zawiera odpowiedź (ocena kontekstu, nie modelu) |
| `refused` | tak / nie | czy model odmówił odpowiedzi |
| `correctness` | 0–2 | zgodność z odpowiedzią wzorcową (2 = poprawna i kompletna, odmowa = 0) |
| `faithfulness` | 0–2 | czy każde twierdzenie wynika z kontekstu (0 = istotne treści zmyślone, odmowa = 2) |
| `citations` | 0–2 lub brak | czy znaczniki `[n]` są obecne i wskazują fragmenty, które popierają zdanie (brak przy odmowie) |
| `language` | 0–2 | poprawna, naturalna polszczyzna |
| `rationale` | tekst | 1–2 zdania uzasadnienia |

Odmowa jest wykrywana też dokładnie: odpowiedź zawierająca zdanie `NO_ANSWER_FALLBACK` liczy się jako odmowa niezależnie od sędziego.

### Metryki

* **Correctness, faithfulness, citations, language** — średnia oceny przeliczonej na skalę 0–1.
* **Fully correct** — odsetek odpowiedzi z oceną correctness = 2.
* **Hallucination** — odsetek odpowiedzi z faithfulness = 0.
* **Right refusal** — odsetek odmów wśród pytań, których kontekst **nie** zawierał odpowiedzi (sędzia: `no`); im wyżej, tym lepiej.
* **False refusal** — odsetek odmów wśród pytań, których kontekst zawierał odpowiedź (sędzia: `yes`); im niżej, tym lepiej.
* **TTFT / total** — mediana czasu do pierwszego tokenu i do końca odpowiedzi.

Przedziały ufności 95% z bootstrapu klastrowego po dokumentach docelowych (2000 prób); różnice względem `gpt-4o-mini` z parowanego bootstrapu z korektą Holma po modelach.

### Ocena człowieka

Skrypt zapisuje `judge_review.csv`: 60 losowych ocenionych odpowiedzi z pustymi kolumnami `my_*`. Po wpisaniu własnych ocen polecenie

```bash
python evaluate_models.py --agreement models_eva_stats/run_20261117_234121/judge_review.csv
```

liczy zgodność z sędzią: zgodność dokładną, zgodność z dokładnością do 1 punktu i ważoną kwadratowo kappę Cohena.

### Pamięć podręczna i koszt

Odpowiedzi i oceny są zapisywane w `.eval_cache/model_eval.sqlite` — przerwany przebieg kontynuuje od miejsca przerwania, a ponowne uruchomienie z tymi samymi danymi nic nie kosztuje. Koszt pełnego przebiegu: generowanie ok. 0,14 USD (`gpt-4o-mini`) i 2,38 USD (`gpt-4o`), sędzia ok. 4,90 USD; Bielik działa lokalnie (ok. 10 godzin na 300 pytań).

```bash
python evaluate_models.py --dry-run                     # podzbiór, kontekst i szacunek kosztu, bez wywołań LLM
python evaluate_models.py                               # gpt-4o-mini, gpt-4o, bielik na 300 pytaniach
python evaluate_models.py --models gpt-4o-mini,gpt-4o --n 100
python evaluate_models.py --retrieval current           # kontekst z dzisiejszej konfiguracji produkcji
```

Pliki wynikowe: `report.txt`, `results.json`, `per_answer.csv` (każda odpowiedź z ocenami i uzasadnieniem), `judge_review.csv`, wykresy `models.png` i `refusals.png`.

## 2. Wyniki (300 pytań, 205 dokumentów, kontekst `planned`)

Sędzia uznał kontekst za wystarczający dla 255 pytań, częściowy dla 29 i niewystarczający dla 16.

| Model | Correctness [95% CI] | Fully correct | Faithfulness [95% CI] | Citations [95% CI] | Language |
|---|---|---|---|---|---|
| `gpt-4o-mini` | **0.850** [0.81–0.88] | 0.787 | 0.958 [0.94–0.98] | 0.966 [0.95–0.98] | 1.000 |
| `gpt-4o` | 0.847 [0.81–0.88] | **0.803** | **0.972** [0.96–0.99] | **0.983** [0.97–0.99] | 1.000 |
| `bielik` | 0.630 [0.58–0.68] | 0.570 | 0.940 [0.92–0.96] | 0.879 [0.84–0.92] | 1.000 |

### Różnice względem `gpt-4o-mini`

| Model | Correctness | Faithfulness | Citations | Language |
|---|---|---|---|---|
| `gpt-4o` | −0.003 (p=0.83) | +0.013 (p=0.32) | +0.008 (p=0.26) | 0.000 |
| `bielik` | **−0.220*** (p=0.001) | −0.018 (p=0.32) | **−0.097*** (p=0.001) | 0.000 |

\* p_holm < 0,05.

### Odmowy, halucynacje, szybkość

| Model | Odmowy | Right refusal | False refusal | Hallucination | TTFT (mediana) | Odpowiedź (mediana) |
|---|---|---|---|---|---|---|
| `gpt-4o-mini` | 8% | 0.88 (n=16) | **0.01** (n=255) | 0.017 | 0.7 s | 2.4 s |
| `gpt-4o` | 11% | **0.95** (n=20) | 0.02 (n=257) | **0.010** | 0.7 s | **1.3 s** |
| `bielik` | 31% | 0.91 (n=22) | 0.23 (n=257) | **0.010** | 29.2 s | 104.7 s |

Liczba pytań z kontekstem `no`/`yes` różni się nieco między modelami, bo sędzia ocenia kontekst przy każdej odpowiedzi osobno.

### Correctness według podgrup

| Model | dosłowne (100) | semantyczne (100) | potoczne (100) | dokument (69) | strona (138) | sylabus (93) |
|---|---|---|---|---|---|---|
| `gpt-4o-mini` | **0.915** | 0.840 | 0.795 | **0.790** | 0.888 | **0.839** |
| `gpt-4o` | 0.880 | **0.845** | **0.815** | **0.790** | **0.902** | 0.806 |
| `bielik` | 0.695 | 0.640 | 0.555 | 0.486 | 0.707 | 0.624 |

| Model | dowód w kontekście (239) | dowodu brak w kontekście (61) |
|---|---|---|
| `gpt-4o-mini` | 0.941 | 0.492 |
| `gpt-4o` | 0.952 | 0.434 |
| `bielik` | 0.699 | 0.361 |

## 3. Wnioski — co wynika z danych

1. **`gpt-4o` nie jest lepszy od `gpt-4o-mini`.** Żadna różnica nie jest istotna (correctness −0.003). `gpt-4o` nieco częściej trafnie odmawia (0.95 vs 0.88, ale na 16–20 pytaniach) i ma odrobinę lepsze cytowania, przy ok. 17 razy wyższym koszcie generowania.
2. **Bielik przegrywa głównie przez fałszywe odmowy.** Odmawia w 23% pytań, na które odpowiedź **była** w kontekście (GPT: 1–2%). Odmowa liczy się jako correctness = 0, więc to ona w dużej mierze tłumaczy spadek o 0.22. Kiedy Bielik odpowiada, rzadko zmyśla: faithfulness i odsetek halucynacji są na poziomie GPT (różnica nieistotna). Słabsze są też cytowania (−0.097): częściej brakuje znaczników `[n]` lub wskazują zły fragment.
3. **Bielik jest za wolny na obecnym sprzęcie.** Mediana 29 s do pierwszego tokenu i 105 s na odpowiedź — dla użytkownika czatu to za dużo, niezależnie od jakości. Pomiar na laptopie z kartą NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GB VRAM; procesor Intel Core i7-12700H, 16 GB RAM. Model w kwantyzacji Q4_K_M zajmuje ok. 7 GB, więc nie mieści się w 4 GB pamięci karty i Ollama wykonuje większą część obliczeń na procesorze. Na karcie z co najmniej 8–12 GB pamięci czas odpowiedzi byłby wielokrotnie krótszy (nie mierzono), ale nie zmieniłoby to jakości odpowiedzi ani odsetka fałszywych odmów.
4. **Halucynacje są rzadkie we wszystkich modelach** (1–2% odpowiedzi z faithfulness = 0). Prompt produkcji skutecznie ogranicza model do źródeł.
5. **Jakość odpowiedzi zależy przede wszystkim od wyszukiwania.** Gdy chunk z dowodem jest w kontekście, `gpt-4o-mini` odpowiada poprawnie w 94%; gdy go nie ma — w 49%. Te 49% to przypadki, w których odpowiedź była w innym chunku (np. w kopii dokumentu lub w sąsiednim fragmencie), więc CtxHit@5 z testów wyszukiwania nieco zaniża rzeczywistą użyteczność kontekstu.
6. **Ocena języka nie różnicuje modeli** — wszystkie dostały 1.000. Sędzia nie znalazł błędów polszczyzny ani wtrąceń w żadnej odpowiedzi.
7. **Najtrudniejsze są pytania potoczne i dokumenty** (PDF-y: regulaminy, zarządzenia) — dla wszystkich modeli. Możliwe wyjaśnienie (hipoteza): pytania potoczne używają innych słów niż dokument, a dokumenty mają długie, gęste fragmenty, z których trzeba wybrać właściwą wartość.

## 4. Decyzja i jej granice

**Produkcja zostaje przy `gpt-4o-mini` jako modelu domyślnym.** Odpowiada tak samo dobrze jak `gpt-4o`, przy ułamku kosztu. `gpt-4o` pozostaje na liście `ALLOWED_MODELS` jako opcja bez zmierzonej przewagi. Bielik pozostaje opcją lokalną (bez wysyłania danych do zewnętrznego API), ale nie nadaje się na model domyślny: częściej odmawia, gdy odpowiedź jest w kontekście, a na obecnym sprzęcie (4 GB pamięci karty) odpowiada ok. 1,5–2 minuty.

Granice:
* **Sędzia to model językowy** (`gpt-4.1`) z tej samej rodziny co dwa z trzech ocenianych modeli; możliwe jest, że lepiej ocenia styl odpowiedzi GPT. Zgodność sędziego z oceną człowieka (`judge_review.csv`, 60 odpowiedzi) — **[uzupełnić]**.
* Odpowiedzi wzorcowe pochodzą z generatora pytań ([`eva_datasets.md`](eva_datasets.md)), a pytania są jednoturowe — test nie obejmuje przeformułowania pytań w rozmowie.
* Prompt systemowy jest jeden dla wszystkich modeli; mógłby nie być optymalny dla Bielika (np. jego skłonność do odmowy może zależeć od sformułowania instrukcji). Test mierzy modele z obecnym promptem produkcji.
* Kontekst pochodzi z próbki 2000 dokumentów i konfiguracji `planned`; na pełnej bazie (107 tys. plików) kontekst bywa trudniejszy. Całość zostanie sprawdzona w walidacji końcowej z pytaniami pisanymi ręcznie.