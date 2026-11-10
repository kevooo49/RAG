# Ewaluacja modeli generujących (`evaluate_models.py`)

Skrypt porównuje trzy modele odpowiadające na pytania na podstawie **zadanego, idealnego kontekstu** (cały właściwy dokument), niezależnie od wyszukiwarki. Mierzy więc zachowanie modelu przy poprawnym wejściu, a nie jakość całego RAG. Zbiór: `llm_eva_dataset.json` — 90 rekordów z odpowiedziami wzorcowymi modelu `o3` ([`eva_datasets.md`](eva_datasets.md), §3). Wynik: `models_eva_stats/run_20261105_105251/`.

## 1. Metoda

* **Modele** (wszystkie `temperature=0`): `gpt-4o-mini` (baza porównań, `--baseline`), `gpt-4o`, `bielik-11b` (`SpeakLeash/bielik-11b-v3.0-instruct:Q4_K_M` przez Ollamę). Produkcja używa `temperature=0.1`.
* **Prompt generowania** jest prosty i własny dla skryptu: „Odpowiedz krótko i bezpośrednio… WYŁĄCZNIE na poniższym tekście”, po nim tekst i pytanie. To **nie jest prompt systemowy produkcji** (`backend.md`, §6.3), więc nie testuje cytowania `[n]`, formatu znaczników `<source>` ani zachowania odmowy.
* **Szum:** z prawdopodobieństwem 50% (`NOISE_PROBABILITY`, `random.seed(42)`) do kontekstu dopisywane jest pierwsze 400 znaków **innego** dokumentu z etykietą `[DODATKOWE INFORMACJE Z INNEGO DOKUMENTU]`. Ten sam kontekst (z szumem lub bez) dostają wszystkie modele dla danego pytania.
* **Sędzia:** model strukturalny (`with_structured_output`) ocenia odpowiedź w skali 1–5 w trzech wymiarach: *faithfulness* (wierność), *relevance* (trafność), *language_quality* (jakość polszczyzny), z krótkim uzasadnieniem.
* **Cache:** SQLite `.eval_cache/llm_eval.sqlite`, dwie tabele: `gen` (odpowiedź + opóźnienie, klucz = SHA-256 modelu i promptu) i `eval` (oceny, klucz = SHA-256 pytania, wzorca i odpowiedzi).
* **Statystyka:** 2000 prób bootstrapowych, ziarno 42; **próbkowanie po pytaniach** (nie po dokumentach, jak w testach wyszukiwania); różnice względem modelu bazowego z parowanym bootstrapem (CI 95%, dwustronne *p*).
* **Opóźnienie:** czas pojedynczego, nie-strumieniowego wywołania `invoke` (pełna odpowiedź). Dla Bielika zależy od sprzętu, którego raport nie opisuje.

```bash
python evaluate_models.py                       # baza: gpt-4o-mini
python evaluate_models.py --baseline gpt-4o
```

## 2. Wyniki (`run_20261105_105251`, 90 pytań)

| Model | Wierność | Trafność | Język | Opóźnienie |
|---|---|---|---|---|
| `gpt-4o-mini` | 4.49 | 4.62 | 4.84 | 1.44 s |
| `gpt-4o` | 4.57 | 4.61 | 4.94 | 1.40 s |
| `bielik-11b` | 3.18 | 3.89 | 4.63 | 96.6 s |

Różnice względem `gpt-4o-mini`:

| Model | Δ wierność [95% CI] | *p* | Δ trafność [95% CI] | *p* |
|---|---|---|---|---|
| `gpt-4o` | +0.08 [−0.09, +0.24] | 0.398 | −0.01 [−0.16, +0.13] | 0.942 |
| `bielik-11b` | −1.31 [−1.68, −0.96] | < 0.001 | −0.73 [−1.06, −0.41] | < 0.001 |

Wierność w zależności od szumu w kontekście:

| Model | Czysty kontekst | Z szumem | Spadek |
|---|---|---|---|
| `gpt-4o-mini` | 4.61 | 4.34 | 0.27 |
| `gpt-4o` | 4.59 | 4.54 | 0.06 |
| `bielik-11b` | 3.12 | 3.24 | −0.12 |

## 3. Wnioski i ich granice

* **`gpt-4o` i `gpt-4o-mini` są statystycznie nieodróżnialne** pod względem wierności i trafności (CI obejmują 0); `gpt-4o` ma jedynie wyższą ocenę języka (4.94 vs 4.84; różnicy nie testowano). Przy braku różnicy jakościowej tańszy model jest rozsądnym wyborem — ale domyślny model w kodzie to **`gpt-4o`** (`schemas.py`, pierwsza opcja selektora we frontendzie), więc wybór nie jest wymuszony na poziomie systemu.
* **Bielik 11B (kwantyzacja Q4_K_M) wypada wyraźnie gorzej** (wierność −1.31, trafność −0.73, *p* < 0.001) i odpowiada przeciętnie ok. 97 s. Dotyczy to konkretnego wariantu modelu uruchomionego na sprzęcie testowym; wniosków nie należy uogólniać na inne kwantyzacje czy konfiguracje GPU.
* **Odporność na szum:** spadek wierności `gpt-4o-mini` (0.27) jest większy niż `gpt-4o` (0.06), co *sugeruje* większą podatność mniejszego modelu na zmyłki. To jednak porównanie dwóch średnich z ok. 45 pytań z szumem; spadku nie testowano statystycznie, a ujemny „spadek” Bielika pokazuje, jak duży jest szum pomiarowy. Sformułowania typu „udowodnił wybitną stabilność” nie są wspierane przez dane.

## 4. Zastrzeżenia metodologiczne

1. **Sędzia nie widzi kontekstu.** Prompt sędziego zawiera pytanie, **odpowiedź wzorcową (`o3`)** i odpowiedź modelu. „Wierność” to więc w praktyce zgodność z wzorcem, a nie weryfikacja względem dokumentu źródłowego.
2. **Błąd sędziego = ocena 1/1/1.** Wyjątek podczas oceniania jest zamieniany na wynik „Błąd” z najniższymi notami, a błąd generowania zapisuje się jako odpowiedź `ERROR: …` i jest oceniany jak zwykły tekst.

## 5. Decyzja

Wynik dobrze uzasadnia odrzucenie Bielika 11B Q4 jako domyślnego modelu odpowiedzi oraz fakt, że droższy `gpt-4o` nie daje mierzalnej poprawy wierności i trafności w tym teście. Wybór `gpt-4o-mini` jest racjonalny kosztowo, ale test nie rozstrzyga ewentualnych różnic w zachowaniu na prawdziwym prompcie systemowym, z cytowaniami i z prawdziwym wyszukiwaniem.