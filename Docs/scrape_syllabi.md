# `scrape_syllabi.py` — scraper sylabusów

Dedykowany skrypt dla serwisu `sylabusy.agh.edu.pl`. Sylabusy nie są zwykłymi stronami (są ładowane jako okna modalne), więc nie obsługuje ich `scrape_html.py`. Wynik trafia do `{data_root}/Sylabusy_Markdown/` i jest indeksowany jako kolekcje `extra_collections` typu `sylabus` ([`config.md`](config.md)).

## 1. Uruchamianie i konfiguracja

```bash
python scrape_syllabi.py
```

Skrypt **nie ma argumentów CLI** i wykonuje się w całości przy imporcie modułu (kod na poziomie pliku, bez funkcji `main`). Ustawienia zmienia się w stałych na górze pliku:

| Stała | Wartość | Znaczenie |
|---|---|---|
| `TEST_LIMIT` | `False` | po zmianie na `True` skrypt zatrzymuje się po `MAX_SYLLABUSES` (5) zapisanych plikach |
| `START_URL` | `https://sylabusy.agh.edu.pl/pl/1/2/22/0/0` | strona startowa (bieżący rocznik) |
| `BASE_FOLDER` | `data_root` z `sites.yaml` | korzeń zapisu; nazwa `Sylabusy_Markdown` jest wpisana na sztywno |
| `User-Agent` | `Mozilla/5.0 … Chrome/120` | **nie** identyfikuje bota (inaczej niż `scrape_html.py`) |

Brak `robots.txt` i brak kontaktu w nagłówku — skrypt ogranicza obciążenie tylko przerwą 0,2 s po każdym sylabusie. Ponowienia: do 3 prób (429/500/502/503/504, `backoff_factor=1`, `Retry-After`).

## 2. Przebieg

1. Pobranie strony startowej; z filtra „Rozpoczęcie studiów” odczytywana jest lista **roczników** (aktywny = `START_URL`, pozostałe z linków). Nazwa rocznika przechodzi przez `clean_name`, więc `2025/2026` → `2025_2026`. Brak strony startowej kończy skrypt kodem 1.
2. Dla każdego rocznika: lista **wydziałów** → dla każdego wydziału lista **kierunków** (nagłówki) → dla każdego kierunku lista **trybów/rodzajów studiów** (linki, np. „stacjonarne, I stopień” → `stacjonarne_I stopień`).
3. Na stronie trybu zbierane są przyciski z `data-syllabus-id`; w obrębie jednej strony trybu powtórzone identyfikatory są pomijane.
4. Dla każdego sylabusa pobierany jest fragment `/pl/document/{id}.html?partialType=modal&forceEpd=false` i konwertowany `markdownify` (nagłówki ATX, bez `script`/`style`, zredukowane puste linie). **Nie jest dodawany front matter** — tytuł dokumentu ustala indeksator z pierwszego nagłówka `#` lub nazwy pliku.

Struktura wyniku:

```text
Sylabusy_Markdown/{rocznik}/{wydział}/{kierunek}/{tryb}/{przedmiot}.md
```

## 3. Nazwy plików

`clean_name` zamienia znaki niedozwolone w Windows (`\ / * ? : " < > |`) na `_`, usuwa końcowe spacje i kropki, **przycina do 120 znaków i dopisuje `...`**, a nazwy zarezerwowane (`CON`, `PRN`, `AUX`, `NUL`, `COM1–3`, `LPT1–3`) uzupełnia sufiksem `_file`. Nazwa pliku = nazwa przedmiotu, bez identyfikatora sylabusa.

## 4. Wznawianie i idempotentność

Brak pliku stanu: jeśli plik docelowy już istnieje, sylabus jest pomijany (`Skipped`). Błąd sieci przy sylabusie zwiększa licznik błędów i nie tworzy pliku, więc następne uruchomienie ponowi próbę. Konsekwencje:

* **zmiany treści istniejącego sylabusa nie są odświeżane** — trzeba usunąć plik;
* **dwa przedmioty o tej samej nazwie w tym samym trybie** dają jeden plik — drugi jest pomijany jako „istniejący”, więc jego sylabus nie trafi do bazy;
* ten sam sylabus występujący w kilku kierunkach jest pobierany osobno dla każdej ścieżki (deduplikacja dotyczy tylko jednej strony trybu).

Foldery są tworzone przed pobraniem listy przedmiotów, więc mogą zostać puste.

## 5. Raport

Zapisywany do `syllabuses_statistics.txt` w **bieżącym katalogu roboczym** (nie w katalogu danych): czas działania, liczba przetworzonych plików, zapisanych, pominiętych, błędów, rozmiar tekstu (MB) i średni czas na plik.

## 6. Powiązanie z indeksem

Rocznik wykryty w serwisie, którego nie ma w `extra_collections`, zostanie pobrany, ale **nie zostanie zindeksowany**, dopóki nie dopiszesz wpisu (`path: Sylabusy_Markdown/2027_2028`, `source: Sylabusy AGH`, `type: sylabus`). Nazwa folderu musi dokładnie odpowiadać wynikowi `clean_name` z nazwy rocznika.