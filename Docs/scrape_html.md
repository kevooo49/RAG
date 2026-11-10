# `scrape_html.py` — crawler stron HTML → Markdown

Jeden crawler dla wszystkich witryn AGH. Zachowanie każdej witryny opisuje wpis w `config/sites.yaml` ([`config.md`](config.md)). Wynik: pliki Markdown z front matter, lista linków do PDF-ów oraz stan umożliwiający wznowienie.

Crawler jest **synchroniczny i jednowątkowy** (`requests` + `sync_playwright`), przechodzi witrynę wszerz (BFS, kolejka `deque`). Import `playwright` jest bezwarunkowy — pakiet trzeba mieć zainstalowany nawet dla witryn bez JavaScriptu.

## 1. Uruchamianie

```bash
python scrape_html.py --list            # lista witryn z sites.yaml
python scrape_html.py bg                # crawl (wznawia, jeśli istnieje stan)
python scrape_html.py bg --max-pages 50 # test
python scrape_html.py bg --reset        # crawl od zera
python scrape_html.py bg --config inny.yaml
```

`--reset` ignoruje zapisany `stan.json` i usuwa `manifest.jsonl`; **nie usuwa** wcześniej zapisanych plików `.md` (zostaną nadpisane lub pozostaną jako osierocone, jeśli strona zniknęła).

## 2. Parametry globalne (stałe w kodzie)

| Stała | Wartość | Znaczenie |
|---|---|---|
| `USER_AGENT` | `AGH-RAG-Crawler/1.0 (+contact: …)` | jawny identyfikator z adresem kontaktowym |
| `DELAY` | 0,3 s | minimalna przerwa; jeśli `robots.txt` ma większy `Crawl-delay`, obowiązuje on |
| `TIMEOUT` | 5 s połączenie / 20 s odczyt | |
| `MAX_BYTES` | 5 MB | większe strony są odrzucane |
| `MAX_QUERY_VARIANTS` | 15 | limit wariantów zapytania (`?a=…`) dla jednej ścieżki |
| `SAVE_STATE_EVERY` | 25 | częstotliwość zapisu stanu |
| `MAX_JUNK_CHARS` | 3000 | ochrona przed usunięciem „zbyt dużego” elementu (§5) |

Ponowienia HTTP: do 3 prób z narastającym opóźnieniem (`backoff_factor=1`) dla kodów 429/500/502/503/504, z uwzględnieniem `Retry-After`.

## 3. Wybór pobierania: `requests` albo Playwright

To **alternatywa wybierana per witryna** (`requires_js` w profilu), a nie dwa równoległe mechanizmy. Dla `requires_js: true` (w repo: `rekrutacja`, `metal`, `imir`, `skos`) każda strona jest ładowana w Chromium (`wait_until="networkidle"`, limit 30 s); **dla każdej strony uruchamiana jest nowa przeglądarka**, co jest wolne, ale proste. Wymaga `playwright install chromium`. Dla stron z Playwrighta nie ma nagłówka `Last-Modified`, więc `modification_date` jest pusty. Odpowiedzi nie-HTML (po `Content-Type`) są odrzucane w obu trybach.

## 4. Kolejka i filtrowanie adresów

**Start:** adres startowy trafia do kolejki, a następnie (tylko przy świeżym starcie, nie przy wznowieniu) adresy z map witryny — z `robots.txt` lub, gdy ich brak, `/sitemap.xml`; indeksy map są rozwijane do głębokości 2.

**Normalizacja (`normalize_url`):** wymuszony `https`, bez fragmentu `#…`, bez domyślnych portów, sklejone `//`, usunięty `/index.html|htm|php` i końcowy `/`, posortowane parametry zapytania, usunięte parametry śledzące (`utm_*`, `mc_*`, `pk_*`, `fbclid`, `gclid`, `chash`, `no_cache`, `phpsessid`, `sid`, `jsessionid`).

**Odrzucane adresy (`evaluate_url`)** — kolejność sprawdzeń: schemat inny niż http(s) → inna domena → wersja obcojęzyczna (prefiks z `foreign_language_prefixes` lub `?lang=` ≠ `pl`) → ścieżka z `excluded_paths` → PDF (trafia na listę, nie do kolejki) → inny plik binarny (`doc(x)`, `xls(x)`, `zip`, obrazy, `csv`, `txt`, …) → parametr „widoku” (`print`, `sort`, `order`, `orderby`, `sortby`, `format`, `output`, `ical`).

**Domena:** porównanie hosta bez `www.` i portu; wyjątki zaszyte w kodzie: `cdn.imir.agh.edu.pl`, `cok.agh.edu.pl`, `www.cok.agh.edu.pl`.

**Pułapki zapytań:** dla jednej ścieżki dozwolonych jest najwyżej 15 różnych zestawów parametrów; kolejne są liczone jako `skipped_query_trap`. Dodatkowo odrzucana jest głębokość powyżej `max_depth`.

**Przekierowania:** po pobraniu adres końcowy jest ponownie normalizowany i sprawdzany (domena, ścieżki); odwiedzone adresy nie są powtarzane.

**robots.txt:** respektowany (`can_fetch`). Brak pliku lub odpowiedź inna niż poprawna oznacza zgodę na wszystko.

## 5. Czyszczenie HTML i konwersja

1. **Wybór treści:** pierwszy selektor z `content_selectors`, który zwraca niepusty element; gdy lista pusta lub nic nie pasuje — `<body>`.
2. **Usuwanie bezwarunkowe:** `script, style, noscript, template, svg, iframe, canvas, object, embed, link, meta`.
3. **Usuwanie ostrożne** (nawigacja, formularze, przyciski, breadcrumbs, cookies, udostępnianie, paginacja…): element jest usuwany **tylko jeśli ma ≤ 3000 znaków tekstu** — by selektor typu `[class*=share]` nie skasował całej strony.
4. **Usuwanie strukturalne** (`header`, `footer`, `aside`) — tylko gdy treść pochodzi z `<body>` (fallback).
5. Selektory z `additional_junk_selectors` są usuwane dodatkowo.
6. Konwersja `markdownify` (nagłówki ATX); obrazy są usuwane; linki względne zamieniane na bezwzględne; odnośniki `#kotwica` i `javascript:` zostają samym tekstem, `mailto:`/`tel:` pozostają.
7. **Stopka:** jeśli profil ma `footer_selectors`, stopka jest wyciągana z **pierwszej** odwiedzonej strony i zapisywana raz jako `_stopka.md` (`document_type: footer`). Dzięki temu dane kontaktowe są w indeksie, ale nie powielają się na każdej stronie.

**Wyjątki zależne od witryny (zaszyte w kodzie, nie w YAML):**
* adresy zawierające `historia` — usuwana sekcja „Rocznice”, a linki w sekcjach „Bibliografia”/„Źródła” są odlinkowywane (struktura Wikipedii/MediaWiki);
* strona główna `imir` — do kolejki wstrzykiwane są ręcznie wskazane adresy podstron (11 URL-i), bo nie wynikają z nawigacji.

Dlatego „nowa witryna = tylko wpis w YAML” jest prawdą dla typowych serwisów, ale nie dla takich przypadków.

## 6. Kryteria zapisu strony

Strona jest pomijana (ze statystyką), gdy: `<meta robots>` zawiera `noindex`; `<html lang>` nie należy do `allowed_languages` (puste `lang` przechodzi; `allowed_languages: null` wyłącza filtr); Markdown ma mniej niż `min_chars` znaków; jej **treść** powtarza już zapisaną stronę — porównywany jest SHA-1 tekstu po usunięciu adresów linków, zamianie na małe litery i normalizacji znaków (`skipped_duplicates`).

## 7. Wynik

```text
{Prefix}_Strony/
   nazwa_ze_ścieżki_ab12cd34.md     # slug ścieżki (do 80 zn.) + 8 znaków SHA-1 URL
   _stopka.md
{Prefix}_Strony_meta/
   stan.json        # visited, queue, hashes, statystyki — wznowienie
   pdf_linki.txt    # posortowane adresy .pdf, po jednym w linii
   manifest.jsonl   # wpis na każdą zapisaną stronę
   crawler.log
```

Front matter każdej strony (wartości zapisane jako JSON): `url`, `title`, `document_type: html`, `language`, `url_path` (segmenty ścieżki), `modification_date` (nagłówek `Last-Modified`), `download_date`, `char_count`. Indekser używa `url` jako linku w odpowiedzi.

Stan jest zapisywany w pętli (przy zapisanej stronie, gdy licznik przetworzonych adresów jest wielokrotnością 25), przy `Ctrl+C` i na końcu (`finally`); adres przetwarzany w chwili przerwania wraca na początek kolejki. Wznowienie = to samo polecenie.

**Raport końcowy:** odwiedzone, zapisane `.md`, znalezione linki PDF, zbyt krótkie, duplikaty treści, błędy/nie-HTML, wykluczone ścieżki, pułapki zapytań, pozostałe w kolejce.

## 8. Ograniczenia

* `pdf_linki.txt` zawiera **tylko PDF-y**; DOCX/XLS/ODT są pomijane jako pliki binarne (patrz [`scrape_documents.md`](scrape_documents.md)).
* Playwright startuje przeglądarkę per strona (wolno przy dużych serwisach z JS).
* Nie ma równoległości — duże witryny crawluje się godzinami.
* Duplikaty wykrywane są tylko dla identycznej treści, nie dla podobnej.