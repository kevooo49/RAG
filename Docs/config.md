# Konfiguracja źródeł (`config/`)

Katalog `config/` zawiera dwa pliki:

* `sites.yaml` — deklaratywny opis: gdzie leżą dane, które witryny się pobiera i jak, które foldery trafiają do indeksu;
* `sites.py` — loader i walidator (`load_config`, klasa `SitesConfig`), współdzielony przez `scrapers/*` i `backend/scripts/*`. Serwer FastAPI **nie** korzysta z tego modułu.

Dodanie zwykłej witryny sprowadza się do nowego wpisu w `sites:`. Istnieją jednak wyjątki zaszyte w kodzie crawlera (patrz [`scrape_html.md`](scrape_html.md), §8), więc „bez zmian w kodzie” dotyczy większości, ale nie wszystkich przypadków.

## 1. Zawartość `sites.yaml` (stan repozytorium)

* **40 witryn** w `sites:`, każda z sekcją `crawl` i `pages: true`;
* **36** z nich ma `documents` (przetwarzanie PDF/DOCX); bez dokumentów: `eaiiib`, `odlewnictwo`, `historia`, `skos`;
* **6** wpisów w `extra_collections` — sylabusy roczników 2021/2022 … 2026/2027 (`type: sylabus`);
* `requires_js: true` ustawione w 4 witrynach (`rekrutacja`, `metal`, `imir`, `skos`).

### Sekcja `paths`

| Klucz | Wartość w repo | Znaczenie |
|---|---|---|
| `data_root` | `data/Baza_Wiedzy_AGH` | korpus Markdown (jedyny klucz obowiązkowy) |
| `eval_dir` | `data/Ewaluacja_Probka` | próbka plików do ewaluacji (patrz [`eva_datasets.md`](eva_datasets.md)) |
| `index_dir` | `data/Baza_Wektorowa_Chroma_ASYM` | folder **manifestu indeksera** (`_manifest_indeksu.json`). Same wektory leżą w kontenerze Chroma (wolumen Dockera), nie tutaj |
| `id_prefix` | `../../data/Baza_Wiedzy_AGH` | stały przedrostek identyfikatorów chunków; domyślnie nazwa folderu `data_root` |

Ścieżki względne są liczone od korzenia repozytorium (`REPO_ROOT`), nie od bieżącego katalogu.

### Sekcja `layout`

Szablony nazw folderów w `data_root`, każdy musi zawierać `{prefix}`:

| Klucz | Szablon | Zawartość |
|---|---|---|
| `pages` | `{prefix}_Strony` | strony HTML jako `.md` |
| `raw_documents` | `{prefix}_Dokumenty_Surowe` | pobrane PDF/DOCX |
| `processed_documents` | `{prefix}_Dokumenty_Przetworzone` | dokumenty po `document_parser.py` |
| `temp_documents` | `{prefix}_Dokumenty_Tymczasowe` | pliki pośrednie (DOCX→PDF) |

Crawler dodatkowo tworzy obok folderu stron folder `{prefix}_Strony_meta` (stan, logi, manifesty).

### Sekcja `crawl_defaults`

Wartości domyślne dla wszystkich witryn:

| Klucz | Domyślnie | Rola |
|---|---|---|
| `content_selectors` | `#mainContent`, `main`, `[role=main]`, `article`, `#main`, `#content` | pierwszy pasujący selektor z niepustym tekstem wyznacza treść strony |
| `footer_selectors` | `#pageFooter`, `footer` | skąd wyciągnąć stopkę (zapisywaną raz) |
| `additional_junk_selectors` | `[]` | dodatkowe elementy do usunięcia |
| `excluded_paths` | `[]` | ścieżki pomijane (dopasowanie po prefiksie) |
| `foreign_language_prefixes` | `/en /de /ru /uk /fr /es` | prefiksy ścieżek wersji obcojęzycznych — pomijane |
| `allowed_languages` | `[pl]` | dozwolony język z `<html lang>` |
| `min_chars` | `300` | minimalna długość Markdownu strony |
| `max_depth` | `12` | maksymalna głębokość BFS |
| `requires_js` | `false` | renderowanie Playwright zamiast `requests` |

### Sekcja `sites`

Dozwolone klucze profilu: `prefix`, `start_url`, `source`, `pages`, `documents`, `crawl` — inne powodują błąd.

* `prefix` — nazwa używana w nazwach folderów (bez `/` i `\`), unikalna;
* `start_url` — adres startowy (`http://` lub `https://`); obowiązkowy, gdy jest `crawl`;
* `source` — nazwa źródła zapisywana w metadanych (`zrodlo`); obowiązkowa, gdy jest `pages` lub `documents`;
* `pages: true` — indeksuj folder stron;
* `documents` — `true` lub słownik opcji parsera/indeksera: `legal_headers` (rozpoznawanie `Rozdział`/`§`/`Art.`; domyślnie `true`), `bold_as_headers` (domyślnie włączone w indekserze);
* `crawl` — nadpisania dla tej witryny (patrz niżej).

Przykład wpisu:

```yaml
sites:
  bg:
    prefix: Biblioteka
    start_url: https://bg.agh.edu.pl/
    source: Biblioteka Główna AGH
    pages: true
    documents: { legal_headers: true }
    crawl:
      content_selectors: []          # pusta lista = nie szukaj po selektorach, użyj <body> z czyszczeniem strukturalnym
      additional_junk_selectors: ['#pageFooter', '#unitMenu']
      excluded_paths: [/wydarzenia, /aktualnosci, /galeria]
```

> **Uwaga:** `content_selectors: []` oznacza brak selektorów, więc crawler przechodzi na `<body>` i usuwa `header/footer/aside`. W repozytorium tak robi większość witryn.

### Sekcja `extra_collections`

Foldery spoza crawlera HTML, np. `Sylabusy_Markdown/2025_2026`. Wymagane klucze: `path` (względem `data_root`), `source`, `type` ∈ {`strona`, `dokument`, `sylabus`}. Opcjonalnie `legal_headers`, `bold_as_headers`. **Nowy rocznik sylabusów trzeba tu dopisać ręcznie** — `scrape_syllabi.py` tworzy foldery według nazw z serwisu, ale indeksator zobaczy tylko te zadeklarowane.

## 2. Walidacja i mechanizmy `sites.py`

`SitesConfig._validate` zgłasza `ConfigError` (a skrypty CLI — przez `load_or_exit` — czytelny komunikat) gdy:

* klucze `layout` są nieznane albo szablon nie zawiera `{prefix}`;
* profil nie jest słownikiem, zawiera nieznane klucze, nie ma `prefix`, `prefix` zawiera ukośnik lub się powtarza;
* brakuje `source` przy `pages`/`documents` albo `start_url` przy `crawl`; `start_url` nie zaczyna się od `http(s)://`;
* wpis `extra_collections` nie ma `path`/`source`/`type` lub ma niedozwolony `type`.

Pozostałe mechanizmy:

* **Nadpisania środowiskowe:** `RAG_SITES_FILE`, `RAG_DATA_ROOT`, `RAG_EVAL_DIR`, `RAG_INDEX_DIR` mają pierwszeństwo przed YAML-em.
* **`crawl_profile(klucz)`** składa profil crawlera: kopia (`deepcopy`) `crawl_defaults`, na którą nakładane są klucze z `sites.<klucz>.crawl` metodą `dict.update`. To **nadpisanie na poziomie kluczy**, nie scalanie list: np. własne `excluded_paths` *zastępują* domyślną listę, a `allowed_languages: null` wyłącza filtr języka (tak ma `wggiis-studiuj`). Wynik zawiera też `start_url`, `folder` (strony) i `documents_folder` (surowe dokumenty).
* **`collections()`** zwraca listę kolekcji dla indeksera: dla każdej witryny z `pages` — folder stron (`type: strona`), dla `documents` — folder dokumentów przetworzonych (`type: dokument`, z `legal_headers`, domyślnie `true`), a potem `extra_collections`. Każda ma `path` (absolutna) i `id_path` = `{id_prefix}/{folder}` — **stabilna podstawa identyfikatorów, niezależna od miejsca na dysku**.
* **`document_dirs(klucz)`** dostarcza `document_parser.py` ścieżki `raw`/`processed`/`temp` i `start_url` jako domenę źródłową.

## 3. Raport diagnostyczny

```bash
python config/sites.py            # lub: python config/sites.py inny.yaml
```

Wypisuje ścieżki (z informacją czy `data_root` istnieje), liczbę witryn i crawlowalnych, liczbę kolekcji indeksu (strony/dokumenty/sylabusy) oraz foldery, których jeszcze nie ma na dysku (`not on disk yet`).

## 4. Dlaczego `id_prefix`

Identyfikator chunku jest skrótem m.in. ścieżki pliku względem `id_path`. Gdyby opierał się na ścieżce bezwzględnej, przeniesienie projektu na inną maszynę zmieniłoby wszystkie identyfikatory i wymusiło płatne ponowne wektoryzowanie całego korpusu. Dlatego `id_prefix` jest ustawiony jawnie (w repo: `../../data/Baza_Wiedzy_AGH`). **Zmiana `id_prefix` lub `prefix` witryny = nowe identyfikatory dla wszystkich jej plików** i ponowne embeddowanie (patrz [`build_database.md`](build_database.md), §8).

## 5. Testy

`backend/tests/test_sites_config.py` sprawdza m.in., że prawdziwy `sites.yaml` jest poprawny, a `id_path` i `path` kolekcji są unikalne.
