# `scrape_documents.py` — pobieranie dokumentów

Drugi krok potoku danych. Skrypt **nie przeszukuje witryny ponownie** — wczytuje listę linków zapisaną przez `scrape_html.py` i pobiera z niej pliki. Importuje `scrape_html` (sesja HTTP, `User-Agent`, konfiguracja), więc oba pliki muszą leżeć w `scrapers/`.

```bash
python scrape_html.py bg            # 1) tworzy {Prefix}_Strony_meta/pdf_linki.txt
python scrape_documents.py bg       # 2) pobiera pliki z tej listy
python scrape_documents.py bg --max 20     # test na 20 plikach
python scrape_documents.py bg --reset      # usuwa manifest i pobiera wszystko od nowa
```

Argumenty: `profile` (klucz z `sites.yaml`), `--config`, `--max`, `--reset`. Brak pliku `pdf_linki.txt` → komunikat z poleceniem uruchomienia crawlera.

## 1. Wejście i wyjście

| Element | Ścieżka |
|---|---|
| lista linków | `{Prefix}_Strony_meta/pdf_linki.txt` (jeden URL na linię) |
| pobrane pliki | folder `documents_folder` witryny (`{Prefix}_Dokumenty_Surowe`) |
| manifest | `{Prefix}_Strony_meta/documents_manifest.jsonl` (dopisywany linia po linii) |
| log | `{Prefix}_Strony_meta/documents.log` |

**Uwaga o zakresie:** crawler dopisuje do `pdf_linki.txt` wyłącznie linki kończące się na `.pdf`. Linki do `.docx`, `.xlsx`, `.odt` itp. są przez niego tylko zliczane jako pominięte pliki binarne. Mimo że `scrape_documents.py` zna rozszerzenia `.pdf`, `.docx`, `.doc`, `.odt`, w obecnym potoku realnie pobiera **tylko PDF-y**. Plik listy zawiera same adresy, więc pole „roboczy tytuł” w manifeście zostaje puste (kod przewiduje teksty linków, ale crawler ich nie zapisuje).

## 2. Przebieg pobierania

Dla każdego adresu z rozszerzeniem ze zbioru dozwolonych, który nie ma w manifeście statusu końcowego:

1. **robots.txt** — sprawdzany per host (`can_fetch` z tym samym `User-Agent` co crawler). Zakaz → status `robots`.
2. **Pobranie strumieniowe** (timeout 5 s połączenie / 30 s odczyt) do pliku `.part`.
3. **Kontrole:** odpowiedź `404/410` → `missing`; inne `≥400` albo błąd sieci → `error`; `Content-Type: text/html` → `rejected` (serwer podał stronę zamiast pliku); `Content-Length` lub rzeczywisty rozmiar > 50 MB → `rejected`; pusty plik → `rejected`.
4. **Sygnatura pliku** (pierwszy kawałek strumienia, sprawdzane pierwsze 1024 B): PDF musi zawierać `%PDF` w tym zakresie; DOCX/ODT — nagłówek `PK\x03\x04`; DOC — `D0 CF 11 E0`. Niezgodność → `rejected`.
5. **Deduplikacja po treści:** skrót SHA-256 pobranego pliku. Jeśli już jest wśród plików o statusie `ok` — `.part` jest usuwany, status `duplicate` (z polem `duplicate_of`).
6. W pozostałych przypadkach `.part` → `os.replace` do docelowej nazwy (w folderze nigdy nie ma pliku urwanego w połowie).
7. Przerwa po pobraniu: `max(0,5 s, Crawl-delay z robots.txt)` (robots.txt jest pobierany raz na host). Przerwanie Ctrl+C zachowuje manifest i pozwala wznowić tym samym poleceniem; pliki `.part` są sprzątane w bloku `finally`.

Nazwa pliku: czytelna nazwa z URL (znaki spoza `[\w\-.]` → `_`, do 80 znaków) + `_` + 8 znaków SHA-1 adresu URL + rozszerzenie małymi literami.

## 3. Statusy i wznawianie

Statusy końcowe (nie są ponawiane): `ok`, `duplicate`, `missing`, `rejected`, `robots`. Status `error` (problem przejściowy) jest ponawiany przy następnym uruchomieniu. Manifest jest ostatnim stanem dla każdego URL-a (wystarczy rzut oka na ostatni wpis).

Wpis manifestu: `url`, `file`, `extension`, `working_title` i `link_texts` (puste, patrz wyżej), `source_pages` (puste), `download_date` (UTC), `status` oraz — zależnie od wyniku — `sha256`, `size`, `content_type`, `last_modified`, `reason`, `duplicate_of`.

Na końcu skrypt drukuje podsumowanie: liczby `ok / duplicate / missing / rejected / robots / error`.

## 4. Ograniczenia

* Zakres ograniczony do PDF przez format `pdf_linki.txt` (sekcja 1).
* Dalsze przetwarzanie (`document_parser.py`) obsługuje tylko `pdf` i `docx`; `.doc` i `.odt` zostałyby pobrane, ale pominięte.
* `--reset` kasuje manifest, ale **nie usuwa** już pobranych plików (nazwy zależą od URL-a, więc zostaną nadpisane tym samym plikiem).