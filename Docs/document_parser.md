# `document_parser.py` — PDF/DOCX → Markdown

Skrypt zamienia pobrane pliki (`{prefix}_Dokumenty_Surowe`) na pliki Markdown z front matter (`{prefix}_Dokumenty_Przetworzone`), gotowe dla indeksera ([`build_database.md`](build_database.md)). Jest trzecim krokiem potoku: `scrape_html.py` → `scrape_documents.py` → **`document_parser.py`**.

## 1. Uruchamianie

```bash
python document_parser.py bg                          # foldery z config/sites.yaml
python document_parser.py --raw R --processed P --temp T --domain https://…   # bez konfiguracji
```

Argumenty: `site` (opcjonalny klucz z `sites.yaml`), `--raw`, `--processed`, `--temp`, `--domain` (adres zapisywany do front matter), `--config`. Wartości podane jawnie mają pierwszeństwo nad tymi z konfiguracji. Skrypt kończy się błędem, gdy folder źródłowy nie istnieje. Klucz `LLAMA_CLOUD_API_KEY` jest czytany z `.env` przez `load_dotenv()` (nie przez `Settings` backendu); jego brak daje tylko ostrzeżenie — pliki wymagające chmury zakończą się wtedy błędem i zostaną pominięte.

## 2. Przetwarzanie pliku

Folder źródłowy jest czytany **płasko** (bez podfolderów). Dla każdego pliku:

1. Rozszerzenie inne niż `pdf`/`docx` → komunikat `Skipped … unsupported format` (dotyczy m.in. `.doc`, `.odt`).
2. Jeśli wynik `{nazwa}.md` już istnieje i ma > 150 bajtów → plik pomijany (idempotentność; plik krótszy, np. sam front matter po pustej odpowiedzi chmury, zostanie przetworzony ponownie).
3. **DOCX** jest konwertowany do tymczasowego PDF-a przez `docx2pdf`. Ten pakiet jest opcjonalny i (zgodnie ze swoją konstrukcją) wymaga zainstalowanego Microsoft Word. **Bez `docx2pdf` pliki DOCX są po cichu pomijane** — jedyną informacją jest ostrzeżenie przy starcie skryptu. Błąd konwersji kończy się pominięciem pliku.
4. **Wybór metody dla PDF** (`requires_advanced_parsing`) — przetwarzanie w chmurze, jeśli *którakolwiek* strona:
   * ma mniej niż 50 znaków tekstu (skan, strona-obraz, ale też prawie pusta strona zwykłego dokumentu),
   * zawiera więcej niż 2 obrazy,
   * zawiera tabelę wykrytą przez `page.find_tables()`.

   Błąd analizy też wymusza chmurę. W pozostałych przypadkach używane jest lokalne PyMuPDF (`fitz`), bez kosztów.
5. Zapis `{nazwa}.md`; plik tymczasowy jest usuwany.

Konsekwencja heurystyki „dowolna strona”: długi, jednolity dokument z jedną tabelą lub jedną niemal pustą stroną trafi w całości do płatnego LlamaParse.

## 3. Metody ekstrakcji

| Metoda | Wynik | Uwagi |
|---|---|---|
| `pymupdf` | **zwykły tekst** (`get_text("text")`), strony rozdzielone pustą linią | bez nagłówków Markdown, tabel i list — struktura, na której opiera się chunkowanie, jest wtedy uboższa |
| `llamaparse` | Markdown (`markdown_full`) | API v2 (`/api/v2/parse/upload`), `tier: agentic`, `version: latest`; zadanie odpytywane co 2 s, maksymalnie 60 razy (~2 min) |

Błędy LlamaParse (brak klucza, odrzucone zadanie `FAILED/CANCELLED`, przekroczony czas, błąd HTTP) powodują pominięcie pliku bez zapisu, więc kolejne uruchomienie spróbuje go ponownie. Plik jest przetwarzany sekwencyjnie i synchronicznie.

## 4. Front matter

```yaml
---
plik: "regulamin_studiow_ab12cd34.pdf"   # nazwa pliku źródłowego (z rozszerzeniem)
typ_dokumentu: docx                      # rozszerzenie ORYGINAŁU (DOCX zostaje docx)
zrodlo: https://bg.agh.edu.pl/           # start_url witryny lub --domain; puste, jeśli brak
metoda_ekstrakcji: pymupdf               # pymupdf | llamaparse
---
```

`zrodlo` to adres **domeny**, a nie URL konkretnego dokumentu. Indekser nadpisuje `zrodlo` nazwą źródła z `sites.yaml`, a pola `url` dokumenty nie mają — w efekcie chunki dokumentów nie niosą linku do pliku (patrz [`build_database.md`](build_database.md), §9.1a). Tytuł dokumentu wybierany jest przez indekser: pierwszy nagłówek `#`, a gdy go brak (tekst z PyMuPDF) — nazwa pliku.

## 5. Raport

Na końcu: `DOCX->PDF: n | Local (Free): n | AI Cloud: n`.

## 6. Ograniczenia

* Brak obsługi `.doc`, `.odt` i podfolderów.
* Zależność DOCX od Microsoft Word (przez `docx2pdf`) ogranicza ten krok do Windows/macOS z Wordem.
* Jakość lokalnej ekstrakcji (zwykły tekst) jest niższa niż chmurowej; nie ma kontroli jakości wyniku poza progiem 150 bajtów.
* Koszt zależy od liczby plików kierowanych do chmury; skrypt go nie szacuje.