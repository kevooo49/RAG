# Dokumentacja Architektoniczna: Parser Dokumentów Binarnych (`document_parser.py`)

## 1. Wprowadzenie i Rola Modułu

Skrypt `document_parser.py` pełni funkcję pomostu technologicznego między surowymi plikami binarnymi (pobranymi uprzednio przez moduł `scrape_documents.py`) a indeksatorem bazy wektorowej. Modele osadzające operują wyłącznie na tekście, co wymusza transformację dokumentów takich jak regulaminy (PDF) czy wnioski dziekanatowe (DOCX) do ustrukturyzowanego formatu Markdown. Skrypt wyróżnia się inteligentnym zarządzaniem kosztami ekstrakcji oraz pełną integracją z centralnym systemem konfiguracji, automatycznie mapując katalogi źródłowe i docelowe.

## 2. Hybrydowy Silnik Ekstrakcji (Heurystyka Kosztów)

Zastosowanie potężnych modeli wizyjnych (VLM) do każdego pliku PDF wygenerowałoby gigantyczne koszty i opóźnienia. Aby zoptymalizować budżet, parser implementuje autorską heurystykę (`requires_advanced_parsing`), która analizuje strukturę dokumentu i decyduje o ścieżce przetwarzania:

* **Przetwarzanie Lokalne (PyMuPDF / fitz)**: Jeśli dokument zawiera czysty, wyselekcjonowany tekst, moduł przetwarza go lokalnie na procesorze maszyny. Operacja ta jest błyskawiczna i całkowicie darmowa.
* **Przetwarzanie w Chmurze (LlamaParse)**: Algorytm kieruje plik do płatnego API LlamaCloud w trzech precyzyjnie zdefiniowanych przypadkach:
1. **Brak warstwy tekstowej (skany)**: Gdy strona zawiera mniej niż 50 wyekstrahowanych znaków (konieczność użycia OCR).
2. **Nagromadzenie grafik**: Gdy wykryto więcej niż 2 obrazy na stronie, co sugeruje złożony układ graficzny wymagający modelu wizyjnego.
3. **Obecność tabel strukturalnych**: Gdy lokalny detektor wykryje osadzone tabele. Klasyczne parsery niszczą strukturę tabel, zlewając kolumny w nieczytelny tekst. LlamaParse w trybie `agentic` perfekcyjnie rekonstruuje je do składni tabelarycznej Markdown, co jest krytyczne dla późniejszego podziału tekstu w `build_database.py`.



## 3. Normalizacja Formatów i Obsługa DOCX

System RAG wymaga ujednoliconego wejścia tekstowego. Dokumenty pakietu Microsoft Office (pliki `.docx`) są przechwytywane i normalizowane przed procesem ekstrakcji:

* Moduł weryfikuje dostępność biblioteki `docx2pdf`. W przypadku jej obecności, plik Word przenoszony jest do wyizolowanego katalogu tymczasowego (`temp`).
* Następuje lokalna konwersja do formatu PDF.
* Wygenerowany PDF trafia do hybrydowego silnika ekstrakcji, a po pomyślnym wyciągnięciu tekstu, plik tymczasowy jest bezpowrotnie usuwany. Gwarantuje to czystość przestrzeni roboczej.

## 4. Wstrzykiwanie Metadanych (Frontmatter)

Wyekstrahowany tekst nie jest zapisywany w formie surowej. Aby system wektorowy nie zgubił kontekstu pliku binarnego, funkcja `add_frontmatter` dokleja na samym szczycie pliku nagłówek w formacie YAML. Obejmuje on:

* `plik`: oryginalną nazwę pliku z rozszerzeniem, niezbędną do generowania poprawnych cytowań w interfejsie użytkownika.
* `typ_dokumentu`: rozszerzenie (np. `pdf`, `docx`), używane przy logicznym filtrowaniu bazy.
* `zrodlo`: oficjalną domenę źródłową lub nazwę jednostki AGH.
* `metoda_ekstrakcji`: znacznik (`llamaparse` lub `pymupdf`) ułatwiający późniejsze debugowanie błędów w parsowaniu tekstu.

## 5. Integracja z Konfiguracją i Interfejs CLI

Skrypt bezpośrednio dziedziczy architekturę ścieżek z pliku `config/sites.yaml`. Dzięki importowi modułu `sites.py` nie wymaga on ręcznego wprowadzania skomplikowanych adresów systemowych.

Interfejs linii poleceń pozwala na elastyczne uruchamianie procesu w dwóch trybach:

* **Wywołanie z kluczem domeny (np. `python document_parser.py cok`)**: Skrypt automatycznie pobiera z konfiguracji YAML ścieżki do folderu surowego (`raw`), docelowego (`processed`) oraz tymczasowego (`temp`), a także ładuje oficjalną nazwę domeny.
* **Wywołanie z jawnym nadpisaniem ścieżek (Override)**: Przydatne w potokach CI/CD lub przy testowaniu niestandardowych zbiorów danych. Użytkownik może samodzielnie zdefiniować flagi: `--raw`, `--processed`, `--temp`, `--domain` oraz wskazać alternatywny plik konfiguracyjny flagą `--config`.

## 6. Bezstanowość i Odporność (Resilience)

Podobnie jak poprzednie narzędzia potoku pobierania, skrypt został zaprojektowany do długotrwałej pracy asynchronicznej:

* **Idempotentność**: Algorytm sprawdza, czy docelowy plik `.md` już istnieje i czy jego rozmiar przekracza 150 bajtów (co chroni przed plikami zapisanymi z błędem pustego frontmattera). Jeśli warunek jest spełniony, kosztowne parsowanie jest pomijane. Pozwala to na bezpieczne przerywanie i wznawianie procesu.
* **Odporność na Timeouty API**: Komunikacja z chmurą LlamaParse opiera się na mechanizmie asynchronicznego odpytywania (ang. polling). Skrypt odpytuje serwer o status zadania co 2 sekundy (przez maksymalnie 2 minuty), a w przypadku odrzucenia (status `FAILED` lub `CANCELLED`) precyzyjnie raportuje komunikat błędu z serwera, po czym przechodzi do kolejnego dokumentu bez przerywania całej kolejki.
* Na koniec działania generowany jest czytelny raport w konsoli, sumujący liczbę pomyślnie znormalizowanych plików DOCX, plików przetworzonych za darmo lokalnie oraz dokumentów poddanych zaawansowanej obróbce w chmurze AI.