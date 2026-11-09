# Dokumentacja Architektoniczna: Crawler HTML (`scrape_html.py`)

## 1. Wprowadzenie i Rola w Systemie

Skrypt `scrape_html.py` to główny moduł odpowiedzialny za zasilanie bazy wiedzy systemu RAG danymi ze stron internetowych uczelni. Działa on jako asynchroniczny crawler oparty o algorytm przeszukiwania wszerz (BFS). Skrypt jest w pełni sterowany przez plik konfiguracyjny `sites.yaml` i nie wymaga modyfikacji kodu źródłowego przy dodawaniu nowych wydziałów czy jednostek AGH. Jego głównym celem jest pobranie treści, rygorystyczne oczyszczenie jej z elementów interfejsu graficznego (szumu) i wyeksportowanie czystego tekstu do formatu Markdown z zachowaniem metadanych.

## 2. Hybrydowy Silnik Pobierania Danych

Ze względu na różnorodność technologiczną uczelnianych stron internetowych, crawler implementuje dwa równoległe mechanizmy pobierania treści:

* **Sesje HTTP (`requests` + `urllib3.util.retry`)**: Używane domyślnie dla klasycznych, statycznych stron HTML. Mechanizm sesji wykorzystuje wbudowany adapter Retry, który automatycznie ponawia żądania w przypadku napotkania błędów serwera (np. 500, 502) lub uderzenia w Rate Limiter (błąd 429).
* **Renderowanie Dynamiczne (Playwright)**: Wiele nowych systemów uczelnianych to aplikacje SPA (Single Page Application) oparte na frameworkach takich jak React. Jeśli profil źródła posiada flagę `requires_js: true`, skrypt uruchamia przeglądarkę Chromium w trybie headless, czeka na pełne załadowanie asynchronicznych zapytań sieciowych (stan `networkidle`) i dopiero wtedy ekstrahuje wyrenderowany obiekt DOM.

## 3. Nawigacja i Zarządzanie Kolejką (BFS)

Proces poruszania się po witrynie opiera się na kolejce FIFO (`deque`), co gwarantuje płaskie i przewidywalne przeszukiwanie struktury drzewa podstron.

* **Inicjalizacja i Sitemapy**: Crawler rozpoczyna pracę od adresu `start_url` oraz weryfikacji pliku `robots.txt`. Jeśli serwer udostępnia mapę witryny (`sitemap.xml`), jej linki są natychmiast wrzucane do kolejki.
* **Higiena i Normalizacja URL**: Każdy wykryty adres jest normalizowany przed dodaniem do kolejki. Crawler usuwa parametry śledzące (np. `fbclid`, `utm_*`), ujednolica schematy i obcina kotwice (`#`). Dodatkowo mechanizm detekcji wyłapuje tzw. pułapki zapytań – jeśli dany widok ma wygenerowanych zbyt wiele wariantów parametrów sortowania, crawler ignoruje kolejne.
* **Walidacja Zasięgu**: Skrypt pilnuje, aby nie opuścić zdefiniowanej domeny (z uwzględnieniem wyjątków takich jak `cdn.imir.agh.edu.pl`), pomija strony w językach obcych (np. prefiksy `/en`, `lang=en`) oraz respektuje wykluczenia ścieżek zdefiniowane w konfiguracji YAML.

## 4. Odszumianie DOM i Konwersja (Junk Removal)

Jakość systemu RAG zależy bezpośrednio od czystości wektorów. Pobranie strony wraz z jej bocznym menu czy stopką doprowadziłoby do drastycznego spadku precyzji wyszukiwania semantycznego (Context Poisoning).

* **Filtrowanie Strukturalne**: Biblioteka BeautifulSoup namierza główny kontener z treścią bazując na selektorach z pliku `sites.yaml` (np. `#mainContent`, `article`). Następnie bezlitośnie usuwane są tagi nawigacji (`nav`), stopki, ukryte skrypty, iframy, panele logowania, ciasteczka oraz karuzele zdjęć.
* **Optymalizacja Linków**: Linki względne (np. `/kontakt`) są w locie zamieniane na bezwzględne, aby zachowały użyteczność w wygenerowanym dokumencie Markdown.
* **Ekstrakcja Stopki**: Elementy takie jak adres wydziału, e-mail czy telefon dziekanatu powtarzają się na każdej podstronie. Crawler ekstrahuje te informacje ze stopki tylko z pierwszej odwiedzonej strony i zapisuje do pojedynczego pliku `_stopka.md`, co zapobiega zanieczyszczaniu każdego chunku w bazie wektorowej powtarzającymi się danymi kontaktowymi.
* **Transformacja do Markdown**: Oczyszczony obiekt DOM trafia do biblioteki `markdownify`, która generuje ustrukturyzowany tekst. Do każdego pliku dopisywany jest nagłówek Frontmatter zawierający kluczowe metadane (oryginalny URL, język, czas pobrania, wyekstrahowany tytuł), co warunkuje późniejsze poprawne generowanie cytowań w interfejsie użytkownika.

## 5. Odporność i Persystencja Stanu

Aby zapobiec utracie danych podczas wielogodzinnych operacji pobierania (np. w przypadku awarii sieci lub ręcznego przerwania skryptu), wprowadzono system ciągłego zapisu stanu:

* Zrzut środowiska (checkpoint) następuje co 25 pobranych stron. Stan kolejki, zbiór odwiedzonych linków oraz statystyki zapisywane są do pliku `stan.json`. Pozwala to na wznowienie crawlera w dowolnym momencie.
* **Deduplikacja treści**: Zdarza się, że pod różnymi adresami URL kryje się ta sama treść. Skrypt liczy skrót SHA-1 z tekstu każdego dokumentu; jeśli zawartość jest identyczna z już zapisaną, plik jest pomijany.
* **Delegacja zadań binarnych**: Crawler HTML nie pobiera plików PDF ani DOCX. Po natrafieniu na link binarny dodaje go do osobnego rejestru w pliku `pdf_linki.txt`. Lista ta służy jako punkt startowy dla kolejnego modułu pipeline'u: skryptu `scrape_documents.py`.

## 6. CLI i Generowanie Raportów

Skrypt oferuje przejrzysty interfejs linii poleceń. Flagą `--list` można wylistować wszystkie zdefiniowane źródła, a flagą `--max-pages` ograniczyć czas działania w celach testowych.

Na zakończenie każdego cyklu (lub po przechwyceniu sygnału przerwania klawiatury), crawler generuje w konsoli i logach wyczerpujący raport końcowy. Prezentuje on podsumowanie operacji, wyszczególniając:

* Liczbę odwiedzonych podstron oraz faktycznie zapisanych plików `.md`.
* Liczbę wyodrębnionych linków do plików PDF.
* Powody odrzucenia dokumentów (zbyt krótki tekst, duplikat treści, zablokowanie przez `robots.txt`, błąd pobierania).
* Stan kolejki oczekującej, uświadamiając operatora, ile pracy pozostało do zakończenia pełnego cyklu dla danego profilu.