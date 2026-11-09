# Dokumentacja Architektoniczna: Pobieranie Dokumentów Binarnych (`scrape_documents.py`)

## 1. Wprowadzenie i Rola w Systemie

Skrypt `scrape_documents.py` to moduł odpowiedzialny za masowe i bezpieczne pobieranie plików binarnych (takich jak PDF, DOCX, XLSX) ze stron internetowych uczelni. Został zaprojektowany jako drugi etap potoku zasilania danych (pipeline'u) – uruchamia się go dopiero po zakończeniu działania crawlera HTML (`scrape_html.py`). Moduł ten nie crawluje sieci ponownie; opiera się na wygenerowanym wcześniej rejestrze `pdf_linki.txt`, co znacząco redukuje obciążenie uczelnianych serwerów i optymalizuje czas wykonania.

## 2. Pamięć Stanu i Zarządzanie Rejestrem

W celu zapewnienia odporności na przerwania (np. utratę połączenia internetowego) i eliminacji powtarzalnych operacji, skrypt implementuje mechanizm logowania strukturalnego (JSONL):

* **Manifest (`documents_manifest.jsonl`)**: Każda próba pobrania pliku jest atomowo logowana do pliku JSONL. Manifest ten pełni funkcję bazy danych dla operacji pobierania. Rejestruje on nie tylko udane pobrania, ale również błędy sieciowe (np. HTTP 404, 410) czy odrzucenia (np. błędna sygnatura pliku).
* **Wznawianie (Resuming)**: Przed podjęciem próby pobrania, algorytm weryfikuje historię w manifeście. Pliki o statusach uznanych za finalne (np. `ok`, `duplicate`, `missing`, `rejected`) są całkowicie pomijane. Pobierane są wyłącznie nowe linki oraz te, które wcześniej zakończyły się tymczasowym błędem sieciowym.
* **Flaga `--reset**`: Przeznaczona dla wymuszenia ponownego pobrania wszystkich plików z listy, ignorując wygenerowany dotychczas manifest.

## 3. Bezpieczeństwo i Weryfikacja Plików

Pobieranie plików binarnych niesie za sobą ryzyko zablokowania pamięci RAM serwera lub zapisania danych uszkodzonych. Moduł wdraża rygorystyczne warstwy obronne:

* **Strumieniowanie Pobierania (Streaming)**: Skrypt nie ładuje plików w całości do pamięci RAM. Korzysta z `iter_content` w pakietach po 64 KB, co pozwala na bezpieczne przetwarzanie plików niezależnie od ich rozmiaru.
* **Obrona przed atakami (Bomba dekompresyjna/OOM)**: Ustawiono sztywny limit wielkości pobieranego pliku (`MAX_SIZE = 50 MB`). Pliki przekraczające tę wartość są natychmiast odrzucane z logiem błędu.
* **Weryfikacja Sygnatur (Magic Numbers)**: Ze względu na częste błędy systemów CMS, adres URL z rozszerzeniem `.pdf` może w rzeczywistości serwować stronę błędu w HTML. Zamiast ufać nagłówkom HTTP, system pobiera pierwszy 1 KB pliku i analizuje jego fizyczną sygnaturę (np. sprawdzając, czy plik zaczyna się od bajtów `b"%PDF"` dla PDF lub `b"PK\x03\x04"` dla dokumentów Office).
* **Zapis Atomowy**: Pliki są pobierane z dopiskiem `.part`. Dopiero po pomyślnej weryfikacji i zakończeniu strumieniowania, system operacyjny wykonuje atomową zmianę nazwy (`os.replace`). Zapobiega to przerwaniu działania w połowie zapisu i pozostawieniu uszkodzonego pliku na dysku.

## 4. Deduplikacja i Ujednolicanie Nazw

W środowiskach akademickich te same zarządzenia czy druki (np. "wniosek o urlop") często występują pod różnymi adresami URL lub ukrywają się pod nieczytelnymi nazwami typu `download.php?id=123`.

* **Tworzenie Nazwy Pliku (Slugify)**: Algorytm odkodowuje parametry URL (`unquote`), wydobywa ostatni logiczny człon ścieżki i poddaje go operacji slugify, zamieniając znaki specjalne na podkreślenia.
* **Haszowanie Zabezpieczające**: Ponieważ nazwy plików mogłyby być identyczne ("zalacznik_1.pdf"), na końcu nazwy każdego dokumentu doklejany jest 8-znakowy skrót SHA-1 wygenerowany z oryginalnego adresu URL.
* **Deduplikacja Zawartości (SHA-256)**: W locie (podczas strumieniowania chunków po 64 KB) generowany jest również hash SHA-256 całej zawartości pliku. Jeśli po pobraniu okaże się, że identyczny dokument został już wcześniej zapisany na dysku przez inny URL, nowo pobrana kopia jest natychmiast usuwana (`os.remove`), a w manifeście oznaczana jako `duplicate` ze wskazaniem referencji do oryginału. Oszczędza to pamięć dyskową, czas późniejszego przetwarzania i miejsce w przestrzeni wektorowej.

## 5. Przestrzeganie Etykiety Sieciowej

Moduł w pełni szanuje reguły serwerów zewnętrznych:

* Używa instancji parsera `RobotFileParser` z wbudowanym mechanizmem buforowania (per domena), by respektować blokady zdefiniowane w pliku `robots.txt`.
* Posiada wbudowany opóźniacz zapytań (Throttle), domyślnie zdefiniowany na `0.5` sekundy między kolejnymi plikami lub ustawiany dynamicznie, jeśli serwer zadeklarował parametr `Crawl-delay`.

## 6. Generowanie Raportu Diagnostycznego

Podobnie jak narzędzia HTML, downloader implementuje statystyki użycia. Raport generowany po zakończeniu pobierania prezentuje jasny podział na:

* Sukcesy pobierania,
* Oszczędność wynikającą z deduplikacji (`duplicate`),
* Braki w systemie (`missing` - HTTP 404),
* Odrzucenia z uwagi na politykę wielkości lub bezpieczeństwa (`rejected`),
* Żądania zablokowane przez administratora serwisu (`robots`).