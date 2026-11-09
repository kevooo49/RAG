# Dokumentacja Architektoniczna: Backend Aplikacji RAG (FastAPI)

## 1. Wprowadzenie i Wzorzec Architektoniczny

Backend systemu RAG został zrealizowany w oparciu o asynchroniczny framework FastAPI, co pozwala na wysoce wydajne przetwarzanie współbieżnych zapytań we/wy (takich jak odpytywanie zewnętrznych baz danych czy interfejsów API modeli językowych).

Architektura aplikacji opiera się na ścisłym wzorcu warstwowym, który gwarantuje separację odpowiedzialności:

* **Warstwa Transportowa (API/Routing)**: Przyjmuje żądania HTTP, deleguje zadania i zwraca ustrukturyzowane odpowiedzi lub strumienie danych.
* **Warstwa Logiki Biznesowej (Services)**: Hermetyzuje złożone operacje wyszukiwania, fuzji wyników oraz orkiestracji modeli AI, odseparowując kontrolery od technicznych szczegółów implementacji.
* **Warstwa Infrastruktury (Core)**: Zapewnia fundamentalne mechanizmy działania systemu, takie jak walidacja typów, zarządzanie konfiguracją środowiskową, logowanie zdarzeń oraz kontrola przepustowości.

Takie podejście sprawia, że kod jest wysoce modularny, łatwy w testowaniu i przygotowany na ewentualną wymianę poszczególnych komponentów (np. zmiana dostawcy bazy wektorowej) bez ingerencji w główny przepływ sterowania.

## 2. Warstwa Infrastruktury i Danych Wejściowych

Fundamentem stabilności backendu są moduły z katalogu głównego, które unifikują sposób, w jaki aplikacja komunikuje się ze środowiskiem operacyjnym i przetwarza dane wejściowe.

### Moduł `config.py`

Skrypt ten stanowi jedyne centralne miejsce zarządzania konfiguracją systemu. Odpowiada za bezpieczne wczytywanie i walidację zmiennych środowiskowych z pliku `.env`. Mapuje on krytyczne poświadczenia (klucze API OpenAI) oraz parametry połączeniowe (adresy hostów dla ChromaDB i Elasticsearch) na statyczne obiekty konfiguracyjne. Dzięki temu aplikacja przy starcie weryfikuje kompletność środowiska, zapobiegając niekontrolowanym awariom w późniejszych fazach działania (tzw. zasada *Fail-Fast*).

### Moduł `schemas.py`

Zamiast operować na luźnych słownikach, backend rygorystycznie waliduje każde żądanie przy użyciu biblioteki Pydantic. Plik `schemas.py` definiuje modele danych wejściowych i wyjściowych. Określa m.in. ścisłą strukturę zapytania użytkownika oraz format przesyłanej historii konwersacji. Modele te automatycznie odrzucają uszkodzone pakiety (np. brakujące pola, nieprawidłowe typy danych) jeszcze przed przekazaniem ich do warstwy serwisowej, chroniąc system przed atakami typu wstrzykiwania złośliwych tagów oraz oszczędzając niepotrzebne wywołania do płatnych API.

### Moduł `logging.py`

Zunifikowany system logowania zdarzeń zastępuje standardowe instrukcje drukowania w konsoli. Rejestruje on kluczowe etapy przepływu żądań, czasy wykonywania zapytań do baz danych oraz ewentualne wyjątki sieciowe. Wdrożenie ustrukturyzowanego logowania jest niezbędne z perspektywy utrzymania systemu na produkcji – pozwala na szybką diagnozę wąskich gardeł wydajnościowych (np. opóźnień wynikających z działania modelu krzyżowego) oraz monitorowanie stabilności połączeń z zewnętrznymi mikrousługami.

## 3. Warstwa Zabezpieczeń i Kontroli Ruchu

Udostępnienie interfejsu opartego o zaawansowane modele generatywne wiąże się z ryzykiem nadużyć oraz potencjalnie wysokimi kosztami operacyjnymi. Zabezpieczeniem na tym polu zajmuje się moduł limitujący.

### Moduł `limiter.py`

Moduł ten implementuje mechanizm Rate Limitingu, chroniący infrastrukturę uczelnianą oraz budżet API przed atakami typu DDoS, skryptami automatycznymi oraz zwykłym spamem. Działa on jako oprogramowanie pośredniczące, które zlicza żądania przychodzące z poszczególnych adresów IP.
W przypadku przekroczenia zdefiniowanego progu bezpiecznej przepustowości (np. 5 zapytań na minutę dla danego użytkownika), mechanizm ten odcina dostęp i natychmiastowo zwraca błąd HTTP 429 Too Many Requests. Frontend aplikacji jest zintegrowany z tym rozwiązaniem i w odpowiedzi na ten kod błędu wyświetla użytkownikowi stosowny komunikat o konieczności odczekania, zapobiegając eskalacji żądań i chroniąc stabilność serwera.

## 4. Warstwa Wyszukiwania i Pobierania Danych (`rag_service.py`)

Moduł `rag_service.py` stanowi serce operacyjne systemu RAG. Odpowiada za optymalizację zapytań, równoległą orkiestrację silników bazodanowych oraz ostateczną fuzję wyników. Jego architektura została zaprojektowana z naciskiem na minimalizację opóźnień oraz maksymalizację trafności kontekstu dostarczanego do głównego modelu generacyjnego. Posiadam ten kod w pamięci z naszej wcześniejszej analizy, co pozwala na precyzyjne odtworzenie jego logiki.

Kluczowe mechanizmy i decyzje projektowe:

* **Rozszerzanie i Przeformułowywanie Zapytań**: Zanim zapytanie trafi do baz danych, jest poddawane obróbce przez szybki, pomocniczy model językowy. Jeśli zapytanie nawiązuje do poprzednich wiadomości, algorytm buduje na podstawie historii konwersacji pytanie samodzielne. Następnie system wzbogaca je o oficjalne uczelniane słowa kluczowe, korzystając z wbudowanego słownika skrótów AGH (rozwiązującego żargon studencki, taki jak słowa "kolos", "sesja" czy "WF").
* **Zarządzanie Pamięcią Podręczną**: Aby zminimalizować koszty odpytywania API OpenAI i drastycznie obniżyć opóźnienia, wdrożono autorski mechanizm pamięci podręcznej z określonym czasem życia. Moduł zachowuje w pamięci serwera operacyjnego wygenerowane wcześniej rozszerzenia zapytań oraz wektory, serwując je natychmiastowo przy powtarzających się pytaniach użytkowników.
* **Asynchroniczne Wyszukiwanie Hybrydowe**: Skrypt eliminuje wąskie gardła wydajnościowe poprzez jednoczesne wywoływanie zapytań do dwóch niezależnych silników. Żądanie trafia równolegle do bazy wektorowej ChromaDB (odpowiadającej za dopasowanie semantyczne) oraz do indeksu Elasticsearch (odpowiadającego za precyzyjne dopasowanie leksykalne).
* **Fuzja Wyników RRF**: Otrzymane z obu baz danych listy kandydatów są łączone za pomocą matematycznego algorytmu oceny odwrotności rangi. Mechanizm ten przelicza pozycje fragmentów w obu rankingach, naturalnie promując te dokumenty, które zostały wysoko ocenione przez oba silniki, i zwraca ścisłą czołówkę najbardziej przydatnych fragmentów tekstu.
* **Progi Odcięcia i Tolerancja na Błędy**: Warstwa ta implementuje progi minimalnego podobieństwa wektorowego. Jeśli żaden fragment w bazie ChromaDB nie spełnia rygorystycznych kryteriów precyzji, wyniki z tego silnika są ignorowane, aby nie zanieczyścić kontekstu. Dodatkowo system jest odporny na awarię jednego z silników bazodanowych – potrafi płynnie przełączyć się na drugą, wciąż działającą bazę, logując to zdarzenie do celów diagnostycznych.

## 5. Warstwa Generacyjna i Syntetyzująca (`llm_service.py`)

Moduł `llm_service.py` odpowiada za ostateczną syntezę odpowiedzi i komunikację z chmurowymi modelami językowymi (API OpenAI). Jego głównym zadaniem jest nałożenie surowych, wyszukanych fragmentów wiedzy na intencję użytkownika przy zachowaniu rygorystycznych barier (tzw. guardrails) zapobiegających halucynacjom.

* **Budowa System Promptu**: Serwis dynamicznie konstruuje instrukcję systemową, wymuszającą na modelu rolę profesjonalnego asystenta uczelnianego. Prompt kategorycznie zabrania modelowi korzystania z wiedzy własnej i nakazuje bazowanie wyłącznie na dostarczonych fragmentach tekstów.
* **Wstrzykiwanie Kontekstu**: Pobrane przez `rag_service.py` chunki są formatowane do ujednoliconej postaci tekstowej (zawierającej metadane źródłowe) i doklejane do zapytania. Dzięki zachowaniu odpowiedniego formatowania (np. ścieżek nagłówków i list), model generatywny bez problemu interpretuje relacje przestrzenne i hierarchię regulaminów.
* **Zarządzanie Oknem Kontekstowym**: Moduł inteligentnie zarządza przesyłaną historią konwersacji. Aplikacja dba o to, aby rozmiar dodawanej historii czatu nie przekroczył ustalonego limitu tokenów, odcinając najstarsze wiadomości, co chroni przed błędami przepełnienia okna kontekstowego (Context Window Overlap) i minimalizuje koszty API.

## 6. Warstwa Transportowa i Routing (`chat.py`)

Zwieńczeniem całego systemu backendowego jest kontroler główny w pliku `chat.py`. Definiuje on endpointy API (np. `/api/chat`), łącząc wszystkie opisane wcześniej warstwy w jeden spójny pipeline komunikacyjny z interfejsem graficznym.

* **Orkiestracja Żądań**: Kontroler przyjmuje zapytanie wejściowe od użytkownika, przepuszcza je przez weryfikator `limiter.py`, zleca wyszukanie wiedzy do `rag_service.py`, a na koniec przekazuje cały pakiet do `llm_service.py`. Stanowi centralny punkt spinający logikę, obsługując również globalne przechwytywanie wyjątków (np. braki połączenia z bazą, timeouty API) i zwracanie ich jako odpowiednich kodów błędów HTTP.
* **Strumieniowanie Odpowiedzi (SSE)**: Najważniejszą decyzją architektoniczną w tym module jest implementacja protokołu Server-Sent Events. Zamiast blokować połączenie i czekać na wygenerowanie pełnej odpowiedzi przez model językowy, endpoint nawiązuje trwałe połączenie z frontendem i asynchronicznie przesyła do niego nowo wygenerowane tokeny w czasie rzeczywistym. Gwarantuje to płynne, natychmiastowe pojawianie się tekstu na ekranie użytkownika, drastycznie zmniejszając odczuwalne opóźnienia systemu. Wraz z pierwszymi pakietami SSE zwracana jest również lista źródeł wygenerowana z metadanych.

## 7. Punkt Wejścia i Inicjalizacja Aplikacji (`main.py`)

Moduł `main.py` pełni rolę głównego rdzenia (entry point), który spina wszystkie warstwy, kontrolery i serwisy w jedną działającą usługę sieciową. To tutaj framework FastAPI jest inicjalizowany i konfigurowany przed przyjęciem pierwszego żądania HTTP. Moduł celowo nie zawiera logiki biznesowej, działając jako dyspozytor i zarządca środowiska.

Kluczowe odpowiedzialności i mechanizmy:

* **Zarządzanie Cyklem Życia (Lifespan Events)**: Plik definiuje asynchroniczne operacje wykonywane podczas startu i zamykania serwera. Na etapie uruchamiania aplikacja nawiązuje trwałe połączenia z zewnętrznymi bazami danych (np. inicjalizacja klienta Elasticsearch i puli połączeń ChromaDB), co eliminuje narzut czasowy przy późniejszej obsłudze zapytań. Przy zamykaniu serwera mechanizm ten dba o bezpieczne zamknięcie sesji i zwolnienie zasobów w pamięci RAM.
* **Konfiguracja CORS (Cross-Origin Resource Sharing)**: Rejestruje polityki bezpieczeństwa określające, z jakich domen przeglądarki mogą wysyłać zapytania do API. Skonfigurowanie odpowiedniego oprogramowania pośredniczącego (CORS Middleware) jest krytyczne dla umożliwienia bezpiecznej komunikacji między odseparowanym frontendem a backendem, zapobiegając jedocześnie nieautoryzowanym odpytaniom z obcych witryn.
* **Rejestracja Routingów i Modułów**: Skrypt działa jako centralny rejestr. Importuje kontrolery z warstwy transportowej (takie jak router zdefiniowany w `chat.py`) i podłącza je do głównej instancji aplikacji za pomocą metody `include_router`. Pozwala to na eleganckie grupowanie ścieżek URL (np. z prefiksem `/api/v1/`).
* **Globalna Obsługa Błędów (Exception Handling)**: Zamiast pozwalać na niekontrolowane zamykanie aplikacji przy błędach, moduł instaluje globalne przechwytywacze wyjątków. Gwarantują one, że nawet nieprzewidziany błąd w głębokiej warstwie serwisowej (np. zerwanie połączenia z API OpenAI) zostanie pochwycony, poprawnie zalogowany za pomocą modułu `logging.py`, a do frontendu powróci ustandaryzowana odpowiedź JSON z czytelnym komunikatem i bezpiecznym kodem błędu (np. 500 Internal Server Error).