# Dokumentacja Architektoniczna: Środowisko Kontenerowe (Docker)

## 1. Wprowadzenie i Architektura Środowiska

Aplikacja oraz jej zależności infrastrukturalne zostały w pełni skonteneryzowane przy użyciu platformy Docker, a ich orkiestracją zarządza narzędzie Docker Compose. Środowisko zostało podzielone na cztery odseparowane mikrousługi (wyszukiwarka wektorowa, wyszukiwarka leksykalna, backend, frontend), co gwarantuje pełną powtarzalność wdrożenia na dowolnym serwerze, izolację procesów oraz spójność między środowiskiem deweloperskim a produkcyjnym.

## 2. Warstwa Baz Danych

Infrastruktura bazodanowa opiera się na dwóch niezależnych silnikach, z których każdy posiada utrwalony wolumen danych, chroniący przed utratą indeksów po restarcie kontenerów.

### Elasticsearch (Wyszukiwarka Leksykalna)

Standardowy obraz Elasticsearch nie posiada wbudowanej obsługi zaawansowanej fleksji języka polskiego. Z tego powodu zaimplementowano dedykowany plik `Dockerfile_ES`, który rozszerza bazowy obraz w wersji `8.15.2` o instalację oficjalnego modułu `analysis-stempel`.

* **Zarządzanie Zapasami (JVM)**: Aby zapobiec niestabilności i nadmiernemu zużyciu pamięci hosta, kontenerowi narzucono sztywny limit sterty maszyny wirtualnej Javy (`ES_JAVA_OPTS=-Xms2g -Xmx2g`), rezerwując dokładnie 2 GB pamięci RAM.
* **Bezpieczeństwo i Architektura**: Usługa działa w trybie jednowęzłowym (`single-node`) z wyłączonym wbudowanym systemem autoryzacji `xpack.security.enabled=false`. Aby zrekompensować brak autoryzacji bazodanowej, port 9200 został przypisany ściśle do interfejsu pętli zwrotnej serwera (`127.0.0.1:9200`), blokując jakikolwiek dostęp z zewnątrz sieci hosta.
* **Sondy Stanu (Healthcheck)**: Wdrożono rygorystyczne sprawdzanie dostępności klastra poprzez odpytywanie głównego endpointu. Pozostałe usługi nie zostaną uruchomione, dopóki Elasticsearch nie zwróci poprawnego statusu.

### ChromaDB (Baza Wektorowa)

Usługa korzysta z lekkiego, natywnego obrazu `chromadb/chroma:1.5.9`. Została skonfigurowana do pracy z utrwalaniem danych na wolumenie (`IS_PERSISTENT=TRUE`, katalog `/data`) oraz wyłączono w niej domyślną telemetrię. Podobnie jak w przypadku indeksu leksykalnego, jej port (8001) wyeksponowano wyłącznie na interfejsie lokalnym hosta.

## 3. Warstwa Logiki Biznesowej (Backend)

Serwer FastAPI został zamknięty w lekkim obrazie bazującym na dystrybucji `python:3.11-slim`.

* **Zależności Systemowe**: Oprócz bibliotek z pliku `requirements.txt`, instalowane są pakiety `tzdata` (wraz ze zmienną wymuszającą strefę czasową `Europe/Warsaw`) oraz `curl` wykorzystywany do weryfikacji stanu.
* **Integracja z Modelami Lokalnymi**: Kontener posiada zdefiniowaną regułę rutingu wewnętrznego (`extra_hosts: "host.docker.internal:host-gateway"`), co pozwala backendowi na komunikację z modelem Ollama uruchomionym bezpośrednio na maszynie hosta (poza siecią Dockera), unikając problemów z izolacją sieciową.
* **Serwer Aplikacyjny**: Aplikacja uruchamiana jest przez serwer Uvicorn z włączoną flagą `--proxy-headers`. Jest to kluczowe ustawienie, które pozwala backendowi poprawnie odczytywać oryginalne adresy IP użytkowników przekazywane przez serwer Nginx, co stanowi fundament działania modułu limitującego zapytania (Rate Limiter).

## 4. Warstwa Prezentacji (Frontend i Odwrotne Proxy)

Interfejs użytkownika dystrybuowany jest poprzez wysoce zoptymalizowany, wieloetapowy proces budowania obrazu (Multi-stage build).

* **Etap Budowania (Builder)**: Wykorzystuje obraz `node:22-alpine` do instalacji zależności (`npm ci`) i kompilacji plików TypeScript/Vite do statycznej paczki wynikowej (`dist`).
* **Etap Dystrybucji (Nginx)**: Skompilowane pliki są kopiowane do minimalistycznego obrazu serwera `nginx:alpine`. Taki zabieg zmniejsza wagę końcowego obrazu o kilkaset megabajtów (brak środowiska Node.js na produkcji).
* **Odwrotne Proxy (Reverse Proxy) i Strumieniowanie**: Skonfigurowany na nowo plik `default.conf` serwera Nginx rozwiązuje dwa kluczowe problemy architektoniczne:
1. **SPA Routing**: Dyrektywa `try_files` przekierowuje wszystkie nieodnalezione ścieżki do pliku `index.html`, co jest niezbędne do poprawnego działania aplikacji jednowymiarowych (Single Page Application).
2. **Eliminacja CORS i SSE**: Serwer przechwytuje żądania kierowane na ścieżkę `/api/` i bezszelestnie przekazuje je do kontenera `backend:8000`. Aby strumieniowanie tokenów z modelu LLM (Server-Sent Events) działało poprawnie, w konfiguracji proxy celowo i całkowicie wyłączono buforowanie oraz pamięć podręczną (`proxy_buffering off`, `proxy_cache off`), a połączenie utrzymywane jest w trybie `keep-alive`.



## 5. Orkiestracja i Cykl Życia (Docker Compose)

Plik kompozycyjny definiuje ścisłą zależność topologiczną (kolejność uruchamiania). Backend nie podejmie próby startu, dopóki sonda stanu (Healthcheck) kontenera Elasticsearch nie zgłosi pełnej gotowości. Z kolei frontend uruchamia się dopiero po zweryfikowaniu dostępności interfejsu API backendu. Zapobiega to lawinowym błędom wynikającym z prób łączenia się z niedostępnymi jeszcze usługami (tzw. Race Conditions). Całość została objęta polityką `restart: unless-stopped`, gwarantującą automatyczne podniesienie systemu po awarii lub ponownym uruchomieniu serwera fizycznego.