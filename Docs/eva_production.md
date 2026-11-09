# Dokumentacja Architektoniczna: Ewaluacja Ostatecznego Stosu Produkcyjnego (`evaluate_production.py`)

## 1. Wprowadzenie i Rola Modułu

Skrypt `evaluate_production.py` stanowi ostateczny punkt weryfikacji całego rurociągu przetwarzania danych w systemie RAG. W przeciwieństwie do poprzednich modułów analitycznych, które testowały izolowane komponenty w celu podjęcia decyzji projektowych, ten skrypt ewaluuje zintegrowany, finalny stos technologiczny. Sprawdza on, jak wszystkie wybrane wcześniej optymalizacje współpracują ze sobą w jednym, spójnym środowisku, dostarczając ostateczne metryki gotowości systemu do wdrożenia na produkcję.

## 2. Architektura Docelowa (Stos Produkcyjny)

Moduł implementuje i testuje następujący zestaw technologii, uznany za optymalny na podstawie wcześniejszych badań:

* **Model Osadzający**: `text-embedding-3-small` – wybrany ze względu na optymalny stosunek kosztów do wydajności.
* **Strategia Chunkowania**: Asymetryczna produkcja (`Production_Asymmetric`) – podział strukturalny zachowujący spójność tabel i list, minimalizujący jednocześnie nadmierną fragmentację.
* **Wyszukiwarka**: Fuzja algorytmu wektorowego (ChromaDB) oraz wyszukiwania leksykalnego (Elasticsearch), łączona za pomocą algorytmu RRF ze wskaźnikiem stałym równym 60.
* **Warstwa Lingwistyczna**: Algorytm rozszerzania zapytań obsługiwany przez szybki model `gpt-4o-mini`, weryfikujący i transformujący zapytanie użytkownika przed wysłaniem go do bazy danych.



## 3. Asymetryczne Zarządzanie Kontekstem

Skrypt wdraża kluczową innowację architektoniczną w postaci asymetrycznego generowania treści dla różnych wyszukiwarek, rozwiązaną w funkcji `chunk_file_asymmetric`.
Ponieważ bazy wektorowe są wysoce wrażliwe na powtarzający się szum, a bazy leksykalne korzystają na maksymalnym nasyceniu słowami kluczowymi, pojedynczy chunk jest przed wdrożeniem rozwidlany na dwie wersje:

* Wersja dla bazy wektorowej (ChromaDB) otrzymuje skrócony, sterylny nagłówek w formacie ścieżki nawigacyjnej (Tytuł › Sekcja), dołączony bezpośrednio nad właściwą treścią.
* Wersja dla indeksu leksykalnego (Elasticsearch) zachowuje pełny, techniczny format nagłówka generowany przez system chunkowania, maksymalizując szansę na twarde dopasowanie słów kluczowych.



## 4. Wyniki Ewaluacji (Precyzja i Opóźnienia)

Uruchomienie skryptu dla 90 zapytań testowych (z próbnikiem ustawionym na 2000 iteracji klastrowego bootstrapu) potwierdziło wysoką stabilność i trafność zintegrowanego systemu.

* **Precyzja Wyszukiwania**: Ostateczna architektura osiągnęła metrykę MRR@10 na poziomie 0.752. Wskaźnik Hit@1 wyniósł 0.667, co oznacza, że w 2/3 przypadków idealny dokument znajduje się na absolutnie pierwszym miejscu. Co więcej, w 85.6% przypadków oczekiwany plik trafia do czołowej piątki wyników (Hit@5), co gwarantuje pełne nasycenie okna kontekstowego dla ostatecznego modelu generacyjnego.
* **Opóźnienia Systemowe (Wydajność)**: Zapytanie do połączonych baz danych (wektorowej i leksykalnej) oraz sprzężenie ich algorytmem RRF zajmuje zaledwie ułamek sekundy (mediana 278.7 ms, 95. percentyl 390.7 ms). Główne obciążenie czasowe generuje weryfikacja i rozszerzanie zapytania przez model językowy, trwająca średnio 1209.1 ms. Sumaryczny narzut nie przekracza jednak bariery 1.5 sekundy, co gwarantuje doskonałe doświadczenie użytkownika.



## 5. Rozbicie Semantyczne i Analiza Błędów

Szczegółowa metryka potwierdza skuteczność obsługi różnych intencji użytkownika. System dobrze radzi sobie z oficjalnymi terminami w trybie leksykalnym (Hit@1: 0.77). Dobrze interpretowane są również zapytania potoczne, w których studenci używają slangu uczelnianego (Hit@1: 0.67). Zgodnie z przewidywaniami, najtrudniejsze pozostają silnie sparafrazowane zapytania semantyczne (Hit@1: 0.57).

Raport wyszczególnia również skrajne przypadki błędów (zapytania poza Top 5). Dogłębna analiza tych przypadków udowadnia granice możliwości systemu wektorowego. Przykładowo, system zapytany potocznie o to, co trzeba zaliczyć na kolokwium z konkretnego przedmiotu, zwraca sylabusy z języków obcych oraz uchwały o warunkach rekrutacji, myląc luźny żargon z ogólnymi regulacjami uczelni. Zapytania te stanowią unikalny materiał do późniejszego dostrajania (fine-tuningu) własnych modeli embeddingowych.