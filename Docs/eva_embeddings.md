# Dokumentacja Architektoniczna: Ewaluacja Modeli Osadzających (`evaluate_embeddings.py`)

## 1. Cel i Środowisko Badawcze

Skrypt `evaluate_embeddings.py` to zautomatyzowane środowisko testowe służące do benchmarkingu modeli wektorowych w systemie RAG. Moduł całkowicie izoluje warstwę wyszukiwania wektorowego (pomijając np. Elasticsearch), aby precyzyjnie zmierzyć trzy krytyczne parametry: precyzję wyszukiwania, latencję (czas zapytania) oraz narzut czasowy na indeksowanie bazy.

Zastosowano w nim zaawansowane mechanizmy optymalizacyjne:

* **Odcisk palca korpusu (Fingerprinting):** Autorski mechanizm oblicza skrót SHA-256 dla wszystkich plików w zbiorze referencyjnym, kodu chunkera oraz nazwy weryfikowanego modelu. Jeśli korpus i kod nie uległy zmianie, skrypt ładuje wektory z lokalnej pamięci podręcznej (ChromaDB Persistent Client), co pozwala uniknąć przepalania budżetu API i powtarzania wielogodzinnych obliczeń lokalnych.


* **Izolacja przez Mockowanie:** Moduł używa systemowej biblioteki `unittest.mock` do wirtualizacji połączeń z Elasticsearch w skrypcie `build_database.py`. Pozwala to mierzyć czystą wydajność bazy wektorowej bez ryzyka zakłóceń wywołanych narzutem sieciowym lub timeoutami innych serwisów.


* **Statystyka Bootstrapowa:** Podobnie jak w teście chunkowania, skrypt korzysta z techniki Cluster Bootstrap (2000 iteracji ze zwracaniem), grupując zapytania względem plików referencyjnych w celu wygenerowania twardych, 95-procentowych przedziałów ufności dla miar `Hit@k` oraz `MRR@10`.



## 2. Profilowanie Modeli (Jakość vs Koszt)

Ewaluacji poddano cztery wiodące modele: dwa komercyjne udostępniane przez API (OpenAI) oraz dwa otwarte modele lokalne. Wyniki testów na 90 zróżnicowanych zapytaniach przy `K=10` jednoznacznie zdefiniowały ich możliwości:

* **OAI-Large (`text-embedding-3-large`):** Zdecydowany lider jakościowy z `MRR@10` na poziomie 0.820 i najwyższym współczynnikiem `Hit@1` (0.711). Czas wektoryzacji korpusu, wspierany infrastrukturą OpenAI, wyniósł zaledwie 38,7 sekundy, przy w pełni akceptowalnej latencji pojedynczych zapytań (220 ms).


* **BGE-m3 (`BAAI/bge-m3`):** Ciężki model open-source, który osiągnął wysoką precyzję (`MRR@10`: 0.757), jednak jego uruchomienie bez potężnej akceleracji GPU spowodowało drastyczny czas budowy indeksu trwający ponad 81 minut (4903 sekundy) i relatywnie powolne odpytywanie bazy (415 ms).


* **PL-Retriever (`ipipan/silver-retriever-base-v1`):** Specjalistyczny, lokalny model przetrenowany pod język polski. Wykazał bardzo solidną celność (`MRR@10`: 0.748), z rozsądniejszym czasem budowy indeksu (około 20 minut) i świetnym czasem zapytania na poziomie 235 ms.


* **OAI-Small (`text-embedding-3-small`):** Budżetowy model bazowy, który zamknął stawkę z najniższym wynikiem `MRR@10` (0.662). Wyróżnił się natomiast bezkonkurencyjnym czasem indeksowania całego zbioru (26,3 sekundy) i najniższym opóźnieniem w wyszukiwaniu (186 ms).



## 3. Luki Semantyczne w Wyszukiwaniu

Rozbicie twardych wyników na poszczególne typy zapytań zidentyfikowało krytyczną słabość budżetowego modelu OAI-Small w kategorii zapytań "semantic" (mocno sparafrazowanych, pozbawionych dokładnych słów kluczowych). Zauważono tu gigantyczną przepaść:

* OAI-Small dla abstrakcyjnych zapytań osiąga `Hit@1` na poziomie zaledwie 0.30.


* OAI-Large radzi sobie z takimi strukturami bez większego problemu, notując `Hit@1` równe 0.70.


* Modele lokalne (BGE-m3 i PL-Retriever) utrzymały stabilny kompromis (w obu przypadkach `Hit@1` równe 0.50).



## 4. Wnioski Architektoniczne

Pomimo ewidentnej, wręcz miażdżącej przewagi technicznej wariantu `text-embedding-3-large`, to model `text-embedding-3-small` pozostaje z racjonalnych powodów docelowym wyborem dla środowiska produkcyjnego. Wynika to wprost ze skali systemu – model Large charakteryzuje się ogromnym kosztem tokenów, co przy potężnej, żywej i aktualizowanej bazie danych uczelni AGH niepotrzebnie windowałoby budżet.

Udowodniona empirycznie "luka semantyczna" mniejszego modelu wektorowego nie przekreśla projektu, ponieważ zostaje kompensowana na późniejszych etapach RAG. Moduł korzysta z mechanizmu wstrzykiwania hybrydowego (Reciprocal Rank Fusion - RRF) wspieranego przez silnik tekstowy Elasticsearch, a także posiłkuje się Query Expansion (przepisywaniem kontekstowym przez LLM), dzięki czemu jakość wyszukiwania na samej górze potoku zostaje wyrównana do poziomu droższych systemów.