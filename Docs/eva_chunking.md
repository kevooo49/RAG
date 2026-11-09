# Dokumentacja Architektoniczna: Ewaluacja Strategii Chunkowania (`evaluate_chunking.py`)

## 1. Wprowadzenie i Rola Modułu

Skrypt `evaluate_chunking.py` jest środowiskiem analitycznym przeznaczonym do empirycznego badania i optymalizacji strategii segmentacji tekstu (tzw. chunkowania) dla bazy wektorowej. Jakość podziału tekstu bezpośrednio determinuje zdolność modelu osadzającego (embeddingowego) do odnajdywania relewantnych fragmentów. Moduł izoluje warstwę wyszukiwania wektorowego od silnika hybrydowego, analizując zachowanie algorytmów na wyizolowanej próbce Ground Truth, dokonując precyzyjnego pomiaru zysków i strat (względem tzw. wariantu bazowego) przy wykorzystaniu rygorystycznych testów statystycznych.

## 2. Metodologia Badawcza i Architektura Ewaluacji

Aby zapewnić miarodajność eksperymentów i wyeliminować szum (np. różnice w przybliżonym wyszukiwaniu k-NN), skrypt implementuje następujące mechanizmy:

* **Wyszukiwanie "Brute-force" (Exact Cosine)**: Zamiast polegać na algorytmie indeksowania w ChromaDB (który narzuca sztuczne limity np. `FETCH_POOL`), skrypt oblicza dokładne podobieństwo kosinusowe na surowych macierzach `numpy` dla absolutnie wszystkich chunków w korpusie ewaluacyjnym. Gwarantuje to poprawny odczyt metryki `Hit@10`, faworyzując jakość samego chunku, a nie parametry heurystyki bazy danych.
* **Separacja Zmiennych Kontekstowych (Ablation Study)**: Skrypt dekomponuje strategie produkcyjne, oddzielając od siebie efekt rozmiaru chunku od efektu wzbogacania o metadane (tzw. chlebek kontekstowy / nagłówek). Utworzono w tym celu warianty sterylne, np. `Fixed_Matched` (surowy podział) w kontrze do `Fixed_Matched+Title` (podział wzbogacony nagłówkiem), w celu jednoznacznego wyizolowania zysku z tzw. wstrzykiwania kontekstu.
* **Persistent Cache (SQLite)**: Aby operacje na tysiącach wygenerowanych chunków nie obciążały API OpenAI i nie generowały zbędnych kosztów, skrypt wyposażono w trwałą pamięć podręczną na dysku (`embeddings.sqlite`). Każdy unikalny ciąg tekstowy powiązany z danym modelem zostaje zapisany w formie skompresowanego binarnego blobu `BLOB (np.float32)`, co sprawia, że ponowne ewaluacje kosztują 0,00 USD.
* **Wielopoziomowe Metryki (Plik vs Chunk)**: Skrypt odróżnia odnalezienie odpowiedniego pliku od jakości dostarczonego kontekstu do okna LLM. Oprócz standardowego `Hit@1`, `MRR@10` i `nDCG@10` na poziomie plików, liczona jest *Precyzja Chunków* (`CPrec@5`). Weryfikuje ona, jaki ułamek (np. 3/5) wyłuskanych przez wyszukiwarkę chunków faktycznie należy do pliku oczekiwanego przez zbiór ewaluacyjny (zapobiega to premiowaniu strategii zalewającej "top 5" bezwartościowymi duplikatami z jednego dokumentu).
* **Cluster Bootstrap Sampling**: Testowanie różnic używa próbkowania (bootstrapingu), przy czym uwzględniono tzw. gęstość zapytań (zapytania z jednego dokumentu są skorelowane). Próbkowanie dotyczy klastrów (całych plików referencyjnych), a nie pojedynczych zapytań, gwarantując surowe, bezpieczne 95% przedziały ufności dla miary `dMRR@10` (delta Mean Reciprocal Rank).

## 3. Analiza Wyników Ewaluacji Strategii (Raport 90 Zapytań)

Uruchomienie skryptu dla zbalansowanego zbioru ewaluacyjnego wykazało kluczowe dysproporcje pomiędzy strategiami dzielenia tekstu.

Zmierzone strategie i ich wyniki `MRR@10`:

* Strategie sztywne:
* `Fixed_500`: 0.756
* `Fixed_1000`: 0.674
* `Fixed_2000`: 0.615


* Strategie bazujące na akapitach:
* `Paragraphs_2500`: 0.679


* Strategie z wstrzyknięciem tytułu (Ablacja):
* `Fixed_Matched`: 0.648 vs `Fixed_Matched+Title`: 0.706
* `Paragraphs_Matched`: 0.670 vs `Paragraphs_Matched+Title`: 0.734


* Strategie produkcyjne (strukturalne):
* `Production_Advanced`: 0.701
* `Production_Asymmetric`: 0.730





### Główne Wnioski Architektoniczne:

1. **Przewaga Małych Chunków nad Dużymi**: Podział `Fixed_500` wygenerował najlepszy wynik `MRR@10` (0.756) oraz `Hit@1` (0.678) spośród wszystkich badanych wariantów surowych. Dowodzi to zjawisku rozmywania się semantyki – w chunkach o wielkości rzędu 2000 znaków, precyzyjne dane (np. warunki uzyskania stypendium) giną w gąszczu wektorów niepowiązanych z treścią pytania (`Fixed_2000` zaliczyło wyraźny spadek `MRR` do 0.615). Różnica statystyczna między modelem produkcyjnym a modelem wielkoobjętościowym (`Fixed_2000`) wyniosła `p=0.034` (znacząca statystycznie utrata jakości na szkodę wariantu `Fixed_2000`).
2. **Krytyczna Wartość Wstrzykiwania Tytułu**: Porównanie wariantów izolowanych (np. `Paragraphs_Matched` bez tytułu: MRR 0.670 vs z tytułem: 0.734) bezsprzecznie dowodzi skuteczności wstrzykiwania tzw. chlebka (breadcrumb) na początku chunku. Jest to dowód na to, że kontekst metadanych jest ważniejszy niż sam bazowy rozmiar fragmentu.
3. **Sukces Hybrydowej Produkcji (`Production_Asymmetric`)**: Wdrożony w systemie asymetryczny wariant produkcyjny okazał się optymalny (MRR 0.730). Co prawda `Fixed_500` wygenerowało statystycznie minimalnie wyższy `Hit@1` (o 0.100), ale kosztem potężnej fragmentaryzacji bazy danych (wygenerowanie 4837 chunków – średnio 25,5 chunku na plik, co oznacza trzykrotnie wyższe koszty środowiskowe i przeciążenie LLM powielającymi się urywkami zdań).
4. **Odporność na typy zapytań**: Rozbicie wyników pokazuje, że algorytmy wektorowe radzą sobie doskonale z zapytaniami "lexical" (`Hit@1` rzędu 0.83 dla asymetrycznej produkcji) i potocznymi "colloquial" (ok. 0.60), lecz notują załamanie przy zapytaniach silnie sparafrazowanych "semantic" (Hit@1 rzędu 0.43). Definiuje to potrzebę wdrożenia algorytmów takich jak RRF oraz Query Expansion w głównej wyszukiwarce.