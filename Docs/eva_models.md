# Dokumentacja Architektoniczna: Ewaluacja Modeli LLM (`evaluate_models.py`)

## 1. Wprowadzenie i Rola Modułu

Skrypt `evaluate_models.py` odpowiada za rygorystyczną weryfikację końcowego etapu działania systemu RAG, czyli fazy generacji odpowiedzi (Generation). Jego głównym zadaniem jest ocena zdolności modeli językowych do syntetyzowania spójnych, precyzyjnych i bezbłędnych wypowiedzi na podstawie dostarczonych fragmentów dokumentów. Moduł celowo izoluje warstwę generacyjną od wyszukiwarki, dostarczając modelom z góry ustalony, idealny kontekst z bazy ewaluacyjnej, co pozwala obiektywnie zmierzyć skłonność modeli do halucynacji w kontrolowanych warunkach.

## 2. Metodologia Badawcza i Model Sędziowski

Zamiast polegać na tradycyjnych i często zawodnych metrykach opartych na nakładaniu się tekstu, zaimplementowano nowoczesne podejście ewaluacyjne z wykorzystaniem potężniejszego modelu językowego w roli arbitra. Skrypt wykorzystuje model sędziowski z wymuszonym, strukturalnym formatowaniem wyjścia, który bezstronnie ocenia każdą wygenerowaną odpowiedź w skali od 1 do 5 w trzech wymiarach:

* Wierność: ocena, czy odpowiedź opiera się wyłącznie na dostarczonym dokumencie (od 1 dla jawnego zmyślania informacji do 5 dla idealnej zgodności z tekstem źródłowym).
* Trafność: weryfikacja, czy wygenerowany tekst bezpośrednio i precyzyjnie odpowiada na zadane przez studenta pytanie.
* Jakość języka: ocena płynności, stylu i poprawności użycia języka polskiego.



## 3. Architektura Środowiska Testowego

Aby środowisko analityczne precyzyjnie odzwierciedlało zagrożenia i obciążenia środowiska produkcyjnego, wdrożono zaawansowane mechanizmy inżynieryjne:

* **Wstrzykiwanie Szumu**: Nawet najlepszy system RAG z bazy wektorowej czasami wyłuskuje nieistotne dokumenty. Aby zbadać odporność testowanych modeli na zmyłki, skrypt z prawdopodobieństwem 50% celowo wstrzykuje do oryginalnego kontekstu wycinek zupełnie niezwiązanego dokumentu jako szum informacyjny.
* **Dwustopniowa Baza Buforująca**: Moduł operuje na dwóch oddzielnych tabelach w pliku pamięci podręcznej SQLite. Tabela generacji bezpiecznie loguje wyprodukowane przez model odpowiedzi oraz ich latencję, natomiast tabela ewaluacji przechowuje wystawione przez sędziego oceny i uzasadnienia. Eliminuje to konieczność ponownego i kosztownego odpytywania modeli w przypadku przerwania testów lub potrzeby reewaluacji danych.
* **Rygor Statystyczny**: Zebrane wyniki poddawane są próbkowaniu metodą klastrowego bootstrapu (2000 iteracji przy zablokowanym ziarnie losowości), co generuje precyzyjne 95% przedziały ufności oraz umożliwia weryfikację istotności statystycznej względem modelu bazowego.



## 4. Analiza Raportu Skuteczności Modeli

Testom poddano dwa wydajne modele chmurowe (gpt-4o-mini jako wariant bazowy oraz gpt-4o jako wariant premium) oraz otwarty, lokalny model dedykowany dla języka polskiego (bielik-11b). Analiza 90 zapytań ujawniła kluczowe różnice w ich architekturze decyzyjnej:

* **Jakość Generacji i Wierność**: Zarówno model bazowy gpt-4o-mini (Wierność: 4.49, Trafność: 4.62), jak i potężniejszy gpt-4o (Wierność: 4.57, Trafność: 4.61) wygenerowały odpowiedzi o niemal identycznej, bardzo wysokiej precyzji. Eksperyment z użyciem lokalnego modelu bielik-11b zakończył się niepowodzeniem – model uzyskał krytycznie niskie noty za wierność (3.18) oraz trafność (3.89), wykazując ogromną tendencję do halucynowania. Różnice między modelem lokalnym a bazowym są wysoce istotne statystycznie i go dyskwalifikują.
* **Odporność na Szum Informacyjny**: Wstrzyknięcie fałszywego kontekstu obnażyło minimalną podatność modelu gpt-4o-mini, którego wierność spadła w takich warunkach z 4.61 do 4.34. W tym samym teście model gpt-4o udowodnił wybitną stabilność poznawczą, notując marginalny spadek wierności o zaledwie 0.06 punktu i bezbłędnie ignorując wstrzyknięte zmyłki.
* **Opóźnienia (Latencja)**: Oba modele chmurowe zapewniły błyskawiczny czas generacji na niemal identycznym poziomie 1.4 sekundy. Lokalny model bielik-11b, pozbawiony wsparcia potężnych klastrów obliczeniowych, zablokował środowisko, wymagając średnio niemal 97 sekund na udzielenie pojedynczej odpowiedzi.



## 5. Konkluzje i Wybór Architektoniczny

Powyższy raport dostarcza twardych dowodów na to, że do obsługi ostatecznej fazy generacji w systemie RAG na środowisku produkcyjnym należy zaimplementować model **gpt-4o-mini**.

Mimo że pełny wariant gpt-4o wykazuje się imponującą odpornością na zaszumiony kontekst dokumentów, różnice w sumarycznej wierności i trafności między oboma wariantami chmurowymi nie osiągnęły progu istotności statystycznej. Biorąc pod uwagę budżet projektu, wdrażanie wielokrotnie droższego modelu gpt-4o jest pozbawione technologicznego uzasadnienia, ponieważ wariant mini dostarcza syntezę o identycznej merytoryce w identycznym ułamku czasu. Model z rodziny Bielik zostaje kategorycznie odrzucony ze względu na nieakceptowalne dla systemów webowych opóźnienia i skrajnie wysoką podatność na zniekształcanie faktów z dostarczonych regulaminów.