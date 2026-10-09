# Prompty dla Grok 4.7

Uruchamiaj **po jednym**, w katalogu repozytorium, w kolejności. Każdy prompt zakłada, że poprzedni jest skończony i testy z niego są zielone. Nie łącz promptów w jednej sesji. Nie rozszerzaj zakresu „przy okazji”.

Wspólna rama, którą każdy prompt już zawiera w środku: przeczytaj `PLAN.md` i `docs/SETUP.md`, trzymaj parametry z tabeli „Parametry v1”, nie pobieraj wag, nie dodawaj kanału języka naturalnego.

Grok instaluje PyTorch sam, według `docs/SETUP.md`. Osobnej zgody na to nie potrzeba. Pobierania modelu nie ma w żadnym prompcie.

---

## Prompt 1 — szkielet i środowisko

```
Przeczytaj PLAN.md i docs/SETUP.md. Zrób tylko szkielet eksperymentu Kanon.

Z katalogu repozytorium utwórz venv, zainstaluj torch z indeksu CPU i pytest, tak jak w docs/SETUP.md. Dopisz requirements.txt z komentarzem, że wagi biorą się z losowej inicjalizacji.

Utwórz puste moduły wymienione w sekcji „Układ repozytorium” w PLAN.md oraz tests/test_smoke_env.py, który importuje torch i sprawdza, że nn.Linear(8, 8) ma wagę o kształcie (8, 8).

Nie pisz jeszcze parsera, sieci ani pętli rundy. Nie dodawaj zależności huggingface, transformers, ollama ani żadnych wag.

Na końcu uruchom pytest -q i w .cursor/work/kanon.md dopisz, że środowisko stoi, razem z wersją torch.
```

## Prompt 2 — parser i reduktor Kalk

```
Przeczytaj PLAN.md, sekcje „Język Kalk” i „Parametry v1”. Zaimplementuj wyłącznie src/kanon/language.py i src/kanon/reduce.py oraz testy.

Najpierw testy, potem kod. Przypadki:
- (succ (succ (zero))) redukuje się do 2
- (pred (zero)) redukuje się do 0
- (if0 (zero) (succ (zero)) (zero)) daje 0; (if0 (succ (zero)) (succ (zero)) (zero)) daje 1
- def dodawania z przykładu w PLAN.md dla (num 2) (num 3) daje 5; przykład jest tylko w teście, nie w obserwacji agenta
- (self ...) bez def kończy się błędem parsowania
- (num 4) w ciele def jest błędem parsowania
- (num 4) w argumencie break jest legalne i rozwija się do czterech succ
- paliwo 256: rekurencja, która nie schodzi do zera, kończy się błędem fuel, a nie nieskończoną pętlą
- term ponad 4096 węzłów kończy się błędem memory
- komunikat po polsku i komentarz w stylu Pythona nie parsują się

Słownik tokenów ma być dokładnie listą z PLAN.md. Arność maksymalna 3. Sloty tylko f0–f31.

Nie pisz generatora zadań ani sieci.
```

## Prompt 3 — generator zadań i zbiór pomiarowy

```
Przeczytaj PLAN.md, sekcję „Generator zadań”. Zaimplementuj src/kanon/tasks.py i testy.

Generator jest zwykłym programem, bez sieci. Z ziarna buduje ukryty term danej głębokości, 5 par publicznych na wejściach 0..6 i 16 par audytowych na wejściach 0..10, rozłącznych z publicznymi.

Głębokość 1 obejmuje tylko identyczność, succ, pred i stałe 0, 1, 2. Głębokość 2 dokłada złożenie i jedną prymitywną rekurencję (dodawanie albo odejmowanie ograniczone). Głębokość 3, przy pustym kanonie, dokłada tylko głębsze złożenie prymitywów.

Funkcja build_holdout(seed) zapisuje 50 zadań na głębokości 1..4 i przy drugim wywołaniu z tym samym seed zwraca te same termy. W obiekcie, który wolno pokazać agentowi, nie ma pola z termem ukrytym. Term ukryty zostaje w obiekcie audytowym używanym wyłącznie przez automat.

Test: dwa przebiegi z seed 0 są identyczne; przebieg z seed 1 różni się; serializacja obserwacji zadania nie zawiera nazwy pola hidden.
```

## Prompt 4 — kanon, cmentarz, wygasanie, ława

```
Przeczytaj PLAN.md, sekcję „Kanon, cmentarz, wygasanie”. Zaimplementuj src/kanon/canon.py i testy. Najpierw testy.

Wpis przyjmuje def tylko razem z parami publicznymi, 16 parami audytowymi i numerem rundy. last_used_round aktualizuje się, gdy jakiś term rozwiązania zawiera (call slot ...).

Po 12 rundach bez wywołania slot wraca do puli i def nie występuje w eksporcie powtórki. Co 4 rundy najstarszy wpis jest redukowny na swoich audit_pairs; rozjazd usuwa wpis.

Cmentarz zapisuje def, wejście, wynik def, wynik targetu i rundę. Eksport dla łamacza zawiera wejście i fakt rozjazdu, bez termu targetu.

Maksimum 32 wpisów, sloty f0–f31. Test na przepełnienie: przyjęcie 33. definicji bez wygasania kończy się odmową, nie cichym nadpisaniem.

Nie podłączaj jeszcze sieci.
```

## Prompt 5 — role, rotacja, obserwacja

```
Przeczytaj PLAN.md, sekcje „Role”, „Obserwacja” i „Runda” (tylko kształt komunikatów). Zaimplementuj src/kanon/roles.py i testy.

Dla rundy r role są: autor r%4, łamacz (r+1)%4, użytkownik (r+2)%4, obcy (r+3)%4. Test na 8 kolejnych rund: każdy agent był każdą rolą dokładnie dwa razy.

Parser slotu przyjmuje wyłącznie kształt ze swojej roli. (propose ...) od łamacza, zdanie po polsku i def z (num k) w ciele dają odrzut. Odrzut jest tokenem [ABS], bez tekstu wyjaśnienia.

Przeczytaj też sekcję „Budżet szkiców”. Autor składa (draft (def slot arity expr)). Użytkownik i obcy składają (draft expr). (propose ...), (use ...) i (solve ...) przed pierwszym szkicem są odrzutem. Po czwartym szkicu draft jest odrzutem. Odpowiedź na szkic to tokeny [TRY] i pięć wyników par publicznych, (num k) albo (err). W tej odpowiedzi nie ma par audytowych. Nazwy spoza f0–f31 nie istnieją.

Obserwacja jest ciągiem tokenów ze słownika. Autor widzi kanon, pary publiczne i jeden wolny slot. Łamacz widzi dodatkowo ciało kandydata. Użytkownik widzi swoje pary i slot kandydata. Obcy nie dostaje ciała kandydata odrzuconego ani termu ukrytego. Test: w obserwacji obcego nie ma podciągu ukrytego termu, który generator trzyma w obiekcie audytowym.

Nie implementuj jeszcze pełnej pętli przyjęcia do kanonu.
```

## Prompt 6 — sampler legalnych termów

```
Przeczytaj PLAN.md, akapit o zimnym starcie w sekcji „Sieć”. Zaimplementuj src/kanon/sample.py i testy.

Sampler dostaje marker roli i obserwację, a zwraca komunikat, który parser tej roli akceptuje. Losowanie jest jednostajne po legalnych produkcjach w danym kroku, z limitem 64 tokenów. Seed daje powtarzalny wynik.

Dla autora ciało def używa zero, succ, pred, if0, var, self, call (call tylko do slotów faktycznie obecnych w kanonie). (num k) w ciele nie występuje nigdy. Sampler autora umie złożyć (draft (def slot arity expr)) oraz (propose (def slot arity expr)) dopiero gdy w obserwacji jest już co najmniej jeden [TRY]. Dla łamacza sampler umie złożyć (abstain slot) oraz (break slot (num k) ...) z k z zakresu 0..10, bez draft. Dla użytkownika (draft expr) i (use slot expr) zawierają (call slot ...) wskazanego kandydata. Dla obcego (draft expr) i (solve expr) wołają tylko sloty kanonu. Nazwy spoza f0–f31 nie istnieją.

Test: 50 próbek każdej roli z pustym kanonem i z kanonem zawierającym f0 parsuje się do końca. Żadna próbka autora nie zawiera tokenu num.

Sieci jeszcze nie pisz. Sampler ma działać sam, bo pierwsze rundy opierają się na ε = 1.
```

## Prompt 7 — cztery sieci od losowych wag

```
Przeczytaj PLAN.md, sekcję „Sieć” i tabelę parametrów. Zaimplementuj src/kanon/model.py i testy.

Cztery transformery dekodujące: 4 warstwy, d_model 128, 4 głowy, d_ff 512, dropout 0. Ziarna 0, 1, 2, 3. Asercja: wagi agenta 0 różnią się od wag agenta 1. Asercja: w pliku źródłowym nie ma load, from_pretrained, huggingface, download.

Dekodowanie ograniczone: funkcja przyjmuje maskę legalnych produkcji (dostarcza ją sampler / parser) i zwraca rozkład tylko na legalnych. Test: produkcja nielegalna ma prawdopodobieństwo 0.

Zapis i odczyt czterech zestawów wag do katalogu runs/. Po odczycie tensor wagi jest równy zapisanemu.

Nie podłączaj pętli rundy. Nie pobieraj żadnego pliku wag.
```

## Prompt 8 — jedna runda i reguły kanonu

```
Przeczytaj PLAN.md, sekcję „Runda”. Zaimplementuj src/kanon/round.py i testy. Politykę na razie podstaw samplerem (ε = 1), sieć podepnij tylko jako opcjonalne źródło logitów, jeśli maska jest dostępna. Gdy test wymusza konkretny komunikat, wstrzyknij go, nie czekaj na sieć.

Przeczytaj też sekcję „Budżet szkiców”. Szkic redukowany jest tylko na 5 parach publicznych. propose, use i solve przed szkicem nie wchodzą w życie.

Scenariusze testowe, każdy z wstrzykniętymi komunikatami:
- propose autora bez wcześniejszego draft jest odrzutem i nie otwiera audytu
- draft, który trafia 5 par publicznych, a pada na audycie 16, nie wchodzi do kanonu
- łamacz podaje wejście, na którym def różni się od targetu: kandydat nie wchodzi, cmentarz rośnie
- łamacz wstrzymuje się, audyt 16 wejść przechodzi, użytkownik woła slot i jego rozwiązanie przechodzi: kandydat wchodzi do kanonu
- użytkownik rozwiązuje zadanie bez (call slot ...): kandydat nie wchodzi, nawet jeśli audyt autora przeszedł
- audyt znajduje rozjazd po wstrzymaniu łamacza: kandydat nie wchodzi
- obcy nie dostaje w obserwacji termu ukrytego
- po rundzie rola każdego agenta przesuwa się o jeden

Nie rób jeszcze kroku wstecznego. Zaloguj zdarzenie rundy jako obiekt, który da się zapisać do jsonl.
```

## Prompt 9 — krok uczenia po rundzie

```
Przeczytaj PLAN.md, sekcję „Krok uczenia”. Zaimplementuj src/kanon/update.py i testy.

Taśma trzyma zaakceptowane komunikaty końcowe z rolą agenta oraz aktualny eksport kanonu. Odrzucony komunikat nie trafia na taśmę. (draft ...) nie trafia na taśmę, także gdy 5 par publicznych się zgadzało. Cmentarz trafia na taśmę łamacza tylko jako potwierdzone zbicie.

Jeden krok AdamW, lr 3e-4, batch 32, przycięcie gradientu 1.0, entropia krzyżowa następnego tokenu, ciągi obcięte do 64 tokenów. Pusta taśma: krok pominięty, wagi bitowo te same.

Test: sztuczny ślad jednego poprawnego (solve ...) zmienia co najmniej jedną wagę agenta, który ten ślad dostał, i nie zmienia wag pozostałych trzech. Test: po kroku na eksporcie kanonu token (call f0 ...) jest nadal legalnym ciągiem uczącym, gdy f0 jest w kanonie.

Nie dodawaj drugiego optimizera ani zamrożonej kopii „referencyjnej” jako trybu pracy. Ewaluacja pomiarowa, którą dopiszesz w promptcie 11, po prostu nie woła tej funkcji.
```

## Prompt 10 — baselines

```
Przeczytaj PLAN.md, sekcję „Baselines”. Zaimplementuj src/kanon/baselines.py i testy.

Solo: jedna sieć, 4 propozycje autora na rundę, przyjęcie do kanonu wyłącznie po 16 wejściach audytowych, 4 kroki uczenia na rundę. Brak roli łamacza i użytkownika. Każda propozycja ma ten sam budżet szkiców 1..4 co autor grupy. Liczba pełnych sprawdzeń automatu na rundę jest tego samego rzędu co w grupie (cztery próby plus audyt).

Losowe wagi: architektura obcego, wagi z ziarna zapisanego przy starcie, krok uczenia nie jest wołany. W teście porównaj hash wag przed i po 5 rundach ewaluacji na kanonie grupy: hash ten sam.

Nie zrównuj jeszcze krzywych na zbiorze pomiarowym. Tu tylko pętle i gwarancje budżetu oraz braku zapisu wag w wariancie losowym.
```

## Prompt 11 — runner, metryki, smoke

```
Przeczytaj PLAN.md, sekcje „Metryki” i „Parametry v1”. Zaimplementuj src/kanon/run.py, krótki README.md (jak odpalić smoke i jak czytać runs/) i test smoke.

Pętla: runda grupy, potem update. Co 10 rund ewaluacja zbioru pomiarowego bez wołania update: sukces obcego grupy per głębokość, sukces solo, sukces obcego losowych wag na kanonie grupy, rozmiar kanonu, rozmiar cmentarza, odsetek zbić łamacza, odsetek zbić dopiero przez audyt, ε. Zapis runs/<seed>/events.jsonl, metrics.csv i snapshot kanonu.

Zbiór pomiarowy powstaje raz, przed rundą 0, z seedu niezależnego od seedu rund. Głębokość bieżąca startuje od 1 i rośnie według progu 0.6 z PLAN.md, sufit 4.

Ewaluacja obcego, solo i losowych wag używa tego samego budżetu szkiców 1..4. Szkice sprawdzane są na 5 parach publicznych zadania pomiarowego. Zaliczenie dalej wymaga 16 par audytowych.

Smoke: 20 rund, seed 0, CPU. Test sprawdza, że proces kończy się kodem 0, że powstają oba pliki logu i że w events.jsonl nie ma ciągu znaków ukrytego termu w obiektach obserwacji.

README ma mówić wprost: nie ma fazy dociągania modelu; torch jest w venv; smoke może skończyć się pustym kanonem i to nie jest błąd maszynerii.

Nie zwiększaj liczby agentów. Nie dodawaj piątego łamacza. Jeśli w logu widać odsetek fałszywych przepuszczeń, tylko go zapisz.
```

## Prompt 12 — przegląd zgodności z planem

```
Przeczytaj PLAN.md od deski do deski i porównaj z kodem. Nic nie dopisuj poza listą rozjazdów.

Sprawdź po kolei: brak pobierania wag, cztery niezależne zestawy wag, krok uczenia wołany po rundzie grupy i solo, krok nieużywany w ewaluacji i w baseline losowych wag, obserwacja bez ukrytego termu, (num) tylko w danych, rotacja ról, warunek kanonu (łamacz i audyt i wywołanie przez użytkownika), wygasanie 12, ława co 4, ε z podłogą 0.05, parametry sieci zgodne z tabelą, budżet szkiców 1..4 przed propose, use i solve, szkice tylko na parach publicznych, szkice nieobecne w taśmie uczącej, ten sam budżet w grupie, solo i losowych wagach.

Wynik zapisz w .cursor/work/kanon.md jako listę: zgodne / rozjazd z plikiem i linijką. Rozjazdów nie naprawiaj w tej sesji.
```
