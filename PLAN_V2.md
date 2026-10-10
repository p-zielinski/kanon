# Kanon v2

> Status: plan po ósmej krytyce. Prompty wdrożenia są w PROMPTS_V2.md. Kodu v2 jeszcze nie ma.
> Odczyt: `runs/0/`, 1218 zdarzeń. Ten katalog zostaje świadectwem i nie jest wznawiany.
> Jeden proces ma iść wiele dni na CPU, także na VPS z 1 GB RAM. Schemat jest pętlą reduktora w paczkach i ma się skończyć. Dni po nim są warsztatem: co runda nowe zadanie z dwóch kart w oknie, oraz co dziesiąta runda mutacją faktu, którą kernel sprawdza.

Wiedza ma dwa źródła: reduktor i kernel, które liczą deterministycznie, oraz cztery własne sieci. Sieć ma widzieć zadanie. Produkcja komunikatu startuje z prefiksu 176 tokenów obserwacji. Wag z internetu nie ma. Cudzej biblioteki twierdzeń nie ma. Piątej sieci w pierwszym długim biegu nie ma.

## Co pokazało 1218 rund

Maszyna żyje. Kanon jest pusty, bo pętla nie szuka, a potem kopiuje jeden term.

| Fakt | Liczba |
|---|---|
| Zdarzenia | 1218 |
| Różne cele autora | 1: za każdym razem `(succ (zero))` |
| Przyjęcia do kanonu | 0 |
| Komunikaty łamacza | 1218 razy `( abstain f0 )` |
| Audyt zgodny z celem | 1, runda 0, definicja i tak spadła |
| Propozycje `(def f0 1 (zero))` | 1209 z 1218 |
| ε od rundy 600 | podłoga 0,05 |
| Grupa i zamrożone wagi, głębokość 1, rundy 610–1200 | obie średnio około 0,16 |

Ziarno zadania i ziarno komunikatu nie zawierają numeru rundy. Łamacz z ziarnem `seed + 20` zawsze wstrzymuje się od głosu. Taśma uczy się z propozycji przyjętych składniowo. Przy ε = 0,05 sieć ma 95% głosu i zostaje przy `(zero)`.

Ziarno v2 zawiera numer rundy. Inaczej wielodniowy bieg powtarza jeden cel.

## Dwa limity

`MAX_TOKENS` w samplerze jest zwykłą liczbą. Wolno jej zależeć od piętra.

`MAX_LEN` w sieci jest tabelą pozycji `nn.Embedding(MAX_LEN, 128)`. Sufit okna wybiera się raz, przed startem, i zostaje **256**.

Budżet komunikatu:

| Piętra | Budżet | Dlaczego tyle |
|---|---|---|
| 1 | 24 | identyczność z `propose` ma 12, następnik zmiennej 15 |
| 2–3 | 48 | dodawanie z `propose` ma 36, odejmowanie ograniczone 46, mnożenie przez `(call …)` ma 40 |
| 4–12 i ćwiczenia | 80 | zdanie z `step` i łańcuch taktyk mieszczą się w 80, a w 48 już nie |

Term ponad budżet piętra odpada, zanim reduktor go liczy.

## Okno, które się nie przesuwa

Pozycje są bezwzględne. Obserwacja jest uzupełniana tokenem `[PAD]` do **176** tokenów. Komunikat zawsze zaczyna się na indeksie 176 i ma do 80 tokenów. Razem 256.

W tych 176 są trzy stałe odcinki:

1. **48 tokenów na jedno ciało albo jedną tezę.** W ćwiczeniu dowodowym leży tu teza, bez taktyki. Krótszy napis dopełnia `[PAD]`. Teza, która się tu nie mieści, nie wchodzi do puli i nie jest losowana. Przy schemacie, dopóki specyfikacja nie padnie, odcinek jest pusty.
2. **Pięć par publicznych.** `(num k)` dla `k` od 0 do 10 zostaje czterema tokenami `( num k )`. Wieży `succ` w oknie nie ma.
3. **Dwie nazwy z półki**, po dwie pary. Generator najpierw wybiera te dwie karty, potem buduje ukryty term wyłącznie z nich. Co czwarte ćwiczenie jedna karta jest rodziną zadania. W pozostałych obie są składnikami, bez znacznika „to jest odpowiedź”. Zbiór pomiarowy pokazuje składniki i też nie ma tego znacznika.

Forward obserwacji liczy się raz na komunikat i zostaje prefiksem produkcji. Krok uczenia idzie po jednym agencie, batch **4**.

## Słownik w dniu zero

Długi bieg startuje jednym modelem. Od pierwszej rundy w słowniku sieci są `"4"` … `"10"`, `[PAD]` oraz głowy `eval`, `rewrite`, `base`, `step`, `claim`, `eq`, `dom`, `ih`. Formy termów zostają: `zero`, `succ`, `pred`, `if0`, `var`, `call`, `self`, `def`. Poza `(num …)` cyfra nadal jest indeksem zmiennej albo arnością i dalej tylko `0`–`3`.

## Przegląd do 16 tokenów

Zanim wstanie wielodniowy proces, komputer przegląda termy do **16** tokenów, bez `self` i bez `call`. Na półkę schodzi jeden najkrótszy term na rodzinę: identyczność, następnik, poprzednik, stała 0, 1 lub 2. Kolejność jest od najkrótszych termów, a przy tej samej długości deterministyczna. Ten sam wynik wchodzi na półkę grupy i na półkę solo. Zamrożone wagi nic nie dostają do wypisania.

Gdy czterech rodzin nie ma, tygodnia nie ma. Test 20 rund (różne cele, choć jeden `break`, ziarno z numerem rundy) dotyczy pętli, nie przeglądu.

## Kształty

Piętra 2–8 oraz pierwsze fakty pięter 9–12 są skończoną listą drzew. Numer termu dekoduje się na dziury. Lista termów nie powstaje. Kształt konkretnego dodawania nie jest wpisany. Wpisane jest drzewo i specyfikacja zachowania na dziedzinie.

Złożenie wołań jest poziomem warsztatu. Osobnego piętra na term, który „nie jest stałą”, nie ma: najkrótsze takie wołanie to opakowana identyczność i półce nic nie daje.

Dziura prosta `S`, bez `call`:

- `(zero)`, `(var i)` dla `i` mniejszego od arności,
- jeden `succ` albo jeden `pred` wokół `(zero)` albo `(var i)`.

Przy arności 2 jest ich 9, przy arności 1 jest ich 5. `call` stoi tylko w miejscach wypisanych przy kształcie.

`(if0 warunek gałąź-zera gałąź-niezera)`: zero bierze gałąź zera, tak jak reduktor v1.

| Kształt | Drzewo | Specyfikacja, która zamyka piętro |
|---|---|---|
| R1 | `(if0 S S (W (self S …)))`. `self` ma tyle argumentów, ile arność. `W` to nic, jeden `succ`, jeden `pred`, albo jeden `(call nazwa …)`, w którym dokładnie jeden argument jest tym `self`, a reszta to `S` | dodawanie; mnożenie; suma `0+…+n`; potęga, przy `0^0 = 1` |
| R2 | `(if0 S S (if0 S S (W (self S …))))`, to samo `W` | odejmowanie ograniczone. Wisi na półce, samo frontery piętra 2 nie zamyka |
| N1 | `(if0 (call nazwa S …) S S)` | porównanie: `1`, gdy `x ≤ y`, inaczej `0`. Minimum i maksimum, jeśli padną, wiszą i frontery nie zamykają |
| P | `(if0 (var 0) S (if0 (pred (var 0)) S (self (pred (pred (var 0))))))`. Dwa `pred` stoją tylko w tym jednym argumencie. Dziury bez `call` | parzystość na `0..10` |
| G | stałe zmienne, dwie nazwy binarne `A` i `B` z półki: `(if0 (var 1) (var 0) (if0 (call A (var 0) (var 1)) (self (var 1) (call B (var 0) (var 1))) (self (var 1) (call B (var 1) (var 0)))))`. W termie dwa `self`, na jednej ścieżce redukcji jedno | NWD na parach `0..6` |

R1 przy dziurach prostych i owinięciach z półki piętra 1 ma około 50 tysięcy termów. R2 około 4 milionów. N1, P i G są mniejsze. Dziedzina arności 2 to pary `0..6`. Każde piętro ma własny numer termu, od zera. W `resume.pt` leży trójka: piętro, kształt, numer. Kolejność wewnątrz kształtu jest od najkrótszych, więc pierwsze trafienie rodziny jest najkrótsze i przebieg tej rodziny na tym piętrze staje.

Bez odejmowania z R2 porównanie i NWD mogą wyczerpać się puste. Frontera i tak schodzi.

### Paczka

Schemat nie jest rundą na term. Jedna paczka liczy **2000** termów albo **2 minuty**, co pierwsze. Potem zapis `resume.pt` i, gdy jakieś piętro jest już zamknięte, jedno ćwiczenie. R2 przy tym rytmie zostaje pracą na godziny. Runda na każdy z 4 milionów termów sama z siebie, przy zapisie około 50 MB, zajmuje tygodnie dysku, zanim wstanie mnożenie.

Co 10 paczek dopisuje się wiersz metryki: piętro, kształt, numer. Wiersza na term nie ma.

### Taktyki

Schemat taktyki też jest paczką i też dekoduje numer, nie iloczyn wszystkich ciągów.

| Kształt | Jak rośnie | Zamyka piętro |
|---|---|---|
| E | `(claim (eq L R) (dom lo hi) (eval))`. `L` i `R` to `S` albo jeden `(call …)`. `hi > lo`, oba z `0..10`. Napisy stron się różnią. Co najmniej jedna strona woła rodzinę z piętra 2 albo wyższego. Jedno wstawienie ciała wpisu w drugi napis faktem E nie jest | jeden fakt równości na tej dziedzinie |
| W | `(claim (eq L R) (rewrite nazwa))`. `R` powstaje z `L` przez jedno wstawienie ciała wpisu, w jego dziedzinie | jeden przyjęty `rewrite` |
| C | start od faktu E albo W już na półce. Doklej jedną taktykę `eval` albo `rewrite`. Kernel ma przyjąć dłuższy napis. Sufit: 8 taktyk i 80 tokenów | jeden przyjęty łańcuch |
| I | `(claim (eq L R) (base) (step (ih) (rewrite nazwa)))` na szkielecie `f(x,0)`, `f(0,x)`, `f(x,1)`. `f` i `nazwa` przebiegają półkę | jedno zdanie, które przeszło `base` i `step` |

Na półkę schodzi jeden najkrótszy fakt, który kernel przyjął i który zamyka piętro, oraz co najwyżej **osiem** dalszych, jako pula ćwiczeń. Do puli wchodzą najpierw fakty, które przechodzą filtr E. Fakty W dokładają się, dopóki ośmiu miejsc brakuje. Reszta zbioru na półkę nie wchodzi. Tautologia, czyli obie strony identyczne przed taktyką albo kopia faktu już wiszącego, odpada.

Ten sam term i ten sam fakt ze schematu wchodzą na półkę grupy i na półkę solo. Różnica grupy, solo i zamrożonych wag zaczyna się na ćwiczeniu i na mutacji.

## Frontera

Losowalne jako schemat jest najniższe piętro, którego wymaganej rodziny jeszcze nie ma i którego kształt się nie wyczerpał. Po przeglądzie do 16 tokenów jest nim piętro 2.

| Piętro | Kształt | Frontera schodzi, gdy |
|---|---|---|
| 1 | przegląd do 16 tokenów, bez `self` | cztery rodziny |
| 2 | R1, potem R2 | półka ma dodawanie, a R2 już znalazło odejmowanie albo się wyczerpało |
| 3 | R1 | półka ma mnożenie, albo kształt się wyczerpał |
| 4 | N1 | półka ma porównanie, albo kształt się wyczerpał |
| 5 | P | półka ma parzystość, albo kształt się wyczerpał |
| 6 | R1 | półka ma sumę ograniczoną, albo kształt się wyczerpał |
| 7 | G | półka ma NWD, albo kształt się wyczerpał |
| 8 | R1 | półka ma potęgę, albo kształt się wyczerpał |
| 9 | E | jeden fakt z filtrem E, albo kształt się wyczerpał |
| 10 | W | jeden przyjęty `rewrite`, albo kształt się wyczerpał |
| 11 | C | jeden przyjęty łańcuch, albo kształt się wyczerpał |
| 12 | I | jedno zdanie z `base` i `step`, albo kształt się wyczerpał |

Gdy numer termu dojdzie do końca kształtu i rodziny nie ma, log pisze „schemat wyczerpany”, rodzina zostaje pusta, frontera schodzi na następne piętro. Wyczerpany kształt nie startuje od nowa. Puste NWD nie zatrzymuje potęgi ani dowodu. Ćwiczenie, które potrzebuje pustej rodziny, nie jest losowane.

Złożenie dwóch następników piętra 2 nie zamyka. Zadanie, którego ukryty term przekracza paliwo 256, 4096 węzłów albo 10000 kroków, nie powstaje. Piętra powyżej 12 śpią w tym samym kodzie: kolejne kształty dopisuje się bez ruszania wag.

## Ćwiczenia

Po zejściu frontery z piętra zostaje ono warsztatem. Warsztat nie ma ostatniej rundy. To jest treść wielodniowego biegu.

Komputer buduje zadanie z dwóch kart okna, ziarno zawiera numer rundy, reduktor albo kernel liczy werdykt, zanim ktokolwiek mówi. Grupa, solo i zamrożone wagi dostają to samo zadanie wspólne. Taśmy i ε są osobne. Zamrożone wagi kroku nie dostają i śladu nie zapisują.

Poziom warsztatu, osobna oś od frontery. Złożenie używa tylko dwóch nazw z okna. Głębszy poziom to głębsze złożenie tych dwóch nazw.

| Poziom | Zadanie |
|---|---|
| 0 | jedno `(call …)` do jednej z dwóch kart, świeże wejścia z `0..6` |
| 1 | złożenie dwóch wołań tych kart |
| 2 | złożenie trzech |
| 3 | złożenie czterech |
| 4 | odtworzenie taktyki faktu z puli wspólnej. W oknie jest teza, taktyki nie ma |
| 5 | ta sama taktyka na świeżych `(num k)` z `0..10`, w paliwie. Kernel ocenia taktykę mówcy |
| 6 | łańcuch 2..8 taktyk |

Ostatnie 50 świeżych ćwiczeń grupy na poziomie, ze sukcesem co najmniej 0,6, podnosi poziom o jeden. Poniżej 0,3 poziom stoi. Poziomy 4–6 czekają, aż pula faktów ze schematu taktyki jest niepusta. Do tej chwili sufit rampy to poziom 3. Sufit całego warsztatu to poziom 6. Na suficie losowanie trwa dalej: nowe wejścia, nowe tezy z puli, nowe liczby. Podłoga ε na znanym poziomie nie kończy biegu, bo cel co rundę jest inny.

Runda ćwiczenia ma jednego mówcę. Rotacja wskazuje, która z czterech sieci mówi. Jedna próba. Solo ma jedną sieć i też jedną próbę. Zamrożone wagi mają jedną próbę. Reduktor albo kernel stawia werdykt. Trafienie dopisuje ten sam ślad do taśm wszystkich czterech sieci grupy. Półka stoi. Bramka `admit` z v1 tej rundy nie otwiera: ćwiczenie nie dokłada definicji.

W trakcie schematu ćwiczenie jest jedno na paczkę i zostaje na najniższym odblokowanym poziomie. Gdy frontera zejdzie z piętra 12, dziewięć rund na dziesięć jest ćwiczeniem wspólnym, jedna jest mutacją.

## Mutacja

Mutacja wstaje, gdy frontera zeszła z piętra 12. Co dziesiąta runda komputer bierze fakt z puli tej krzywej i robi **64** mutacje: inna liczba z `0..10`, inne złożenie dwóch nazw widocznych w oknie, albo jedna doklejona taktyka w suficie 8 i 80 tokenów. Kernel każdą ocenia. Lista szkieletów nie powstaje i stan „pąk wyczerpany” nie istnieje, bo liczby i złożenia dwóch kart nie mają końca.

Trafienie, które przechodzi filtr E albo jest dłuższym łańcuchem przyjętym przez kernel i nie jest kopią, wchodzi do puli tylko tej krzywej. Pula wspólna, z której losuje się ćwiczenie grupy, solo i wag zamrożonych, zostaje przy faktach schematu obecnych na obu półkach. Powtórka faktu lokalnego jest zadaniem wyłącznie krzywej, która go ma, w rundzie mutacji. Do luki 20 punktów lokalny fakt nie wchodzi.

Na tej powtórce łamacz ma cztery próby wskazać punkt, który kernel sam potwierdzi jako rozjazd. Potwierdzone zbicie wchodzi na taśmę łamacza. 64 mutacji łamacz nie przegląda: te sprawdza kernel.

## Półka

Przyjęta definicja i przyjęty fakt ze schematu albo z mutacji schodzą na półkę od razu. Wpis nazwany albo wołany nie znika po 12 rundach. Wygasanie 12 rund zostaje tylko przy buforze roboczym, który jeszcze nie przeszedł dziedziny.

Nazwy `f0`–`f31` są oknem. Półka ma stałe identyfikatory. Zadania pomiaru, ćwiczenia i powołania trzymają identyfikator. Przy składaniu obserwacji identyfikator dostaje nazwę z okna. W jednej obserwacji widać dwie nazwy.

Ława co 50 rund liczy pierścień tuż za dziedziną oraz kilka punktów z wnętrza. Rozjazd liczby zdejmuje wpis i każdy zapis, który się na niego powołuje, także z puli. Limit kroków albo paliwa jest „nie wiadomo” i wpisu nie zdejmuje.

Dziedzina arności 1 to `0..10`. Arności 2 to pary `0..6`, o ile reduktor zdąży. Rodzinę nadaje komputer, gdy zachowanie zgadza się ze specyfikacją na dziedzinie.

## Taśma i ε

Ciąg uczący to 176 tokenów obserwacji i potem komunikat, który automat sprawdził. Na taśmę wchodzi też term albo fakt przyjęty ze schematu, z obserwacją tej rodziny. `(abstain …)` nie wchodzi. Komunikat rozjechany na dziedzinie albo na kernelu nie wchodzi. Pusta taśma nie rusza wag.

Taśma ma sufit: **256** śladów na agenta i na poziom warsztatu. Starszy ślad odpada. `events.jsonl` obraca się przy **32 MB**; zostaje bieżący plik i jeden poprzedni. `metrics.csv` zostaje w całości i rośnie o wiersz na 10 paczek albo 10 rund warsztatu.

Zegar ε tyka tylko w rundzie, która gra ten poziom.

- Przy schemacie sieć produkcji nie wybiera.
- Autor ćwiczenia schodzi ×0,995 do podłogi 0,25. Nowy poziom warsztatu zaczyna od 1.
- Łamacz schodzi ×0,995 do podłogi 0,5. Zegar tyka na powtórce faktu lokalnego, gdzie szuka rozjazdu.
- Wykonawca schodzi ×0,995 do podłogi 0,05 dopiero od pierwszej swojej odpowiedzi przyjętej na tym poziomie.
- Mówca rundy mutacji, dopóki ta krzywa nie ma własnego faktu lokalnego, ma ε = 1.

Krok jest AdamW, lr `3e-4`, przycięcie 1, batch 4, jeden agent naraz. Batch 32 przy oknie 256 dochodzi na maszynie deweloperskiej do około 626 MB samego procesu. Batch 4 zostaje przy około 316 MB.

## Kernel taktyk

`eval` liczy obie strony na dziedzinie ze zdania. Fakt mówi „równe na tej dziedzinie”.

`rewrite nazwa` podmienia podterm ciałem wpisu, w jego dziedzinie. Poza dziedziną kernel odrzuca. Jedno takie wstawienie jest faktem W, nie faktem E.

`base` redukuje obie strony przy zerze.

`step` stawia cel jako całą równość przy `(succ (var 0))`. Lokalne `ih` mówi, że teza zachodzi dla `(var 0)`. Kernel wolno raz podmienić podterm zgodny z lewą stroną tej tezy na prawą stronę. Przy `add(x, 0) = x` po redukcji zostaje `succ(add(n, 0))` i ta jedna podmiana zamienia `add(n, 0)` na `n`. Założenie nie jest wpisem półki. Samo przepisanie dodawania, bez tej podmiany, zamienia `add(x, 0)` w `if0`.

Lean jako program i jego biblioteka zostają poza v2.

## Zbiór pomiarowy

Pięćdziesiąt zadań powstaje raz, w chwili gdy frontera schodzi z piętra 8. Źródłem są wpisy położone przez schemat i obecne na obu półkach. Karty są składnikami zadania, bez znacznika rodziny. Potem zbiór się nie zmienia.

Do tej chwili metryka ma świeżą dokładność ćwiczeń i nie ma luki. Co 50 rund warsztatu, bez kroku uczenia, liczy się ten stały zbiór. Luka 20 punktów dotyczy złożenia. Dwa punkty to szum. Kto pierwszy znalazł dodawanie, tą luką nie jest: znalazł je ten sam kształt.

## Metryki

Co 10 paczek albo co 10 rund warsztatu: runda albo numer paczki, frontera, kształt i numer termu, czy kształt się wyczerpał, poziom warsztatu, świeża dokładność ostatnich 50 ćwiczeń grupy, solo i wag zamrożonych, rozmiar półki, rozmiar puli wspólnej i pul lokalnych, liczba śladów na taśmie, zdjęcia ławy.

Osobno numer paczki wejścia dodawania, mnożenia i pierwszego faktu E oraz znacznik, czy fakt wszedł ze schematu, czy z mutacji. Sam rozmiar półki od rundy zero nic nie mówi: przegląd do 16 tokenów kładzie cztery rodziny od razu.

`canon/snapshot.json` pisze półkę (identyfikator, nazwa w oknie albo brak, rodzina, dziedzina, ciało, powołania), pulę wspólną, pule lokalne, numer termu i poziom warsztatu. Ciało jest dla człowieka.

## Bieg na wiele dni

Wielodniowy start jest jednym poleceniem, gdy przegląd do 16 tokenów położył cztery rodziny, a w kodzie śpią kształty, paczka, warsztat, mutacja, okno 256, `[PAD]`, `ih` i sufit taśmy.

Jedno wywołanie zostaje w procesie. Zapis `resume.pt` jest atomowy, przez plik tymczasowy, i podmienia jeden plik: wagi, optimizery, taśmę przyciętą do sufitu, numer termu, poziom warsztatu, pule. W schemacie zapis jest po każdej paczce. W warsztacie co **50** rund albo co **2 minuty**, co pierwsze. Przerwanie gubi co najwyżej niedokończoną paczkę. Katalog: `runs/v2-seed0/`. `runs/0/` nie jest kasowany i nie jest wznawiany. `runs/` zostaje w `.gitignore`.

Schemat ma się skończyć. Dni po nim są warsztatem na suficie poziomu oraz mutacją co dziesiątą rundę. Wyczerpanie kształtu schodzi fronterą.

Osobna komenda tylko do odczytu ładuje zapis, zadaje jeden term albo jedno zdanie i wypisuje werdykt. Kroku, półki, kształtu, warsztatu i puli nie rusza.

## VPS i kopia na GitHubie

Docelowa maszyna wielodniowa ma 1 GB RAM i jeden CPU. Pierwszy długi bieg na niej nie czeka na GPU. Na czas biegu pudełko nie robi nic innego. Swap 1 GB jest bezpiecznikiem na wypadek szczytu reduktora.

Co **6 godzin** ten sam proces wysyła kopię, po udanym zapisie lokalnym:

- `snapshot.json`
- `metrics.csv`
- sześć plików wag: agenty 0–3, solo, zamrożone

Wagi sześciu sieci tego rozmiaru to około 20 MB. Optimizer, taśma i `resume.pt` zostają na dysku VPS. Z nich proces wznawia rundę. Z wydania GitHuba da się odtworzyć wagi i przeczytać półkę, nie wznowić krok AdamW.

Wydanie `kanon-v2-latest` jest podmieniane (`--clobber`). Raz na dobę powstaje osobny tag `kanon-v2-RRRR-MM-DD`, bez podmiany, żeby miesiąc dało się cofnąć o dzień. Commit co kilka godzin z plikiem wag na `main` zostawia w historii gigabajty, których usunięcie pliku nie kasuje.

Token wypychania siedzi w środowisku VPS, poza repozytorium. Skrypt wysyłki stoi poza `model.py`. Nieudana wysyłka pisze się w logu, bieg idzie dalej. Gdy repozytorium jest publiczne, wagi i metryki też są publiczne.

## Kolejność kodu

1. Etap 0, test 20 rund: różne cele, ziarno z numerem rundy, choć jeden `break`.
2. Model z oknem 256 i pełnym słownikiem. Obserwacja do 176. Produkcja komunikatu startuje z tego prefiksu. Batch 4.
3. Przegląd do 16 tokenów na obie półki. Bez czterech rodzin piętra 1 tygodnia nie ma.
4. Kształty R1 i R2 piętra 2, paczka 2000 termów albo 2 minuty, zapis po paczce, jedno ćwiczenie na paczkę. Koniec bez dodawania schodzi fronterą.
5. Reszta arytmetyki: mnożenie, porównanie, parzystość, suma, NWD, potęga. Pusta rodzina schodzi fronterą. Po zejściu z piętra 8 powstaje stały zbiór 50 zadań.
6. Kształty E, W, C i I. Filtr E odcina synonim jednego `call`. Łańcuch rośnie z faktu, nie z iloczynu taktyk. `ih` podmienia jeden podterm.
7. Warsztat poziomów 0–6 na dwóch kartach. Trafienie idzie na taśmę wszystkich czterech i nie woła `admit`. Mutacja: 64 sprawdzeń kernela na rundę, pula lokalna, bez stanu wyczerpania.
8. Sufit taśmy 256, obrót `events.jsonl` przy 32 MB, zapis warsztatu co 50 rund albo co 2 minuty, wysyłka co 6 godzin.
9. Jedno polecenie na wiele dni, katalog `runs/v2-seed0/`, maszyna z 1 GB RAM.

Piątej sieci i profilu GPU w tej kolejności nie ma.
