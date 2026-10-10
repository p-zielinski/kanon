# Prompty v2

Uruchamiaj **po jednym**, w katalogu repozytorium, w kolejności od 1 do 22. Każdy zakłada, że poprzedni jest skończony i `pytest -q` jest zielone. Nie łącz promptów w jednej sesji. Nie rozszerzaj zakresu „przy okazji”.

Potem odpalaj prompt 23. Werdykt jest ostatnią linią sekcji „## Kontrola” w `.cursor/work/kanon.md`, nie ostatnią linią całego pliku. Tylko gdy ta linia to `SĄ ROZJAZDY`, odpal prompt 24. Powtarzaj tę parę, aż sekcja skończy się linią `WSZYSTKO DOBRZE`. Wtedy promptu 24 nie odpalaj.

Wspólna rama, którą każdy prompt implementacyjny już niesie: przeczytaj `PLAN_V2.md`. Kod v2 stoi w `src/kanon/v2/`. Wejście wielodniowe to `python -m kanon.v2`. `python -m kanon.run` i katalog `runs/0/` zostają ścieżką v1: nie czytaj ich, nie zapisuj, nie migruj `resume.pt`.

Zakazy, które obowiązują w każdym prompcie:

- Nie pobieraj wag. Nie dodawaj huggingface, transformers, ollama.
- Krotka `TOKENS` w `src/kanon/language.py` zostaje listą z `PLAN.md`. Nowe słowa są w słowniku sieci v2, nie w tej krotce.
- `if0`: zero bierze gałąź zera, tak jak reduktor v1. Wzoru nie odwracaj.
- Rotacja: autor `r%4`, łamacz `(r+1)%4`, użytkownik `(r+2)%4`, obcy `(r+3)%4`.
- Ziarna: grupa 0–3, zamrożone 4, solo 5. Piątej sieci nie ma.
- W plikach sieci v2 nie ma podciągów `load`, `from_pretrained`, `huggingface`, `download`. Słowo `payload` zawiera `load` i w tych plikach nie występuje.
- Nagłówek własnego zapisu wag czytaj przez `ast.literal_eval`, tak jak v1.
- Testy v1 zostają zielone. Nie przerabiaj ich pod v2.
- TDD, w każdym prompcie 1–22 i w 24: najpierw testy wypisane w prompcie. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu `pytest -q`, razem z testami v1, i krótki dopisek w `.cursor/work/kanon.md`. To samo zdanie jest wklejone w treść każdego promptu.

---

## Prompt 1 — pakiet v2 i ziarno z numerem rundy

```
Przeczytaj PLAN_V2.md, sekcję „Co pokazało 1218 rund” i „Kolejność kodu” punkt 1. Zrób szkielet src/kanon/v2/ i testy etapu 0. Ścieżki v1 nie ruszaj.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

python -m kanon.run oraz runs/0/ zostają. Ten prompt nie odpala biegu v1 i nie otwiera runs/0.

Nowa funkcja ziarna przyjmuje seed, numer rundy i indeks próby. Dwa numery rundy dają dwa różne ziarna zadania i dwa różne ziarna komunikatu łamacza. Samo seed bez numeru rundy nie wystarcza: to jest błąd z 1218 rund, gdzie cel autora był zawsze (succ (zero)), a łamacz zawsze wstrzymywał się od głosu.

Nie wołaj `play()` z v1. Jego `task_seed` nie przyjmuje numeru rundy, a łamacz bierze `seed + 20`, więc test na tej funkcji zawsze powtórzy jeden cel i jeden `(abstain …)`.

Test w katalogu tymczasowym, ε = 1, sampler v1, nowa funkcja ziarna:
- 20 rund: cele zadań autora, brane z generatora przez nowe ziarno, nie są wszystkie tym samym termem
- numery rundy 0..39: choć jeden komunikat łamacza to `(break …)`. Przejdź cały zakres, nie jedną losową dwudziestkę
- if0 zostaje wzorem v1. Runda 0 przy grupie ziaren 0–3: autor jest ziarnem 0, łamacz 1, użytkownik 2, obcy 3
- po teście katalog tymczasowy znika, runs/0/ ma ten sam rozmiar co przed testem

Nie pisz jeszcze okna 256, półki ani schematu.
```

## Prompt 2 — słownik dnia zero i budżet komunikatu

```
Przeczytaj PLAN_V2.md, sekcje „Dwa limity” i „Słownik w dniu zero”. Dopisz słownik sieci v2 i funkcję budżetu. TOKENS w language.py nie rośnie.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Słownik v2 jest krotką. Modelu jeszcze nie buduj. W krotce są napisy "4" … "10", [PAD] oraz głowy eval, rewrite, base, step, claim, eq, dom, ih. Formy termów zostają: zero, succ, pred, if0, var, call, self, def. Poza (num …) cyfra nadal jest indeksem zmiennej albo arnością i dalej tylko 0–3. W parach (num k) dla k od 0 do 10 jest w słowniku v2 jednym tokenem liczby. Tezę zdania parsuje dopiero prompt 10.

Budżet komunikatu, zwykła liczba, nie waga:
- piętro 1: 24
- piętra 2–3: 48
- piętra 4–12 i ćwiczenia: 80
Term ponad budżet odpada, zanim reduktor go liczy. Test: sztuczny term o 49 tokenach odpada na piętrze 2 i przechodzi bramkę budżetu na piętrze 4. Test: krotka TOKENS ma tę samą długość co przed tym promptem.

Sieci jeszcze nie składaj. Okna jeszcze nie koduj.
```

## Prompt 3 — sieć v2, okno 256, prefiks obserwacji

```
Przeczytaj PLAN_V2.md, sekcje „Okno, które się nie przesuwa” i „Dwa limity”. Zaimplementuj src/kanon/v2/model.py. src/kanon/model.py zostaje przy MAX_LEN 64.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Architektura: 4 warstwy, d_model 128, 4 głowy, d_ff 512, dropout 0. Sześć sieci da się zbudować z ziaren 0, 1, 2, 3, 4 i 5. Wagi ziarna 0 różnią się od ziarna 1. Zamrożone są ziarno 4, solo ziarno 5. Piątej sieci uczących się, poza tą szóstką, nie twórz.

MAX_LEN = 256 jest tabelą pozycji i wybiera się raz w konstruktorze. Test: próba podania czasu 257 kończy się błędem. Test: w pliku źródłowym sieci v2 nie ma podciągów load, from_pretrained, huggingface, download ani słowa payload.

Obserwacja jest uzupełniana [PAD] do 176 tokenów. Komunikat zaczyna się na indeksie 176. W 176 są trzy odcinki: 48 tokenów ciała albo tezy, pięć par publicznych, dwie nazwy z półki po dwie pary. (num 10) w oknie to cztery tokeny ( num 10 ), nie wieża succ. Test: dwa różne ciągi par dają dwa różne prefiksy o długości 176, a pierwszy token komunikatu stoi na indeksie 176.

Forward prefiksu da się policzyć raz i podać produkcji. Test: rozkład następnego tokenu przy dwóch różnych prefiksach nie jest identyczny. Maska produkcji zeruje prawdopodobieństwo tokenu spoza maski.

Nie podłączaj pętli wielu dni. Nie pisz do runs/.
```

## Prompt 4 — półka, identyfikatory, ława

```
Przeczytaj PLAN_V2.md, sekcję „Półka”. Zaimplementuj półkę v2 i testy. Kanon v1 w src/kanon/canon.py zostaw.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Wpis przyjęty na dziedzinie dostaje stały identyfikator i nie znika po 12 rundach, także gdy jest wołany. Wygasanie 12 rund dotyczy tylko bufora roboczego, który dziedziny jeszcze nie przeszedł. Nazwy f0–f31 są oknem nakładanym przy składaniu obserwacji. Powołanie wewnątrz jednej półki trzyma lokalny identyfikator, nie napis f3. Test: po przewinięciu okna powołanie nadal wskazuje ten sam identyfikator, a napis f3 w obserwacji może oznaczać inny identyfikator niż rundę wcześniej.

Dziedzina arności 1 to 0..10. Arności 2 to pary 0..6, o ile reduktor zdąży. Rodzinę nadaje komputer, gdy zachowanie zgadza się ze specyfikacją. Test: term identyczności dostaje rodzinę identyczności; `(succ (succ (zero)))` nie dostaje rodziny identyczności ani następnika. Specyfikacji dodawania w tym prompcie jeszcze nie ma.

Ta sama definicja ze schematu kładziona na półkę grupy i na półkę solo ma to samo ciało i tę samą rodzinę. Identyfikator wpisu jest lokalny dla półki. Zadanie wspólne trzyma rodzinę; przy składaniu obserwacji każda półka podstawia swój identyfikator.

Ława co 50 rund liczy pierścień tuż za dziedziną i kilka punktów z wnętrza. Rozjazd liczby zdejmuje wpis i każdy zapis, który się na niego powołuje. Limit kroków albo paliwa jest „nie wiadomo” i wpisu nie zdejmuje. Test obu przypadków.

Zamrożone wagi nie dostają metody zapisu.
```

## Prompt 5 — przegląd do 16 tokenów

```
Przeczytaj PLAN_V2.md, sekcję „Przegląd do 16 tokenów”. Zaimplementuj przegląd i testy.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Komputer idzie od najkrótszych termów, bez self i bez call, przy tej samej długości deterministycznie, i staje, gdy cztery rodziny już leżą, nie później niż na 16 tokenach. Wewnątrz jednej długości najpierw termy bez if0, potem z if0. Inaczej krzak if0 zjada 60 sekund, zanim padnie (pred (var 0)). Nie wylicza wszystkich drzew if0 do 16 tokenów. Na półkę schodzi jeden najkrótszy term na rodzinę: identyczność, następnik, poprzednik, stała 0, 1 lub 2. To samo ciało i ta sama rodzina wchodzą na półkę grupy i solo. Zamrożone wagi nic nie dostają.

Test na suficie 8 znajduje co najmniej identyczność i stałą. Test na suficie 16 znajduje cztery rodziny, jest deterministyczny przy dwóch uruchomieniach i mieści się w 60 sekundach, bo staje po czwartej rodzinie. Gdy którejś rodziny nie ma, funkcja gotowości tygodnia zwraca fałsz. Test tego fałszu na suficie 1: najkrótszy term ma 3 tokeny, więc żadna rodzina nie wchodzi.

Nie uruchamiaj tygodnia. Nie pisz schematu R1.
```

## Prompt 6 — dziura prosta i numer termu

```
Przeczytaj PLAN_V2.md, sekcję „Kształty”, akapit o dziurze prostej S. Zaimplementuj dekoder numeru. Listy termów nie buduj.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

S, bez call:
- (zero), (var i) dla i mniejszego od arności
- jeden succ albo jeden pred wokół (zero) albo (var i)
Przy arności 2 jest ich 9, przy arności 1 jest ich 5. Test liczy dokładnie te liczby i sprawdza, że żaden element S nie zawiera call.

Numer dekoduje się na dziury. Przy arności 2 legalne numery S to 0..8. Test: długość termu przy numerze n+1 jest nie mniejsza niż przy n na tych dziewięciu. Odczyt jednego numeru z iloczynu dwóch dziur, na przykład numeru 80 przy podstawie 9, zwraca parę i nie tworzy listy 81 par.

Nie przechodź jeszcze specyfikacji dodawania.
```

## Prompt 7 — piętro 2, R1 i R2

```
Przeczytaj PLAN_V2.md, tabelę kształtów R1 i R2 oraz sekcję „Frontera” dla pięter 1–2.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

R1: (if0 S S (W (self S …))). self ma tyle argumentów, ile arność. W to nic, jeden succ, jeden pred, albo jeden (call nazwa …), w którym dokładnie jeden argument jest tym self, a reszta to S.
R2: (if0 S S (if0 S S (W (self S …)))) z tym samym W.

Specyfikacja dodawania na parach 0..6 zamyka rodzinę dodawania. Odejmowanie ograniczone, max(x-y, 0), wchodzi na półkę i samo piętra nie zamyka. Złożenie dwóch następników dodawaniem nie jest. Pierwsze trafienie w kolejności od najkrótszych jest tym, które zostaje, i przebieg tej rodziny wtedy staje. Po dodawaniu R1 staje, a R2 idzie do odejmowania albo do wyczerpania. Frontera schodzi dopiero wtedy. Samo dodawanie, przy nietkniętym R2, piętra 2 nie zamyka.

Każde piętro ma własny numer od zera. W stanie schematu leży trójka: piętro, kształt, numer. Pliku resume.pt jeszcze nie składaj, to prompt 18. Koniec kształtu bez rodziny pisze w logu „schemat wyczerpany”, rodzina zostaje pusta, frontera schodzi na następne piętro. Wyczerpany kształt nie startuje od zera drugi raz. Test na sztucznym kształcie o pięciu termach, z których żaden nie pasuje: numer piętra rośnie o jeden, drugi przebieg nie generuje termów od nowa.

Test wyroczni, bez pełnego skanu: term
(if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))
spełnia dodawanie i jest w obrazie R1. Term
(if0 (var 0) (zero) (if0 (var 1) (var 0) (self (pred (var 0)) (pred (var 1)))))
spełnia odejmowanie ograniczone i jest w obrazie R2. Indeks wyroczni wyliczasz z kształtu. Odczyt tego indeksu zwraca ten term. Szukanie startuje kilka numerów wcześniej i staje na pierwszym trafieniu specyfikacji. Skanu R1 od zera i pełnego R2, około 4 milionów termów, w teście nie odpalaj.

Ten sam term kładziesz na półkę grupy i solo.
```

## Prompt 8 — paczka schematu

```
Przeczytaj PLAN_V2.md, podsekcję „Paczka”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Jedna paczka liczy 2000 termów albo 2 minuty, co pierwsze. Potem zapis stanu jest wołany raz. Gdy jakieś piętro jest już zamknięte, po paczce jest dokładnie jedno ćwiczenie; w tym prompcie ćwiczenie może być atrapą, która liczy wywołania. Wiersz metryki co 10 paczek zawiera piętro, kształt i numer. Wiersza na term nie ma.

Test: kształt o 5000 szybkich termach daje trzy paczki (2000, 2000, 1000) i trzy zapisy, nie 5000. Test zegara: przy sztucznym zegarze, który po 10 termach zgłasza upływ 2 minut, paczka ma 10 termów i jeden zapis. Test: piętro 1 zamknięte, trzy paczki, trzy wywołania atrapy ćwiczenia. Test: żadne piętro nie zamknięte, atrapa ćwiczenia ma zero wywołań.

Nie uruchamiaj R2 w całości.
```

## Prompt 9 — reszta arytmetyki

```
Przeczytaj PLAN_V2.md, tabelę frontery pięter 3–8. C1 i C2 nie istnieją. Złożenie wołań nie jest piętrem.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Każde piętro ma własny numer od zera i własną specyfikację:
- 3, R1: mnożenie na parach 0..6
- 4, N1: porównanie, 1 gdy x ≤ y, inaczej 0. Minimum i maksimum, jeśli padną, wiszą i frontery nie zamykają
- 5, P: parzystość na 0..10. Dwa pred tylko w argumencie self, tak jak drzewo P w planie
- 6, R1: suma 0+…+n
- 7, G: NWD na parach 0..6, drzewo G z planu, dwa self w termie, jedno na ścieżce
- 8, R1: potęga, 0^0 = 1

Pusta rodzina po wyczerpaniu kształtu schodzi fronterą o jedno piętro, nie przeskakuje od razu do 8. Test: wyczerpanie piętra 4 otwiera 5, nie 8. Osobno: puste porównanie albo puste NWD nie zatrzymuje potęgi. Po ich wyczerpaniu frontera idzie dalej, aż piętro 8 wystartuje.

Test wyroczni, bez pełnego skanu R1: podane niżej termy są w obrazie swojego kształtu i spełniają specyfikację. Zanim sprawdzisz mnożenie i sumę, na półce testu leży dodawanie. Zanim sprawdzisz potęgę, leży mnożenie. Zanim sprawdzisz porównanie, leży odejmowanie ograniczone. Bez tych wpisów kształt nie ma wołania, którym te termy są owinięte. Pełne szukanie odpal tylko dla P: ma znaleźć parzystość. G jest małe, ale na półce bez pary funkcji binarnych kończy się wyczerpaniem i schodzi fronterą. Nie wymagaj trafienia NWD w teście jednostkowym.
- mnożenie: (if0 (var 0) (zero) (call add (var 1) (self (pred (var 0)) (var 1)))))
- porównanie, gdy monus jest na półce: (if0 (call monus (var 0) (var 1)) (succ (zero)) (zero))
- parzystość: (if0 (var 0) (zero) (if0 (pred (var 0)) (succ (zero)) (self (pred (pred (var 0))))))
- suma: (if0 (var 0) (zero) (call add (var 0) (self (pred (var 0)))))
- potęga: (if0 (var 1) (succ (zero)) (call mul (var 0) (self (var 0) (pred (var 1)))))

Pełnego skanu R1 od zera w teście nie odpalaj.
```

## Prompt 10 — kernel taktyk

```
Przeczytaj PLAN_V2.md, sekcję „Kernel taktyk”. Zaimplementuj kernel i testy. Lean i cudza biblioteka twierdzeń nie wchodzą.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

eval liczy obie strony na dziedzinie ze zdania. Fakt mówi „równe na tej dziedzinie”. Test: równość na 0..2, która pada na 3, jest faktem na 0..2 i nie jest faktem na 0..3.

rewrite nazwa podmienia podterm ciałem wpisu, w jego dziedzinie. Poza dziedziną kernel odrzuca. Jedno takie wstawienie jest rodzajem W, nie rodzajem E. Test obu.

base redukuje obie strony przy zerze.

step stawia cel jako całą równość przy (succ (var 0)). ih mówi, że teza zachodzi dla (var 0), i nie jest wpisem półki. Kernel wolno raz podmienić podterm zgodny z lewą stroną tej tezy na prawą stronę. Test na add(x, 0) = x, przy dodawaniu na półce. Samo rewrite, zanim cokolwiek zredukuje warunek, zostawia if0 i kroku nie domyka. Po redukcji zostaje succ(add(n, 0)). Jedna podmiana ih daje succ(n). Druga podmiana ih w tym samym kroku jest odrzucana. Test obu braków: rewrite bez ih oraz ih użyte drugi raz. Pełny filtr piętra 9 jest w prompcie 11; tu wystarczy, że jedno rewrite nie dostaje znacznika E.

Tautologia, obie strony identyczne przed taktyką, odpada.
```

## Prompt 11 — filtr E, pula, łańcuch

```
Przeczytaj PLAN_V2.md, podsekcję „Taktyki” i wiersze frontery 9–12.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

E: (claim (eq L R) (dom lo hi) (eval)). hi > lo, oba z 0..10, napisy stron się różnią, co najmniej jedna strona woła rodzinę z piętra 2 albo wyższego. Jedno wstawienie ciała wpisu faktem E nie jest. Test, który to naprawdę sprawdza, bierze slot dodawania z półki. Lewa strona to (call slot (var 0) (var 1)). Prawa powstaje przez jedno wstawienie ciała tego slotu: var podstawione argumentami, a self w ciele zamienione z powrotem na (call slot …). Surowe ciało z gołym self obok wołania nie jest tym testem, bo self poza definicją nie jest prawą stroną równości. Ta para, na dziedzinie szerszej niż jeden punkt, nie zamyka piętra 9. Para następnika z wołaniem jego slotu na (zero) tego nie sprawdza, bo odpada już filtrem „rodzina z piętra 2”. W termie nie ma tokenu „dodawanie” ani „następnik”: slot bierzesz z półki.

W: jedno rewrite, jak w kernelu. Zamyka piętro 10.

C rośnie z faktu E albo W już na półce: doklej jedną taktykę eval albo rewrite, kernel ma przyjąć dłuższy napis, sufit 8 taktyk i 80 tokenów. Iloczynu wszystkich ciągów taktyk nie buduj. Test: przy stałej liczbie wpisów na półce liczba kandydatów o dwóch taktykach rośnie liniowo z liczbą faktów, od których łańcuch startuje. Test: dziewiąta taktyka odpada.

I: (claim (eq L R) (base) (step (ih) (rewrite nazwa))) na szkieletach f(x,0), f(0,x), f(x,1). f i nazwa przebiegają półkę. Zamyka piętro 12, gdy base i step przechodzą.

Na półkę schodzi jeden najkrótszy fakt zamykający piętro oraz co najwyżej osiem dalszych. Do puli najpierw wchodzą fakty z filtrem E, potem W, dopóki miejsc brakuje. Dziewiąty dodatkowy fakt odpada. Kopia faktu już wiszącego odpada. Koniec kształtu bez faktu pisze „schemat wyczerpany” i schodzi fronterą. Ten sam fakt schematu wchodzi na półkę grupy i solo.

Teza, która nie mieści się w 48 tokenach, nie wchodzi do puli. Test.
```

## Prompt 12 — dwie karty i poziomy złożenia

```
Przeczytaj PLAN_V2.md, sekcję „Ćwiczenia”, poziomy 0–3, oraz akapit okna o dwóch nazwach.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Generator najpierw wybiera dwie karty. Ukryty term używa tylko ich identyfikatorów. Poziom 0: jedno call. Poziom 1: dwa wołania. Poziom 2: trzy. Poziom 3: cztery. Test: każda nazwa w termie poziomu 3 jest jedną z dwóch kart. Trzeciej nazwy nie ma. Nie wymagaj, żeby obie karty wystąpiły w tym samym termie.

Zadanie, którego ukryty term przekracza paliwo 256, 4096 węzłów albo 10000 kroków, nie powstaje. Test: generator odrzuca taki zadatek, zanim reduktor liczy całą dziedzinę.

Licznik ćwiczeń rośnie tylko na ćwiczeniu. Mutacja go nie podbija. Ćwiczenie, którego licznik jest podzielny przez 4, kładzie rodzinę zadania na jedną kartę. Pozostałe mają obie karty jako składniki, bez znacznika rodziny w obserwacji. Test: w serializacji obserwacji rundy bez wskazówki nie ma pola ani tokenu, który mówi „rodzina celu”. Numer rundy warsztatu 12 nie jest wskazówką sam z siebie, jeśli licznik ćwiczeń w tym miejscu nie dzieli się przez 4.

Ziarno zadania zawiera numer rundy. Dwie rundy dają dwa różne wejścia. Werdykt reduktora jest policzony, zanim mówca dostaje obserwację. Test: obiekt obserwacji nie zawiera termu ukrytego.

Poziomy 4–6 w tym prompcie tylko odrzucają losowanie, gdy pula taktyk jest pusta. Sufit rampy do tego czasu to 3. Test.
```

## Prompt 13 — runda ćwiczenia, taśma, ε

```
Przeczytaj PLAN_V2.md, resztę sekcji „Ćwiczenia” i sekcję „Taśma i ε”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Runda ćwiczenia ma jednego mówcę i jedną próbę. Sieć wybiera rotacja v1, także gdy wypadnie łamacz. Komunikat jest wyrażeniem albo taktyką. (break …) i (abstain …) na ćwiczeniu są pudłem, nie szukaniem luki. Solo: jedna sieć, jedna próba, własna taśma. Zamrożone wagi: jedna próba, bez śladu i bez kroku.

Trafienie dopisuje ten sam ślad do taśm wszystkich czterech sieci grupy. Ślad to 176 tokenów obserwacji i potem komunikat sprawdzony. Półka po ćwiczeniu ma ten sam zbiór identyfikatorów. Bramka admit v1 nie jest wołana. Test: liczba wpisów półki przed i po trafionym ćwiczeniu jest równa.

Pudło, (abstain …) i komunikat rozjechany na kernelu na taśmę nie wchodzą. Sufit: 256 śladów na agenta i na poziom warsztatu. 257. ślad usuwa najstarszy. Test.

Zegar ε tyka tylko na poziomie, który runda gra. Na ćwiczeniu rola autora rusza zegar autora, a rola użytkownika, obcego albo łamacza rusza zegar wykonawcy. Jedno ćwiczenie nie rusza dwóch zegarów. Zegar łamacza do podłogi 0,5 rusza dopiero na powtórce faktu lokalnego, w prompcie 15. Ta powtórka jest wyjątkiem: łamacz szuka rozjazdu, a mówca rozwiązania ma osobny zegar, więc w tej jednej rundzie tykają dwa.
- Sama pętla schematu, bez rundy ćwiczenia, nie rusza ε. Test: po paczce bez ćwiczenia ε zostaje 1.
- Autor schodzi ×0,995 do podłogi 0,25. Nowy poziom zaczyna od 1.
- Wykonawca schodzi ×0,995 do podłogi 0,05 dopiero od pierwszej swojej odpowiedzi przyjętej na tym poziomie. Test: sama porażka zegara wykonawcy nie rusza.
- Pusta taśma nie rusza wag.

Kroku AdamW jeszcze nie podłączaj, jeśli atrapa kroku wystarcza do testu „wagi zamrożone bitowo te same”.
```

## Prompt 14 — rampa warsztatu

```
Przeczytaj PLAN_V2.md, akapit o progu 0,6 i 0,3.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Oś poziomu jest osobna od frontery. Ostatnie 50 świeżych ćwiczeń grupy na poziomie podnosi poziom o jeden wyłącznie przy sukcesie co najmniej 0,6. Poniżej 0,6, także w pasie od 0,3 do 0,6, poziom stoi. Poziom nie schodzi w dół. Test: 50 sukcesów podnosi; potem seria porażek zostawia poziom na nowym stopniu rampy.

Poziomy 4–6 czekają na niepustą pulę faktów schematu. Wcześniej próba wejścia na 4 zostaje na 3. Sufit to 6. Na suficie, przy dalszych sukcesach, poziom zostaje 6, a losowanie nowego zadania trwa. Test obu.

W trakcie schematu ćwiczenie po paczce jest na najniższym odblokowanym poziomie. Test.
```

## Prompt 15 — mutacja

```
Przeczytaj PLAN_V2.md, sekcję „Mutacja”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Mutacja wstaje, gdy frontera zeszła z piętra 12. Wcześniej rytm zostaje ten z paczki: jedno ćwiczenie na paczkę, mutacji nie ma. Test na frontere 11: po paczce jest ćwiczenie i zero mutacji.

Po zejściu z piętra 12 w każdej dziesiątce rund warsztatu (0–9, 10–19, …) ostatnia jest mutacją: numery 9, 19, 29, …. To jest dziewięć ćwiczeń i jedna mutacja. Runda 0 jest ćwiczeniem. Numery 10, 20, 30 mutacją nie są: w pierwszej dziesiątce byłoby wtedy dziesięć ćwiczeń i mutacja dopiero w następnej. Komputer bierze fakt z puli wspólnej albo z puli lokalnej tej krzywej i robi 64 mutacje. Zapis trafienia jest tylko lokalny. Rodzaje: inna liczba z 0..10, inne złożenie dwóch nazw widocznych w oknie, jedna doklejona taktyka w suficie 8 i 80 tokenów. Kernel ocenia każdą. Listy szkieletów nie ma. Napisu „pąk wyczerpany” nie ma i po 64 mutacjach bez trafienia następna runda mutacji znowu robi 64. Test.

Trafienie, które przechodzi filtr E albo jest dłuższym łańcuchem przyjętym przez kernel i nie jest kopią, wchodzi do puli tylko tej krzywej. Pula wspólna ćwiczenia grupy, solo i wag zamrożonych zostaje przy faktach schematu obecnych na obu półkach. Test: fakt lokalny grupy nie pojawia się w zadaniu solo ani w zbiorze luki.

Powtórka faktu lokalnego jest zadaniem tej krzywej w rundzie mutacji. Łamacz ma cztery próby wskazać punkt, który kernel sam potwierdzi jako rozjazd. Potwierdzone zbicie wchodzi na taśmę łamacza. 64 mutacji łamacz nie przegląda. ε łamacza, ×0,995 do podłogi 0,5, tyka tylko na tej powtórce. Mówca rundy mutacji ma ε = 1, dopóki ta krzywa nie ma własnego faktu lokalnego.
```

## Prompt 16 — zbiór 50 zadań

```
Przeczytaj PLAN_V2.md, sekcję „Zbiór pomiarowy”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Pięćdziesiąt zadań powstaje raz, w chwili gdy frontera schodzi z piętra 8. Wcześniej pliku zbioru nie ma. Test: przy frontere 7 zbiór nie istnieje; po zejściu z 8 istnieje i drugie wywołanie budowniczego nie podmienia termów.

Źródłem są wpisy schematu obecne na obu półkach. Karty są składnikami, bez znacznika rodziny. Test serializacji. Ziarno zbioru jest niezależne od ziarna rundy, ale zbiór nie powstaje przed piętrem 8, nawet jeśli seed jest znany.

Co 50 rund warsztatu zbiór jest liczony bez kroku uczenia. Test: hash wag po tej ewaluacji jest ten sam. Luka jest różnicą sukcesu na tych 50 i jest liczbą w metryce. Kod nie ma progu, który uruchamia piątą sieć. Test: przy luce 2 i przy luce 20 piąta sieć nadal nie istnieje. Kolumna luki może być w nagłówku wcześniej, lecz do zejścia z piętra 8 jest pusta.
```

## Prompt 17 — metryki, snapshot, obrót logu

```
Przeczytaj PLAN_V2.md, sekcję „Metryki”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Wiersz co 10 paczek albo co 10 rund warsztatu ma: numer, fronterę, kształt, numer termu, czy kształt się wyczerpał, poziom warsztatu, świeżą dokładność ostatnich 50 ćwiczeń grupy, solo i wag zamrożonych, rozmiar półki, rozmiar puli wspólnej, rozmiary pul lokalnych, liczbę śladów, zdjęcia ławy.

Osobno zapisz numer paczki wejścia dodawania, mnożenia i pierwszego faktu E oraz znacznik „schemat” albo „mutacja”. Test: sama wielkość półki po przeglądzie 16 tokenów nie ustawia znacznika dodawania.

canon/snapshot.json pisze półkę: identyfikator, nazwa w oknie albo brak, rodzina, dziedzina, ciało, powołania. Do tego pula wspólna, pule lokalne, numer termu, poziom warsztatu. Ciało jest w pliku. W obserwacji agenta tego ciała, poza odcinkiem 48 tokenów gdy plan je tam kładzie, nie ma.

events.jsonl obraca się przy 32 MB. Zostaje bieżący plik i jeden poprzedni. Test na małym progu, podstawionym w teście jako 1 KB: trzeci obrót kasuje najstarszy, zostają dwa pliki. metrics.csv przy obrocie zdarzeń zostaje w całości.
```

## Prompt 18 — zapis i wznowienie

```
Przeczytaj PLAN_V2.md, sekcję „Bieg na wiele dni”, akapit o resume.pt.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Zapis jest atomowy: plik tymczasowy, potem podmiana jednego resume.pt. W środku: wagi, optimizery, taśma przycięta do sufitu, trójka piętro-kształt-numer, poziom warsztatu, pule. W schemacie zapis po każdej paczce. W warsztacie co 50 rund albo co 2 minuty, co pierwsze. Test zegara i test licznika 50.

Przerwanie w trakcie paczki: po wznowieniu numer termu jest tym z ostatniej pełnej paczki, nie numerem z środka. Test symuluje to, zamykając proces atrapą po niepełnej paczce.

Katalog docelowy nazywa się runs/v2-seed0/. Test używa katalogu tymczasowego. Po teście runs/0/resume.pt ma ten sam rozmiar co przed testem. Komentarz w kodzie wolno mieć; katalogiem domyślnym zapisu v2 jest runs/v2-seed0/. Optimizery w zapisie mogą być puste: AdamW dochodzi w prompcie 19.

Komenda v1 python -m kanon.run --seed 0 nadal dopisuje do runs/0 i tego promptu nie używaj do jej odpalenia.
```

## Prompt 19 — krok uczenia v2

```
Przeczytaj PLAN_V2.md, koniec sekcji „Taśma i ε”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Krok v2: AdamW, lr 3e-4, przycięcie 1, batch 4, jeden agent naraz, entropia krzyżowa następnego tokenu. Ciąg to prefiks 176 i komunikat. src/kanon/update.py zostaje przy batch 32, bo to v1; test v1 tego pliku ma zostać zielony.

Produkcja komunikatu startuje z prefiksu 176, nie z pustego ciągu. Test: pierwsze wywołanie sieci przy produkcji dostaje ciąg o długości co najmniej 176. Nie porównuj dwóch wylosowanych tekstów: jedna próbka przy podobnych prefiksach może wyjść taka sama.

Test: ślad przyjęty zmienia wagę agenta, który dostał ślad, i przy osobnych taśmach solo nie zmienia wag grupy, gdy krok solo nie dostał tego śladu. Test: ten sam ślad ćwiczenia grupy zmienia wagi wszystkich czterech, bo taśma grupy dostała cztery kopie. Test: pusta taśma, wagi bitowo te same. Test: stała batcha w kodzie v2 jest równa 4. Nie sprawdzaj tego wyszukiwaniem cyfry 4 w pliku.

Ewaluacja zbioru 50 tej funkcji nie woła.
```

## Prompt 20 — komenda tylko do odczytu

```
Przeczytaj PLAN_V2.md, ostatni akapit sekcji „Bieg na wiele dni”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Osobna komenda ładuje zapis, przyjmuje jeden term albo jedno zdanie i wypisuje werdykt reduktora albo kernela. Nie woła kroku. Nie zmienia półki, numeru kształtu, poziomu warsztatu ani puli.

Test: hash półki, numer termu i poziom przed i po komendzie są równe. Test: werdykt dodawania na (num 2) (num 3) to 5, gdy półka ma dodawanie. Test: zdanie z podwójnym użyciem ih jest odrzucane.

Nie uruchamiaj pętli wielu dni.
```

## Prompt 21 — kopia co 6 godzin

```
Przeczytaj PLAN_V2.md, sekcję „VPS i kopia na GitHubie”. Skrypt wysyłki stoi poza plikami sieci. W plikach sieci v2 nadal nie ma napisu download.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Co 6 godzin, po udanym zapisie lokalnym, proces woła wysyłkę. Test zegara: przy upływie 5 godzin wysyłki nie ma; przy 6 jest jedna.

Paczka wysyłki: snapshot.json, metrics.csv, sześć plików wag (0–3, solo, zamrożone). Nie ma w niej optimizera, taśmy ani resume.pt. Test listy plików.

Wydanie kanon-v2-latest jest podmieniane. Raz na dobę powstaje tag kanon-v2-RRRR-MM-DD bez podmiany. Test na atrapie klienta: dwa wywołania tego samego dnia klepią latest dwa razy i tworzą jeden tag dobowy. Token czytaj ze środowiska. W repozytorium tokenu nie zapisuj. Test: brak tokenu kończy się wpisem w logu i kodem kontynuacji, nie wyjątkiem, który zabija pętlę.

W teście nie łącz się z GitHubem. Atrapa klienta wystarcza.
```

## Prompt 22 — jedno polecenie

```
Przeczytaj PLAN_V2.md, sekcje „Bieg na wiele dni” i „Kolejność kodu”.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

python -m kanon.v2 --seed 0 zostaje w procesie. Najpierw przegląd do 16 tokenów. Gdy czterech rodzin nie ma, proces kończy się błędem i tygodnia nie zaczyna. Potem paczki schematu, potem warsztat. Katalog domyślny: runs/v2-seed0/. Flaga katalogu w teście wskazuje katalog tymczasowy.

Smoke testu nie przechodzi pełnego R2 ani 2000 termów R1. Domyślna paczka procesu, bez flagi testowej, zostaje 2000 termów albo 2 minuty, bez limitu liczby paczek. W teście paczka ma 5 termów, limit to 2 paczki, kształt jest mały i sztuczny. Test: wywołanie bez flagi limitu tego limitu 2 nie ustawia i paczki 5 nie ustawia. Smoke: kod wyjścia 0, powstaje resume w katalogu tymczasowym. Wiersza metryki po 2 paczkach nie wymagaj. Osobny test metryki obniża próg do 1 paczki na tym samym małym kształcie. Nie rób 10 paczek R1. Test gotowości z sufitem 1 kończy proces błędem, zanim powstanie paczka. runs/0/resume.pt nie zmienia rozmiaru.

README: dopisz akapit v2. Jedno polecenie, jeden proces, katalog runs/v2-seed0/. Pętla while z README dla v1 zostaje opisem v1 i nie jest instrukcją v2. Przerwanie gubi co najwyżej niedokończoną paczkę. GPU nie jest wymagane. Piątej sieci nie ma.

Nie odpalaj biegu wielodniowego w tej sesji.
```

---

## Pętla kontrolna

Prompt 23 tylko czyta. Prompt 24 tylko naprawia to, co 23 oznaczyło jako rozjazd. Werdykt „lepiej, przyjęte” prompt 24 pomija.

Odpal 23. Werdykt czytaj z ostatniej linii sekcji „## Kontrola”, nie z końca całego pliku i nie z ostatniego zdania czatu, jeśli sekcja mówi co innego. Gdy ta linia to `WSZYSTKO DOBRZE`, skończ. Gdy `SĄ ROZJAZDY`, odpal 24, potem znowu 23. Powtarzaj, aż sekcja skończy się na `WSZYSTKO DOBRZE`.

## Prompt 23 — zgodność z planem

```
Przeczytaj PLAN_V2.md od deski do deski. Porównaj z src/kanon/v2/ i z testami v2. Nic nie naprawiaj. Nie uruchamiaj biegu wielodniowego. runs/0/ nie otwieraj.

Dla każdego punktu napisz „zgodne” albo „rozjazd” z plikiem i linijką, albo „lepiej, przyjęte” razem z analizą. Analiza „lepiej” ma cztery zdania: co jest inaczej, jaki cel planu to obsługuje, jakie ograniczenie kod poluzował, dlaczego poluzowanie nie psuje wielu dni, paczki, sufitu taśmy, filtru E, dwóch kart, zbioru 50 po piętrze 8 ani zakazu runs/0. Bez tych czterech zdań odchylenie jest rozjazdem, nie ulepszeniem.

Sprawdź po kolei:

1. runs/0/ i python -m kanon.run nie są ścieżką v2. TOKENS ma długość z PLAN.md. if0 bierze gałąź zera. Rotacja jest wzorem r%4, (r+1)%4, (r+2)%4, (r+3)%4.
2. Ziarna 0–3, 4, 5. Piątej sieci nie ma. W plikach sieci v2 nie ma load, from_pretrained, huggingface, download ani słowa payload.
3. Ziarno zadania i ziarno komunikatu zawierają numer rundy.
4. Okno 256, obserwacja dopełniona do 176, komunikat od indeksu 176. (num k) dla k 0..10 w oknie nie jest wieżą succ. Produkcja startuje z prefiksu 176.
5. Budżet 24 / 48 / 80. Term ponad budżet nie wchodzi do reduktora.
6. Przegląd bez self i bez call staje, gdy cztery rodziny leżą, nie później niż na 16 tokenach. Nie wylicza wszystkich if0. Te same ciała wchodzą na grupę i solo. Zamrożone wagi przeglądu nie zapisują. Bez czterech rodzin tydzień nie startuje.
7. Numer termu dekoduje dziury. Listy wszystkich termów R2 nie ma. Paczka to 2000 albo 2 minuty. Zapis po paczce, nie po termie. Metryka co 10 paczek.
8. Piętro 2 schodzi, gdy jest dodawanie, a R2 znalazło odejmowanie albo się wyczerpało. Samo dodawanie, przy nietkniętym R2, nie schodzi. Odejmowanie samo nie zamyka. Dwa następniki nie zamykają. Wyczerpanie pisze „schemat wyczerpany” i schodzi o jedno piętro, bez restartu kształtu.
9. Pięter C1 i C2 nie ma. Mnożenie, porównanie, parzystość, suma, NWD, potęga są piętrami 3–8. Pusta rodzina schodzi.
10. E wymaga rodziny z piętra 2+, dziedziny szerszej niż punkt i nie jest jednym rewrite. Pula ma co najwyżej osiem faktów obok faktu zamykającego. Łańcuch rośnie z faktu. ih podmienia jeden podterm przy add(n, 0) wewnątrz succ.
11. Ćwiczenie używa tylko dwóch kart. Co czwarte ćwiczenie, licząc licznikiem ćwiczeń, ma kartę rodziny. Mutacja tego licznika nie podbija. Pomiar karty rodziny nie ma. Trafienie idzie na taśmę wszystkich czterech i nie woła admit. Sufit taśmy 256 na agenta i poziom.
12. Rampa podnosi się wyłącznie od 0,6. Poniżej 0,6 stoi. Poziom nie spada. Sufit 6. Poziomy 4–6 czekają na pulę.
13. Przed zejściem z piętra 12 mutacji nie ma, ćwiczenie jest jedno na paczkę. Po piętrze 12 mutacja jest w rundach 9, 19, 29, …: dziewięć ćwiczeń i jedna mutacja, 64 sprawdzenia kernela. Stanu wyczerpania mutacji nie ma. Fakt lokalny nie wchodzi do wspólnego zadania ani do luki. Łamacz ma cztery próby na powtórce lokalnej.
14. Zbiór 50 powstaje przy zejściu z piętra 8, raz, bez znacznika rodziny. Wcześniej pliku zbioru nie ma, a kolumna luki jest pusta. Ewaluacja zbioru nie woła kroku.
15. Batch kroku v2 to 4, jeden agent naraz. ε autora ćwiczenia do 0,25. Na ćwiczeniu użytkownik, obcy i łamacz tykają zegar wykonawcy do 0,05, od pierwszego trafienia. Jedno ćwiczenie nie rusza dwóch zegarów. Powtórka lokalna jest wyjątkiem: zegar łamacza do 0,5 i zegar mówcy tykają w tej samej rundzie. Nowe piętro rampy zaczyna ε od 1. Pusta taśma nie rusza wag.
16. resume.pt jest jeden, podmiana atomowa. Warsztat zapisuje co 50 rund albo co 2 minuty. Katalog domyślny runs/v2-seed0/.
17. events.jsonl obraca się, zostają dwa pliki, metrics.csv zostaje. Wysyłka co 6 godzin ma snapshot, metryki i sześć wag, bez resume.pt. Skrypt wysyłki jest poza plikiem sieci. Brak tokenu nie zabija pętli.
18. Komenda odczytu nie zmienia półki ani numeru. Domyślne python -m kanon.v2 nie ma limitu paczek używanego w smoke.

Wynik dopisz do .cursor/work/kanon.md jako nową sekcję „## Kontrola” z datą. Ostatnia linia tej sekcji to dokładnie WSZYSTKO DOBRZE albo dokładnie SĄ ROZJAZDY. Nie oceniaj po ostatniej linii całego pliku: później mogą dojść inne notatki. WSZYSTKO DOBRZE wolno napisać, gdy każdy punkt jest „zgodne” albo „lepiej, przyjęte”. Jeden rozjazd daje SĄ ROZJAZDY.
```

## Prompt 24 — poprawa rozjazdów

```
Przeczytaj ostatnią sekcję „## Kontrola” w .cursor/work/kanon.md i PLAN_V2.md. Jeśli ostatnia linia tej sekcji to WSZYSTKO DOBRZE, nic nie zmieniaj i skończ.

Popraw wyłącznie punkty oznaczone „rozjazd”. Punktów „zgodne” i „lepiej, przyjęte” nie ruszaj i nie „ulepszaj”. Nie dodawaj piątej sieci, nie pobieraj wag, nie otwieraj runs/0/, nie uruchamiaj biegu wielodniowego.

Dla każdego rozjazdu TDD: najpierw test, który pada. Uruchom go i zobacz czerwony wynik, zanim dopiszesz kod. Potem kod. pytest -q ma być zielone, łącznie z testami v1.

Na końcu wypisz, które rozjazdy zeszły. Nie pisz WSZYSTKO DOBRZE. To zdanie należy do następnego przebiegu promptu 23.
```
