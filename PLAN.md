# Kanon

> Ostatnia aktualizacja planu: 2026-10-09
> Status: specyfikacja v1, bez implementacji
> Cel dokumentu: wierny zapis ustaleń z rozmowy oraz decyzje dopięte tak, żeby dało się to zbudować bez dalszych pytań.

## Po co to jest

Pytanie eksperymentu: czy cztery małe sieci, startujące z losowych wag i zamknięte w języku termów, zbudują kanon procedur, dzięki któremu rozwiązują głębsze zadania niż sieć ucząca się sama i niż sieć, która kanon widzi, ale wag nie aktualizuje.

To nie jest próba zbudowania AGI. To jest mikrospołeczność z plastycznymi wagami. Wynik ma być liczbą na zadaniach, których agenty nie widziały.

Motyw z rozmowy, który tu zostaje:

- Ludzkie społeczności rosną, bo myśl odkleja się od osoby. Zostaje procedura, którą ktoś inny naprawdę użył, albo kontrprzykład, który ją zabił.
- Wdzięczność jest użyciem. Krytyka jest kontrprzykładem. Uprzejme zdanie nie ma gdzie wejść, bo kanał nie przyjmuje zdań.
- Uczenie na opinii drugiego modelu zjeżdża w dół, gdy nie ma sprawdzianu. Uczenie przy automacie, który redukuje term do liczby, ma szansę iść w górę.
- Duży model z internetu już umie arytmetykę. Wtedy eksperyment mierzyłby odtwarzanie, nie budowanie.
- Zamrożenie wag po treningu jest poza harmonogramem. Krok uczenia jest częścią rundy.

## Teza

Po N rundach obcy (agent oceniany na zamrożonym pomiarze, bez zapisu wag) rozwiązuje więcej zadań ze zbioru trzymanego z boku niż:

1. ten sam obcy z losowymi wagami, czytający ten sam kanon,
2. pojedyncza sieć po tylu samo krokach uczenia i tylu samo wywołaniach automatu, bez ról łamacza i użytkownika.

Jeśli krzywa grupy nie odkleja się od obu baz, protokół społecznej selekcji w tej skali nic nie dodaje. To jest uczciwy wynik negatywny, nie powód, żeby dorzucić model z internetu.

## Poza zakresem v1

- Pobieranie wag (Llama, GPT, Hugging Face, Ollama, GGUF).
- Własny silnik mnożenia macierzy, własna implementacja CUDA, własny autograd.
- Język naturalny w którąkolwiek stronę po starcie procesu.
- Python jako język agentów. Komentarze, stringi i nazwy byłyby kanałem mowy oraz wiedzą z pretreningu.
- Lean i biblioteka standardowa. To byłby cudzy kanon wgrany na starcie.
- Piąty agent. Wchodzi dopiero w v2, i tylko jako drugi łamacz, jeśli automat regularnie zabija definicje, które łamacz przepuścił.
- Porównanie z dużym modelem językowym. Dopiero gdy trzy krzywe (grupa, solo, losowe wagi) istnieją.
- Twierdzenie, że wzrost kanonu jest inteligencją ogólną. Wzrost jest głębszą składaniem w jednym formalnym świecie.

## Architektura

```
cztery sieci (osobne wagi, losowy start)
        |  termy zgodne ze slotem roli
        v
   harnaś rundy
        |  legalny term
        v
   automat redukcji  --->  liczba albo błąd (limit kroków / pamięci)
        |
        +--> kanon (dysk)
        +--> cmentarz (dysk)
        +--> powtórka akceptowanych śladów
        |
        v
   jeden krok uczenia na każdej z czterech sieci
```

Pożyczony jest wyłącznie automat różniczkowania (PyTorch). Agent, gramatyka, reduktor, kanon, rotacja ról i reguła aktualizacji są nasze.

Cztery agenty istnieją po to, żeby w jednej rundzie cztery role robiły różną robotę. Pamięć wspólna jest w kanonie, nie w wagach. Dzięki temu krok, który nadpisze wagę, nie kasuje od razu dorobku. Powtórka z kanonu co rundę trzyma starą umiejętność przy życiu.

| Byt | Trzymany w | Dzielony |
|---|---|---|
| Polityka składania termów | wagi sieci | nie, cztery zestawy |
| Procedury, które przetrwały | kanon na dysku | tak |
| Zbite hipotezy i wejścia, które je zbiły | cmentarz | tak, widoczny dla łamacza jako lista wejść |
| Zadania pomiarowe | plik wygenerowany raz z ziarna | nikt ich nie używa w rundach |

## Język Kalk

Jedyny język agentów. Term redukuje się do liczby naturalnej. Rozmowa jest ciągiem termów.

### Składnia

```
expr ::=
    (zero)
  | (succ expr)
  | (pred expr)
  | (if0 expr expr expr)
  | (var n)                 ; n całkowite, 0 .. arność-1
  | (call slot expr ...)    ; slot to f0 .. f31
  | (self expr ...)         ; rekurencja, legalna tylko w ciele def

def  ::= (def slot arity expr)

num  ::= (num k)            ; wyłącznie w danych: pary IO i argumenty łamacza
                            ; k to liczba dziesiętna >= 0
                            ; rozwija się do succ^k (zero)
                            ; w ciele def jest nielegalne

draft-autora     ::= (draft def)          ; slot = jedyny wolny slot z obserwacji
draft-reszty     ::= (draft expr)         ; użytkownik i obcy; arność bierze się z zadania
propose          ::= (propose def)
use              ::= (use slot expr)      ; expr musi zawierać (call slot ...)
solve            ::= (solve expr)
```

Nazwy spoza `f0`–`f31` nie istnieją. Rozwiązanie użytkownika i obcego jest wyrażeniem, nie definicją kanonu.

Słoty kanonu są zamknięte: `f0` … `f31`. Słownik modelu dzięki temu nie rośnie. Wygasła definicja zwalnia slot i slot może wrócić.

`pred` z zera daje zero. `if0` bierze gałąź „else”, gdy warunek redukuje się do zera, w przeciwnym razie gałąź „then”.

`(self ...)` kosztuje 1 jednostkę paliwa. Paliwo startuje od 256 na każde wywołanie z zewnątrz. Przy zerze redukcja kończy się błędem `fuel`. Limit rozmiaru termu w trakcie redukcji: 4096 węzłów, błąd `memory`. Limit kroków redukcji: 10000, błąd `steps`. Błąd oznacza, że definicja nie przechodzi testu.

### Przykład, którego agenty nie dostają jako podpowiedzi

Dodawanie, zapisane tylko po to, żeby człowiek czytający plan wiedział, że język jest dostateczny:

```
(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))
```

Generator zadań umie zbudować takie termy sam. Żaden prompt agenta tego nie zawiera.

### Słownik tokenów

Zamknięty. Nic spoza listy nie jest tokenem.

```
( ) zero succ pred if0 var call self def arity num draft err
f0 f1 ... f31
0 1 2 3          ; indeksy zmiennych, arność (arność maksymalna 3)
[AUTHOR] [BREAK] [USE] [OUTSIDER] [PAIRS] [CANON] [SEP] [ABS] [TRY]
```

Tokeny w nawiasach kwadratowych są markerami slotu i sekcji obserwacji. To nie są zdania. Obserwacja jest ciągiem tokenów z tej listy, budowanym przez harnaś.

## Obserwacja

Harnaś składa obserwację. Agent nie dostaje `hidden_term`.

Wspólny rdzeń:

```
[CANON] (def f0 2 ...) [SEP] (def f3 1 ...) [SEP]
[PAIRS] (num ...) ...
```

Potem marker roli: `[AUTHOR]`, `[BREAK]`, `[USE]` albo `[OUTSIDER]`.

Łamacz dodatkowo dostaje ciało definicji będącej pod oceną. Użytkownik dostaje nazwę slotu, który wolno mu wywołać, jeśli definicja jest kandydatem (jeszcze nie w kanonie). Obcy dostaje wyłącznie kanon zatwierdzony oraz pary swojego zadania pomiarowego albo zadania rundy przeznaczonego dla obcego. Obcy nie dostaje transkryptu sporu.

## Role

Rola jest kształtem komunikatu, który parser slotu przyjmie w tej rundzie. Komunikat innego kształtu jest odrzucany. Harnaś nie odsyła wyjaśnienia. Odsyła token `[ABS]` i idzie dalej.

| Rola | Legalny komunikat | Skutek |
|---|---|---|
| Autor | `(draft (def slot arity expr))`, potem `(propose (def slot arity expr))` | szkic sprawdzany na 5 parach; propose jest kandydatem do kanonu |
| Łamacz | `(break slot (num k) ...)` albo `(abstain slot)` | próba zbicia albo wstrzymanie; te 4 wejścia są już jego budżetem szukania |
| Użytkownik | `(draft expr)`, potem `(use slot expr)` | szkic na jego 5 parach; expr musi wołać slot |
| Obcy | `(draft expr)`, potem `(solve expr)` | szkic na 5 parach zadania; solve idzie do metryki |

Rotacja co rundę o jeden: autor → łamacz → użytkownik → obcy → autor. Po czterech rundach każdy był każdą rolą. Graf „zawsze tych dwóch przeciw sobie” się nie zestala.

Przypisanie ról na rundzie `r` (indeksy agentów 0..3):

```
autor        = r % 4
łamacz       = (r + 1) % 4
użytkownik   = (r + 2) % 4
obcy         = (r + 3) % 4
```

## Runda

1. Generator losuje zadanie autora: ukryty term głębokości `d`, 5 par publicznych, 16 wejść audytowych niewidocznych dla agentów.
2. Generator losuje inne zadanie użytkownika tej samej głębokości.
3. Autor składa od 1 do 4 szkiców `(draft expr)`, a potem `(propose ...)`. Slot musi być wolny albo wskazywany przez harnaś (harnaś podaje jeden wolny slot w obserwacji jako jedyny, którego `propose` może użyć). `propose` przed pierwszym szkicem jest odrzutem `[ABS]`. Po czwartym szkicu legalny jest już tylko `propose`.
4. Łamacz dostaje do 4 prób wejść. Automat liczy `def(wejście)` i `target(wejście)`. Różnica oznacza trafione zbicie: kandydat odpada, para ląduje na cmentarzu.
5. Jeśli łamacz wstrzyma się albo żadne z jego wejść nie rozróżnia, automat puszcza 16 świeżych wejść. Jedna różnica kasuje kandydata.
6. Użytkownik składa od 1 do 4 szkiców `(draft expr)`, potem `(use slot expr)`. Wyrażenie musi zawierać `(call slot ...)`. Automat sprawdza `use` na świeżych wejściach jego zadania. Brak wywołania, brak szkicu albo błąd testu oznacza, że kandydat nie wchodzi.
7. Warunek wejścia do kanonu, wszystkie naraz:
   - niezbity przez łamacza,
   - zgodny z targetem na 16 wejściach audytowych,
   - wywołany w rozwiązaniu użytkownika, które samo przechodzi audyt.
8. Obcy składa od 1 do 4 szkiców i `(solve expr)` na jednym zadaniu ze zbioru bieżącego (nie ze zbioru pomiarowego), widząc tylko kanon. To daje szybką krzywą w logu. Zbioru pomiarowego nie tyka.
9. Każda z czterech sieci dostaje jeden krok na powtórce.
10. Wygasanie i powrót na ławę, według zasad niżej.

Fałszywy kontrprzykład nie istnieje jako osobna wiadomość tekstowa. Łamacz nie deklaruje oczekiwanej wartości. Podaje tylko wejście. Oczekiwanie bierze automat z ukrytego targetu. Dzięki temu łamacz nie dogaduje się z autorem co do „prawdy”.

## Budżet szkiców

To jest odpowiednik tego, co przy dużych modelach nazywa się myśleniem. Tam „myślenie” jest dłuższym ciągiem tokenów przed odpowiedzią końcową. Wagi w trakcie pytania stoją. Dodatkowy koszt idzie w przeszukanie tekstu. Ślad po polsku albo po angielsku bywa przy tym opowieścią obok obliczenia, a nie samym obliczeniem.

U nas szkic jest termem, a automat go redukuje. Ślad jest wierny, bo da się go wykonać.

- Autor, użytkownik i obcy: minimum 1 szkic, maksimum 4, potem komunikat, który się liczy. Szkic autora jest `(draft (def ...))` o arności zadania. Szkic użytkownika i obcego jest `(draft expr)` o arności zadania. Inna arność daje pięć razy `(err)` i zużywa jedną próbę.
- Po każdym szkicu harnaś dopisuje do obserwacji `[TRY]` i pięć wyników na parach publicznych: `(num k)` albo `(err)`, gdy redukcja padła na paliwie, pamięci albo kroku.
- Pary audytowe i ukryty term w tej odpowiedzi nie występują. Szkic, który trafia 5 par publicznych, może dalej polec na audycie.
- Łamacz szkiców nie składa. Jego cztery wejścia są już szukaniem.
- Szkice nie wchodzą do kanonu i nie wchodzą do taśmy uczącej. Klonuje się tylko zaakceptowany komunikat końcowy.
- Grupa, solo i losowe wagi mają ten sam budżet 1..4. Inaczej krzywa z dłuższym szukaniem wygrałaby bez lepszych wag.
- Wymuszenie dotyczy obowiązku jednego sprawdzenia przed oddaniem, nie treści szkicu i nie jego długości. Minimalna liczba tokenów w szkicu produkowałaby puste `succ` dopisane dla limitu.

Osobny bieg grupy z budżetem 0, tym samym ziarnem, czytany w rundzie 600, mówi, czy szkice w ogóle podniosły krzywą. Do zakończenia v1 w rundzie 1200 nie jest potrzebny.

## Kanon, cmentarz, wygasanie

Wpis kanonu:

```
slot
def
public_pairs          ; podzbiór pokazany przy przyjęciu
audit_pairs           ; 16 par z przyjęcia, do powtórnego sprawdzenia
last_used_round
born_round
```

Ukryty term targetu zostaje w logu audytowym eksperymentu, poza obserwacją agentów.

Cmentarz trzyma: slot-kandydat, def, wejście, wynik def, wynik targetu, numer rundy. Łamacz widzi wejścia i to, że wynik się rozjechał, bez target-term.

Wygasanie: brak wywołania przez 12 rund (trzy pełne obroty) usuwa wpis. Slot wraca do puli. Wpis wypada też z powtórki.

Powrót na ławę: co 4 rundy najstarszy wpis jest sprawdzany ponownie na swoich `audit_pairs`. Rozjazd usuwa wpis. Łamacz w tej samej rundzie i tak szuka nowego zbicia kandydata; ława dotyczy tylko starego wpisu i robi ją automat, nie model.

## Generator zadań

Generator jest programem. Nie jest modelem.

- Głębokość 1: identyczność, `succ`, `pred`, stała 0, 1, 2.
- Głębokość 2: złożenie dwóch konstrukcji głębokości 1 oraz prymitywna rekurencja w stylu dodawania i odejmowania ograniczonego.
- Głębokość 3 i wyżej: złożenie termu głębokości `d-1` z funkcją, która już jest w kanonie, a gdy kanon jest pusty, głębsza rekurencja na prymitywach.

Głębokość bieżąca rośnie o 1, gdy obcy na zadaniach bieżącej głębokości (osobny, mały zestaw rozwojowy, nie zbiór pomiarowy) przekroczy 0,6 sukcesu przez dwie kolejne ewaluacje. Sufit v1: głębokość 4.

Zadania mają arność 1 albo 2. Wejście to tyle liczb, ile wynosi arność. Pary publiczne: 5 losowych wejść, każda współrzędna z zakresu 0..6. Wejścia audytowe: 16 losowych, współrzędne 0..10, rozłączne z publicznymi. Ziarno generatora jest zapisane.

Zbiór pomiarowy powstaje raz, zanim ruszy pierwsza runda, z innego ziarna. 50 zadań na głębokość, głębokości 1..4. Żadna runda ich nie używa. Ewaluacja co 10 rund nie aktualizuje wag.

## Sieć

Polityka wyborów produkcji gramatyki, nie swobodny generator tekstu.

- Transformer dekodujący: 4 warstwy, `d_model` 128, 4 głowy, `d_ff` 512, dropout 0.
- Cztery niezależne zestawy wag. Inicjalizacja losowa, cztery różne ziarna (0, 1, 2, 3).
- Brak wspólnego checkpointu bazowego.
- Dekodowanie ograniczone gramatyką: w każdym kroku legalne są tylko produkcje, które parser slotu jeszcze może dokończyć. Sieć daje logity nad produkcjami.
- Zimny start: przy losowych wagach rozkład jest prawie płaski, więc harnaś miesza sampler. Z prawdopodobieństwem `ε` produkcję losuje sampler jednostajny spośród legalnych, w przeciwnym razie bierze próbkę z sieci. `ε` startuje od 1,0, mnoży się przez 0,995 co rundę, podłoga 0,05.
- Sampler jednostajny jest częścią eksperymentu. Bez niego pierwsze tysiące rund to same odrzuty parsera i zero sygnału.

Długość generacji: maksymalnie 64 tokeny. Niedomknięty term jest odrzutem.

## Krok uczenia

Dzieje się po każdej rundzie, dla każdej sieci, jeden krok AdamW.

Dane powtórki (wspólna taśma, próbkowanie zależne od roli, którą agent właśnie grał):

- zaakceptowane komunikaty końcowe tego agenta z przeszłości (`propose`, `use`, `solve`, `break`, `abstain`), bez szkiców,
- definicje obecnie w kanonie, jako ciągi tokenów,
- dla łamacza: potwierdzone zbicia z cmentarza.

Odrzucony komunikat nie wchodzi do klonowania. Strata to entropia krzyżowa następnego tokenu. Przycięcie gradientu: 1,0. Learning rate: 3e-4. Batch: 32 ciągi, obcięte do 64 tokenów.

Gdy taśma jest pusta (pierwsze rundy), krok jest pomijany. Wagi wtedy zostają, jakie są. Od pierwszej akceptacji krok jest obowiązkowy.

To jest celowo skromne uczenie: klonowanie tego, co automat przepuścił, plus powtórka kanonu. Plastyczność jest prawdziwa (wagi po rundzie są inne). Nie obiecuje wynalezienia analizy matematycznej. Obiecuje, że da się zmierzyć, czy selekcja społeczna zmienia krzywą.

Ewaluacja pomiarowa nie wykonuje kroku. Pętla rund po ewaluacji robi krok tak samo jak wcześniej. Osobnego „trybu readonly” nie ma w harmonogramie.

## Baselines

Ta sama gramatyka, ten sam automat, ten sam budżet:

| Krzywa | Wagi | Kto proponuje | Krok uczenia | Kanon |
|---|---|---|---|---|
| grupa | 4 sieci | 4 role | 4 kroki na rundę | pełne reguły przyjęcia |
| solo | 1 sieć | tylko autor, 4 propozycje na rundę | 4 kroki na rundę | wchodzi to, co przejdzie 16 wejść audytowych, bez łamacza i użytkownika |
| losowe-wagi | wagi nigdy nie ruszane, architektura obcego | obcy czyta kanon zbudowany przez grupę | brak | kanon grupy |

Porównanie grupy z „losowe-wagi” mówi, ile wyniku siedzi w tekście kanonu, a ile w wagach. Porównanie z solo mówi, ile dokładają łamacz i użytkownik.

Liczba wywołań automatu na rundę jest zrównana (rząd 4 pełnych sprawdzeń plus audyty). Inaczej grupa wygrałaby samym budżetem sprawdzania.

## Metryki

Co 10 rund, bez aktualizacji wag, na zbiorze pomiarowym:

- sukces obcego grupy, osobno per głębokość,
- sukces solo,
- sukces obcego z losowymi wagami na kanonie grupy,
- rozmiar kanonu,
- rozmiar cmentarza,
- odsetek kandydatów zbitych przez łamacza,
- odsetek kandydatów zbitych dopiero przez audyt (fałszywe przepuszczenia łamacza),
- `ε`.

Próg „fałszywe przepuszczenia są częste”: audyt zabija więcej niż 30% kandydatów przepuszczonych przez łamacza, liczone na 50 kolejnych przyjęć do audytu. To jest jedyny sygnał, że w v2 warto dodać drugiego łamacza. v1 go nie dodaje. v1 zapisuje ten odsetek.

Log: `runs/<seed>/events.jsonl`, `runs/<seed>/metrics.csv`, `runs/<seed>/canon/`.

Jedno zadanie pomiarowe jest rozwiązane, gdy term obcego zgadza się z ukrytym targetem na wszystkich 16 parach audytowych tego zadania. Zgodność z 5 parami publicznymi nie liczy się jako sukces. Ewaluacja nie dopisuje tych zadań do taśmy uczącej.

Smoke, który ma przejść w CI lokalnym: 20 rund, ziarno 0, CPU. Asercje: proces kończy się, wagi grupy różnią się od inicjalizacji albo taśma jest wciąż pusta (dopuszczalne na 20 rundach, ale krok po sztucznie wrzuconym śladzie musi zmienić wagę — to jest test jednostkowy, nie smoke), obserwacja zserializowana nie zawiera ciągu `hidden`.

## Jak czytać bieg

Prompty 1–12 budują instrument. Smoke 20 rund sprawdza, że instrument kończy pracę i pisze log. Pusty kanon po 20 rundach jest dopuszczalny. Wynik naukowy zaczyna się dopiero na dłuższym biegu.

Zapis do `metrics.csv` idzie co 10 rund. Człowiek nie czyta co 10 rund. Są cztery spojrzenia, ustalone z góry:

| Runda | Po co |
|---|---|
| 20 | Maszyneria: pliki logu istnieją, obserwacje bez ukrytego termu, proces żyje. |
| 100 | Lejek, nie ranking. Ile komunikatów odrzucił parser, ile zbił łamacz, ile zbił audyt, ile odpadło brakiem `call`, ile weszło do kanonu. |
| 600 | Pierwszy odczyt krzywych. `ε` jest już przy podłodze 0,05, więc proponuje głównie sieć, nie sampler. |
| 1200 | Drugi odczyt, te same krzywe. Koniec v1. Biegu się nie przedłuża, bo „jeszcze chwila i ruszy”. |

`ε` spada 0,995 razy na rundę. Z 1,0 do 0,05 schodzi około rundy 600. Wcześniejszy sukces na głębokości 1 jest zwykle trafieniem samplera, które kanon zapisał.

Na 50 zadaniach błąd jednej proporcji przy wyniku w okolicy połowy to około 7 punktów procentowych. Luka między dwiema krzywymi zaczyna coś znaczyć od około 20 punktów na tej samej głębokości, i ten sam znak musi być w odczycie 600 i 1200. Dwa punkty to szum.

Czytanie krzywych, osobno dla każdej głębokości:

- Grupa wyraźnie nad solo i nad losowymi wagami: protokół ról coś dodaje, i wagi umieją z kanonu korzystać.
- Grupa i losowe wagi razem, solo niżej: wynik siedzi w tekście kanonu. Sieć grupy nie nauczyła się nic ponad to, co da się przeczytać z zeszytu.
- Wszystkie trzy razem: albo kanon pusty, albo procedury w nim nie przenoszą się na zbiór pomiarowy.
- Głębokość 1 wysoko, 3 i 4 na zerze: składanie nie wyszło poza prymitywy. To jest wynik, nie awaria runnera.

Lejek z rundy 100 mówi, czy w ogóle warto czekać do 600. Kanon pusty i prawie same odrzuty parsera: sampler nie składa legalnych termów, błąd maszynerii. Kanon pusty, termy legalne, wszystko zabija audyt: prymitywy nie trafiają w target, eksperyment żyje, tylko jeszcze nic nie utrwalił. Kanon rośnie, a obcy na zbiorze pomiarowym stoi: procedury przechodzą własne testy i nie przenoszą się dalej. Tego ostatniego nie wolno leczyć dopisaniem podpowiedzi do obserwacji.

## Układ repozytorium

```
PLAN.md                 ten dokument
PROMPTS.md              prompty implementacyjne, po jednym na sesję
docs/SETUP.md           środowisko; Grok wykonuje to sam
README.md               jak uruchomić, gdy kod już jest
.cursor/work/kanon.md   handoff
src/kanon/language.py   tokeny, parser, kształty komunikatów
src/kanon/reduce.py     reduktor
src/kanon/tasks.py      generator i zbiór pomiarowy
src/kanon/canon.py      kanon, cmentarz, wygasanie, ława
src/kanon/roles.py      sloty, rotacja, składanie obserwacji
src/kanon/sample.py     sampler legalnych produkcji
src/kanon/model.py      transformer, 4 zestawy wag
src/kanon/update.py     powtórka i jeden krok
src/kanon/round.py      jedna runda
src/kanon/baselines.py  solo i losowe wagi
src/kanon/run.py        pętla, logi, ewaluacja
tests/                  po jednym pliku na moduł powyżej
```

## Parametry v1 (zablokowane)

| Parametr | Wartość |
|---|---|
| warstwy / d_model / głowy / d_ff | 4 / 128 / 4 / 512 |
| dropout | 0 |
| sloty kanonu | 32 (`f0`–`f31`) |
| arność maksymalna | 3 |
| paliwo / kroki / węzły | 256 / 10000 / 4096 |
| pary publiczne / audyt | 5 / 16 |
| próby łamacza | 4 |
| wygasanie | 12 rund bez wywołania |
| ława | co 4 rundy, najstarszy wpis |
| lr / batch / grad clip | 3e-4 / 32 / 1.0 |
| ε | 1.0 → ×0.995 / runda → podłoga 0.05 |
| ewaluacja | co 10 rund |
| szkice przed odpowiedzią | minimum 1, maksimum 4, tylko pary publiczne |
| zbiór pomiarowy | 50 zadań × głębokości 1..4 |
| głębokość start / sufit | 1 / 4 |
| próg wzrostu głębokości | 0.6 przez dwie ewaluacje |
| smoke | 20 rund, seed 0 |
| ziarna wag | 0, 1, 2, 3 |
| optymalizator | AdamW |

## Ryzyka, których nie wolno „naprawiać” dopiskiem po polsku

- Losowa sieć długo nie włoży nic do kanonu. Na to jest sampler i `ε`, nie podpowiedź treści definicji.
- Kanon może skostnieć. Na to jest wygasanie i ława.
- Dwie sieci o wspólnych ślepych plamkach kanonizują błąd. Audyt 16 świeżych wejść jest tu ważniejszy od kolejnego modelu.
- Klonowanie akceptowanych termów nagradza to, co już przeszło, i słabo nagradza nowość. Dlatego `ε` nie spada do zera.
- 20 rund smoke może dać pusty kanon. Smoke sprawdza maszynerię. Krzywe czyta się w rundach 600 i 1200, według sekcji „Jak czytać bieg”.

## Skąd się wzięły te liczby agentów

Dwa agenty zostawiają użytkownika wewnątrz sporu, więc zmowa „ja ciebie nie złamię” jest stabilna. Trzy domykają spór autorem, łamaczem i użytkownikiem, ale nikt z zewnątrz nie mierzy kanonu. Cztery role są dokładnie tym, co w jednej rundzie robi różną robotę. Tłum kopii tego samego modelu utrwala wspólny błąd pewniej, niż go znajduje. Piąty agent jest drugim łamaczem i tylko wtedy, gdy metryka fałszywych przepuszczeń przekroczy próg z sekcji Metryki.
