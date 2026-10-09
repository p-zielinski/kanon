# Kanon v2

> Status: plan do odczytu rano, bez zmian w biegu v1
> Bieg v1: `runs/0/`, wagi w `runs/0/resume.pt`, odczyt w `runs/0/metrics.csv`
> Prompty: `PROMPTS_V2.md`, po jednym na sesję

v1 zostaje. Polecenie `python -m kanon.run --seed 0` dalej dopisuje jedną rundę do tego samego zapisu, z jednym łamaczem. v2 niczego w `runs/0/` nie przestawia i tego pliku nie nadpisuje.

## Co rano jest już w metrykach

Przy pisaniu tego planu `metrics.csv` miał wiersze 10, 20 i 30. To nie jest decyzja. Rano plik ma być przeczytany od nowa.

Wtedy widać było: głębokość 1, kanon 0, cmentarz 0, łamacz 0, audyt kasuje około 0,90–0,97 kandydatów przepuszczonych dalej, a trzy krzywe na każdej głębokości siedzą między około 0,02 a 0,18. ε jest jeszcze wysoko, więc proponuje głównie sampler. Luki 20 punktów o stałym znaku tam nie ma.

## Jedyna zmiana protokołu

Piąty agent wchodzi tylko jako drugi łamacz. Cztery sieci 0–3 dalej rotują wzorem v1. Piąta, ziarno 6, w każdej rundzie jest łamaczem i niczym innym. Zamrożone wagi zostają ziarnem 4, solo ziarnem 5.

Drugi łamacz dostaje to samo co pierwszy: do czterech `(break slot (num k) …)` albo `(abstain slot)`, bez szkicu i bez deklarowania oczekiwanej wartości. Widzi ciało kandydata, wejścia z cmentarza i to, czy wynik się rozjechał. Termu celu nie widzi. Trafienie któregokolwiek z dwóch łamaczy zdejmuje kandydata i nie otwiera audytu 16. Budżety są osobne: cztery próby i cztery próby, nie jedna ósemka.

Krok uczenia piątej sieci jest jeden na rundę, na tych samych danych co u łamacza: własne zaakceptowane `break` i `abstain`, definicje kanonu, potwierdzone zbicia. Czwórka grupy dalej dostaje po jednym kroku. Solo dalej cztery. Zamrożone wagi dalej zero. To świadomie psuje równość liczby zbić. Tą nierównością jest właśnie interwencja v2. Solo nie dostaje za to dodatkowych propozycji.

Bieg v2 startuje osobno, od rundy 0, w `runs/v2/`. Ma ten sam zbiór pomiarowy co v1 (`holdout_seed` ziarna 0), żeby krzywe leżały na tych samych zadaniach. Checkpoint v2 zapisuje, że łamaczy jest dwóch. Stary `runs/0/resume.pt` tego pola nie ma i dalej znaczy jednego łamacza.

## Bramka

Bramka drugiego łamacza otwiera się tylko wtedy, gdy w `metrics.csv` jest wiersz z rundą co najmniej 100 i `audit_miss_rate` większym niż 0,30, a w `events.jsonl` widać legalne propozycje, nie sam `[ABS]` parsera. Liczenie progu jest na oknie 50 przyjęć do audytu. Krótsze okno przed rundą 100 bramki nie otwiera.

Zamknięta bramka oznacza: drugiego łamacza się nie pisze. Pusty kanon i audyt, który zabija prawie wszystko, to obraz „eksperyment żyje, jeszcze nic nie utrwalił”. Drugi łamacz takiego kanonu nie zapełni. Zabija kandydatów wcześniej. Nie wymyśla definicji, która przechodzi 16 par.

## Pytanie do zapisanego modelu

Osobna funkcja w `kanon.run` ładuje `resume.pt` i zadaje jedno zadanie. Nie woła `step`, nie woła `end_round`, nie dopisuje rundy, nie rusza `resume.pt`, `events.jsonl` ani `metrics.csv`.

Zadanie jest termem Kalka, nie zdaniem. Agent dostaje kanon grupy i pary publiczne. Odpowiedź to `(solve …)`. Sukces to zgodność na 16 parach audytowych. Domyślnie sieć 0 i głębokość zapisana w checkpointcie. ε bierze się z checkpointu. Term celu wolno wypisać człowiekowi na stdout. Do obserwacji agenta nie wchodzi.

## Czego v2 nie robi

- Nie przedłuża v1 ponad 1200 jako sposobu na wynik. Odczyt 600 i 1200 zostaje. Dwa punkty to szum. Luka zaczyna się od około 20 punktów na tej samej głębokości i ten sam znak musi być w obu odczytach.
- Nie dokleja podpowiedzi do obserwacji, także gdy kanon rośnie, a obcy na zbiorze pomiarowym stoi.
- Nie pobiera wag. Nie dodaje języka naturalnego. Nie zmienia `if0`, listy `TOKENS` ani wzoru rotacji czwórki.
- Nie wznawia `runs/0/` kodem dwóch łamaczy.
