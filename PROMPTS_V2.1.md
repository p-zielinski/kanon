# Prompty v2.1

Plan zostaje `PLAN_V2.md`. Kod v1 i `PROMPTS_V2.md` zostają. Te dwa prompty dopinają do polecenia `python -m kanon.v2` to, czego plan wymaga, a bieżący bieg nie robi: wypowiedź w ćwiczeniu i krok AdamW na przyjętym śladzie.

Odpalaj **po jednym**. Najpierw prompt 1. Gdy jego testy są zielone i długi bieg nie został przez niego uruchomiony, odpal prompt 2. Nie łącz ich w jednej sesji.

Wspólna rama:

- Kod v2 stoi w `src/kanon/v2/`. `python -m kanon.run` i `runs/0/` zostają ścieżką v1: nie czytaj ich, nie zapisuj, nie migruj `resume.pt`.
- Nie pobieraj wag. Nie dodawaj huggingface, transformers, ollama.
- Krotka `TOKENS` w `src/kanon/language.py` zostaje. Nowe słowa są w słowniku v2.
- `if0`: zero bierze gałąź zera. Rotacja: autor `r%4`, łamacz `(r+1)%4`, użytkownik `(r+2)%4`, obcy `(r+3)%4`.
- Ziarna: grupa 0–3, zamrożone 4, solo 5. Piątej sieci nie ma.
- W `src/kanon/v2/model.py` nie ma podciągów `load`, `from_pretrained`, `huggingface`, `download` ani słowa `payload`.
- Testy: `.venv/bin/python -m pytest -q` z katalogu repozytorium. NumPy nie dochodzi.
- Długi bieg to `PYTHONPATH=src python -m kanon.v2 --seed 0` bez `--max-packages` i bez `--shape-count`. Prompt 1 go nie odpala. `_stay` nie jest wołane z testu.
- Flagi `--max-packages` i `--shape-count` zostają atrapą kształtu `T`. Nie zamieniaj ich w skan R1 i nie rób z nich dowodu uczenia.
- Sześć dziur przy arności 1 zostaje. Poziomów 4–6 nie buduj. Wysyłka zostaje przy atrapie, która nic nie publikuje.
- `ask` i ocena zbioru 50 kroku nie wołają.
- Po promptcie dopisek w `.cursor/work/kanon.md`. Commit tylko, gdy użytkownik o niego poprosi.

---

## Prompt 1 — wypowiedź i krok w tym samym poleceniu

```
Przeczytaj PLAN_V2.md, sekcje „Ćwiczenia”, „Taśma i ε” i „Bieg na wiele dni”. Potem src/kanon/v2/command.py, play.py, learn.py, resume.py, workshop.py, ramp.py i record.py. Nie otwieraj runs/0/.

TDD: najpierw testy wypisane niżej. Uruchom je, zanim dopiszesz kod tego zachowania, i zobacz, że padają. Potem kod. Na końcu pytest -q, razem z testami v1.

Dziś _workshop_round i _one_exercise podają message="(abstain f0)". learn.step nie jest wołane z polecenia. _one_exercise buduje świeżą taśmę i świeże zegary, więc ślad ginie przed zapisem. _metric wpisuje slady=0. _remember bierze wagi raz, na starcie. Drugie python -m kanon.v2 nie woła Session.open i składa półkę od przeglądu, więc trójka z resume.pt leży na innej półce niż ta, przy której powstała.

To ma być ten sam proces, nie druga komenda. Przy schemacie sieć produkcji nie wybiera termu. Ćwiczenie mówi. Trafienie idzie na taśmę. Pusta taśma nie rusza wag.

Wypowiedź ćwiczenia

Nowa funkcja, w pliku v2 poza model.py, składa jeden komunikat jednej próby.

- Obserwacja to task.observation.tokens, długość 176. Ukrytego task.term ta funkcja nie czyta.
- Mówi nazwami okna f0 i f1, nie identyfikatorem półki. play._rename już mapuje f0 i f1 na dwie karty zadania.
- Komunikat jest wyrażeniem albo taktyką, w budżecie ćwiczenia 80 tokenów, domkniętym nawiasem. (abstain …) i (break …) nie są wynikiem tej funkcji.
- Ziarno to round_seed(seed, round_no, attempt). Grupa bierze attempt 0, solo 1, zamrożone 2.
- Przy ε >= 1 sieci nie wołaj. Losowanie idzie po legalnych złożeniach dwóch kart z okna, tej samej głębokości co poziom zadania: poziom 0 to jedno (call …), poziom 1 dwa, poziom 2 trzy, poziom 3 cztery. Argumenty to (var i) w zakresie arności albo (num k) dla k z 0..10. Dwa numery rundy dają dwa komunikaty, które nie są tym samym napisem na każdym ziarnie.
- Przy ε < 1 pierwszy kontakt z siecią to forward_prefix na całym prefiksie 176. Dalej każdy token: z prawdopodobieństwem ε legalny wybór samplera, w przeciwnym razie rozkład sieci na masce tokenów, które jeszcze domykają wyrażenie w budżecie. learn.produce zwraca jeden argmax i samo nie jest komunikatem. Pętla staje na domkniętym wyrażeniu albo na budżecie.
- Zamrożone wagi dostają próbę przy ε = 1, samplerem, bez forwardu i bez kroku.
- kanon.sample.speak zostaje samplerem v1. Ćwiczenie v2 go nie woła: tamten sampler mówi definicją i abstainem, a werdykt ćwiczenia przyjmuje wyrażenie.

Krok

- Jedna taśma grupy, jedna taśma solo, jedne zegary i sześć sieci żyją przez cały _stay. Ćwiczenie paczki ich nie tworzy od nowa.
- W schemacie zadanie jest na poziomie 0, także gdy rampa stoi wyżej. Po piętrze 12 zadanie jest na bieżącym poziomie rampy.
- Trafienie grupy dopisuje ten sam ślad do czterech taśm i woła learn.step po jednym agencie 0–3, na taśmie tego agenta i tego poziomu. Batch zostaje 4, lr 3e-4, przycięcie 1.
- Trafienie solo woła krok tylko sieci 5, na taśmie solo. Pudło, abstain i komunikat rozjechany na reduktorze albo kernelu taśmy nie dostają i kroku nie wołają.
- Sieć 4 po ćwiczeniu jest bitowo tą samą, co przed nim.
- Rampa notuje trafienie grupy. W schemacie losowanie i tak zostaje na poziomie 0.
- Optimizer AdamW siedzi w session.optimizers i w resume.pt. Krok bierze ten optimizer, nie słownik, który umiera z procesem. Stary zapis z pustymi optimizerami dostaje nowego AdamW na wczytanych wagach.
- Przed session.write wagi w zapisie są żywym state_dict sześciu sieci. Krok sprzed zapisu widać po Session.open.

Ślad ze schematu

Gdy _place albo settle przyjmie term albo fakt, na taśmę grupy wchodzi ten napis z obserwacją rodziny: ciało w odcinku 48 tokenów, gdy się mieści. Potem krok czterech sieci grupy. Forward produkcji na tej ścieżce nie idzie. Solo dostaje kartę na półkę, tak jak dziś, i taśmy ze schematu nie dostaje. Abstain na tę taśmę nie wchodzi.

Zapis i ciąg dalszy

Gdy w katalogu biegu jest resume.pt i canon/snapshot.json, start woła Session.open i składa półkę grupy oraz solo z ciał w snapshocie, w tej kolejności, bez drugiego przeglądu. Trójka piętro, kształt, numer zostaje. Dodawanie obecne w snapshocie jest na półce po wczytaniu. Brak resume.pt zostawia dzisiejszy przegląd do 16 tokenów i start od piętra 2. Stary resume bez klucza zegarów startuje ε od 1. Zegary wolno dopisać do tego samego payloadu, żeby restart nie cofał ε.

Metryka

Kolumna slady to liczba śladów taśmy grupy, nie stałe 0. Świeża dokładność grupy, solo i zamrożonych bierze prawdziwe ostatnie trafienia, nie puste krotki.

README

Zastąp zdania, że komunikat ćwiczenia to (abstain f0), że krok nie idzie i że drugie uruchomienie nie wczytuje resume.pt. Zostaw w pliku te napisy, bo test je wymaga: pętla while true; do PYTHONPATH=src python -m kanon.run --seed 0 || exit; done, python -m kanon.v2 --seed 0, runs/v2-seed0/, GPU nie jest wymagane, Piątej sieci nie ma, Ta pętla jest opisem v1.

Testy, katalog tymczasowy, bez _stay i bez runs/v2-seed0

- Po trafionym ćwiczeniu waga agenta 0 różni się od świeżego build_net(0), a waga ziarna 4 jest bitowo ta sama. Pusta taśma i pudło zostawiają wagę 0 bitowo tą samą.
- Dwa numery rundy przy ε = 1 dają dwa różne komunikaty. Żaden nie jest napisem ukrytego termu podanym wprost do funkcji wypowiedzi. W 200 rundach na półce z identycznością i następnikiem, poziom 0, choć jeden komunikat reduktor przyjmuje.
- Przyjęcie dodawania ze schematu dopisuje ślad i rusza wagę grupy. forward_prefix na tej ścieżce nie jest wołany.
- Katalog z resume.pt o trójce (3, R1, 100) i snapshotem, którego półka ma ciało dodawania, wczytuje index 100 i local_id("dodawanie"). Przegląd przy tym wczytaniu nie idzie.
- Zapis po jednym trafieniu, otwarty nowym Session, ma wagę różną od świeżego ziarna.
- W src/kanon/v2/command.py nie ma literału "(abstain f0)" jako komunikatu ćwiczenia ani rundy warsztatu.

Nie odpalaj python -m kanon.v2 bez flag limitu. Jeśli taki proces już żyje, nie zabijaj go i nie startuj drugiego. Na końcu pytest -q i dopisek w .cursor/work/kanon.md: wypowiedź i krok są w poleceniu, długi bieg tym promptem nie wystartował.
```

---

## Prompt 2 — odpalenie i pilnowanie, aż wagi się ruszą

```
Przeczytaj PLAN_V2.md, sekcję „Taśma i ε”, oraz dopisek promptu 1 w .cursor/work/kanon.md. Prompt 1 ma być już w kodzie. Jeśli w command.py ćwiczenie nadal podaje "(abstain f0)" albo learn.step nie jest na ścieżce ćwiczenia i schematu, wróć do promptu 1. Tego promptu nie zaczynaj od implementacji.

Nie otwieraj runs/0/. Nie pobieraj wag. Nie commituj.

Start

Sprawdź, czy żyje już proces python -m kanon.v2 bez --max-packages i bez --shape-count. Jeśli żyje, nie zabijaj go i nie startuj drugiego. Zakończ i napisz, że bieg już idzie.

Gdy nie żyje, z katalogu repozytorium, w tle:

source .venv/bin/activate
PYTHONPATH=src python -m kanon.v2 --seed 0

Katalog to runs/v2-seed0/. Nie dawaj --max-packages ani --shape-count. Nie kasuj tego katalogu: resume.pt i canon/snapshot.json są postępem schematu. Ostrzeżenie torcha o braku NumPy nie jest awarią i na standardowe wyjście nic więcej lecieć nie musi.

Co jest uczeniem

Uczenie zaszło dopiero, gdy po starcie tego procesu spełnione są oba warunki:

- w resume.pt taśma grupy ma choć jeden ślad, albo wiersz metrics.csv dopisany po starcie ma slady większe od zera
- waga ziarna 0, 1, 2 albo 3 w tym resume.pt różni się od świeżego build_net tego ziarna

Ziarno 4 zostaje bitowo świeżym build_net(4). Ręczna podmiana state_dict, krok na pudle i krok na abstain uczeniem nie są. Dokładność, która rośnie, bo sampler dostał ukryty term, uczeniem nie jest.

Jak pilnować

Patrz na pliki, nie na milczenie stdout. Paczka trwa do 2000 termów albo 2 minut. Metryka schodzi co 10 paczek, więc snapshot i czas zapisu resume.pt są wcześniejsze niż nowy wiersz.

Pudło ćwiczenia zostawia pustą taśmę i te same wagi. To jest zgodne z planem. Nie traktuj pierwszych pustych paczek jako błędu.

Błąd jest, gdy półka ma dodawanie, po starcie zapisały się co najmniej trzy pełne paczki, a w kodzie ścieżki, która właśnie szła, komunikat nadal jest stałym abstainem albo learn.step w ogóle nie jest wołane. Wtedy zatrzymaj ten proces, popraw tylko tę ścieżkę, zostaw runs/v2-seed0/ i uruchom to samo polecenie jeszcze raz. Wczytanie resume.pt ma utrzymać trójkę i półkę.

Gdy komunikaty są prawdziwymi wyrażeniami, krok jest podłączony, a przez sześć zapisanych paczek po zamkniętym piętrze taśma dalej jest pusta, nie wklejaj ukrytego termu do samplera i nie obniżaj ε do zera. Sprawdź testem z promptu 1, że w 200 rundach przy ε = 1 choć jedno wyrażenie pada. Gdy ten test pada, popraw support samplera i uruchom bieg ponownie. Gdy ten test przechodzi, bieg jest zgodny z planem i jeszcze nie trafił: zostaw proces włączony.

Granica czekania to 90 minut albo 30 zapisanych paczek od startu, co pierwsze. W tej granicy albo widzisz oba warunki uczenia, albo zostawiasz żywy proces i piszesz, ile paczek zeszło, jaka jest trójka, ile śladów leży na taśmie i że ziarno 4 stoi.

Koniec

Gdy oba warunki uczenia zaszły i ziarno 4 stoi, procesu nie zatrzymuj. W .cursor/work/kanon.md dopisz trójkę, liczbę śladów i to, że waga grupy ruszyła się krokiem, a ziarno 4 nie. Napisz to samo w odpowiedzi.
```
