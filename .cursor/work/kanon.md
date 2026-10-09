# Kanon

> Last updated: 2026-10-09
> Branch: `main`
> Status: done

## Problem & goal

Cztery plastyczne sieci w zamkniętym języku termów mają zbudować kanon procedur i rozwiązywać głębsze zadania niż sieć solo oraz niż sieć z losowymi wagami czytająca ten sam kanon. „Zrobione” dla v1 oznacza działający smoke 20 rund i metryki z `PLAN.md`, nie wzrost inteligencji.

## Scope

- **In scope:** gramatyka Kalk, reduktor, generator, kanon, role, sampler, transformer od losowych wag, krok uczenia po rundzie, baseline solo i losowe wagi, runner.
- **Out of scope:** wagi z internetu, język naturalny, Python jako język agentów, piąty agent, porównanie z dużym LLM.

## Key files & context

| Area | Path | Role |
|------|------|------|
| Spec | `PLAN.md` | ustalenia i zablokowane parametry |
| Prompty | `PROMPTS.md` | 12 sesji implementacyjnych |
| Setup | `docs/SETUP.md` | venv, torch CPU, zakaz wag |
| Handoff | `.cursor/work/kanon.md` | ten plik |
| Pakiet | `src/kanon/*.py` | pętla, baseline i krok uczenia |
| Język | `src/kanon/language.py` | tokeny, parser, kształty komunikatów |
| Reduktor | `src/kanon/reduce.py` | liczba albo `fuel` / `memory` / `steps` |
| Testy języka | `tests/test_language.py`, `tests/test_reduce.py` | prompt 2 |
| Generator | `src/kanon/tasks.py` | term celu, pary, zbiór pomiarowy |
| Testy generatora | `tests/test_tasks.py` | prompt 3 |
| Kanon | `src/kanon/canon.py` | wpisy, cmentarz, wygasanie, ława |
| Testy kanonu | `tests/test_canon.py` | prompt 4 |
| Role | `src/kanon/roles.py` | rotacja, parser slotu, obserwacja |
| Testy ról | `tests/test_roles.py` | prompt 5 |
| Sampler | `src/kanon/sample.py` | legalne produkcje, budżet 64 tokenów |
| Testy samplera | `tests/test_sample.py` | prompt 6 |
| Sieć | `src/kanon/model.py` | cztery dekodery, maska produkcji, zapis wag |
| Testy sieci | `tests/test_model.py` | prompt 7 |
| Runda | `src/kanon/round.py` | szkice, audyt, przyjęcie, zdarzenie jsonl |
| Testy rundy | `tests/test_round.py` | prompt 8 |
| Krok uczenia | `src/kanon/update.py` | taśma, jeden AdamW, entropia krzyżowa |
| Testy kroku | `tests/test_update.py` | prompt 9 |
| Baselines | `src/kanon/baselines.py` | solo: 4 propozycje i 4 kroki; losowe wagi bez kroku |
| Testy baselines | `tests/test_baselines.py` | prompt 10 |
| Pętla | `src/kanon/run.py` | runda, krok, pomiar co 10, log `runs/` |
| Testy pętli | `tests/test_run.py` | próg głębokości i smoke 20 rund |
| Opis uruchomienia | `README.md` | smoke i układ `runs/` |
| Smoke | `tests/test_smoke_env.py` | import torch i kształt wagi `(8, 8)` |

## Decisions

- 2026-10-09: cztery agenty plus automat, role rotowane slotem komunikatu.
- 2026-10-09: wdzięczność = wywołanie `call`, krytyka = wejście, na którym automat widzi rozjazd.
- 2026-10-09: PyTorch tylko jako autograd. Sieć 4×128, losowy start, krok AdamW po każdej rundzie.
- 2026-10-09: zimny start przez sampler legalnych termów i ε od 1.0 do podłogi 0.05.
- 2026-10-09: duży model językowy wchodzi najwcześniej po trzech krzywych (grupa, solo, losowe wagi).
- 2026-10-09: odczyt człowieka w rundach 20, 100, 600 i 1200. Log co 10 rund. Sukces = 16 par audytowych, nie 5 publicznych. Luka poniżej około 20 punktów na 50 zadaniach to szum.
- 2026-10-09: „myślenie” = 1–4 szkice termu sprawdzone na 5 parach publicznych przed odpowiedzią, ten sam budżet we wszystkich krzywych. Szkice nie wchodzą do taśmy uczącej.
- 2026-10-09: `(if0 cond zero-branch nonzero-branch)`. Zero bierze pierwszą gałąź, bo tylko wtedy def dodawania z PLAN.md na `(num 2)` `(num 3)` daje 5. Prompt 2 podaje dla dwóch przykładowych termów odwrotne wyniki 0 i 1.
- 2026-10-09: `TOKENS` jest dokładnie listą z PLAN.md. `propose`, `use`, `solve`, `break`, `abstain` parsują się jako głowy komunikatów, ale nie wchodzą do `TOKENS` — gramatyka ról ich wymaga, a słownik ich nie wymienia.
- 2026-10-09: `(use …)` bez `(call slot …)` jest `[ABS]` w parserze slotu i nie zamyka sesji. Runda i tak wymaga tego wywołania przy przyjęciu do kanonu.
- 2026-10-09: `(call …)` dostaje świeże paliwo 256. `(self …)` zużywa 1 z bieżącego wywołania. 256 zejść `pred` przechodzi, 257. kończy się `fuel`. Limit węzłów liczy AST oraz rozwinięcie `(num k)`. Ewaluacja idzie stosem na stercie.
- 2026-10-09: `build_holdout(seed)` jest czystą funkcją ziarna, 50 zadań na głębokości 1..4. `run` zapisuje ten zbiór raz do `runs/<seed>/holdout.json` przed rundą 0. Rundy go nie czytają.
- 2026-10-09: widok agenta (`TaskView`) ma głębokość, arność i 5 par publicznych. Term celu jest tylko na `TaskAudit.target`. Obserwacja to `[PAIRS]` i `(num k)`.
- 2026-10-09: głębokość 1 to identyczność, succ, pred i stałe 0, 1, 2. Głębokość 2 dokłada złożenie dwóch takich konstrukcji oraz dodawanie i odejmowanie ograniczone. Głębokość 3 i 4 przy pustym kanonie to kolejne złożenie succ/pred, bez `call` i bez nowego `self`. Niepusty kanon na głębokości 3 i 4 daje `(call slot …)` do funkcji z kanonu.
- 2026-10-09: pary publiczne są pięcioma różnymi wejściami z 0..6. Audyt ma 16 wejść z 0..10, rozłącznych ze zbiorem publicznych. Przy arności 1 dziedzina 0..10 ma 11 wartości, więc 16 wejść audytowych musi się powtarzać.
- 2026-10-09: wpis kanonu startuje z `last_used_round = born_round`. Znika, gdy `runda >= last_used_round + 12`. Ława w rundach 4, 8, 12, … redukuje tylko najstarszy wpis na zapisanych parach audytowych. `end_round` najpierw wygasza, potem robi ławę. Zajęty slot albo 33. definicja zwraca `None` i nie nadpisuje wpisu. Eksport łamacza to wejście i `diverged`, bez termu targetu.
- 2026-10-09: `assign(r)` daje autor `r%4`, łamacz `(r+1)%4`, użytkownik `(r+2)%4`, obcy `(r+3)%4`. Odrzut parsera slotu to dokładnie `[ABS]` i nie zużywa szkicu. Szkic autora to `(draft (def …))`, użytkownika i obcego `(draft expr)`. Zła arność daje pięć `(err)` i zużywa próbę. Po czterech szkicach kolejny `draft` jest `[ABS]`, a `propose` / `use` / `solve` przechodzą. `propose` nie woła `canon.admit`. Łamacz ma cztery `(break …)`; `(abstain …)` zamyka sesję także po tym budżecie.
- 2026-10-09: obserwacja to `[CANON] … [SEP] [PAIRS] (num k) …`, potem dopisek roli i marker. Autor dostaje jeden wolny slot, łamacz ciało kandydata, użytkownik samą nazwę slotu. Obcy nie dostaje ciała odrzuconego kandydata ani termu z `TaskAudit.target`. Odpowiedź `[TRY]` liczy wyłącznie pięć par publicznych.
- 2026-10-09: sampler (`sample(marker, observation, seed=…)`) losuje jednostajnie produkcję, która jeszcze mieści się w 64 tokenach i domyka komunikat. Autor: szkic albo `propose` po co najmniej jednym `[TRY]`, po czterech `[TRY]` tylko `propose`; ciało bez `num`, `call` tylko do slotów z kanonu, slot i arność z obserwacji. Użytkownik woła wskazanego kandydata. Obcy woła tylko sloty kanonu. Łamacz: `(abstain slot)` albo `(break slot (num k)…)` z `k` z 0..10, bez szkicu. `speak` przy ε = 1 woła ten sampler. Poniżej 1 każdy krok produkcji z prawdopodobieństwem ε bierze sampler, inaczej wagę z logitów sieci na tokenach, które tę produkcję odróżniają.
- 2026-10-09: `build_agents()` daje cztery dekodery, ziarna 0..3, 4 warstwy, d_model 128, 4 głowy, d_ff 512, dropout 0. `distribution(ids, legal)` zwraca rozkład następnej produkcji i stawia 0 tam, gdzie maska jest fałszywa. Słownik to `TOKENS` plus głowy komunikatów. Zapis to `agent-0.bin`…`agent-3.bin` w podanym katalogu. Pętla rundy nie jest podłączona.
- 2026-10-09: `play` przy ε = 1 bierze sampler. Poniżej 1 wymaga `choose`. Skrypt wstrzykuje komunikaty. Audyt 16 wejść startuje dopiero po przyjętym `propose`. Do kanonu wchodzi definicja niezbita, zgodna z targetem na audycie i wywołana w `use`, które samo przechodzi audyt zadania użytkownika. Zbicie łamacza idzie na cmentarz i audytu nie otwiera. Obcy startuje po `admit`: zadanie i obserwacja biorą kanon już z tą definicją, bez termu celu i bez transkryptu sporu. `targets` w zdarzeniu trzyma termy autora, użytkownika i obcego obok obserwacji. `play` nie woła kroku uczenia.
- 2026-10-09: taśma (`Tape`) trzyma zaakceptowane finały z rolą agenta, aktualny eksport kanonu i, tylko dla roli łamacza, zbicia z rozjazdem. Szkic i odrzut nie wchodzą. `step` to jeden AdamW (lr 3e-4, batch 32, przycięcie 1.0, entropia krzyżowa następnego tokenu, ciąg do 64). Pusta taśma pomija krok i nie tworzy optimizera. Jedna taśma, jeden optimizer na sieć. Ewaluacja tej funkcji nie woła.
- 2026-10-09: `solo_round` to jedna sieć, cztery propozycje autora, szkice 1..4 przez ten sam `SlotSession`, przyjęcie tylko gdy 16 par audytowych się zgadza. Brak łamacza i użytkownika. Potem cztery wywołania `step`. Na taśmę idzie tylko przyjęty `propose`. `start_frozen` trzyma ziarno i jedną sieć obcego. `frozen_eval` czyta kanon grupy, nie woła `step` i nie zapisuje kanonu.
- 2026-10-09: `run` przed rundą 0 buduje zbiór pomiarowy z `seed + 1_000_003`, zapisuje `holdout.json` i buduje zestaw rozwojowy 10 zadań na głębokość z `seed + 2_000_006`. Pętla woła `play` i `solo_round` z `close=False`, notuje taśmę, a gdy wypada pomiar — `score_tasks` bez `step`. Potem `step_all`, cztery kroki solo, i dopiero `end_round` na kanonie grupy i solo. Pomiar używa `speak` oraz wag obcego grupy, sieci solo i sieci zamrożonej. ε startuje w 1 i maleje ×0,995 do podłogi 0,05. Głębokość rośnie po dwóch ewaluacjach powyżej 0,6, sufit 4. Losowe wagi (ziarno 4) nie dostają kroku. Log: `events.jsonl`, `metrics.csv`, `holdout.json`, `weight_check.json`, `canon/snapshot.json`. `(num k)` spoza tokenów `0`–`3` rozwija się w `encode` do wieży `succ`.

## Progress

- [x] Specyfikacja, prompty, setup
- [x] Prompt 1 — szkielet. Środowisko stoi: Python 3.12, torch 2.14.1 (indeks CPU), pytest 9.1.1. `pytest -q`: 1 passed.
- [x] Prompt 2 — parser i reduktor Kalk. `pytest -q`: 20 passed.
- [x] Prompt 3 — generator zadań i zbiór pomiarowy. `pytest -q`: 28 passed.
- [x] Prompt 4 — kanon, cmentarz, wygasanie, ława. `pytest -q`: 37 passed.
- [x] Prompt 5 — role, rotacja, budżet szkiców, obserwacja. `pytest -q`: 51 passed.
- [x] Prompt 6 — sampler legalnych termów. `pytest -q`: 61 passed.
- [x] Prompt 7 — cztery dekodery od losowych wag. `pytest -q`: 67 passed.
- [x] Prompt 8 — jedna runda i reguły kanonu. `pytest -q`: 77 passed.
- [x] Prompt 9 — krok uczenia po rundzie. `pytest -q`: 81 passed.
- [x] Prompt 10 — baselines solo i losowe wagi. `pytest -q`: 83 passed.
- [x] Prompt 11 — runner, metryki, smoke. `pytest -q`: 87 passed.
- [x] Prompt 12 — przegląd zgodności. Lista niżej.
- [x] Poprawka rozjazdów z pierwszego przeglądu. `pytest -q`: 95 passed.
- [x] Prompt 12 ponownie, na kodzie po poprawce. Lista niżej.
- [x] Poprawka trzech rozjazdów z drugiego przeglądu: cmentarz w obserwacji łamacza, krok przed wygasaniem, ε na produkcjach. `pytest -q`: 99 passed.
- [x] Prompt 12 trzeci raz, na kodzie po tej poprawce. Lista niżej. Kodu nie zmieniano.
- [x] Obcy startuje po `admit`. `pytest -q`: 100 passed.
- [x] Prompt 12 po tej poprawce. Lista niżej. Nowych rozjazdów nie było.
- [x] Start i stop: domyślnie jedna runda, kontynuacja z `resume.pt`, `--fresh` usuwa zapis.

## Next steps

1. Rano, gdy `runs/0/metrics.csv` ma wiersze, odpal po jednym promptcie z `PROMPTS_V2.md`. Najpierw odczyt. Piątego agenta dodaje dopiero otwarta bramka.

## Blockers / open questions

- Import torch wypisuje ostrzeżenie, że nie ma NumPy. Test przechodzi. NumPy nie jest w `requirements.txt` i nie wchodzi w zakres promptu 1.

## Verification

- 2026-10-09: `.venv` z Python 3.12 (`/opt/homebrew/bin/python3.12`; systemowy `python3` to 3.9 i nie spełnia `docs/SETUP.md`).
- `pip install torch --index-url https://download.pytorch.org/whl/cpu` oraz `pip install pytest`.
- `python -c "import torch; ..."`: `2.14.1 (8, 8)`.
- `pytest -q`: 1 passed (`test_torch_import`). Ostrzeżenie o braku NumPy, bez porażki.
- 2026-10-09: `pytest -q` po promptcie 2: 20 passed. To samo ostrzeżenie NumPy ze smoke torcha.
- 2026-10-09: `pytest -q` po promptcie 3: 28 passed.
- 2026-10-09: `pytest -q` po promptcie 4: 37 passed.
- 2026-10-09: `pytest -q` po promptcie 5: 51 passed. To samo ostrzeżenie NumPy ze smoke torcha.
- 2026-10-09: `pytest -q` po promptcie 6: 61 passed. To samo ostrzeżenie NumPy.
- 2026-10-09: `pytest -q` po promptcie 7: 67 passed. To samo ostrzeżenie NumPy.
- 2026-10-09: `pytest -q` po promptcie 8: 77 passed. To samo ostrzeżenie NumPy.
- 2026-10-09: `pytest -q` po promptcie 9: 81 passed. To samo ostrzeżenie NumPy.
- 2026-10-09: `pytest -q` po promptcie 10: 83 passed. To samo ostrzeżenie NumPy.
- 2026-10-09: `pytest -q` po promptcie 11: 87 passed. Smoke 20 rund, seed 0, kończy się kodem 0. To samo ostrzeżenie NumPy.
- 2026-10-09: drugi przebieg promptu 12 porównał PLAN.md z kodem po poprawce. Testów nie odpalano. Kodu nie zmieniano.
- 2026-10-09: poprawka trzech rozjazdów z drugiego przeglądu. `pytest -q`: 99 passed. To samo ostrzeżenie NumPy.
- 2026-10-09: trzeci przebieg promptu 12 porównał PLAN.md z kodem po tej poprawce. Testów nie odpalano. Kodu nie zmieniano.
- 2026-10-10: obcy liczy `solve` na kanonie po `admit`. `pytest -q`: 100 passed. To samo ostrzeżenie NumPy.
- 2026-10-10: czwarty przebieg promptu 12. Nowych rozjazdów nie było. Kodu poza tą poprawką nie zmieniano.
- 2026-10-10: jedno wywołanie `kanon.run` dopisuje rundy do `runs/<seed>/resume.pt`. Domyślnie jedną. `--fresh` kasuje zapis i zaczyna od rundy 0.

## Przegląd zgodności

> 2026-10-10. Czwarty przebieg promptu 12, po przesunięciu obcego za `admit`. `pytest -q`: 100 passed.

### Lista kontrolna

- **zgodne.** Brak pobierania wag. `requirements.txt:2-3` ma torch i pytest. `tests/test_model.py:20` zabrania w źródle sieci napisów `load`, `from_pretrained`, `huggingface`, `download`. W `src/kanon/model.py` tych napisów nie ma.
- **zgodne.** Cztery niezależne zestawy wag, ziarna 0, 1, 2, 3. `src/kanon/model.py:27`, `89-90`. Solo i sieć zamrożona to osobne baseline (`src/kanon/run.py:55-56`, `208`, `211`), nie piąty agent grupy.
- **zgodne.** Po rundzie grupy jest jeden krok na sieć (`src/kanon/run.py:283`), potem wygasanie (`286-287`). Solo robi cztery kroki przed `end_round` (`src/kanon/baselines.py:85-89`; w runnerze `src/kanon/run.py:284-287`).
- **zgodne.** Ewaluacja pomiarowa nie woła `step` (`src/kanon/run.py:256-281` jest przed `283`). `frozen_eval` kroku nie woła (`src/kanon/baselines.py:102-119`). Koniec `run` sprawdza hash wag z ziarna 4 (`src/kanon/run.py:292-293`).
- **zgodne.** Obserwacja nie czyta termu celu (`src/kanon/roles.py:129-148`). Termy autora, użytkownika i obcego są w `targets`, obok `observations` (`src/kanon/round.py:93-94`, `250-254`).
- **zgodne.** `(num k)` w ciele def jest błędem parsera (`src/kanon/language.py:365-367`). Ciało losowane dla autora nie oferuje `num` (`src/kanon/sample.py:95-129`). `num` zostaje w parach, w `[TRY]`, w argumentach `break` i we wejściach cmentarza.
- **zgodne.** Rotacja jest wzorem z planu: autor `r%4`, łamacz `(r+1)%4`, użytkownik `(r+2)%4`, obcy `(r+3)%4` (`src/kanon/roles.py:91-95`). Strzałka PLAN.md:158 daje inną ścieżkę osobistą niż ten wzór. Kod trzyma wzór.
- **zgodne.** Do kanonu wchodzi definicja niezbita, zgodna na 16 parach audytowych i wywołana w `use`, które samo te pary przechodzi (`src/kanon/round.py:205-213`). `use` bez `(call slot …)` jest `[ABS]` i nie zamyka sesji (`src/kanon/roles.py:226-232`).
- **zgodne.** Wpis znika, gdy `runda >= last_used_round + 12` (`src/kanon/canon.py:22`, `104`).
- **zgodne.** Ława w rundach podzielnych przez 4, od rundy 4, tylko najstarszy wpis (`src/kanon/canon.py:23`, `110-118`). `end_round` najpierw wygasza (`121-122`).
- **zgodne.** ε startuje w 1, ×0,995, podłoga 0,05 (`src/kanon/run.py:51-52`, `117-120`, `217`, `290`). Przy ε = 1 `speak` oddaje sampler (`src/kanon/sample.py:484-485`). Poniżej 1 każdy krok produkcji bierze sampler z prawdopodobieństwem ε, inaczej wagę z logitów na tokenach odróżniających tę produkcję (`src/kanon/sample.py:279-298`, `300-312`, `492`). Pomiar podaje to samo ε (`src/kanon/run.py:274-280`, `333-360`).
- **zgodne.** Sieć: 4 warstwy, d_model 128, 4 głowy, d_ff 512, dropout 0, ziarna 0..3 (`src/kanon/model.py:21-27`). AdamW, lr 3e-4, batch 32, przycięcie 1,0 (`src/kanon/update.py:37-39`, `164-194`). Długość generacji 64 (`src/kanon/sample.py:22`).
- **zgodne.** `propose`, `use` i `solve` wymagają co najmniej jednego szkicu i nie przechodzą piątego (`src/kanon/roles.py:48`, `201-202`, `221`, `227`, `237`). Po czterech `[TRY]` sampler autora zostawia sam `propose` (`src/kanon/sample.py:151-154`).
- **zgodne.** Szkic redukuje się na pięciu parach publicznych sesji (`src/kanon/roles.py:164-165`, `209-217`). Szesnaście par audytowych wchodzi dopiero przy audycie (`src/kanon/round.py:185-188`).
- **zgodne.** Szkic nie wchodzi na taśmę (`src/kanon/update.py:71-72`). Pętla notuje komunikat, gdy odpowiedź rundy to `accepted` (`src/kanon/run.py:398-402`).
- **zgodne.** Grupa, solo i pomiar losowych wag używają tego samego `SlotSession` z budżetem 1..4 (`src/kanon/roles.py:48`). Solo ma cztery propozycje (`src/kanon/baselines.py:26`, `77-82`). Pomiar obcego ma ten sam limit szkiców (`src/kanon/run.py:146-150`).

### Dalsze punkty

- **zgodne.** Przy zerze `if0` bierze drugą gałąź, pole `zero_branch` (`src/kanon/reduce.py:117-119`, `src/kanon/language.py:78-81`). Przykład dodawania PLAN.md:113-114 bierze wtedy `(var 1)`.
- **zgodne.** `(num k)` spoza tokenów `0`–`3` rozwija się w `encode` do wieży `succ` (`src/kanon/update.py:127-144`). Reduktor `(num k)` zwraca liczbę `k` (`src/kanon/reduce.py:154-157`). `(num 0)` … `(num 3)` zostają tokenami.
- **zgodne.** Zbiór pomiarowy jest plikiem `runs/<seed>/holdout.json`, pisanym raz przed rundą 0 (`src/kanon/run.py:203-204`, `453`). 50 zadań na głębokość 1..4 (`src/kanon/tasks.py:40`, `90-97`). Rundy używają `task_seed` (`src/kanon/round.py:102-104`).
- **zgodne.** Pary publiczne: 5 różnych wejść, współrzędne 0..6. Audyt: 16 wejść, współrzędne 0..10, rozłączne ze zbiorem publicznych (`src/kanon/tasks.py:36-39`, `112-119`, `231-243`). Zadania mają arność 1 albo 2 (`110`).
- **zgodne.** Głębokość 1 to identyczność, succ, pred i stałe 0, 1, 2 (`src/kanon/tasks.py:170-182`). Głębokość 2 dokłada złożenie oraz dodawanie i odejmowanie ograniczone (`162-167`). Głębokość od 3 woła kanon, a przy pustym kanonie składa prymitywy (`157-159`). Próg wzrostu: dwie ewaluacje powyżej 0,6, sufit 4, zestaw rozwojowy osobny od pomiaru (`src/kanon/run.py:45-50`, `124-131`, `274-281`, `380-387`).
- **zgodne.** Paliwo 256, węzły 4096, kroki 10000 (`src/kanon/reduce.py:26-28`, `93-95`, `142`). `(self …)` zużywa 1 (`127`). `(call …)` dostaje nowe paliwo (`139`). `pred` z zera daje zero (`115`).
- **zgodne.** Smoke 20 rund, ziarno 0 (`src/kanon/run.py:58-59`, `195`). Obserwacje nie zawierają ciągu `hidden` (`tests/test_run.py:246`). Wagi grupy ruszyły się albo taśma jest pusta (`tests/test_run.py:258-259`). Gdy taśma coś ma, a wagi stoją, `run` kończy się błędem (`src/kanon/run.py:294-301`).
- **zgodne.** Łamacz dostaje z cmentarza wejścia i znacznik rozjazdu, bez termu celu (`src/kanon/roles.py:145-146`, `252-258`; `src/kanon/canon.py:155-158`). `(err …)` gdy wynik się rozjechał, `(zero …)` gdy się zgadzał.
- **zgodne.** Krok jest przed wygasaniem. Runner zamyka kanon po `step_all` i krokach solo (`src/kanon/run.py:283-289`). `play` i `solo_round` domyślnie nadal zamykają kanon; pętla woła je z `close=False` (`src/kanon/round.py:118`, `230-231`; `src/kanon/baselines.py:73`, `83-90`).
- **zgodne.** Odsetek zbić audytu jest w `metrics.csv` na oknie 50 (`src/kanon/run.py:57`, `242-243`, `363-365`). Progu 30% kod nie zamienia na drugiego łamacza.
- **zgodne.** Szkic autora jest `(draft (def …))` (`src/kanon/roles.py:68`, `src/kanon/sample.py:157-158`). Tak ma gramatyka PLAN.md:94, tabela :153 i budżet :193. Zdanie PLAN.md:173 pisze w tym miejscu `(draft expr)`.
- **zgodne.** PLAN.md:121-127 zamyka listę tokenów i nie ma w niej `propose`, `use`, `solve`, `break`, `abstain`. `TOKENS` tej listy nie rozszerza (`src/kanon/language.py:21-50`). Te słowa są w `MESSAGE_HEADS` (`language.py:19`) i w słowniku sieci (`src/kanon/model.py:28`), bo gramatyka roli ich wymaga.
- **zgodne.** Najpierw warunek kanonu, potem obcy. `admit` jest w `src/kanon/round.py:205-213`. Zadanie i obserwacja obcego biorą kanon po tym przyjęciu (`214-223`). `solve` liczy się na tym kanonie (`224-230`, `408-410`). Wygasanie idzie później (`234-235`).
