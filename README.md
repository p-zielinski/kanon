# Kanon

Cztery sieci startują z losowych wag i po każdej rundzie dostają jeden krok AdamW. Nie ma fazy dociągania modelu. Torch jest w `.venv`.

Pusty kanon po smoke nie jest błędem maszynerii. Smoke sprawdza, że proces kończy pracę i pisze log.

## Jedna runda

Z katalogu repozytorium:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m kanon.run --seed 0
```

To dopisuje jedną rundę do `runs/0/` i kończy pracę. Następne takie samo polecenie idzie dalej: runda 1, 2, 3, … Stan wag, kanonu i taśmy jest w `runs/0/resume.pt`. Log, który da się czytać, jest obok.

Na noc, aż do Ctrl-C:

```bash
source .venv/bin/activate
while true; do PYTHONPATH=src python -m kanon.run --seed 0 || exit; done
```

Przerwanie w trakcie rundy gubi tylko tę rundę. Ostatnia dokończona zostaje. Rano widać ją w `runs/0/events.jsonl` i, co 10 rund, w `metrics.csv`.

Usunięcie zapisu i start od rundy 0:

```bash
PYTHONPATH=src python -m kanon.run --fresh --seed 0
```

`--rounds 20` dopisuje dwadzieścia rund za jednym razem. Smoke, od zera:

```bash
PYTHONPATH=src python -m kanon.run --fresh --rounds 20 --seed 0
```

## runs/

`runs/<seed>/events.jsonl` — jedna runda, jeden obiekt JSON. Numer rundy liczy się od 0. Termy zadań autora, użytkownika i obcego są w `targets`, obok `observations`. W `observations` ich nie ma.

`runs/<seed>/holdout.json` — zbiór pomiarowy zapisany raz, przed rundą 0, z ziarna innego niż ziarno rund.

`runs/<seed>/metrics.csv` — wiersz po 10, 20, 30, … ukończonych rundach. Kolumny: ε, głębokość bieżąca, rozmiar kanonu, rozmiar cmentarza, odsetek zbić łamacza, odsetek zbić dopiero przez audyt, oraz sukces obcego grupy, solo i obcego o losowych wagach na każdej głębokości 1..4. Ewaluacja nie wykonuje kroku uczenia; krok grupy i kroki solo idą po niej. Zgodność z pięcioma parami publicznymi nie jest sukcesem; liczy się 16 par audytowych. Przy ε = 1 komunikat bierze sampler. Poniżej 1 każdy krok produkcji miesza sampler z siecią. Wygasanie i ława idą po kroku uczenia.

`runs/<seed>/canon/snapshot.json` — definicje kanonu grupy i cmentarz. Nie ma w nim termu celu.

`runs/<seed>/weight_check.json` — czy taśma grupy była pusta i czy wagi grupy ruszyły się od startu.

Specyfikacja v1 jest w [PLAN.md](PLAN.md). Plan v2 i prompty na rano są w [PLAN_V2.md](PLAN_V2.md) i [PROMPTS_V2.md](PROMPTS_V2.md). Środowisko jest w [docs/SETUP.md](docs/SETUP.md).
