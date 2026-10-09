# Kanon

Cztery sieci startują z losowych wag i po każdej rundzie dostają jeden krok AdamW. Nie ma fazy dociągania modelu. Torch jest w `.venv`.

Pusty kanon po smoke nie jest błędem maszynerii. Smoke sprawdza, że proces kończy pracę i pisze log.

## Smoke

Z katalogu repozytorium:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m kanon.run --rounds 20 --seed 0
```

Proces kończy się kodem 0. Zapis ląduje w `runs/0/`.

## runs/

`runs/<seed>/events.jsonl` — jedna runda, jeden obiekt JSON. Numer rundy liczy się od 0. Termy zadań autora, użytkownika i obcego są w `targets`, obok `observations`. W `observations` ich nie ma.

`runs/<seed>/holdout.json` — zbiór pomiarowy zapisany raz, przed rundą 0, z ziarna innego niż ziarno rund.

`runs/<seed>/metrics.csv` — wiersz po 10, 20, 30, … ukończonych rundach. Kolumny: ε, głębokość bieżąca, rozmiar kanonu, rozmiar cmentarza, odsetek zbić łamacza, odsetek zbić dopiero przez audyt, oraz sukces obcego grupy, solo i obcego o losowych wagach na każdej głębokości 1..4. Ewaluacja nie wykonuje kroku uczenia; krok grupy i kroki solo idą po niej. Zgodność z pięcioma parami publicznymi nie jest sukcesem; liczy się 16 par audytowych. Przy ε = 1 komunikat bierze sampler. Poniżej 1 każdy krok produkcji miesza sampler z siecią. Wygasanie i ława idą po kroku uczenia.

`runs/<seed>/canon/snapshot.json` — definicje kanonu grupy i cmentarz. Nie ma w nim termu celu.

`runs/<seed>/weight_check.json` — czy taśma grupy była pusta i czy wagi grupy ruszyły się od startu.

Specyfikacja jest w [PLAN.md](PLAN.md). Środowisko jest w [docs/SETUP.md](docs/SETUP.md).
