# Kanon

Bieżący bieg to v2. Sześć sieci startuje z losowych wag: cztery w grupie, jedna solo i jedna zamrożona. Jedno polecenie zostaje w jednym procesie. GPU nie jest wymagane. Piątej sieci nie ma. Torch jest w `.venv`.

## Uruchomienie

Z katalogu repozytorium:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m kanon.v2 --seed 0
```

Katalog to `runs/v2-seed0/`. Najpierw przegląd termów do 16 tokenów. Gdy czterech rodzin piętra 1 nie ma, proces kończy się kodem 1 i katalog zostaje pusty. Przy domyślnym limicie przegląd te rodziny kładzie i przechodzi do paczek.

Jedna paczka liczy 2000 termów albo 2 minuty, co pierwsze. Po każdej pełnej paczce są `resume.pt` i `canon/snapshot.json`. Co 10 paczek dochodzi wiersz w `metrics.csv`. Na standardowe wyjście nic nie leci. Postęp jest w snapshocie: piętro, kształt i numer termu. Start to piętro 2, kształt R1, numer 0. Dodawanie zostawia piętro 2 i otwiera R2. R2 zostaje pracą na wiele godzin. Po piętrze 12 proces wchodzi w warsztat i sam nie kończy pracy. Zatrzymanie to Ctrl-C.

Przerwanie w trakcie paczki zostawia ostatnią dokończoną. Drugie takie samo polecenie nie wczytuje `resume.pt`: przegląd i schemat zaczynają się od nowa, a pierwsza pełna paczka nadpisuje zapis.

Komunikat ćwiczenia w tym poleceniu to `(abstain f0)`. Sieć nie jest losowana i krok uczenia nie idzie, więc wagi w zapisie zostają wagami startowymi. Wysyłka co 6 godzin nic nie publikuje. Gdy w środowisku nie ma `KANON_V2_TOKEN`, do `events.jsonl` dochodzi `wysyłka nie wyszła: brak tokenu`, a proces idzie dalej.

`runs/0/` to świadectwo v1. To polecenie go nie rusza.

## Krótki przebieg

Kończy się sam. Kształt `T` jest atrapą testu, nie skanem R1:

```bash
PYTHONPATH=src python -m kanon.v2 --seed 0 \
  --directory /tmp/kanon-smoke \
  --package-terms 5 --max-packages 2 --shape-count 40
```

Kod wyjścia 0. W katalogu są `resume.pt` i snapshot. Wiersza metryki nie ma, bo próg to 10 paczek.

## Pliki `runs/v2-seed0/`

`canon/snapshot.json` — półka, pula, numer termu i poziom warsztatu.

`metrics.csv` — wiersz co 10 paczek: numer, frontera, kształt, numer termu i reszta liczników biegu.

`resume.pt` — wagi, optimizery, taśma, trójka piętro-kształt-numer. Powstaje po pełnej paczce.

`events.jsonl` — log wysyłki. Obraca się przy 32 MB; zostaje bieżący plik i jeden poprzedni.

Plan jest w [PLAN_V2.md](PLAN_V2.md). Prompty wdrożenia są w [PROMPTS_V2.md](PROMPTS_V2.md). Środowisko jest w [docs/SETUP.md](docs/SETUP.md).

## v1, wycofane

`python -m kanon.run` dopisuje jedną rundę do `runs/0/` i kończy pracę. Nocna pętla v1:

```bash
while true; do PYTHONPATH=src python -m kanon.run --seed 0 || exit; done
```

Ta pętla jest opisem v1. Nie uruchamiaj jej dla bieżącego biegu. Specyfikacja v1 zostaje w [PLAN.md](PLAN.md).
