# Prompty v2 dla Grok 4.7

Uruchamiaj **po jednym**, rano, gdy `runs/0/metrics.csv` ma wiersze danych. Nie łącz promptów w jednej sesji. Nie rozszerzaj zakresu. Bieg v1 w `runs/0/` ma w tym czasie dalej dopisywać rundy poleceniem bez `--fresh` i bez dwóch łamaczy.

Wspólna rama: przeczytaj `PLAN.md`, `PLAN_V2.md` i `docs/SETUP.md`. Parametry v1 zostają. Wag z internetu nie pobieraj.

---

## Prompt 1 — odczyt metryk

```
Przeczytaj PLAN.md, sekcję „Jak czytać bieg”, oraz PLAN_V2.md. Nic nie zmieniaj w kodzie.

Przeczytaj runs/0/metrics.csv od nowa i, jeśli jest, runs/0/canon/snapshot.json. Przy rundzie co najmniej 100 zajrzyj też w runs/0/events.jsonl na tyle, by odróżnić odrzuty parsera ([ABS]) od legalnych propozycji, które zbił łamacz, które zbił audyt, które odpadły brakiem call i które weszły do kanonu.

Wynik zapisz w .cursor/work/kanon-v2.md. Podaj ostatni wiersz metryk: runda, ε, głębokość, rozmiar kanonu, rozmiar cmentarza, breaker_rate, audit_miss_rate oraz trzy krzywe na głębokościach 1–4. Dopisz, który z czterech obrazów z PLAN.md widać. Luka poniżej około 20 punktów na 50 zadaniach to szum.

Na końcu pliku jedna linia, dosłownie:
bramka-łamacza: otwarta
albo
bramka-łamacza: zamknięta

Otwórz bramkę tylko wtedy, gdy jakiś wiersz ma rundę co najmniej 100, audit_miss_rate większe niż 0.30 i w events.jsonl są legalne propozycje, nie sam [ABS]. Przed rundą 100 bramka zostaje zamknięta, nawet gdy odsetek audytu jest wysoki. Pusty kanon przy wysokim audycie opisz jako „jeszcze nic nie utrwalił”. Nie lecz tego podpowiedzią ani piątym agentem w tej sesji.
```

## Prompt 2 — jedno zadanie do zapisanego modelu

```
Przeczytaj PLAN_V2.md, sekcję „Pytanie do zapisanego modelu”. Zaimplementuj tylko odczyt jednego zadania z runs/<seed>/resume.pt.

Funkcja w src/kanon/run.py ładuje wagi grupy i kanon grupy z resume.pt. Stawia jedno zadanie głębokości zapisanej w checkpointcie, chyba że podano inną. Agentem domyślnym jest sieć 0. Komunikat idzie przez speak, z ε z checkpointu, rolą obcego. Sukces to zgodność na 16 parach audytowych. Pięć par publicznych sukcesem nie jest.

Nie wołaj step ani end_round. Nie zapisuj resume.pt, events.jsonl, metrics.csv ani snapshotu. Term celu wypisz człowiekowi. W obserwacji agenta go nie ma.

CLI: python -m kanon.run --pose --seed 0, z opcjonalnym --depth i --agent 0..3. Bez --pose dotychczasowe polecenie dalej dopisuje jedną rundę jednego łamacza.

Najpierw test na tymczasowym katalogu: jedna zapisana runda, potem pose nie zmienia resume.pt i nie dopisuje events.jsonl. Uruchom pytest -q dla testów, które dotykasz. W .cursor/work/kanon-v2.md dopisz, jak zadać pytanie.
```

## Prompt 3 — drugi łamacz, tylko przy otwartej bramce

```
Przeczytaj .cursor/work/kanon-v2.md i PLAN_V2.md. Jeśli linia bramki nie brzmi dokładnie „bramka-łamacza: otwarta”, nie zmieniaj kodu i zakończ sesję.

Gdy bramka jest otwarta, dodaj piątego agenta, ziarno 6, który w każdej rundzie jest tylko łamaczem. Czwórka 0–3 rotuje dotychczasowym wzorem. Drugi łamacz ma cztery próby, osobno od pierwszego, komunikat (break …) albo (abstain …), bez szkicu i bez oczekiwanej wartości. Obserwacja: ciało kandydata, cmentarz bez termu celu. Trafienie któregokolwiek zdejmuje kandydata przed audytem 16. Jeden krok AdamW na rundę, dane jak u łamacza. Solo zostaje przy czterech krokach, zamrożone wagi przy zerze.

Domyślne python -m kanon.run --seed 0 zostaje przy jednym łamaczu i umie czytać dotychczasowy resume.pt. Bieg dwóch łamaczy startuje od rundy 0 w runs/v2/, ze zbiorem pomiarowym holdout_seed ziarna 0. Nie wznawiaj runs/0/.

Najpierw testy: jeden łamacz na starym zapisie, dwóch łamaczy na świeżym katalogu, zbicie przez drugiego pomija audyt. Uruchom pytest -q dla testów, które dotykasz. Dopisz polecenie startu do README i do .cursor/work/kanon-v2.md.
```
