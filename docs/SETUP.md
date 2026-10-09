# Środowisko

Grok wykonuje te kroki sam w katalogu repozytorium. Nie ma nic do pobrania z Hugging Face ani z żadnego repozytorium wag.

## Czego nie pobierać

Eksperyment startuje z losowych wag. Checkpoint modelu językowego psuje pomiar (patrz `PLAN.md`, sekcja „Poza zakresem v1”).

Nie uruchamiaj i nie wpisuj do zależności:

- `transformers` z pobraniem modelu,
- `huggingface_hub`, `ollama`, `llama-cpp-python`,
- żadnego pliku `.gguf`, `.safetensors`, `.bin` z wagami cudzego modelu.

`torch` jest silnikiem tensorów i autogradu. Wagi powstają w `src/kanon/model.py` przez inicjalizację losową.

## Wersje

- Python 3.11 albo 3.12.
- CPU wystarcza. Sieć ma `d_model` 128 i cztery kopie. GPU jest zbędne w v1.

## Instalacja

Z katalogu głównego repozytorium:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest
```

`requirements.txt` w katalogu głównym:

```
torch
pytest
```

Komentarz na górze `requirements.txt`: wagi biorą się z losowej inicjalizacji, instalacja to wyłącznie biblioteka.

## Sprawdzenie

```bash
python -c "import torch; t = torch.nn.Linear(8, 8); print(torch.__version__, tuple(t.weight.shape))"
pytest -q
```

Oczekiwane: wersja torch wypisana, kształt `(8, 8)`, pytest zielony (na pustym szkielecie dopuszczalny jest jeden test `test_torch_import`).

## Sprzęt i czas

Smoke 20 rund ma kończyć się na CPU w kilka minut. Jeśli pojedyncza runda trwa dłużej niż około 30 sekund, najpierw sprawdź, czy reduktor nie dostaje paliwa większego niż 256 i czy sieć nie jest większa niż w tabeli parametrów w `PLAN.md`. Nie powiększaj modelu, żeby „szybciej się uczył”.
