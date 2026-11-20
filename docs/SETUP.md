# Environment

Do this in the repository root. There is nothing to download from Hugging Face or from any weight repository.

## What not to install

The experiment starts from random weights. A language-model checkpoint would spoil the measurement. See [v1-plan.md](v1-plan.md), section "Out of scope for v1".

Do not run, and do not add as dependencies:

- `transformers` together with a model download,
- `huggingface_hub`, `ollama`, `llama-cpp-python`,
- any `.gguf`, `.safetensors`, or `.bin` file of someone else's model.

`torch` is the tensor and autograd engine. Weights are created in `src/kanon/model.py` and `src/kanon/v2/model.py` by random initialization.

## Versions

- Python 3.11 or 3.12. On the machine that built this tree, Python 3.12 was `/opt/homebrew/bin/python3.12`. The system `python3` was 3.9 and does not meet this file.
- CPU is enough. Each network has `d_model` 128. v1 has four copies. v2 builds six. A GPU is not required.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pytest
```

Root `requirements.txt` lists `torch` and `pytest`. The comment at the top says weights come from random initialization and the install is only the library.

The recorded install on 2026-10-09 was torch 2.14.1 from the CPU index and pytest 9.1.1. `pytest -q` then reported 1 passed (`test_torch_import`) plus a warning that NumPy was missing.

## Check

```bash
python -c "import torch; t = torch.nn.Linear(8, 8); print(torch.__version__, tuple(t.weight.shape))"
PYTHONPATH=src python -m pytest -q
```

Expected: a torch version is printed, the shape is `(8, 8)`, and pytest is green. The NumPy warning is not a failure. Do not add NumPy to chase that warning.

## Hardware and time

The 20-round v1 smoke should finish on CPU in a few minutes. If a single v1 round takes longer than about 30 seconds, check that the reducer is not given more than 256 fuel and that the network is not larger than the parameter table in [v1-plan.md](v1-plan.md). Do not enlarge the model so that it "learns faster".

The default v2 command is not a smoke. One package is 2000 terms or 2 minutes. R2 is hours. The closed run in [v2.1-result.md](v2.1-result.md) ran 19 hours 25 minutes on CPU, about two cores, and was stopped on purpose. Use the short command in the README when you only want to see that the process writes `resume.pt`.
