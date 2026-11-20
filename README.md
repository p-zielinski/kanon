# Kanon

Kanon asks whether small networks that start from random weights, and that may speak only in a closed term language, can build a shared shelf of procedures and then use it. PyTorch is the tensor and autograd engine. No language-model checkpoint is loaded. There is no fifth network.

Three experiments were written down. Only the first two exist in code.

| Experiment | What it is | Where it stands |
|---|---|---|
| v1 | Four role-rotating networks, one solo network, and one frozen reader of the same canon. | Retired. `runs/0/` is the witness (1218 events). Do not resume it. |
| v2 | Six networks. A reducer searches a fixed shape list, then a workshop drills the shelf. | Implemented. The long run is closed. Read [docs/v2.1-result.md](docs/v2.1-result.md). |
| v3 | Frozen language models plus an optional Lean check, scored by a counterfactual transfer test. | Not implemented. The plan is [docs/future/v3-plan.md](docs/future/v3-plan.md). There is no `src/v3/`. |

The v2 plan is [docs/v2-plan.md](docs/v2-plan.md). The v1 specification is [docs/v1-plan.md](docs/v1-plan.md). How the code was built, prompt by prompt, is [docs/history.md](docs/history.md). The environment is [docs/SETUP.md](docs/SETUP.md). A long browser deck of the same numbers is [docs/v2-story.html](docs/v2-story.html).

## Run v2

From the repository root, after [docs/SETUP.md](docs/SETUP.md):

```bash
source .venv/bin/activate
PYTHONPATH=src python -m kanon.v2 --seed 0
```

The directory is `runs/v2-seed0/`. A GPU is not required. One command stays in one process. `runs/` is gitignored. Do not commit that directory or any weight file.

The process starts with a survey of terms up to 16 tokens. If the four families of floor 1 are missing, it exits with code 1 and leaves the directory empty. With the default limit the survey lays those families down and moves on to packages.

One package is 2000 terms or 2 minutes, whichever comes first. After each full package the process writes `resume.pt` and `canon/snapshot.json`. Every 10 packages it appends a row to `metrics.csv`. Nothing else is printed on standard output, apart from a torch warning when NumPy is absent. Progress is in the snapshot: floor, shape, and term index. The start is floor 2, shape R1, index 0. Addition leaves floor 2 and opens R2. R2 is many hours of work. After floor 12 the process enters the workshop and does not finish by itself. Stop it with Ctrl-C.

A stop in the middle of a package keeps the last finished package. The same command again loads `resume.pt` and the shelf from `canon/snapshot.json`, so the triple floor-shape-index stays. Without those two files the start is the survey and floor 2.

During the schema search the network does not choose the term. The exercise, one per package after a floor has closed, is the utterance. At epsilon = 1 the sampler speaks. Below 1 the network starts from the 176-token observation prefix. A hit goes on the tape and takes an AdamW step. An empty tape does not move weights. A ship every 6 hours publishes nothing in this tree: the shipper is a stub. When `KANON_V2_TOKEN` is absent, `events.jsonl` gains a line that the ship did not go out, and the process continues.

`runs/0/` is the v1 witness. This command does not touch it.

### Short run

This one ends by itself. Shape `T` is a test stand-in, not an R1 scan:

```bash
PYTHONPATH=src python -m kanon.v2 --seed 0 \
  --directory /tmp/kanon-smoke \
  --package-terms 5 --max-packages 2 --shape-count 40
```

Exit code 0. The directory contains `resume.pt` and a snapshot. There is no metrics row, because the threshold is 10 packages.

### Files in `runs/v2-seed0/`

`canon/snapshot.json` holds the shelf, the pool, the term index, and the workshop level.

`metrics.csv` gets a row every 10 packages: index, frontier, shape, term index, and the other counters of the run.

`resume.pt` holds weights, optimizers, the tape, and the floor-shape-index triple. It appears after a full package.

`events.jsonl` is the ship log. It rotates at 32 MB. The current file and one previous file remain.

## What v2 trains

Six decoders, each 4 layers, `d_model` 128, 4 heads, `d_ff` 512, dropout 0. Seeds 0, 1, 2, and 3 are the group. Seed 4 is frozen: it is sampled at epsilon = 1 and never stepped. Seed 5 is solo. There is no fifth network.

A hit by the group copies one trace onto all four group tapes and steps seeds 0-3. A solo hit steps only seed 5. The observation is padded to 176 tokens. The message starts at index 176. The position table is 256 and is chosen once.

The schema is a reducer over a finite list of term shapes (addition, bounded subtraction, multiplication, comparison, parity, bounded sum, gcd, exponent, then equality and tactic facts). The workshop, after a floor closes, drills tasks built from two cards on the shelf. Levels 0-3 are deeper compositions. Levels 4-6 ask for a tactic the kernel accepts. The workshop has no last round.

The closed run did not finish that schema. It stopped on purpose on 2026-10-11 after 19 hours 25 minutes, at workshop level 6, with seven shelf cards and three shallow facts about zero. Group tactic accuracy on the last 200 metric rows was about 0.84, solo about 0.65, frozen sampler about 0.17. That is a learning result on this shelf. It is not an arithmetic canon. The full table is [docs/v2.1-result.md](docs/v2.1-result.md).

## v1, retired

`python -m kanon.run` appends one round to `runs/0/` and stops. The overnight v1 loop was:

```bash
while true; do PYTHONPATH=src python -m kanon.run --seed 0 || exit; done
```

That loop describes v1. Do not run it for the current experiment.

v1 is the social-selection protocol: author, breaker, user, and outsider rotate each round. A definition enters the canon only if the breaker does not kill it, 16 audit inputs agree with the hidden target, and the user calls the new slot in a solution that itself passes the audit. Solo is one network with four author proposals and no breaker. The frozen curve reads the group canon and does not step. The question was whether the group curve would separate from both baselines on a holdout. The 1218-event read of `runs/0/` did not support a second breaker: the author target was always `(succ (zero))`, because the task seed omitted the round number, and the breaker abstained 1218 times for the same reason. v2 puts the round number in the seed and searches with the reducer instead. Specification: [docs/v1-plan.md](docs/v1-plan.md).

## v3, not in this tree

[docs/future/v3-plan.md](docs/future/v3-plan.md) is a future experiment. It is not implemented. It does not train weights. Frozen language models would grow a typed graph of integer-arithmetic claims, and a Python clerk would score transfer by a Full versus Ablation versus Empty counterfactual. Lean 4 is an optional later milestone in that plan. Nothing under `src/kanon/` imports it, and `python -m v3` does not exist.

## Tests

```bash
source .venv/bin/activate
PYTHONPATH=src python -m pytest -q
```

The smoke of 20 v1 rounds and the short v2 command are in that suite. A torch warning about missing NumPy is expected. NumPy is not a dependency.

## License

MIT. See [LICENSE](LICENSE).
