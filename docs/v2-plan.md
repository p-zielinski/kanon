# Kanon v2

> Status when this plan was frozen: plan after the eighth critique. The implementation prompts are summarized in [history.md](history.md). At that moment the v2 code did not exist yet. It now lives in `src/kanon/v2/`. The closed run is [v2.1-result.md](v2.1-result.md). This file is still the plan, and the run is incomplete against it.
> Read: `runs/0/`, 1218 events. That directory stays a witness and is not resumed.
> One process is meant to run for many days on CPU, including a VPS with 1 GB of RAM. The schema is a reducer loop in packages and is meant to finish. The days after it are the workshop: each round a new task from two cards in the window, and every tenth round a mutation of a fact, which the kernel checks.

Knowledge has two sources: the reducer and the kernel, which compute deterministically, and four networks of our own. The network has to see the task. Production of a message starts from a prefix of 176 observation tokens. There are no weights from the internet. There is no foreign theorem library. There is no fifth network in the first long run.

## What 1218 rounds showed

The machine is alive. The canon is empty, because the loop does not search, and then copies one term.

| Fact | Count |
|---|---|
| Events | 1218 |
| Distinct author targets | 1: every time `(succ (zero))` |
| Admissions to the canon | 0 |
| Breaker messages | 1218 times `( abstain f0 )` |
| Audit matching the target | 1, round 0, and the definition still fell |
| Proposals `(def f0 1 (zero))` | 1209 of 1218 |
| Epsilon from round 600 | floor 0.05 |
| Group and frozen weights, depth 1, rounds 610-1200 | both averaged about 0.16 |

The task seed and the message seed do not contain the round number. A breaker with seed `seed + 20` always abstains. The tape learns from proposals that were accepted syntactically. At epsilon = 0.05 the network has 95% of the voice and stays on `(zero)`.

The v2 seed contains the round number. Otherwise a multi-day run repeats one target.

## Two limits

`MAX_TOKENS` in the sampler is an ordinary number. It may depend on the floor.

`MAX_LEN` in the network is a position table `nn.Embedding(MAX_LEN, 128)`. The window ceiling is chosen once, before the start, and stays **256**.

Message budget:

| Floors | Budget | Why that much |
|---|---|---|
| 1 | 24 | identity from `propose` is 12, a variable successor is 15 |
| 2-3 | 48 | addition from `propose` is 36, bounded subtraction is 46, multiplication through `(call ...)` is 40 |
| 4-12 and exercises | 80 | a sentence with `step` and a tactic chain fit in 80, and no longer fit in 48 |

A term over the floor budget is dropped before the reducer counts it.

## A window that does not slide

Positions are absolute. The observation is padded with the token `[PAD]` to **176** tokens. The message always starts at index 176 and has up to 80 tokens. Together, 256.

Those 176 hold three fixed segments:

1. **48 tokens for one body or one claim.** In a proof exercise the claim lies here, without the tactic. A shorter string is padded with `[PAD]`. A claim that does not fit does not enter the pool and is not drawn. During the schema, until a specification lands, the segment is empty.
2. **Five public pairs.** `(num k)` for `k` from 0 to 10 stays four tokens `( num k )`. There is no tower of `succ` in the window.
3. **Two names from the shelf**, two pairs each. The generator first picks those two cards, then builds the hidden term only from them. Every fourth exercise, one card is the task family. In the others both are ingredients, with no marker "this is the answer". The measurement set shows ingredients and also has no such marker.

The observation forward is computed once per message and stays the production prefix. The learning step goes one agent at a time, batch **4**.

## Vocabulary on day zero

The long run starts as one model. From the first round the network vocabulary contains `"4"` through `"10"`, `[PAD]`, and the heads `eval`, `rewrite`, `base`, `step`, `claim`, `eq`, `dom`, `ih`. Term forms stay: `zero`, `succ`, `pred`, `if0`, `var`, `call`, `self`, `def`. Outside `(num ...)`, a digit is still a variable index or an arity, and still only `0`-`3`.

## Survey to 16 tokens

Before the multi-day process stands up, the computer surveys terms up to **16** tokens, without `self` and without `call`. One shortest term per family comes down to the shelf: identity, successor, predecessor, constant 0, 1, or 2. The order is shortest first, and deterministic at equal length. The same result enters the group shelf and the solo shelf. Frozen weights get nothing to write.

If four families are missing, there is no week. A test of 20 rounds (different targets, at least one `break`, a seed that contains the round number) concerns the loop, not the survey.

## Shapes

Floors 2-8 and the first facts of floors 9-12 are a finite list of trees. A term index decodes to holes. A list of terms is not built. The shape of one particular addition is not written in. What is written in is the tree and a specification of behavior on a domain.

A composition of calls is a workshop level. There is no separate floor for a term that "is not a constant": the shortest such call is a wrapped identity and gives the shelf nothing.

Simple hole `S`, no `call`:

- `(zero)`, `(var i)` for `i` smaller than the arity,
- one `succ` or one `pred` around `(zero)` or `(var i)`.

At arity 2 there are 9 of them. The plan's sentence says 5 at arity 1. The list itself includes `(pred (zero))`, and the code counts 6 at arity 1. The compliance pass kept that count. See [history.md](history.md). `call` stands only in the places written next to the shape.

`(if0 condition zero-branch nonzero-branch)`: zero takes the zero branch, as in the v1 reducer.

| Shape | Tree | Specification that closes the floor |
|---|---|---|
| R1 | `(if0 S S (W (self S ...)))`. `self` has as many arguments as the arity. `W` is nothing, one `succ`, one `pred`, or one `(call name ...)` in which exactly one argument is that `self` and the rest are `S` | addition; multiplication; sum `0+...+n`; exponent, with `0^0 = 1` |
| R2 | `(if0 S S (if0 S S (W (self S ...))))`, the same `W` | bounded subtraction. It hangs on the shelf. It does not by itself close the frontier of floor 2 |
| N1 | `(if0 (call name S ...) S S)` | comparison: `1` when `x <= y`, otherwise `0`. Minimum and maximum, if they land, hang and do not close the frontier |
| P | `(if0 (var 0) S (if0 (pred (var 0)) S (self (pred (pred (var 0))))))`. Two `pred` stand only in that one argument. Holes without `call` | parity on `0..10` |
| G | fixed variables, two binary names `A` and `B` from the shelf: `(if0 (var 1) (var 0) (if0 (call A (var 0) (var 1)) (self (var 1) (call B (var 0) (var 1))) (self (var 1) (call B (var 1) (var 0)))))`. Two `self` in the term, one on a reduction path | gcd on pairs `0..6` |

R1, with simple holes and wrappers from the floor-1 shelf, has about 50 thousand terms. R2 has about 4 million. N1, P, and G are smaller. The arity-2 domain is pairs `0..6`. Each floor has its own term index, from zero. `resume.pt` holds the triple: floor, shape, index. Order inside a shape is shortest first, so the first hit of a family is the shortest, and the scan of that family on that floor stops.

Without subtraction from R2, comparison and gcd can exhaust empty. The frontier still drops.

### Package

The schema is not a round per term. One package counts **2000** terms or **2 minutes**, whichever comes first. Then a save of `resume.pt` and, when some floor is already closed, one exercise. R2 at this rhythm is hours of work. A round on each of 4 million terms, at a save of about 50 MB, would take weeks of disk before multiplication stood up.

Every 10 packages a metrics row is appended: floor, shape, index. There is no per-term row.

### Tactics

The tactic schema is also a package and also decodes an index, not the product of all sequences.

| Shape | How it grows | Closes the floor |
|---|---|---|
| E | `(claim (eq L R) (dom lo hi) (eval))`. `L` and `R` are `S` or one `(call ...)`. `hi > lo`, both from `0..10`. The side strings differ. At least one side calls a family from floor 2 or higher. One substitution of an entry body into the other string is not an E fact | one equality fact on that domain |
| W | `(claim (eq L R) (rewrite name))`. `R` comes from `L` by one substitution of an entry body, inside its domain | one accepted `rewrite` |
| C | start from an E or W fact already on the shelf. Append one tactic `eval` or `rewrite`. The kernel must accept the longer string. Ceiling: 8 tactics and 80 tokens | one accepted chain |
| I | `(claim (eq L R) (base) (step (ih) (rewrite name)))` on the skeleton `f(x,0)`, `f(0,x)`, `f(x,1)`. `f` and `name` range over the shelf | one sentence that passed `base` and `step` |

One shortest fact that the kernel accepted and that closes the floor comes down to the shelf, plus at most **eight** further facts, as the exercise pool. Facts that pass filter E enter the pool first. W facts are added while eight places are still short. The rest of the set does not enter the shelf. A tautology, both sides identical before the tactic, or a copy of a fact already hanging, is dropped.

The same term and the same fact from the schema enter the group shelf and the solo shelf. The difference between the group, solo, and frozen weights starts at the exercise and at the mutation.

## Frontier

What can be drawn as schema is the lowest floor whose required family is still missing and whose shape is not exhausted. After the survey to 16 tokens, that floor is 2.

| Floor | Shape | The frontier drops when |
|---|---|---|
| 1 | survey to 16 tokens, no `self` | four families |
| 2 | R1, then R2 | the shelf has addition, and R2 has found subtraction or is exhausted |
| 3 | R1 | the shelf has multiplication, or the shape is exhausted |
| 4 | N1 | the shelf has comparison, or the shape is exhausted |
| 5 | P | the shelf has parity, or the shape is exhausted |
| 6 | R1 | the shelf has a bounded sum, or the shape is exhausted |
| 7 | G | the shelf has gcd, or the shape is exhausted |
| 8 | R1 | the shelf has exponent, or the shape is exhausted |
| 9 | E | one fact with filter E, or the shape is exhausted |
| 10 | W | one accepted `rewrite`, or the shape is exhausted |
| 11 | C | one accepted chain, or the shape is exhausted |
| 12 | I | one sentence with `base` and `step`, or the shape is exhausted |

When the term index reaches the end of the shape and the family is missing, the log writes "schemat wyczerpany" (schema exhausted), the family stays empty, and the frontier drops to the next floor. An exhausted shape does not start over. An empty gcd does not stop exponent or proof. An exercise that needs an empty family is not drawn.

A composition of two successors from floor 2 does not close. A task whose hidden term exceeds fuel 256, 4096 nodes, or 10000 steps is not created. Floors above 12 sleep in the same code: further shapes are appended without moving weights.

## Exercises

After the frontier leaves a floor, that floor becomes a workshop. The workshop has no last round. That is the content of the multi-day run.

The computer builds a task from two cards in the window. The seed contains the round number. The reducer or the kernel computes the verdict before anyone speaks. The group, solo, and frozen weights receive the same shared task. Tapes and epsilon are separate. Frozen weights do not receive a step and do not store a trace.

The workshop level is a separate axis from the frontier. A composition uses only the two names in the window. A deeper level is a deeper composition of those two names.

| Level | Task |
|---|---|
| 0 | one `(call ...)` of one of the two cards, fresh inputs from `0..6` |
| 1 | a composition of two calls of those cards |
| 2 | a composition of three |
| 3 | a composition of four |
| 4 | reconstruct the tactic of a fact from the shared pool. The window has the claim. The tactic is absent |
| 5 | the same tactic on fresh `(num k)` from `0..10`, inside the fuel. The kernel grades the speaker's tactic |
| 6 | a chain of 2..8 tactics |

The last 50 fresh group exercises on a level, with success of at least 0.6, raise the level by one. Below 0.3 the level stands. Levels 4-6 wait until the pool of tactic-schema facts is non-empty. Until then the ramp ceiling is level 3. The ceiling of the whole workshop is level 6. On the ceiling the draw continues: new inputs, new claims from the pool, new numbers. An epsilon floor on a known level does not end the run, because the target is different each round.

An exercise round has one speaker. Rotation says which of the four networks speaks. One attempt. Solo has one network and also one attempt. Frozen weights have one attempt. The reducer or the kernel sets the verdict. A hit appends the same trace to the tapes of all four group networks. The shelf stands. The v1 `admit` gate does not open this round: an exercise does not add a definition.

During the schema the exercise is one per package and stays on the lowest unlocked level. When the frontier leaves floor 12, nine rounds out of ten are a shared exercise, and one is a mutation.

## Mutation

Mutation stands up when the frontier has left floor 12. Every tenth round the computer takes a fact from that curve's pool and makes **64** mutations: another number from `0..10`, another composition of two names visible in the window, or one appended tactic inside the ceiling of 8 and 80 tokens. The kernel grades each. A list of skeletons is not built, and a state "bud exhausted" does not exist, because numbers and compositions of two cards have no end.

A hit that passes filter E, or is a longer chain the kernel accepted, and is not a copy, enters only that curve's pool. The shared pool, from which the group, solo, and frozen-weight exercise is drawn, stays the schema facts present on both shelves. A replay of a local fact is a task only of the curve that has it, in the mutation round. A local fact does not enter the 20-point gap.

On that replay the breaker has four tries to name a point the kernel itself confirms as a split. A confirmed kill enters the breaker tape. The breaker does not scan the 64 mutations: the kernel checks those.

The implemented rhythm, checked in the compliance pass: before floor 12 is left, a package leaves one exercise and zero mutations. Afterwards, in each ten workshop rounds the last is a mutation (rounds 9, 19, 29). Round 0 is an exercise. Rounds 10, 20, and 30 are not mutations.

## Shelf

An accepted definition and an accepted fact, from the schema or from a mutation, come down to the shelf at once. A named or called entry does not disappear after 12 rounds. The 12-round expiry remains only for a working buffer that has not yet passed the domain.

Names `f0`-`f31` are a window. The shelf has stable identifiers. Measurement tasks, exercises, and citations hold the identifier. When the observation is assembled, the identifier receives a name from the window. One observation shows two names.

The bench every 50 rounds computes a ring just past the domain and a few interior points. A numeric split removes the entry and every record that cites it, including from the pool. A step limit or a fuel limit is "unknown" and does not remove the entry.

The arity-1 domain is `0..10`. Arity 2 is pairs `0..6`, if the reducer finishes in time. The computer assigns the family when the behavior matches the specification on the domain.

## Tape and epsilon

A training sequence is 176 observation tokens and then the message the machine checked. A term or a fact accepted from the schema also enters the tape, with the observation of that family. `(abstain ...)` does not enter. A message that splits on the domain or on the kernel does not enter. An empty tape does not move weights.

The tape has a ceiling: **256** traces per agent and per workshop level. An older trace falls off. `events.jsonl` rotates at **32 MB**. The current file and one previous file remain. `metrics.csv` stays whole and grows by one row per 10 packages or 10 workshop rounds.

The epsilon clock ticks only in a round that plays that level.

- During the schema the production network does not choose.
- The exercise author descends by 0.995 to a floor of 0.25. A new workshop level starts at 1.
- The breaker descends by 0.995 to a floor of 0.5. The clock ticks on the replay of a local fact, where it looks for a split.
- The executor descends by 0.995 to a floor of 0.05 only from their first accepted answer on that level.
- The speaker of a mutation round, until that curve has its own local fact, has epsilon = 1.

The step is AdamW, lr `3e-4`, clip 1, batch 4, one agent at a time. Batch 32 at a window of 256 reaches about 626 MB of process on the development machine. Batch 4 stays at about 316 MB.

## Tactic kernel

`eval` computes both sides on the domain in the sentence. A fact says "equal on this domain".

`rewrite name` replaces a subterm by the entry body, inside its domain. Outside the domain the kernel rejects. One such substitution is a W fact, not an E fact.

`base` reduces both sides at zero.

`step` sets the goal as the whole equality at `(succ (var 0))`. Local `ih` says the claim holds for `(var 0)`. The kernel may once replace a subterm matching the left side of that claim by the right side. On `add(x, 0) = x`, reduction leaves `succ(add(n, 0))`, and that one substitution turns `add(n, 0)` into `n`. The assumption is not a shelf entry. Rewriting addition alone, without that substitution, turns `add(x, 0)` into `if0`.

Lean as a program, and its library, stay outside v2.

## Measurement set

Fifty tasks are created once, at the moment the frontier leaves floor 8. The source is entries laid down by the schema and present on both shelves. Cards are ingredients of the task, with no family marker. After that the set does not change.

Until then the metric has fresh exercise accuracy and has no gap. Every 50 workshop rounds, with no learning step, that fixed set is scored. A gap of 20 points concerns composition. Two points are noise. Whoever found addition first is not that gap: the same shape found it.

## Metrics

Every 10 packages or every 10 workshop rounds: the round or the package index, the frontier, the shape and the term index, whether the shape is exhausted, the workshop level, fresh accuracy of the last 50 exercises of the group, of solo, and of the frozen weights, the shelf size, the size of the shared pool and of the local pools, the number of traces on the tape, bench snapshots.

Separately, the package index where addition, multiplication, and the first E fact entered, and a marker of whether the fact came from the schema or from a mutation. Shelf size alone from round zero says nothing: the survey to 16 tokens lays four families down at once.

`canon/snapshot.json` writes the shelf (identifier, window name or none, family, domain, body, citations), the shared pool, the local pools, the term index, and the workshop level. The body is for a human.

## A run of many days

The multi-day start is one command, once the survey to 16 tokens has laid down four families, and the code already contains the sleeping shapes, the package, the workshop, the mutation, the window of 256, `[PAD]`, `ih`, and the tape ceiling.

One invocation stays in the process. The save `resume.pt` is atomic, through a temporary file, and replaces one file: weights, optimizers, the tape trimmed to the ceiling, the term index, the workshop level, the pools. In the schema the save is after every package. In the workshop it is every **50** rounds or every **2 minutes**, whichever comes first. An interrupt loses at most an unfinished package. Directory: `runs/v2-seed0/`. `runs/0/` is not deleted and is not resumed. `runs/` stays in `.gitignore`.

The schema is meant to finish. The days after it are the workshop at the level ceiling, and a mutation every tenth round. Exhaustion of a shape drops the frontier.

A separate read-only command loads the save, asks one term or one sentence, and prints the verdict. It does not touch the step, the shelf, the shape, the workshop, or the pool.

## VPS and a copy on GitHub

The target multi-day machine has 1 GB of RAM and one CPU. The first long run on it does not wait for a GPU. For the duration of the run the box does nothing else. A swap of 1 GB is a fuse for a reducer peak.

Every **6 hours** the same process sends a copy, after a successful local save:

- `snapshot.json`
- `metrics.csv`
- six weight files: agents 0-3, solo, frozen

The weights of six networks of this size are about 20 MB. The optimizer, the tape, and `resume.pt` stay on the VPS disk. The process resumes the round from those. From a GitHub release one can restore the weights and read the shelf, not resume the AdamW step.

The release `kanon-v2-latest` is replaced (`--clobber`). Once a day a separate tag `kanon-v2-RRRR-MM-DD` is created, without replacement, so a month can be rewound by a day. A commit every few hours with a weight file on `main` leaves gigabytes in history, which deleting the file does not erase.

The push token sits in the VPS environment, outside the repository. The ship script stands outside `model.py`. A failed ship is written in the log, and the run continues. When the repository is public, the weights and the metrics are public too.

In the tree as published, the shipper does not publish. A missing `KANON_V2_TOKEN` writes a log line and the process continues. Do not commit `runs/` or weight files.

## Code order, as it was built

1. Stage 0, a test of 20 rounds: different targets, a seed with the round number, at least one `break`.
2. A model with a window of 256 and the full vocabulary. Observation to 176. Message production starts from that prefix. Batch 4.
3. A survey to 16 tokens onto both shelves. Without four families of floor 1 there is no week.
4. Shapes R1 and R2 of floor 2, a package of 2000 terms or 2 minutes, a save after the package, one exercise per package. An end without addition drops the frontier.
5. The rest of the arithmetic: multiplication, comparison, parity, sum, gcd, exponent. An empty family drops the frontier. After leaving floor 8 a fixed set of 50 tasks is created.
6. Shapes E, W, C, and I. Filter E cuts a synonym of one `call`. A chain grows from a fact, not from a product of tactics. `ih` substitutes one subterm.
7. Workshop levels 0-6 on two cards. A hit goes to the tape of all four and does not call `admit`. Mutation: 64 kernel checks per round, a local pool, no exhaustion state.
8. Tape ceiling 256, rotation of `events.jsonl` at 32 MB, a workshop save every 50 rounds or every 2 minutes, a ship every 6 hours.
9. One command for many days, directory `runs/v2-seed0/`, a machine with 1 GB of RAM.

There is no fifth network and no GPU profile in that order.
