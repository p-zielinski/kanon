# Kanon v1

> Plan last updated: 2026-10-09
> Status: v1 specification. The code now exists under `src/kanon/` and the path is retired. See the README. This file stays the specification the code was checked against.
> The original aim of the document was a faithful record of the design conversation, closed tightly enough to build without further questions.

## Why this exists

The experimental question: can four small networks, starting from random weights and closed inside a term language, build a canon of procedures with which they solve deeper tasks than a network that learns alone, and than a network that can read the canon but does not update its weights.

This is not an attempt to build AGI. It is a micro-community with plastic weights. The result is a number on tasks the agents have not seen.

What stays from the conversation:

- Human communities grow because a thought comes unstuck from a person. What remains is a procedure someone else actually used, or a counterexample that killed it.
- Gratitude is a use. Criticism is a counterexample. A polite sentence has nowhere to go, because the channel does not accept sentences.
- Learning on another model's opinion drifts down when there is no check. Learning next to a machine that reduces a term to a number has a chance to go up.
- A large model from the internet already knows arithmetic. The experiment would then measure recall, not construction.
- Freezing weights after training is off the schedule. The learning step is part of the round.

## Thesis

After N rounds an outsider (an agent scored on a frozen measurement, with no weight write) solves more tasks from a set held aside than:

1. the same outsider with random weights, reading the same canon,
2. a single network after the same number of learning steps and the same number of machine calls, without the breaker and user roles.

If the group curve does not come unstuck from both baselines, the social-selection protocol adds nothing at this scale. That is an honest negative result, not a reason to add a model from the internet.

## Out of scope for v1

- Downloading weights (Llama, GPT, Hugging Face, Ollama, GGUF).
- A private matrix-multiply engine, a private CUDA implementation, a private autograd.
- Natural language in either direction after the process starts.
- Python as the agent language. Comments, strings, and names would be a speech channel and pretrained knowledge.
- Lean and a standard library. That would be someone else's canon loaded at the start.
- A fifth agent. It enters only in v2, and only as a second breaker, if the machine regularly kills definitions the breaker let through. The 1218-event read dropped this. See [history.md](history.md).
- A comparison with a large language model. Only once three curves exist (group, solo, random weights).
- A claim that growth of the canon is general intelligence. Growth is deeper composition in one formal world.

## Architecture

```
four networks (separate weights, random start)
        |  terms that match the role slot
        v
   round harness
        |  legal term
        v
   reduction machine  --->  a number or an error (step / memory limit)
        |
        +--> canon (disk)
        +--> cemetery (disk)
        +--> replay of accepted traces
        |
        v
   one learning step on each of the four networks
```

The only borrowed piece is the differentiation engine (PyTorch). The agent, the grammar, the reducer, the canon, the role rotation, and the update rule are ours.

Four agents exist so that four roles do different work in one round. Shared memory is in the canon, not in the weights. A step that overwrites a weight does not immediately erase the record. A replay from the canon each round keeps the old skill alive.

| Thing | Held in | Shared |
|---|---|---|
| Policy for composing terms | network weights | no, four sets |
| Procedures that survived | canon on disk | yes |
| Killed hypotheses and the inputs that killed them | cemetery | yes, visible to the breaker as a list of inputs |
| Measurement tasks | a file generated once from a seed | nobody uses them in rounds |

## The Kalk language

The only agent language. A term reduces to a natural number. A conversation is a sequence of terms.

### Syntax

```
expr ::=
    (zero)
  | (succ expr)
  | (pred expr)
  | (if0 expr expr expr)
  | (var n)                 ; n an integer, 0 .. arity-1
  | (call slot expr ...)    ; slot is f0 .. f31
  | (self expr ...)         ; recursion, legal only in a def body

def  ::= (def slot arity expr)

num  ::= (num k)            ; only in data: IO pairs and breaker arguments
                            ; k is a decimal number >= 0
                            ; expands to succ^k (zero)
                            ; illegal in a def body

draft-author     ::= (draft def)          ; slot = the only free slot in the observation
draft-rest       ::= (draft expr)         ; user and outsider; arity comes from the task
propose          ::= (propose def)
use              ::= (use slot expr)      ; expr must contain (call slot ...)
solve            ::= (solve expr)
```

Names outside `f0`-`f31` do not exist. A user or outsider solution is an expression, not a canon definition.

Canon slots are closed: `f0` through `f31`. The model vocabulary therefore does not grow. An expired definition frees its slot, and the slot can return.

`pred` of zero is zero. `if0` takes the zero branch when the condition reduces to zero, otherwise the nonzero branch. The implementation names these `zero_branch` and the other branch. Zero takes the first branch, because only then the addition definition below returns 5 on `(num 2)` `(num 3)`.

`(self ...)` costs 1 fuel. Fuel starts at 256 on every call from outside. At zero, reduction ends with error `fuel`. Term-size limit during reduction: 4096 nodes, error `memory`. Reduction-step limit: 10000, error `steps`. An error means the definition fails the test.

### An example the agents do not receive as a hint

Addition, written only so a human reading the plan can see that the language is sufficient:

```
(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))
```

The task generator can build such terms itself. No agent prompt contains this.

### Token vocabulary

Closed. Nothing outside the list is a token.

```
( ) zero succ pred if0 var call self def arity num draft err
f0 f1 ... f31
0 1 2 3          ; variable indices and arity (maximum arity 3)
[AUTHOR] [BREAK] [USE] [OUTSIDER] [PAIRS] [CANON] [SEP] [ABS] [TRY]
```

The bracketed tokens are slot and section markers. They are not sentences. The observation is a sequence of tokens from this list, built by the harness.

The plan's list does not contain `propose`, `use`, `solve`, `break`, or `abstain`. `TOKENS` does not grow to include them. The role grammar still requires them, so they live in the message-head set and in the network vocabulary. The recorded length of `TOKENS` is 59.

## Observation

The harness builds the observation. The agent does not receive `hidden_term`.

Shared core:

```
[CANON] (def f0 2 ...) [SEP] (def f3 1 ...) [SEP]
[PAIRS] (num ...) ...
```

Then a role marker: `[AUTHOR]`, `[BREAK]`, `[USE]`, or `[OUTSIDER]`.

The breaker also receives the body of the definition under judgment. The user receives the slot name they may call, if the definition is a candidate (not yet in the canon). The outsider receives only the approved canon and the pairs of their measurement task or of the round task meant for the outsider. The outsider does not receive the dispute transcript.

## Roles

A role is the shape of message the slot parser will accept that round. A message of another shape is rejected. The harness does not send an explanation. It sends the token `[ABS]` and moves on.

| Role | Legal message | Effect |
|---|---|---|
| Author | `(draft (def slot arity expr))`, then `(propose (def slot arity expr))` | draft checked on 5 pairs; propose is a canon candidate |
| Breaker | `(break slot (num k) ...)` or `(abstain slot)` | an attempt to kill, or an abstain; these 4 inputs are already the search budget |
| User | `(draft expr)`, then `(use slot expr)` | draft on their 5 pairs; expr must call the slot |
| Outsider | `(draft expr)`, then `(solve expr)` | draft on 5 pairs of the task; solve goes to the metric |

Rotation by one each round: author, breaker, user, outsider, author. After four rounds each has held each role. A graph of "always these two against each other" does not set.

Role assignment on round `r` (agent indices 0..3):

```
author    = r % 4
breaker   = (r + 1) % 4
user      = (r + 2) % 4
outsider  = (r + 3) % 4
```

The arrow written in an earlier draft of the rotation described a different personal path. The code keeps this formula.

## Round

1. The generator draws the author's task: a hidden term of depth `d`, 5 public pairs, 16 audit inputs invisible to the agents.
2. The generator draws a different user task of the same depth.
3. The author composes 1 to 4 drafts `(draft ...)`, then `(propose ...)`. The slot must be free or the one the harness names (the harness puts one free slot in the observation as the only one `propose` may use). `propose` before the first draft is a rejection `[ABS]`. After the fourth draft the only legal message is `propose`.
4. The breaker gets up to 4 input attempts. The machine computes `def(input)` and `target(input)`. A difference is a hit kill: the candidate falls, and the pair lands in the cemetery.
5. If the breaker abstains, or none of their inputs distinguish, the machine runs 16 fresh inputs. One difference cancels the candidate.
6. The user composes 1 to 4 drafts `(draft expr)`, then `(use slot expr)`. The expression must contain `(call slot ...)`. The machine checks `use` on fresh inputs of their task. No call, no draft, or a test error means the candidate does not enter.
7. Entry to the canon requires all of these at once:
   - not killed by the breaker,
   - matching the target on 16 audit inputs,
   - called in a user solution that itself passes the audit.
8. The outsider composes 1 to 4 drafts and `(solve expr)` on one task from the current set (not from the measurement set), seeing only the canon. This gives a fast curve in the log. It does not touch the measurement set. The outsider starts after `admit`, so the task and the observation already include the new definition.
9. Each of the four networks gets one step on the replay.
10. Expiry and return to the bench, by the rules below.

A false counterexample does not exist as a separate text message. The breaker does not declare an expected value. They give only an input. The expectation comes from the hidden target. The breaker therefore cannot agree a "truth" with the author.

## Draft budget

This is the counterpart of what large models call thinking. There, "thinking" is a longer token string before the final answer. Weights stand still during the question. The extra cost goes into searching text. A trace in Polish or in English can be a story beside the computation, rather than the computation.

Here a draft is a term, and the machine reduces it. The trace is faithful because it can be executed.

- Author, user, and outsider: minimum 1 draft, maximum 4, then the message that counts. An author draft is `(draft (def ...))` at the task arity. A user or outsider draft is `(draft expr)` at the task arity. A wrong arity yields five `(err)` and spends one attempt.
- After each draft the harness appends `[TRY]` and five results on the public pairs: `(num k)`, or `(err)` when reduction failed on fuel, memory, or steps.
- Audit pairs and the hidden term do not appear in that reply. A draft that hits 5 public pairs can still fail the audit.
- The breaker does not draft. Their four inputs are already the search.
- Drafts do not enter the canon and do not enter the learning tape. Only an accepted final message is cloned.
- Group, solo, and random weights have the same budget 1..4. Otherwise a curve with a longer search would win without better weights.
- The force is an obligation to check once before handing in, not the content of the draft and not its length. A minimum token count in a draft would produce empty `succ` written to meet the limit.

A separate group run with budget 0, the same seed, read at round 600, says whether drafts raised the curve at all. It is not required to finish v1 at round 1200.

## Canon, cemetery, expiry

A canon entry:

```
slot
def
public_pairs          ; the subset shown at admission
audit_pairs           ; 16 pairs from admission, for a later check
last_used_round
born_round
```

The hidden target term stays in the experiment's audit log, outside the agent observation.

The cemetery holds: candidate slot, def, input, def result, target result, round number. The breaker sees the inputs and that the result split, without the target term.

Expiry: no call for 12 rounds (three full rotations) removes the entry. The slot returns to the pool. The entry also leaves the replay. An entry starts with `last_used_round = born_round` and disappears when `round >= last_used_round + 12`.

Return to the bench: every 4 rounds the oldest entry is checked again on its `audit_pairs`. A mismatch removes the entry. The breaker in that same round still searches for a new kill of the candidate. The bench concerns only the old entry, and the machine does it, not the model. `end_round` expires first, then runs the bench. A taken slot, or a 33rd definition, returns no entry and does not overwrite.

## Task generator

The generator is a program. It is not a model.

- Depth 1: identity, `succ`, `pred`, constant 0, 1, 2.
- Depth 2: a composition of two depth-1 constructions, and primitive recursion in the style of addition and bounded subtraction.
- Depth 3 and above: a composition of a depth `d-1` term with a function already in the canon, and, when the canon is empty, a deeper recursion on primitives.

The current depth grows by 1 when the outsider, on tasks of the current depth (a separate small development set, not the measurement set), exceeds 0.6 success for two evaluations in a row. v1 ceiling: depth 4.

Tasks have arity 1 or 2. An input has as many numbers as the arity. Public pairs: 5 random inputs, each coordinate from 0..6. Audit inputs: 16 random, coordinates 0..10, disjoint from the public set. At arity 1 the domain 0..10 has 11 values, so 16 audit inputs must repeat. The generator seed is stored.

The measurement set is created once, before the first round, from another seed. 50 tasks per depth, depths 1..4. No round uses them. Evaluation every 10 rounds does not update weights.

## Network

A policy over grammar productions, not a free text generator.

- Decoder transformer: 4 layers, `d_model` 128, 4 heads, `d_ff` 512, dropout 0.
- Four independent weight sets. Random initialization, four different seeds (0, 1, 2, 3).
- No shared base checkpoint.
- Grammar-constrained decoding: at each step the only legal productions are those the slot parser can still finish. The network gives logits over productions.
- Cold start: at random weights the distribution is almost flat, so the harness mixes in a sampler. With probability epsilon the production is drawn by a uniform sampler among the legal ones, otherwise a sample from the network. Epsilon starts at 1.0, multiplies by 0.995 each round, floor 0.05.
- The uniform sampler is part of the experiment. Without it the first thousand rounds are parser rejections and zero signal.

Generation length: at most 64 tokens. An unclosed term is a rejection.

## Learning step

It happens after every round, for every network, one AdamW step.

Replay data (a shared tape, sampled according to the role the agent just played):

- accepted final messages of this agent from the past (`propose`, `use`, `solve`, `break`, `abstain`), without drafts,
- definitions currently in the canon, as token sequences,
- for the breaker: confirmed kills from the cemetery.

A rejected message does not enter cloning. The loss is next-token cross entropy. Gradient clip: 1.0. Learning rate: 3e-4. Batch: 32 sequences, cut to 64 tokens.

When the tape is empty (the first rounds), the step is skipped. Weights stay as they are. From the first acceptance the step is mandatory.

This is deliberately modest learning: clone what the machine let through, plus a replay of the canon. Plasticity is real (weights after a round are different). It does not promise the invention of mathematical analysis. It promises that one can measure whether social selection changes the curve.

A measurement evaluation does not take a step. The round loop after an evaluation steps the same way as before. There is no separate "readonly mode" on the schedule.

## Baselines

The same grammar, the same machine, the same budget:

| Curve | Weights | Who proposes | Learning step | Canon |
|---|---|---|---|---|
| group | 4 networks | 4 roles | 4 steps per round | full admission rules |
| solo | 1 network | author only, 4 proposals per round | 4 steps per round | whatever passes 16 audit inputs, with no breaker and no user |
| random weights | weights never moved, outsider architecture | outsider reads the canon the group built | none | the group canon |

Comparing the group with "random weights" says how much of the result sits in the canon text, and how much in the weights. Comparing with solo says how much the breaker and the user add.

The number of machine calls per round is matched (on the order of 4 full checks plus audits). Otherwise the group would win by the checking budget alone.

## Metrics

Every 10 rounds, without a weight update, on the measurement set:

- success of the group outsider, separately per depth,
- solo success,
- success of the random-weight outsider on the group canon,
- canon size,
- cemetery size,
- fraction of candidates killed by the breaker,
- fraction of candidates killed only by the audit (false passes by the breaker),
- epsilon.

Threshold "false passes are frequent": the audit kills more than 30% of candidates the breaker let through, counted on 50 consecutive admissions to the audit. That is the only signal that v2 should add a second breaker. v1 does not add one. v1 records the fraction. The code does not turn the 30% threshold into a second breaker.

Log: `runs/<seed>/events.jsonl`, `runs/<seed>/metrics.csv`, `runs/<seed>/canon/`.

One measurement task is solved when the outsider's term agrees with the hidden target on all 16 audit pairs of that task. Agreement on 5 public pairs does not count as success. Evaluation does not append those tasks to the learning tape.

Smoke that must pass in local CI: 20 rounds, seed 0, CPU. Assertions: the process ends, group weights differ from initialization or the tape is still empty (acceptable on 20 rounds, but a step on an artificially inserted trace must change a weight: that is a unit test, not the smoke), and the serialized observation does not contain the string `hidden`.

## How to read a run

Prompts 1-12 build the instrument. A smoke of 20 rounds checks that the instrument finishes and writes a log. An empty canon after 20 rounds is acceptable. A scientific result starts only on a longer run.

Writes to `metrics.csv` happen every 10 rounds. A human does not read every 10 rounds. There are four looks, fixed in advance:

| Round | What for |
|---|---|
| 20 | Machinery: log files exist, observations have no hidden term, the process is alive. |
| 100 | A funnel, not a ranking. How many messages the parser rejected, how many the breaker killed, how many the audit killed, how many fell for lack of `call`, how many entered the canon. |
| 600 | First read of the curves. Epsilon is already at the floor 0.05, so the network proposes, not the sampler. |
| 1200 | Second read, the same curves. End of v1. The run is not extended because "one more moment and it will move". |

Epsilon falls by a factor 0.995 per round. From 1.0 to 0.05 it arrives around round 600. An earlier success at depth 1 is usually a sampler hit that the canon recorded.

On 50 tasks the error of one proportion, at a result near one half, is about 7 percentage points. A gap between two curves starts to mean something from about 20 points at the same depth, and the same sign must appear at the read of 600 and of 1200. Two points are noise.

Reading the curves, separately for each depth:

- Group clearly above solo and above random weights: the role protocol adds something, and the weights can use the canon.
- Group and random weights together, solo lower: the result sits in the canon text. The group network learned nothing beyond what can be read from the notebook.
- All three together: either the canon is empty, or the procedures in it do not transfer to the measurement set.
- Depth 1 high, depths 3 and 4 at zero: composition did not get past the primitives. That is a result, not a runner failure.

The funnel at round 100 says whether it is worth waiting until 600. Empty canon and almost only parser rejections: the sampler is not composing legal terms, a machinery error. Empty canon, legal terms, the audit kills everything: primitives miss the target, the experiment is alive, it has just not stored anything yet. Canon grows, and the outsider on the measurement set stands still: procedures pass their own tests and do not transfer. That last case must not be treated by writing a hint into the observation.

## What the 1218-event read actually was

`runs/0/` held 1218 events. The author target was one term, `(succ (zero))`, because the task seed omitted the round number. The breaker abstained 1218 times, because the message seed also omitted the round number. Admissions: 0. That read is why v2 exists and why a second breaker was not the next step. The v1 command remains `python -m kanon.run`. It appends one round by default and can continue from `resume.pt`. `--fresh` deletes the save.

## Repository map, as specified

```
PLAN.md                 this document, now docs/v1-plan.md
PROMPTS.md              implementation prompts, now summarized in docs/history.md
docs/SETUP.md           environment
README.md               how to run, once the code exists
src/kanon/language.py   tokens, parser, message shapes
src/kanon/reduce.py     reducer
src/kanon/tasks.py      generator and measurement set
src/kanon/canon.py      canon, cemetery, expiry, bench
src/kanon/roles.py      slots, rotation, observation
src/kanon/sample.py     sampler of legal productions
src/kanon/model.py      transformer, 4 weight sets
src/kanon/update.py     replay and one step
src/kanon/round.py      one round
src/kanon/baselines.py  solo and random weights
src/kanon/run.py        loop, logs, evaluation
tests/                  one file per module above
```

The handoff that used to live at `.cursor/work/kanon.md` is not part of the published tree.

## v1 parameters (locked)

| Parameter | Value |
|---|---|
| layers / d_model / heads / d_ff | 4 / 128 / 4 / 512 |
| dropout | 0 |
| canon slots | 32 (`f0`-`f31`) |
| maximum arity | 3 |
| fuel / steps / nodes | 256 / 10000 / 4096 |
| public pairs / audit | 5 / 16 |
| breaker attempts | 4 |
| expiry | 12 rounds without a call |
| bench | every 4 rounds, the oldest entry |
| lr / batch / grad clip | 3e-4 / 32 / 1.0 |
| epsilon | 1.0, times 0.995 per round, floor 0.05 |
| evaluation | every 10 rounds |
| drafts before the answer | minimum 1, maximum 4, public pairs only |
| measurement set | 50 tasks times depths 1..4 |
| depth start / ceiling | 1 / 4 |
| depth-growth threshold | 0.6 for two evaluations |
| smoke | 20 rounds, seed 0 |
| weight seeds | 0, 1, 2, 3 |
| optimizer | AdamW |

## Risks that must not be "fixed" by a prose hint

- A random network puts nothing in the canon for a long time. The answer is the sampler and epsilon, not a hint of the definition's content.
- The canon can stiffen. The answer is expiry and the bench.
- Two networks with a shared blind spot canonize an error. The audit of 16 fresh inputs matters more here than another model.
- Cloning accepted terms rewards what already passed, and weakly rewards novelty. That is why epsilon does not fall to zero.
- A smoke of 20 rounds can yield an empty canon. Smoke checks the machinery. Curves are read at rounds 600 and 1200, by the section "How to read a run".

## Where the agent count came from

Two agents leave the user inside the dispute, so a pact "I will not break you" is stable. Three close the dispute with author, breaker, and user, but nobody outside measures the canon. Four roles are exactly the different jobs one round does. A crowd of copies of the same model locks in a shared error more reliably than it finds it. A fifth agent is a second breaker, and only when the false-pass metric crosses the threshold in the Metrics section. v1 does not add that agent.
