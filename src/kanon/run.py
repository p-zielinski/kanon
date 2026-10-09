"""Pętla v1: runda grupy, pomiar, potem krok.

Zbiór pomiarowy powstaje raz, przed rundą 0, z ziarna innego niż ziarno rund,
i ląduje w pliku. Głębokość startuje od 1. Rośnie o 1, gdy obcy na małym
zestawie rozwojowym tej głębokości przekroczy 0,6 przez dwie kolejne
ewaluacje. Sufit to 4.

Ewaluacja nie woła kroku i nie dopisuje zadań do taśmy. Szkice 1..4 liczą
się na pięciu parach publicznych. Zaliczenie wymaga 16 par audytowych.
Przy ε = 1 komunikat bierze sampler. Poniżej 1 każdy krok produkcji miesza
sampler z siecią. Krok grupy i cztery kroki solo idą po pomiarze.

Jedno wywołanie dopisuje rundy do zapisu tego ziarna. Bez `--fresh` kolejny
start idzie dalej od ostatniej dokończonej rundy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from kanon.baselines import LEARNING_STEPS, solo_round, start_frozen
from kanon.canon import Canon, Entry, Grave
from kanon.language import Expr, Solve, parse_def
from kanon.model import Decoder, build_agent, build_agents
from kanon.reduce import ReduceError, reduce_expr
from kanon.roles import (
    MAX_DRAFTS,
    Accepted,
    Role,
    SlotSession,
    assign,
    format_def,
    format_expr,
    observe,
)
from kanon.round import EPSILON, dumps_event, play
from kanon.sample import speak
from kanon.tasks import AUDIT_COUNT, Holdout, IOPair, TaskAudit, build_holdout, generate_task
from kanon.update import Tape, step, step_all

EVAL_EVERY = 10
DEPTH_START = 1
DEPTH_MAX = 4
GROWTH = 0.6
GROWTH_STREAK = 2
DEV_PER_DEPTH = 10
EPSILON_DECAY = 0.995
EPSILON_FLOOR = 0.05
HOLDOUT_GAP = 1_000_003
DEV_GAP = 2_000_006
FROZEN_SEED = 4
SOLO_SEED = 5
AUDIT_WINDOW = 50
SMOKE_ROUNDS = 20
SMOKE_SEED = 0

METRICS = (
    "round",
    "epsilon",
    "depth",
    "canon",
    "cemetery",
    "breaker_rate",
    "audit_miss_rate",
    "group_1",
    "group_2",
    "group_3",
    "group_4",
    "solo_1",
    "solo_2",
    "solo_3",
    "solo_4",
    "frozen_1",
    "frozen_2",
    "frozen_3",
    "frozen_4",
)

_ROLE_NAMES = {
    "author": Role.AUTHOR,
    "breaker": Role.BREAKER,
    "user": Role.USER,
    "outsider": Role.OUTSIDER,
}
_MARKERS = {
    Role.AUTHOR: "[AUTHOR]",
    Role.BREAKER: "[BREAK]",
    Role.USER: "[USE]",
    Role.OUTSIDER: "[OUTSIDER]",
}


@dataclass(frozen=True, slots=True)
class SolveAttempt:
    drafts: int
    public_checks: int
    audit_pairs: int
    solved: bool


def holdout_seed(seed: int) -> int:
    """Ziarno zbioru pomiarowego, inne niż ziarno rund."""
    _seed(seed)
    return seed + HOLDOUT_GAP


def dev_seed(seed: int) -> int:
    """Ziarno małego zestawu rozwojowego, osobne od rund i od pomiaru."""
    _seed(seed)
    return seed + DEV_GAP


def decay_epsilon(epsilon: float) -> float:
    nxt = float(epsilon) * EPSILON_DECAY
    if nxt < EPSILON_FLOOR:
        return EPSILON_FLOOR
    return nxt


def advance_depth(depth: int, streak: int, rate: float) -> tuple[int, int]:
    """Dwie kolejne ewaluacje powyżej 0,6 podnoszą głębokość. Sufit to 4."""
    if depth not in (1, 2, 3, 4):
        raise ValueError("depth")
    followed = streak + 1 if rate > GROWTH else 0
    if followed >= GROWTH_STREAK and depth < DEPTH_MAX:
        return depth + 1, 0
    return depth, followed


def attempt_task(
    canon: Canon,
    task: TaskAudit,
    *,
    seed: int,
    messages: Sequence[str] | None = None,
    agent: Decoder | None = None,
    epsilon: float = EPSILON,
) -> SolveAttempt:
    """Szkice na parach publicznych, zaliczenie dopiero po 16 parach audytowych."""
    if type(seed) is not int:
        raise ValueError("seed")
    session = SlotSession(Role.OUTSIDER, task.view.public_pairs, canon=canon)
    observation = observe(Role.OUTSIDER, canon=canon, pairs=task.view.public_pairs)
    queue = None if messages is None else list(messages)
    solved_expr: Expr | None = None
    for offset in range(MAX_DRAFTS * 2):
        if session.closed:
            break
        message = _pull(queue, observation, seed + offset, agent, epsilon)
        if message is None:
            break
        result = session.submit(message)
        if isinstance(result, Accepted) and isinstance(result.message, Solve):
            solved_expr = result.message.expr
            break
        if isinstance(result, str) and result.startswith("[TRY]"):
            observation = f"{observation} {result}"
    audit_pairs = 0
    solved = False
    if solved_expr is not None:
        audit_pairs = len(task.audit_pairs)
        solved = audit_pairs == AUDIT_COUNT and _expr_matches(solved_expr, task.audit_pairs, canon)
    public = session.drafts * len(task.view.public_pairs)
    return SolveAttempt(session.drafts, public, audit_pairs, solved)


def score_tasks(
    canon: Canon,
    tasks: Sequence[TaskAudit],
    *,
    seed: int,
    agent: Decoder | None = None,
    epsilon: float = EPSILON,
) -> float:
    if not tasks:
        return 0.0
    hits = 0
    for index, task in enumerate(tasks):
        solved = attempt_task(
            canon,
            task,
            seed=seed + index * 16,
            agent=agent,
            epsilon=epsilon,
        ).solved
        if solved:
            hits += 1
    return hits / len(tasks)


def run(
    *,
    seed: int = SMOKE_SEED,
    rounds: int = 1,
    root: Path | str = "runs",
    fresh: bool = False,
) -> Path:
    """Dopisuje `rounds` rund do zapisu ziarna. Bez `fresh` kontynuuje poprzedni zapis."""
    _seed(seed)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds")
    directory = Path(root) / str(seed)
    canon_dir = directory / "canon"
    canon_dir.mkdir(parents=True, exist_ok=True)
    resume_path = directory / "resume.pt"
    events_path = directory / "events.jsonl"
    metrics_path = directory / "metrics.csv"
    if fresh:
        _wipe(directory, canon_dir, resume_path, events_path, metrics_path)
    saved = None if not resume_path.is_file() else torch.load(resume_path, weights_only=False)
    holdout = build_holdout(holdout_seed(seed))
    if not (directory / "holdout.json").is_file():
        _write_holdout(directory / "holdout.json", holdout)
    developmental = _dev_tasks(dev_seed(seed))
    by_depth = {depth: holdout.by_depth(depth) for depth in (1, 2, 3, 4)}
    agents = build_agents()
    solo_agent = build_agent(SOLO_SEED)
    solo_canon = Canon()
    solo_tape = Tape()
    frozen = start_frozen(FROZEN_SEED)
    group = Canon()
    tape = Tape()
    if saved is None:
        completed = 0
        depth = DEPTH_START
        streak = 0
        epsilon = EPSILON
        proposed = 0
        broken_n = 0
        audit_flags: list[bool] = []
        initial = tuple(_weight_hash(agent) for agent in agents)
        events_path.write_text("", encoding="utf-8")
        _write_header(metrics_path)
    else:
        completed = int(saved["completed"])
        depth = int(saved["depth"])
        streak = int(saved["streak"])
        epsilon = float(saved["epsilon"])
        proposed = int(saved["proposed"])
        broken_n = int(saved["broken_n"])
        audit_flags = [bool(item) for item in saved["audit_flags"]]
        initial = tuple(saved["initial"])
        _install_weights(agents, saved["group_weights"])
        _install_weights((solo_agent,), (saved["solo_weights"],))
        _install_canon(group, saved["group_canon"])
        _install_canon(solo_canon, saved["solo_canon"])
        tape.import_traces(saved["group_traces"])
        solo_tape.import_traces(saved["solo_traces"])
        tape.note_canon(group)
        tape.note_breaks(group)
        solo_tape.note_canon(solo_canon)
        for agent, state in zip(agents, saved["group_optim"], strict=True):
            tape.import_optimizer(agent, state)
        solo_tape.import_optimizer(solo_agent, saved["solo_optim"])
        _align_events(events_path, completed, saved["last_line"])
        if not metrics_path.is_file():
            _write_header(metrics_path)
    for _ in range(rounds):
        round_no = completed
        owners = _agent_roles(round_no)
        event = play(
            group,
            round_no=round_no,
            depth=depth,
            seed=seed,
            epsilon=epsilon,
            choose=_policy(agents, owners, epsilon),
            close=False,
        )
        line = dumps_event(event) + "\n"
        if event.broken or event.audit_ran:
            proposed += 1
        if event.broken:
            broken_n += 1
        if event.audit_ran:
            audit_flags.append(not event.audit_ok)
        _note(tape, group, event)
        solo_round(
            solo_agent,
            solo_canon,
            solo_tape,
            round_no=round_no,
            depth=depth,
            seed=seed,
            learn=False,
            choose=_solo_policy(solo_agent, epsilon),
            close=False,
        )
        if (round_no + 1) % EVAL_EVERY == 0:
            outsider = agents[owners.index(Role.OUTSIDER)]
            _write_metrics(
                metrics_path,
                completed=round_no + 1,
                epsilon=epsilon,
                depth=depth,
                group=group,
                proposed=proposed,
                broken_n=broken_n,
                audit_flags=audit_flags,
                by_depth=by_depth,
                solo_canon=solo_canon,
                seed=seed,
                group_agent=outsider,
                solo_agent=solo_agent,
                frozen_agent=frozen.agent,
            )
            rate = score_tasks(
                group,
                developmental[depth],
                seed=seed + 80_000 + round_no,
                agent=outsider,
                epsilon=epsilon,
            )
            depth, streak = advance_depth(depth, streak, rate)
            _snapshot(group, canon_dir / "snapshot.json", round_no + 1)
        step_all(agents, tape, owners)
        for _ in range(LEARNING_STEPS):
            step((solo_agent,), solo_tape, agent=0, role=Role.AUTHOR)
        group.end_round(round_no)
        solo_canon.end_round(round_no)
        tape.note_canon(group)
        solo_tape.note_canon(solo_canon)
        epsilon = decay_epsilon(epsilon)
        completed += 1
        _commit(
            resume_path,
            events_path,
            line,
            completed=completed,
            epsilon=epsilon,
            depth=depth,
            streak=streak,
            proposed=proposed,
            broken_n=broken_n,
            audit_flags=audit_flags,
            initial=initial,
            group=group,
            solo_canon=solo_canon,
            tape=tape,
            solo_tape=solo_tape,
            agents=agents,
            solo_agent=solo_agent,
        )
        print(f"runda {round_no} zapisana, {directory}", file=sys.stderr)
    _snapshot(group, canon_dir / "snapshot.json", completed)
    if _weight_hash(frozen.agent) != _weight_hash(build_agent(frozen.seed)):
        raise RuntimeError("frozen weights")
    changed = tuple(_weight_hash(agent) for agent in agents) != initial
    empty = _tape_empty(tape)
    (directory / "weight_check.json").write_text(
        json.dumps({"tape_empty": empty, "weights_changed": changed}) + "\n",
        encoding="utf-8",
    )
    if empty == changed:
        raise RuntimeError("group step")
    return directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kanon.run")
    parser.add_argument("--seed", type=int, default=SMOKE_SEED)
    parser.add_argument("--rounds", type=int, default=1, help="ile rund dopisać do zapisu")
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="usuwa zapis tego ziarna i zaczyna od rundy 0",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(seed=args.seed, rounds=args.rounds, root=args.root, fresh=args.fresh)
    return 0


def _wipe(directory: Path, canon_dir: Path, resume_path: Path, events_path: Path, metrics_path: Path) -> None:
    for path in (resume_path, events_path, metrics_path, directory / "weight_check.json", directory / "holdout.json"):
        if path.is_file():
            path.unlink()
    snapshot = canon_dir / "snapshot.json"
    if snapshot.is_file():
        snapshot.unlink()


def _commit(
    resume_path: Path,
    events_path: Path,
    line: str,
    *,
    completed: int,
    epsilon: float,
    depth: int,
    streak: int,
    proposed: int,
    broken_n: int,
    audit_flags: list[bool],
    initial: tuple[str, ...],
    group: Canon,
    solo_canon: Canon,
    tape: Tape,
    solo_tape: Tape,
    agents: Sequence,
    solo_agent,
) -> None:
    payload = {
        "completed": completed,
        "epsilon": epsilon,
        "depth": depth,
        "streak": streak,
        "proposed": proposed,
        "broken_n": broken_n,
        "audit_flags": list(audit_flags),
        "initial": list(initial),
        "last_line": line,
        "group_canon": _canon_payload(group),
        "solo_canon": _canon_payload(solo_canon),
        "group_traces": list(tape.export_traces()),
        "solo_traces": list(solo_tape.export_traces()),
        "group_weights": [_state(agent) for agent in agents],
        "solo_weights": _state(solo_agent),
        "group_optim": [tape.export_optimizer(agent) for agent in agents],
        "solo_optim": solo_tape.export_optimizer(solo_agent),
    }
    temporary = resume_path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(resume_path)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()


def _align_events(path: Path, completed: int, last_line: str) -> None:
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = [line for line in text.splitlines() if line]
    if len(lines) > completed:
        body = "".join(f"{line}\n" for line in lines[:completed])
        path.write_text(body, encoding="utf-8")
        return
    if len(lines) == completed:
        return
    if len(lines) == completed - 1:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(last_line if last_line.endswith("\n") else last_line + "\n")
        return
    raise ValueError("log")


def _canon_payload(canon: Canon) -> dict:
    return {
        "entries": [
            {
                "def": format_def(entry.definition),
                "public": _pairs_payload(entry.public_pairs),
                "audit": _pairs_payload(entry.audit_pairs),
                "last_used_round": entry.last_used_round,
                "born_round": entry.born_round,
            }
            for entry in canon.replay()
        ],
        "graves": [
            {
                "slot": grave.slot,
                "def": format_def(grave.definition),
                "inputs": list(grave.inputs),
                "def_result": grave.def_result,
                "target_result": grave.target_result,
                "round": grave.round_no,
            }
            for grave in canon.graves
        ],
    }


def _install_canon(canon: Canon, payload: dict) -> None:
    entries = []
    for item in payload["entries"]:
        definition = parse_def(item["def"])
        entries.append(
            Entry(
                slot=definition.slot,
                definition=definition,
                public_pairs=_pairs_from(item["public"]),
                audit_pairs=_pairs_from(item["audit"]),
                last_used_round=int(item["last_used_round"]),
                born_round=int(item["born_round"]),
            )
        )
    graves = []
    for item in payload["graves"]:
        graves.append(
            Grave(
                slot=item["slot"],
                definition=parse_def(item["def"]),
                inputs=tuple(int(value) for value in item["inputs"]),
                def_result=int(item["def_result"]),
                target_result=int(item["target_result"]),
                round_no=int(item["round"]),
            )
        )
    canon.restore(entries, graves)


def _pairs_payload(pairs: Sequence[IOPair]) -> list[dict]:
    return [{"inputs": list(pair.inputs), "output": pair.output} for pair in pairs]


def _pairs_from(rows: Sequence[dict]) -> tuple[IOPair, ...]:
    return tuple(IOPair(tuple(int(value) for value in row["inputs"]), int(row["output"])) for row in rows)


def _state(module) -> dict:
    reader = getattr(module, "state_dict", None)
    if reader is None:
        return {}
    return reader()


def _install_weights(modules: Sequence, states: Sequence[dict]) -> None:
    for module, state in zip(modules, states, strict=True):
        writer = getattr(module, "load_state_dict", None)
        if writer is None:
            continue
        writer(state)


def _write_metrics(
    path: Path,
    *,
    completed: int,
    epsilon: float,
    depth: int,
    group: Canon,
    proposed: int,
    broken_n: int,
    audit_flags: list[bool],
    by_depth: dict[int, tuple[TaskAudit, ...]],
    solo_canon: Canon,
    seed: int,
    group_agent: Decoder,
    solo_agent: Decoder,
    frozen_agent: Decoder,
) -> None:
    base = seed + completed * 100_003
    group_rates = [
        score_tasks(
            group,
            by_depth[level],
            seed=base + level * 1_000,
            agent=group_agent,
            epsilon=epsilon,
        )
        for level in (1, 2, 3, 4)
    ]
    solo_rates = [
        score_tasks(
            solo_canon,
            by_depth[level],
            seed=base + 20_000 + level * 1_000,
            agent=solo_agent,
            epsilon=epsilon,
        )
        for level in (1, 2, 3, 4)
    ]
    frozen_rates = [
        score_tasks(
            group,
            by_depth[level],
            seed=base + 40_000 + level * 1_000,
            agent=frozen_agent,
            epsilon=epsilon,
        )
        for level in (1, 2, 3, 4)
    ]
    breaker = broken_n / proposed if proposed else 0.0
    window = audit_flags[-AUDIT_WINDOW:]
    missed = sum(window) / len(window) if window else 0.0
    row = [
        completed,
        f"{epsilon:.6f}",
        depth,
        len(group.replay()),
        len(group.graves),
        f"{breaker:.6f}",
        f"{missed:.6f}",
        *[f"{rate:.6f}" for rate in (*group_rates, *solo_rates, *frozen_rates)],
    ]
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(row)


def _dev_tasks(seed: int) -> dict[int, tuple[TaskAudit, ...]]:
    found: dict[int, tuple[TaskAudit, ...]] = {}
    for depth in (1, 2, 3, 4):
        rows = [
            generate_task(seed + depth * 1_000 + index, depth)
            for index in range(DEV_PER_DEPTH)
        ]
        found[depth] = tuple(rows)
    return found


def _note(tape: Tape, canon: Canon, event) -> None:
    owners = {
        Role.AUTHOR: event.roles[0],
        Role.BREAKER: event.roles[1],
        Role.USER: event.roles[2],
        Role.OUTSIDER: event.roles[3],
    }
    for name, message, reply in event.steps:
        if reply != "accepted":
            continue
        role = _ROLE_NAMES[name]
        tape.note(owners[role], role, message, accepted=True)
    tape.note_canon(canon)
    tape.note_breaks(canon)


def _agent_roles(round_no: int) -> tuple[Role, Role, Role, Role]:
    held = assign(round_no)
    roles = {
        held.author: Role.AUTHOR,
        held.breaker: Role.BREAKER,
        held.user: Role.USER,
        held.outsider: Role.OUTSIDER,
    }
    return tuple(roles[index] for index in range(4))


def _pull(
    queue: list[str] | None,
    observation: str,
    seed: int,
    agent: Decoder | None,
    epsilon: float,
) -> str | None:
    if queue is not None:
        if not queue:
            return None
        return queue.pop(0)
    return speak("[OUTSIDER]", observation, seed=seed, epsilon=epsilon, agent=agent)


def _policy(agents: Sequence[Decoder], owners: Sequence[Role], epsilon: float):
    def choose(role: Role, observation: str, seed: int) -> str:
        return speak(
            _MARKERS[role],
            observation,
            seed=seed,
            epsilon=epsilon,
            agent=agents[owners.index(role)],
        )

    return choose


def _solo_policy(agent: Decoder, epsilon: float):
    def choose(role: Role, observation: str, seed: int) -> str:
        del role
        return speak("[AUTHOR]", observation, seed=seed, epsilon=epsilon, agent=agent)

    return choose


def _write_holdout(path: Path, holdout: Holdout) -> None:
    payload = {
        "seed": holdout.seed,
        "tasks": [
            {
                "depth": task.view.depth,
                "arity": task.view.arity,
                "target": format_expr(task.target),
                "public": [
                    {"inputs": list(pair.inputs), "output": pair.output}
                    for pair in task.view.public_pairs
                ],
                "audit": [
                    {"inputs": list(pair.inputs), "output": pair.output}
                    for pair in task.audit_pairs
                ],
            }
            for task in holdout.tasks
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def _tape_empty(tape: Tape) -> bool:
    return all(not tape.sources(index, role) for index in range(4) for role in Role)


def _expr_matches(expr: Expr, pairs, canon: Canon) -> bool:
    defs = {entry.definition.slot: entry.definition for entry in canon.replay()}
    for pair in pairs:
        try:
            got = reduce_expr(expr, defs=defs, env=pair.inputs)
        except (ReduceError, ValueError):
            return False
        if got != pair.output:
            return False
    return True


def _snapshot(canon: Canon, path: Path, completed: int) -> None:
    payload = {
        "round": completed,
        "entries": [
            {
                "slot": entry.slot,
                "def": format_def(entry.definition),
                "born_round": entry.born_round,
                "last_used_round": entry.last_used_round,
            }
            for entry in canon.replay()
        ],
        "graves": [
            {
                "slot": grave.slot,
                "def": format_def(grave.definition),
                "inputs": list(grave.inputs),
                "def_result": grave.def_result,
                "target_result": grave.target_result,
                "round": grave.round_no,
            }
            for grave in canon.graves
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_header(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerow(METRICS)


def _weight_hash(agent) -> str:
    digest = hashlib.sha256()
    for name, param in agent.named_parameters():
        data = param.detach().cpu().contiguous()
        if data.storage_offset() != 0:
            data = data.clone()
        raw = bytes(data.untyped_storage())[: data.nbytes]
        digest.update(name.encode())
        digest.update(raw)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    if type(seed) is not int or seed < 0:
        raise ValueError("seed")


if __name__ == "__main__":
    raise SystemExit(main())
