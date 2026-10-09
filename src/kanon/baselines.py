"""Solo i losowe wagi.

Solo to jedna sieć w roli autora: cztery propozycje na rundę, każda ze
szkicami 1..4, przyjęcie dopiero po 16 parach audytowych, potem cztery
kroki AdamW. Nie ma łamacza ani użytkownika.

Losowe wagi to ta sama architektura w roli obcego. Ziarno jest zapisane
przy starcie, a pętla ewaluacji nie woła kroku uczenia i nie rusza kanonu.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kanon.canon import Canon
from kanon.language import Def, Expr, Propose, Solve
from kanon.model import Decoder, build_agent
from kanon.reduce import ReduceError, reduce_def, reduce_expr
from kanon.roles import MAX_DRAFTS, Accepted, Role, SlotSession, observe
from kanon.round import task_seed
from kanon.sample import sample
from kanon.tasks import AUDIT_COUNT, generate_task
from kanon.update import Tape, step

PROPOSALS = 4
LEARNING_STEPS = 4


@dataclass(frozen=True, slots=True)
class Attempt:
    drafts: int
    public_checks: int
    audit_pairs: int
    admitted: str | None


@dataclass(frozen=True, slots=True)
class SoloRound:
    attempts: tuple[Attempt, ...]
    learning_steps: int
    tries: int
    audits: int


@dataclass(slots=True)
class Frozen:
    """Obcy o wagach z ziarna startu. Krok uczenia do niego nie sięga."""

    seed: int
    agent: Decoder


@dataclass(frozen=True, slots=True)
class OutsiderRound:
    drafts: int
    public_checks: int
    audit_pairs: int
    solved: bool


def solo_round(
    agent: Decoder,
    canon: Canon,
    tape: Tape,
    *,
    round_no: int,
    depth: int,
    seed: int,
    proposals: Sequence[Sequence[str]] | None = None,
    learn: bool = True,
    choose=None,
    close: bool = True,
) -> SoloRound:
    """Cztery propozycje autora. Krok jest przed wygasaniem, chyba że pomiar ma być pierwszy."""
    _round(round_no, depth, seed)
    queues = _queues(proposals, PROPOSALS)
    attempts: list[Attempt] = []
    for index, queue in enumerate(queues):
        attempts.append(
            _author_attempt(canon, queue, round_no, depth, seed, index, tape, choose)
        )
    tape.note_canon(canon)
    taken = LEARNING_STEPS if learn else 0
    if learn:
        for _ in range(LEARNING_STEPS):
            step((agent,), tape, agent=0, role=Role.AUTHOR)
    if close:
        canon.end_round(round_no)
        tape.note_canon(canon)
    tries = sum(item.drafts for item in attempts)
    audits = sum(1 for item in attempts if item.audit_pairs == AUDIT_COUNT)
    return SoloRound(tuple(attempts), taken, tries, audits)


def start_frozen(seed: int) -> Frozen:
    if type(seed) is not int:
        raise ValueError("seed")
    return Frozen(seed, build_agent(seed))


def frozen_eval(
    frozen: Frozen,
    canon: Canon,
    *,
    rounds: int,
    depth: int,
    seed: int,
    scripts: Sequence[Sequence[str]] | None = None,
) -> tuple[OutsiderRound, ...]:
    """Rundy obcego na kanonie grupy. Wag i kanonu nie zapisuje."""
    _round(0, depth, seed)
    if type(rounds) is not int or rounds < 1:
        raise ValueError("rounds")
    queues = _queues(scripts, rounds)
    played = []
    for index, queue in enumerate(queues):
        played.append(_outsider_attempt(frozen.agent, canon, queue, depth, seed, index))
    return tuple(played)


def _author_attempt(
    canon: Canon,
    queue: list[str] | None,
    round_no: int,
    depth: int,
    seed: int,
    index: int,
    tape: Tape,
    choose=None,
) -> Attempt:
    defs = tuple(entry.definition for entry in canon.replay())
    task = generate_task(task_seed(seed, index), depth, canon=defs)
    free = canon.free_slots()
    slot = free[0] if free else None
    if slot is None:
        return Attempt(0, 0, 0, None)
    session = SlotSession(
        Role.AUTHOR,
        task.view.public_pairs,
        canon=canon,
        free_slot=slot,
    )
    observation = observe(
        Role.AUTHOR,
        canon=canon,
        pairs=task.view.public_pairs,
        free_slot=slot,
    )
    proposed: Def | None = None
    source = ""
    for attempt in range(MAX_DRAFTS * 2):
        if session.closed:
            break
        message = _pull(queue, "[AUTHOR]", observation, seed + index * 100 + attempt, choose)
        if message is None:
            break
        result = session.submit(message)
        if isinstance(result, Accepted) and isinstance(result.message, Propose):
            proposed = result.message.definition
            source = message
            break
        if isinstance(result, str) and result.startswith("[TRY]"):
            observation = f"{observation} {result}"
    audit_pairs = 0
    admitted = None
    if proposed is not None:
        audit_pairs = len(task.audit_pairs)
        if audit_pairs == AUDIT_COUNT and _def_matches(proposed, task.audit_pairs, canon):
            admitted = canon.admit(
                proposed,
                task.view.public_pairs,
                task.audit_pairs,
                round_no,
            )
    if admitted is not None:
        tape.note(0, Role.AUTHOR, source, accepted=True)
    public = session.drafts * len(task.view.public_pairs)
    return Attempt(session.drafts, public, audit_pairs, admitted)


def _outsider_attempt(
    agent: Decoder,
    canon: Canon,
    queue: list[str] | None,
    depth: int,
    seed: int,
    index: int,
) -> OutsiderRound:
    if not isinstance(agent, Decoder):
        raise ValueError("agent")
    defs = tuple(entry.definition for entry in canon.replay())
    task = generate_task(task_seed(seed, index), depth, canon=defs)
    session = SlotSession(Role.OUTSIDER, task.view.public_pairs, canon=canon)
    observation = observe(Role.OUTSIDER, canon=canon, pairs=task.view.public_pairs)
    solved_expr = None
    for attempt in range(MAX_DRAFTS * 2):
        if session.closed:
            break
        message = _pull(queue, "[OUTSIDER]", observation, seed + 1_000 + index * 100 + attempt)
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
    return OutsiderRound(session.drafts, public, audit_pairs, solved)


def _pull(
    queue: list[str] | None,
    marker: str,
    observation: str,
    seed: int,
    choose=None,
) -> str | None:
    if queue is not None:
        if not queue:
            return None
        return queue.pop(0)
    if choose is not None:
        return choose(Role.AUTHOR, observation, seed)
    return sample(marker, observation, seed=seed)


def _queues(
    scripts: Sequence[Sequence[str]] | None,
    count: int,
) -> list[list[str] | None]:
    if scripts is None:
        return [None] * count
    if len(scripts) != count:
        raise ValueError("scripts")
    return [list(item) for item in scripts]


def _def_matches(defn: Def, pairs, canon: Canon) -> bool:
    defs = {entry.definition.slot: entry.definition for entry in canon.replay()}
    defs[defn.slot] = defn
    for pair in pairs:
        try:
            got = reduce_def(defn, pair.inputs, defs=defs)
        except (ReduceError, ValueError):
            return False
        if got != pair.output:
            return False
    return True


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


def _round(round_no: int, depth: int, seed: int) -> None:
    if type(round_no) is not int or round_no < 0:
        raise ValueError("round")
    if depth not in (1, 2, 3, 4):
        raise ValueError("depth")
    if type(seed) is not int:
        raise ValueError("seed")
