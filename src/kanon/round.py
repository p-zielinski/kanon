"""Jedna runda: szkice, łamacz, audyt, użytkownik, obcy.

Przy ε = 1 komunikat bierze sampler. Poniżej 1 polityka `choose` miesza
sampler z siecią na masce legalnych tokenów. Wstrzyknięty skrypt zastępuje
politykę. Krok uczenia jest poza tą rundą.

Do kanonu wchodzi definicja niezbita przez łamacza, zgodna z targetem na
16 wejściach audytowych i wywołana w rozwiązaniu użytkownika, które samo
te wejścia przechodzi. Szkic liczy się tylko na pięciu parach publicznych.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from kanon.canon import Canon
from kanon.language import (
    SLOTS,
    Call,
    Def,
    Expr,
    If0,
    Pred,
    Propose,
    Self,
    Solve,
    Succ,
    Use,
)
from kanon.language import Break as BreakMessage
from kanon.reduce import ReduceError, reduce_def, reduce_expr
from kanon.roles import Accepted, Role, SlotSession, assign, format_expr, observe
from kanon.sample import sample
from kanon.tasks import IOPair, TaskAudit, generate_task

EPSILON = 1.0
_MARKERS = {
    Role.AUTHOR: "[AUTHOR]",
    Role.BREAKER: "[BREAK]",
    Role.USER: "[USE]",
    Role.OUTSIDER: "[OUTSIDER]",
}


@dataclass(frozen=True, slots=True)
class Script:
    """Komunikaty wstawione zamiast polityki. Pusta krotka nic nie wysyła."""

    author: tuple[str, ...] = ()
    breaker: tuple[str, ...] = ()
    user: tuple[str, ...] = ()
    outsider: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RoundEvent:
    round_no: int
    depth: int
    seed: int
    roles: tuple[int, int, int, int]
    admitted: str | None
    audit_ran: bool
    broken: bool
    graves_added: int
    outsider_ok: bool
    observations: dict[str, str]
    steps: tuple[tuple[str, str, str], ...]
    audit_ok: bool = False
    epsilon: float = EPSILON
    targets: tuple[tuple[str, str], ...] = ()

    def to_json(self) -> dict:
        author, breaker, user, outsider = self.roles
        return {
            "round": self.round_no,
            "depth": self.depth,
            "seed": self.seed,
            "epsilon": self.epsilon,
            "roles": {
                "author": author,
                "breaker": breaker,
                "user": user,
                "outsider": outsider,
            },
            "admitted": self.admitted,
            "audit_ran": self.audit_ran,
            "audit_ok": self.audit_ok,
            "broken": self.broken,
            "graves_added": self.graves_added,
            "outsider_ok": self.outsider_ok,
            "observations": self.observations,
            "targets": {role: term for role, term in self.targets},
            "steps": [
                {"role": role, "message": message, "reply": reply}
                for role, message, reply in self.steps
            ],
        }


def task_seed(seed: int, index: int) -> int:
    """0 autor, 1 użytkownik, 2 obcy. Zadanie rundy, nie zbiór pomiarowy."""
    return seed * 1_000_003 + index + 1


def play(
    canon: Canon,
    *,
    round_no: int,
    depth: int,
    seed: int,
    script: Script | None = None,
    legal=None,
    logits=None,
    epsilon: float = EPSILON,
    choose=None,
    close: bool = True,
) -> RoundEvent:
    if type(round_no) is not int or round_no < 0:
        raise ValueError("round")
    if depth not in (1, 2, 3, 4):
        raise ValueError("depth")
    if type(seed) is not int:
        raise ValueError("seed")
    epsilon = _epsilon(epsilon)
    if script is None and epsilon < 1 and choose is None:
        raise ValueError("choose")
    roles = assign(round_no)
    defs = tuple(entry.definition for entry in canon.replay())
    author_task = generate_task(task_seed(seed, 0), depth, canon=defs)
    user_task = generate_task(task_seed(seed, 1), depth, canon=defs)
    free = canon.free_slots()
    free_slot = free[0] if free else None
    graves_before = len(canon.graves)
    steps: list[tuple[str, str, str]] = []
    observations = {
        "author": _author_view(canon, author_task, free_slot),
    }
    proposed = _author_phase(
        canon,
        author_task,
        free_slot,
        observations["author"],
        _queue(script, "author"),
        seed,
        steps,
        legal,
        logits,
        choose,
    )
    observations["breaker"] = observe(
        Role.BREAKER,
        canon=canon,
        pairs=author_task.view.public_pairs,
        candidate=proposed,
    )
    observations["user"] = observe(
        Role.USER,
        canon=canon,
        pairs=user_task.view.public_pairs,
        candidate=proposed,
    )
    broken = False
    if proposed is not None:
        broken = _breaker_phase(
            canon,
            author_task,
            proposed,
            _queue(script, "breaker"),
            seed,
            steps,
            legal,
            logits,
            round_no,
            choose,
        )
    audit_ran = False
    audit_ok = False
    if proposed is not None and not broken:
        audit_ran = True
        audit_ok = _def_matches(proposed, author_task.audit_pairs, _defs(canon))
    use_expr = _user_phase(
        canon,
        user_task,
        proposed,
        _queue(script, "user"),
        seed,
        steps,
        legal,
        logits,
        choose,
    )
    user_ok = (
        proposed is not None
        and use_expr is not None
        and _calls(use_expr, proposed.slot)
        and _expr_matches(
            use_expr,
            user_task.audit_pairs,
            {**_defs(canon), proposed.slot: proposed},
        )
    )
    admitted = None
    if proposed is not None and not broken and audit_ok and user_ok and free_slot is not None:
        admitted = canon.admit(
            proposed,
            author_task.view.public_pairs,
            author_task.audit_pairs,
            round_no,
        )
        if admitted is not None and use_expr is not None:
            canon.note_use(use_expr, round_no)
    outsider_task = generate_task(
        task_seed(seed, 2),
        depth,
        canon=tuple(entry.definition for entry in canon.replay()),
    )
    observations["outsider"] = observe(
        Role.OUTSIDER,
        canon=canon,
        pairs=outsider_task.view.public_pairs,
    )
    outsider_ok = _outsider_phase(
        canon,
        outsider_task,
        _queue(script, "outsider"),
        seed,
        steps,
        legal,
        logits,
        choose,
    )
    if close:
        canon.end_round(round_no)
    return RoundEvent(
        round_no=round_no,
        depth=depth,
        seed=seed,
        roles=(roles.author, roles.breaker, roles.user, roles.outsider),
        admitted=admitted,
        audit_ran=audit_ran,
        broken=broken,
        graves_added=len(canon.graves) - graves_before,
        outsider_ok=outsider_ok,
        observations=observations,
        steps=tuple(steps),
        audit_ok=audit_ok,
        epsilon=epsilon,
        targets=(
            ("author", format_expr(author_task.target)),
            ("user", format_expr(user_task.target)),
            ("outsider", format_expr(outsider_task.target)),
        ),
    )


def dumps_event(event: RoundEvent) -> str:
    return json.dumps(event.to_json(), ensure_ascii=False)


def _epsilon(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("epsilon")
    number = float(value)
    if not 0 < number <= 1:
        raise ValueError("epsilon")
    return number


def _author_view(canon: Canon, task: TaskAudit, free_slot: str | None) -> str:
    return observe(
        Role.AUTHOR,
        canon=canon,
        pairs=task.view.public_pairs,
        free_slot=free_slot,
    )


def _author_phase(
    canon,
    task,
    free_slot,
    observation,
    queue,
    seed,
    steps,
    legal,
    logits,
    choose,
) -> Def | None:
    if free_slot is None:
        return None
    session = SlotSession(
        Role.AUTHOR,
        task.view.public_pairs,
        canon=canon,
        free_slot=free_slot,
    )
    local = observation
    proposed = None
    for attempt in range(8):
        if session.closed:
            break
        message = _pull(queue, Role.AUTHOR, local, seed + attempt, legal, logits, choose)
        if message is None:
            break
        result = session.submit(message)
        if isinstance(result, Accepted) and isinstance(result.message, Propose):
            steps.append(("author", message, "accepted"))
            proposed = result.message.definition
            break
        reply = result if isinstance(result, str) else "accepted"
        steps.append(("author", message, reply))
        if isinstance(result, str) and result.startswith("[TRY]"):
            local = f"{local} {result}"
    return proposed


def _breaker_phase(
    canon,
    task,
    proposed: Def,
    queue,
    seed,
    steps,
    legal,
    logits,
    round_no: int,
    choose,
) -> bool:
    session = SlotSession(Role.BREAKER, task.view.public_pairs, canon=canon)
    local = observe(
        Role.BREAKER,
        canon=canon,
        pairs=task.view.public_pairs,
        candidate=proposed,
    )
    for attempt in range(8):
        if session.closed:
            break
        message = _pull(queue, Role.BREAKER, local, seed + 20 + attempt, legal, logits, choose)
        if message is None:
            break
        result = session.submit(message)
        reply = "accepted" if isinstance(result, Accepted) else result
        steps.append(("breaker", message, reply))
        if not isinstance(result, Accepted):
            continue
        if isinstance(result.message, BreakMessage):
            if _breaker_hit(canon, task, proposed, result.message, round_no):
                return True
        else:
            return False
    return False


def _breaker_hit(canon, task, proposed: Def, message: BreakMessage, round_no: int) -> bool:
    if message.slot != proposed.slot or len(message.args) != proposed.arity:
        return False
    want = _value(task.target, message.args, _defs(canon), task.canon)
    got = _defined(proposed, message.args, _defs(canon))
    if want is None or (got is not None and got == want):
        return False
    canon.bury(
        slot=proposed.slot,
        definition=proposed,
        inputs=message.args,
        def_result=got if got is not None else want + 1,
        target_result=want,
        round_no=round_no,
    )
    return True


def _user_phase(canon, task, proposed, queue, seed, steps, legal, logits, choose) -> Expr | None:
    session = SlotSession(
        Role.USER,
        task.view.public_pairs,
        canon=canon,
        candidate_slot=None if proposed is None else proposed.slot,
    )
    local = observe(
        Role.USER,
        canon=canon,
        pairs=task.view.public_pairs,
        candidate=proposed,
    )
    chosen = None
    for attempt in range(8):
        if session.closed:
            break
        message = _pull(queue, Role.USER, local, seed + 40 + attempt, legal, logits, choose)
        if message is None:
            break
        result = session.submit(message)
        if isinstance(result, Accepted) and isinstance(result.message, Use):
            steps.append(("user", message, "accepted"))
            chosen = result.message.expr
            break
        reply = result if isinstance(result, str) else "accepted"
        steps.append(("user", message, reply))
        if isinstance(result, str) and result.startswith("[TRY]"):
            local = f"{local} {result}"
    return chosen


def _outsider_phase(canon, task, queue, seed, steps, legal, logits, choose) -> bool:
    session = SlotSession(Role.OUTSIDER, task.view.public_pairs, canon=canon)
    local = observe(Role.OUTSIDER, canon=canon, pairs=task.view.public_pairs)
    solved = None
    for attempt in range(8):
        if session.closed:
            break
        message = _pull(queue, Role.OUTSIDER, local, seed + 60 + attempt, legal, logits, choose)
        if message is None:
            break
        result = session.submit(message)
        if isinstance(result, Accepted) and isinstance(result.message, Solve):
            steps.append(("outsider", message, "accepted"))
            solved = result.message.expr
            break
        reply = result if isinstance(result, str) else "accepted"
        steps.append(("outsider", message, reply))
        if isinstance(result, str) and result.startswith("[TRY]"):
            local = f"{local} {result}"
    if solved is None:
        return False
    return _expr_matches(solved, task.audit_pairs, _defs(canon))


def _pull(queue, role: Role, observation: str, seed: int, legal, logits, choose) -> str | None:
    if queue is not None:
        if not queue:
            return None
        return queue.pop(0)
    if choose is not None:
        return choose(role, observation, seed)
    return _choose(_MARKERS[role], observation, seed, legal, logits)


def _choose(marker: str, observation: str, seed: int, legal, logits) -> str:
    if legal is not None and logits is not None and EPSILON < 1:
        return _from_logits(logits, legal)
    return sample(marker, observation, seed=seed)


def _from_logits(logits, legal) -> str:
    del logits, legal
    raise RuntimeError("epsilon is 1")


def _queue(script: Script | None, name: str) -> list[str] | None:
    if script is None:
        return None
    return list(getattr(script, name))


def _defs(canon: Canon) -> dict[str, Def]:
    return {entry.definition.slot: entry.definition for entry in canon.replay()}


def _def_matches(defn: Def, pairs: Sequence[IOPair], defs: dict[str, Def]) -> bool:
    merged = dict(defs)
    merged[defn.slot] = defn
    for pair in pairs:
        if _defined(defn, pair.inputs, merged) != pair.output:
            return False
    return True


def _expr_matches(expr: Expr, pairs: Sequence[IOPair], defs: dict[str, Def]) -> bool:
    for pair in pairs:
        try:
            got = reduce_expr(expr, defs=defs, env=pair.inputs)
        except (ReduceError, ValueError):
            return False
        if got != pair.output:
            return False
    return True


def _defined(defn: Def, inputs: tuple[int, ...], defs: dict[str, Def]) -> int | None:
    merged = dict(defs)
    merged[defn.slot] = defn
    try:
        return reduce_def(defn, inputs, defs=merged)
    except (ReduceError, ValueError):
        return None


def _value(expr: Expr, inputs: tuple[int, ...], defs: dict[str, Def], canon: tuple[Def, ...]) -> int | None:
    known = dict(defs)
    for item in canon:
        known.setdefault(item.slot, item)
    try:
        if _contains_self(expr):
            slot = next(name for name in SLOTS if name not in known)
            return reduce_def(Def(slot, len(inputs), expr), inputs, defs=known)
        return reduce_expr(expr, defs=known, env=inputs)
    except (ReduceError, ValueError):
        return None


def _contains_self(expr: Expr) -> bool:
    return any(isinstance(node, Self) for node in _walk(expr))


def _calls(expr: Expr, slot: str) -> bool:
    return any(isinstance(node, Call) and node.slot == slot for node in _walk(expr))


def _walk(expr: Expr):
    stack = [expr]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, (Call, Self)):
            stack.extend(node.args)
