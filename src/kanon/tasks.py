"""Generator zadań Kalk.

Ziarno wyznacza term, pary publiczne i pary audytowe. Drugie wywołanie
`build_holdout` z tym samym ziarnem zwraca ten sam zbiór. Term celu
zostaje na obiekcie audytowym. Widok dla agenta ma tylko głębokość,
arność i pięć par publicznych.

Głębokość 1: identyczność, succ, pred, stałe 0, 1 i 2.
Głębokość 2: złożenie dwóch konstrukcji głębokości 1 albo dodawanie
lub odejmowanie ograniczone.
Głębokość 3 i 4 przy pustym kanonie: głębsze złożenie tych prymitywów.
Gdy kanon nie jest pusty, głębokość 3 i 4 woła jedną funkcję z kanonu
i podaje jej term głębokości o jeden mniejszej.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from kanon.language import (
    SLOTS,
    Call,
    Def,
    Expr,
    If0,
    Pred,
    Self,
    Succ,
    Var,
    Zero,
)
from kanon.reduce import reduce_def, reduce_expr

PUBLIC_COUNT = 5
AUDIT_COUNT = 16
PUBLIC_MAX = 6
AUDIT_MAX = 10
HOLDOUT_PER_DEPTH = 50
DEPTHS = (1, 2, 3, 4)

_ADD = If0(
    Var(0),
    Var(1),
    Succ(Self((Pred(Var(0)), Var(1)))),
)
_MONUS = If0(
    Var(1),
    Var(0),
    If0(Var(0), Zero(), Self((Pred(Var(0)), Pred(Var(1))))),
)


@dataclass(frozen=True, slots=True)
class IOPair:
    inputs: tuple[int, ...]
    output: int


@dataclass(frozen=True, slots=True)
class TaskView:
    """To, co wolno pokazać agentowi. Nie ma tu termu celu."""

    depth: int
    arity: int
    public_pairs: tuple[IOPair, ...]


@dataclass(frozen=True, slots=True)
class TaskAudit:
    """Obiekt automatu. `target` nie wchodzi do obserwacji agenta."""

    seed: int
    view: TaskView
    target: Expr
    audit_pairs: tuple[IOPair, ...]
    canon: tuple[Def, ...] = ()


@dataclass(frozen=True, slots=True)
class Holdout:
    seed: int
    tasks: tuple[TaskAudit, ...]

    def by_depth(self, depth: int) -> tuple[TaskAudit, ...]:
        return tuple(task for task in self.tasks if task.view.depth == depth)


def build_holdout(seed: int) -> Holdout:
    """50 zadań na każdej głębokości 1..4. Czysta funkcja ziarna."""
    tasks = []
    for depth in DEPTHS:
        for index in range(HOLDOUT_PER_DEPTH):
            child = _child_seed(seed, depth, index)
            tasks.append(generate_task(child, depth))
    return Holdout(seed=seed, tasks=tuple(tasks))


def generate_task(
    seed: int,
    depth: int,
    *,
    canon: Sequence[Def] = (),
) -> TaskAudit:
    if depth not in DEPTHS:
        raise ValueError("depth must be 1..4")
    canon_defs = tuple(canon)
    rng = random.Random(seed)
    arity = 1 if rng.randrange(2) == 0 else 2
    target = _target(rng, depth, arity, canon_defs)
    public_inputs = _sample(rng, arity, PUBLIC_MAX, PUBLIC_COUNT, banned=())
    audit_inputs = _sample(
        rng,
        arity,
        AUDIT_MAX,
        AUDIT_COUNT,
        banned=set(public_inputs),
    )
    view = TaskView(
        depth=depth,
        arity=arity,
        public_pairs=_pairs(target, public_inputs, canon_defs),
    )
    return TaskAudit(
        seed=seed,
        view=view,
        target=target,
        audit_pairs=_pairs(target, audit_inputs, canon_defs),
        canon=canon_defs,
    )


def task_observation(view: TaskView) -> str:
    """Pary publiczne jako tokeny Kalk. Bez par audytowych i bez termu celu."""
    chunks = ["[PAIRS]"]
    for pair in view.public_pairs:
        for value in (*pair.inputs, pair.output):
            chunks.append(f"(num {value})")
    return " ".join(chunks)


def _child_seed(seed: int, depth: int, index: int) -> int:
    return (seed + 1) * 1_000_003 + depth * 100 + index


def _target(
    rng: random.Random,
    depth: int,
    arity: int,
    canon: tuple[Def, ...],
) -> Expr:
    if depth == 1:
        return _depth1(rng, arity)
    if depth == 2:
        return _depth2(rng, arity)
    if canon:
        return _call_canon(rng, depth, arity, canon)
    return _tower(rng, arity, depth)


def _depth2(rng: random.Random, arity: int) -> Expr:
    if arity == 2 and rng.randrange(3) == 0:
        return _ADD
    if arity == 2 and rng.randrange(2) == 0:
        return _MONUS
    return _compose(rng, arity)


def _depth1(rng: random.Random, arity: int) -> Expr:
    kind = rng.randrange(6)
    if kind == 0:
        return Var(rng.randrange(arity))
    if kind == 1:
        return Succ(Var(rng.randrange(arity)))
    if kind == 2:
        return Pred(Var(rng.randrange(arity)))
    if kind == 3:
        return Zero()
    if kind == 4:
        return Succ(Zero())
    return Succ(Succ(Zero()))


def _compose(rng: random.Random, arity: int) -> Expr:
    for _ in range(32):
        inner = _depth1(rng, arity)
        term = Succ(inner) if rng.randrange(2) == 0 else Pred(inner)
        if not _is_depth1(term, arity):
            return term
    return Succ(Succ(Var(0)))


def _tower(rng: random.Random, arity: int, depth: int) -> Expr:
    if depth == 1:
        return _depth1(rng, arity)
    if depth == 2:
        return _compose(rng, arity)
    inner = _tower(rng, arity, depth - 1)
    return Succ(inner) if rng.randrange(2) == 0 else Pred(inner)


def _call_canon(
    rng: random.Random,
    depth: int,
    arity: int,
    canon: tuple[Def, ...],
) -> Expr:
    defn = canon[rng.randrange(len(canon))]
    if defn.arity == 0:
        return Call(defn.slot, ())
    sub = _tower(rng, arity, depth - 1)
    args: list[Expr] = [sub]
    while len(args) < defn.arity:
        args.append(_depth1(rng, arity))
    return Call(defn.slot, tuple(args))


def _is_depth1(expr: Expr, arity: int) -> bool:
    if expr in (Zero(), Succ(Zero()), Succ(Succ(Zero()))):
        return True
    if isinstance(expr, Var):
        return expr.index < arity
    if isinstance(expr, Succ) and isinstance(expr.arg, Var):
        return expr.arg.index < arity
    if isinstance(expr, Pred) and isinstance(expr.arg, Var):
        return expr.arg.index < arity
    return False


def _sample(
    rng: random.Random,
    arity: int,
    upper: int,
    count: int,
    *,
    banned: set[tuple[int, ...]] | tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    blocked = set(banned)
    domain = [inputs for inputs in _domain(arity, upper) if inputs not in blocked]
    if len(domain) >= count:
        return tuple(rng.sample(domain, count))
    return tuple(domain[rng.randrange(len(domain))] for _ in range(count))


def _domain(arity: int, upper: int) -> tuple[tuple[int, ...], ...]:
    if arity == 1:
        return tuple((value,) for value in range(upper + 1))
    return tuple(
        (left, right)
        for left in range(upper + 1)
        for right in range(upper + 1)
    )


def _pairs(
    target: Expr,
    rows: tuple[tuple[int, ...], ...],
    canon: tuple[Def, ...],
) -> tuple[IOPair, ...]:
    return tuple(IOPair(inputs, _evaluate(target, inputs, canon)) for inputs in rows)


def _evaluate(target: Expr, inputs: tuple[int, ...], canon: tuple[Def, ...]) -> int:
    defs = {item.slot: item for item in canon}
    if _contains_self(target):
        defn = Def(_fresh_slot(set(defs)), len(inputs), target)
        return reduce_def(defn, inputs, defs=defs)
    return reduce_expr(target, defs=defs, env=inputs)


def _contains_self(expr: Expr) -> bool:
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Self):
            return True
        if isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, Call):
            stack.extend(node.args)
    return False


def _fresh_slot(used: set[str]) -> str:
    for slot in SLOTS:
        if slot not in used:
            return slot
    raise ValueError("no free slot")
