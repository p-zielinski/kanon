"""Warsztat na dwóch kartach. Poziomy 0–3 składają ich wołania.

Generator najpierw wybiera dwie karty, potem buduje ukryty term tylko
z ich identyfikatorów. Poziomy 4–6 w tym prompcie jedynie odrzucają
losowanie, gdy pula taktyk jest pusta. Sufit rampy do tej chwili to 3.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from kanon.language import Call, Def, Expr, Succ, Var, node_count
from kanon.v2.budget import message_budget, term_length
from kanon.reduce import NODE_LIMIT, ReduceError, reduce_expr
from kanon.v2.seed import round_seed
from kanon.v2.shelf import Shelf
from kanon.v2.vocab import VOCAB
from kanon.v2.window import PUBLIC_PAIRS, observation_prefix

RAMP_CAP = 3
WORKSHOP_CAP = 6
FRESH = PUBLIC_PAIRS
_ATTEMPTS = 4

_Reduce = Callable[..., int]


@dataclass(frozen=True, slots=True)
class Observation:
    """To, co dostaje mówca. Termu ukrytego tu nie ma."""

    names: tuple[str, str]
    pairs: tuple[tuple[int, ...], ...]
    tokens: tuple[int, ...]
    family: str | None


@dataclass(frozen=True, slots=True)
class Task:
    level: int
    cards: tuple[str, str]
    term: Expr
    inputs: tuple[tuple[int, ...], ...]
    verdicts: tuple[int, ...]
    observation: Observation
    family: str | None
    marked: str | None
    seed: int
    round_no: int


@dataclass(frozen=True, slots=True)
class Pending:
    """Poziom 4–6. Treść taktyki dochodzi później. Wejścia są świeże na każdą rundę."""

    level: int
    inputs: tuple[tuple[int, ...], ...] = ()
    round_no: int = 0


class Workshop:
    """Licznik ćwiczeń. Mutacja go nie rusza."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.count = 0

    def ceiling(self, shelf: Shelf) -> int:
        return ramp_ceiling(shelf)

    def exercise(
        self,
        shelf: Shelf,
        *,
        level: int,
        round_no: int,
        reduce: _Reduce = reduce_expr,
    ) -> Task | Pending | None:
        self.count += 1
        return draw_task(
            shelf,
            level=level,
            round_no=round_no,
            seed=self.seed,
            hint=self.count % 4 == 0,
            reduce=reduce,
        )

    def mutate(self) -> None:
        """Mutacja nie jest ćwiczeniem i licznika nie podbija."""


def ramp_ceiling(shelf: Shelf) -> int:
    """Pusta pula taktyk trzyma rampę na poziomie 3."""
    if shelf.hung() or shelf.fact_pool():
        return WORKSHOP_CAP
    return RAMP_CAP


def draw_task(
    shelf: Shelf,
    *,
    level: int,
    round_no: int,
    seed: int,
    hint: bool = False,
    reduce: _Reduce = reduce_expr,
) -> Task | Pending | None:
    """Jedno zadanie. Werdykt jest policzony, zanim powstaje obserwacja."""
    if type(level) is not int or not 0 <= level <= WORKSHOP_CAP:
        return None
    if level > ramp_ceiling(shelf):
        return None
    if level > RAMP_CAP:
        rng = _rng(seed, round_no)
        return Pending(level, _inputs(rng, 1), round_no)
    rng = _rng(seed, round_no)
    cards = _pick(shelf, rng)
    if cards is None:
        return None
    defs = _defs(shelf)
    arities = {card.id: card.arity for card in shelf.entries()}
    term: Expr | None = None
    inputs: tuple[tuple[int, ...], ...] = ()
    verdicts: tuple[int, ...] | None = None
    for _ in range(_ATTEMPTS):
        built = _compose(cards, arities, rng, level + 1)
        fresh = _inputs(rng, _free_arity(built))
        judged = vet(built, defs, fresh, reduce=reduce, exercise=True)
        if judged is None:
            continue
        term = built
        inputs = fresh
        verdicts = judged
        break
    if term is None or verdicts is None:
        return None
    marked, family = _mark(shelf, cards, hint)
    observation = _observe(shelf, cards, inputs, verdicts, family, defs, reduce)
    return Task(
        level,
        cards,
        term,
        inputs,
        verdicts,
        observation,
        family,
        marked,
        round_seed(seed, round_no, 0),
        round_no,
    )


def vet(
    expr: Expr,
    defs: Mapping[str, Def],
    domain: Sequence[Sequence[int]],
    *,
    reduce: _Reduce = reduce_expr,
    floor: int = 12,
    exercise: bool = False,
) -> tuple[int, ...] | None:
    """Odrzuca zadatek na pierwszym limicie, bez reszty dziedziny."""
    if node_count(expr) > NODE_LIMIT:
        return None
    if term_length(expr) > message_budget(floor, exercise=exercise):
        return None
    found: list[int] = []
    for point in domain:
        try:
            found.append(reduce(expr, defs=defs, env=tuple(point)))
        except ReduceError:
            return None
    return tuple(found)


def serialize(observation: Observation) -> str:
    """Napisy mówcy. Bez wskazówki nie ma pola rodziny celu."""
    payload: dict[str, object] = {
        "names": observation.names,
        "pairs": observation.pairs,
        "tokens": [VOCAB[token] for token in observation.tokens],
    }
    if observation.family is not None:
        payload["rodzina"] = observation.family
    return json.dumps(payload, ensure_ascii=False)


def _observe(
    shelf: Shelf,
    cards: tuple[str, str],
    inputs: tuple[tuple[int, ...], ...],
    verdicts: tuple[int, ...],
    family: str | None,
    defs: Mapping[str, Def],
    reduce: _Reduce,
) -> Observation:
    pairs = tuple(point + (value,) for point, value in zip(inputs, verdicts, strict=True))
    names = ("f0", "f1")
    examples = tuple(_examples(shelf.entry(card), defs, reduce) for card in cards)
    tokens = observation_prefix(
        inputs,
        names=(
            (names[0], examples[0]),
            (names[1], examples[1]),
        ),
    )
    return Observation(names, pairs, tokens, family)


def _examples(
    card: object,
    defs: Mapping[str, Def],
    reduce: _Reduce,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shown: list[tuple[int, ...]] = []
    for number in (0, 1):
        point = (number,) * card.arity
        call = Call(card.id, tuple(Var(index) for index in range(card.arity)))
        value = reduce(call, defs=defs, env=point)
        shown.append(point + (value,))
    return (shown[0], shown[1])


def _mark(shelf: Shelf, cards: tuple[str, str], hint: bool) -> tuple[str | None, str | None]:
    if not hint:
        return None, None
    chosen = cards[0]
    for ident in cards:
        if shelf.entry(ident).family is not None:
            chosen = ident
            break
    return chosen, shelf.entry(chosen).family


def _pick(shelf: Shelf, rng: random.Random) -> tuple[str, str] | None:
    entries = shelf.entries()
    if len(entries) < 2:
        return None
    left, right = rng.sample(list(entries), 2)
    return (left.id, right.id)


def _compose(
    cards: tuple[str, str],
    arities: Mapping[str, int],
    rng: random.Random,
    calls: int,
) -> Expr:
    expr: Expr | None = None
    for _ in range(calls):
        slot = cards[rng.randrange(len(cards))]
        arity = arities[slot]
        if expr is None:
            args = tuple(Var(index) for index in range(arity))
        else:
            args = (expr,) + tuple(Var(0) for _ in range(arity - 1))
        expr = Call(slot, args)
    if expr is None:
        raise ValueError("calls")
    return expr


def _inputs(rng: random.Random, arity: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(rng.randint(0, 6) for _ in range(arity))
        for _ in range(FRESH)
    )


def _free_arity(expr: Expr) -> int:
    """Ile liczb potrzebuje term. Zagnieżdżone wołanie binarne czyta `(var 1)`."""
    highest = -1
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Var):
            highest = max(highest, node.index)
        elif isinstance(node, Succ):
            stack.append(node.arg)
        elif isinstance(node, Call):
            stack.extend(node.args)
    if highest < 0:
        raise ValueError("var")
    return highest + 1


def _defs(shelf: Shelf) -> dict[str, Def]:
    return {card.id: Def(card.id, card.arity, card.body) for card in shelf.entries()}


def _rng(seed: int, round_no: int) -> random.Random:
    return random.Random(round_seed(seed, round_no, 0))
