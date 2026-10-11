"""Warsztat na dwóch kartach. Poziomy 0–3 składają ich wołania.

Generator najpierw wybiera dwie karty, potem buduje ukryty term tylko
z ich identyfikatorów. Poziomy 4–6 biorą fakt z puli. W oknie jest teza,
taktyki nie ma. Pusta pula taktyk trzyma sufit rampy na poziomie 3.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from kanon.language import Call, Def, Expr, If0, Num, Pred, Succ, Var, Zero, node_count
from kanon.v2.budget import message_budget, term_length
from kanon.reduce import NODE_LIMIT, ReduceError, reduce_expr
from kanon.v2.kernel import Kernel
from kanon.v2.seed import round_seed
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import DomEval, Fact, Rewrite, expr_tokens
from kanon.v2.vocab import VOCAB
from kanon.v2.window import BODY_LEN, PUBLIC_PAIRS, observation_prefix

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
    fact: Fact | None = None


@dataclass(frozen=True, slots=True)
class Pending:
    """Pula taktyk jeszcze nie dała zadania. Wejścia zostają świeże na rundę."""

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
        return _draw_proof(
            shelf,
            level=level,
            round_no=round_no,
            seed=seed,
            hint=hint,
        )
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


def _draw_proof(
    shelf: Shelf,
    *,
    level: int,
    round_no: int,
    seed: int,
    hint: bool,
) -> Task | None:
    """Teza w oknie. Taktyka zostaje po stronie mówcy."""
    rng = _rng(seed, round_no)
    facts = [
        fact
        for fact in tuple(shelf.hung()) + tuple(shelf.fact_pool())
        if isinstance(fact, Fact) and fact.left != fact.right
    ]
    if level < WORKSHOP_CAP:
        preferred = [fact for fact in facts if fact.kind in ("E", "W")]
        if preferred:
            facts = preferred
    rng.shuffle(facts)
    for fact in facts:
        if not _playable(shelf, fact):
            continue
        task = _proof_task(shelf, fact, level, rng, round_no, seed, hint)
        if task is not None:
            return task
    return None


def _proof_task(
    shelf: Shelf,
    fact: Fact,
    level: int,
    rng: random.Random,
    round_no: int,
    seed: int,
    hint: bool,
) -> Task | None:
    cards = _window_cards(shelf, fact, rng)
    if cards is None:
        return None
    shown = _fresh_fact(fact, rng) if level == 5 else fact
    words = _thesis_words(shown, cards)
    if words is None or len(words) > BODY_LEN:
        return None
    high = 10 if level >= 5 else 6
    inputs = tuple((rng.randint(0, high),) for _ in range(PUBLIC_PAIRS))
    verdicts = tuple(0 for _ in inputs)
    marked, family = _mark(shelf, cards, hint)
    observation = _observe(
        shelf,
        cards,
        inputs,
        verdicts,
        family,
        _defs(shelf),
        reduce_expr,
        body=words,
    )
    return Task(
        level,
        cards,
        shown.left,
        inputs,
        verdicts,
        observation,
        family,
        marked,
        round_seed(seed, round_no, 0),
        round_no,
        shown,
    )


def _playable(shelf: Shelf, fact: Fact) -> bool:
    kernel = Kernel(shelf)
    if fact.kind == "E":
        for tactic in fact.tactics:
            if isinstance(tactic, DomEval) and kernel.eval(fact.left, fact.right, tactic.lo, tactic.hi):
                return True
        return kernel.eval(fact.left, fact.right, 0, 1)
    if fact.kind == "W":
        names = [tactic.name for tactic in fact.tactics if isinstance(tactic, Rewrite)]
        names.extend(card.id for card in shelf.entries())
        return any(kernel.rewrite(fact.left, fact.right, name).accepted for name in names)
    if fact.kind in ("C", "I"):
        return kernel.eval(fact.left, fact.right, 0, 1)
    return False


def _window_cards(shelf: Shelf, fact: Fact, rng: random.Random) -> tuple[str, str] | None:
    mentioned: list[str] = []
    for expr in (fact.left, fact.right):
        for slot in _mentioned(expr):
            if slot not in mentioned:
                mentioned.append(slot)
    if len(mentioned) > 2:
        return None
    entries = [card.id for card in shelf.entries()]
    if len(entries) < 2:
        return None
    chosen = [slot for slot in mentioned if slot in entries]
    rest = [slot for slot in entries if slot not in chosen]
    rng.shuffle(rest)
    while len(chosen) < 2 and rest:
        chosen.append(rest.pop())
    if len(chosen) != 2:
        return None
    return (chosen[0], chosen[1])


def _fresh_fact(fact: Fact, rng: random.Random) -> Fact:
    number = rng.randint(0, 10)
    return Fact(
        _subst_var(fact.left, number),
        _subst_var(fact.right, number),
        fact.kind,
        fact.tactics,
    )


def _subst_var(expr: Expr, number: int) -> Expr:
    if isinstance(expr, Var) and expr.index == 0:
        return Num(number)
    if isinstance(expr, Succ):
        return Succ(_subst_var(expr.arg, number))
    if isinstance(expr, Pred):
        return Pred(_subst_var(expr.arg, number))
    if isinstance(expr, Call):
        return Call(expr.slot, tuple(_subst_var(arg, number) for arg in expr.args))
    if isinstance(expr, If0):
        return If0(
            _subst_var(expr.cond, number),
            _subst_var(expr.zero_branch, number),
            _subst_var(expr.nonzero_branch, number),
        )
    return expr


def _thesis_words(fact: Fact, cards: tuple[str, str]) -> tuple[str, ...] | None:
    mapping = {cards[0]: "f0", cards[1]: "f1"}
    left = _rename_window(fact.left, mapping)
    right = _rename_window(fact.right, mapping)
    if left is None or right is None:
        return None
    return ("(", "eq", *expr_tokens(left), *expr_tokens(right), ")")


def _rename_window(expr: Expr, mapping: Mapping[str, str]) -> Expr | None:
    if isinstance(expr, Call):
        if expr.slot not in mapping:
            return None
        args: list[Expr] = []
        for arg in expr.args:
            renamed = _rename_window(arg, mapping)
            if renamed is None:
                return None
            args.append(renamed)
        return Call(mapping[expr.slot], tuple(args))
    if isinstance(expr, Succ):
        arg = _rename_window(expr.arg, mapping)
        return None if arg is None else Succ(arg)
    if isinstance(expr, Pred):
        arg = _rename_window(expr.arg, mapping)
        return None if arg is None else Pred(arg)
    if isinstance(expr, If0):
        parts = (
            _rename_window(expr.cond, mapping),
            _rename_window(expr.zero_branch, mapping),
            _rename_window(expr.nonzero_branch, mapping),
        )
        if any(part is None for part in parts):
            return None
        return If0(parts[0], parts[1], parts[2])
    if isinstance(expr, (Zero, Var, Num)):
        return expr
    return None


def _mentioned(expr: Expr) -> tuple[str, ...]:
    found: list[str] = []
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Call):
            found.append(node.slot)
            stack.extend(node.args)
        elif isinstance(node, Succ):
            stack.append(node.arg)
        elif isinstance(node, Pred):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
    return tuple(found)


def _observe(
    shelf: Shelf,
    cards: tuple[str, str],
    inputs: tuple[tuple[int, ...], ...],
    verdicts: tuple[int, ...],
    family: str | None,
    defs: Mapping[str, Def],
    reduce: _Reduce,
    body: tuple[str, ...] = (),
) -> Observation:
    pairs = tuple(point + (value,) for point, value in zip(inputs, verdicts, strict=True))
    names = ("f0", "f1")
    examples = tuple(_examples(shelf.entry(card), defs, reduce) for card in cards)
    tokens = observation_prefix(
        inputs,
        body=body,
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
