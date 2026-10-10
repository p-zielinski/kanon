"""Mutacja po zejściu frontery z piętra 12.

Wcześniej paczka ma jedno ćwiczenie i zero mutacji. Potem w każdej
dziesiątce rund ostatnia, numer 9, 19, 29, …, jest mutacją. Komputer
robi 64 mutacje i kernel ocenia każdą. Trafienie zostaje w puli tej
krzywej. Lista szkieletów nie powstaje.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from kanon.language import Call, Def, Expr, Num, Var
from kanon.reduce import ReduceError, reduce_expr
from kanon.roles import assign
from kanon.v2.batch import run_package
from kanon.v2.play import Clocks, Tape, speaker
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import (
    CHAIN_TACTICS,
    CHAIN_TOKENS,
    DomEval,
    Eval,
    Fact,
    admits_e,
    chain_ok,
    thesis_tokens,
)

MUTATIONS = 64
_KINDS = ("number", "compose", "tactic")
_Judge = Callable[[Fact], bool]


@dataclass(frozen=True, slots=True)
class Variant:
    kind: str
    number: int | None
    fact: Fact
    accepted: bool


@dataclass(frozen=True, slots=True)
class ReplayResult:
    checked: tuple[int, ...]
    confirmed: tuple[int, ...]


class Gap:
    """Luka 20 punktów. Fakt lokalny do niej nie wchodzi."""

    def __init__(self) -> None:
        self._facts: list[Fact] = []

    @property
    def facts(self) -> tuple[Fact, ...]:
        return tuple(self._facts)

    def admit(self, fact: Fact, *, local: bool) -> bool:
        if local or len(self._facts) >= 20 or fact in self._facts:
            return False
        self._facts.append(fact)
        return True


def skeletons() -> tuple[object, ...]:
    """Listy szkieletów nie ma."""
    return ()


def is_mutation(round_no: int, floor: int) -> bool:
    """Ostatnia runda dziesiątki, dopiero po zejściu z piętra 12."""
    return floor > 12 and round_no % 10 == 9


def finish_package(
    schema: Schema,
    shape: object,
    *,
    exercise: Callable[[], None],
    mutate: Callable[[], None],
) -> None:
    """Paczka. Mutacja wstaje dopiero, gdy frontera zeszła z piętra 12."""
    run_package(
        schema,
        shape,
        save=lambda state: None,
        exercise=exercise,
        clock=lambda done: 0.0,
        closed=lambda: True,
    )
    if schema.floor > 12:
        mutate()


def shared_facts(group: Shelf, solo: Shelf) -> tuple[Fact, ...]:
    """Fakty schematu obecne na obu półkach."""
    other = list(solo.hung()) + list(solo.fact_pool())
    found: list[Fact] = []
    for fact in list(group.hung()) + list(group.fact_pool()):
        if fact in other and fact not in found:
            found.append(fact)
    return tuple(found)


def solo_task(shared: Sequence[Fact], group_local: Sequence[Fact]) -> tuple[Fact, ...]:
    """Zadanie solo bierze pulę wspólną. Lokalny fakt grupy tam nie wchodzi."""
    return tuple(fact for fact in shared if fact not in group_local)


def choose_fact(
    curve: str,
    shared: Sequence[Fact],
    local: Mapping[str, Sequence[Fact]],
) -> Fact | None:
    """Pula wspólna albo pula lokalna tej krzywej. Cudzej puli nie bierze."""
    own = tuple(local.get(curve, ()))
    pool = tuple(shared) + own
    if not pool:
        return None
    return pool[0]


def mutate(
    shelf: Shelf,
    fact: Fact,
    names: tuple[str, str],
    *,
    judge: _Judge | None = None,
    log: list[str] | None = None,
    curve: str = "group",
    local: Sequence[Fact] = (),
    clocks: Clocks | None = None,
    tape: Tape | None = None,
) -> tuple[Variant, ...]:
    """64 mutacje. Kernel ocenia każdą. Łamacz tej listy nie dostaje."""
    del curve, clocks, tape
    owned = list(local)
    found: list[Variant] = []
    for index in range(MUTATIONS):
        built = _build(index, fact, names)
        if judge is None:
            accepted = _kernel_yes(shelf, fact, built.fact, owned)
        else:
            accepted = bool(judge(built.fact)) and built.fact not in owned
        if accepted:
            owned.append(built.fact)
        found.append(Variant(built.kind, built.number, built.fact, accepted))
    if log is not None:
        log.append("mutacja")
    return tuple(found)


def replay(
    shelf: Shelf,
    fact: Fact,
    guesses: tuple[int, ...],
    *,
    round_no: int,
    tape: Tape,
    clocks: Clocks,
    level: int,
    local: bool,
) -> ReplayResult:
    """Powtórka faktu lokalnego. Cztery punkty, zegar łamacza tyka raz."""
    if not local:
        return ReplayResult((), ())
    if len(guesses) != 4:
        raise ValueError("guesses")
    confirmed = tuple(point for point in guesses if _diverges(shelf, fact, point))
    breaker = assign(round_no).breaker
    for point in confirmed:
        tape.note(breaker, level, (point,))
    clocks.tick_breaker(level)
    _role = speaker(round_no)[1]
    if _role == "author":
        clocks.tick_author(level)
    elif _role in ("user", "outsider", "breaker"):
        clocks.tick_executor(level)
    return ReplayResult(tuple(guesses), confirmed)


def _kernel_yes(shelf: Shelf, source: Fact, candidate: Fact, owned: Sequence[Fact]) -> bool:
    if candidate == source or candidate in owned:
        return False
    if len(candidate.tactics) > len(source.tactics) and chain_ok(shelf, candidate):
        return True
    for tactic in candidate.tactics:
        if isinstance(tactic, DomEval) and admits_e(
            shelf, candidate.left, candidate.right, tactic.lo, tactic.hi
        ):
            return True
    return False


def _build(index: int, fact: Fact, names: tuple[str, str]) -> Variant:
    kind = _KINDS[index % 3]
    if kind == "number":
        number = index % 11
        return Variant(
            kind,
            number,
            Fact(fact.left, Num(number), fact.kind, fact.tactics),
            False,
        )
    if kind == "compose":
        return Variant(
            kind,
            None,
            Fact(_compose(names, index), fact.right, fact.kind, fact.tactics),
            False,
        )
    tactics = fact.tactics
    if len(tactics) < 8:
        longer = Fact(fact.left, fact.right, "C", tactics + (Eval(),))
        if len(longer.tactics) <= CHAIN_TACTICS and thesis_tokens(longer) <= CHAIN_TOKENS:
            return Variant(kind, None, longer, False)
    return Variant(kind, None, fact, False)


def _compose(names: tuple[str, str], index: int) -> Expr:
    first, second = names
    outer = first if index % 2 == 0 else second
    inner = second if outer == first else first
    return Call(outer, (Call(inner, (Var(0),)),))


def _diverges(shelf: Shelf, fact: Fact, point: int) -> bool:
    defs = {card.id: Def(card.id, card.arity, card.body) for card in shelf.entries()}
    try:
        left = reduce_expr(fact.left, defs=defs, env=(point,))
        right = reduce_expr(fact.right, defs=defs, env=(point,))
    except (ReduceError, ValueError):
        return False
    return left != right
