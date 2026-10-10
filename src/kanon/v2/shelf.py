"""Półka v2. Identyfikator zostaje, nazwa f0–f31 jest oknem.

Wpis przyjęty na dziedzinie dostaje lokalny identyfikator. Bufor roboczy,
który dziedziny nie przeszedł, znika po 12 rundach. Ława co 50 rund
liczy pierścień tuż za dziedziną i kilka punktów z wnętrza. Rozjazd
liczby zdejmuje wpis, powołania i zapis puli. Paliwo, kroki albo pamięć
to „nie wiadomo”.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from kanon.language import (
    SLOTS,
    Call,
    Def,
    Expr,
    If0,
    Num,
    Pred,
    Self,
    Succ,
    Var,
    Zero,
)
from kanon.reduce import FUEL_LIMIT, ReduceError, reduce_def

EXPIRE_AFTER = 12
BENCH_EVERY = 50
_UNKNOWN = frozenset(("fuel", "steps", "memory"))

_Spec = Callable[[int, tuple[int, ...]], int | None]


@dataclass(frozen=True, slots=True)
class Card:
    id: str
    body: Expr
    arity: int
    family: str | None
    citations: tuple[str, ...]
    born: int


@dataclass(frozen=True, slots=True)
class _Held:
    definition: Def
    born: int


class Shelf:
    """Półka grupy albo solo. Identyfikatory liczy od zera, osobno."""

    def __init__(self) -> None:
        self._cards: dict[str, Card] = {}
        self._order: list[str] = []
        self._buffer: dict[str, _Held] = {}
        self._pool: dict[str, tuple[str, ...]] = {}
        self._hung: list[object] = []
        self._exercise: list[object] = []
        self._window: dict[str, str] = {}
        self._next = 0
        self._buf_next = 0
        self._pool_next = 0

    def write(self, definition: Def, round_no: int = 0) -> str:
        """Przyjmuje definicję, która na dziedzinie wraca liczbą."""
        _check_round(round_no)
        if definition.arity not in (1, 2):
            raise ValueError("arity")
        body = _rewrite(definition.body, self._window)
        ident = f"e{self._next}"
        self._next += 1
        defn = Def(ident, definition.arity, body)
        family = _classify(defn, self._defs())
        card = Card(ident, body, definition.arity, family, _calls(body), round_no)
        self._cards[ident] = card
        self._order.append(ident)
        return ident

    def stage(self, definition: Def, round_no: int) -> str:
        """Bufor roboczy. Dziedziny jeszcze nie liczy."""
        _check_round(round_no)
        ident = f"b{self._buf_next}"
        self._buf_next += 1
        self._buffer[ident] = _Held(definition, round_no)
        return ident

    def note_use(self, ident: str, round_no: int) -> None:
        """Wołanie nie przedłuża bufora i nie zdejmuje półki."""
        _check_round(round_no)
        if ident not in self._cards and ident not in self._buffer:
            raise ValueError("id")

    def expire(self, round_no: int) -> tuple[str, ...]:
        """Zdejmuje bufor, gdy runda minęła urodzenie o 12. Półka zostaje."""
        _check_round(round_no)
        doomed = tuple(
            ident
            for ident, held in self._buffer.items()
            if round_no >= held.born + EXPIRE_AFTER
        )
        for ident in doomed:
            del self._buffer[ident]
        return doomed

    def bind(self, ids: Sequence[str]) -> None:
        """Nakłada nazwy f0, f1, … na podane identyfikatory."""
        if len(ids) > len(SLOTS):
            raise ValueError("window")
        for ident in ids:
            if ident not in self._cards:
                raise ValueError("window")
        self._window = {f"f{index}": ident for index, ident in enumerate(ids)}

    def id_at(self, name: str) -> str | None:
        """Identyfikator, który teraz stoi pod napisem okna."""
        return self._window.get(name)

    def local_id(self, family: str) -> str | None:
        """Identyfikator tej półki dla rodziny zadania wspólnego."""
        for ident in self._order:
            if self._cards[ident].family == family:
                return ident
        return None

    def entry(self, ident: str) -> Card:
        return self._cards[ident]

    def entries(self) -> tuple[Card, ...]:
        """Wpisy półki, w kolejności przyjęcia."""
        return tuple(self._cards[ident] for ident in self._order)

    def contains(self, ident: str) -> bool:
        return ident in self._cards or ident in self._buffer

    def pool_add(self, cites: Sequence[str]) -> str:
        cited = tuple(cites)
        for ident in cited:
            if ident not in self._cards:
                raise ValueError("pool")
        ident = f"p{self._pool_next}"
        self._pool_next += 1
        self._pool[ident] = cited
        return ident

    def pool(self) -> tuple[str, ...]:
        return tuple(self._pool)

    def hang(self, fact: object) -> bool:
        """Fakt, który zamyka piętro. Kopia już wiszącego odpada."""
        if fact in self._hung:
            return False
        self._hung.append(fact)
        return True

    def hung(self) -> tuple[object, ...]:
        return tuple(self._hung)

    def pool_fact(self, fact: object) -> bool:
        """Co najwyżej osiem dalszych faktów. Kopia wiszącego albo ósemki odpada."""
        if len(self._exercise) >= 8 or fact in self._hung or fact in self._exercise:
            return False
        self._exercise.append(fact)
        return True

    def fact_pool(self) -> tuple[object, ...]:
        return tuple(self._exercise)

    def bench(self, round_no: int) -> tuple[str, ...]:
        """Co 50 rund. Rozjazd liczby zdejmuje wpis i to, co się na niego powołuje."""
        _check_round(round_no)
        if round_no == 0 or round_no % BENCH_EVERY != 0:
            return ()
        doomed = {
            ident
            for ident in self._order
            if self._cards[ident].family is not None and self._diverges(self._cards[ident])
        }
        changed = True
        while changed:
            changed = False
            for ident, card in self._cards.items():
                if ident not in doomed and any(cite in doomed for cite in card.citations):
                    doomed.add(ident)
                    changed = True
            for ident, cited in self._pool.items():
                if ident not in doomed and any(cite in doomed for cite in cited):
                    doomed.add(ident)
                    changed = True
        for ident in list(self._order):
            if ident in doomed:
                del self._cards[ident]
                self._order.remove(ident)
        for ident in list(self._pool):
            if ident in doomed:
                del self._pool[ident]
        self._window = {
            name: ident for name, ident in self._window.items() if ident not in doomed
        }
        return tuple(ident for ident in doomed)

    def _defs(self) -> dict[str, Def]:
        return {
            ident: Def(ident, card.arity, card.body) for ident, card in self._cards.items()
        }

    def _diverges(self, card: Card) -> bool:
        assert card.family is not None
        spec = _SPEC[card.family]
        defn = Def(card.id, card.arity, card.body)
        defs = self._defs()
        for point in _interior(card.arity) + _ring(card.arity):
            try:
                got = reduce_def(defn, point, defs=defs)
            except ReduceError as exc:
                if exc.kind in _UNKNOWN:
                    continue
                raise
            expected = spec(card.arity, point)
            if expected is None or got != expected:
                return True
        return False


class FrozenShelf:
    """Wagi zamrożone. Metody zapisu nie ma."""


def _classify(defn: Def, defs: Mapping[str, Def]) -> str | None:
    points = _domain(defn.arity)
    got: list[int | None] = []
    for point in points:
        try:
            got.append(reduce_def(defn, point, defs=defs))
        except ReduceError as exc:
            if exc.kind in _UNKNOWN:
                got.append(None)
            else:
                raise
    if any(value is None for value in got):
        return _power_if_known(defn.arity, points, got)
    for name, spec in _SPEC.items():
        expected = [spec(defn.arity, point) for point in points]
        if any(value is None for value in expected):
            continue
        if expected == got:
            return name
    return None


def _power_if_known(
    arity: int,
    points: tuple[tuple[int, ...], ...],
    got: list[int | None],
) -> str | None:
    """Potęga zostaje, gdy znane punkty się zgadzają, a nieznane są większe niż paliwo."""
    matched = False
    for point, value in zip(points, got, strict=True):
        expected = _power(arity, point)
        if expected is None:
            return None
        if value is None:
            if expected <= FUEL_LIMIT:
                return None
            continue
        if value != expected:
            return None
        matched = True
    if not matched:
        return None
    return "potęga"


def _domain(arity: int) -> tuple[tuple[int, ...], ...]:
    if arity == 1:
        return tuple((n,) for n in range(11))
    if arity == 2:
        return tuple((x, y) for x in range(7) for y in range(7))
    raise ValueError("arity")


def _ring(arity: int) -> tuple[tuple[int, ...], ...]:
    if arity == 1:
        return ((11,),)
    if arity == 2:
        return tuple((x, y) for x in range(8) for y in range(8) if x == 7 or y == 7)
    raise ValueError("arity")


def _interior(arity: int) -> tuple[tuple[int, ...], ...]:
    if arity == 1:
        return ((0,), (5,), (10,))
    if arity == 2:
        return ((0, 0), (3, 3), (6, 6))
    raise ValueError("arity")


def _identity(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 1:
        return None
    return point[0]


def _successor(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 1:
        return None
    return point[0] + 1


def _predecessor(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 1:
        return None
    return 0 if point[0] == 0 else point[0] - 1


def _addition(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    return point[0] + point[1]


def _subtraction(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    return max(0, point[0] - point[1])


def _product(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    return point[0] * point[1]


def _compare(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    return 1 if point[0] <= point[1] else 0


def _minimum(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    return min(point)


def _maximum(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    return max(point)


def _parity(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 1:
        return None
    return point[0] % 2


def _triangular(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 1:
        return None
    number = point[0]
    return number * (number + 1) // 2


def _gcd(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    left, right = point
    while right:
        left, right = right, left % right
    return left


def _power(arity: int, point: tuple[int, ...]) -> int | None:
    if arity != 2:
        return None
    base, exponent = point
    if exponent == 0:
        return 1
    return base**exponent


def _constant(value: int) -> _Spec:
    def spec(arity: int, point: tuple[int, ...]) -> int | None:
        if arity not in (1, 2):
            return None
        return value

    return spec


_SPEC: dict[str, _Spec] = {
    "identyczność": _identity,
    "następnik": _successor,
    "poprzednik": _predecessor,
    "stała 0": _constant(0),
    "stała 1": _constant(1),
    "stała 2": _constant(2),
    "dodawanie": _addition,
    "odejmowanie": _subtraction,
    "mnożenie": _product,
    "porównanie": _compare,
    "minimum": _minimum,
    "maksimum": _maximum,
    "parzystość": _parity,
    "suma": _triangular,
    "NWD": _gcd,
    "potęga": _power,
}


def _rewrite(expr: Expr, window: dict[str, str]) -> Expr:
    if isinstance(expr, Call):
        if expr.slot in SLOTS:
            if expr.slot not in window:
                raise ValueError("name")
            slot = window[expr.slot]
        else:
            slot = expr.slot
        return Call(slot, tuple(_rewrite(arg, window) for arg in expr.args))
    if isinstance(expr, Succ):
        return Succ(_rewrite(expr.arg, window))
    if isinstance(expr, Pred):
        return Pred(_rewrite(expr.arg, window))
    if isinstance(expr, If0):
        return If0(
            _rewrite(expr.cond, window),
            _rewrite(expr.zero_branch, window),
            _rewrite(expr.nonzero_branch, window),
        )
    if isinstance(expr, Self):
        return Self(tuple(_rewrite(arg, window) for arg in expr.args))
    if isinstance(expr, (Zero, Var, Num)):
        return expr
    raise TypeError(type(expr))


def _calls(expr: Expr) -> tuple[str, ...]:
    found: list[str] = []
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Call):
            found.append(node.slot)
            stack.extend(node.args)
        elif isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, Self):
            stack.extend(node.args)
    return tuple(found)


def _check_round(round_no: int) -> None:
    if type(round_no) is not int or round_no < 0:
        raise ValueError("round")
