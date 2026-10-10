"""Piętra 2–8. Numer dekoduje się na dziury. Lista termów nie powstaje.

R1 i R2 są piętrem 2. Dalej mnożenie (R1), porównanie (N1), parzystość (P),
suma (R1), NWD (G) i potęga (R1). C1 i C2 nie istnieją. Złożenie wołań
nie jest piętrem. Pusta rodzina schodzi o jedno piętro. Minimum i maksimum
wiszą i porównania nie zamykają.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from kanon.language import Call, Def, Expr, If0, Pred, Self, Succ, Var, Zero
from kanon.v2.budget import message_budget, term_length
from kanon.v2.shape import decode_hole, hole_count
from kanon.v2.shelf import Shelf

_Target = tuple[str, int]
_Layout = tuple[str, object, int]
FLOOR_SHAPE = {
    3: "R1",
    4: "N1",
    5: "P",
    6: "R1",
    7: "G",
    8: "R1",
    9: "E",
    10: "W",
    11: "C",
    12: "I",
}
FLOOR_FAMILY = {
    3: "mnożenie",
    4: "porównanie",
    5: "parzystość",
    6: "suma",
    7: "NWD",
    8: "potęga",
}


class Schema:
    """Trójka stanu: piętro, kształt, numer."""

    def __init__(self) -> None:
        self.floor = 2
        self.shape = "R1"
        self.index = 0
        self.log: list[str] = []
        self._stopped: set[str] = set()

    def state(self) -> tuple[int, str, int]:
        return (self.floor, self.shape, self.index)

    def seek_addition(
        self,
        group: Shelf,
        solo: Shelf,
        start: int,
        on_decode: Callable[[Expr], None] | None = None,
    ) -> Expr | None:
        """Szuka dodawania od `start`. Po trafieniu R1 staje, R2 jest nietknięte."""
        if "R1" in self._stopped:
            return None
        _check_index(start)
        index, term = _walk(lambda i: decode_r1(i, 2), start, "dodawanie", on_decode)
        _place(group, solo, term)
        self._stopped.add("R1")
        self.floor = 2
        self.shape = "R2"
        self.index = 0
        return term

    def seek_subtraction(
        self,
        group: Shelf,
        solo: Shelf,
        start: int,
        on_decode: Callable[[Expr], None] | None = None,
    ) -> Expr | None:
        """Szuka odejmowania od `start`. Samo nie zamyka piętra."""
        if "R2" in self._stopped:
            return None
        _check_index(start)
        index, term = _walk(lambda i: decode_r2(i, 2), start, "odejmowanie", on_decode)
        _place(group, solo, term)
        self._stopped.add("R2")
        if group.local_id("dodawanie") is not None:
            self.floor = 3
            self.shape = "R1"
            self.index = 0
        else:
            self.floor = 2
            self.shape = "R2"
            self.index = index
        return term

    def seek_parity(
        self,
        group: Shelf,
        solo: Shelf,
        on_decode: Callable[[Expr], None] | None = None,
    ) -> Expr | None:
        """Pełny kształt P. Pierwsza parzystość zamyka piętro."""
        if "P" in self._stopped:
            return None
        self.floor = 5
        self.shape = "P"
        total = hole_count(1) ** 2
        for index in range(self.index, total):
            term = decode_p(index)
            if on_decode is not None:
                on_decode(term)
            if _family(term, 1, floor=self.floor) == "parzystość":
                _place(group, solo, term, 1)
                self._stopped.add("P")
                self._advance()
                return term
        self.log.append("schemat wyczerpany")
        self._stopped.add("P")
        self._advance()
        return None

    def advance_if_closed(self, shelf: Shelf) -> bool:
        """Schodzi, gdy na półce jest rodzina albo fakt tego piętra."""
        if not _floor_ready(self.floor, shelf):
            return False
        self._advance()
        return True

    def scan(
        self,
        shape: object,
        matches: Callable[[Expr], bool],
        group: Shelf,
        solo: Shelf,
        on_decode: Callable[[Expr], None] | None = None,
    ) -> Expr | None:
        """Jeden kształt. Brak rodziny pisze „schemat wyczerpany” i schodzi piętro."""
        name = shape.name
        if name in self._stopped:
            return None
        for index in range(shape.count):
            term = shape.decode(index)
            if on_decode is not None:
                on_decode(term)
            if matches(term):
                _place(group, solo, term)
                self._stopped.add(name)
                self.shape = name
                self.index = index
                return term
        self.log.append("schemat wyczerpany")
        self._stopped.add(name)
        self._advance()
        return None

    def _advance(self) -> None:
        self.floor += 1
        self.shape = FLOOR_SHAPE.get(self.floor, "R1")
        self.index = 0


def _floor_ready(floor: int, shelf: Shelf) -> bool:
    family = FLOOR_FAMILY.get(floor)
    if family is not None:
        return shelf.local_id(family) is not None
    kind = {9: "E", 10: "W", 11: "C", 12: "I"}.get(floor)
    if kind is None:
        return False
    return any(getattr(fact, "kind", None) == kind for fact in shelf.hung())


def floor_shape(floor: int) -> str:
    """Kształt piętra 3–12."""
    return FLOOR_SHAPE[floor]


def floor_family(floor: int) -> str:
    """Rodzina, która zamyka to piętro."""
    return FLOOR_FAMILY[floor]


def comparison_shape(shelf: Shelf):
    """N1. Pusta półka nie ma wołania, więc kształt ma zero termów."""
    targets = tuple((card.id, card.arity) for card in shelf.entries())
    base = hole_count(2)
    count = 0
    for _, callee in targets:
        _check_callee(callee)
        count += base ** (callee + 2)
    return _Shape("N1", count, lambda index: decode_n1(index, targets))


def gcd_shape(shelf: Shelf):
    """G. Pary różnych funkcji binarnych. Jedna funkcja pary nie tworzy."""
    names = tuple(card.id for card in shelf.entries() if card.arity == 2)
    pairs = tuple((left, right) for left in names for right in names if left != right)
    return _Shape("G", len(pairs), lambda index: _gcd_term(pairs[index]))


def decode_n1(index: int, targets: Sequence[_Target]) -> Expr:
    """Term N1 o tym numerze."""
    _check_index(index)
    base = hole_count(2)
    cursor = index
    for name, callee in targets:
        _check_callee(callee)
        width = callee + 2
        block = base**width
        if cursor < block:
            digits = _digits(cursor, base, width)
            args = tuple(_hole(digit, 2) for digit in digits[:callee])
            return If0(
                Call(name, args),
                _hole(digits[callee], 2),
                _hole(digits[callee + 1], 2),
            )
        cursor -= block
    raise ValueError("index")


def encode_n1(expr: Expr, targets: Sequence[_Target]) -> int:
    """Numer termu N1 wyliczony z kształtu."""
    if not isinstance(expr, If0) or not isinstance(expr.cond, Call):
        raise ValueError("shape")
    call = expr.cond
    holes = [_encode_hole(arg, 2) for arg in call.args]
    holes.append(_encode_hole(expr.zero_branch, 2))
    holes.append(_encode_hole(expr.nonzero_branch, 2))
    base = hole_count(2)
    offset = 0
    for name, callee in targets:
        _check_callee(callee)
        width = callee + 2
        block = base**width
        if name == call.slot and callee == len(call.args):
            if len(holes) != width:
                raise ValueError("shape")
            value = 0
            for digit in holes:
                value = value * base + digit
            return offset + value
        offset += block
    raise ValueError("shape")


def decode_p(index: int) -> Expr:
    """Term P. Dwa `pred` w argumencie `self` są częścią drzewa."""
    _check_index(index)
    base = hole_count(1)
    if index >= base * base:
        raise ValueError("index")
    outer, inner = divmod(index, base)
    return If0(
        Var(0),
        _hole(outer, 1),
        If0(Pred(Var(0)), _hole(inner, 1), Self((Pred(Pred(Var(0))),))),
    )


def encode_p(expr: Expr) -> int:
    """Numer termu P wyliczony z dwóch dziur."""
    if not isinstance(expr, If0) or expr.cond != Var(0):
        raise ValueError("shape")
    inner = expr.nonzero_branch
    if (
        not isinstance(inner, If0)
        or inner.cond != Pred(Var(0))
        or inner.nonzero_branch != Self((Pred(Pred(Var(0))),))
    ):
        raise ValueError("shape")
    base = hole_count(1)
    return _encode_hole(expr.zero_branch, 1) * base + _encode_hole(inner.zero_branch, 1)


def decode_r1(
    index: int,
    arity: int,
    targets: Sequence[_Target] = (),
) -> Expr:
    """Term R1 o tym numerze."""
    return _decode(index, arity, targets, _r1_width, _build_r1)


def encode_r1(
    expr: Expr,
    arity: int,
    targets: Sequence[_Target] = (),
) -> int:
    """Numer termu R1 wyliczony z kształtu."""
    if not isinstance(expr, If0):
        raise ValueError("shape")
    kind, payload, self_expr, extras = _peel(expr.nonzero_branch, arity)
    holes = [_encode_hole(expr.cond, arity), _encode_hole(expr.zero_branch, arity)]
    holes.extend(_encode_hole(arg, arity) for arg in self_expr.args)
    holes.extend(_encode_hole(extra, arity) for extra in extras)
    return _pack(arity, targets, kind, payload, holes, _r1_width)


def decode_r2(
    index: int,
    arity: int,
    targets: Sequence[_Target] = (),
) -> Expr:
    """Term R2 o tym numerze."""
    return _decode(index, arity, targets, _r2_width, _build_r2)


def encode_r2(
    expr: Expr,
    arity: int,
    targets: Sequence[_Target] = (),
) -> int:
    """Numer termu R2 wyliczony z kształtu."""
    if not isinstance(expr, If0) or not isinstance(expr.nonzero_branch, If0):
        raise ValueError("shape")
    inner = expr.nonzero_branch
    kind, payload, self_expr, extras = _peel(inner.nonzero_branch, arity)
    holes = [
        _encode_hole(expr.cond, arity),
        _encode_hole(expr.zero_branch, arity),
        _encode_hole(inner.cond, arity),
        _encode_hole(inner.zero_branch, arity),
    ]
    holes.extend(_encode_hole(arg, arity) for arg in self_expr.args)
    holes.extend(_encode_hole(extra, arity) for extra in extras)
    return _pack(arity, targets, kind, payload, holes, _r2_width)


def _decode(index, arity, targets, width_of, build) -> Expr:
    _check_index(index)
    _check_arity(arity)
    base = hole_count(arity)
    cursor = index
    for kind, payload, width in _layouts(arity, targets, width_of):
        block = base**width
        if cursor < block:
            return build(arity, kind, payload, _digits(cursor, base, width))
        cursor -= block
    raise ValueError("index")


def _pack(arity, targets, kind, payload, holes, width_of) -> int:
    _check_arity(arity)
    base = hole_count(arity)
    offset = 0
    for item_kind, item_payload, width in _layouts(arity, targets, width_of):
        block = base**width
        if item_kind == kind and item_payload == payload:
            if len(holes) != width:
                raise ValueError("shape")
            value = 0
            for digit in holes:
                value = value * base + digit
            return offset + value
        offset += block
    raise ValueError("shape")


def _layouts(arity: int, targets: Sequence[_Target], width_of: Callable[[int], int]):
    fixed = width_of(arity)
    layouts: list[_Layout] = [("plain", kind, fixed) for kind in range(3)]
    for name, callee in targets:
        if type(callee) is not int or callee < 1:
            raise ValueError("target")
        for pos in range(callee):
            layouts.append(("call", (name, pos, callee), fixed + callee - 1))
    return layouts


def _r1_width(arity: int) -> int:
    return 2 + arity


def _r2_width(arity: int) -> int:
    return 4 + arity


def _build_r1(arity: int, kind: str, payload: object, digits: tuple[int, ...]) -> Expr:
    fixed = _r1_width(arity)
    cond = _hole(digits[0], arity)
    zero = _hole(digits[1], arity)
    self_expr = Self(tuple(_hole(digit, arity) for digit in digits[2:fixed]))
    wrapped = _wrap(kind, payload, self_expr, digits[fixed:], arity)
    return If0(cond, zero, wrapped)


def _build_r2(arity: int, kind: str, payload: object, digits: tuple[int, ...]) -> Expr:
    fixed = _r2_width(arity)
    outer_cond = _hole(digits[0], arity)
    outer_zero = _hole(digits[1], arity)
    inner_cond = _hole(digits[2], arity)
    inner_zero = _hole(digits[3], arity)
    self_expr = Self(tuple(_hole(digit, arity) for digit in digits[4:fixed]))
    wrapped = _wrap(kind, payload, self_expr, digits[fixed:], arity)
    return If0(outer_cond, outer_zero, If0(inner_cond, inner_zero, wrapped))


def _wrap(kind: str, payload: object, self_expr: Self, extra: tuple[int, ...], arity: int) -> Expr:
    if kind == "plain":
        if payload == 0:
            return self_expr
        if payload == 1:
            return Succ(self_expr)
        return Pred(self_expr)
    name, pos, callee = payload
    extras = [_hole(digit, arity) for digit in extra]
    args: list[Expr] = []
    cursor = 0
    for slot in range(callee):
        if slot == pos:
            args.append(self_expr)
        else:
            args.append(extras[cursor])
            cursor += 1
    return Call(name, tuple(args))


def _peel(expr: Expr, arity: int):
    if _is_self(expr, arity):
        return "plain", 0, expr, ()
    if isinstance(expr, Succ) and _is_self(expr.arg, arity):
        return "plain", 1, expr.arg, ()
    if isinstance(expr, Pred) and _is_self(expr.arg, arity):
        return "plain", 2, expr.arg, ()
    if isinstance(expr, Call):
        positions = [slot for slot, arg in enumerate(expr.args) if _is_self(arg, arity)]
        if len(positions) != 1:
            raise ValueError("shape")
        pos = positions[0]
        self_expr = expr.args[pos]
        extras = tuple(arg for slot, arg in enumerate(expr.args) if slot != pos)
        return "call", (expr.slot, pos, len(expr.args)), self_expr, extras
    raise ValueError("shape")


def _is_self(expr: Expr, arity: int) -> bool:
    return isinstance(expr, Self) and len(expr.args) == arity


def _hole(index: int, arity: int) -> Expr:
    return decode_hole(index, arity)


def _encode_hole(expr: Expr, arity: int) -> int:
    if isinstance(expr, Zero):
        return 0
    if isinstance(expr, Var) and expr.index < arity:
        return 1 + expr.index
    if isinstance(expr, Succ) and isinstance(expr.arg, Zero):
        return 1 + arity
    if isinstance(expr, Pred) and isinstance(expr.arg, Zero):
        return 2 + arity
    if isinstance(expr, Succ) and isinstance(expr.arg, Var) and expr.arg.index < arity:
        return 3 + arity + expr.arg.index
    if isinstance(expr, Pred) and isinstance(expr.arg, Var) and expr.arg.index < arity:
        return 3 + 2 * arity + expr.arg.index
    raise ValueError("hole")


def _digits(value: int, base: int, width: int) -> tuple[int, ...]:
    digits = [0] * width
    for pos in range(width - 1, -1, -1):
        value, digits[pos] = divmod(value, base)
    return tuple(digits)


def _walk(decode, start: int, family: str, on_decode) -> tuple[int, Expr]:
    for step in range(8):
        index = start + step
        term = decode(index)
        if on_decode is not None:
            on_decode(term)
        if _family(term) == family:
            return index, term
    raise ValueError("scan")


def _family(expr: Expr, arity: int = 2, *, floor: int = 2) -> str | None:
    if term_length(expr) > message_budget(floor):
        return None
    shelf = Shelf()
    try:
        ident = shelf.write(Def("f0", arity, expr))
    except ValueError:
        return None
    return shelf.entry(ident).family


def _place(group: Shelf, solo: Shelf, expr: Expr, arity: int = 2) -> None:
    defn = Def("f0", arity, expr)
    group.write(defn)
    solo.write(defn)


class _Shape:
    def __init__(self, name: str, count: int, decode: Callable[[int], Expr]) -> None:
        self.name = name
        self.count = count
        self._decode = decode

    def decode(self, index: int) -> Expr:
        _check_index(index)
        if index >= self.count:
            raise ValueError("index")
        return self._decode(index)


def _gcd_term(pair: tuple[str, str]) -> Expr:
    left, right = pair
    return If0(
        Var(1),
        Var(0),
        If0(
            Call(left, (Var(0), Var(1))),
            Self((Var(1), Call(right, (Var(0), Var(1))))),
            Self((Var(1), Call(right, (Var(1), Var(0))))),
        ),
    )


def _check_callee(callee: int) -> None:
    if type(callee) is not int or callee < 1:
        raise ValueError("target")


def _check_arity(arity: int) -> None:
    if type(arity) is not int or arity < 1:
        raise ValueError("arity")


def _check_index(index: int) -> None:
    if type(index) is not int or index < 0:
        raise ValueError("index")
