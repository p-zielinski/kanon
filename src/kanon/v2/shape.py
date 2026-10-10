"""Dziura prosta S. Numer składa się na term, lista termów nie powstaje.

S, bez `call`: `(zero)`, `(var i)` dla `i` mniejszego od arności, oraz jeden
`succ` albo jeden `pred` wokół `(zero)` albo `(var i)`. Jest ich
`3 * (1 + arność)`: 6 przy arności 1, 9 przy arności 2. Kolejność jest
od najkrótszych.
"""

from __future__ import annotations

from kanon.language import Expr, Pred, Succ, Var, Zero


def hole_count(arity: int) -> int:
    """Ile jest dziur prostych przy tej arności."""
    _check_arity(arity)
    return 3 * (1 + arity)


def decode_hole(index: int, arity: int) -> Expr:
    """Dziura o numerze `index`. Numery arności 2 to 0..8."""
    _check_arity(arity)
    _check_index(index)
    if index >= hole_count(arity):
        raise ValueError("index")
    if index == 0:
        return Zero()
    cursor = index - 1
    if cursor < arity:
        return Var(cursor)
    cursor -= arity
    if cursor == 0:
        return Succ(Zero())
    if cursor == 1:
        return Pred(Zero())
    cursor -= 2
    if cursor < arity:
        return Succ(Var(cursor))
    return Pred(Var(cursor - arity))


def decode_pair(index: int, base: int) -> tuple[Expr, Expr]:
    """Jedna para z iloczynu dwóch dziur. Podstawa 9, numer 80, to dziury 8 i 8."""
    _check_index(index)
    if type(base) is not int or base < 1:
        raise ValueError("base")
    high, low = divmod(index, base)
    if high >= base:
        raise ValueError("index")
    arity = _arity(base)
    return (decode_hole(high, arity), decode_hole(low, arity))


def _arity(count: int) -> int:
    if count % 3 != 0:
        raise ValueError("base")
    arity = count // 3 - 1
    if arity < 1:
        raise ValueError("base")
    return arity


def _check_arity(arity: int) -> None:
    if type(arity) is not int or arity < 1:
        raise ValueError("arity")


def _check_index(index: int) -> None:
    if type(index) is not int or index < 0:
        raise ValueError("index")
