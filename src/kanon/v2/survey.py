"""Przegląd do 16 tokenów.

Idzie od najkrótszych termów, bez `self` i bez `call`. Przy tej samej
długości najpierw termy bez `if0`. Staje, gdy na półce leżą cztery
rodziny. Drzew `if0` do 16 tokenów nie wylicza.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from kanon.language import Def, Expr, If0, Pred, Succ, Var, Zero
from kanon.v2.shelf import Shelf

_FAMILIES = ("identyczność", "następnik", "poprzednik", "stała 0", "stała 1", "stała 2")
_READY = 4


def survey(group: Shelf, solo: Shelf, *, limit: int = 16) -> None:
    """Kładzie jeden najkrótszy term na rodzinę, na półkę grupy i solo."""
    _check_limit(limit)
    found: set[str] = set()
    probe = Shelf()
    plain: dict[int, tuple[Expr, ...]] = {}
    for length in range(3, limit + 1):
        bucket = _plain(length, plain)
        plain[length] = bucket
        if _admit(group, solo, probe, found, bucket):
            return
        if _admit(group, solo, probe, found, _if0(length, plain)):
            return


def week_ready(shelf: Shelf) -> bool:
    """Tydzień startuje, gdy na półce leżą cztery rodziny piętra 1."""
    return sum(shelf.local_id(name) is not None for name in _FAMILIES) >= _READY


def _admit(
    group: Shelf,
    solo: Shelf,
    probe: Shelf,
    found: set[str],
    exprs: Iterable[Expr],
) -> bool:
    for expr in exprs:
        family = _family(probe, expr)
        if family is None or family in found:
            continue
        defn = Def("f0", 1, expr)
        group.write(defn)
        solo.write(defn)
        found.add(family)
        if len(found) == _READY:
            return True
    return False


def _family(probe: Shelf, expr: Expr) -> str | None:
    try:
        ident = probe.write(Def("f0", 1, expr))
    except ValueError as exc:
        if str(exc) != "domain":
            raise
        return None
    return probe.entry(ident).family


def _plain(length: int, earlier: dict[int, tuple[Expr, ...]]) -> tuple[Expr, ...]:
    if length == 3:
        return (Zero(),)
    if length == 4:
        return (Var(0),)
    sub_len = length - 3
    if sub_len < 3:
        return ()
    subs = earlier[sub_len]
    wrapped = tuple(Succ(sub) for sub in subs)
    return wrapped + tuple(Pred(sub) for sub in subs)


def _if0(length: int, plain: dict[int, tuple[Expr, ...]]) -> Iterator[Expr]:
    """`if0` z dziećmi bez `if0`. Zagnieżdżonego krzaka nie buduje."""
    need = length - 3
    if need < 9:
        return
    for cond_len in range(3, need - 5):
        for zero_len in range(3, need - cond_len - 2):
            nonzero_len = need - cond_len - zero_len
            if nonzero_len < 3:
                continue
            for cond in plain[cond_len]:
                for zero_branch in plain[zero_len]:
                    for nonzero in plain[nonzero_len]:
                        yield If0(cond, zero_branch, nonzero)


def _check_limit(limit: int) -> None:
    if type(limit) is not int or limit < 0:
        raise ValueError("limit")
