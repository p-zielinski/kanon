"""Kernel taktyk. Liczy równość, jedno przepisanie i krok indukcyjny.

Lean i cudza biblioteka twierdzeń tu nie wchodzą. `ih` nie jest wpisem półki.
Jedno wstawienie ciała jest faktem W. Znacznika E ten kernel nie stawia.
"""

from __future__ import annotations

from dataclasses import dataclass

from kanon.language import (
    Call,
    Def,
    Expr,
    If0,
    Num,
    ParseError,
    Pred,
    Self,
    Succ,
    Var,
    Zero,
)
from kanon.reduce import ReduceError, reduce_expr
from kanon.v2.shelf import Card, Shelf


@dataclass(frozen=True, slots=True)
class Verdict:
    """Przyjęcie taktyki. `kind` to „W” albo nic. „E” tu nie występuje."""

    accepted: bool
    kind: str | None = None


class Kernel:
    """Półka jest źródłem ciał. Założenie indukcyjne na niej nie leży."""

    def __init__(self, shelf: Shelf) -> None:
        self.shelf = shelf

    def eval(self, left: Expr, right: Expr, lo: int, hi: int) -> bool:
        """Obie strony równe na każdym punkcie `lo..hi` włącznie."""
        if left == right or not _interval(lo, hi):
            return False
        defs = self._defs()
        for point in range(lo, hi + 1):
            if not _same(left, right, Num(point), defs):
                return False
        return True

    def rewrite(self, left: Expr, right: Expr, name: str) -> Verdict:
        """Jedno wstawienie ciała w dziedzinie wpisu. Poza dziedziną odrzuca."""
        if left == right:
            return Verdict(False)
        card = self._card(name)
        if card is None:
            return Verdict(False)
        defs = self._defs()
        for index in range(_call_count(left, card, defs)):
            candidate, found = _replace_call(left, card, defs, index)
            if found and candidate == right:
                return Verdict(True, "W")
        return Verdict(False)

    def base(self, left: Expr, right: Expr) -> bool:
        """Obie strony po wstawieniu zera za `(var 0)`."""
        if left == right:
            return False
        return _same(left, right, Zero(), self._defs())

    def step(self, left: Expr, right: Expr) -> Step:
        """Cel to cała równość przy `(succ (var 0))`."""
        if left == right:
            raise ValueError("tautology")
        return Step(self, left, right)

    def _defs(self) -> dict[str, Def]:
        return {
            card.id: Def(card.id, card.arity, card.body) for card in self.shelf.entries()
        }

    def _card(self, name: str) -> Card | None:
        for card in self.shelf.entries():
            if card.id == name:
                return card
        return None


class Step:
    """Jedno `ih` na krok. Drugie wołanie jest odrzucane."""

    def __init__(self, kernel: Kernel, left: Expr, right: Expr) -> None:
        bump = Succ(Var(0))
        self._kernel = kernel
        self.thesis_left = left
        self.thesis_right = right
        self.current = _subst_index(left, 0, bump)
        self.goal_right = _subst_index(right, 0, bump)
        self._ih = False

    def rewrite(self, name: str) -> Expr:
        """Wstawia ciało. Warunku `if0` jeszcze nie rusza."""
        card = self._kernel._card(name)
        if card is None:
            raise ValueError("rewrite")
        candidate, found = _replace_call(self.current, card, self._kernel._defs(), 0)
        if not found:
            raise ValueError("rewrite")
        self.current = candidate
        return self.current

    def reduce(self) -> Expr:
        """Kasuje `if0`, którego warunek jest zerem albo następnikiem, i `pred` po `succ`."""
        self.current = _symbolic_reduce(self.current)
        return self.current

    def ih(self) -> Expr:
        """Raz podmienia lewą stronę tezy na prawą. Drugi raz w tym kroku odpada."""
        if self._ih:
            raise ValueError("ih")
        candidate, found = _replace_equal(self.current, self.thesis_left, self.thesis_right)
        if not found:
            raise ValueError("ih")
        self._ih = True
        self.current = candidate
        return self.current

    def closed(self) -> bool:
        return self.current == self.goal_right


def _interval(lo: int, hi: int) -> bool:
    return type(lo) is int and type(hi) is int and 0 <= lo <= hi


def _same(left: Expr, right: Expr, point: Expr, defs: dict[str, Def]) -> bool:
    try:
        got = reduce_expr(_subst_index(left, 0, point), defs=defs)
        expected = reduce_expr(_subst_index(right, 0, point), defs=defs)
    except (ReduceError, ValueError, ParseError):
        return False
    return got == expected


def _subst_index(expr: Expr, index: int, replacement: Expr) -> Expr:
    if isinstance(expr, Var):
        if expr.index == index:
            return replacement
        return expr
    if isinstance(expr, Succ):
        return Succ(_subst_index(expr.arg, index, replacement))
    if isinstance(expr, Pred):
        return Pred(_subst_index(expr.arg, index, replacement))
    if isinstance(expr, If0):
        return If0(
            _subst_index(expr.cond, index, replacement),
            _subst_index(expr.zero_branch, index, replacement),
            _subst_index(expr.nonzero_branch, index, replacement),
        )
    if isinstance(expr, Call):
        args = tuple(_subst_index(arg, index, replacement) for arg in expr.args)
        return Call(expr.slot, args)
    if isinstance(expr, Self):
        args = tuple(_subst_index(arg, index, replacement) for arg in expr.args)
        return Self(args)
    if isinstance(expr, (Zero, Num)):
        return expr
    raise TypeError(type(expr))


def _unfold(body: Expr, args: tuple[Expr, ...], slot: str) -> Expr:
    if isinstance(body, Var):
        return args[body.index]
    if isinstance(body, Self):
        return Call(slot, tuple(_unfold(arg, args, slot) for arg in body.args))
    if isinstance(body, Succ):
        return Succ(_unfold(body.arg, args, slot))
    if isinstance(body, Pred):
        return Pred(_unfold(body.arg, args, slot))
    if isinstance(body, If0):
        return If0(
            _unfold(body.cond, args, slot),
            _unfold(body.zero_branch, args, slot),
            _unfold(body.nonzero_branch, args, slot),
        )
    if isinstance(body, Call):
        return Call(body.slot, tuple(_unfold(arg, args, slot) for arg in body.args))
    if isinstance(body, (Zero, Num)):
        return body
    raise TypeError(type(body))


def _fits(card: Card, args: tuple[Expr, ...], defs: dict[str, Def]) -> bool:
    if card.arity == 1:
        limit = 10
    elif card.arity == 2:
        limit = 6
    else:
        return False
    if len(args) != card.arity:
        return False
    for arg in args:
        if _has_var(arg):
            continue
        try:
            value = reduce_expr(arg, defs=defs)
        except (ReduceError, ValueError, ParseError):
            return False
        if type(value) is not int or value < 0 or value > limit:
            return False
    return True


def _has_var(expr: Expr) -> bool:
    if isinstance(expr, Var):
        return True
    if isinstance(expr, (Succ, Pred)):
        return _has_var(expr.arg)
    if isinstance(expr, If0):
        return any(_has_var(part) for part in (expr.cond, expr.zero_branch, expr.nonzero_branch))
    if isinstance(expr, (Call, Self)):
        return any(_has_var(arg) for arg in expr.args)
    return False


def _call_count(expr: Expr, card: Card, defs: dict[str, Def]) -> int:
    count = 0
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Call):
            if node.slot == card.id and _fits(card, node.args, defs):
                count += 1
            stack.extend(node.args)
        elif isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, Self):
            stack.extend(node.args)
    return count


def _replace_call(
    expr: Expr,
    card: Card,
    defs: dict[str, Def],
    nth: int,
) -> tuple[Expr, bool]:
    seen = 0

    def go(node: Expr) -> Expr:
        nonlocal seen
        if isinstance(node, Call) and node.slot == card.id and _fits(card, node.args, defs):
            if seen == nth:
                seen += 1
                return _unfold(card.body, node.args, card.id)
            seen += 1
            return Call(node.slot, tuple(go(arg) for arg in node.args))
        if isinstance(node, Succ):
            return Succ(go(node.arg))
        if isinstance(node, Pred):
            return Pred(go(node.arg))
        if isinstance(node, If0):
            return If0(go(node.cond), go(node.zero_branch), go(node.nonzero_branch))
        if isinstance(node, Call):
            return Call(node.slot, tuple(go(arg) for arg in node.args))
        if isinstance(node, Self):
            return Self(tuple(go(arg) for arg in node.args))
        return node

    return go(expr), seen > nth


def _replace_equal(expr: Expr, target: Expr, replacement: Expr) -> tuple[Expr, bool]:
    seen = 0

    def go(node: Expr) -> Expr:
        nonlocal seen
        if node == target:
            if seen == 0:
                seen += 1
                return replacement
            seen += 1
        if isinstance(node, Succ):
            return Succ(go(node.arg))
        if isinstance(node, Pred):
            return Pred(go(node.arg))
        if isinstance(node, If0):
            return If0(go(node.cond), go(node.zero_branch), go(node.nonzero_branch))
        if isinstance(node, Call):
            return Call(node.slot, tuple(go(arg) for arg in node.args))
        if isinstance(node, Self):
            return Self(tuple(go(arg) for arg in node.args))
        return node

    return go(expr), seen > 0


def _symbolic_reduce(expr: Expr) -> Expr:
    current = expr
    for _ in range(64):
        current, changed = _reduce_once(current)
        if not changed:
            return current
    return current


def _reduce_once(expr: Expr) -> tuple[Expr, bool]:
    if isinstance(expr, Succ):
        arg, changed = _reduce_once(expr.arg)
        return Succ(arg), changed
    if isinstance(expr, Pred):
        arg, changed = _reduce_once(expr.arg)
        if isinstance(arg, Succ):
            return arg.arg, True
        if isinstance(arg, Zero) or (isinstance(arg, Num) and arg.value == 0):
            return Zero(), True
        return Pred(arg), changed
    if isinstance(expr, If0):
        cond, c1 = _reduce_once(expr.cond)
        zero, c2 = _reduce_once(expr.zero_branch)
        other, c3 = _reduce_once(expr.nonzero_branch)
        if _is_zero(cond):
            return zero, True
        if _is_nonzero(cond):
            return other, True
        return If0(cond, zero, other), c1 or c2 or c3
    if isinstance(expr, Call):
        args = []
        changed = False
        for arg in expr.args:
            reduced, one = _reduce_once(arg)
            args.append(reduced)
            changed = changed or one
        return Call(expr.slot, tuple(args)), changed
    if isinstance(expr, Self):
        args = []
        changed = False
        for arg in expr.args:
            reduced, one = _reduce_once(arg)
            args.append(reduced)
            changed = changed or one
        return Self(tuple(args)), changed
    return expr, False


def _is_zero(expr: Expr) -> bool:
    return isinstance(expr, Zero) or (isinstance(expr, Num) and expr.value == 0)


def _is_nonzero(expr: Expr) -> bool:
    return isinstance(expr, Succ) or (isinstance(expr, Num) and expr.value > 0)
