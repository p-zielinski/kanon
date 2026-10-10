"""Filtr E, pula faktów i łańcuch taktyk. Piętra 9–12.

Jedno wstawienie ciała nie jest faktem E. Łańcuch dokleja jedną taktykę
do faktu, który już jest. Iloczynu ciągów taktyk tu nie ma.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from kanon.language import Call, Expr, If0, Num, Pred, Self, Succ, Var, Zero
from kanon.roles import format_expr
from kanon.v2.kernel import Kernel
from kanon.v2.shape import decode_hole, hole_count
from kanon.v2.shelf import Shelf
from kanon.v2.window import BODY_LEN

CHAIN_TACTICS = 8
CHAIN_TOKENS = 80
_HIGH = frozenset(
    {
        "dodawanie",
        "odejmowanie",
        "mnożenie",
        "porównanie",
        "minimum",
        "maksimum",
        "parzystość",
        "suma",
        "NWD",
        "potęga",
    }
)
_FLOOR_KIND = {9: "E", 10: "W", 11: "C", 12: "I"}


@dataclass(frozen=True, slots=True)
class DomEval:
    lo: int
    hi: int


@dataclass(frozen=True, slots=True)
class Eval:
    """Doklejone `(eval)`. Dziedzina tego kroku to 0..1."""


@dataclass(frozen=True, slots=True)
class Rewrite:
    name: str


@dataclass(frozen=True, slots=True)
class Base:
    pass


@dataclass(frozen=True, slots=True)
class Induct:
    name: str


@dataclass(frozen=True, slots=True)
class Fact:
    left: Expr
    right: Expr
    kind: str
    tactics: tuple[object, ...]


def admits_e(shelf: Shelf, left: Expr, right: Expr, lo: int, hi: int) -> bool:
    """Fakt E. Jedno wstawienie ciała, nawet na szerokiej dziedzinie, odpada."""
    if type(lo) is not int or type(hi) is not int:
        return False
    if not (0 <= lo <= 10 and 0 <= hi <= 10 and hi > lo):
        return False
    if format_expr(left) == format_expr(right):
        return False
    if not (_side_ok(left) and _side_ok(right)):
        return False
    if not (_calls_high(left, shelf) or _calls_high(right, shelf)):
        return False
    if one_insertion(shelf, left, right):
        return False
    return Kernel(shelf).eval(left, right, lo, hi)


def one_insertion(shelf: Shelf, left: Expr, right: Expr) -> bool:
    """Prawa albo lewa strona jest jednym wstawieniem ciała wpisu."""
    kernel = Kernel(shelf)
    for card in shelf.entries():
        if kernel.rewrite(left, right, card.id).accepted:
            return True
        if kernel.rewrite(right, left, card.id).accepted:
            return True
    return False


def make_e(shelf: Shelf, left: Expr, right: Expr, lo: int, hi: int) -> Fact | None:
    if not admits_e(shelf, left, right, lo, hi):
        return None
    return Fact(left, right, "E", (DomEval(lo, hi),))


def make_w(shelf: Shelf, left: Expr, right: Expr, name: str) -> Fact | None:
    verdict = Kernel(shelf).rewrite(left, right, name)
    if not verdict.accepted or verdict.kind != "W":
        return None
    return Fact(left, right, "W", (Rewrite(name),))


def make_i(shelf: Shelf, left: Expr, right: Expr, name: str) -> Fact | None:
    if not induction_ok(shelf, left, right, name):
        return None
    return Fact(left, right, "I", (Base(), Induct(name)))


def e_count(shelf: Shelf) -> int:
    """Ile jest par stron i dziedzin. Lista termów nie powstaje."""
    width = len(_sides(shelf))
    return width * width * len(_domains())


def encode_e(shelf: Shelf, left: Expr, right: Expr, lo: int, hi: int) -> int:
    """Numer równania E. Ten sam numer wraca z `decode_e`."""
    sides = _sides(shelf)
    width = len(sides)
    domain = _domains().index((lo, hi))
    return (domain * width + sides.index(left)) * width + sides.index(right)


def decode_e(shelf: Shelf, index: int) -> tuple[Expr, Expr, int, int]:
    """Strony i dziedzina o tym numerze."""
    sides = _sides(shelf)
    width = len(sides)
    domain, rest = divmod(index, width * width)
    left, right = divmod(rest, width)
    lo, hi = _domains()[domain]
    return sides[left], sides[right], lo, hi


def w_count(shelf: Shelf) -> int:
    names = shelf.entries()
    if not names:
        return 0
    width = len(_sides(shelf))
    return width * width * len(names)


def decode_w(shelf: Shelf, index: int) -> tuple[Expr, Expr, str]:
    sides = _sides(shelf)
    names = tuple(card.id for card in shelf.entries())
    width = len(sides)
    name, rest = divmod(index, width * width)
    left, right = divmod(rest, width)
    return sides[left], sides[right], names[name]


def c_candidates(shelf: Shelf) -> tuple[Fact, ...]:
    """Jedna doklejona taktyka do faktu, który już leży. Iloczynu ciągów nie ma."""
    facts = tuple(fact for fact in tuple(shelf.hung()) + tuple(shelf.fact_pool()) if isinstance(fact, Fact))
    return extend_once(facts, shelf)


def i_count(shelf: Shelf) -> int:
    binaries = tuple(card for card in shelf.entries() if card.arity == 2)
    names = shelf.entries()
    if not binaries or not names:
        return 0
    return len(binaries) * 3 * len(_sides(shelf)) * len(names)


def decode_i(shelf: Shelf, index: int) -> tuple[Expr, Expr, str]:
    binaries = tuple(card for card in shelf.entries() if card.arity == 2)
    names = tuple(card.id for card in shelf.entries())
    sides = _sides(shelf)
    block = 3 * len(sides) * len(names)
    which, rest = divmod(index, block)
    skeleton, rest = divmod(rest, len(sides) * len(names))
    right, name = divmod(rest, len(names))
    return _skeleton(binaries[which].id, skeleton), sides[right], names[name]


def _sides(shelf: Shelf) -> tuple[Expr, ...]:
    holes = tuple(decode_hole(index, 2) for index in range(hole_count(2)))
    calls: list[Expr] = []
    for card in shelf.entries():
        base = hole_count(card.arity)
        for index in range(base**card.arity):
            digits = _digits(index, base, card.arity)
            args = tuple(decode_hole(digit, card.arity) for digit in digits)
            calls.append(Call(card.id, args))
    return holes + tuple(calls)


def _domains() -> tuple[tuple[int, int], ...]:
    return tuple((lo, hi) for lo in range(11) for hi in range(lo + 1, 11))


def _digits(value: int, base: int, width: int) -> tuple[int, ...]:
    digits = [0] * width
    for pos in range(width - 1, -1, -1):
        value, digits[pos] = divmod(value, base)
    return tuple(digits)


def _skeleton(slot: str, which: int) -> Expr:
    if which == 0:
        return Call(slot, (Var(0), Zero()))
    if which == 1:
        return Call(slot, (Zero(), Var(0)))
    return Call(slot, (Var(0), Succ(Zero())))


def is_skeleton(expr: Expr, shelf: Shelf) -> bool:
    """`f(x,0)`, `f(0,x)` albo `f(x,1)`. `f` jest wpisem binarnym półki."""
    if not isinstance(expr, Call) or len(expr.args) != 2:
        return False
    if not any(card.id == expr.slot and card.arity == 2 for card in shelf.entries()):
        return False
    left, right = expr.args
    zero = Zero()
    one = Succ(zero)
    variable = Var(0)
    return (left, right) in ((variable, zero), (zero, variable), (variable, one))


def induction_ok(shelf: Shelf, left: Expr, right: Expr, name: str) -> bool:
    """`base` i jedno `step` z `ih` oraz `rewrite`."""
    if not is_skeleton(left, shelf):
        return False
    if not any(card.id == name for card in shelf.entries()):
        return False
    kernel = Kernel(shelf)
    if not kernel.base(left, right):
        return False
    try:
        step = kernel.step(left, right)
        step.rewrite(name)
        step.reduce()
        step.ih()
    except ValueError:
        return False
    return step.closed()


def extend_once(facts: Sequence[Fact], shelf: Shelf) -> tuple[Fact, ...]:
    """Dokleja jedną taktykę `eval` albo `rewrite`. Nie składa iloczynu ciągów."""
    extras: list[object] = [Eval()]
    extras.extend(Rewrite(card.id) for card in shelf.entries())
    found: list[Fact] = []
    for fact in facts:
        if fact.kind not in ("E", "W") or len(fact.tactics) >= CHAIN_TACTICS:
            continue
        for extra in extras:
            longer = Fact(fact.left, fact.right, "C", fact.tactics + (extra,))
            if chain_ok(shelf, longer):
                found.append(longer)
    return tuple(found)


def chain_ok(shelf: Shelf, fact: Fact) -> bool:
    if not 1 <= len(fact.tactics) <= CHAIN_TACTICS:
        return False
    if thesis_tokens(fact) > CHAIN_TOKENS:
        return False
    kernel = Kernel(shelf)
    return all(_tactic_ok(kernel, fact.left, fact.right, tac) for tac in fact.tactics)


def settle(
    group: Shelf,
    solo: Shelf,
    facts: Sequence[Fact],
    *,
    floor: int,
    source: Shelf,
) -> Fact | None:
    """Najkrótszy fakt piętra na obie półki, potem co najwyżej osiem do puli."""
    kind = _FLOOR_KIND.get(floor)
    if kind is None:
        return None
    closing = [fact for fact in facts if fact.kind == kind and _holds(source, fact)]
    if not closing:
        return None
    closing.sort(key=thesis_tokens)
    best = closing[0]
    if best in group.hung() or best in solo.hung():
        return None
    group.hang(best)
    solo.hang(best)
    rest = [
        fact
        for fact in facts
        if fact != best and fact.kind in ("E", "W") and _holds(source, fact)
    ]
    rest.sort(key=lambda fact: (0 if fact.kind == "E" else 1, thesis_tokens(fact)))
    for fact in rest:
        if thesis_tokens(fact) > BODY_LEN:
            continue
        if group.pool_fact(fact):
            solo.pool_fact(fact)
    return best


def admit_pool(shelf: Shelf, fact: Fact) -> bool:
    """Teza dłuższa niż 48 tokenów do puli nie wchodzi."""
    if thesis_tokens(fact) > BODY_LEN:
        return False
    return shelf.pool_fact(fact)


def thesis_tokens(fact: Fact) -> int:
    return len(claim_tokens(fact))


def claim_tokens(fact: Fact) -> tuple[str, ...]:
    parts = ["(", "claim", "(", "eq", *expr_tokens(fact.left), *expr_tokens(fact.right), ")"]
    for tactic in fact.tactics:
        parts.extend(_tactic_tokens(tactic))
    parts.append(")")
    return tuple(parts)


def _holds(shelf: Shelf, fact: Fact) -> bool:
    if fact.kind == "E" and fact.tactics and isinstance(fact.tactics[0], DomEval):
        dom = fact.tactics[0]
        return admits_e(shelf, fact.left, fact.right, dom.lo, dom.hi)
    if fact.kind == "W" and fact.tactics and isinstance(fact.tactics[0], Rewrite):
        verdict = Kernel(shelf).rewrite(fact.left, fact.right, fact.tactics[0].name)
        return verdict.accepted and verdict.kind == "W"
    if fact.kind == "C":
        return len(fact.tactics) >= 2 and chain_ok(shelf, fact)
    if fact.kind == "I":
        names = [tac.name for tac in fact.tactics if isinstance(tac, Induct)]
        return bool(names) and induction_ok(shelf, fact.left, fact.right, names[0])
    return False


def _tactic_ok(kernel: Kernel, left: Expr, right: Expr, tactic: object) -> bool:
    if isinstance(tactic, DomEval):
        return kernel.eval(left, right, tactic.lo, tactic.hi)
    if isinstance(tactic, Eval):
        return kernel.eval(left, right, 0, 1)
    if isinstance(tactic, Rewrite):
        return kernel.rewrite(left, right, tactic.name).accepted
    if isinstance(tactic, Base):
        return kernel.base(left, right)
    if isinstance(tactic, Induct):
        return False
    return False


def _side_ok(expr: Expr) -> bool:
    if _is_s(expr):
        return True
    return isinstance(expr, Call) and bool(expr.args) and all(_is_s(arg) for arg in expr.args)


def _is_s(expr: Expr) -> bool:
    if isinstance(expr, Zero):
        return True
    if isinstance(expr, Var):
        return expr.index >= 0
    if isinstance(expr, (Succ, Pred)):
        return isinstance(expr.arg, (Zero, Var))
    return False


def _calls_high(expr: Expr, shelf: Shelf) -> bool:
    wanted = {card.id for card in shelf.entries() if card.family in _HIGH}
    return any(slot in wanted for slot in _slots(expr))


def _slots(expr: Expr) -> tuple[str, ...]:
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


def _tactic_tokens(tactic: object) -> tuple[str, ...]:
    if isinstance(tactic, DomEval):
        return ("(", "dom", str(tactic.lo), str(tactic.hi), ")", "(", "eval", ")")
    if isinstance(tactic, Eval):
        return ("(", "eval", ")")
    if isinstance(tactic, Rewrite):
        return ("(", "rewrite", tactic.name, ")")
    if isinstance(tactic, Base):
        return ("(", "base", ")")
    if isinstance(tactic, Induct):
        return ("(", "step", "(", "ih", ")", "(", "rewrite", tactic.name, ")", ")")
    raise TypeError(type(tactic))


def expr_tokens(expr: Expr) -> tuple[str, ...]:
    if isinstance(expr, Zero):
        return ("(", "zero", ")")
    if isinstance(expr, Var):
        return ("(", "var", str(expr.index), ")")
    if isinstance(expr, Num):
        return ("(", "num", str(expr.value), ")")
    if isinstance(expr, Succ):
        return ("(", "succ", *expr_tokens(expr.arg), ")")
    if isinstance(expr, Pred):
        return ("(", "pred", *expr_tokens(expr.arg), ")")
    if isinstance(expr, If0):
        return (
            "(",
            "if0",
            *expr_tokens(expr.cond),
            *expr_tokens(expr.zero_branch),
            *expr_tokens(expr.nonzero_branch),
            ")",
        )
    if isinstance(expr, Call):
        parts = ["(", "call", expr.slot]
        for arg in expr.args:
            parts.extend(expr_tokens(arg))
        parts.append(")")
        return tuple(parts)
    if isinstance(expr, Self):
        parts = ["(", "self"]
        for arg in expr.args:
            parts.extend(expr_tokens(arg))
        parts.append(")")
        return tuple(parts)
    raise TypeError(type(expr))
