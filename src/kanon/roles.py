"""Role rundy, parser slotu i obserwacja.

Przypisanie na rundzie r: autor r%4, łamacz (r+1)%4, użytkownik (r+2)%4,
obcy (r+3)%4. Parser przyjmuje tylko kształt swojej roli. Odrzut to sam
token [ABS]. Szkiców jest od 1 do 4; odpowiedź na szkic to [TRY] i pięć
wyników par publicznych.

Obserwacja jest ciągiem tokenów ze słownika (liczba po `num` jest dana).
Obcy nie dostaje ciała kandydata ani termu trzymanego na obiekcie audytowym.
Łamacz dostaje dodatkowo wejścia z cmentarza: `(err …)` gdy wynik się
rozjechał, `(zero …)` gdy się zgadzał. Termu celu tam nie ma.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from kanon.canon import Canon
from kanon.language import (
    MAX_ARITY,
    Abstain,
    Break,
    Call,
    Def,
    DraftDef,
    DraftExpr,
    Expr,
    If0,
    Message,
    Num,
    ParseError,
    Pred,
    Propose,
    Self,
    Solve,
    Succ,
    Use,
    Var,
    Zero,
    parse_message,
)
from kanon.reduce import ReduceError, reduce_def, reduce_expr
from kanon.tasks import IOPair

ABS = "[ABS]"
MAX_DRAFTS = 4
MAX_BREAKS = 4
_FIVE_ERR = "[TRY] (err) (err) (err) (err) (err)"


class Role(Enum):
    AUTHOR = "author"
    BREAKER = "breaker"
    USER = "user"
    OUTSIDER = "outsider"


_MARKER = {
    Role.AUTHOR: "[AUTHOR]",
    Role.BREAKER: "[BREAK]",
    Role.USER: "[USE]",
    Role.OUTSIDER: "[OUTSIDER]",
}

_SHAPES: dict[Role, tuple[type, ...]] = {
    Role.AUTHOR: (DraftDef, Propose),
    Role.BREAKER: (Break, Abstain),
    Role.USER: (DraftExpr, Use),
    Role.OUTSIDER: (DraftExpr, Solve),
}


@dataclass(frozen=True, slots=True)
class RoundRoles:
    author: int
    breaker: int
    user: int
    outsider: int


@dataclass(frozen=True, slots=True)
class Accepted:
    message: Message


def assign(round_no: int) -> RoundRoles:
    if type(round_no) is not int or round_no < 0:
        raise ValueError("round")
    return RoundRoles(
        author=round_no % 4,
        breaker=(round_no + 1) % 4,
        user=(round_no + 2) % 4,
        outsider=(round_no + 3) % 4,
    )


def format_expr(expr: Expr) -> str:
    if isinstance(expr, Zero):
        return "(zero)"
    if isinstance(expr, Succ):
        return f"(succ {format_expr(expr.arg)})"
    if isinstance(expr, Pred):
        return f"(pred {format_expr(expr.arg)})"
    if isinstance(expr, If0):
        return (
            f"(if0 {format_expr(expr.cond)} "
            f"{format_expr(expr.zero_branch)} "
            f"{format_expr(expr.nonzero_branch)})"
        )
    if isinstance(expr, Var):
        return f"(var {expr.index})"
    if isinstance(expr, Num):
        return f"(num {expr.value})"
    if isinstance(expr, Call):
        rendered = " ".join([expr.slot, *(format_expr(arg) for arg in expr.args)])
        return f"(call {rendered})"
    if isinstance(expr, Self):
        rendered = " ".join(format_expr(arg) for arg in expr.args)
        return f"(self {rendered})" if rendered else "(self)"
    raise TypeError(type(expr))


def format_def(definition: Def) -> str:
    return f"(def {definition.slot} {definition.arity} {format_expr(definition.body)})"


def observe(
    role: Role,
    *,
    canon: Canon,
    pairs: Sequence[IOPair],
    free_slot: str | None = None,
    candidate: Def | None = None,
) -> str:
    """Ciąg tokenów dla roli. Nie czyta termu celu ani par audytowych."""
    parts = [_canon_section(canon), _pairs_section(pairs)]
    if role is Role.AUTHOR and free_slot is not None:
        parts.append(free_slot)
    elif role is Role.BREAKER and candidate is not None:
        parts.append(format_def(candidate))
    elif role is Role.USER and candidate is not None:
        parts.append(candidate.slot)
    if role is Role.BREAKER:
        parts.extend(_cemetery(canon))
    parts.append(_MARKER[role])
    return " ".join(parts)


class SlotSession:
    """Parser jednego slotu roli. Nie przyjmuje definicji do kanonu."""

    def __init__(
        self,
        role: Role,
        pairs: Sequence[IOPair],
        *,
        canon: Canon | None = None,
        free_slot: str | None = None,
        candidate_slot: str | None = None,
    ) -> None:
        public = tuple(pairs)
        if len(public) != 5:
            raise ValueError("public pairs must be 5")
        self.role = role
        self.pairs = public
        self.arity = len(public[0].inputs)
        self.canon = Canon() if canon is None else canon
        self.free_slot = free_slot
        self.candidate_slot = candidate_slot
        self.drafts = 0
        self.breaks = 0
        self.closed = False

    def submit(self, source: str) -> str | Accepted:
        if self.closed:
            return ABS
        try:
            message = parse_message(source, arity=MAX_ARITY)
        except ParseError:
            return ABS
        if not isinstance(message, _SHAPES[self.role]):
            return ABS
        if isinstance(message, (DraftDef, DraftExpr)):
            return self._draft(message)
        if isinstance(message, Propose):
            return self._finish_def(message)
        if isinstance(message, Use):
            return self._finish_use(message)
        if isinstance(message, Solve):
            return self._finish(message)
        if isinstance(message, Break):
            return self._break(message)
        if isinstance(message, Abstain):
            self.closed = True
            return Accepted(message)
        return ABS

    def _draft(self, message: DraftDef | DraftExpr) -> str:
        if self.drafts >= MAX_DRAFTS:
            return ABS
        if isinstance(message, DraftDef) and not self._slot_ok(message.definition.slot):
            return ABS
        self.drafts += 1
        if isinstance(message, DraftDef):
            if message.definition.arity != self.arity:
                return _FIVE_ERR
            values = [
                _reduce_def(message.definition, pair.inputs, _canon_defs(self.canon))
                for pair in self.pairs
            ]
            return _try_line(values)
        if _too_wide(message.expr, self.arity):
            return _FIVE_ERR
        defs = _canon_defs(self.canon)
        values = [_reduce_expr(message.expr, pair.inputs, defs) for pair in self.pairs]
        return _try_line(values)

    def _finish_def(self, message: Propose) -> str | Accepted:
        if self.drafts < 1 or not self._slot_ok(message.definition.slot):
            return ABS
        self.closed = True
        return Accepted(message)

    def _finish_use(self, message: Use) -> str | Accepted:
        if self.drafts < 1:
            return ABS
        if self.candidate_slot is not None and message.slot != self.candidate_slot:
            return ABS
        if not _calls_slot(message.expr, message.slot):
            return ABS
        self.closed = True
        return Accepted(message)

    def _finish(self, message: Solve) -> str | Accepted:
        if self.drafts < 1:
            return ABS
        self.closed = True
        return Accepted(message)

    def _break(self, message: Break) -> str | Accepted:
        if self.breaks >= MAX_BREAKS:
            return ABS
        self.breaks += 1
        return Accepted(message)

    def _slot_ok(self, slot: str) -> bool:
        return self.free_slot is None or slot == self.free_slot


def _cemetery(canon: Canon) -> list[str]:
    rendered = []
    for case in canon.breaker_export():
        head = "err" if case.diverged else "zero"
        nums = " ".join(f"(num {value})" for value in case.inputs)
        rendered.append(f"({head} {nums})" if nums else f"({head})")
    return rendered


def _canon_section(canon: Canon) -> str:
    parts = ["[CANON]"]
    entries = canon.replay()
    if not entries:
        parts.append("[SEP]")
    for entry in entries:
        parts.append(format_def(entry.definition))
        parts.append("[SEP]")
    return " ".join(parts)


def _pairs_section(pairs: Sequence[IOPair]) -> str:
    parts = ["[PAIRS]"]
    for pair in pairs:
        for value in (*pair.inputs, pair.output):
            parts.append(f"(num {value})")
    return " ".join(parts)


def _canon_defs(canon: Canon) -> dict[str, Def]:
    return {entry.definition.slot: entry.definition for entry in canon.replay()}


def _reduce_def(definition: Def, inputs: tuple[int, ...], defs: dict[str, Def]) -> int | None:
    try:
        return reduce_def(definition, inputs, defs=defs)
    except (ReduceError, ValueError, ParseError):
        return None


def _reduce_expr(expr: Expr, inputs: tuple[int, ...], defs: dict[str, Def]) -> int | None:
    try:
        return reduce_expr(expr, defs=defs, env=inputs)
    except (ReduceError, ValueError, ParseError):
        return None


def _too_wide(expr: Expr, arity: int) -> bool:
    return any(isinstance(node, Var) and node.index >= arity for node in _walk(expr))


def _calls_slot(expr: Expr, slot: str) -> bool:
    return any(isinstance(node, Call) and node.slot == slot for node in _walk(expr))


def _walk(expr: Expr):
    stack = [expr]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, (Call, Self)):
            stack.extend(node.args)


def _try_line(values: list[int | None]) -> str:
    parts = ["[TRY]"]
    for value in values:
        parts.append("(err)" if value is None else f"(num {value})")
    return " ".join(parts)
