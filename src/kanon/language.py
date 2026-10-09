"""Tokeny, parser i kształty komunikatów języka Kalk.

Słownik `TOKENS` jest listą z PLAN.md. Głowy `propose`, `use`, `solve`,
`break` i `abstain` nie są w tej liście, a parser komunikatów je przyjmuje,
bo bez nich nie da się zapisać kształtów z sekcji „Role”.

`(if0 cond zero-branch nonzero-branch)`: zero bierze pierwszą gałąź.
Inaczej definicja dodawania z PLAN.md nie redukuje się do 5.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Union

MAX_ARITY = 3
SLOTS: tuple[str, ...] = tuple(f"f{i}" for i in range(32))
MESSAGE_HEADS: tuple[str, ...] = ("propose", "use", "solve", "break", "abstain")

TOKENS: tuple[str, ...] = (
    "(",
    ")",
    "zero",
    "succ",
    "pred",
    "if0",
    "var",
    "call",
    "self",
    "def",
    "arity",
    "num",
    "draft",
    "err",
    *SLOTS,
    "0",
    "1",
    "2",
    "3",
    "[AUTHOR]",
    "[BREAK]",
    "[USE]",
    "[OUTSIDER]",
    "[PAIRS]",
    "[CANON]",
    "[SEP]",
    "[ABS]",
    "[TRY]",
)

_WORDS = frozenset(TOKENS) - frozenset(SLOTS) - {"(", ")"}
_MARKERS = frozenset(tok for tok in TOKENS if tok.startswith("["))
_LEXEMES = _WORDS | frozenset(SLOTS) | frozenset(MESSAGE_HEADS)
_VAR_DIGITS = frozenset("0123")


class ParseError(Exception):
    """Term albo komunikat jest poza gramatyką Kalk."""


@dataclass(frozen=True, slots=True)
class Zero:
    pass


@dataclass(frozen=True, slots=True)
class Succ:
    arg: Expr


@dataclass(frozen=True, slots=True)
class Pred:
    arg: Expr


@dataclass(frozen=True, slots=True)
class If0:
    cond: Expr
    zero_branch: Expr
    nonzero_branch: Expr


@dataclass(frozen=True, slots=True)
class Var:
    index: int


@dataclass(frozen=True, slots=True)
class Call:
    slot: str
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Self:
    args: tuple[Expr, ...]


@dataclass(frozen=True, slots=True)
class Num:
    """`(num k)` w danych. W ciele def parser tego nie wpuszcza."""

    value: int


Expr = Union[Zero, Succ, Pred, If0, Var, Call, Self, Num]


@dataclass(frozen=True, slots=True)
class Def:
    slot: str
    arity: int
    body: Expr


@dataclass(frozen=True, slots=True)
class DraftDef:
    definition: Def


@dataclass(frozen=True, slots=True)
class DraftExpr:
    expr: Expr


@dataclass(frozen=True, slots=True)
class Propose:
    definition: Def


@dataclass(frozen=True, slots=True)
class Use:
    slot: str
    expr: Expr


@dataclass(frozen=True, slots=True)
class Solve:
    expr: Expr


@dataclass(frozen=True, slots=True)
class Break:
    slot: str
    args: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Abstain:
    slot: str


Message = Union[DraftDef, DraftExpr, Propose, Use, Solve, Break, Abstain]


@dataclass(frozen=True, slots=True)
class _Ctx:
    arity: int
    allow_self: bool
    allow_num: bool


def expand_num(k: int) -> Expr:
    """`(num k)` rozwinięte do `succ` powtórzonego k razy wokół `(zero)`."""
    if type(k) is not int or k < 0:
        raise ValueError("num requires a non-negative int")
    expr: Expr = Zero()
    for _ in range(k):
        expr = Succ(expr)
    return expr


def node_count(expr: Expr) -> int:
    count = 0
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        count += 1
        if isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.append(node.cond)
            stack.append(node.zero_branch)
            stack.append(node.nonzero_branch)
        elif isinstance(node, (Call, Self)):
            stack.extend(node.args)
    return count


def tokenize(source: str) -> list[str]:
    return [str(tok) if type(tok) is int else tok for tok in _lex(source)]


def parse_expr(
    source: str,
    *,
    arity: int = 0,
    allow_self: bool = False,
    allow_num: bool = False,
) -> Expr:
    node = _nest(_lex(source))
    return _convert(lambda: _as_expr(node, _ctx(arity, allow_self, allow_num)))


def parse_data(source: str) -> Expr:
    """Wyrażenie w danych: pary IO i argumenty łamacza. `(num k)` jest legalne."""
    return parse_expr(source, allow_num=True)


def parse_def(source: str) -> Def:
    node = _nest(_lex(source))
    return _convert(lambda: _as_def(node))


def parse_message(source: str, *, arity: int = 0) -> Message:
    """Komunikat roli. `arity` dotyczy expr użytkownika i obcego, nie def autora."""
    node = _nest(_lex(source))
    ctx = _ctx(arity, False, False)
    return _convert(lambda: _as_message(node, ctx))


def _ctx(arity: int, allow_self: bool, allow_num: bool) -> _Ctx:
    if type(arity) is not int or not 0 <= arity <= MAX_ARITY:
        raise ParseError("arity out of range")
    return _Ctx(arity, allow_self, allow_num)


def _lex(source: str) -> list[str | int]:
    tokens: list[str | int] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(ch)
            i += 1
            continue
        if ch == "[":
            end = source.find("]", i + 1)
            if end < 0:
                raise ParseError("unclosed marker")
            marker = source[i : end + 1]
            if marker not in _MARKERS:
                raise ParseError("unknown marker")
            tokens.append(marker)
            i = end + 1
            continue
        if ch.isdigit():
            j = i + 1
            while j < n and source[j].isdigit():
                j += 1
            digits = source[i:j]
            if len(digits) > 1 and digits[0] == "0":
                raise ParseError("leading zero")
            previous = tokens[-1] if tokens else None
            if previous == "num":
                tokens.append(int(digits))
            elif digits in _VAR_DIGITS and len(digits) == 1:
                tokens.append(int(digits))
            else:
                raise ParseError("numeral out of vocabulary")
            i = j
            continue
        if ch.isascii() and ch.isalpha():
            j = i + 1
            while j < n and source[j].isascii() and source[j].isalnum():
                j += 1
            word = source[i:j]
            if word not in _LEXEMES:
                raise ParseError("unknown token")
            tokens.append(word)
            i = j
            continue
        raise ParseError("illegal character")
    return tokens


def _nest(tokens: list[str | int]) -> list:
    stack: list[list] = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
            continue
        if tok == ")":
            if len(stack) == 1:
                raise ParseError("unbalanced paren")
            finished = stack.pop()
            stack[-1].append(finished)
            continue
        if len(stack) == 1:
            raise ParseError("bare token")
        stack[-1].append(tok)
    if len(stack) != 1 or len(stack[0]) != 1 or not isinstance(stack[0][0], list):
        raise ParseError("expected one term")
    return stack[0][0]


def _as_def(node: list) -> Def:
    if len(node) != 4 or node[0] != "def":
        raise ParseError("expected def")
    _, slot, arity, body = node
    if slot not in SLOTS or type(arity) is not int:
        raise ParseError("def")
    ctx = _ctx(arity, True, False)
    return Def(slot, ctx.arity, _as_expr(body, ctx))


def _convert(build):
    # Kręgosłup o długości limitu węzłów mieści się w termie, a domyślny
    # limit rekurencji Pythona kończy parsowanie wcześniej.
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 8000))
    try:
        return build()
    except RecursionError as exc:
        raise ParseError("term too deep") from exc
    finally:
        sys.setrecursionlimit(old)


def _as_expr(node: object, ctx: _Ctx) -> Expr:
    if not isinstance(node, list) or not node:
        raise ParseError("expected term")
    head, *rest = node
    if head == "zero":
        if rest:
            raise ParseError("zero takes no arguments")
        return Zero()
    if head == "succ":
        if len(rest) != 1:
            raise ParseError("succ takes one argument")
        return Succ(_as_expr(rest[0], ctx))
    if head == "pred":
        if len(rest) != 1:
            raise ParseError("pred takes one argument")
        return Pred(_as_expr(rest[0], ctx))
    if head == "if0":
        if len(rest) != 3:
            raise ParseError("if0 takes three arguments")
        return If0(
            _as_expr(rest[0], ctx),
            _as_expr(rest[1], ctx),
            _as_expr(rest[2], ctx),
        )
    if head == "var":
        if len(rest) != 1 or type(rest[0]) is not int or rest[0] >= ctx.arity:
            raise ParseError("var out of range")
        return Var(rest[0])
    if head == "call":
        if not rest or not isinstance(rest[0], str) or rest[0] not in SLOTS:
            raise ParseError("call needs a slot")
        if len(rest) - 1 > MAX_ARITY:
            raise ParseError("call arity")
        return Call(rest[0], tuple(_as_expr(arg, ctx) for arg in rest[1:]))
    if head == "self":
        if not ctx.allow_self:
            raise ParseError("self outside def")
        if len(rest) != ctx.arity:
            raise ParseError("self arity")
        return Self(tuple(_as_expr(arg, ctx) for arg in rest))
    if head == "num":
        if not ctx.allow_num:
            raise ParseError("num outside data")
        if len(rest) != 1 or type(rest[0]) is not int:
            raise ParseError("num takes one integer")
        return Num(rest[0])
    raise ParseError("not an expression")


def _as_message(node: list, ctx: _Ctx) -> Message:
    if not node:
        raise ParseError("empty message")
    head, *rest = node
    if head == "draft":
        if len(rest) != 1 or not isinstance(rest[0], list) or not rest[0]:
            raise ParseError("draft")
        inner = rest[0]
        if inner[0] == "def":
            return DraftDef(_as_def(inner))
        return DraftExpr(_as_expr(inner, ctx))
    if head == "propose":
        if len(rest) != 1 or not isinstance(rest[0], list):
            raise ParseError("propose")
        return Propose(_as_def(rest[0]))
    if head == "use":
        if len(rest) != 2 or rest[0] not in SLOTS:
            raise ParseError("use")
        return Use(rest[0], _as_expr(rest[1], ctx))
    if head == "solve":
        if len(rest) != 1:
            raise ParseError("solve")
        return Solve(_as_expr(rest[0], ctx))
    if head == "break":
        if len(rest) < 2 or rest[0] not in SLOTS:
            raise ParseError("break")
        return Break(rest[0], tuple(_as_num(arg) for arg in rest[1:]))
    if head == "abstain":
        if len(rest) != 1 or rest[0] not in SLOTS:
            raise ParseError("abstain")
        return Abstain(rest[0])
    raise ParseError("unknown message")


def _as_num(node: object) -> int:
    if (
        not isinstance(node, list)
        or len(node) != 2
        or node[0] != "num"
        or type(node[1]) is not int
    ):
        raise ParseError("break argument must be num")
    return node[1]
