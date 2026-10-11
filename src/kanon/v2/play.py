"""Runda ćwiczenia, taśma i zegar ε.

Jeden mówca, jedna próba. Rotacja v1 wybiera sieć, także gdy mówi łamacz.
`(break …)` i `(abstain …)` są pudłem. Trafienie dopisuje ten sam ślad
do czterech taśm grupy. Solo ma własną taśmę. Zamrożone wagi nie dostają
śladu ani kroku. Pusta taśma nie rusza wag.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from kanon.language import (
    Abstain,
    Break,
    Call,
    Def,
    Expr,
    If0,
    Num,
    ParseError,
    Pred,
    Succ,
    Var,
    Zero,
    expand_num,
    parse_expr,
    parse_message,
)
from kanon.reduce import ReduceError, reduce_expr
from kanon.roles import assign
from kanon.v2.batch import run_package
from kanon.v2.kernel import Kernel
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import Base, DomEval, Eval, Fact, Induct, Rewrite, chain_ok
from kanon.v2.window import TOKEN_ID, place_message
from kanon.v2.workshop import Task

TRACE_CAP = 256
AUTHOR_FLOOR = 0.25
EXECUTOR_FLOOR = 0.05
BREAKER_FLOOR = 0.5
DECAY = 0.995
_GROUP = 4
_ABSTAIN = "(abstain f0)"
_ROLES = ("author", "breaker", "user", "outsider")
_EXECUTOR = frozenset(("breaker", "user", "outsider"))

_Step = Callable[[object], None]


@dataclass(frozen=True, slots=True)
class RoundResult:
    speaker: int
    role: str
    attempts: int
    hit: bool
    solo_attempts: int
    solo_hit: bool
    frozen_attempts: int
    frozen_hit: bool


class Tape:
    """Ślady na agenta i na poziom warsztatu. Sufit 256, najstarszy odpada."""

    def __init__(self) -> None:
        self._rows: dict[tuple[int, int], list[tuple[int, ...]]] = {}

    def note(self, agent: int, level: int, tokens: tuple[int, ...]) -> None:
        row = self._rows.setdefault((agent, level), [])
        row.append(tuple(tokens))
        if len(row) > TRACE_CAP:
            del row[0]

    def traces(self, agent: int, level: int) -> tuple[tuple[int, ...], ...]:
        return tuple(self._rows.get((agent, level), ()))

    def count(self) -> int:
        """Ile śladów leży na taśmie, po wszystkich agentach i poziomach."""
        return sum(len(rows) for rows in self._rows.values())

    def export(self) -> dict[tuple[int, int], tuple[tuple[int, ...], ...]]:
        return {key: tuple(rows) for key, rows in self._rows.items()}

    def ingest(self, raw: dict) -> None:
        self._rows = {}
        for key, rows in (raw or {}).items():
            agent, level = int(key[0]), int(key[1])
            self._rows[(agent, level)] = [tuple(item) for item in rows]


class Clocks:
    """ε osobno dla autora i wykonawcy, osobno na każdym poziomie."""

    def __init__(self) -> None:
        self._author: dict[int, float] = {}
        self._executor: dict[int, float] = {}
        self._breaker: dict[int, float] = {}

    def author(self, level: int) -> float:
        return self._author.get(level, 1.0)

    def executor(self, level: int) -> float:
        return self._executor.get(level, 1.0)

    def breaker(self, level: int) -> float:
        """Podłoga 0,5 rusza dopiero na powtórce faktu lokalnego."""
        return self._breaker.get(level, 1.0)

    def tick_author(self, level: int) -> None:
        self._author[level] = _decay(self.author(level), AUTHOR_FLOOR)

    def tick_executor(self, level: int) -> None:
        self._executor[level] = _decay(self.executor(level), EXECUTOR_FLOOR)

    def tick_breaker(self, level: int) -> None:
        """×0,995 do 0,5. Wołane tylko na powtórce faktu lokalnego."""
        self._breaker[level] = _decay(self.breaker(level), BREAKER_FLOOR)

    def export(self) -> dict[str, dict[int, float]]:
        return {
            "author": dict(self._author),
            "executor": dict(self._executor),
            "breaker": dict(self._breaker),
        }

    def ingest(self, raw: dict | None) -> None:
        if not raw:
            return
        self._author = {int(key): float(value) for key, value in raw.get("author", {}).items()}
        self._executor = {int(key): float(value) for key, value in raw.get("executor", {}).items()}
        self._breaker = {int(key): float(value) for key, value in raw.get("breaker", {}).items()}


def speaker(round_no: int) -> tuple[int, str]:
    """Sieć i rola tej jednej próby. Rotacja jest taka jak w v1."""
    roles = assign(round_no)
    agents = (roles.author, roles.breaker, roles.user, roles.outsider)
    index = round_no % 4
    return agents[index], _ROLES[index]


def play_exercise(
    shelf: Shelf,
    task: Task,
    *,
    round_no: int,
    message: str,
    tape: Tape,
    clocks: Clocks,
    solo_tape: Tape | None = None,
    solo_message: str | None = None,
    frozen: object = None,
    frozen_message: str | None = None,
    weights: list[object] | None = None,
    step: _Step | None = None,
    admit: Callable[..., object] | None = None,
) -> RoundResult:
    """Jedna próba grupy, jedna solo, jedna zamrożonych wag. `admit` nie jest wołane."""
    del admit, frozen
    agent, role = speaker(round_no)
    hit = _accept(shelf, task, message)
    if hit:
        _note_group(tape, task, message)
    if role == "author":
        clocks.tick_author(task.level)
    elif hit and role in _EXECUTOR:
        clocks.tick_executor(task.level)
    if weights is not None and step is not None:
        for index, weight in enumerate(weights):
            apply_step(weight, tape, agent=index, level=task.level, step=step)
    solo_hit = _accept(shelf, task, _ABSTAIN if solo_message is None else solo_message)
    if solo_hit and solo_tape is not None and solo_message is not None:
        solo_tape.note(0, task.level, _trace(task, solo_message))
    frozen_hit = _accept(shelf, task, _ABSTAIN if frozen_message is None else frozen_message)
    return RoundResult(
        agent,
        role,
        1,
        hit,
        1,
        solo_hit,
        1,
        frozen_hit,
    )


def apply_step(
    weight: object,
    tape: Tape,
    *,
    agent: int,
    level: int,
    step: _Step,
) -> bool:
    """Pusta taśma nie woła kroku."""
    if not tape.traces(agent, level):
        return False
    step(weight)
    return True


def run_schema(schema: Schema, shape: object, *, clocks: Clocks, closed: bool) -> None:
    """Pętla schematu bez rundy ćwiczenia. Zegar zostaje."""
    del clocks
    run_package(
        schema,
        shape,
        save=lambda state: None,
        exercise=lambda: None,
        clock=lambda done: 0.0,
        closed=lambda: closed,
    )


def encode_message(source: str) -> tuple[int, ...]:
    return tuple(TOKEN_ID[word] for word in _words(source))


def _note_group(tape: Tape, task: Task, message: str) -> None:
    tokens = _trace(task, message)
    for agent in range(_GROUP):
        tape.note(agent, task.level, tokens)


def _trace(task: Task, message: str) -> tuple[int, ...]:
    return place_message(task.observation.tokens, encode_message(message))


def _accept(shelf: Shelf, task: Task, message: str) -> bool:
    if task.level >= 4 and task.fact is not None:
        return _proof(shelf, task, message)
    try:
        parsed = parse_message(message, arity=2)
    except ParseError:
        parsed = None
    if isinstance(parsed, (Break, Abstain)) or parsed is not None:
        return False
    try:
        expr = parse_expr(message, arity=2, allow_num=True)
    except ParseError:
        expr = None
    if expr is not None:
        return _matches(shelf, task, _rename(_expand_nums(expr), task.cards))
    return _tactic(shelf, task, message)


def _matches(shelf: Shelf, task: Task, expr: Expr) -> bool:
    defs = {card.id: Def(card.id, card.arity, card.body) for card in shelf.entries()}
    found: list[int] = []
    for point in task.inputs:
        try:
            found.append(reduce_expr(expr, defs=defs, env=point))
        except (ReduceError, ValueError):
            return False
    return tuple(found) == task.verdicts


def _proof(shelf: Shelf, task: Task, message: str) -> bool:
    """Kernel ocenia taktykę mówcy. Teza jest już w zadaniu."""
    fact = task.fact
    if not isinstance(fact, Fact):
        return False
    try:
        tactics = _tactics_from(message, task.cards)
    except (ValueError, IndexError, TypeError):
        return False
    if task.level >= 6:
        if not 2 <= len(tactics) <= 8:
            return False
        return chain_ok(shelf, Fact(fact.left, fact.right, "C", tactics))
    if task.level == 5 and not _fuel_ok(shelf, fact):
        return False
    kernel = Kernel(shelf)
    if fact.kind == "E":
        tactic = tactics[0] if len(tactics) == 1 else None
        return isinstance(tactic, DomEval) and kernel.eval(fact.left, fact.right, tactic.lo, tactic.hi)
    if fact.kind == "W":
        tactic = tactics[0] if len(tactics) == 1 else None
        return isinstance(tactic, Rewrite) and kernel.rewrite(fact.left, fact.right, tactic.name).accepted
    if fact.kind == "I":
        return (
            len(tactics) == 2
            and isinstance(tactics[0], Base)
            and isinstance(tactics[1], Induct)
            and kernel.base(fact.left, fact.right)
        )
    return bool(tactics) and all(_tactic_holds(kernel, fact, tactic) for tactic in tactics)


def _fuel_ok(shelf: Shelf, fact: Fact) -> bool:
    defs = {card.id: Def(card.id, card.arity, card.body) for card in shelf.entries()}
    try:
        reduce_expr(fact.left, defs=defs, env=(0, 0))
        reduce_expr(fact.right, defs=defs, env=(0, 0))
    except (ReduceError, ValueError):
        return False
    return True


def _tactic_holds(kernel: Kernel, fact: Fact, tactic: object) -> bool:
    if isinstance(tactic, DomEval):
        return kernel.eval(fact.left, fact.right, tactic.lo, tactic.hi)
    if isinstance(tactic, Eval):
        return kernel.eval(fact.left, fact.right, 0, 1)
    if isinstance(tactic, Rewrite):
        return kernel.rewrite(fact.left, fact.right, tactic.name).accepted
    if isinstance(tactic, Base):
        return kernel.base(fact.left, fact.right)
    return False


def _tactics_from(message: str, cards: tuple[str, str]) -> tuple[object, ...]:
    words = _words(message)
    index = 0
    found: list[object] = []
    while index < len(words):
        tactic, index = _read_tactic(words, index, cards)
        found.append(tactic)
    if not found:
        raise ValueError("empty")
    return tuple(found)


def _read_tactic(words: list[str], index: int, cards: tuple[str, str]) -> tuple[object, int]:
    if words[index] != "(":
        raise ValueError("open")
    head = words[index + 1]
    index += 2
    if head == "eval":
        return Eval(), _expect(words, index, ")")
    if head == "base":
        return Base(), _expect(words, index, ")")
    if head == "rewrite":
        name = _window_card(words[index], cards)
        return Rewrite(name), _expect(words, index + 1, ")")
    if head == "dom":
        lo = int(words[index])
        hi = int(words[index + 1])
        index = _expect(words, index + 2, ")")
        index = _expect(words, index, "(")
        index = _expect(words, index, "eval")
        index = _expect(words, index, ")")
        if not 0 <= lo < hi <= 10:
            raise ValueError("dom")
        return DomEval(lo, hi), index
    if head == "step":
        index = _expect(words, index, "(")
        index = _expect(words, index, "ih")
        index = _expect(words, index, ")")
        index = _expect(words, index, "(")
        index = _expect(words, index, "rewrite")
        name = _window_card(words[index], cards)
        index = _expect(words, index + 1, ")")
        return Induct(name), _expect(words, index, ")")
    raise ValueError("head")


def _expect(words: list[str], index: int, token: str) -> int:
    if words[index] != token:
        raise ValueError("token")
    return index + 1


def _window_card(token: str, cards: tuple[str, str]) -> str:
    if token == "f0":
        return cards[0]
    if token == "f1":
        return cards[1]
    raise ValueError("name")


def _tactic(shelf: Shelf, task: Task, message: str) -> bool:
    try:
        tree = _nest(_words(message))
    except ValueError:
        return False
    if not isinstance(tree, list) or not tree or tree[0] != "claim" or len(tree) != 3:
        return False
    eq, tactic = tree[1], tree[2]
    if (
        not isinstance(eq, list)
        or len(eq) != 3
        or eq[0] != "eq"
        or not isinstance(tactic, list)
        or not tactic
        or tactic[0] != "eval"
    ):
        return False
    try:
        left = _from_tree(eq[1], task.cards)
        right = _from_tree(eq[2], task.cards)
    except (ValueError, TypeError, IndexError):
        return False
    return Kernel(shelf).eval(left, right, 0, 1)


def _expand_nums(expr: Expr) -> Expr:
    """`(num k)` w argumencie jest wieżą `succ`, którą reduktor już liczy."""
    if isinstance(expr, Num):
        return expand_num(expr.value)
    if isinstance(expr, Call):
        return Call(expr.slot, tuple(_expand_nums(arg) for arg in expr.args))
    if isinstance(expr, Succ):
        return Succ(_expand_nums(expr.arg))
    if isinstance(expr, Pred):
        return Pred(_expand_nums(expr.arg))
    if isinstance(expr, If0):
        return If0(
            _expand_nums(expr.cond),
            _expand_nums(expr.zero_branch),
            _expand_nums(expr.nonzero_branch),
        )
    return expr


def _rename(expr: Expr, cards: tuple[str, str]) -> Expr:
    window = {"f0": cards[0], "f1": cards[1]}
    if isinstance(expr, Call):
        slot = window.get(expr.slot, expr.slot)
        return Call(slot, tuple(_rename(arg, cards) for arg in expr.args))
    if isinstance(expr, Succ):
        return Succ(_rename(expr.arg, cards))
    if isinstance(expr, Pred):
        return Pred(_rename(expr.arg, cards))
    if isinstance(expr, If0):
        return If0(
            _rename(expr.cond, cards),
            _rename(expr.zero_branch, cards),
            _rename(expr.nonzero_branch, cards),
        )
    return expr


def _from_tree(node: object, cards: tuple[str, str]) -> Expr:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("expr")
    head, *rest = node
    if head == "zero" and not rest:
        return Zero()
    if head == "succ" and len(rest) == 1:
        return Succ(_from_tree(rest[0], cards))
    if head == "pred" and len(rest) == 1:
        return Pred(_from_tree(rest[0], cards))
    if head == "var" and len(rest) == 1:
        return Var(int(rest[0]))
    if head == "call" and rest:
        slot = rest[0]
        if slot == "f0":
            slot = cards[0]
        elif slot == "f1":
            slot = cards[1]
        if not isinstance(slot, str):
            raise ValueError("slot")
        return Call(slot, tuple(_from_tree(arg, cards) for arg in rest[1:]))
    raise ValueError("expr")


def _decay(value: float, floor: float) -> float:
    nxt = value * DECAY
    if nxt < floor:
        return floor
    return nxt


def _words(source: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    size = len(source)
    while index < size:
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char in "()":
            tokens.append(char)
            index += 1
            continue
        if char.isdigit():
            end = index + 1
            while end < size and source[end].isdigit():
                end += 1
            digits = source[index:end]
            if digits not in TOKEN_ID:
                raise ValueError("token")
            tokens.append(digits)
            index = end
            continue
        if char.isascii() and char.isalpha():
            end = index + 1
            while end < size and source[end].isascii() and source[end].isalnum():
                end += 1
            word = source[index:end]
            if word not in TOKEN_ID:
                raise ValueError("token")
            tokens.append(word)
            index = end
            continue
        raise ValueError("token")
    return tokens


def _nest(words: list[str]) -> list:
    stack: list[list] = [[]]
    for word in words:
        if word == "(":
            stack.append([])
            continue
        if word == ")":
            if len(stack) == 1:
                raise ValueError("paren")
            finished = stack.pop()
            stack[-1].append(finished)
            continue
        if len(stack) == 1:
            raise ValueError("bare")
        stack[-1].append(word)
    if len(stack) != 1 or len(stack[0]) != 1 or not isinstance(stack[0][0], list):
        raise ValueError("term")
    return stack[0][0]
