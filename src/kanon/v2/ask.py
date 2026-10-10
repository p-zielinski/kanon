"""Odczyt zapisu. Jeden term albo jedno zdanie, bez kroku i bez zapisu.

Ładuje `resume.pt` i półkę ze snapshotu. Werdykt jest liczbą reduktora
albo przyjęciem kernela. Półki, numeru termu, poziomu i puli nie rusza.
"""

from __future__ import annotations

import json
from pathlib import Path

from kanon.language import Call, Def, Expr, If0, Num, Pred, Self, Succ, Var, Zero, parse_expr
from kanon.reduce import reduce_expr
from kanon.v2.kernel import Kernel
from kanon.v2.resume import Session
from kanon.v2.shelf import Shelf


def ask(directory: Path, text: str) -> str:
    """Wypisuje werdykt. Kroku uczenia nie woła i nic nie zapisuje."""
    directory = Path(directory)
    Session.open(directory)
    shelf = _shelf(directory)
    if text.lstrip().startswith("(step"):
        return _sentence(shelf, text)
    return str(_term(shelf, text))


def _shelf(directory: Path) -> Shelf:
    document = json.loads((directory / "canon" / "snapshot.json").read_text(encoding="utf-8"))
    shelf = Shelf()
    names: dict[str, str] = {}
    for card in document["polka"]:
        points = card["dziedzina"]
        arity = len(points[0]) if points else 1
        body = parse_expr(card["ciało"], arity=arity, allow_self=True)
        ident = shelf.write(Def("f0", arity, body))
        name = card["nazwa"]
        if isinstance(name, str):
            names[name] = ident
    bound = tuple(names[f"f{index}"] for index in range(32) if f"f{index}" in names)
    if bound:
        shelf.bind(bound)
    return shelf


def _term(shelf: Shelf, text: str) -> int:
    expr = parse_expr(text, arity=2, allow_num=True)
    return reduce_expr(_resolve(expr, shelf), defs=_defs(shelf))


def _sentence(shelf: Shelf, text: str) -> str:
    tree = _nest(_lex(text))
    if not isinstance(tree, list) or len(tree) < 2 or tree[0] != "step":
        return "odrzucone"
    equation = tree[1]
    if not isinstance(equation, list) or len(equation) != 3 or equation[0] != "eq":
        return "odrzucone"
    try:
        walked = Kernel(shelf).step(_node(equation[1], shelf), _node(equation[2], shelf))
        for tactic in tree[2:]:
            if not isinstance(tactic, list) or not tactic:
                raise ValueError("tactic")
            if tactic == ["ih"]:
                walked.ih()
            elif tactic == ["reduce"]:
                walked.reduce()
            elif tactic[0] == "rewrite" and len(tactic) == 2:
                name = tactic[1]
                bound = shelf.id_at(name) if isinstance(name, str) else None
                walked.rewrite(bound if bound is not None else str(name))
            else:
                raise ValueError("tactic")
    except ValueError:
        return "odrzucone"
    if walked.closed():
        return "przyjęte"
    return "odrzucone"


def _resolve(expr: Expr, shelf: Shelf) -> Expr:
    if isinstance(expr, Call):
        bound = shelf.id_at(expr.slot)
        slot = bound if bound is not None else expr.slot
        return Call(slot, tuple(_resolve(arg, shelf) for arg in expr.args))
    if isinstance(expr, Succ):
        return Succ(_resolve(expr.arg, shelf))
    if isinstance(expr, Pred):
        return Pred(_resolve(expr.arg, shelf))
    if isinstance(expr, If0):
        return If0(
            _resolve(expr.cond, shelf),
            _resolve(expr.zero_branch, shelf),
            _resolve(expr.nonzero_branch, shelf),
        )
    if isinstance(expr, Self):
        return Self(tuple(_resolve(arg, shelf) for arg in expr.args))
    return expr


def _node(node: object, shelf: Shelf) -> Expr:
    if not isinstance(node, list) or not node or not isinstance(node[0], str):
        raise ValueError("expr")
    head, *rest = node
    if head == "zero" and not rest:
        return Zero()
    if head == "succ" and len(rest) == 1:
        return Succ(_node(rest[0], shelf))
    if head == "pred" and len(rest) == 1:
        return Pred(_node(rest[0], shelf))
    if head == "var" and len(rest) == 1:
        return Var(int(str(rest[0])))
    if head == "num" and len(rest) == 1:
        return Num(int(str(rest[0])))
    if head == "call" and rest and isinstance(rest[0], str):
        bound = shelf.id_at(rest[0])
        slot = bound if bound is not None else rest[0]
        return Call(slot, tuple(_node(arg, shelf) for arg in rest[1:]))
    raise ValueError("expr")


def _defs(shelf: Shelf) -> dict[str, Def]:
    return {card.id: Def(card.id, card.arity, card.body) for card in shelf.entries()}


def _lex(source: str) -> list[str]:
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
            tokens.append(source[index:end])
            index = end
            continue
        if char.isascii() and char.isalpha():
            end = index + 1
            while end < size and source[end].isascii() and source[end].isalnum():
                end += 1
            tokens.append(source[index:end])
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
                raise ValueError("nest")
            done = stack.pop()
            stack[-1].append(done)
            continue
        stack[-1].append(word)
    if len(stack) != 1:
        raise ValueError("nest")
    return stack[0][0]
