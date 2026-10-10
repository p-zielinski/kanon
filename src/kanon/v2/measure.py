"""Zbiór pomiarowy. Pięćdziesiąt zadań powstaje raz, po zejściu z piętra 8.

Źródłem są wpisy schematu obecne na obu półkach. Karty są składnikami,
bez znacznika rodziny. Co 50 rund warsztatu zbiór jest liczony bez kroku.
Luka jest różnicą sukcesu grupy i mocniejszej z dwóch pozostałych krzywych.
Progu, który uruchamia piątą sieć, nie ma.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from kanon.language import Call, Def, Expr, Succ, Var
from kanon.roles import format_expr
from kanon.v2.shelf import Card, Shelf
from kanon.v2.workshop import vet

SET_SIZE = 50
SET_FILE = "holdout.json"
SET_GAP = 1_000_003
_FRESH = 5
_HEADER = ("runda", "frontera", "luka")
_SEEDS = (0, 1, 2, 3, 4, 5)

_Answer = Callable[[object], bool]
_Step = Callable[[], None]


@dataclass(frozen=True, slots=True)
class Score:
    gap: int | str
    seeds: tuple[int, ...]
    counted: bool


def set_seed(seed: int) -> int:
    """Ziarno zbioru. Nie jest ziarnem rundy."""
    if type(seed) is not int:
        raise ValueError("seed")
    return seed + SET_GAP


def build_set(
    directory: Path,
    *,
    floor: int,
    group: Shelf,
    solo: Shelf,
    seed: int,
) -> Path | None:
    """Pisze plik raz, gdy frontera jest już za piętrem 8. Wcześniej go nie ma."""
    if floor <= 8:
        return None
    path = Path(directory) / SET_FILE
    if path.is_file():
        return path
    drawn = set_seed(seed)
    tasks = _tasks(group, solo, drawn)
    document = {"seed": drawn, "tasks": [_task(item) for item in tasks]}
    path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_metric(path: Path, *, round_no: int, floor: int, gap: int) -> None:
    """Kolumna luki jest w nagłówku. Do zejścia z piętra 8 komórka jest pusta."""
    cell = "" if floor <= 8 else str(int(gap))
    fresh = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        if fresh:
            handle.write(",".join(_HEADER) + "\n")
        handle.write(f"{round_no},{floor},{cell}\n")


def score_set(
    round_no: int,
    tasks: Sequence[object],
    *,
    floor: int,
    answers: tuple[_Answer, _Answer, _Answer],
    step: _Step,
) -> Score:
    """Co 50 rund. Kroku nie woła. Sześć sieci zostaje przy luce 2 i przy luce 20."""
    del step
    if round_no <= 0 or round_no % 50 != 0 or floor <= 8:
        return Score("", _SEEDS, False)
    group, solo, frozen = answers
    hits = tuple(sum(1 for task in tasks if answer(task)) for answer in (group, solo, frozen))
    return Score(hits[0] - max(hits[1], hits[2]), _SEEDS, True)


def _tasks(group: Shelf, solo: Shelf, seed: int) -> tuple[tuple[tuple[str, str], Expr, tuple[tuple[int, ...], ...], tuple[int, ...]], ...]:
    cards = _shared(group, solo)
    if not cards:
        raise ValueError("shared")
    rng = random.Random(seed)
    arities = {card.id: card.arity for card in cards}
    defs = {card.id: Def(card.id, card.arity, card.body) for card in cards}
    found: list[tuple[tuple[str, str], Expr, tuple[tuple[int, ...], ...], tuple[int, ...]]] = []
    guard = 0
    while len(found) < SET_SIZE:
        guard += 1
        if guard > SET_SIZE * 20:
            raise ValueError("set")
        pair = _pair(cards, rng)
        term = _compose(pair, arities, rng, 1 + len(found) % 4)
        inputs = _inputs(rng, _free_arity(term))
        outputs = vet(term, defs, inputs)
        if outputs is None:
            continue
        found.append((pair, term, inputs, outputs))
    return tuple(found)


def _task(
    item: tuple[tuple[str, str], Expr, tuple[tuple[int, ...], ...], tuple[int, ...]],
) -> dict[str, object]:
    cards, term, inputs, outputs = item
    return {
        "cards": list(cards),
        "term": format_expr(term),
        "inputs": [list(point) for point in inputs],
        "outputs": list(outputs),
    }


def _shared(group: Shelf, solo: Shelf) -> tuple[Card, ...]:
    other = {card.id: card for card in solo.entries()}
    found: list[Card] = []
    for card in group.entries():
        twin = other.get(card.id)
        if twin is not None and twin.body == card.body and twin.arity == card.arity:
            found.append(card)
    return tuple(found)


def _pair(cards: Sequence[Card], rng: random.Random) -> tuple[str, str]:
    if len(cards) == 1:
        return (cards[0].id, cards[0].id)
    left, right = rng.sample(list(cards), 2)
    return (left.id, right.id)


def _compose(
    cards: tuple[str, str],
    arities: Mapping[str, int],
    rng: random.Random,
    calls: int,
) -> Expr:
    expr: Expr | None = None
    for _ in range(calls):
        slot = cards[rng.randrange(len(cards))]
        arity = arities[slot]
        if expr is None:
            args = tuple(Var(index) for index in range(arity))
        else:
            args = (expr,) + tuple(Var(0) for _ in range(arity - 1))
        expr = Call(slot, args)
    if expr is None:
        raise ValueError("calls")
    return expr


def _inputs(rng: random.Random, arity: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(rng.randint(0, 6) for _ in range(arity)) for _ in range(_FRESH))


def _free_arity(expr: Expr) -> int:
    highest = -1
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Var):
            highest = max(highest, node.index)
        elif isinstance(node, Succ):
            stack.append(node.arg)
        elif isinstance(node, Call):
            stack.extend(node.args)
    if highest < 0:
        raise ValueError("var")
    return highest + 1
