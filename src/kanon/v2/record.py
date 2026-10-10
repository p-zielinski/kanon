"""Metryki, snapshot półki i obrót zdarzeń.

Wiersz co 10 paczek albo co 10 rund warsztatu. Numer paczki dodawania,
mnożenia i pierwszego faktu E jest osobno, ze znacznikiem „schemat”
albo „mutacja”. `events.jsonl` zostawia bieżący plik i jeden poprzedni.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from kanon.language import Expr, tokenize
from kanon.roles import format_expr
from kanon.v2.mutate import shared_facts
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.window import BODY_LEN, observation_prefix

EVENT_LIMIT = 32 * 1024 * 1024
_WINDOW = 50
_ORIGINS = ("schemat", "mutacja")
_HEADER = (
    "numer",
    "zrodlo",
    "frontera",
    "ksztalt",
    "term",
    "wyczerpany",
    "poziom",
    "grupa",
    "solo",
    "zamrozone",
    "polka",
    "wspolna",
    "lokalna_grupa",
    "lokalna_solo",
    "slady",
    "lawa",
    "dodawanie",
    "dodawanie_z",
    "mnozenie",
    "mnozenie_z",
    "fakt_e",
    "fakt_e_z",
    "luka",
)


@dataclass
class Milestone:
    package: int | None = None
    origin: str | None = None


class Milestones:
    """Numer paczki wejścia. Wielkość półki tego numeru nie ustawia."""

    def __init__(self) -> None:
        self.addition = Milestone()
        self.multiplication = Milestone()
        self.first_e = Milestone()

    def observe(self, group: Shelf, solo: Shelf, *, package: int, origin: str) -> None:
        if origin not in _ORIGINS:
            raise ValueError("origin")
        self._family(group, solo, "dodawanie", "addition", package, origin)
        self._family(group, solo, "mnożenie", "multiplication", package, origin)
        if self.first_e.package is not None:
            return
        for fact in tuple(group.hung()) + tuple(group.fact_pool()):
            if getattr(fact, "kind", None) == "E":
                self.first_e = Milestone(package, origin)
                return

    def _family(
        self,
        group: Shelf,
        solo: Shelf,
        family: str,
        attr: str,
        package: int,
        origin: str,
    ) -> None:
        slot: Milestone = getattr(self, attr)
        if slot.package is not None:
            return
        if group.local_id(family) is None or solo.local_id(family) is None:
            return
        setattr(self, attr, Milestone(package, origin))


def append_row(
    path: Path,
    *,
    count: int,
    source: str,
    schema: Schema,
    exhausted: bool,
    level: int,
    group: Shelf,
    solo: Shelf,
    marks: Milestones,
    group_hits: Sequence[bool],
    solo_hits: Sequence[bool],
    frozen_hits: Sequence[bool],
    local_group: Sequence[object],
    local_solo: Sequence[object],
    traces: int,
    bench: Sequence[str],
    gap: int | str,
    every: int = 10,
) -> bool:
    """Jeden wiersz na dziesiątą paczkę i na dziesiątą rundę warsztatu."""
    if count <= 0 or count % every != 0:
        return False
    fresh = not path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if fresh:
            writer.writerow(_HEADER)
        writer.writerow(
            [
                count,
                source,
                schema.floor,
                schema.shape,
                schema.index,
                "tak" if exhausted else "nie",
                level,
                _rate(group_hits),
                _rate(solo_hits),
                _rate(frozen_hits),
                len(group.entries()),
                len(shared_facts(group, solo)),
                len(local_group),
                len(local_solo),
                traces,
                " ".join(bench),
                _cell(marks.addition.package),
                _cell(marks.addition.origin),
                _cell(marks.multiplication.package),
                _cell(marks.multiplication.origin),
                _cell(marks.first_e.package),
                _cell(marks.first_e.origin),
                "" if schema.floor <= 8 or gap == "" else str(int(gap)),
            ]
        )
    return True


def write_snapshot(
    directory: Path,
    *,
    group: Shelf,
    solo: Shelf,
    local_group: Sequence[object],
    local_solo: Sequence[object],
    term: int,
    level: int,
) -> Path:
    """`canon/snapshot.json`. Ciało jest dla człowieka."""
    path = Path(directory) / "canon" / "snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "term": term,
        "poziom": level,
        "polka": [_card(group, card.id) for card in group.entries()],
        "wspolna": [_fact(fact) for fact in shared_facts(group, solo)],
        "lokalna": {
            "grupa": [_fact(fact) for fact in local_group],
            "solo": [_fact(fact) for fact in local_solo],
        },
    }
    path.write_text(json.dumps(document, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def agent_observation(pairs: Sequence[Sequence[int]], *, body: Expr | None = None) -> tuple[int, ...]:
    """Obserwacja agenta. Ciało, gdy się mieści, leży tylko w odcinku 48 tokenów."""
    words: tuple[str, ...] = ()
    if body is not None:
        found = tuple(tokenize(format_expr(body)))
        if len(found) <= BODY_LEN:
            words = found
    return observation_prefix(pairs, body=words)


def append_event(directory: Path, event: object, *, limit: int = EVENT_LIMIT) -> bool:
    """Dopisuje zdarzenie. Przy progu zostaje bieżący plik i jeden poprzedni."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    current = directory / "events.jsonl"
    previous = directory / "events.1.jsonl"
    encoded = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
    rotated = False
    if current.exists() and current.stat().st_size + len(encoded) > limit:
        if previous.exists():
            previous.unlink()
        current.rename(previous)
        rotated = True
    with current.open("ab") as handle:
        handle.write(encoded)
    return rotated


def _card(shelf: Shelf, ident: str) -> dict[str, object]:
    card = shelf.entry(ident)
    return {
        "id": ident,
        "nazwa": _name(shelf, ident),
        "rodzina": card.family,
        "dziedzina": _domain(card.arity),
        "ciało": format_expr(card.body),
        "powołania": list(card.citations),
    }


def _fact(fact: object) -> dict[str, object]:
    left = getattr(fact, "left", None)
    right = getattr(fact, "right", None)
    return {
        "rodzaj": getattr(fact, "kind", None),
        "lewa": format_expr(left) if isinstance(left, Expr) else None,
        "prawa": format_expr(right) if isinstance(right, Expr) else None,
    }


def _name(shelf: Shelf, ident: str) -> str | None:
    for index in range(32):
        name = f"f{index}"
        if shelf.id_at(name) == ident:
            return name
    return None


def _domain(arity: int) -> list[list[int]]:
    if arity == 1:
        return [[number] for number in range(11)]
    if arity == 2:
        return [[left, right] for left in range(7) for right in range(7)]
    raise ValueError("arity")


def _rate(flags: Sequence[bool]) -> str:
    window = tuple(flags)[-_WINDOW:]
    if not window:
        return ""
    return f"{sum(bool(item) for item in window) / len(window):.6f}"


def _cell(value: object) -> str:
    if value is None:
        return ""
    return str(value)
