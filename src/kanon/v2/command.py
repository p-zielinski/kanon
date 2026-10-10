"""Jedno polecenie na wiele dni.

`python -m kanon.v2 --seed 0` zostaje w procesie. Najpierw przegląd do
16 tokenów. Bez czterech rodzin tydzień się nie zaczyna. Potem paczki
schematu, potem warsztat. Katalog domyślny to `runs/v2-seed0/`.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from kanon.language import Zero
from kanon.v2.batch import PACKAGE_SECONDS, PACKAGE_TERMS, run_package
from kanon.v2.model import build_nets
from kanon.v2.mutate import choose_fact, is_mutation, mutate, shared_facts
from kanon.v2.play import Clocks, Tape, play_exercise
from kanon.v2.record import Milestones, append_event, append_row, write_snapshot
from kanon.v2.resume import DEFAULT_DIR, Session
from kanon.v2.schema import (
    FLOOR_FAMILY,
    Schema,
    _family,
    _layouts,
    _place,
    _r1_width,
    _r2_width,
    comparison_shape,
    decode_p,
    decode_r1,
    decode_r2,
    gcd_shape,
)
from kanon.v2.shape import hole_count
from kanon.v2.shelf import Shelf
from kanon.v2.ship import ShipClock, send
from kanon.v2.survey import survey, week_ready
from kanon.v2.tactics import (
    Fact,
    c_candidates,
    decode_e,
    decode_i,
    decode_w,
    e_count,
    i_count,
    make_e,
    make_i,
    make_w,
    settle,
    w_count,
)
from kanon.v2.workshop import Task, Workshop

_Decode = Callable[[int], object]


@dataclass(frozen=True, slots=True)
class Plan:
    seed: int
    directory: Path
    package_terms: int
    package_seconds: int
    max_packages: int | None
    shape_count: int | None
    metric_every: int
    survey_limit: int


class _Probe:
    """Mały kształt testu. Nie jest R1 ani R2."""

    def __init__(self, count: int) -> None:
        self.name = "T"
        self.count = count

    def decode(self, index: int):
        return Zero()


class _Quiet:
    """Wydanie bez połączenia. Test biegu wielodniowego tu nie dochodzi."""

    def publish(self, name: str, *, clobber: bool, files: tuple[str, ...]) -> None:
        return None


class _Seek:
    """Kształt piętra. Pierwsze trafienie rodziny zatrzymuje paczkę."""

    def __init__(
        self,
        name: str,
        count: int,
        decode: _Decode,
        wanted: str | None,
        group: Shelf,
        solo: Shelf,
        schema: Schema,
        arity: int,
    ) -> None:
        self.name = name
        self.count = count
        self._decode = decode
        self.wanted = wanted
        self.group = group
        self.solo = solo
        self.schema = schema
        self.arity = arity

    def decode(self, index: int):
        term = self._decode(index)
        if self.schema.floor >= 9:
            self._tactic(term)
            return term
        if self.wanted is None or _family(term, self.arity, floor=self.schema.floor) != self.wanted:
            return term
        _place(self.group, self.solo, term, self.arity)
        self.wanted = None
        self.count = 0
        self._switch(index)
        return term

    def _tactic(self, found: object) -> None:
        fact = _fact(self.schema.shape, self.group, found)
        if fact is None:
            return
        if settle(self.group, self.solo, (fact,), floor=self.schema.floor, source=self.group) is None:
            return
        self.count = 0
        self.schema._advance()
        self.schema.index = -1

    def _switch(self, index: int) -> None:
        schema = self.schema
        if schema.floor == 2 and self.name == "R1":
            schema.shape = "R2"
            schema.index = -1
            return
        if schema.floor == 2 and self.name == "R2":
            if self.group.local_id("dodawanie") is not None:
                schema.floor = 3
                schema.shape = "R1"
                schema.index = -1
                return
            schema.shape = "R2"
            schema.floor = 2
            schema.index = index - 1
            self.count = index
            return
        schema._advance()
        schema.index = -1


class ExerciseBook:
    """Licznik ćwiczeń polecenia. Mutacja go nie podbija."""

    def __init__(self, seed: int) -> None:
        self._shop = Workshop(seed)

    def draw(self, shelf: Shelf, *, level: int, round_no: int):
        return self._shop.exercise(shelf, level=level, round_no=round_no)

    def mutate(self) -> None:
        self._shop.mutate()


def configure(argv: Sequence[str] | None = None) -> Plan:
    parser = argparse.ArgumentParser(prog="kanon.v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--package-terms", type=int, default=PACKAGE_TERMS)
    parser.add_argument("--max-packages", type=int, default=None)
    parser.add_argument("--shape-count", type=int, default=None)
    parser.add_argument("--metric-every", type=int, default=10)
    parser.add_argument("--survey-limit", type=int, default=16)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return Plan(
        args.seed,
        args.directory,
        args.package_terms,
        PACKAGE_SECONDS,
        args.max_packages,
        args.shape_count,
        args.metric_every,
        args.survey_limit,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run(configure(argv))


def run(plan: Plan) -> int:
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=plan.survey_limit)
    if not week_ready(group):
        return 1
    if plan.max_packages is None and plan.shape_count is None:
        _stay(plan, group, solo)
        return 0
    _bounded(plan, group, solo)
    return 0


def _bounded(plan: Plan, group: Shelf, solo: Shelf) -> None:
    session = Session(plan.directory)
    _remember(session)
    _snapshot(session, group, solo)
    width = plan.shape_count
    if width is None:
        width = plan.package_terms * (plan.max_packages or 1)
    shape = _Probe(width)
    done = 0
    cap = plan.max_packages if plan.max_packages is not None else width
    while done < cap and session.schema.index < shape.count:
        size = run_package(
            session.schema,
            shape,
            save=lambda state: _save(session, group, solo),
            exercise=lambda: None,
            clock=lambda count: 0.0,
            closed=lambda: False,
            terms=plan.package_terms,
            seconds=plan.package_seconds,
        )
        if size == 0:
            break
        done += 1
        _metric(plan, session, group, solo, done)


def _stay(plan: Plan, group: Shelf, solo: Shelf) -> None:
    """Paczki, potem warsztat. To wołanie zostaje w procesie."""
    session = Session(plan.directory)
    _remember(session)
    _snapshot(session, group, solo)
    origin = time.monotonic()
    ship = ShipClock()
    book = ExerciseBook(plan.seed)
    done = 0
    while session.schema.floor <= 12:
        shape = _shape_for(session.schema, group, solo)
        if session.schema.index >= shape.count:
            session.schema.log.append("schemat wyczerpany")
            if session.schema.floor == 2 and session.schema.shape == "R1":
                session.schema.shape = "R2"
                session.schema.index = 0
            else:
                session.schema._advance()
            continue
        started = time.monotonic()
        size = run_package(
            session.schema,
            shape,
            save=lambda state: _save(session, group, solo),
            exercise=lambda: _one_exercise(book, group, done),
            clock=lambda count: time.monotonic() - started,
            closed=lambda: _closed(group),
            terms=plan.package_terms,
            seconds=plan.package_seconds,
        )
        if size == 0:
            session.schema._advance()
            continue
        done += 1
        _metric(plan, session, group, solo, done)
        _ship(plan, ship, origin)
    clocks = Clocks()
    tape = Tape()
    solo_tape = Tape()
    round_no = 0
    while True:
        round_no += 1
        _workshop_round(plan, session, group, solo, round_no, clocks, tape, solo_tape, book)
        if session.workshop(round_no, time.monotonic() - origin):
            _ship(plan, ship, origin)


def _workshop_round(plan, session, group, solo, round_no, clocks, tape, solo_tape, book: ExerciseBook) -> None:
    if is_mutation(round_no, session.schema.floor):
        book.mutate()
        _mutate(group, solo)
        return
    task = book.draw(group, level=session.level, round_no=round_no)
    if not isinstance(task, Task):
        return
    play_exercise(
        group,
        task,
        round_no=round_no,
        message="(abstain f0)",
        tape=tape,
        clocks=clocks,
        solo_tape=solo_tape,
    )


def _one_exercise(book: ExerciseBook, group: Shelf, done: int) -> None:
    task = book.draw(group, level=0, round_no=done + 1)
    if not isinstance(task, Task):
        return
    play_exercise(
        group,
        task,
        round_no=done + 1,
        message="(abstain f0)",
        tape=Tape(),
        clocks=Clocks(),
    )


def _mutate(group: Shelf, solo: Shelf) -> None:
    names = tuple(f"f{index}" for index in range(32) if group.id_at(f"f{index}") is not None)
    if len(names) < 2:
        return
    fact = choose_fact("group", shared_facts(group, solo), {"group": ()})
    if fact is None:
        return
    mutate(group, fact, (names[0], names[1]))


def _shape_for(schema: Schema, group: Shelf, solo: Shelf) -> _Seek:
    targets = tuple((card.id, card.arity) for card in group.entries())
    name = schema.shape
    if schema.floor == 2 and name == "R2":
        return _Seek(
            "R2",
            _width_count(2, targets, _r2_width),
            lambda index: decode_r2(index, 2, targets),
            "odejmowanie",
            group,
            solo,
            schema,
            2,
        )
    if name == "P":
        return _Seek("P", hole_count(1) ** 2, decode_p, "parzystość", group, solo, schema, 1)
    if name == "N1":
        built = comparison_shape(group)
        return _Seek(built.name, built.count, built.decode, "porównanie", group, solo, schema, 2)
    if name == "G":
        built = gcd_shape(group)
        return _Seek(built.name, built.count, built.decode, "NWD", group, solo, schema, 2)
    if name == "R1":
        family = "dodawanie" if schema.floor == 2 else FLOOR_FAMILY.get(schema.floor)
        return _Seek(
            "R1",
            _width_count(2, targets, _r1_width),
            lambda index: decode_r1(index, 2, targets),
            family,
            group,
            solo,
            schema,
            2,
        )
    if name in ("E", "W", "C", "I"):
        return _tactic_shape(schema, group, solo)
    return _Seek(name, 0, lambda index: Zero(), None, group, solo, schema, 2)


def _tactic_shape(schema: Schema, group: Shelf, solo: Shelf) -> _Seek:
    name = schema.shape
    if name == "E":
        count, decode = e_count(group), lambda index: decode_e(group, index)
    elif name == "W":
        count, decode = w_count(group), lambda index: decode_w(group, index)
    elif name == "C":
        facts = c_candidates(group)
        count, decode = len(facts), lambda index, row=facts: row[index]
    elif name == "I":
        count, decode = i_count(group), lambda index: decode_i(group, index)
    else:
        count, decode = 0, lambda index: Zero()
    return _Seek(name, count, decode, None, group, solo, schema, 2)


def _fact(kind: str, shelf: Shelf, found: object):
    if kind == "E" and isinstance(found, tuple) and len(found) == 4:
        left, right, lo, hi = found
        return make_e(shelf, left, right, lo, hi)
    if kind == "W" and isinstance(found, tuple) and len(found) == 3:
        left, right, name = found
        return make_w(shelf, left, right, name)
    if kind == "C" and isinstance(found, Fact):
        return found
    if kind == "I" and isinstance(found, tuple) and len(found) == 3:
        left, right, name = found
        return make_i(shelf, left, right, name)
    return None


def _width_count(arity: int, targets: tuple[tuple[str, int], ...], width_of) -> int:
    base = hole_count(arity)
    return sum(base**width for _, _, width in _layouts(arity, targets, width_of))


def _closed(group: Shelf) -> bool:
    if group.local_id("dodawanie") is not None or group.local_id("odejmowanie") is not None:
        return True
    return any(group.local_id(name) is not None for name in FLOOR_FAMILY.values())


def _remember(session: Session) -> None:
    session.weights = {str(net.seed): net.state_dict() for net in build_nets()}


def _snapshot(session: Session, group: Shelf, solo: Shelf) -> None:
    write_snapshot(
        session.directory,
        group=group,
        solo=solo,
        local_group=(),
        local_solo=(),
        term=session.schema.index,
        level=session.level,
    )


def _save(session: Session, group: Shelf, solo: Shelf) -> None:
    session.write()
    _snapshot(session, group, solo)


def _metric(plan: Plan, session: Session, group: Shelf, solo: Shelf, done: int) -> None:
    append_row(
        plan.directory / "metrics.csv",
        count=done,
        source="paczka",
        schema=session.schema,
        exhausted=False,
        level=session.level,
        group=group,
        solo=solo,
        marks=Milestones(),
        group_hits=(),
        solo_hits=(),
        frozen_hits=(),
        local_group=(),
        local_solo=(),
        traces=0,
        bench=(),
        gap="",
        every=plan.metric_every,
    )


def _ship(plan: Plan, clock: ShipClock, origin: float) -> None:
    log: list[str] = []
    send(
        plan.directory,
        now=(time.monotonic() - origin) / 3600,
        saved=True,
        client=_Quiet(),
        env=os.environ,
        log=log,
        day=date.today(),
        clock=clock,
    )
    for line in log:
        append_event(plan.directory, line)
