"""Jedno polecenie na wiele dni.

`python -m kanon.v2 --seed 0` zostaje w procesie. Najpierw przegląd do
16 tokenów. Bez czterech rodzin tydzień się nie zaczyna. Potem paczki
schematu, potem warsztat. Katalog domyślny to `runs/v2-seed0/`.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import torch

from kanon.language import (
    SLOTS,
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
    parse_expr,
)
from kanon.roles import format_expr
from kanon.v2.batch import PACKAGE_SECONDS, PACKAGE_TERMS, run_package
from kanon.v2.kernel import Kernel
from kanon.v2.learn import LR, step
from kanon.v2.model import build_nets
from kanon.v2.mutate import choose_fact, is_mutation, mutate, shared_facts
from kanon.v2.play import Clocks, Tape, encode_message, play_exercise, speaker
from kanon.v2.ramp import Ramp
from kanon.v2.record import Milestones, agent_observation, append_event, append_row, write_snapshot
from kanon.v2.resume import DEFAULT_DIR, Session, arm_workshop
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
    Base,
    DomEval,
    Eval,
    Fact,
    Induct,
    Rewrite,
    c_candidates,
    decode_e,
    decode_i,
    decode_w,
    e_count,
    i_count,
    induction_ok,
    make_e,
    make_i,
    make_w,
    settle,
    w_count,
)
from kanon.v2.voice import utter
from kanon.v2.window import place_message
from kanon.v2.workshop import Task, Workshop

_SHELF_ID = re.compile(r"\be\d+\b")

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
        on_accept: Callable[[object], None] | None = None,
    ) -> None:
        self.name = name
        self.count = count
        self._decode = decode
        self.wanted = wanted
        self.group = group
        self.solo = solo
        self.schema = schema
        self.arity = arity
        self.on_accept = on_accept

    def decode(self, index: int):
        term = self._decode(index)
        if self.schema.floor >= 9:
            self._tactic(term)
            return term
        if self.wanted is None or _family(term, self.arity, floor=self.schema.floor) != self.wanted:
            return term
        _place(self.group, self.solo, term, self.arity)
        if self.on_accept is not None:
            self.on_accept(term)
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
        if self.on_accept is not None:
            self.on_accept(fact)
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
    if plan.max_packages is None and plan.shape_count is None:
        session, group, solo, resumed = prepare(plan)
        if not resumed and not week_ready(group):
            return 1
        _stay(plan, group, solo, session)
        return 0
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=plan.survey_limit)
    if not week_ready(group):
        return 1
    _bounded(plan, group, solo)
    return 0


def prepare(plan: Plan) -> tuple[Session, Shelf, Shelf, bool]:
    """Zapis i snapshot wracają bez przeglądu. Inaczej przegląd składa półki."""
    restored = open_run(plan.directory)
    if restored is not None:
        session, group, solo = restored
        return session, group, solo, True
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=plan.survey_limit)
    return Session(plan.directory), group, solo, False


def open_run(directory: Path) -> tuple[Session, Shelf, Shelf] | None:
    directory = Path(directory)
    snapshot = directory / "canon" / "snapshot.json"
    if not (directory / "resume.pt").is_file() or not snapshot.is_file():
        return None
    session = Session.open(directory)
    return session, shelf_from_snapshot(snapshot), shelf_from_snapshot(snapshot)


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


def _stay(plan: Plan, group: Shelf, solo: Shelf, session: Session) -> None:
    """Paczki, potem warsztat. To wołanie zostaje w procesie."""
    run = _boot(plan, session)
    _snapshot(session, group, solo)
    origin = time.monotonic()
    ship = ShipClock()
    book = ExerciseBook(plan.seed)
    done = 0
    while session.schema.floor <= 12:
        shape = _shape_for(
            session.schema,
            group,
            solo,
            on_accept=lambda found: note_schema(run.nets[:4], run.tape, found),
        )
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
            save=lambda state: _save(session, group, solo, run),
            exercise=lambda: _one_exercise(book, group, solo, done, run),
            clock=lambda count: time.monotonic() - started,
            closed=lambda: _closed(group),
            terms=plan.package_terms,
            seconds=plan.package_seconds,
        )
        if size == 0:
            session.schema._advance()
            continue
        done += 1
        write_package_metric(plan, session, group, solo, done, run.scores, run.tape)
        _ship(plan, ship, origin)
    round_no = arm_workshop(session)
    while True:
        round_no += 1
        _workshop_round(book, group, solo, round_no, run)
        write_package_metric(
            plan,
            session,
            group,
            solo,
            round_no,
            run.scores,
            run.tape,
            source="runda",
        )
        elapsed = time.monotonic() - origin
        if session.clock.due(round_no, elapsed):
            _save(session, group, solo, run)
            _ship(plan, ship, origin)


def _workshop_round(book: ExerciseBook, group: Shelf, solo: Shelf, round_no: int, run: _Live) -> None:
    if is_mutation(round_no, run.session.schema.floor):
        book.mutate()
        _mutate(group, solo)
        return
    task = book.draw(group, level=run.session.level, round_no=round_no)
    if not isinstance(task, Task):
        return
    _speak(book, group, solo, task, round_no, run)


def _one_exercise(book: ExerciseBook, group: Shelf, solo: Shelf, done: int, run: _Live) -> None:
    task = book.draw(group, level=0, round_no=done + 1)
    if not isinstance(task, Task):
        return
    _speak(book, group, solo, task, done + 1, run)


def _mutate(group: Shelf, solo: Shelf) -> None:
    names = tuple(f"f{index}" for index in range(32) if group.id_at(f"f{index}") is not None)
    if len(names) < 2:
        return
    fact = choose_fact("group", shared_facts(group, solo), {"group": ()})
    if fact is None:
        return
    mutate(group, fact, (names[0], names[1]))


def _shape_for(
    schema: Schema,
    group: Shelf,
    solo: Shelf,
    on_accept: Callable[[object], None] | None = None,
) -> _Seek:
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
            on_accept,
        )
    if name == "P":
        return _Seek("P", hole_count(1) ** 2, decode_p, "parzystość", group, solo, schema, 1, on_accept)
    if name == "N1":
        built = comparison_shape(group)
        return _Seek(built.name, built.count, built.decode, "porównanie", group, solo, schema, 2, on_accept)
    if name == "G":
        built = gcd_shape(group)
        return _Seek(built.name, built.count, built.decode, "NWD", group, solo, schema, 2, on_accept)
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
            on_accept,
        )
    if name in ("E", "W", "C", "I"):
        return _tactic_shape(schema, group, solo, on_accept)
    return _Seek(name, 0, lambda index: Zero(), None, group, solo, schema, 2, on_accept)


def _tactic_shape(
    schema: Schema,
    group: Shelf,
    solo: Shelf,
    on_accept: Callable[[object], None] | None = None,
) -> _Seek:
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
    return _Seek(name, count, decode, None, group, solo, schema, 2, on_accept)


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


def _save(session: Session, group: Shelf, solo: Shelf, run: _Live | None = None) -> None:
    if run is not None:
        capture(session, run.nets, run.tape, run.solo_tape, run.clocks, run.solo_clocks)
    session.write()
    _snapshot(session, group, solo)


def _metric(plan: Plan, session: Session, group: Shelf, solo: Shelf, done: int) -> None:
    write_package_metric(plan, session, group, solo, done, Scores(), Tape())


class Scores:
    """Ostatnie trafienia grupy, solo i wag zamrożonych."""

    def __init__(self) -> None:
        self.group: list[bool] = []
        self.solo: list[bool] = []
        self.frozen: list[bool] = []

    def add(self, hit: bool, solo_hit: bool, frozen_hit: bool) -> None:
        for row, flag in (
            (self.group, hit),
            (self.solo, solo_hit),
            (self.frozen, frozen_hit),
        ):
            row.append(bool(flag))
            del row[:-50]


@dataclass
class _Live:
    plan: Plan
    session: Session
    nets: tuple
    tape: Tape
    solo_tape: Tape
    clocks: Clocks
    solo_clocks: Clocks
    scores: Scores
    ramp: Ramp


def _boot(plan: Plan, session: Session) -> _Live:
    live = _Live(
        plan,
        session,
        attach(session),
        Tape(),
        Tape(),
        Clocks(),
        Clocks(),
        Scores(),
        Ramp(),
    )
    live.tape.ingest(session.tape)
    live.solo_tape.ingest(session.solo_tape)
    live.clocks.ingest(session.clocks_state)
    live.solo_clocks.ingest(session.solo_clocks_state)
    live.ramp.level = session.level
    return live


def _speak(book: ExerciseBook, group: Shelf, solo: Shelf, task: Task, round_no: int, run: _Live) -> None:
    del book, solo
    agent, role = speaker(round_no)
    epsilon = run.clocks.author(task.level) if role == "author" else run.clocks.executor(task.level)
    solo_epsilon = run.solo_clocks.author(task.level)
    message = utter(
        task,
        group,
        seed=run.plan.seed,
        round_no=round_no,
        attempt=0,
        epsilon=epsilon,
        net=run.nets[agent] if epsilon < 1 else None,
    )
    solo_message = utter(
        task,
        group,
        seed=run.plan.seed,
        round_no=round_no,
        attempt=1,
        epsilon=solo_epsilon,
        net=run.nets[5] if solo_epsilon < 1 else None,
    )
    frozen_message = utter(
        task,
        group,
        seed=run.plan.seed,
        round_no=round_no,
        attempt=2,
        epsilon=1.0,
        net=None,
    )
    result = settle_round(
        group,
        task,
        run.nets,
        round_no=round_no,
        message=message,
        solo_message=solo_message,
        frozen_message=frozen_message,
        tape=run.tape,
        solo_tape=run.solo_tape,
        clocks=run.clocks,
    )
    run.solo_clocks.tick_author(task.level)
    run.scores.add(result.hit, result.solo_hit, result.frozen_hit)
    run.ramp.note(result.hit, group)
    run.session.level = run.ramp.level


def settle_round(
    shelf: Shelf,
    task: Task,
    nets: Sequence[object],
    *,
    round_no: int,
    message: str,
    solo_message: str,
    frozen_message: str,
    tape: Tape,
    solo_tape: Tape,
    clocks: Clocks,
):
    """Jedna próba. Krok idzie tylko po trafieniu i nigdy na ziarnie 4."""
    result = play_exercise(
        shelf,
        task,
        round_no=round_no,
        message=message,
        tape=tape,
        clocks=clocks,
        solo_tape=solo_tape,
        solo_message=solo_message,
        frozen_message=frozen_message,
    )
    if result.hit:
        commit_group(nets, tape, task.level)
    if result.solo_hit and len(nets) > 5:
        step(nets[5], solo_tape, agent=0, level=task.level)
    return result


def commit_group(nets: Sequence[object], tape: Tape, level: int) -> bool:
    """Cztery sieci grupy. Pusta taśma zostawia wagi."""
    moved = False
    for agent, net in enumerate(tuple(nets)[:4]):
        moved = step(net, tape, agent=agent, level=level) or moved
    return moved


def note_schema(nets: Sequence[object], tape: Tape, found: object) -> None:
    """Term albo fakt schematu wchodzi na taśmę grupy. Produkcja tu nie mówi."""
    body, text = _schema_text(found)
    if not text:
        return
    try:
        prefix = agent_observation(((0, 0),) * 5, body=body)
        trace = place_message(prefix, encode_message(text))
    except (ParseError, ValueError):
        return
    for agent in range(4):
        tape.note(agent, 0, trace)
    commit_group(nets, tape, 0)


def capture(
    session: Session,
    nets: Sequence[object],
    tape: Tape,
    solo_tape: Tape,
    clocks: Clocks,
    solo_clocks: Clocks,
) -> None:
    """Żywe wagi i optimizery, zanim zapis zejdzie na dysk."""
    session.weights = {
        str(net.seed): {key: value.detach().clone() for key, value in net.state_dict().items()}
        for net in nets
    }
    session.optimizers = {}
    for net in nets:
        opt = getattr(net, "optimizer", None)
        if isinstance(opt, torch.optim.AdamW):
            session.optimizers[str(net.seed)] = opt.state_dict()
    session.tape = tape.export()
    session.solo_tape = solo_tape.export()
    session.clocks_state = clocks.export()
    session.solo_clocks_state = solo_clocks.export()


def attach(session: Session):
    """Sześć sieci. Pusty optimizer zapisu dostaje nowego AdamW."""
    nets = build_nets()
    saved = session.weights if isinstance(session.weights, dict) else {}
    optimizers = session.optimizers if isinstance(session.optimizers, dict) else {}
    for net in nets:
        weights = saved.get(str(net.seed))
        if isinstance(weights, dict) and weights:
            net.load_state_dict(weights)
        if any(param.requires_grad for param in net.parameters()):
            opt = torch.optim.AdamW(net.parameters(), lr=LR)
            state = optimizers.get(str(net.seed))
            if isinstance(state, dict) and state:
                opt.load_state_dict(state)
            net.optimizer = opt
    return nets


def write_package_metric(
    plan: Plan,
    session: Session,
    group: Shelf,
    solo: Shelf,
    done: int,
    scores: Scores,
    tape: Tape,
    source: str = "paczka",
) -> bool:
    return append_row(
        plan.directory / "metrics.csv",
        count=done,
        source=source,
        schema=session.schema,
        exhausted=False,
        level=session.level,
        group=group,
        solo=solo,
        marks=Milestones(),
        group_hits=scores.group,
        solo_hits=scores.solo,
        frozen_hits=scores.frozen,
        local_group=(),
        local_solo=(),
        traces=tape.count(),
        bench=(),
        gap="",
        every=plan.metric_every,
    )


def shelf_from_snapshot(path: Path) -> Shelf:
    import json

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    shelf = Shelf()
    for card in document["polka"]:
        domain = card["dziedzina"]
        arity = len(domain[0]) if domain else 1
        body = _parse_body(str(card["ciało"]), arity)
        shelf.write(Def("f0", arity, body))
    for raw in document.get("wspolna") or []:
        fact = _fact_from_document(shelf, raw)
        if fact is not None:
            shelf.hang(fact)
    return shelf


def _fact_from_document(shelf: Shelf, raw: object) -> Fact | None:
    """Snapshot trzyma strony i rodzaj. Taktykę odtwarza kernel."""
    if not isinstance(raw, dict):
        return None
    left_src = raw.get("lewa")
    right_src = raw.get("prawa")
    kind = raw.get("rodzaj")
    if not isinstance(left_src, str) or not isinstance(right_src, str) or kind not in ("E", "W", "C", "I"):
        return None
    try:
        left = _parse_body(left_src, 2)
        right = _parse_body(right_src, 2)
    except (ParseError, ValueError, TypeError):
        return None
    if left == right:
        return None
    kernel = Kernel(shelf)
    if kind == "W":
        for card in shelf.entries():
            if kernel.rewrite(left, right, card.id).accepted:
                return Fact(left, right, "W", (Rewrite(card.id),))
        return None
    if kind == "E":
        for lo in range(10):
            for hi in range(lo + 1, 11):
                if kernel.eval(left, right, lo, hi):
                    return Fact(left, right, "E", (DomEval(lo, hi),))
        return None
    if kind == "C" and kernel.eval(left, right, 0, 1):
        return Fact(left, right, "C", (Eval(), Eval()))
    if kind == "I":
        for card in shelf.entries():
            if induction_ok(shelf, left, right, card.id):
                return Fact(left, right, "I", (Base(), Induct(card.id)))
    return None


def _parse_body(source: str, arity: int) -> Expr:
    found = list(dict.fromkeys(_SHELF_ID.findall(source)))
    if not found:
        return parse_expr(source, arity=arity, allow_self=True)
    free = [slot for slot in SLOTS if slot not in source]
    if len(free) < len(found):
        raise ValueError("slots")
    renamed = source
    mapping: dict[str, str] = {}
    for ident, slot in zip(found, free):
        renamed = re.sub(rf"\b{ident}\b", slot, renamed)
        mapping[slot] = ident
    return _rename_slots(parse_expr(renamed, arity=arity, allow_self=True), mapping)


def _rename_slots(expr: Expr, mapping: dict[str, str]) -> Expr:
    if isinstance(expr, Call):
        slot = mapping.get(expr.slot, expr.slot)
        return Call(slot, tuple(_rename_slots(arg, mapping) for arg in expr.args))
    if isinstance(expr, Succ):
        return Succ(_rename_slots(expr.arg, mapping))
    if isinstance(expr, Pred):
        return Pred(_rename_slots(expr.arg, mapping))
    if isinstance(expr, If0):
        return If0(
            _rename_slots(expr.cond, mapping),
            _rename_slots(expr.zero_branch, mapping),
            _rename_slots(expr.nonzero_branch, mapping),
        )
    if isinstance(expr, Self):
        return Self(tuple(_rename_slots(arg, mapping) for arg in expr.args))
    if isinstance(expr, (Zero, Var, Num)):
        return expr
    raise TypeError(type(expr))


def _schema_text(found: object) -> tuple[Expr | None, str]:
    if isinstance(found, Expr):
        return found, format_expr(found)
    left = getattr(found, "left", None)
    right = getattr(found, "right", None)
    if not isinstance(left, Expr):
        return None, ""
    text = format_expr(left)
    if isinstance(right, Expr):
        text = f"(eq {text} {format_expr(right)})"
    return left, text


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
