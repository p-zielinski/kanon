import csv
import json
from pathlib import Path

from kanon.language import Call, Def, If0, Pred, Self, Var, Zero, parse_def
from kanon.v2.record import (
    agent_observation,
    append_event,
    append_row,
    write_snapshot,
)
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.survey import survey
from kanon.v2.tactics import Eval, Fact
from kanon.v2.vocab import VOCAB

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_SUCC = "(def f0 1 (succ (var 0)))"


def test_every_tenth_row_records_the_workshop_and_not_the_survey_size(tmp_path: Path):
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=16)
    schema = Schema()
    schema.shape = "R1"
    schema.index = 7
    marks = _marks()
    for package in (1, 2, 3):
        marks.observe(group, solo, package=package, origin="schemat")
    assert marks.addition.package is None
    assert len(group.entries()) == 4
    path = tmp_path / "metrics.csv"
    for count in range(1, 10):
        assert append_row(path, count=count, source="paczka", schema=schema, exhausted=False, level=0, group=group, solo=solo, marks=marks, group_hits=(), solo_hits=(), frozen_hits=(), local_group=(), local_solo=(), traces=0, bench=(), gap=5) is False
    assert not path.exists()
    group_hits = (False,) * 10 + (True,) * 50
    assert append_row(
        path,
        count=10,
        source="paczka",
        schema=schema,
        exhausted=True,
        level=1,
        group=group,
        solo=solo,
        marks=marks,
        group_hits=group_hits,
        solo_hits=(True,) * 25 + (False,) * 25,
        frozen_hits=(False,) * 50,
        local_group=("a",),
        local_solo=("b", "c"),
        traces=3,
        bench=("e0", "e3"),
        gap=5,
    )
    assert append_row(
        path,
        count=10,
        source="runda",
        schema=schema,
        exhausted=False,
        level=1,
        group=group,
        solo=solo,
        marks=marks,
        group_hits=group_hits,
        solo_hits=(True,) * 25 + (False,) * 25,
        frozen_hits=(False,) * 50,
        local_group=("a",),
        local_solo=("b", "c"),
        traces=3,
        bench=(),
        gap=5,
    )
    assert append_row(path, count=11, source="runda", schema=schema, exhausted=False, level=1, group=group, solo=solo, marks=marks, group_hits=(), solo_hits=(), frozen_hits=(), local_group=(), local_solo=(), traces=0, bench=(), gap=5) is False
    _enter(group, solo, marks)
    schema.floor = 9
    assert append_row(
        path,
        count=20,
        source="runda",
        schema=schema,
        exhausted=False,
        level=4,
        group=group,
        solo=solo,
        marks=marks,
        group_hits=group_hits,
        solo_hits=(True,) * 25 + (False,) * 25,
        frozen_hits=(False,) * 50,
        local_group=("a",),
        local_solo=("b", "c"),
        traces=3,
        bench=(),
        gap=3,
    )
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert [row["zrodlo"] for row in rows] == ["paczka", "runda", "runda"]
    assert [row["numer"] for row in rows] == ["10", "10", "20"]
    first, _, last = rows
    assert first["frontera"] == "2"
    assert first["ksztalt"] == "R1"
    assert first["term"] == "7"
    assert first["wyczerpany"] == "tak"
    assert first["poziom"] == "1"
    assert first["grupa"] == "1.000000"
    assert first["solo"] == "0.500000"
    assert first["zamrozone"] == "0.000000"
    assert first["polka"] == "4"
    assert first["wspolna"] == "0"
    assert first["lokalna_grupa"] == "1"
    assert first["lokalna_solo"] == "2"
    assert first["slady"] == "3"
    assert first["lawa"] == "e0 e3"
    assert first["dodawanie"] == ""
    assert first["dodawanie_z"] == ""
    assert first["luka"] == ""
    assert last["dodawanie"] == "12"
    assert last["dodawanie_z"] == "schemat"
    assert last["mnozenie"] == "14"
    assert last["mnozenie_z"] == "mutacja"
    assert last["fakt_e"] == "16"
    assert last["fakt_e_z"] == "schemat"
    assert last["luka"] == "3"
    assert last["poziom"] == "4"
    assert last["frontera"] == "9"


def test_snapshot_keeps_the_body_out_of_the_agent_view(tmp_path: Path):
    group, solo = Shelf(), Shelf()
    ident = group.write(parse_def(_SUCC))
    solo.write(parse_def(_SUCC))
    group.bind((ident,))
    caller = group.write(parse_def("(def f0 1 (succ (call f0 (var 0))))"))
    shared = Fact(Var(0), Zero(), "E", (Eval(),))
    local = Fact(Var(0), parse_def(_SUCC).body, "C", (Eval(),))
    group.hang(shared)
    solo.hang(shared)
    path = write_snapshot(
        tmp_path,
        group=group,
        solo=solo,
        local_group=(local,),
        local_solo=(),
        term=12,
        level=2,
    )
    assert path == tmp_path / "canon" / "snapshot.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["term"] == 12
    assert data["poziom"] == 2
    body = format_body(group, ident)
    assert body in path.read_text(encoding="utf-8")
    card = next(item for item in data["polka"] if item["id"] == ident)
    assert card["nazwa"] == "f0"
    assert card["rodzina"] == "następnik"
    assert card["ciało"] == body
    assert card["powołania"] == []
    assert card["dziedzina"][0] == [0]
    assert card["dziedzina"][-1] == [10]
    assert [11] not in card["dziedzina"]
    other = next(item for item in data["polka"] if item["id"] == caller)
    assert other["nazwa"] is None
    assert other["powołania"] == [ident]
    assert data["wspolna"][0]["rodzaj"] == "E"
    assert data["lokalna"]["grupa"][0]["rodzaj"] == "C"
    assert data["lokalna"]["solo"] == []
    pairs = ((0,), (1,), (2,), (3,), (4,))
    quiet = [VOCAB[token] for token in agent_observation(pairs)]
    shown = [VOCAB[token] for token in agent_observation(pairs, body=group.entry(ident).body)]
    assert "succ" not in quiet
    assert body not in " ".join(quiet)
    assert "succ" in shown[:48]
    assert "succ" not in shown[48:]
    assert len(shown) == 176


def test_third_rotation_drops_the_oldest_and_keeps_metrics(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("numer,luka\n10,\n20,3\n", encoding="utf-8")
    saved = metrics.read_bytes()
    names = ("najstarszy", "drugi", "trzeci", "najnowszy")
    rotated = [
        append_event(tmp_path, {"name": name, "pad": "x" * 2000}, limit=1024) for name in names
    ]
    assert rotated == [False, True, True, True]
    files = sorted(path.name for path in tmp_path.iterdir() if path.name.startswith("events"))
    assert files == ["events.1.jsonl", "events.jsonl"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir() if path.suffix == ".jsonl")
    assert "najstarszy" not in text
    assert "drugi" not in text
    assert "trzeci" in text
    assert "najnowszy" in text
    assert metrics.read_bytes() == saved
    assert metrics.read_text(encoding="utf-8").count("\n") == 3


def _marks():
    from kanon.v2.record import Milestones

    return Milestones()


def _enter(group: Shelf, solo: Shelf, marks) -> None:
    for shelf in (group, solo):
        shelf.write(parse_def(_ADD))
    marks.observe(group, solo, package=12, origin="schemat")
    for shelf in (group, solo):
        add = shelf.local_id("dodawanie")
        assert add is not None
        term = If0(Var(0), Zero(), Call(add, (Var(1), Self((Pred(Var(0)), Var(1))))))
        shelf.write(Def("f0", 2, term))
    marks.observe(group, solo, package=14, origin="mutacja")
    group.hang(Fact(Var(0), Zero(), "E", ()))
    marks.observe(group, solo, package=16, origin="schemat")
    group.hang(Fact(Zero(), Var(0), "E", ()))
    marks.observe(group, solo, package=18, origin="mutacja")


def format_body(shelf: Shelf, ident: str) -> str:
    from kanon.roles import format_expr

    return format_expr(shelf.entry(ident).body)
