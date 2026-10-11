import csv
from pathlib import Path

import torch

from kanon.language import Call, Zero, parse_def, parse_expr
from kanon.roles import format_expr
from kanon.v2.command import (
    shelf_from_snapshot,
    Scores,
    attach,
    capture,
    commit_group,
    configure,
    note_schema,
    prepare,
    settle_round,
    write_package_metric,
)
from kanon.v2.learn import produce
from kanon.v2.model import build_net, build_nets
from kanon.v2.play import Clocks, Tape, play_exercise
from kanon.v2.record import write_snapshot
from kanon.v2.resume import Session
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import DomEval, Fact
from kanon.v2.voice import utter
from kanon.v2.window import OBS_LEN
from kanon.v2.workshop import Task, draw_task

_IDENTITY = "(def f0 1 (var 0))"
_SUCC = "(def f0 1 (succ (var 0)))"
_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_MISS = "(succ (succ (succ (succ (succ (var 0))))))"


def test_a_hit_moves_the_group_and_a_miss_or_empty_tape_does_not():
    shelf = _pair()
    nets = build_nets()
    fresh = _rows(build_net(0))
    frozen = _rows(nets[4])
    task, message = _until_hit(shelf)
    tape = Tape()
    settle_round(
        shelf,
        task,
        nets,
        round_no=task.round_no,
        message=message,
        solo_message=_MISS,
        frozen_message=_MISS,
        tape=tape,
        solo_tape=Tape(),
        clocks=Clocks(),
    )
    assert not _same(nets[0], fresh)
    assert _same(nets[4], frozen)
    assert tape.traces(0, task.level)

    missed = build_nets()
    before = _rows(missed[0])
    settle_round(
        shelf,
        task,
        missed,
        round_no=task.round_no,
        message=_MISS,
        solo_message=_MISS,
        frozen_message=_MISS,
        tape=Tape(),
        solo_tape=Tape(),
        clocks=Clocks(),
    )
    assert _same(missed[0], before)
    held = build_net(0)
    image = _rows(held)
    assert commit_group((held,), Tape(), 0) is False
    assert _same(held, image)


def test_two_rounds_differ_and_do_not_copy_the_hidden_term():
    shelf = _pair()
    bomb = _Bomb()
    seen: list[str] = []
    for round_no in range(40):
        task = _task(shelf, round_no)
        message = utter(
            task,
            shelf,
            seed=0,
            round_no=round_no,
            attempt=0,
            epsilon=1.0,
            net=bomb,
        )
        assert message != format_expr(task.term)
        if message not in seen:
            seen.append(message)
        if len(seen) >= 2:
            break
    assert len(seen) >= 2
    assert bomb.calls == 0


def test_two_hundred_level_zero_rounds_include_one_accepted_reply():
    shelf = _pair()
    hits = 0
    for round_no in range(200):
        task = _task(shelf, round_no)
        message = utter(
            task,
            shelf,
            seed=0,
            round_no=round_no,
            attempt=0,
            epsilon=1.0,
            net=None,
        )
        result = play_exercise(
            shelf,
            task,
            round_no=round_no,
            message=message,
            tape=Tape(),
            clocks=Clocks(),
        )
        hits += int(result.hit)
    assert hits >= 1


def test_level_four_sampler_hits_without_reading_the_tactic():
    shelf = Shelf()
    zero = shelf.write(parse_def("(def f0 1 (zero))"))
    shelf.write(parse_def(_IDENTITY))
    shelf.hang(Fact(Call(zero, (Zero(),)), Zero(), "W", ()))
    bomb = _Bomb()
    seen: list[str] = []
    hits = 0
    for round_no in range(80):
        task = draw_task(shelf, level=4, round_no=round_no, seed=0)
        assert isinstance(task, Task)
        text = serialize_obs(task)
        assert "rewrite" not in text
        message = utter(
            task,
            shelf,
            seed=0,
            round_no=round_no,
            attempt=0,
            epsilon=1.0,
            net=bomb,
        )
        if message not in seen:
            seen.append(message)
        result = play_exercise(
            shelf,
            task,
            round_no=round_no,
            message=message,
            tape=Tape(),
            clocks=Clocks(),
        )
        hits += int(result.hit)
    assert len(seen) >= 2
    assert hits >= 1
    assert bomb.calls == 0


def serialize_obs(task: Task) -> str:
    from kanon.v2.workshop import serialize

    return serialize(task.observation)


def test_restored_fact_opens_a_level_four_exercise(tmp_path: Path):
    shelf = Shelf()
    shelf.write(parse_def(_ADD))
    shelf.write(parse_def(_IDENTITY))
    ident = shelf.entries()[0].id
    shelf.hang(Fact(Zero(), Call(ident, (Zero(), Zero())), "E", (DomEval(0, 1),)))
    write_snapshot(
        tmp_path,
        group=shelf,
        solo=shelf,
        local_group=(),
        local_solo=(),
        term=0,
        level=4,
    )
    restored = shelf_from_snapshot(tmp_path / "canon" / "snapshot.json")
    assert restored.hung()
    task = draw_task(restored, level=4, round_no=0, seed=0)
    assert isinstance(task, Task)
    assert task.fact is not None
    assert task.fact.kind == "E"


def test_low_epsilon_starts_from_the_observation_prefix():
    shelf = _pair()
    task = _task(shelf, 0)
    net = build_net(0)
    lengths: list[int] = []
    real = net.forward_prefix

    def spy(token_ids):
        lengths.append(int(token_ids.shape[1]))
        return real(token_ids)

    net.forward_prefix = spy
    message = utter(task, shelf, seed=0, round_no=0, attempt=0, epsilon=0.0, net=net)
    parse_expr(message, arity=2, allow_num=True)
    assert lengths
    assert lengths[0] == OBS_LEN


def test_schema_addition_steps_without_the_production_forward(monkeypatch):
    expr = parse_def(_ADD).body
    nets = build_nets()
    fresh = _rows(nets[0])
    tape = Tape()
    shapes: list[tuple[int, ...]] = []
    real = nets[0].forward_prefix

    def spy(token_ids):
        shapes.append(tuple(token_ids.shape))
        return real(token_ids)

    nets[0].forward_prefix = spy

    def refuse(*_args, **_kwargs):
        raise AssertionError("produkcja")

    monkeypatch.setattr("kanon.v2.voice.utter", refuse)
    monkeypatch.setattr("kanon.v2.learn.produce", refuse)
    note_schema(nets[:4], tape, expr)
    assert tape.traces(0, 0)
    assert not _same(nets[0], fresh)
    assert shapes
    assert all(shape[0] == 4 for shape in shapes)
    assert produce.__name__ == "produce"


def test_restore_keeps_the_frontier_and_skips_the_survey(tmp_path: Path, monkeypatch):
    shelf = Shelf()
    shelf.write(parse_def(_ADD))
    session = Session(tmp_path)
    session.schema.floor = 3
    session.schema.shape = "R1"
    session.schema.index = 100
    session.write()
    write_snapshot(
        tmp_path,
        group=shelf,
        solo=shelf,
        local_group=(),
        local_solo=(),
        term=100,
        level=0,
    )

    def boom(*_args, **_kwargs):
        raise AssertionError("survey")

    monkeypatch.setattr("kanon.v2.command.survey", boom)
    plan = configure(["--seed", "0", "--directory", str(tmp_path)])
    restored, group, solo, resumed = prepare(plan)
    assert resumed is True
    assert restored.schema.state() == (3, "R1", 100)
    assert group.local_id("dodawanie") is not None
    assert solo.local_id("dodawanie") is not None


def test_a_saved_hit_reopens_away_from_the_fresh_seed(tmp_path: Path):
    shelf = _pair()
    nets = build_nets()
    task, message = _until_hit(shelf)
    tape = Tape()
    solo_tape = Tape()
    clocks = Clocks()
    solo_clocks = Clocks()
    settle_round(
        shelf,
        task,
        nets,
        round_no=task.round_no,
        message=message,
        solo_message=_MISS,
        frozen_message=_MISS,
        tape=tape,
        solo_tape=solo_tape,
        clocks=clocks,
    )
    session = Session(tmp_path)
    capture(session, nets, tape, solo_tape, clocks, solo_clocks)
    session.write()
    loaded = Session.open(tmp_path)
    fresh = build_net(0).state_dict()
    saved = loaded.weights["0"]
    assert any(not torch.equal(saved[key], fresh[key]) for key in fresh)
    assert loaded.optimizers["0"]
    rebound = attach(loaded)
    assert isinstance(rebound[0].optimizer, torch.optim.AdamW)
    empty = Session(tmp_path)
    empty.optimizers = {}
    empty.weights = {}
    assert isinstance(attach(empty)[0].optimizer, torch.optim.AdamW)


def test_metrics_record_traces_and_hits(tmp_path: Path):
    plan = configure(
        ["--seed", "0", "--directory", str(tmp_path), "--metric-every", "1"]
    )
    session = Session(tmp_path)
    scores = Scores()
    scores.group.append(True)
    scores.solo.append(False)
    scores.frozen.append(False)
    tape = Tape()
    tape.note(0, 0, (1, 2, 3))
    assert write_package_metric(plan, session, Shelf(), Shelf(), 1, scores, tape) is True
    rows = list(csv.reader((tmp_path / "metrics.csv").open(encoding="utf-8")))
    assert rows[1][7] != ""
    assert rows[1][14] == "1"


def test_the_command_does_not_hardcode_abstain():
    text = Path("src/kanon/v2/command.py").read_text(encoding="utf-8")
    assert '"(abstain f0)"' not in text
    assert "'(abstain f0)'" not in text


class _Bomb:
    def __init__(self) -> None:
        self.calls = 0

    def forward_prefix(self, token_ids):
        self.calls += 1
        raise AssertionError("prefix")


def _pair() -> Shelf:
    shelf = Shelf()
    shelf.write(parse_def(_IDENTITY))
    shelf.write(parse_def(_SUCC))
    return shelf


def _task(shelf: Shelf, round_no: int) -> Task:
    task = draw_task(shelf, level=0, round_no=round_no, seed=0)
    assert isinstance(task, Task)
    return task


def _until_hit(shelf: Shelf) -> tuple[Task, str]:
    for round_no in range(200):
        task = _task(shelf, round_no)
        message = utter(
            task,
            shelf,
            seed=0,
            round_no=round_no,
            attempt=0,
            epsilon=1.0,
            net=None,
        )
        result = play_exercise(
            shelf,
            task,
            round_no=round_no,
            message=message,
            tape=Tape(),
            clocks=Clocks(),
        )
        if result.hit:
            return task, message
    raise AssertionError("hit")


def _rows(net):
    return tuple(param.detach().clone() for param in net.parameters())


def _same(net, rows) -> bool:
    return all(
        torch.equal(param.detach(), saved)
        for param, saved in zip(net.parameters(), rows, strict=True)
    )
