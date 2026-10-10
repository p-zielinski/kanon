from kanon.language import Call, Succ, Var, Zero, parse_def, tokenize
from kanon.roles import format_expr
from kanon.v2.batch import run_package
from kanon.v2.budget import message_budget
from kanon.v2.command import ExerciseBook, _Seek, _shape_for
from kanon.v2.schema import Schema, _family
from kanon.v2.shelf import Shelf
from kanon.v2.survey import survey
from kanon.v2.tactics import decode_e, encode_e
from kanon.v2.workshop import vet

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"


def test_a_term_past_the_floor_budget_does_not_reach_the_reducer(monkeypatch):
    expr = _tower(message_budget(2) + 5)
    assert len(tokenize(format_expr(expr))) > message_budget(2)
    calls: list[object] = []

    def spy(body, defs=None, env=None):
        calls.append(env)
        return 0

    assert vet(expr, {}, [(0,)], reduce=spy, floor=2) is None
    assert calls == []
    assert _family(expr, 1, floor=2) is None
    assert calls == []

    def boom(defn, point, defs=None):
        calls.append(point)
        return 0

    monkeypatch.setattr("kanon.v2.shelf.reduce_def", boom)
    assert _family(expr, 1, floor=2) is None
    assert calls == []
    short = Zero()
    assert vet(short, {}, [(0,)], reduce=spy, floor=2) == (0,)
    assert calls == [(0,)]


def test_subtraction_alone_does_not_close_floor_two_in_the_command():
    oracle = parse_def(
        "(def f0 2 (if0 (var 0) (zero) "
        "(if0 (var 1) (var 0) (self (pred (var 0)) (pred (var 1))))))"
    ).body
    schema = Schema()
    group, solo = Shelf(), Shelf()
    seek = _Seek("R2", 3, lambda index: oracle, "odejmowanie", group, solo, schema, 2)
    run_package(
        schema,
        seek,
        save=lambda state: None,
        exercise=lambda: None,
        clock=lambda count: 0.0,
        closed=lambda: False,
        terms=1,
    )
    assert schema.floor == 2
    assert schema.shape == "R2"
    assert group.local_id("dodawanie") is None
    assert group.local_id("odejmowanie") is not None


def test_floor_nine_decodes_an_addition_equation_before_exhaustion():
    group, solo = Shelf(), Shelf()
    slot = group.write(parse_def(_ADD))
    solo.write(parse_def(_ADD))
    left = Call(slot, (Var(0), Zero()))
    right = Var(0)
    index = encode_e(group, left, right, 0, 6)
    assert decode_e(group, index) == (left, right, 0, 6)
    schema = Schema()
    schema.floor = 9
    schema.shape = "E"
    schema.index = index
    shape = _shape_for(schema, group, solo)
    assert shape.count > index
    run_package(
        schema,
        shape,
        save=lambda state: None,
        exercise=lambda: None,
        clock=lambda count: 0.0,
        closed=lambda: False,
        terms=1,
    )
    assert schema.state() == (10, "W", 0)
    assert group.hung() == solo.hung()
    assert group.hung()


def test_the_command_exercise_counts_to_the_family_card():
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=16)
    book = ExerciseBook(0)
    tasks = [book.draw(group, level=0, round_no=round_no) for round_no in range(4)]
    assert book.mutate() is None
    assert all(task.family is None for task in tasks[:3])
    assert tasks[3].family is not None


def _tower(extra: int):
    expr = Zero()
    while len(tokenize(format_expr(expr))) <= extra:
        expr = Succ(expr)
    return expr
