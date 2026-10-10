from kanon.language import Call, If0, Pred, Succ, Var, Zero, parse_def
from kanon.roles import format_expr
from kanon.v2.schema import Schema, floor_shape
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import (
    DomEval,
    Eval,
    Fact,
    admit_pool,
    admits_e,
    chain_ok,
    extend_once,
    induction_ok,
    is_skeleton,
    make_e,
    make_i,
    make_w,
    one_insertion,
    settle,
    thesis_tokens,
)

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_SUCC = "(def f0 1 (succ (var 0)))"


def test_floor_shapes_nine_through_twelve():
    assert [floor_shape(n) for n in range(9, 13)] == ["E", "W", "C", "I"]


def test_one_insertion_of_addition_does_not_close_floor_9():
    shelf = Shelf()
    slot = shelf.write(parse_def(_ADD))
    left = Call(slot, (Var(0), Var(1)))
    right = If0(Var(0), Var(1), Succ(Call(slot, (Pred(Var(0)), Var(1)))))
    rendered = format_expr(left) + format_expr(right)
    assert "dodawanie" not in rendered
    assert "następnik" not in rendered
    assert one_insertion(shelf, left, right) is True
    assert admits_e(shelf, left, right, 0, 6) is False
    group, solo = Shelf(), Shelf()
    assert settle(group, solo, [_rejected(left, right)], floor=9, source=shelf) is None
    schema = Schema()
    schema.floor = 9
    schema.shape = "E"
    schema.index = 4
    assert schema.advance_if_closed(group) is False
    assert schema.state() == (9, "E", 4)


def test_successor_call_misses_the_floor_two_filter():
    shelf = Shelf()
    slot = shelf.write(parse_def(_SUCC))
    assert shelf.entry(slot).family == "następnik"
    left = Call(slot, (Zero(),))
    right = Succ(Zero())
    assert "następnik" not in format_expr(left)
    assert one_insertion(shelf, left, right) is True
    assert admits_e(shelf, left, right, 0, 2) is False


def test_addition_equation_is_e_and_the_same_fact_lands_on_both_shelves():
    shelf = Shelf()
    slot = shelf.write(parse_def(_ADD))
    left = Call(slot, (Var(0), Zero()))
    right = Var(0)
    fact = make_e(shelf, left, right, 0, 6)
    assert fact is not None and fact.kind == "E"
    group, solo = Shelf(), Shelf()
    assert settle(group, solo, [fact], floor=9, source=shelf) == fact
    assert group.hung() == solo.hung() == (fact,)
    schema = Schema()
    schema.floor = 9
    schema.shape = "E"
    assert schema.advance_if_closed(group) is True
    assert schema.state() == (10, "W", 0)
    assert settle(group, solo, [fact], floor=9, source=shelf) is None
    assert group.hung() == (fact,)


def test_rewrite_closes_floor_10():
    shelf = Shelf()
    slot = shelf.write(parse_def(_ADD))
    left = Call(slot, (Var(0), Var(1)))
    right = If0(Var(0), Var(1), Succ(Call(slot, (Pred(Var(0)), Var(1)))))
    fact = make_w(shelf, left, right, slot)
    assert fact is not None and fact.kind == "W"
    group, solo = Shelf(), Shelf()
    assert settle(group, solo, [fact], floor=10, source=shelf) == fact
    assert group.hung() == solo.hung() == (fact,)
    schema = Schema()
    schema.floor = 10
    schema.shape = "W"
    assert schema.advance_if_closed(group) is True
    assert schema.state() == (11, "C", 0)


def test_two_tactic_candidates_grow_linearly_and_the_ninth_drops():
    shelf = Shelf()
    slot = shelf.write(parse_def(_ADD))
    left = Call(slot, (Var(0), Zero()))
    facts = []
    for hi in (1, 2, 3):
        fact = make_e(shelf, left, Var(0), 0, hi)
        assert fact is not None
        facts.append(fact)
    one = extend_once(facts[:1], shelf)
    three = extend_once(facts, shelf)
    assert len(one) >= 1
    assert len(three) == 3 * len(one)
    assert {len(fact.tactics) for fact in three} == {2}
    longer = one[0]
    assert chain_ok(shelf, longer) is True
    schema = Schema()
    schema.floor = 11
    schema.shape = "C"
    group, solo = Shelf(), Shelf()
    assert settle(group, solo, [longer], floor=11, source=shelf) == longer
    assert schema.advance_if_closed(group) is True
    assert schema.state() == (12, "I", 0)
    eight = Fact(left, Var(0), "C", (Eval(),) * 8)
    nine = Fact(left, Var(0), "C", (Eval(),) * 9)
    assert chain_ok(shelf, eight) is True
    assert chain_ok(shelf, nine) is False
    assert extend_once([eight], shelf) == ()
    packed = Fact(left, Var(0), "C", (DomEval(0, 1),) * 8)
    assert thesis_tokens(packed) > 80
    assert chain_ok(shelf, packed) is False


def test_induction_on_the_three_skeletons_closes_floor_12():
    shelf = Shelf()
    slot = shelf.write(parse_def(_ADD))
    forms = (
        Call(slot, (Var(0), Zero())),
        Call(slot, (Zero(), Var(0))),
        Call(slot, (Var(0), Succ(Zero()))),
    )
    for form in forms:
        assert is_skeleton(form, shelf) is True
    assert is_skeleton(Call(slot, (Var(0), Var(1))), shelf) is False
    left = forms[0]
    assert induction_ok(shelf, left, Var(0), slot) is True
    fact = make_i(shelf, left, Var(0), slot)
    assert fact is not None and fact.kind == "I"
    group, solo = Shelf(), Shelf()
    assert settle(group, solo, [fact], floor=12, source=shelf) == fact
    assert group.hung() == solo.hung() == (fact,)
    schema = Schema()
    schema.floor = 12
    schema.shape = "I"
    assert schema.advance_if_closed(group) is True
    assert schema.floor == 13


def test_pool_takes_e_before_w_and_drops_the_ninth_extra_and_copies():
    shelf = Shelf()
    slot = shelf.write(parse_def(_ADD))
    succ = shelf.write(parse_def(_SUCC))
    short = make_e(shelf, Call(slot, (Var(0), Zero())), Var(0), 0, 1)
    extras = [
        make_e(shelf, Call(slot, (Var(0), Zero())), Var(0), 0, hi) for hi in range(2, 10)
    ]
    long_e = make_e(
        shelf,
        Call(slot, (Succ(Var(0)), Zero())),
        Succ(Var(0)),
        0,
        6,
    )
    short_w = make_w(shelf, Call(succ, (Var(0),)), Succ(Var(0)), succ)
    assert short is not None and long_e is not None and short_w is not None
    assert all(item is not None for item in extras)
    assert len(extras) == 8
    group, solo = Shelf(), Shelf()
    kept = settle(group, solo, [short, *extras, long_e], floor=9, source=shelf)
    assert kept == short
    assert group.hung() == solo.hung() == (short,)
    assert short not in group.fact_pool()
    assert long_e not in group.fact_pool()
    assert len(group.fact_pool()) == 8
    assert group.fact_pool() == solo.fact_pool()
    assert settle(group, solo, [short, *extras, long_e], floor=9, source=shelf) is None
    assert len(group.hung()) == 1
    assert len(group.fact_pool()) == 8
    other, other_solo = Shelf(), Shelf()
    kept = settle(other, other_solo, [short, long_e, short_w], floor=9, source=shelf)
    assert kept == short
    assert [fact.kind for fact in other.fact_pool()] == ["E", "W"]
    assert other.fact_pool()[0] == long_e
    assert other.fact_pool()[1] == short_w
    huge = Fact(_tower(Var(0), 20), Succ(Var(0)), "E", (DomEval(0, 1),))
    assert thesis_tokens(huge) > 48
    assert admit_pool(other, huge) is False
    assert huge not in other.fact_pool()


def test_empty_e_shape_opens_the_next_floor():
    schema = Schema()
    schema.floor = 9
    schema.shape = "E"
    schema.index = 2
    group, solo = Shelf(), Shelf()
    seen: list = []
    schema.scan(_Empty("E"), lambda _term: False, group, solo, on_decode=seen.append)
    assert seen == []
    assert schema.log == ["schemat wyczerpany"]
    assert schema.state() == (10, "W", 0)
    assert group.hung() == ()
    assert solo.hung() == ()


def _tower(expr, times: int):
    for _ in range(times):
        expr = Succ(expr)
    return expr


def _rejected(left, right):
    return Fact(left, right, "E", (DomEval(0, 6),))


class _Empty:
    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def decode(self, index: int):
        raise AssertionError(index)
