import pytest

from kanon.language import Call, If0, Num, Pred, Succ, Var, Zero, parse_def
from kanon.v2.kernel import Kernel
from kanon.v2.shelf import Shelf

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"


def test_eval_fact_holds_on_0_through_2_and_fails_when_3_is_included():
    left = If0(Pred(Pred(Var(0))), Var(0), Zero())
    right = Var(0)
    kernel = Kernel(Shelf())
    assert kernel.eval(left, right, 0, 2) is True
    assert kernel.eval(left, right, 0, 3) is False


def test_rewrite_inside_the_domain_is_w_and_outside_is_rejected():
    shelf = Shelf()
    add = shelf.write(parse_def(_ADD))
    kernel = Kernel(shelf)
    left = Call(add, (Var(0), Var(1)))
    right = If0(Var(0), Var(1), Succ(Call(add, (Pred(Var(0)), Var(1)))))
    verdict = kernel.rewrite(left, right, add)
    assert verdict.accepted is True
    assert verdict.kind == "W"
    assert verdict.kind != "E"
    assert kernel.rewrite(left, Var(0), add).accepted is False
    outside = Call(add, (Num(7), Zero()))
    unfolded = If0(Num(7), Zero(), Succ(Call(add, (Pred(Num(7)), Zero()))))
    assert kernel.rewrite(outside, unfolded, add).accepted is False


def test_base_reduces_both_sides_at_zero():
    shelf = Shelf()
    add = shelf.write(parse_def(_ADD))
    kernel = Kernel(shelf)
    left = Call(add, (Var(0), Zero()))
    assert kernel.base(left, Var(0)) is True
    assert kernel.base(left, Succ(Var(0))) is False


def test_step_of_addition_needs_one_ih_after_reduction():
    shelf = Shelf()
    add = shelf.write(parse_def(_ADD))
    assert shelf.local_id("ih") is None
    kernel = Kernel(shelf)
    left = Call(add, (Var(0), Zero()))
    right = Var(0)
    step = kernel.step(left, right)
    assert isinstance(step.rewrite(add), If0)
    assert step.closed() is False
    assert step.reduce() == Succ(Call(add, (Var(0), Zero())))
    assert step.closed() is False
    assert step.ih() == Succ(Var(0))
    assert step.closed() is True
    with pytest.raises(ValueError):
        step.ih()
    assert shelf.local_id("ih") is None


def test_tautology_is_rejected_before_a_tactic():
    kernel = Kernel(Shelf())
    same = Var(0)
    assert kernel.eval(same, same, 0, 2) is False
    assert kernel.base(same, same) is False
    assert kernel.rewrite(same, same, "e0").accepted is False
    with pytest.raises(ValueError):
        kernel.step(same, same)
