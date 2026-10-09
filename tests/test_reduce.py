import pytest

from kanon.language import Succ, Zero, parse_data, parse_def, parse_expr
from kanon.reduce import NODE_LIMIT, ReduceError, reduce_def, reduce_expr

ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"


def _chain(n_succ: int):
    expr = Zero()
    for _ in range(n_succ):
        expr = Succ(expr)
    return expr


def test_succ_succ_zero_is_two():
    assert reduce_expr(parse_expr("(succ (succ (zero)))")) == 2


def test_pred_zero_is_zero():
    assert reduce_expr(parse_expr("(pred (zero))")) == 0
    assert reduce_expr(parse_expr("(pred (succ (zero)))")) == 0
    assert reduce_expr(parse_expr("(succ (pred (zero)))")) == 1


def test_if0_branch_order_matches_addition():
    # (if0 cond zero-branch nonzero-branch).
    # PROMPTS.md podaje dla tych dwóch termów wyniki 0 i 1. Ten porządek
    # bierze gałąź else przy zerze i sprawia, że def dodawania z PLAN.md
    # nie dochodzi do 5. Ten sam prompt wymaga 5, więc zero bierze pierwszą gałąź.
    assert reduce_expr(parse_expr("(if0 (zero) (succ (zero)) (zero))")) == 1
    assert reduce_expr(parse_expr("(if0 (succ (zero)) (succ (zero)) (zero))")) == 0


def test_addition_def_on_num_2_and_num_3_is_five():
    defn = parse_def(ADD)
    assert reduce_def(defn, [parse_data("(num 2)"), parse_data("(num 3)")]) == 5
    assert reduce_def(defn, [parse_data("(num 0)"), parse_data("(num 4)")]) == 4


def test_recursion_that_does_not_reach_zero_ends_with_fuel():
    defn = parse_def("(def f0 1 (self (succ (var 0))))")
    with pytest.raises(ReduceError) as caught:
        reduce_def(defn, [0])
    assert caught.value.kind == "fuel"


def test_fuel_budget_is_256_on_an_external_call():
    defn = parse_def("(def f0 1 (if0 (var 0) (zero) (self (pred (var 0)))))")
    assert reduce_def(defn, [256]) == 0
    with pytest.raises(ReduceError) as caught:
        reduce_def(defn, [257])
    assert caught.value.kind == "fuel"


def test_term_over_4096_nodes_is_memory():
    with pytest.raises(ReduceError) as caught:
        reduce_expr(_chain(NODE_LIMIT))
    assert caught.value.kind == "memory"
    assert reduce_expr(_chain(NODE_LIMIT - 1)) == NODE_LIMIT - 1
    spine = "(succ " * NODE_LIMIT + "(zero)" + ")" * NODE_LIMIT
    with pytest.raises(ReduceError) as caught:
        reduce_expr(parse_expr(spine))
    assert caught.value.kind == "memory"
    with pytest.raises(ReduceError) as caught:
        reduce_expr(parse_data(f"(num {NODE_LIMIT})"))
    assert caught.value.kind == "memory"


def test_call_cycle_ends_with_steps_not_a_hang():
    caller = parse_def("(def f0 0 (call f1))")
    callee = parse_def("(def f1 0 (call f0))")
    with pytest.raises(ReduceError) as caught:
        reduce_def(caller, [], defs={callee.slot: callee})
    assert caught.value.kind == "steps"


def test_if0_does_not_reduce_the_other_branch():
    defn = parse_def("(def f0 1 (if0 (var 0) (zero) (self (var 0))))")
    assert reduce_def(defn, [0]) == 0
