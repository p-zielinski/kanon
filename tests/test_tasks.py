import pytest

from kanon.language import (
    Call,
    Def,
    Expr,
    If0,
    Pred,
    Self,
    Succ,
    Var,
    Zero,
    parse_def,
    tokenize,
)
from kanon.reduce import reduce_def, reduce_expr
from kanon.tasks import (
    Holdout,
    TaskAudit,
    TaskView,
    build_holdout,
    generate_task,
    task_observation,
)

ADD = If0(
    Var(0),
    Var(1),
    Succ(Self((Pred(Var(0)), Var(1)))),
)
MONUS = If0(
    Var(1),
    Var(0),
    If0(Var(0), Zero(), Self((Pred(Var(0)), Pred(Var(1))))),
)


def test_holdout_same_seed_is_identical_and_other_seed_differs():
    first = build_holdout(0)
    second = build_holdout(0)
    other = build_holdout(1)
    assert first == second
    assert _terms(first) == _terms(second)
    assert _terms(first) != _terms(other)
    assert first.seed == 0


def test_holdout_stores_fifty_tasks_on_each_depth():
    holdout = build_holdout(0)
    assert isinstance(holdout, Holdout)
    assert len(holdout.tasks) == 200
    for depth in (1, 2, 3, 4):
        group = holdout.by_depth(depth)
        assert len(group) == 50
        assert {task.view.depth for task in group} == {depth}


def test_task_observation_has_no_hidden_field_and_only_public_pairs():
    holdout = build_holdout(0)
    assert "hidden" not in TaskView.__dataclass_fields__
    for task in holdout.tasks:
        assert isinstance(task, TaskAudit)
        assert isinstance(task.view, TaskView)
        assert not hasattr(task.view, "hidden")
        assert not hasattr(task.view, "target")
        obs = task_observation(task.view)
        assert "hidden" not in obs
        assert "hidden" not in repr(task.view)
        assert obs.count("(num ") == 5 * (task.view.arity + 1)
        assert "succ" not in obs and "var" not in obs and "self" not in obs
        tokenize(obs)


def test_pairs_follow_ranges_and_match_the_target():
    holdout = build_holdout(0)
    assert any(task.view.arity == 1 for task in holdout.tasks)
    assert any(task.view.arity == 2 for task in holdout.tasks)
    for task in holdout.tasks:
        _assert_split(task)
        _assert_outputs(task)


def test_depth_shapes_with_empty_canon():
    holdout = build_holdout(0)
    depth2 = holdout.by_depth(2)
    assert any(task.target == ADD or task.target == MONUS for task in depth2)
    assert any(_is_composition(task.target, task.view.arity) for task in depth2)
    for task in holdout.by_depth(1):
        assert _is_depth1(task.target, task.view.arity)
        assert not _contains(task.target, (Self, Call, If0))
    for task in depth2:
        assert task.target == ADD or task.target == MONUS or _is_composition(
            task.target, task.view.arity
        )
        assert not _contains(task.target, (Call,))
    for depth in (3, 4):
        for task in holdout.by_depth(depth):
            assert not _contains(task.target, (Self, Call, If0))
            assert _is_tower(task.target, task.view.arity, depth)


def test_depth3_with_canon_calls_a_canon_slot_and_matches_it():
    unary = parse_def("(def f0 1 (succ (var 0)))")
    binary = parse_def(
        "(def f1 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
    )
    for defn in (unary, binary):
        task = generate_task(0, 3, canon=(defn,))
        assert isinstance(task.target, Call)
        assert task.target.slot == defn.slot
        assert not _contains(task.target.args[0], (Self, Call))
        _assert_split(task)
        _assert_outputs(task)


def test_task_replays_from_its_seed():
    task = build_holdout(0).tasks[10]
    replay = generate_task(task.seed, task.view.depth, canon=task.canon)
    assert replay == task


def test_depth_outside_v1_is_rejected():
    with pytest.raises(ValueError):
        generate_task(0, 0)
    with pytest.raises(ValueError):
        generate_task(0, 5)


def _terms(holdout: Holdout) -> tuple[Expr, ...]:
    return tuple(task.target for task in holdout.tasks)


def _assert_split(task: TaskAudit) -> None:
    assert task.view.arity in (1, 2)
    public = [pair.inputs for pair in task.view.public_pairs]
    audit = [pair.inputs for pair in task.audit_pairs]
    assert len(public) == 5
    assert len(audit) == 16
    assert len(set(public)) == 5
    assert set(audit).isdisjoint(public)
    for inputs in public:
        assert len(inputs) == task.view.arity
        assert all(0 <= value <= 6 for value in inputs)
    for inputs in audit:
        assert len(inputs) == task.view.arity
        assert all(0 <= value <= 10 for value in inputs)
    if task.view.arity == 1:
        assert len(set(audit)) < 16
    else:
        assert len(set(audit)) == 16


def _assert_outputs(task: TaskAudit) -> None:
    pairs = task.view.public_pairs + task.audit_pairs
    for pair in pairs:
        assert pair.output == _expected(task.target, pair.inputs, task.canon)


def _expected(expr: Expr, inputs: tuple[int, ...], canon: tuple[Def, ...]) -> int:
    if expr == ADD:
        return inputs[0] + inputs[1]
    if expr == MONUS:
        return max(0, inputs[0] - inputs[1])
    if isinstance(expr, Call):
        defs = {item.slot: item for item in canon}
        args = [_interpret(arg, inputs) for arg in expr.args]
        return reduce_def(defs[expr.slot], args)
    return _interpret(expr, inputs)


def _interpret(expr: Expr, inputs: tuple[int, ...]) -> int:
    if isinstance(expr, Zero):
        return 0
    if isinstance(expr, Var):
        return inputs[expr.index]
    if isinstance(expr, Succ):
        return _interpret(expr.arg, inputs) + 1
    if isinstance(expr, Pred):
        return max(0, _interpret(expr.arg, inputs) - 1)
    raise AssertionError(expr)


def _is_depth1(expr: Expr, arity: int) -> bool:
    consts = (Zero(), Succ(Zero()), Succ(Succ(Zero())))
    if expr in consts:
        return True
    if isinstance(expr, Var):
        return expr.index < arity
    if isinstance(expr, Succ) and isinstance(expr.arg, Var):
        return expr.arg.index < arity
    if isinstance(expr, Pred) and isinstance(expr.arg, Var):
        return expr.arg.index < arity
    return False


def _is_composition(expr: Expr, arity: int) -> bool:
    return (
        isinstance(expr, (Succ, Pred))
        and _is_depth1(expr.arg, arity)
        and not _is_depth1(expr, arity)
    )


def _is_tower(expr: Expr, arity: int, depth: int) -> bool:
    if depth == 2:
        return _is_composition(expr, arity)
    if depth < 2 or not isinstance(expr, (Succ, Pred)):
        return False
    return _is_tower(expr.arg, arity, depth - 1)


def _contains(expr: Expr, kinds: tuple[type, ...]) -> bool:
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, kinds):
            return True
        if isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, (Call, Self)):
            stack.extend(node.args)
    return False

