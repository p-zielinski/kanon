from kanon.language import Call, Expr, If0, Num, Pred, Self, Succ, Var, Zero, parse_def
from kanon.reduce import NODE_LIMIT, ReduceError, reduce_expr
from kanon.roles import format_expr
from kanon.v2.seed import round_seed
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import Fact
from kanon.v2.workshop import Pending, Workshop, draw_task, serialize, vet

_SUCC = "(def f0 1 (succ (var 0)))"
_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_ZERO = "(def f0 1 (zero))"
_EXPR = (Call, Succ, Pred, If0, Var, Zero, Num, Self)


def test_level_three_uses_only_the_two_chosen_cards():
    shelf = _shelf()
    third = {card.id for card in shelf.entries()}
    for level, calls in ((0, 1), (1, 2), (2, 3), (3, 4)):
        task = draw_task(shelf, level=level, round_no=0, seed=0)
        slots = _slots(task.term)
        assert len(task.cards) == 2
        assert len(slots) == calls
        assert set(slots) <= set(task.cards)
        assert set(task.cards) < third
        assert set(slots).isdisjoint(third - set(task.cards))
        assert all(0 <= number <= 6 for point in task.inputs for number in point)


def test_generator_rejects_a_starter_before_the_whole_domain():
    domain = tuple((n,) for n in range(7))
    starter = Call("e0", (Var(0),))
    for kind in ("fuel", "steps"):
        seen: list[tuple[int, ...]] = []

        def spy(expr, *, defs, env, kind=kind):
            seen.append(tuple(env))
            raise ReduceError(kind)

        assert vet(starter, {}, domain, reduce=spy) is None
        assert seen == [(0,)]
        seen.clear()

    huge: Expr = starter
    for _ in range(NODE_LIMIT):
        huge = Succ(huge)
    called: list[int] = []

    def quiet(expr, *, defs, env):
        called.append(1)
        return 0

    assert vet(huge, {}, domain, reduce=quiet) is None
    assert called == []

    per: dict[str, list] = {}

    def burning(expr, *, defs, env):
        per.setdefault(format_expr(expr), []).append(tuple(env))
        raise ReduceError("fuel")

    assert draw_task(_shelf(), level=0, round_no=0, seed=0, reduce=burning) is None
    assert per
    assert all(len(points) < 5 for points in per.values())


def test_exercise_counter_ignores_mutation_and_round_twelve():
    shelf = _shelf()
    families = {card.family for card in shelf.entries() if card.family}
    shop = Workshop(seed=0)
    shop.mutate()
    assert shop.count == 0
    tasks = []
    for round_no in (0, 12, 4, 8):
        shop.mutate()
        tasks.append(shop.exercise(shelf, level=0, round_no=round_no))
    assert shop.count == 4
    for task in tasks[:3]:
        text = serialize(task.observation)
        assert task.family is None
        assert "rodzina" not in text
        assert "celu" not in text
        for family in families:
            assert family not in text
    marked = tasks[3]
    assert marked.family is not None
    assert marked.marked in marked.cards
    assert marked.family == shelf.entry(marked.marked).family
    assert marked.family in serialize(marked.observation)
    shop.mutate()
    assert shop.count == 4


def test_two_rounds_differ_and_the_observation_hides_the_term():
    shelf = _shelf()
    logged: list[Expr] = []

    def spy(expr, *, defs, env):
        logged.append(expr)
        return reduce_expr(expr, defs=defs, env=env)

    first = draw_task(shelf, level=0, round_no=0, seed=0, reduce=spy)
    second = draw_task(shelf, level=0, round_no=1, seed=0, reduce=spy)
    assert first.seed == round_seed(0, 0, 0)
    assert second.seed == round_seed(0, 1, 0)
    assert first.inputs != second.inputs
    assert first.term in logged
    assert first.verdicts == tuple(
        reduce_expr(first.term, defs=_defs(shelf), env=point) for point in first.inputs
    )
    observation = first.observation
    assert not _has_expr(observation)
    assert format_expr(first.term) not in serialize(observation)
    assert len(observation.names) == 2


def test_levels_four_through_six_wait_for_the_tactic_pool():
    shelf = _shelf()
    shop = Workshop()
    shelf.pool_add((shelf.entries()[0].id,))
    assert shop.ceiling(shelf) == 3
    assert draw_task(shelf, level=4, round_no=0, seed=0) is None
    assert draw_task(shelf, level=5, round_no=0, seed=0) is None
    assert draw_task(shelf, level=6, round_no=0, seed=0) is None
    assert draw_task(shelf, level=3, round_no=0, seed=0) is not None
    shelf.hang(Fact(Var(0), Var(0), "E", ()))
    assert shop.ceiling(shelf) == 6
    opened = draw_task(shelf, level=4, round_no=0, seed=0)
    assert isinstance(opened, Pending)
    assert opened.level == 4


def _shelf() -> Shelf:
    shelf = Shelf()
    shelf.write(parse_def(_SUCC))
    shelf.write(parse_def(_ADD))
    shelf.write(parse_def(_ZERO))
    return shelf


def _defs(shelf: Shelf) -> dict:
    from kanon.language import Def

    return {card.id: Def(card.id, card.arity, card.body) for card in shelf.entries()}


def _slots(expr: Expr) -> tuple[str, ...]:
    found: list[str] = []
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Call):
            found.append(node.slot)
            stack.extend(node.args)
        elif isinstance(node, Succ):
            stack.append(node.arg)
    return tuple(found)


def _has_expr(value: object) -> bool:
    if isinstance(value, _EXPR):
        return True
    if isinstance(value, tuple):
        return any(_has_expr(item) for item in value)
    if isinstance(value, dict):
        return any(_has_expr(item) for item in value.values())
    fields = getattr(value, "__dataclass_fields__", None)
    if fields is not None:
        return any(_has_expr(getattr(value, name)) for name in fields)
    return False
