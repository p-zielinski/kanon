from pathlib import Path

from kanon.language import Call, Expr, Succ, Var, Zero, parse_def
from kanon.roles import assign
from kanon.v2.mutate import (
    Gap,
    choose_fact,
    finish_package,
    is_mutation,
    mutate,
    replay,
    shared_facts,
    skeletons,
    solo_task,
)
from kanon.v2.play import Clocks, Tape
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.tactics import Eval, Fact, make_e, thesis_tokens

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_NAMES = ("f0", "f1")


def test_frontier_eleven_packages_an_exercise_and_no_mutation():
    schema = Schema()
    schema.floor = 11
    log: list[str] = []
    finish_package(
        schema,
        _Shape(),
        exercise=lambda: log.append("exercise"),
        mutate=lambda: log.append("mutation"),
    )
    assert log == ["exercise"]
    assert is_mutation(9, 11) is False
    assert is_mutation(9, 12) is False
    assert [is_mutation(n, 13) for n in range(10)] == [False] * 9 + [True]
    assert [is_mutation(n, 13) for n in (0, 9, 10, 19, 20, 29, 30)] == [
        False,
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_sixty_four_mutations_repeat_without_an_exhausted_bud():
    shelf = _addition()
    fact = _equation(shelf)
    assert fact is not None
    seen: list[Fact] = []

    def judge(candidate: Fact) -> bool:
        seen.append(candidate)
        return False

    book: list[str] = []
    first = mutate(shelf, fact, _NAMES, judge=judge, log=book)
    second = mutate(shelf, fact, _NAMES, judge=judge, log=book)
    assert len(first) == 64
    assert len(second) == 64
    assert len(seen) == 128
    assert skeletons() == ()
    assert "pąk wyczerpany" not in " ".join(book)
    assert "pąk wyczerpany" not in Path(mutate.__code__.co_filename).read_text()
    kinds = {item.kind for item in first}
    assert kinds == {"number", "compose", "tactic"}
    numbers = [item.number for item in first if item.kind == "number"]
    assert numbers
    assert all(isinstance(number, int) and 0 <= number <= 10 for number in numbers)
    assert len(set(numbers)) > 1
    for item in first:
        if item.kind == "compose":
            assert set(_slots(item.fact.left)) <= set(_NAMES)
            assert _slots(item.fact.left)
        if item.kind == "tactic":
            assert len(item.fact.tactics) <= 8
            assert thesis_tokens(item.fact) <= 80
    wide = Fact(fact.left, fact.right, "E", (Eval(),) * 8)
    capped = mutate(shelf, wide, _NAMES, judge=judge)
    assert all(len(item.fact.tactics) <= 8 for item in capped if item.kind == "tactic")


def test_local_hit_stays_off_the_solo_task_and_the_gap():
    shelf = _addition()
    fact = _equation(shelf)
    group = Shelf()
    solo = Shelf()
    group.hang(fact)
    solo.hang(fact)
    before_group = group.fact_pool()
    before_solo = (solo.hung(), solo.fact_pool())
    kept = mutate(shelf, fact, _NAMES, curve="group")
    local = [item.fact for item in kept if item.accepted]
    assert local
    assert all(item.fact not in shared_facts(group, solo) for item in kept if item.accepted)
    assert solo_task(shared_facts(group, solo), local) == (fact,)
    assert all(item not in solo_task(shared_facts(group, solo), local) for item in local)
    gap = Gap()
    for item in local:
        assert gap.admit(item, local=True) is False
        assert item not in gap.facts
    assert choose_fact("solo", (fact,), {"group": tuple(local), "solo": ()}) == fact
    assert group.fact_pool() == before_group
    assert (solo.hung(), solo.fact_pool()) == before_solo
    again = mutate(shelf, fact, _NAMES, curve="group", local=tuple(local))
    assert [item.fact for item in again if item.accepted] == []


def test_replay_gives_the_breaker_four_points_and_ticks_only_then():
    shelf = _addition()
    planted = Fact(Var(0), Zero(), "E", (Eval(),))
    clocks = Clocks()
    tape = Tape()
    mutate(shelf, _equation(shelf), _NAMES, judge=lambda fact: False, clocks=clocks, tape=tape)
    assert clocks.author(0) == 1
    assert clocks.executor(0) == 1
    assert clocks.breaker(0) == 1
    assert tape.traces(assign(19).breaker, 0) == ()

    quiet = replay(
        shelf,
        planted,
        (0, 1, 2, 3),
        round_no=19,
        tape=tape,
        clocks=clocks,
        level=0,
        local=False,
    )
    assert quiet.checked == ()
    assert clocks.executor(0) == 1
    assert clocks.breaker(0) == 1

    result = replay(
        shelf,
        planted,
        (0, 5, 0, 1),
        round_no=19,
        tape=tape,
        clocks=clocks,
        level=0,
        local=True,
    )
    assert result.checked == (0, 5, 0, 1)
    assert result.confirmed == (5, 1)
    breaker = assign(19).breaker
    assert [item[-1] for item in tape.traces(breaker, 0)] == [5, 1]
    assert tape.traces(assign(19).author, 0) == ()
    assert clocks.breaker(0) == 1 * 0.995
    assert clocks.executor(0) == 1 * 0.995
    assert clocks.author(0) == 1
    for _ in range(200):
        replay(
            shelf,
            planted,
            (0, 0, 0, 0),
            round_no=19,
            tape=Tape(),
            clocks=clocks,
            level=0,
            local=True,
        )
    assert clocks.breaker(0) == 0.5


class _Shape:
    name = "R1"
    count = 3

    def decode(self, index: int):
        return Zero()


def _addition() -> Shelf:
    shelf = Shelf()
    shelf.write(parse_def(_ADD))
    return shelf


def _equation(shelf: Shelf) -> Fact:
    slot = shelf.entries()[0].id
    fact = make_e(shelf, Call(slot, (Var(0), Zero())), Var(0), 0, 6)
    assert fact is not None
    return fact


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
