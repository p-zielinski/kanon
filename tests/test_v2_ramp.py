from kanon.language import Zero, parse_def
from kanon.v2.batch import run_package
from kanon.v2.ramp import Ramp
from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.workshop import draw_task

_SUCC = "(def f0 1 (succ (var 0)))"
_ZERO = "(def f0 1 (zero))"


def test_fifty_successes_raise_and_failures_stay_on_the_new_step():
    schema = Schema()
    shelf = Shelf()
    ramp = Ramp()
    for _ in range(49):
        ramp.note(True, shelf, schema)
    assert ramp.level == 0
    assert schema.floor == 2
    assert schema.index == 0
    ramp.note(True, shelf, schema)
    assert ramp.level == 1
    assert ramp.unlocked() == (0, 1)
    assert schema.state()[:2] == (2, "R1")
    ramp.note(False, shelf, schema)
    assert ramp.level == 1
    for _ in range(49):
        ramp.note(False, shelf, schema)
    assert ramp.level == 1
    assert schema.floor == 2

    band = Ramp()
    for _ in range(21):
        band.note(False, shelf)
    for _ in range(29):
        band.note(True, shelf)
    assert band.level == 0

    at_three_tenths = Ramp()
    for _ in range(35):
        at_three_tenths.note(False, shelf)
    for _ in range(15):
        at_three_tenths.note(True, shelf)
    assert at_three_tenths.level == 0

    exact = Ramp()
    for _ in range(20):
        exact.note(False, shelf)
    for _ in range(30):
        exact.note(True, shelf)
    assert exact.level == 1


def test_pool_blocks_four_and_the_ceiling_keeps_drawing():
    shelf = _cards()
    shelf.pool_add((shelf.entries()[0].id,))
    ramp = Ramp()
    for _ in range(200):
        ramp.note(True, shelf)
    assert ramp.level == 3
    shelf.hang(object())
    assert ramp.level == 3
    ramp.note(True, shelf)
    assert ramp.level == 4
    for _ in range(150):
        ramp.note(True, shelf)
    assert ramp.level == 6
    for _ in range(50):
        ramp.note(True, shelf)
    assert ramp.level == 6
    first = draw_task(shelf, level=ramp.level, round_no=0, seed=0)
    second = draw_task(shelf, level=ramp.level, round_no=1, seed=0)
    assert first is not None and second is not None
    assert first.inputs != second.inputs


def test_schema_package_uses_the_lowest_unlocked_level():
    shelf = Shelf()
    ramp = Ramp()
    schema = Schema()
    for _ in range(50):
        ramp.note(True, shelf, schema)
    assert ramp.level == 1
    assert ramp.package_level() == 0
    played: list[int] = []

    def exercise() -> None:
        played.append(ramp.package_level())

    run_package(
        schema,
        _Shape(),
        save=lambda state: None,
        exercise=exercise,
        clock=lambda done: 0.0,
        closed=lambda: True,
    )
    assert played == [0]
    assert ramp.level == 1
    assert schema.floor == 2


class _Shape:
    name = "R1"
    count = 2

    def decode(self, index: int):
        return Zero()


def _cards() -> Shelf:
    shelf = Shelf()
    shelf.write(parse_def(_SUCC))
    shelf.write(parse_def(_ZERO))
    return shelf
