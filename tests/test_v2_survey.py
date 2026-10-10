import time

from kanon.language import parse_def
from kanon.v2.shelf import FrozenShelf, Shelf
from kanon.v2.survey import survey, week_ready

_FAMILIES = ("identyczność", "następnik", "poprzednik", "stała 0", "stała 1", "stała 2")
_CONSTANTS = ("stała 0", "stała 1", "stała 2")


def test_ceiling_8_finds_identity_and_a_constant():
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=8)
    assert _body(group, "identyczność") == parse_def("(def f0 1 (var 0))").body
    assert any(group.local_id(name) is not None for name in _CONSTANTS)
    assert _placed(group) == _placed(solo)
    frozen = FrozenShelf()
    assert not hasattr(frozen, "write")


def test_ceiling_16_stops_at_four_families_and_repeats():
    started = time.monotonic()
    first_group, first_solo = Shelf(), Shelf()
    survey(first_group, first_solo, limit=16)
    second_group, second_solo = Shelf(), Shelf()
    survey(second_group, second_solo, limit=16)
    assert time.monotonic() - started < 60
    assert len(_placed(first_group)) == 4
    assert week_ready(first_group)
    assert _placed(first_group) == _placed(first_solo)
    assert _placed(first_group) == _placed(second_group)
    assert _placed(second_group) == _placed(second_solo)


def test_ceiling_1_is_not_ready_for_the_week():
    group, solo = Shelf(), Shelf()
    survey(group, solo, limit=1)
    assert _placed(group) == ()
    assert _placed(solo) == ()
    assert not week_ready(group)
    assert not week_ready(solo)


def _placed(shelf: Shelf) -> tuple[tuple[str, object], ...]:
    rows = []
    for name in _FAMILIES:
        ident = shelf.local_id(name)
        if ident is None:
            continue
        rows.append((name, shelf.entry(ident).body))
    return tuple(rows)


def _body(shelf: Shelf, family: str):
    ident = shelf.local_id(family)
    assert ident is not None
    return shelf.entry(ident).body
