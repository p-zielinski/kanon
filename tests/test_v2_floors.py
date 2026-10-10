from kanon.language import Call, Def, If0, Pred, Self, Succ, Var, Zero, parse_def
from kanon.reduce import reduce_def
from kanon.v2.schema import (
    Schema,
    comparison_shape,
    decode_n1,
    decode_p,
    encode_n1,
    encode_p,
    encode_r1,
    floor_family,
    floor_shape,
    gcd_shape,
)
from kanon.v2.shelf import Shelf

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_SUB = (
    "(def f0 2 (if0 (var 0) (zero) "
    "(if0 (var 1) (var 0) (self (pred (var 0)) (pred (var 1))))))"
)
_PARITY = (
    "(def f0 1 (if0 (var 0) (zero) "
    "(if0 (pred (var 0)) (succ (zero)) (self (pred (pred (var 0)))))))"
)


def test_floors_three_through_eight_are_not_composition():
    assert [floor_shape(n) for n in range(3, 9)] == ["R1", "N1", "P", "R1", "G", "R1"]
    assert [floor_family(n) for n in range(3, 9)] == [
        "mnożenie",
        "porównanie",
        "parzystość",
        "suma",
        "NWD",
        "potęga",
    ]
    shapes = {floor_shape(n) for n in range(3, 9)}
    assert shapes == {"R1", "N1", "P", "G"}
    assert "C1" not in shapes
    assert "C2" not in shapes


def test_exhausting_floor_4_opens_floor_5():
    group, solo = Shelf(), Shelf()
    shape = comparison_shape(group)
    assert shape.count == 0
    schema = Schema()
    schema.floor = 4
    schema.shape = "N1"
    schema.index = 9
    seen: list = []
    schema.scan(shape, lambda _term: False, group, solo, on_decode=seen.append)
    assert seen == []
    assert schema.log == ["schemat wyczerpany"]
    assert schema.state() == (5, "P", 0)
    assert schema.floor != 8
    assert group.local_id("porównanie") is None
    assert solo.local_id("porównanie") is None


def test_empty_comparison_and_gcd_open_exponentiation_one_floor_at_a_time():
    schema = Schema()
    schema.floor = 4
    schema.shape = "N1"
    schema.index = 4
    group, solo = Shelf(), Shelf()
    visited = []
    while schema.floor < 8:
        visited.append(schema.state())
        seen: list = []
        schema.scan(_Empty(schema.shape), lambda _term: False, group, solo, on_decode=seen.append)
        assert seen == []
    assert visited == [
        (4, "N1", 4),
        (5, "P", 0),
        (6, "R1", 0),
        (7, "G", 0),
    ]
    assert schema.state() == (8, "R1", 0)
    assert schema.log == ["schemat wyczerpany"] * 4
    assert group.local_id("porównanie") is None
    assert group.local_id("NWD") is None
    assert group.local_id("potęga") is None


def test_gcd_without_two_binary_functions_opens_floor_8():
    empty = Shelf()
    assert gcd_shape(empty).count == 0
    group = Shelf()
    group.write(parse_def(_ADD))
    shape = gcd_shape(group)
    assert shape.count == 0
    solo = Shelf()
    schema = Schema()
    schema.floor = 7
    schema.shape = "G"
    schema.index = 3
    seen: list = []
    schema.scan(shape, lambda _term: False, group, solo, on_decode=seen.append)
    assert seen == []
    assert schema.log == ["schemat wyczerpany"]
    assert schema.state() == (8, "R1", 0)
    assert group.local_id("NWD") is None
    assert solo.local_id("NWD") is None


def test_gcd_tree_has_two_selves_and_does_not_need_a_hit():
    group = Shelf()
    left = group.write(parse_def(_ADD))
    right = group.write(parse_def(_SUB))
    shape = gcd_shape(group)
    assert shape.count == 2
    assert shape.decode(0) == _gcd_term(left, right)
    assert shape.decode(1) == _gcd_term(right, left)
    assert _self_count(shape.decode(0)) == 2


def test_multiplication_oracle_is_in_r1_and_gets_its_family():
    shelf = Shelf()
    add = shelf.write(parse_def(_ADD))
    assert shelf.entry(add).family == "dodawanie"
    term = _mul(add)
    targets = ((add, 2),)
    assert decode_r1_back(term, 2, targets) == term
    ident = shelf.write(Def("f0", 2, term))
    assert shelf.entry(ident).family == "mnożenie"


def test_sum_oracle_is_in_r1_and_gets_its_family():
    shelf = Shelf()
    add = shelf.write(parse_def(_ADD))
    term = _sum(add)
    targets = ((add, 2),)
    assert decode_r1_back(term, 1, targets) == term
    ident = shelf.write(Def("f0", 1, term))
    assert shelf.entry(ident).family == "suma"


def test_exponentiation_oracle_is_in_r1_and_zero_to_zero_is_one():
    shelf = Shelf()
    add = shelf.write(parse_def(_ADD))
    mul = shelf.write(Def("f0", 2, _mul(add)))
    assert shelf.entry(mul).family == "mnożenie"
    term = _exp(mul)
    targets = ((mul, 2),)
    assert decode_r1_back(term, 2, targets) == term
    ident = shelf.write(Def("f0", 2, term))
    assert shelf.entry(ident).family == "potęga"
    defs = {
        add: Def(add, 2, shelf.entry(add).body),
        mul: Def(mul, 2, shelf.entry(mul).body),
        ident: Def(ident, 2, shelf.entry(ident).body),
    }
    assert reduce_def(defs[ident], (0, 0), defs=defs) == 1


def test_comparison_oracle_closes_floor_4_and_min_max_do_not():
    shelf = Shelf()
    monus = shelf.write(parse_def(_SUB))
    assert shelf.entry(monus).family == "odejmowanie"
    targets = ((monus, 2),)
    smaller = _bound(monus, Var(0), Var(1))
    larger = _bound(monus, Var(1), Var(0))
    compared = _bound(monus, Succ(Zero()), Zero())
    assert decode_n1(encode_n1(smaller, targets), targets) == smaller
    assert decode_n1(encode_n1(larger, targets), targets) == larger
    assert decode_n1(encode_n1(compared, targets), targets) == compared
    low = shelf.write(Def("f0", 2, smaller))
    high = shelf.write(Def("f0", 2, larger))
    assert shelf.entry(low).family == "minimum"
    assert shelf.entry(high).family == "maksimum"
    schema = Schema()
    schema.floor = 4
    schema.shape = "N1"
    schema.index = 11
    assert schema.advance_if_closed(shelf) is False
    assert schema.state() == (4, "N1", 11)
    assert shelf.local_id("porównanie") is None
    ident = shelf.write(Def("f0", 2, compared))
    assert shelf.entry(ident).family == "porównanie"
    assert schema.advance_if_closed(shelf) is True
    assert schema.state() == (5, "P", 0)
    assert "schemat wyczerpany" not in schema.log


def test_parity_search_stops_at_the_first_hit():
    oracle = parse_def(_PARITY).body
    assert decode_p(encode_p(oracle)) == oracle
    shelf = Shelf()
    ident = shelf.write(parse_def(_PARITY))
    assert shelf.entry(ident).family == "parzystość"
    group, solo = Shelf(), Shelf()
    schema = Schema()
    schema.floor = 5
    schema.shape = "P"
    schema.index = 0
    seen: list = []
    found = schema.seek_parity(group, solo, on_decode=seen.append)
    assert found is not None
    assert 1 <= len(seen) <= 36
    assert seen[-1] == found == decode_p(len(seen) - 1)
    assert _body(group, "parzystość") == _body(solo, "parzystość") == found
    assert schema.state() == (6, "R1", 0)
    assert "schemat wyczerpany" not in schema.log
    again: list = []
    assert schema.seek_parity(group, solo, on_decode=again.append) is None
    assert again == []


def decode_r1_back(term, arity, targets):
    from kanon.v2.schema import decode_r1

    return decode_r1(encode_r1(term, arity, targets), arity, targets)


def _mul(add: str):
    return If0(Var(0), Zero(), Call(add, (Var(1), Self((Pred(Var(0)), Var(1))))))


def _sum(add: str):
    return If0(Var(0), Zero(), Call(add, (Var(0), Self((Pred(Var(0)),)))))


def _exp(mul: str):
    return If0(Var(1), Succ(Zero()), Call(mul, (Var(0), Self((Var(0), Pred(Var(1)))))))


def _bound(monus: str, zero_branch, nonzero_branch):
    return If0(Call(monus, (Var(0), Var(1))), zero_branch, nonzero_branch)


def _gcd_term(left: str, right: str):
    return If0(
        Var(1),
        Var(0),
        If0(
            Call(left, (Var(0), Var(1))),
            Self((Var(1), Call(right, (Var(0), Var(1))))),
            Self((Var(1), Call(right, (Var(1), Var(0))))),
        ),
    )


def _self_count(expr) -> int:
    if isinstance(expr, Self):
        return 1 + sum(_self_count(arg) for arg in expr.args)
    if isinstance(expr, Call):
        return sum(_self_count(arg) for arg in expr.args)
    if isinstance(expr, If0):
        return _self_count(expr.cond) + _self_count(expr.zero_branch) + _self_count(expr.nonzero_branch)
    if isinstance(expr, (Succ, Pred)):
        return _self_count(expr.arg)
    return 0


def _body(shelf: Shelf, family: str):
    ident = shelf.local_id(family)
    assert ident is not None
    return shelf.entry(ident).body


class _Empty:
    def __init__(self, name: str):
        self.name = name
        self.count = 0

    def decode(self, index: int):
        raise AssertionError(index)
