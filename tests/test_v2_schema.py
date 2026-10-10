from kanon.language import Call, If0, Self, Zero, parse_def
from kanon.v2.schema import Schema, decode_r1, decode_r2, encode_r1, encode_r2
from kanon.v2.shelf import Shelf

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_SUB = (
    "(def f0 2 (if0 (var 0) (zero) "
    "(if0 (var 1) (var 0) (self (pred (var 0)) (pred (var 1))))))"
)


def test_two_successors_are_not_addition():
    shelf = Shelf()
    ident = shelf.write(parse_def("(def f0 2 (succ (succ (var 0))))"))
    assert shelf.entry(ident).family != "dodawanie"


def test_addition_oracle_roundtrips_and_stops_the_search():
    oracle = parse_def(_ADD).body
    index = encode_r1(oracle, 2)
    assert decode_r1(index, 2) == oracle
    group, solo = Shelf(), Shelf()
    schema = Schema()
    seen: list = []
    schema.seek_addition(group, solo, index - 3, on_decode=seen.append)
    assert seen[0] == decode_r1(index - 3, 2)
    assert seen[-1] == oracle
    assert len(seen) == 4
    assert schema.state() == (2, "R2", 0)
    assert _body(group, "dodawanie") == _body(solo, "dodawanie") == oracle
    again: list = []
    schema.seek_addition(group, solo, index - 3, on_decode=again.append)
    assert again == []
    assert schema.state() == (2, "R2", 0)


def test_subtraction_alone_stays_on_the_shelf_and_leaves_the_floor():
    oracle = parse_def(_SUB).body
    index = encode_r2(oracle, 2)
    assert decode_r2(index, 2) == oracle
    group, solo = Shelf(), Shelf()
    schema = Schema()
    seen: list = []
    schema.seek_subtraction(group, solo, index - 3, on_decode=seen.append)
    assert seen[0] == decode_r2(index - 3, 2)
    assert seen[-1] == oracle
    assert len(seen) == 4
    assert schema.state() == (2, "R2", index)
    assert schema.floor == 2
    assert _body(group, "odejmowanie") == _body(solo, "odejmowanie") == oracle
    assert group.local_id("dodawanie") is None


def test_floor_closes_after_addition_and_subtraction():
    schema = Schema()
    group, solo = Shelf(), Shelf()
    add = parse_def(_ADD).body
    sub = parse_def(_SUB).body
    schema.seek_addition(group, solo, encode_r1(add, 2) - 3)
    assert schema.floor == 2
    schema.seek_subtraction(group, solo, encode_r2(sub, 2) - 3)
    assert schema.state() == (3, "R1", 0)
    assert _body(group, "dodawanie") == _body(solo, "dodawanie")
    assert _body(group, "odejmowanie") == _body(solo, "odejmowanie")


def test_exhausted_shape_advances_the_floor_once():
    schema = Schema()
    group, solo = Shelf(), Shelf()
    seen: list = []
    schema.scan(_Five(), lambda _term: False, group, solo, on_decode=seen.append)
    assert seen == [Zero(), Zero(), Zero(), Zero(), Zero()]
    assert schema.log == ["schemat wyczerpany"]
    assert schema.state() == (3, "R1", 0)
    assert group.local_id("dodawanie") is None
    assert solo.local_id("dodawanie") is None
    again: list = []
    schema.scan(_Five(), lambda _term: False, group, solo, on_decode=again.append)
    assert again == []
    assert schema.log == ["schemat wyczerpany"]
    assert schema.state() == (3, "R1", 0)


def test_call_wrapper_is_in_the_r1_image():
    expr = If0(
        Zero(),
        Zero(),
        Call("e0", (Self((Zero(), Zero())),)),
    )
    index = encode_r1(expr, 2, targets=(("e0", 1),))
    assert decode_r1(index, 2, targets=(("e0", 1),)) == expr
    assert index == 3 * 9**4


class _Five:
    name = "toy"
    count = 5

    def decode(self, index: int):
        return Zero()


def _body(shelf: Shelf, family: str):
    ident = shelf.local_id(family)
    assert ident is not None
    return shelf.entry(ident).body
