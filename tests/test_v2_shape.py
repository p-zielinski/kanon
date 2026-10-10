import pytest

from kanon.language import Call, Pred, Succ, Var, Zero, tokenize
from kanon.roles import format_expr
from kanon.v2.shape import decode_hole, decode_pair, hole_count


def test_simple_hole_counts_and_has_no_call():
    assert hole_count(2) == 9
    assert hole_count(1) == 6
    for arity in (1, 2):
        terms = [decode_hole(index, arity) for index in range(hole_count(arity))]
        assert len(set(terms)) == hole_count(arity)
        assert all(not _contains_call(term) for term in terms)
    assert {decode_hole(index, 1) for index in range(6)} == {
        Zero(),
        Var(0),
        Succ(Zero()),
        Pred(Zero()),
        Succ(Var(0)),
        Pred(Var(0)),
    }


def test_arity_2_lengths_do_not_shrink():
    lengths = [
        len(tokenize(format_expr(decode_hole(index, 2)))) for index in range(9)
    ]
    assert all(lengths[index + 1] >= lengths[index] for index in range(8))
    with pytest.raises(ValueError):
        decode_hole(9, 2)


def test_product_index_returns_one_pair():
    pair = decode_pair(80, 9)
    assert isinstance(pair, tuple)
    assert len(pair) == 2
    assert pair == (decode_hole(8, 2), decode_hole(8, 2))


def _contains_call(expr) -> bool:
    if isinstance(expr, Call):
        return True
    if isinstance(expr, (Succ, Pred)):
        return _contains_call(expr.arg)
    if isinstance(expr, (Zero, Var)):
        return False
    raise TypeError(type(expr))
