import pytest

from kanon.language import TOKENS, ParseError, parse_def, parse_expr, parse_message, tokenize
from kanon.v2.budget import message_budget
from kanon.v2.vocab import VOCAB

TOKENS_BEFORE = 59
TERM_FORMS = ("zero", "succ", "pred", "if0", "var", "call", "self", "def")
TACTIC_HEADS = ("eval", "rewrite", "base", "step", "claim", "eq", "dom", "ih")
NUMERALS = tuple(str(k) for k in range(11))


def test_day_zero_vocab_holds_numerals_pad_and_tactic_heads():
    assert isinstance(VOCAB, tuple)
    assert len(VOCAB) == len(set(VOCAB))
    for form in TERM_FORMS:
        assert form in VOCAB
    for word in (*NUMERALS, "[PAD]", *TACTIC_HEADS):
        assert word in VOCAB
    for word in ("4", "5", "6", "7", "8", "9", "10", "[PAD]", *TACTIC_HEADS):
        assert word not in TOKENS


def test_num_k_is_one_token_and_other_digits_stay_zero_through_three():
    for k in range(11):
        assert tokenize(f"(num {k})") == ["(", "num", str(k), ")"]
        assert str(k) in VOCAB
    with pytest.raises(ParseError):
        parse_def("(def f0 4 (zero))")
    with pytest.raises(ParseError):
        parse_expr("(var 4)")
    with pytest.raises(ParseError):
        parse_message("(claim (eq (zero) (zero)))")


def test_tokens_tuple_keeps_the_length_from_before_this_prompt():
    assert len(TOKENS) == TOKENS_BEFORE


def test_budget_is_a_plain_number_and_a_49_token_term_meets_the_gate():
    assert type(message_budget(1)) is int
    assert message_budget(1) == 24
    assert message_budget(2) == 48
    assert message_budget(3) == 48
    assert message_budget(4) == 80
    assert message_budget(12) == 80
    assert message_budget(1, exercise=True) == 80
    term = tuple(f"t{i}" for i in range(49))
    assert len(term) > message_budget(2)
    assert len(term) <= message_budget(4)
