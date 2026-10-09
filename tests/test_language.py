import pytest

from kanon.language import (
    TOKENS,
    Abstain,
    Break,
    DraftDef,
    DraftExpr,
    ParseError,
    Propose,
    Solve,
    Succ,
    Use,
    Zero,
    expand_num,
    parse_data,
    parse_def,
    parse_expr,
    parse_message,
    tokenize,
)

PLAN_TOKENS = (
    "(",
    ")",
    "zero",
    "succ",
    "pred",
    "if0",
    "var",
    "call",
    "self",
    "def",
    "arity",
    "num",
    "draft",
    "err",
    *(f"f{i}" for i in range(32)),
    "0",
    "1",
    "2",
    "3",
    "[AUTHOR]",
    "[BREAK]",
    "[USE]",
    "[OUTSIDER]",
    "[PAIRS]",
    "[CANON]",
    "[SEP]",
    "[ABS]",
    "[TRY]",
)

ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"


def test_token_vocabulary_is_exactly_the_plan_list():
    assert TOKENS == PLAN_TOKENS
    assert len(TOKENS) == len(set(TOKENS))
    for head in ("propose", "use", "solve", "break", "abstain"):
        assert head not in TOKENS


def test_slots_are_only_f0_through_f31():
    assert [tok for tok in TOKENS if tok.startswith("f")] == [f"f{i}" for i in range(32)]
    parse_def("(def f0 0 (zero))")
    parse_def("(def f31 0 (zero))")
    with pytest.raises(ParseError):
        parse_def("(def f32 0 (zero))")
    with pytest.raises(ParseError):
        parse_expr("(call f32)")


def test_call_accepts_at_most_three_arguments():
    parse_def("(def f0 0 (call f1 (zero) (zero) (zero)))")
    with pytest.raises(ParseError):
        parse_def("(def f0 0 (call f1 (zero) (zero) (zero) (zero)))")


def test_max_arity_is_three_and_var_index_is_below_arity():
    parse_def("(def f0 3 (var 2))")
    with pytest.raises(ParseError):
        parse_def("(def f0 4 (zero))")
    with pytest.raises(ParseError):
        parse_def("(def f0 3 (var 3))")
    with pytest.raises(ParseError):
        parse_def("(def f0 1 (var 1))")


def test_self_outside_def_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_expr("(self (zero))")
    with pytest.raises(ParseError):
        parse_data("(self)")
    parsed = parse_def("(def f0 0 (self))")
    assert parsed.arity == 0


def test_num_in_def_body_is_a_parse_error():
    with pytest.raises(ParseError):
        parse_def("(def f0 0 (num 4))")
    with pytest.raises(ParseError):
        parse_def("(def f0 1 (succ (num 4)))")
    with pytest.raises(ParseError):
        parse_def("(def f0 1 (if0 (var 0) (num 1) (zero)))")


def test_num_in_break_argument_expands_to_four_succs():
    message = parse_message("(break f0 (num 4))")
    assert message == Break("f0", (4,))
    assert expand_num(4) == Succ(Succ(Succ(Succ(Zero()))))
    assert parse_data("(num 4)") != parse_expr("(succ (succ (succ (succ (zero)))))")
    assert expand_num(message.args[0]) == parse_expr("(succ (succ (succ (succ (zero)))))")


def test_polish_and_python_comment_do_not_parse():
    with pytest.raises(ParseError):
        parse_expr("dodawanie dwóch liczb")
    with pytest.raises(ParseError):
        parse_message("(draft (wynik jest zero))")
    with pytest.raises(ParseError):
        parse_expr("(succ (zero)) # komentarz")
    with pytest.raises(ParseError):
        parse_expr("# komentarz\n(zero)")
    with pytest.raises(ParseError):
        tokenize("(zero) ; komentarz")


def test_message_shapes():
    assert parse_message(f"(draft {ADD})") == DraftDef(parse_def(ADD))
    assert parse_message("(draft (succ (zero)))", arity=0) == DraftExpr(
        parse_expr("(succ (zero))")
    )
    assert parse_message(f"(propose {ADD})") == Propose(parse_def(ADD))
    assert parse_message("(use f3 (call f3 (var 0)))", arity=1) == Use(
        "f3", parse_expr("(call f3 (var 0))", arity=1)
    )
    assert parse_message("(solve (zero))") == Solve(parse_expr("(zero)"))
    assert parse_message("(abstain f31)") == Abstain("f31")
    assert parse_message("(break f2 (num 0) (num 10))") == Break("f2", (0, 10))


def test_tokenize_observation_markers():
    assert tokenize("[CANON] (def f0 1 (zero)) [SEP]") == [
        "[CANON]",
        "(",
        "def",
        "f0",
        "1",
        "(",
        "zero",
        ")",
        ")",
        "[SEP]",
    ]
