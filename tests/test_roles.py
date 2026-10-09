import pytest

from kanon.canon import Canon
from kanon.language import TOKENS, parse_def, tokenize
from kanon.roles import Accepted, Propose, Role, SlotSession, assign, format_def, format_expr, observe
from kanon.tasks import IOPair, generate_task


def test_each_agent_plays_each_role_twice_in_eight_rounds():
    counts = {(agent, role): 0 for agent in range(4) for role in Role}
    for round_no in range(8):
        roles = assign(round_no)
        counts[roles.author, Role.AUTHOR] += 1
        counts[roles.breaker, Role.BREAKER] += 1
        counts[roles.user, Role.USER] += 1
        counts[roles.outsider, Role.OUTSIDER] += 1
    assert all(count == 2 for count in counts.values())
    assert assign(0).author == 0
    assert assign(0).breaker == 1
    assert assign(0).user == 2
    assert assign(0).outsider == 3
    assert assign(1).outsider == 0


@pytest.mark.parametrize(
    ("role", "source"),
    [
        (Role.BREAKER, "(propose (def f0 1 (var 0)))"),
        (Role.AUTHOR, "zdanie po polsku"),
        (Role.AUTHOR, "(draft (def f0 1 (num 4)))"),
        (Role.AUTHOR, "(propose (def f0 1 (num 4)))"),
        (Role.AUTHOR, "(draft (def f32 1 (var 0)))"),
        (Role.USER, "(solve (var 0))"),
        (Role.OUTSIDER, "(break f0 (num 1))"),
        (Role.BREAKER, "(draft (var 0))"),
    ],
)
def test_wrong_shape_is_only_the_abs_token(role, source):
    session = SlotSession(role, _public(), free_slot="f0", candidate_slot="f0")
    session.submit("(draft (var 0))")
    assert session.submit(source) == "[ABS]"


def test_final_message_before_a_draft_is_abs_and_fifth_draft_is_abs():
    session = _author()
    propose = "(propose (def f0 1 (var 0)))"
    draft = "(draft (def f0 1 (var 0)))"
    assert session.submit(propose) == "[ABS]"
    user = SlotSession(Role.USER, _public(), candidate_slot="f0")
    outsider = SlotSession(Role.OUTSIDER, _public())
    assert user.submit("(use f0 (var 0))") == "[ABS]"
    assert outsider.submit("(solve (zero))") == "[ABS]"
    for _ in range(4):
        assert session.submit(draft) == "[TRY] (num 0) (num 1) (num 2) (num 3) (num 4)"
    assert session.submit(draft) == "[ABS]"
    accepted = session.submit(propose)
    assert isinstance(accepted, Accepted)
    assert isinstance(accepted.message, Propose)
    assert session.canon.replay() == ()


def test_wrong_arity_draft_returns_five_errors_and_spends_a_try():
    session = _author()
    assert session.submit("(draft (def f0 2 (var 0)))") == "[TRY] (err) (err) (err) (err) (err)"
    assert isinstance(session.submit("(propose (def f0 1 (var 0)))"), Accepted)

    user = SlotSession(Role.USER, _public(), candidate_slot="f0")
    assert user.submit("(draft (var 1))") == "[TRY] (err) (err) (err) (err) (err)"
    assert user.submit("(use f0 (var 0))") == "[ABS]"
    assert isinstance(user.submit("(use f0 (succ (call f0 (var 0))))"), Accepted)


def test_draft_reply_uses_public_pairs_only():
    session = _author()
    reply = session.submit("(draft (def f0 1 (var 0)))")
    assert isinstance(reply, str)
    assert reply.startswith("[TRY] ")
    assert "(num 10)" not in reply
    assert reply.count("(num ") == 5
    _assert_dictionary(reply)


def test_breaker_takes_four_breaks_and_rejects_propose():
    session = SlotSession(Role.BREAKER, _public())
    assert session.submit("(propose (def f0 1 (var 0)))") == "[ABS]"
    for _ in range(4):
        assert isinstance(session.submit("(break f0 (num 3))"), Accepted)
    assert session.submit("(break f0 (num 4))") == "[ABS]"
    assert isinstance(session.submit("(abstain f0)"), Accepted)


def test_observations_are_dictionary_tokens_and_hide_the_target():
    task = generate_task(0, 2)
    empty = Canon()
    canon = Canon()
    _admit(canon, "(def f1 1 (pred (var 0)))")
    candidate = parse_def("(def f9 1 (succ (succ (var 0))))")
    hidden = format_expr(task.target)
    pairs = task.view.public_pairs

    outsider = observe(Role.OUTSIDER, canon=empty, pairs=pairs, candidate=candidate)
    assert hidden not in outsider
    assert format_def(candidate) not in outsider
    assert outsider.endswith("[OUTSIDER]")
    visible = observe(Role.OUTSIDER, canon=canon, pairs=pairs)
    assert "(def f1 1 (pred (var 0)))" in visible

    breaker = observe(Role.BREAKER, canon=canon, pairs=pairs, candidate=candidate)
    assert format_def(candidate) in breaker
    assert breaker.endswith("[BREAK]")

    user = observe(Role.USER, canon=canon, pairs=pairs, candidate=candidate)
    assert user.split().count("f9") == 1
    assert "(succ (succ (var 0)))" not in user
    assert user.endswith("[USE]")

    author = observe(Role.AUTHOR, canon=canon, pairs=_public(), free_slot="f3")
    assert author.split().count("f3") == 1
    assert author.startswith("[CANON] ")
    assert "[PAIRS]" in author
    assert author.endswith("[AUTHOR]")
    assert format_def(candidate) not in author

    for text in (outsider, visible, breaker, user, author):
        _assert_dictionary(text)


def test_breaker_sees_cemetery_inputs_without_the_target_term():
    canon = Canon()
    killed = parse_def("(def f2 1 (succ (succ (succ (zero)))))")
    canon.bury(
        slot="f2",
        definition=killed,
        inputs=(4,),
        def_result=5,
        target_result=6,
        round_no=1,
    )
    canon.bury(
        slot="f2",
        definition=killed,
        inputs=(2,),
        def_result=2,
        target_result=2,
        round_no=1,
    )
    candidate = parse_def("(def f9 1 (var 0))")
    text = observe(Role.BREAKER, canon=canon, pairs=_public(), candidate=candidate)
    assert "(err (num 4))" in text
    assert "(zero (num 2))" in text
    assert format_expr(killed.body) not in text
    assert text.endswith("[BREAK]")
    _assert_dictionary(text)
    outsider = observe(Role.OUTSIDER, canon=canon, pairs=_public())
    assert "(err (num 4))" not in outsider


def _author() -> SlotSession:
    canon = Canon()
    return SlotSession(Role.AUTHOR, _public(), canon=canon, free_slot="f0")


def _public() -> tuple[IOPair, ...]:
    return tuple(IOPair((i,), i) for i in range(5))


def _admit(canon: Canon, source: str) -> None:
    definition = parse_def(source)
    public = (IOPair((0,), 0),)
    audit = tuple(IOPair((i,), max(0, i - 1)) for i in range(16))
    assert canon.admit(definition, public, audit, 0) == definition.slot


def _assert_dictionary(text: str) -> None:
    tokens = tokenize(text)
    for index, token in enumerate(tokens):
        if token in TOKENS:
            continue
        assert index > 0 and tokens[index - 1] == "num" and token.isdigit()
