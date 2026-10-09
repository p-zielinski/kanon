import pytest

from kanon.canon import Canon
from kanon.language import (
    SLOTS,
    Abstain,
    Break,
    Call,
    DraftDef,
    DraftExpr,
    Expr,
    If0,
    Num,
    ParseError,
    Pred,
    Propose,
    Self,
    Solve,
    Succ,
    Use,
    Zero,
    parse_def,
    parse_message,
    tokenize,
)
from kanon.roles import Accepted, Role, SlotSession, observe
from kanon.sample import legal_tokens, sample, speak
from kanon.tasks import IOPair

MARKERS = {
    Role.AUTHOR: "[AUTHOR]",
    Role.BREAKER: "[BREAK]",
    Role.USER: "[USE]",
    Role.OUTSIDER: "[OUTSIDER]",
}
FREE = "f1"
CANDIDATE = "f9"


@pytest.mark.parametrize("filled", [False, True])
def test_fifty_samples_of_each_role_parse(filled):
    canon = _with_f0() if filled else Canon()
    allowed = {"f0"} if filled else set()
    pairs = _pairs()
    for role in Role:
        seen: set[str] = set()
        for seed in range(50):
            text = sample(MARKERS[role], _obs(role, canon, pairs), seed=seed)
            seen.add(text)
            assert len(tokenize(text)) <= 64
            message = parse_message(text, arity=1)
            _assert_role(role, message, text, allowed)
            session = SlotSession(
                role,
                pairs,
                canon=canon,
                free_slot=FREE,
                candidate_slot=CANDIDATE,
            )
            assert session.submit(text) != "[ABS]"
        assert len(seen) > 1


def test_seed_repeats_the_same_message():
    observation = _obs(Role.AUTHOR, Canon(), _pairs())
    assert sample("[AUTHOR]", observation, seed=0) == sample("[AUTHOR]", observation, seed=0)


def test_author_propose_waits_for_a_try_and_stops_after_four():
    canon = Canon()
    pairs = _pairs()
    kinds_after_one: set[type] = set()
    for seed in range(32):
        draft = sample("[AUTHOR]", _obs(Role.AUTHOR, canon, pairs, tries=0), seed=seed)
        assert isinstance(parse_message(draft, arity=1), DraftDef)
        assert "num" not in tokenize(draft)
        proposed = sample("[AUTHOR]", _obs(Role.AUTHOR, canon, pairs, tries=4), seed=seed)
        message = parse_message(proposed, arity=1)
        assert isinstance(message, Propose)
        assert message.definition.slot == FREE
        session = SlotSession(Role.AUTHOR, pairs, free_slot=FREE)
        assert session.submit("(draft (def f1 1 (zero)))") != "[ABS]"
        assert isinstance(session.submit(proposed), Accepted)
        mixed = sample("[AUTHOR]", _obs(Role.AUTHOR, canon, pairs, tries=1), seed=seed)
        kinds_after_one.add(type(parse_message(mixed, arity=1)))
    assert kinds_after_one == {DraftDef, Propose}


def test_user_and_outsider_final_forms_follow_the_same_budget():
    canon = _with_f0()
    pairs = _pairs()
    for seed in range(8):
        use = sample("[USE]", _obs(Role.USER, canon, pairs, tries=4), seed=seed)
        use_message = parse_message(use, arity=1)
        assert isinstance(use_message, Use)
        assert use_message.slot == CANDIDATE
        assert CANDIDATE in _calls(use_message.expr)
        assert set(_calls(use_message.expr)) <= {"f0", CANDIDATE}
        user = SlotSession(Role.USER, pairs, candidate_slot=CANDIDATE)
        assert user.submit("(draft (call f9 (var 0)))") != "[ABS]"
        assert isinstance(user.submit(use), Accepted)

        solve = sample("[OUTSIDER]", _obs(Role.OUTSIDER, canon, pairs, tries=4), seed=seed)
        solve_message = parse_message(solve, arity=1)
        assert isinstance(solve_message, Solve)
        assert set(_calls(solve_message.expr)) <= {"f0"}
        early = sample("[OUTSIDER]", _obs(Role.OUTSIDER, canon, pairs, tries=0), seed=seed)
        assert isinstance(parse_message(early, arity=1), DraftExpr)


def test_breaker_samples_abstain_or_break_without_a_draft():
    canon = _with_f0()
    pairs = _pairs()
    kinds: set[type] = set()
    for seed in range(32):
        text = sample("[BREAK]", _obs(Role.BREAKER, canon, pairs), seed=seed)
        message = parse_message(text, arity=1)
        kinds.add(type(message))
        assert not isinstance(message, (DraftDef, DraftExpr))
        if isinstance(message, Break):
            assert message.slot == CANDIDATE
            assert len(message.args) == 1
            assert 0 <= message.args[0] <= 10
        else:
            assert isinstance(message, Abstain)
            assert message.slot == CANDIDATE
    assert kinds == {Break, Abstain}


def test_def_arity_follows_the_public_pairs():
    pairs = tuple(IOPair((i, 0), i) for i in range(5))
    text = sample("[AUTHOR]", _obs(Role.AUTHOR, Canon(), pairs), seed=0)
    message = parse_message(text, arity=2)
    assert isinstance(message, DraftDef)
    assert message.definition.arity == 2
    user = sample("[USE]", _obs(Role.USER, Canon(), pairs), seed=0)
    body = parse_message(user, arity=2)
    assert isinstance(body, DraftExpr)
    calls = [node for node in _walk(body.expr) if isinstance(node, Call)]
    assert any(node.slot == CANDIDATE and len(node.args) == 2 for node in calls)


def test_try_before_the_role_marker_keeps_the_free_slot():
    base = _obs(Role.AUTHOR, Canon(), _pairs())
    head, marker, tail = base.rpartition("[AUTHOR]")
    block = "[TRY] (err) (err) (err) (err) (err) "
    observation = f"{head}{block * 4}{marker}{tail}"
    message = parse_message(sample("[AUTHOR]", observation, seed=3), arity=1)
    assert isinstance(message, Propose)
    assert message.definition.slot == FREE


def test_bad_marker_is_rejected():
    with pytest.raises(ValueError):
        sample("[ABS]", _obs(Role.AUTHOR, Canon(), _pairs()), seed=0)


def _assert_role(role: Role, message, text: str, canon_slots: set[str]) -> None:
    if role is Role.AUTHOR:
        assert isinstance(message, DraftDef)
        assert message.definition.slot == FREE
        assert message.definition.arity == 1
        assert "num" not in tokenize(text)
        assert not _has_num(message.definition.body)
        assert set(_calls(message.definition.body)) <= canon_slots
        return
    if role is Role.BREAKER:
        assert isinstance(message, (Break, Abstain))
        assert message.slot == CANDIDATE
        if isinstance(message, Break):
            assert message.args
            assert all(0 <= value <= 10 for value in message.args)
        return
    if role is Role.USER:
        assert isinstance(message, DraftExpr)
        found = _calls(message.expr)
        assert CANDIDATE in found
        assert set(found) <= canon_slots | {CANDIDATE}
        return
    assert isinstance(message, DraftExpr)
    assert set(_calls(message.expr)) <= canon_slots
    for token in tokenize(text):
        if token.startswith("f") and token[1:].isdigit():
            assert token in SLOTS


def _obs(role: Role, canon: Canon, pairs: tuple[IOPair, ...], *, tries: int = 0) -> str:
    text = observe(
        role,
        canon=canon,
        pairs=pairs,
        free_slot=FREE,
        candidate=parse_def("(def f9 1 (succ (var 0)))"),
    )
    if tries:
        text += " " + " ".join(["[TRY] (err) (err) (err) (err) (err)"] * tries)
    return text


def _pairs() -> tuple[IOPair, ...]:
    return tuple(IOPair((i,), i) for i in range(5))


def _with_f0() -> Canon:
    canon = Canon()
    definition = parse_def("(def f0 1 (var 0))")
    public = (IOPair((0,), 0),)
    audit = tuple(IOPair((i,), i) for i in range(16))
    assert canon.admit(definition, public, audit, 0) == "f0"
    return canon


def _calls(expr: Expr) -> list[str]:
    return [node.slot for node in _walk(expr) if isinstance(node, Call)]


def _has_num(expr: Expr) -> bool:
    return any(isinstance(node, Num) for node in _walk(expr))


def _walk(expr: Expr):
    stack = [expr]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, (Call, Self)):
            stack.extend(node.args)


class _Flat:
    def distribution(self, token_ids, legal):
        import torch

        mask = legal if legal.dim() == 2 else legal.unsqueeze(0)
        probs = mask.to(dtype=torch.float)
        return probs / probs.sum(dim=-1, keepdim=True)


def test_unbalanced_observation_does_not_parse_as_a_program():
    with pytest.raises((ValueError, ParseError)):
        sample("[AUTHOR]", "[CANON]", seed=0)


def test_every_sampled_token_is_still_legal():
    pairs = _pairs()
    for role in Role:
        observation = _obs(role, Canon(), pairs)
        for seed in range(12):
            tokens = tokenize(sample(MARKERS[role], observation, seed=seed))
            for index, token in enumerate(tokens):
                legal = legal_tokens(MARKERS[role], observation, tokens[:index])
                assert legal is not None
                assert token in legal
            assert legal_tokens(MARKERS[role], observation, tokens) is None


def test_full_epsilon_matches_the_sampler_and_a_mask_still_closes():
    observation = _obs(Role.OUTSIDER, Canon(), _pairs())
    assert speak("[OUTSIDER]", observation, seed=4, epsilon=1) == sample(
        "[OUTSIDER]", observation, seed=4
    )
    for role in (Role.AUTHOR, Role.USER, Role.BREAKER, Role.OUTSIDER):
        text = speak(MARKERS[role], _obs(role, Canon(), _pairs()), seed=2, epsilon=0, agent=_Flat())
        assert parse_message(text, arity=1) is not None
    with pytest.raises(ValueError):
        speak("[OUTSIDER]", observation, seed=0, epsilon=0)


def test_breaker_observation_may_carry_cemetery_cases():
    canon = Canon()
    canon.bury(
        slot="f2",
        definition=parse_def("(def f2 1 (succ (zero)))"),
        inputs=(4,),
        def_result=1,
        target_result=0,
        round_no=1,
    )
    observation = observe(
        Role.BREAKER,
        canon=canon,
        pairs=_pairs(),
        candidate=parse_def("(def f9 1 (var 0))"),
    )
    message = parse_message(sample("[BREAK]", observation, seed=1), arity=1)
    assert isinstance(message, (Break, Abstain))


def test_epsilon_zero_follows_the_peaked_production():
    observation = _obs(Role.OUTSIDER, Canon(), _pairs(), tries=0)
    text = speak(MARKERS[Role.OUTSIDER], observation, seed=0, epsilon=0, agent=_PeakZero())
    message = parse_message(text, arity=1)
    assert isinstance(message, DraftExpr)
    assert isinstance(message.expr, Zero)


class _PeakZero:
    def distribution(self, token_ids, legal):
        import torch

        from kanon.model import TOKEN_ID, VOCAB

        probs = torch.full((1, len(VOCAB)), 1e-8)
        probs[0, TOKEN_ID["zero"]] = 1.0
        return probs / probs.sum(dim=-1, keepdim=True)


def test_a_real_network_closes_under_the_mask():
    from kanon.model import build_agent

    observation = _obs(Role.USER, Canon(), _pairs())
    text = speak(MARKERS[Role.USER], observation, seed=7, epsilon=0, agent=build_agent(1))
    message = parse_message(text, arity=1)
    assert isinstance(message, DraftExpr)
    assert CANDIDATE in _calls(message.expr)
