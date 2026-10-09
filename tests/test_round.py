import json

import pytest

from kanon.canon import Canon
from kanon.roles import format_def, format_expr
from kanon.round import EPSILON, Script, dumps_event, play, task_seed
from kanon.tasks import IOPair, TaskAudit, generate_task

PASS_SEED = 11
OPEN_SEED = 0


def test_propose_without_a_draft_is_rejected_and_skips_audit(tmp_path):
    canon = Canon()
    event = play(canon, round_no=0, depth=1, seed=OPEN_SEED, script=_propose_only())
    assert event.audit_ran is False
    assert event.admitted is None
    assert event.broken is False
    assert event.graves_added == 0
    assert canon.replay() == ()
    assert canon.graves == ()
    assert event.steps[0][2] == "[ABS]"
    line = dumps_event(event)
    assert json.loads(line)["epsilon"] == EPSILON == 1
    (tmp_path / "events.jsonl").write_text(line + "\n", encoding="utf-8")
    assert json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8"))["audit_ran"] is False


def test_draft_that_fits_public_pairs_and_fails_audit_stays_out():
    canon = Canon()
    author = _task(OPEN_SEED, 0)
    assert author.view.arity == 1
    body = _fits_public_only(author)
    event = play(
        canon,
        round_no=0,
        depth=1,
        seed=OPEN_SEED,
        script=Script(
            author=(_draft(body), _propose(body)),
            breaker=("(abstain f0)",),
        ),
    )
    reply = next(step[2] for step in event.steps if step[2].startswith("[TRY]"))
    assert reply.count("(num ") + reply.count("(err") == 5
    assert reply == "[TRY] " + " ".join(f"(num {pair.output})" for pair in author.view.public_pairs)
    assert event.audit_ran is True
    assert event.admitted is None
    assert canon.replay() == ()


def test_breaker_hit_rejects_the_candidate_and_grows_the_cemetery():
    canon = Canon()
    event = play(
        canon,
        round_no=0,
        depth=1,
        seed=OPEN_SEED,
        script=Script(
            author=(_draft("(zero)"), _propose("(zero)")),
            breaker=("(break f0 (num 0))",),
        ),
    )
    assert event.broken is True
    assert event.audit_ran is False
    assert event.admitted is None
    assert event.graves_added == 1
    assert canon.replay() == ()
    grave = canon.graves[0]
    assert grave.inputs == (0,)
    assert grave.def_result != grave.target_result
    assert grave.round_no == 0


def test_abstain_audit_and_user_call_admit_the_candidate():
    canon = Canon()
    body = "(succ (succ (zero)))"
    event = play(
        canon,
        round_no=3,
        depth=1,
        seed=PASS_SEED,
        script=Script(
            author=(_draft(body), _propose(body)),
            breaker=("(abstain f0)",),
            user=("(draft (call f0 (zero)))", "(use f0 (call f0 (zero)))"),
        ),
    )
    assert event.audit_ran is True
    assert event.broken is False
    assert event.admitted == "f0"
    assert [entry.slot for entry in canon.replay()] == ["f0"]
    assert format_expr(canon.replay()[0].definition.body) == body
    assert canon.graves == ()


def test_outsider_sees_the_definition_admitted_in_this_round(monkeypatch):
    calls: list[tuple[int, tuple[str, ...]]] = []
    real = generate_task

    def wrapped(seed, depth, canon=()):
        calls.append((seed, tuple(item.slot for item in canon)))
        return real(seed, depth, canon=canon)

    monkeypatch.setattr("kanon.round.generate_task", wrapped)
    canon = Canon()
    body = "(succ (succ (zero)))"
    event = play(
        canon,
        round_no=3,
        depth=1,
        seed=PASS_SEED,
        script=Script(
            author=(_draft(body), _propose(body)),
            breaker=("(abstain f0)",),
            user=("(draft (call f0 (zero)))", "(use f0 (call f0 (zero)))"),
            outsider=("(draft (call f0 (zero)))", "(solve (call f0 (zero)))"),
        ),
    )
    assert event.admitted == "f0"
    admitted = format_def(canon.replay()[0].definition)
    observation = event.observations["outsider"]
    assert admitted in observation
    assert observation.index(admitted) < observation.index("[PAIRS]")
    outsider_seed = task_seed(PASS_SEED, 2)
    assert (outsider_seed, ("f0",)) in calls
    assert (task_seed(PASS_SEED, 0), ()) in calls
    draft = next(step for step in event.steps if step[0] == "outsider" and step[1].startswith("(draft"))
    assert draft[2] == "[TRY] (num 2) (num 2) (num 2) (num 2) (num 2)"
    tail = observation.split("[PAIRS]", 1)[1]
    assert event.to_json()["targets"]["outsider"] not in tail


def test_user_without_a_call_blocks_admission():
    canon = Canon()
    body = "(succ (zero))"
    event = play(
        canon,
        round_no=0,
        depth=1,
        seed=OPEN_SEED,
        script=Script(
            author=(_draft(body), _propose(body)),
            breaker=("(abstain f0)",),
            user=("(draft (var 0))", "(use f0 (var 0))"),
        ),
    )
    assert event.audit_ran is True
    assert event.broken is False
    assert event.admitted is None
    assert canon.replay() == ()


def test_use_and_solve_before_a_draft_do_not_count():
    canon = Canon()
    body = "(succ (succ (zero)))"
    event = play(
        canon,
        round_no=0,
        depth=1,
        seed=PASS_SEED,
        script=Script(
            author=(_draft(body), _propose(body)),
            breaker=("(abstain f0)",),
            user=("(use f0 (call f0 (zero)))",),
            outsider=("(solve (zero))",),
        ),
    )
    assert event.admitted is None
    assert canon.replay() == ()
    assert ("user", "(use f0 (call f0 (zero)))", "[ABS]") in event.steps
    assert ("outsider", "(solve (zero))", "[ABS]") in event.steps
    assert event.outsider_ok is False


def test_audit_mismatch_after_abstain_rejects_the_candidate():
    canon = Canon()
    before = len(canon.graves)
    event = play(
        canon,
        round_no=0,
        depth=1,
        seed=OPEN_SEED,
        script=Script(
            author=(_draft("(zero)"), _propose("(zero)")),
            breaker=("(abstain f0)",),
            user=("(draft (call f0 (zero)))", "(use f0 (call f0 (zero)))"),
        ),
    )
    assert event.audit_ran is True
    assert event.broken is False
    assert event.admitted is None
    assert len(canon.graves) == before
    assert canon.replay() == ()


def test_outsider_observation_hides_the_target_term():
    canon = Canon()
    event = play(canon, round_no=0, depth=1, seed=OPEN_SEED, script=_propose_only())
    hidden = format_expr(_task(OPEN_SEED, 2).target)
    observation = event.observations["outsider"]
    assert hidden not in observation
    assert "hidden" not in observation
    assert observation.endswith("[OUTSIDER]")


def test_each_agent_role_shifts_by_one_after_the_round():
    first = play(Canon(), round_no=0, depth=1, seed=OPEN_SEED, script=_propose_only())
    second = play(Canon(), round_no=1, depth=1, seed=OPEN_SEED, script=_propose_only())
    assert first.roles == (0, 1, 2, 3)
    assert second.roles == (1, 2, 3, 0)
    names = ("author", "breaker", "user", "outsider")
    for agent in range(4):
        before = names.index(_name(first.roles, agent))
        after = names.index(_name(second.roles, agent))
        assert after == (before - 1) % 4


def test_unscripted_round_logs_without_a_training_step():
    event = play(Canon(), round_no=0, depth=1, seed=OPEN_SEED)
    payload = json.loads(dumps_event(event))
    assert payload["epsilon"] == 1
    assert set(payload["observations"]) == {"author", "breaker", "user", "outsider"}
    assert event.steps


def _propose_only() -> Script:
    return Script(author=("(propose (def f0 1 (zero)))",))


def _task(seed: int, index: int) -> TaskAudit:
    return generate_task(task_seed(seed, index), 1)


def _draft(body: str) -> str:
    return f"(draft (def f0 1 {body}))"


def _propose(body: str) -> str:
    return f"(propose (def f0 1 {body}))"


def _fits_public_only(task: TaskAudit) -> str:
    """Zgadza się z pięcioma parami publicznymi i myli się poza nimi."""
    body = "(zero)"
    ordered = sorted(task.view.public_pairs, key=lambda pair: pair.inputs[0], reverse=True)
    for pair in ordered:
        body = _arm(pair, body)
    return body


def _arm(pair: IOPair, rest: str) -> str:
    probe = "(var 0)"
    for _ in range(pair.inputs[0]):
        probe = f"(pred {probe})"
    return f"(if0 {probe} {_const(pair.output)} {rest})"


def _const(value: int) -> str:
    expr = "(zero)"
    for _ in range(value):
        expr = f"(succ {expr})"
    return expr


def test_target_sits_beside_the_observations():
    event = play(Canon(), round_no=0, depth=1, seed=OPEN_SEED)
    payload = event.to_json()
    assert "hidden" not in json.dumps(payload["observations"])
    for index, role in ((0, "author"), (1, "user"), (2, "outsider")):
        term = format_expr(generate_task(task_seed(OPEN_SEED, index), 1).target)
        assert payload["targets"][role] == term
        assert term not in payload["observations"][role].split("[PAIRS]", 1)[1]


def test_expiry_waits_until_the_round_is_closed():
    from kanon.language import parse_def

    canon = Canon()
    definition = parse_def("(def f0 1 (var 0))")
    public = (IOPair((0,), 0),)
    audit = tuple(IOPair((index,), index) for index in range(16))
    assert canon.admit(definition, public, audit, 0) == "f0"
    play(canon, round_no=12, depth=1, seed=0, script=Script(), close=False)
    assert [entry.slot for entry in canon.replay()] == ["f0"]
    canon.end_round(12)
    assert canon.replay() == ()


def test_epsilon_below_one_needs_a_policy():
    with pytest.raises(ValueError):
        play(Canon(), round_no=0, depth=1, seed=0, epsilon=0.5)


def _name(roles: tuple[int, int, int, int], agent: int) -> str:
    names = ("author", "breaker", "user", "outsider")
    return names[roles.index(agent)]
