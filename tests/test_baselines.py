import hashlib

import torch

import kanon.baselines as baselines
from kanon.baselines import LEARNING_STEPS, PROPOSALS, frozen_eval, solo_round, start_frozen
from kanon.canon import Canon
from kanon.language import parse_def
from kanon.model import Decoder, build_agent
from kanon.roles import MAX_DRAFTS, Role, format_def, format_expr
from kanon.round import task_seed
from kanon.tasks import AUDIT_COUNT, PUBLIC_COUNT, IOPair, generate_task
from kanon.update import Tape


def test_solo_budget_admits_only_after_sixteen_and_steps_four_times(monkeypatch):
    calls = {"n": 0}
    real = baselines.step

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(baselines, "step", wrapped)
    agent = build_agent(0)
    canon = Canon()
    tape = Tape()
    saved = _snapshot(agent)
    quiet = solo_round(
        agent,
        canon,
        tape,
        round_no=0,
        depth=1,
        seed=0,
        proposals=((), (), (), ()),
    )
    assert PROPOSALS == 4
    assert LEARNING_STEPS == 4
    assert len(quiet.attempts) == 4
    assert quiet.learning_steps == 4
    assert quiet.tries == 0
    assert quiet.audits == 0
    assert calls["n"] == 4
    assert _same(agent, saved)
    assert tape.traces == ()

    event = solo_round(
        agent,
        canon,
        tape,
        round_no=0,
        depth=1,
        seed=0,
        proposals=_author_scripts(),
    )
    assert [item.drafts for item in event.attempts] == [1, 1, MAX_DRAFTS, 0]
    assert [item.public_checks for item in event.attempts] == [5, 5, 20, 0]
    assert [item.audit_pairs for item in event.attempts] == [16, 16, 16, 0]
    assert [item.admitted for item in event.attempts] == ["f0", None, "f1", None]
    assert event.learning_steps == 4
    assert event.tries == 6
    assert event.audits == 3
    assert event.tries <= PROPOSALS * MAX_DRAFTS
    assert event.audits <= PROPOSALS
    assert all(item.public_checks == item.drafts * PUBLIC_COUNT for item in event.attempts)
    assert all(item.audit_pairs in (0, AUDIT_COUNT) for item in event.attempts)
    assert calls["n"] == 8
    assert [entry.slot for entry in canon.replay()] == ["f0", "f1"]
    assert canon.graves == ()
    assert [item.text.startswith("(propose ") for item in tape.traces] == [True, True]
    assert all(item.role is Role.AUTHOR and item.agent == 0 for item in tape.traces)
    assert all("(draft " not in item.text for item in tape.traces)
    assert not _same(agent, saved)

    loose = solo_round(agent, Canon(), Tape(), round_no=0, depth=1, seed=1)
    assert len(loose.attempts) == 4
    assert loose.learning_steps == 4
    assert all(item.drafts <= MAX_DRAFTS for item in loose.attempts)
    assert all(item.audit_pairs in (0, AUDIT_COUNT) for item in loose.attempts)


def test_frozen_weights_keep_the_start_seed_across_five_rounds(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("krok uczenia")

    monkeypatch.setattr(baselines, "step", boom)
    canon = Canon()
    definition = parse_def("(def f0 1 (var 0))")
    public = (IOPair((0,), 0),)
    audit = tuple(IOPair((index,), index) for index in range(16))
    assert canon.admit(definition, public, audit, 0) == "f0"
    exported = tuple(format_def(entry.definition) for entry in canon.replay())

    def refuse(*_args, **_kwargs):
        raise AssertionError("zapis kanonu")

    for name in ("admit", "bury", "note_use", "end_round", "expire", "bench"):
        monkeypatch.setattr(canon, name, refuse)

    frozen = start_frozen(2)
    assert isinstance(frozen.agent, Decoder)
    assert len(frozen.agent.layers) == 4
    before = _weight_hash(frozen.agent)
    assert before == _weight_hash(build_agent(frozen.seed))
    played = frozen_eval(
        frozen,
        canon,
        rounds=5,
        depth=1,
        seed=0,
        scripts=(("(draft (var 0))", "(solve (var 0))"),) * 5,
    )
    assert len(played) == 5
    assert [item.drafts for item in played] == [1, 1, 1, 1, 1]
    assert [item.public_checks for item in played] == [5, 5, 5, 5, 5]
    assert [item.audit_pairs for item in played] == [16, 16, 16, 16, 16]
    assert _weight_hash(frozen.agent) == before
    assert tuple(format_def(entry.definition) for entry in canon.replay()) == exported
    assert canon.graves == ()

    frozen_eval(frozen, canon, rounds=1, depth=1, seed=3)
    assert _weight_hash(frozen.agent) == before


def _author_scripts():
    tasks = [generate_task(task_seed(0, index), 1) for index in range(4)]
    true = [format_expr(task.target) for task in tasks]
    wrong = "(succ (succ (succ (zero))))"
    return (
        (
            "(break f0 (num 0))",
            "(use f0 (var 0))",
            _def("draft", "f0", tasks[0].view.arity, true[0]),
            _def("propose", "f0", tasks[0].view.arity, true[0]),
        ),
        (
            _def("draft", "f1", tasks[1].view.arity, wrong),
            _def("propose", "f1", tasks[1].view.arity, wrong),
        ),
        (
            *(_def("draft", "f1", tasks[2].view.arity, true[2]),) * 5,
            _def("propose", "f1", tasks[2].view.arity, true[2]),
        ),
        (_def("propose", "f2", tasks[3].view.arity, true[3]),),
    )


def _def(kind: str, slot: str, arity: int, body: str) -> str:
    return f"({kind} (def {slot} {arity} {body}))"


def _snapshot(agent):
    return tuple(param.detach().clone() for param in agent.parameters())


def _same(agent, saved) -> bool:
    return all(
        torch.equal(param, row)
        for param, row in zip(agent.parameters(), saved, strict=True)
    )


def _weight_hash(agent) -> str:
    digest = hashlib.sha256()
    for name, param in agent.named_parameters():
        data = param.detach().cpu().contiguous()
        if data.storage_offset() != 0:
            data = data.clone()
        raw = bytes(data.untyped_storage())[: data.nbytes]
        digest.update(name.encode())
        digest.update(raw)
    return digest.hexdigest()
