import torch

from kanon.canon import Canon
from kanon.language import parse_def
from kanon.model import build_agents
from kanon.roles import Role
from kanon.tasks import IOPair
from kanon.update import BATCH, GRAD_CLIP, LR, Tape, batch_tensor, encode, step, step_all, trainable


def test_drafts_rejections_and_unconfirmed_breaks_stay_off_the_tape():
    tape = Tape()
    tape.note(0, Role.AUTHOR, "(draft (def f0 1 (var 0)))", accepted=True)
    tape.note(0, Role.AUTHOR, "(propose (def f0 1 (zero)))", accepted=False)
    tape.note(0, Role.AUTHOR, "(propose (def f0 1 (var 0)))", accepted=True)
    assert [item.text for item in tape.traces] == ["(propose (def f0 1 (var 0)))"]
    assert tape.traces[0].agent == 0
    assert tape.traces[0].role is Role.AUTHOR

    canon = Canon()
    _bury(canon, def_result=1, target_result=1, inputs=(1,))
    _bury(canon, def_result=0, target_result=1, inputs=(0,))
    tape.note_breaks(canon)
    assert tape.sources(1, Role.AUTHOR) == ()
    assert tape.sources(1, Role.BREAKER) == ("(break f0 (num 0))",)
    assert LR == 3e-4
    assert BATCH == 32
    assert GRAD_CLIP == 1.0


def test_batch_is_32_rows_cut_at_64_tokens():
    tape = Tape()
    body = "(zero)"
    for _ in range(40):
        body = f"(succ {body})"
    tape.note(0, Role.OUTSIDER, f"(solve {body})", accepted=True)
    packed = batch_tensor(tape, 0, Role.OUTSIDER)
    assert packed is not None
    ids, _lengths = packed
    assert ids.shape == (32, 64)
    assert encode(f"(solve {body})") is not None
    assert len(encode(f"(solve {body})") or ()) == 64


def test_numeral_outside_the_vocab_expands_before_the_loss():
    tower = "(zero)"
    for _ in range(10):
        tower = f"(succ {tower})"
    assert encode("(break f0 (num 10))") == encode(f"(break f0 {tower})")
    assert encode("(break f0 (num 0))") is not None
    assert encode("(break f0 (num 3))") != encode("(break f0 (zero))")
    agents = build_agents()
    tape = Tape()
    tape.note(0, Role.BREAKER, "(break f0 (num 10))", accepted=True)
    assert step((agents[0],), tape, agent=0, role=Role.BREAKER) is True


def test_empty_tape_skips_and_one_solve_updates_only_that_agent():
    agents = build_agents()
    tape = Tape()
    roles = (Role.AUTHOR, Role.BREAKER, Role.OUTSIDER, Role.USER)
    saved = _snapshot(agents)
    assert step_all(agents, tape, roles) == (False, False, False, False)
    assert _same(agents, saved)

    tape.note(2, Role.OUTSIDER, "(solve (var 0))", accepted=True)
    saved = _snapshot(agents)
    flags = step_all(agents, tape, roles)
    assert flags == (False, False, True, False)
    assert not _same_one(agents[2], saved[2])
    for index in (0, 1, 3):
        assert _same_one(agents[index], saved[index])


def test_call_stays_trainable_after_a_step_on_the_canon_export():
    agents = build_agents()
    canon = Canon()
    call = "(call f0 (var 0))"
    assert trainable(call, canon) is False
    _admit_f0(canon)
    tape = Tape()
    tape.note_canon(canon)
    assert any(text.startswith("(def f0 ") for text in tape.sources(0, Role.AUTHOR))
    assert trainable(call, canon) is True
    assert step(agents, tape, agent=0, role=Role.AUTHOR) is True
    assert trainable(call, canon) is True
    assert trainable("(call f1 (var 0))", canon) is False
    ids = encode(call)
    assert ids is not None and len(ids) <= 64
    agents[0].logits(torch.tensor([ids], dtype=torch.long))


def _snapshot(agents):
    return tuple(
        tuple(param.detach().clone() for param in agent.parameters()) for agent in agents
    )


def _same(agents, saved) -> bool:
    return all(_same_one(agent, rows) for agent, rows in zip(agents, saved, strict=True))


def _same_one(agent, rows) -> bool:
    return all(torch.equal(param, saved) for param, saved in zip(agent.parameters(), rows, strict=True))


def _admit_f0(canon: Canon) -> None:
    definition = parse_def("(def f0 1 (var 0))")
    public = (IOPair((0,), 0),)
    audit = tuple(IOPair((index,), index) for index in range(16))
    assert canon.admit(definition, public, audit, 0) == "f0"


def _bury(canon: Canon, *, def_result: int, target_result: int, inputs: tuple[int, ...]) -> None:
    canon.bury(
        slot="f0",
        definition=parse_def("(def f0 1 (var 0))"),
        inputs=inputs,
        def_result=def_result,
        target_result=target_result,
        round_no=0,
    )
