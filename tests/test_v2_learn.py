import torch

from kanon.v2.learn import BATCH, GRAD_CLIP, LR, produce, step
from kanon.v2.measure import score_set
from kanon.v2.model import build_net
from kanon.v2.play import Tape
from kanon.v2.window import OBS_LEN


def test_production_starts_from_the_observation_prefix():
    net = build_net(0)
    lengths: list[int] = []
    real = net.forward_prefix

    def spy(token_ids):
        lengths.append(int(token_ids.shape[1]))
        return real(token_ids)

    net.forward_prefix = spy
    produce(net, tuple(0 for _ in range(OBS_LEN)))
    assert lengths
    assert lengths[0] >= OBS_LEN


def test_an_accepted_trace_updates_that_agent_and_solo_leaves_the_group():
    group = [build_net(seed) for seed in range(4)]
    solo = build_net(5)
    group_tape = Tape()
    solo_tape = Tape()
    group_tape.note(1, 0, _trace())
    saved = _snapshot(group)
    solo_saved = _snapshot((solo,))
    assert step(group[1], group_tape, agent=1, level=0) is True
    assert not _same(group[1], saved[1])
    for index in (0, 2, 3):
        assert _same(group[index], saved[index])
    held = _snapshot(group)
    assert step(solo, solo_tape, agent=0, level=0) is False
    assert _same(solo, solo_saved[0])
    for net, before in zip(group, held, strict=True):
        assert _same(net, before)


def test_four_copies_of_one_exercise_update_the_group():
    nets = [build_net(seed) for seed in range(4)]
    tape = Tape()
    trace = _trace()
    for agent in range(4):
        tape.note(agent, 0, trace)
    saved = _snapshot(nets)
    for agent, net in enumerate(nets):
        assert step(net, tape, agent=agent, level=0) is True
    for net, before in zip(nets, saved, strict=True):
        assert not _same(net, before)


def test_an_empty_tape_keeps_weights_bitwise():
    net = build_net(0)
    saved = _snapshot((net,))
    assert step(net, Tape(), agent=0, level=0) is False
    assert _same(net, saved[0])


def test_the_batch_is_four_sequences(monkeypatch):
    assert BATCH == 4
    assert LR == 3e-4
    assert GRAD_CLIP == 1.0
    net = build_net(0)
    seen: list[int] = []
    clips: list[float] = []
    real = net.forward_prefix
    original = torch.nn.utils.clip_grad_norm_

    def spy(token_ids):
        seen.append(int(token_ids.shape[0]))
        return real(token_ids)

    def clip(parameters, max_norm, *args, **kwargs):
        clips.append(float(max_norm))
        return original(parameters, max_norm, *args, **kwargs)

    net.forward_prefix = spy
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    tape = Tape()
    tape.note(0, 0, _trace())
    assert step(net, tape, agent=0, level=0) is True
    assert seen[0] == BATCH
    assert clips == [GRAD_CLIP]
    opt = net.optimizer
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["lr"] == LR


def test_the_holdout_evaluation_does_not_call_the_step(monkeypatch):
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append("krok")
        return False

    monkeypatch.setattr("kanon.v2.learn.step", spy)
    tasks = tuple(range(50))
    score = score_set(
        50,
        tasks,
        floor=9,
        answers=(lambda task: True, lambda task: True, lambda task: True),
        step=lambda: calls.append("ewaluacja"),
    )
    assert score.counted is True
    assert score.gap == 0
    assert calls == []


def _trace() -> tuple[int, ...]:
    return tuple(index % 10 for index in range(OBS_LEN + 4))


def _snapshot(nets):
    return tuple(
        tuple(param.detach().clone() for param in net.parameters()) for net in nets
    )


def _same(net, rows) -> bool:
    return all(torch.equal(param, saved) for param, saved in zip(net.parameters(), rows, strict=True))
