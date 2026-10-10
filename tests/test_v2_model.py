from pathlib import Path

import pytest
import torch

from kanon.model import MAX_LEN as V1_MAX_LEN
from kanon.v2.model import (
    D_FF,
    D_MODEL,
    DROPOUT,
    FROZEN_SEED,
    MAX_LEN,
    N_HEADS,
    N_LAYERS,
    SOLO_SEED,
    VOCAB,
    build_net,
    build_nets,
)
from kanon.v2.window import (
    BODY_LEN,
    MSG_AT,
    OBS_LEN,
    observation_prefix,
    place_message,
)

_BANNED = ("load", "from_pretrained", "huggingface", "download", "payload")


def test_six_nets_keep_the_locked_shape_and_seeds():
    nets = build_nets()
    assert len(nets) == 6
    assert [net.seed for net in nets] == [0, 1, 2, 3, 4, 5]
    assert FROZEN_SEED == 4 and SOLO_SEED == 5
    frozen = nets[FROZEN_SEED]
    solo = nets[SOLO_SEED]
    assert not any(param.requires_grad for param in frozen.parameters())
    assert any(param.requires_grad for param in solo.parameters())
    assert any(param.requires_grad for param in nets[0].parameters())
    net = nets[0]
    assert len(net.layers) == N_LAYERS == 4
    assert net.d_model == D_MODEL == 128
    assert net.n_heads == N_HEADS == 4
    assert net.d_ff == D_FF == 512
    assert net.dropout == DROPOUT == 0
    assert net.position.num_embeddings == MAX_LEN == 256
    assert net.position.embedding_dim == D_MODEL
    assert not torch.equal(_flat(nets[0]), _flat(nets[1]))
    assert torch.equal(_flat(nets[0]), _flat(build_net(0)))
    with pytest.raises(ValueError):
        build_net(6)


def test_time_257_is_rejected_and_v1_stays_at_64():
    net = build_net(0)
    with pytest.raises(ValueError):
        net.forward_prefix(torch.zeros(1, MAX_LEN + 1, dtype=torch.long))
    assert V1_MAX_LEN == 64


def test_source_does_not_fetch_weights():
    source = Path(__file__).resolve().parents[1].joinpath("src/kanon/v2/model.py").read_text()
    for banned in _BANNED:
        assert banned not in source


def test_pair_rows_change_the_prefix_and_the_message_starts_at_176():
    left = observation_prefix(_row(0))
    right = observation_prefix(_row(1))
    assert left != right
    assert len(left) == len(right) == OBS_LEN == 176
    message = (VOCAB.index("("),)
    placed = place_message(left, message)
    assert placed[MSG_AT] == message[0]
    assert MSG_AT == 176
    ten = observation_prefix(_row(0)[:-1] + ((10, 1),))
    assert _num_span(ten, 10) == ("(", "num", "10", ")")
    assert "succ" not in _num_span(ten, 10)


def test_prefix_forward_changes_the_next_token_and_the_mask_zeroes_the_rest():
    net = build_net(0)
    net.eval()
    left = _state(net, observation_prefix(_row(0)))
    right = _state(net, observation_prefix(_row(3)))
    legal = torch.ones(len(VOCAB), dtype=torch.bool)
    first = net.distribution(left, legal)
    second = net.distribution(right, legal)
    assert first.shape == second.shape == (1, len(VOCAB))
    assert not torch.equal(first, second)
    only = torch.zeros(len(VOCAB), dtype=torch.bool)
    only[4] = True
    peaked = net.distribution(left, only)
    assert peaked[0, 4].item() == 1
    assert torch.count_nonzero(peaked[0, ~only]).item() == 0


def test_body_is_48_pads_and_a_name_sits_after_the_pairs():
    bare = observation_prefix(_row(0))
    named = observation_prefix(
        _row(0),
        names=(("f0", ((1, 2), (3, 4))), ("f1", ((0, 1), (2, 3)))),
    )
    pad = VOCAB.index("[PAD]")
    assert bare[:BODY_LEN] == (pad,) * BODY_LEN
    assert BODY_LEN == 48
    assert named[:108] == bare[:108]
    assert VOCAB.index("f0") in named[108:]
    assert VOCAB.index("f0") not in named[:108]


def _row(value: int) -> tuple[tuple[int, int], ...]:
    return tuple((value, value) for _ in range(5))


def _state(net, prefix: tuple[int, ...]) -> torch.Tensor:
    ids = torch.tensor([prefix], dtype=torch.long)
    return net.forward_prefix(ids)


def _flat(net) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in net.parameters()])


def _num_span(prefix: tuple[int, ...], value: int) -> tuple[str, ...]:
    wanted = tuple(VOCAB.index(token) for token in ("(", "num", str(value), ")"))
    for start in range(len(prefix) - 3):
        if prefix[start : start + 4] == wanted:
            return tuple(VOCAB[index] for index in wanted)
    raise AssertionError(value)
