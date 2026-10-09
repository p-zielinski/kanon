from pathlib import Path

import pytest
import torch

from kanon.model import (
    D_FF,
    D_MODEL,
    DROPOUT,
    MAX_LEN,
    N_HEADS,
    N_LAYERS,
    VOCAB,
    build_agent,
    build_agents,
    read_agents,
    write_agents,
)

_BANNED = ("load", "from_pretrained", "huggingface", "download")


def test_four_decoders_match_the_locked_shape():
    agents = build_agents()
    assert len(agents) == 4
    agent = agents[0]
    assert len(agent.layers) == N_LAYERS == 4
    assert agent.d_model == D_MODEL == 128
    assert agent.n_heads == N_HEADS == 4
    assert agent.d_ff == D_FF == 512
    assert agent.dropout == DROPOUT == 0
    assert len(VOCAB) == len(set(VOCAB))
    assert "f0" in VOCAB and "f31" in VOCAB and "f32" not in VOCAB
    assert not torch.equal(_flat(agents[0]), _flat(agents[1]))
    for left, right in zip(agents, agents[1:]):
        assert not torch.equal(_flat(left), _flat(right))
    assert torch.equal(_flat(agents[0]), _flat(build_agent(0)))


def test_illegal_production_has_probability_zero():
    agent = build_agent(0)
    agent.eval()
    ids = torch.zeros(2, 3, dtype=torch.long)
    only = torch.zeros(len(VOCAB), dtype=torch.bool)
    only[4] = True
    peaked = agent.distribution(ids, only)
    expected = torch.zeros_like(peaked)
    expected[:, 4] = 1
    assert torch.equal(peaked, expected)

    legal = torch.zeros(len(VOCAB), dtype=torch.bool)
    legal[1] = True
    legal[2] = True
    mixed = agent.distribution(ids, legal)
    assert torch.equal(mixed[:, ~legal], torch.zeros(2, int((~legal).sum())))
    assert torch.allclose(mixed.sum(dim=-1), torch.ones(2))

    rows = legal.unsqueeze(0).repeat(2, 1)
    rows[0, 1] = False
    rows[1, 2] = False
    per_row = agent.distribution(ids, rows)
    assert per_row[0, 1].item() == 0
    assert per_row[1, 2].item() == 0
    assert per_row[0, 2].item() == 1
    assert per_row[1, 1].item() == 1


def test_past_logits_ignore_future_tokens():
    agent = build_agent(2)
    agent.eval()
    left = torch.randint(0, len(VOCAB), (1, 5))
    right = left.clone()
    right[0, -1] = (right[0, -1] + 1) % len(VOCAB)
    with torch.no_grad():
        earlier = agent.logits(left)
        later = agent.logits(right)
    assert torch.equal(earlier[:, :-1, :], later[:, :-1, :])


def test_weights_roundtrip_under_runs(tmp_path):
    agents = build_agents()
    directory = tmp_path / "runs"
    write_agents(directory, agents)
    restored = read_agents(directory)
    assert len(list(directory.glob("agent-*.bin"))) == 4
    for left, right in zip(agents, restored, strict=True):
        for original, copied in zip(left.parameters(), right.parameters(), strict=True):
            assert torch.equal(original, copied)


def test_source_does_not_fetch_weights():
    source = Path(__file__).resolve().parents[1].joinpath("src/kanon/model.py").read_text()
    for banned in _BANNED:
        assert banned not in source


def test_rejects_a_sequence_past_the_generation_limit():
    agent = build_agent(0)
    with pytest.raises(ValueError):
        agent.logits(torch.zeros(1, MAX_LEN + 1, dtype=torch.long))


def _flat(agent) -> torch.Tensor:
    return torch.cat([param.detach().reshape(-1) for param in agent.parameters()])
