"""Krok v2. AdamW, jeden agent naraz, entropia krzyżowa następnego tokenu.

Ciąg to prefiks 176 i komunikat. Produkcja startuje z tego prefiksu.
Pusta taśma nie rusza wag. Ewaluacja zbioru 50 tego kroku nie woła.
`src/kanon/update.py` zostaje krokiem v1.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from kanon.v2.model import MAX_LEN, Decoder
from kanon.v2.play import Tape
from kanon.v2.vocab import VOCAB
from kanon.v2.window import OBS_LEN, TOKEN_ID

LR = 3e-4
BATCH = 4
GRAD_CLIP = 1.0

_PAD = TOKEN_ID["[PAD]"]


def step(net: Decoder, tape: Tape, *, agent: int, level: int) -> bool:
    """Jeden agent. Batch jest stałą `BATCH`. Pusta taśma zostawia wagi."""
    rows = [row for row in tape.traces(agent, level) if len(row) >= 2]
    if not rows:
        return False
    ids, lengths = _batch(rows)
    net.train()
    optimizer = _optimizer(net)
    optimizer.zero_grad()
    hidden = net.forward_prefix(ids)
    logits = net.output(hidden)
    targets = ids[:, 1:].clone()
    for index, length in enumerate(lengths):
        if length < ids.shape[1]:
            targets[index, length - 1 :] = -100
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100,
    )
    if not torch.isfinite(loss):
        raise RuntimeError("loss")
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
    optimizer.step()
    net.optimizer = optimizer
    return True


def produce(net: Decoder, prefix: Sequence[int]) -> tuple[int, ...]:
    """Pierwsze wołanie sieci dostaje cały prefiks obserwacji."""
    if len(prefix) < OBS_LEN:
        raise ValueError("prefix")
    ids = torch.tensor([list(prefix)], dtype=torch.long)
    hidden = net.forward_prefix(ids)
    legal = torch.ones(len(VOCAB), dtype=torch.bool)
    probs = net.distribution(hidden, legal)
    return (int(torch.argmax(probs, dim=-1).item()),)


def _batch(rows: Sequence[Sequence[int]]) -> tuple[torch.Tensor, tuple[int, ...]]:
    chosen = [tuple(rows[index % len(rows)]) for index in range(BATCH)]
    width = min(MAX_LEN, max(len(row) for row in chosen))
    clipped = [row[:width] for row in chosen]
    padded = [list(row) + [_PAD] * (width - len(row)) for row in clipped]
    lengths = tuple(len(row) for row in clipped)
    return torch.tensor(padded, dtype=torch.long), lengths


def _optimizer(net: Decoder) -> torch.optim.AdamW:
    """Optimizer siedzi na sieci. Procesowy słownik nie przeżywa restartu."""
    found = getattr(net, "optimizer", None)
    if isinstance(found, torch.optim.AdamW):
        return found
    found = torch.optim.AdamW(net.parameters(), lr=LR)
    net.optimizer = found
    return found
