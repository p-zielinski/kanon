"""Sześć dekoderów v2. Okno pozycji ma 256 i jest wybierane w konstruktorze.

Warstwy 4, d_model 128, 4 głowy, d_ff 512, dropout 0. Ziarna 0, 1, 2, 3
to grupa, ziarno 4 jest zamrożone, ziarno 5 jest solo. Forward prefiksu
liczy się raz i wchodzi do produkcji następnego tokenu.
"""

from __future__ import annotations

import torch
from torch import nn

from kanon.v2.vocab import VOCAB

N_LAYERS = 4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
DROPOUT = 0.0
MAX_LEN = 256
SEEDS = (0, 1, 2, 3, 4, 5)
FROZEN_SEED = 4
SOLO_SEED = 5


class Decoder(nn.Module):
    """Dekoder przyczynowy. Logity są nad słownikiem v2."""

    def __init__(self) -> None:
        super().__init__()
        self.d_model = D_MODEL
        self.n_heads = N_HEADS
        self.d_ff = D_FF
        self.dropout = DROPOUT
        self.embed = nn.Embedding(len(VOCAB), D_MODEL)
        self.position = nn.Embedding(MAX_LEN, D_MODEL)
        block = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.core = nn.TransformerEncoder(block, N_LAYERS, enable_nested_tensor=False)
        self.layers = self.core.layers
        self.norm = nn.LayerNorm(D_MODEL)
        self.output = nn.Linear(D_MODEL, len(VOCAB), bias=False)
        self.seed = -1

    def forward_prefix(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Ukryty stan prefiksu. Produkcja bierze ten tensor."""
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token ids")
        time = token_ids.shape[1]
        if time == 0 or time > MAX_LEN:
            raise ValueError("length")
        positions = torch.arange(time, device=token_ids.device)
        hidden = self.embed(token_ids) + self.position(positions)
        hidden = self.core(hidden, mask=_causal_mask(time, hidden.device, hidden.dtype))
        return self.norm(hidden)

    def distribution(self, prefix: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """Rozkład następnego tokenu po prefiksie. Poza maską jest zero."""
        if prefix.ndim != 3 or prefix.shape[-1] != self.d_model:
            raise ValueError("prefix")
        if prefix.shape[1] == 0 or prefix.shape[1] > MAX_LEN:
            raise ValueError("length")
        scores = self.output(prefix[:, -1, :])
        mask = _align_mask(legal, scores.shape[0], scores.shape[-1], scores.device)
        hidden = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(hidden, dim=-1)
        return torch.where(mask, probs, torch.zeros_like(probs))


def build_net(seed: int) -> Decoder:
    if type(seed) is not int or seed not in SEEDS:
        raise ValueError("seed")
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        net = Decoder()
    finally:
        torch.set_rng_state(state)
    net.seed = seed
    if seed == FROZEN_SEED:
        net.requires_grad_(False)
    return net


def build_nets() -> tuple[Decoder, Decoder, Decoder, Decoder, Decoder, Decoder]:
    return tuple(build_net(seed) for seed in SEEDS)


def _causal_mask(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    blocked = torch.full((size, size), float("-inf"), device=device, dtype=dtype)
    return torch.triu(blocked, diagonal=1)


def _align_mask(legal: torch.Tensor, batch: int, vocab: int, device: torch.device) -> torch.Tensor:
    if legal.dtype != torch.bool:
        raise ValueError("mask")
    mask = legal.to(device=device)
    if mask.shape == (vocab,):
        return mask.view(1, vocab).expand(batch, vocab)
    if mask.shape == (batch, vocab):
        return mask
    raise ValueError("mask")
