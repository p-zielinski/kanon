"""Cztery dekodery od losowych ziaren.

Warstwy 4, d_model 128, 4 głowy, d_ff 512, dropout 0. Ziarna 0, 1, 2, 3.
Na kroku dekodowania maska zostawia dodatnie prawdopodobieństwo tylko
legalnym produkcjom. Zapis wag jest lokalny; moduł nie sięga po cudze pliki.
"""

from __future__ import annotations

import ast
import json
import struct
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from kanon.language import MESSAGE_HEADS, TOKENS

N_LAYERS = 4
D_MODEL = 128
N_HEADS = 4
D_FF = 512
DROPOUT = 0.0
MAX_LEN = 64
SEEDS = (0, 1, 2, 3)
VOCAB: tuple[str, ...] = (*TOKENS, *MESSAGE_HEADS)
TOKEN_ID = {token: index for index, token in enumerate(VOCAB)}
_MAGIC = b"KNWT1"


class Decoder(nn.Module):
    """Dekoder przyczynowy. Logity są nad `VOCAB`."""

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

    def logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token ids")
        time = token_ids.shape[1]
        if time == 0 or time > MAX_LEN:
            raise ValueError("length")
        positions = torch.arange(time, device=token_ids.device)
        hidden = self.embed(token_ids) + self.position(positions)
        hidden = self.core(hidden, mask=_causal_mask(time, hidden.device, hidden.dtype))
        return self.output(self.norm(hidden))

    def distribution(self, token_ids: torch.Tensor, legal: torch.Tensor) -> torch.Tensor:
        """Prawdopodobieństwa następnej produkcji. Poza maską są zerem."""
        scores = self.logits(token_ids)[:, -1, :]
        mask = _align_mask(legal, scores.shape[0], scores.shape[-1], scores.device)
        hidden = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(hidden, dim=-1)
        return torch.where(mask, probs, torch.zeros_like(probs))


def build_agent(seed: int) -> Decoder:
    if type(seed) is not int:
        raise ValueError("seed")
    state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        return Decoder()
    finally:
        torch.set_rng_state(state)


def build_agents() -> tuple[Decoder, Decoder, Decoder, Decoder]:
    return tuple(build_agent(seed) for seed in SEEDS)


def write_agents(directory: Path | str, agents: Sequence[Decoder]) -> None:
    if len(agents) != len(SEEDS):
        raise ValueError("agents")
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for index, agent in enumerate(agents):
        _write_module(root / f"agent-{index}.bin", agent)


def read_agents(directory: Path | str) -> tuple[Decoder, Decoder, Decoder, Decoder]:
    root = Path(directory)
    agents = []
    for index in range(len(SEEDS)):
        path = root / f"agent-{index}.bin"
        if not path.is_file():
            raise ValueError("weights")
        agent = build_agent(0)
        _read_module(path, agent)
        agents.append(agent)
    return agents[0], agents[1], agents[2], agents[3]


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


def _write_module(path: Path, module: nn.Module) -> None:
    meta = []
    blobs: list[bytes] = []
    cursor = 0
    for name, tensor in module.state_dict().items():
        data = tensor.detach().cpu().contiguous()
        if data.storage_offset() != 0:
            data = data.clone()
        raw = bytes(data.untyped_storage())[: data.nbytes]
        meta.append(
            {
                "name": name,
                "dtype": str(data.dtype).removeprefix("torch."),
                "shape": list(data.shape),
                "start": cursor,
                "size": len(raw),
            }
        )
        blobs.append(raw)
        cursor += len(raw)
    header = json.dumps(meta).encode("utf-8")
    path.write_bytes(_MAGIC + struct.pack("<I", len(header)) + header + b"".join(blobs))


def _read_module(path: Path, module: nn.Module) -> None:
    data = path.read_bytes()
    if not data.startswith(_MAGIC) or len(data) < len(_MAGIC) + 4:
        raise ValueError("weights")
    (header_len,) = struct.unpack_from("<I", data, len(_MAGIC))
    start = len(_MAGIC) + 4
    meta = ast.literal_eval(data[start : start + header_len].decode("utf-8"))
    body = data[start + header_len :]
    stored: dict[str, torch.Tensor] = {}
    for item in meta:
        blob = body[item["start"] : item["start"] + item["size"]]
        stored[item["name"]] = _tensor(blob, item["dtype"], item["shape"])
    live = module.state_dict()
    if set(stored) != set(live):
        raise ValueError("weights")
    for name, tensor in stored.items():
        live[name].copy_(tensor)


def _tensor(blob: bytes, dtype_name: str, shape: list[int]) -> torch.Tensor:
    try:
        raw = torch.frombuffer(bytearray(blob), dtype=getattr(torch, dtype_name))
        return raw.reshape(tuple(shape)).clone()
    except (ValueError, RuntimeError, AttributeError) as exc:
        raise ValueError("weights") from exc
