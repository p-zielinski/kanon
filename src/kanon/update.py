"""Jeden krok AdamW na taśmie zaakceptowanych komunikatów.

Taśma trzyma finały agenta (`propose`, `use`, `solve`, `break`, `abstain`),
aktualny eksport kanonu i, dla łamacza, potwierdzone zbicia. Szkic i odrzut
nie wchodzą. Pusta taśma pomija krok: wagi zostają bitowo te same.

Ewaluacja pomiarowa tej funkcji nie woła.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kanon.canon import Canon, Grave
from kanon.language import (
    MAX_ARITY,
    Abstain,
    Break,
    DraftDef,
    DraftExpr,
    ParseError,
    Propose,
    Solve,
    Use,
    expand_num,
    parse_message,
    tokenize,
)
from kanon.model import MAX_LEN, TOKEN_ID, Decoder
from kanon.roles import Role, format_def, format_expr

LR = 3e-4
BATCH = 32
GRAD_CLIP = 1.0
_FINALS = (Propose, Use, Solve, Break, Abstain)


@dataclass(frozen=True, slots=True)
class Trace:
    agent: int
    role: Role
    text: str


class Tape:
    """Wspólna taśma. Próbka kroku zależy od agenta i roli, którą właśnie gra."""

    def __init__(self) -> None:
        self._traces: list[Trace] = []
        self._canon: tuple[str, ...] = ()
        self._breaks: tuple[str, ...] = ()
        self._optim: dict[int, torch.optim.AdamW] = {}

    @property
    def traces(self) -> tuple[Trace, ...]:
        return tuple(self._traces)

    def note(self, agent: int, role: Role, message: str, *, accepted: bool) -> None:
        _agent(agent)
        if not accepted or not isinstance(role, Role):
            return
        try:
            parsed = parse_message(message, arity=MAX_ARITY)
        except ParseError:
            return
        if isinstance(parsed, (DraftDef, DraftExpr)) or not isinstance(parsed, _FINALS):
            return
        self._traces.append(Trace(agent, role, message))

    def note_canon(self, canon: Canon) -> None:
        self._canon = tuple(format_def(entry.definition) for entry in canon.replay())

    def note_breaks(self, canon: Canon) -> None:
        self._breaks = tuple(
            _break_text(grave)
            for grave in canon.graves
            if grave.def_result != grave.target_result
        )

    def sources(self, agent: int, role: Role) -> tuple[str, ...]:
        _agent(agent)
        own = tuple(item.text for item in self._traces if item.agent == agent)
        breaks = self._breaks if role is Role.BREAKER else ()
        return own + self._canon + breaks

    def _optimizer(self, module: Decoder) -> torch.optim.AdamW:
        key = id(module)
        found = self._optim.get(key)
        if found is None:
            found = torch.optim.AdamW(module.parameters(), lr=LR)
            self._optim[key] = found
        return found


def trainable(text: str, canon: Canon) -> bool:
    """`(call slot ...)` jest ciągiem uczącym, gdy każdy wołany slot jest w kanonie."""
    if encode(text) is None:
        return False
    slots = {entry.definition.slot for entry in canon.replay()}
    tokens = tokenize(text)
    for index, token in enumerate(tokens):
        if token == "call":
            if index + 1 >= len(tokens) or tokens[index + 1] not in slots:
                return False
    return True


def encode(text: str) -> list[int] | None:
    try:
        tokens = tokenize(text)
    except ParseError:
        return None
    flat = _expand_nums(tokens)
    if flat is None:
        return None
    ids = [TOKEN_ID[token] for token in flat]
    if len(ids) < 2:
        return None
    return ids[:MAX_LEN]


def _expand_nums(tokens: list[str]) -> list[str] | None:
    """`(num k)` spoza słownika rozwija się do wieży `succ` wokół `(zero)`."""
    out: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            index + 3 < len(tokens)
            and tokens[index] == "("
            and tokens[index + 1] == "num"
            and tokens[index + 3] == ")"
            and tokens[index + 2] not in TOKEN_ID
        ):
            numeral = tokens[index + 2]
            if not numeral.isdigit() or (len(numeral) > 1 and numeral.startswith("0")):
                return None
            out.extend(tokenize(format_expr(expand_num(int(numeral)))))
            index += 4
            continue
        if tokens[index] not in TOKEN_ID:
            return None
        out.append(tokens[index])
        index += 1
    return out


def batch_tensor(tape: Tape, agent: int, role: Role) -> tuple[torch.Tensor, tuple[int, ...]] | None:
    encoded = [encode(text) for text in tape.sources(agent, role)]
    rows = [row for row in encoded if row is not None]
    if not rows:
        return None
    chosen = [rows[index % len(rows)] for index in range(BATCH)]
    width = max(len(row) for row in chosen)
    padded = [row + [0] * (width - len(row)) for row in chosen]
    lengths = tuple(len(row) for row in chosen)
    return torch.tensor(padded, dtype=torch.long), lengths


def step(agents: Sequence[Decoder], tape: Tape, *, agent: int, role: Role) -> bool:
    """Jeden krok albo pominięcie, gdy ten agent nie ma ciągów."""
    _agent(agent)
    if not isinstance(role, Role):
        raise ValueError("role")
    if agent >= len(agents):
        raise ValueError("agent")
    packed = batch_tensor(tape, agent, role)
    if packed is None:
        return False
    ids, lengths = packed
    module = agents[agent]
    module.train()
    optimizer = tape._optimizer(module)
    optimizer.zero_grad()
    logits = module.logits(ids)
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
    nn.utils.clip_grad_norm_(module.parameters(), GRAD_CLIP)
    optimizer.step()
    return True


def step_all(agents: Sequence[Decoder], tape: Tape, roles: Sequence[Role]) -> tuple[bool, ...]:
    if len(agents) != 4 or len(roles) != 4:
        raise ValueError("agents")
    return tuple(step(agents, tape, agent=index, role=roles[index]) for index in range(4))


def _break_text(grave: Grave) -> str:
    nums = " ".join(f"(num {value})" for value in grave.inputs)
    return f"(break {grave.slot} {nums})"


def _agent(agent: int) -> None:
    if type(agent) is not int or not 0 <= agent < 4:
        raise ValueError("agent")
