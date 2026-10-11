"""Jedna wypowiedź ćwiczenia. Przy ε = 1 mówi sampler, bez sieci.

Do poziomu 3 komunikat jest złożeniem dwóch kart z okna. Od poziomu 4
jest taktyką: teza leży w obserwacji, a ta funkcja jej nie odczytuje
z faktu. Sieć dochodzi dopiero poniżej 1 i startuje z prefiksu 176.
Ukrytego termu ta funkcja nie czyta. `kanon.sample.speak` zostaje przy v1.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import torch

from kanon.v2.budget import message_budget
from kanon.v2.model import Decoder
from kanon.v2.seed import round_seed
from kanon.v2.shelf import Shelf
from kanon.v2.vocab import VOCAB
from kanon.v2.window import OBS_LEN, TOKEN_ID
from kanon.v2.workshop import Task

_LIMIT = message_budget(12, exercise=True)
_SLOTS = ("f0", "f1")


def utter(
    task: Task,
    shelf: Shelf,
    *,
    seed: int,
    round_no: int,
    attempt: int,
    epsilon: float,
    net: Decoder | None = None,
) -> str:
    """Jeden komunikat. ε >= 1 nie woła `forward_prefix`."""
    prefix = task.observation.tokens
    if len(prefix) != OBS_LEN:
        raise ValueError("prefix")
    if len(task.cards) != 2:
        raise ValueError("cards")
    if task.level >= 4:
        return _utter_proof(task, seed, round_no, attempt, epsilon, net, prefix)
    arities = (
        shelf.entry(task.cards[0]).arity,
        shelf.entry(task.cards[1]).arity,
    )
    rng = random.Random(round_seed(seed, round_no, attempt))
    use_net = net is not None and epsilon < 1
    stack: list[tuple] = [("open-call", task.level + 1)]
    spoken: list[str] = []
    hidden = None
    if use_net:
        hidden = _forward(net, prefix)
    while len(spoken) < _LIMIT:
        legal = _options(stack)
        if legal is None:
            break
        token = _choose(legal, rng, epsilon, net if use_net else None, hidden)
        spoken.append(token)
        stack = _advance(stack, token, arities)
        if use_net and stack:
            hidden = _forward(net, prefix + tuple(TOKEN_ID[item] for item in spoken))
    return " ".join(spoken)


def _utter_proof(
    task: Task,
    seed: int,
    round_no: int,
    attempt: int,
    epsilon: float,
    net: Decoder | None,
    prefix: tuple[int, ...],
) -> str:
    """Jedna taktyka na poziomach 4–5, łańcuch 2..8 na poziomie 6."""
    rng = random.Random(round_seed(seed, round_no, attempt))
    count = 1 if task.level < 6 else rng.randrange(2, 9)
    stack: list[tuple] = [("open",) for _ in range(count)]
    stack.reverse()
    use_net = net is not None and epsilon < 1
    spoken: list[str] = []
    hidden = _forward(net, prefix) if use_net else None
    while stack and len(spoken) < _LIMIT:
        legal = _proof_options(stack)
        if not legal:
            break
        token = _choose(legal, rng, epsilon, net if use_net else None, hidden)
        spoken.append(token)
        stack = _advance_proof(stack, token)
        if use_net and stack:
            hidden = _forward(net, prefix + tuple(TOKEN_ID[item] for item in spoken))
    return " ".join(spoken)


def _proof_options(stack: list[tuple]) -> list[str]:
    if not stack:
        return []
    kind = stack[-1][0]
    if kind == "open":
        return ["("]
    if kind == "head":
        return ["eval", "rewrite", "base", "step", "dom"]
    if kind == "lo":
        return [str(index) for index in range(10)]
    if kind == "hi":
        return [str(index) for index in range(stack[-1][1] + 1, 11)]
    if kind == "name":
        return ["f0", "f1"]
    if kind == "word":
        return [stack[-1][1]]
    raise RuntimeError(kind)


def _advance_proof(stack: list[tuple], token: str) -> list[tuple]:
    stack = list(stack)
    job = stack.pop()
    kind = job[0]
    if kind == "open" and token == "(":
        stack.append(("head",))
        return stack
    if kind == "head" and token == "eval":
        stack.append(("word", ")"))
        return stack
    if kind == "head" and token == "base":
        stack.append(("word", ")"))
        return stack
    if kind == "head" and token == "rewrite":
        stack.append(("word", ")"))
        stack.append(("name",))
        return stack
    if kind == "head" and token == "step":
        for word in (")", ")", "name", "rewrite", "(", ")", "ih", "("):
            if word == "name":
                stack.append(("name",))
            else:
                stack.append(("word", word))
        return stack
    if kind == "head" and token == "dom":
        stack.append(("lo",))
        return stack
    if kind == "lo" and token in _proof_options([job]):
        stack.append(("hi", int(token)))
        return stack
    if kind == "hi" and token in _proof_options([job]):
        for word in (")", "eval", "(", ")"):
            stack.append(("word", word))
        return stack
    if kind == "name" and token in ("f0", "f1"):
        return stack
    if kind == "word" and token == job[1]:
        return stack
    raise ValueError("token")


def _forward(net: Decoder, prefix: Sequence[int]) -> torch.Tensor:
    with torch.no_grad():
        return net.forward_prefix(torch.tensor([list(prefix)], dtype=torch.long))


def _choose(legal, rng, epsilon: float, net, hidden) -> str:
    if net is None or hidden is None or rng.random() < epsilon:
        return rng.choice(legal)
    mask = torch.zeros(len(VOCAB), dtype=torch.bool)
    for token in legal:
        mask[TOKEN_ID[token]] = True
    with torch.no_grad():
        probs = net.distribution(hidden, mask)[0]
    draw = rng.random()
    acc = 0.0
    for token in legal:
        acc += float(probs[TOKEN_ID[token]])
        if draw <= acc:
            return token
    return legal[-1]


def _options(stack: list[tuple]) -> list[str] | None:
    if not stack:
        return None
    kind = stack[-1][0]
    if kind in ("open-call", "open-leaf"):
        return ["("]
    if kind == "call-word":
        return ["call"]
    if kind == "slot":
        return list(_SLOTS)
    if kind == "leaf-kind":
        return ["var", "num"]
    if kind == "leaf-digit":
        name, limit = stack[-1][1], stack[-1][2]
        if name == "var":
            return [str(index) for index in range(limit)]
        return [str(index) for index in range(11)]
    if kind in ("close-leaf", "close-call"):
        return [")"]
    raise RuntimeError(kind)


def _advance(stack: list[tuple], token: str, arities: tuple[int, int]) -> list[tuple]:
    stack = list(stack)
    job = stack.pop()
    kind = job[0]
    if kind == "open-call" and token == "(":
        stack.append(("call-word", job[1]))
        return stack
    if kind == "call-word" and token == "call":
        stack.append(("slot", job[1]))
        return stack
    if kind == "slot" and token in _SLOTS:
        remaining = job[1]
        arity = arities[_SLOTS.index(token)]
        stack.append(("close-call",))
        if remaining > 1:
            jobs = [("open-call", remaining - 1)]
            jobs.extend(("open-leaf", arity) for _ in range(arity - 1))
        else:
            jobs = [("open-leaf", arity) for _ in range(arity)]
        stack.extend(reversed(jobs))
        return stack
    if kind == "open-leaf" and token == "(":
        stack.append(("leaf-kind", job[1]))
        return stack
    if kind == "leaf-kind" and token in ("var", "num"):
        stack.append(("leaf-digit", token, job[1]))
        return stack
    if kind == "leaf-digit" and token in _options([job]):
        stack.append(("close-leaf",))
        return stack
    if kind == "close-leaf" and token == ")":
        return stack
    if kind == "close-call" and token == ")":
        return stack
    raise ValueError("token")
