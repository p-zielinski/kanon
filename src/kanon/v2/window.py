"""Okno obserwacji v2.

176 tokenów: 48 na ciało albo tezę, pięć par publicznych, dwie nazwy
z półki po dwie pary. Krótszy odcinek dopełnia `[PAD]`. `(num k)` dla
k od 0 do 10 to cztery tokeny `( num k )`. Komunikat zaczyna się na 176.
"""

from __future__ import annotations

from collections.abc import Sequence

from kanon.language import SLOTS
from kanon.v2.vocab import VOCAB

OBS_LEN = 176
MSG_AT = 176
BODY_LEN = 48
PUBLIC_PAIRS = 5
NUMS_PER_PAIR = 3
NUM_LEN = 4
PUBLIC_LEN = PUBLIC_PAIRS * NUMS_PER_PAIR * NUM_LEN
NAME_COUNT = 2
PAIRS_PER_NAME = 2
NAME_HEAD = 10
NAME_LEN = NAME_HEAD + PAIRS_PER_NAME * NUMS_PER_PAIR * NUM_LEN
NAMES_LEN = NAME_COUNT * NAME_LEN

TOKEN_ID = {token: index for index, token in enumerate(VOCAB)}
_PAD = "[PAD]"

if BODY_LEN + PUBLIC_LEN + NAMES_LEN != OBS_LEN:
    raise RuntimeError("window")


def observation_prefix(
    pairs: Sequence[Sequence[int]],
    *,
    body: Sequence[str] = (),
    names: Sequence[tuple[str, Sequence[Sequence[int]]]] = (),
) -> tuple[int, ...]:
    """Identyfikatory prefiksu. Długość 176."""
    if len(pairs) != PUBLIC_PAIRS:
        raise ValueError("pairs")
    if len(names) > NAME_COUNT:
        raise ValueError("names")
    tokens: list[str] = []
    tokens.extend(_fit(body, BODY_LEN))
    for pair in pairs:
        tokens.extend(_pair(pair))
    name_tokens: list[str] = []
    for name in names:
        name_tokens.extend(_name(name))
    if len(name_tokens) > NAMES_LEN:
        raise ValueError("names")
    name_tokens.extend([_PAD] * (NAMES_LEN - len(name_tokens)))
    tokens.extend(name_tokens)
    return tuple(TOKEN_ID[token] for token in tokens)


def place_message(prefix: Sequence[int], message: Sequence[int]) -> tuple[int, ...]:
    """Dokleja komunikat. Pierwszy jego token stoi na indeksie 176."""
    if len(prefix) != OBS_LEN or len(message) == 0:
        raise ValueError("message")
    return tuple(prefix) + tuple(message)


def _fit(tokens: Sequence[str], length: int) -> tuple[str, ...]:
    if len(tokens) > length:
        raise ValueError("body")
    return tuple(tokens) + (_PAD,) * (length - len(tokens))


def _pair(numbers: Sequence[int]) -> tuple[str, ...]:
    if not 1 <= len(numbers) <= NUMS_PER_PAIR:
        raise ValueError("pair")
    slots: list[int | None] = list(numbers)
    slots.extend([None] * (NUMS_PER_PAIR - len(numbers)))
    out: list[str] = []
    for value in slots:
        out.extend(_num(value))
    return tuple(out)


def _num(value: int | None) -> tuple[str, ...]:
    if value is None:
        return (_PAD,) * NUM_LEN
    if type(value) is not int or not 0 <= value <= 10:
        raise ValueError("num")
    return ("(", "num", str(value), ")")


def _name(name: tuple[str, Sequence[Sequence[int]]]) -> tuple[str, ...]:
    slot, pairs = name
    if slot not in SLOTS or len(pairs) != PAIRS_PER_NAME:
        raise ValueError("name")
    head = (slot,) + (_PAD,) * (NAME_HEAD - 1)
    body: list[str] = []
    for pair in pairs:
        body.extend(_pair(pair))
    return head + tuple(body)
