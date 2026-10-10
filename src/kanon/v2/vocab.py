"""Słownik sieci v2 w dniu zero.

Krotka zaczyna się od tokenów języka i głów komunikatów ról. Dalej są
cyfry „4”…„10”, `[PAD]` i głowy taktyk. „0”…„3” zostają jednym tokenem
liczby w `(num k)` i zarazem indeksem zmiennej albo arnością.
"""

from __future__ import annotations

from kanon.language import MESSAGE_HEADS, TOKENS

_NUMERALS = tuple(str(k) for k in range(4, 11))
_TACTICS = ("eval", "rewrite", "base", "step", "claim", "eq", "dom", "ih")

VOCAB: tuple[str, ...] = (
    *TOKENS,
    *MESSAGE_HEADS,
    *_NUMERALS,
    "[PAD]",
    *_TACTICS,
)
