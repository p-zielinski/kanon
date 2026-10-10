"""Ziarno zadania i komunikatu, z numerem rundy.

v1 brało zadanie z `task_seed(seed, index)`, a komunikat łamacza z
`seed + 20 + próba`. Numer rundy tam nie wchodził, więc 1218 rund
powtarzało jeden cel autora i jeden `(abstain …)`.
"""

from __future__ import annotations


def round_seed(seed: int, round_no: int, attempt: int) -> int:
    """Mieszanka ziarna, numeru rundy i indeksu próby.

    Generator bierze ten wynik jako ziarno zadania, sampler jako ziarno
    komunikatu łamacza. Dwie rundy dają dwa ziarna zadania i dwa ziarna
    komunikatu. Dwie próby tej samej rundy też.
    """
    if type(seed) is not int or type(round_no) is not int or type(attempt) is not int:
        raise ValueError("seed")
    if round_no < 0 or attempt < 0:
        raise ValueError("seed")
    wide = 1_000_003
    return (seed + 1) * wide * wide + round_no * wide + attempt
