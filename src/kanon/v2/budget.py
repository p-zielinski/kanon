"""Budżet komunikatu. Zwykła liczba, liczona z piętra albo z ćwiczenia."""

from __future__ import annotations


def term_length(expr: object) -> int:
    """Długość termu w tokenach. Identyfikator półki jest jednym tokenem."""
    from kanon.v2.tactics import expr_tokens

    return len(expr_tokens(expr))


def message_budget(floor: int, *, exercise: bool = False) -> int:
    """Tokeny komunikatu, które jeszcze wchodzą na dane piętro.

    Piętro 1 ma 24, piętra 2–3 mają 48, piętra 4–12 mają 80.
    Ćwiczenie ma 80. Dłuższy term odpada przed reduktorem.
    """
    if type(floor) is not int or floor < 1:
        raise ValueError("floor")
    if exercise:
        return 80
    if floor == 1:
        return 24
    if floor <= 3:
        return 48
    if floor <= 12:
        return 80
    raise ValueError("floor")
