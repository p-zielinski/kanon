"""Rampa warsztatu. Osobna oś od frontery.

Ostatnie 50 świeżych ćwiczeń grupy na poziomie podnosi je o jeden, gdy
trafień jest co najmniej 30. Poniżej tego, także przy 0,3, poziom stoi
i w dół nie schodzi. Poziomy 4–6 czekają na pulę taktyk. Sufit to 6.
W schemacie ćwiczenie zostaje na najniższym odblokowanym poziomie.
"""

from __future__ import annotations

from kanon.v2.schema import Schema
from kanon.v2.shelf import Shelf
from kanon.v2.workshop import WORKSHOP_CAP, ramp_ceiling

WINDOW = 50
PROMOTE_AT = 30


class Ramp:
    """Poziom grupy. Każdy stopień ma własne ostatnie 50 ćwiczeń."""

    def __init__(self) -> None:
        self.level = 0
        self._rows: dict[int, list[bool]] = {}

    def note(self, hit: bool, shelf: Shelf, schema: Schema | None = None) -> int:
        """Dopisuje ćwiczenie grupy. Frontery nie rusza."""
        del schema
        row = self._rows.setdefault(self.level, [])
        row.append(bool(hit))
        if len(row) > WINDOW:
            del row[0]
        if len(row) < WINDOW or sum(row) < PROMOTE_AT:
            return self.level
        if self.level >= min(WORKSHOP_CAP, ramp_ceiling(shelf)):
            return self.level
        self.level += 1
        return self.level

    def unlocked(self) -> tuple[int, ...]:
        return tuple(range(self.level + 1))

    def package_level(self) -> int:
        """Najniższy odblokowany. Ćwiczenie po paczce zostaje tutaj."""
        return min(self.unlocked())
