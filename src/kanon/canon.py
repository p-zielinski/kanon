"""Kanon procedur, cmentarz i ława.

Wpis przyjmuje definicję tylko razem z parami publicznymi, 16 parami
audytowymi i numerem rundy. `last_used_round` rusza się, gdy term
rozwiązania zawiera `(call slot ...)`. Po 12 rundach bez wywołania slot
wraca do puli i znika z eksportu powtórki. Co 4 rundy automat redukuje
najstarszy wpis na jego parach audytowych; rozjazd usuwa wpis.

Cmentarz trzyma definicję, wejście i oba wyniki. Eksport dla łamacza
ma wejście i fakt rozjazdu, bez termu targetu.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from kanon.language import SLOTS, Call, Def, Expr, If0, Pred, Self, Succ
from kanon.reduce import ReduceError, reduce_def
from kanon.tasks import IOPair

EXPIRE_AFTER = 12
BENCH_EVERY = 4
AUDIT_COUNT = 16


@dataclass(frozen=True, slots=True)
class Entry:
    slot: str
    definition: Def
    public_pairs: tuple[IOPair, ...]
    audit_pairs: tuple[IOPair, ...]
    last_used_round: int
    born_round: int


@dataclass(frozen=True, slots=True)
class Grave:
    slot: str
    definition: Def
    inputs: tuple[int, ...]
    def_result: int
    target_result: int
    round_no: int


@dataclass(frozen=True, slots=True)
class BreakerCase:
    inputs: tuple[int, ...]
    diverged: bool


class Canon:
    def __init__(self) -> None:
        self._entries: dict[str, Entry] = {}
        self._order: list[str] = []
        self._graves: list[Grave] = []

    def admit(
        self,
        definition: Def,
        public_pairs: Sequence[IOPair],
        audit_pairs: Sequence[IOPair],
        round_no: int,
    ) -> str | None:
        _check_round(round_no)
        public = tuple(public_pairs)
        audit = tuple(audit_pairs)
        if not public:
            raise ValueError("public pairs are required")
        if len(audit) != AUDIT_COUNT:
            raise ValueError("audit pairs must be 16")
        if definition.slot not in SLOTS:
            raise ValueError("unknown slot")
        for pair in (*public, *audit):
            if len(pair.inputs) != definition.arity:
                raise ValueError("pair arity does not match def")
        if definition.slot in self._entries or len(self._entries) >= len(SLOTS):
            return None
        self._entries[definition.slot] = Entry(
            slot=definition.slot,
            definition=definition,
            public_pairs=public,
            audit_pairs=audit,
            last_used_round=round_no,
            born_round=round_no,
        )
        self._order.append(definition.slot)
        return definition.slot

    def note_use(self, term: Expr, round_no: int) -> None:
        _check_round(round_no)
        for slot in _called_slots(term):
            entry = self._entries.get(slot)
            if entry is None or round_no <= entry.last_used_round:
                continue
            self._entries[slot] = replace(entry, last_used_round=round_no)

    def expire(self, round_no: int) -> tuple[str, ...]:
        _check_round(round_no)
        doomed = tuple(
            slot
            for slot in self._order
            if round_no >= self._entries[slot].last_used_round + EXPIRE_AFTER
        )
        for slot in doomed:
            self._remove(slot)
        return doomed

    def bench(self, round_no: int) -> str | None:
        """W rundach 4, 8, 12, … sprawdza najstarszy wpis. Zwraca zdjęty slot."""
        _check_round(round_no)
        if round_no == 0 or round_no % BENCH_EVERY != 0:
            return None
        slot = self._oldest()
        if slot is None or self._matches(self._entries[slot]):
            return None
        self._remove(slot)
        return slot

    def end_round(self, round_no: int) -> None:
        self.expire(round_no)
        self.bench(round_no)

    def replay(self) -> tuple[Entry, ...]:
        return tuple(self._entries[slot] for slot in self._order)

    def free_slots(self) -> tuple[str, ...]:
        return tuple(slot for slot in SLOTS if slot not in self._entries)

    def bury(
        self,
        *,
        slot: str,
        definition: Def,
        inputs: Sequence[int],
        def_result: int,
        target_result: int,
        round_no: int,
    ) -> None:
        _check_round(round_no)
        if slot not in SLOTS:
            raise ValueError("unknown slot")
        self._graves.append(
            Grave(
                slot=slot,
                definition=definition,
                inputs=tuple(inputs),
                def_result=def_result,
                target_result=target_result,
                round_no=round_no,
            )
        )

    def breaker_export(self) -> tuple[BreakerCase, ...]:
        return tuple(
            BreakerCase(grave.inputs, grave.def_result != grave.target_result)
            for grave in self._graves
        )

    @property
    def graves(self) -> tuple[Grave, ...]:
        return tuple(self._graves)

    def _oldest(self) -> str | None:
        if not self._order:
            return None
        return min(
            self._order,
            key=lambda slot: (self._entries[slot].born_round, self._order.index(slot)),
        )

    def _matches(self, entry: Entry) -> bool:
        defs = {slot: item.definition for slot, item in self._entries.items()}
        for pair in entry.audit_pairs:
            try:
                got = reduce_def(entry.definition, pair.inputs, defs=defs)
            except (ReduceError, ValueError):
                return False
            if got != pair.output:
                return False
        return True

    def _remove(self, slot: str) -> None:
        del self._entries[slot]
        self._order.remove(slot)


def _called_slots(expr: Expr) -> set[str]:
    found: set[str] = set()
    stack: list[Expr] = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, Call):
            found.add(node.slot)
            stack.extend(node.args)
        elif isinstance(node, (Succ, Pred)):
            stack.append(node.arg)
        elif isinstance(node, If0):
            stack.extend((node.cond, node.zero_branch, node.nonzero_branch))
        elif isinstance(node, Self):
            stack.extend(node.args)
    return found


def _check_round(round_no: int) -> None:
    if type(round_no) is not int or round_no < 0:
        raise ValueError("round")
