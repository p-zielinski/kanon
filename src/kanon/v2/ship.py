"""Wysyłka kopii poza plik sieci.

Token czyta się ze środowiska. W repozytorium go nie ma.
"""

from collections.abc import Mapping, MutableSequence
from datetime import date
from pathlib import Path
from typing import Protocol

INTERVAL = 6
LATEST = "kanon-v2-latest"
TOKEN = "KANON_V2_TOKEN"
CONTINUE = "kontynuacja"

_WEIGHTS = ("0", "1", "2", "3", "solo", "frozen")


class Publisher(Protocol):
    def publish(self, name: str, *, clobber: bool, files: tuple[str, ...]) -> None: ...


class ShipClock:
    def __init__(self) -> None:
        self.last = 0.0

    def due(self, now: float, *, saved: bool) -> bool:
        if not saved:
            return False
        return now - self.last >= INTERVAL

    def mark(self, now: float) -> None:
        self.last = now


def bundle(directory: Path) -> tuple[str, ...]:
    root = Path(directory)
    names: list[str] = []
    if (root / "canon" / "snapshot.json").is_file():
        names.append("snapshot.json")
    if (root / "metrics.csv").is_file():
        names.append("metrics.csv")
    weights = root / "weights"
    for label in _WEIGHTS:
        if (weights / f"{label}.bin").is_file():
            names.append(f"{label}.bin")
    return tuple(names)


def send(
    directory: Path,
    *,
    now: float,
    saved: bool,
    client: Publisher,
    env: Mapping[str, str],
    log: MutableSequence[str],
    day: date,
    clock: ShipClock,
) -> str:
    if not clock.due(now, saved=saved):
        return CONTINUE
    if not env.get(TOKEN, ""):
        log.append("wysyłka nie wyszła: brak tokenu")
        return CONTINUE
    files = bundle(directory)
    client.publish(LATEST, clobber=True, files=files)
    client.publish(f"kanon-v2-{day.isoformat()}", clobber=False, files=files)
    clock.mark(now)
    return CONTINUE
