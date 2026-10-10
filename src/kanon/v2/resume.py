"""Zapis v2. Jeden `resume.pt`, podmieniany z pliku tymczasowego.

Katalog domyślny to `runs/v2-seed0/`. `runs/0/` jest biegiem v1 i ten
zapis go nie otwiera. Optimizery mogą być puste: AdamW dochodzi później.
W schemacie zapis jest po pełnej paczce. W warsztacie co 50 rund albo
co 2 minuty, co pierwsze.
"""

from __future__ import annotations

from pathlib import Path

from kanon.v2.batch import run_package
from kanon.v2.play import TRACE_CAP
from kanon.v2.schema import Schema

# Katalog v2. runs/0/ zostaje świadectwem v1.
DEFAULT_DIR = Path("runs/v2-seed0")
WORKSHOP_ROUNDS = 50
WORKSHOP_SECONDS = 120


class WorkshopClock:
    """Liczy od ostatniego zapisu. 50 rund albo 120 sekund."""

    def __init__(self) -> None:
        self.since_round = 0
        self.since_time = 0.0

    def due(self, round_no: int, now: float) -> bool:
        rounds = round_no - self.since_round >= WORKSHOP_ROUNDS
        elapsed = now - self.since_time >= WORKSHOP_SECONDS
        if not rounds and not elapsed:
            return False
        self.since_round = round_no
        self.since_time = float(now)
        return True


class Session:
    """Stan, który po przerwaniu wstaje z ostatniej pełnej paczki."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.schema = Schema()
        self.weights: dict[str, object] = {}
        self.optimizers: dict[str, object] = {}
        self.tape: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {}
        self.level = 0
        self.pools: dict[str, tuple[object, ...]] = {
            "wspolna": (),
            "grupa": (),
            "solo": (),
        }
        self.clock = WorkshopClock()

    def package(self, shape: object, clock) -> int:
        """Jedna paczka. Zapis jest po niej, nie w jej trakcie."""
        return run_package(
            self.schema,
            shape,
            save=lambda state: self.write(),
            exercise=lambda: None,
            clock=clock,
            closed=lambda: False,
        )

    def workshop(self, round_no: int, now: float) -> bool:
        """Zapis, gdy minęło 50 rund albo 2 minuty."""
        if not self.clock.due(round_no, now):
            return False
        self.write()
        return True

    def write(self) -> Path:
        payload = {
            "weights": self.weights,
            "optimizers": self.optimizers,
            "tape": _trim(self.tape),
            "state": self.schema.state(),
            "level": self.level,
            "pools": {
                "wspolna": tuple(self.pools["wspolna"]),
                "grupa": tuple(self.pools["grupa"]),
                "solo": tuple(self.pools["solo"]),
            },
            "since_round": self.clock.since_round,
            "since_time": self.clock.since_time,
        }
        return _commit(self.directory / "resume.pt", payload)

    @classmethod
    def open(cls, directory: Path) -> Session:
        data = _read(Path(directory) / "resume.pt")
        session = cls(directory)
        floor, shape, index = data["state"]
        session.schema.floor = floor
        session.schema.shape = shape
        session.schema.index = index
        session.weights = data["weights"]
        session.optimizers = data["optimizers"]
        session.level = data["level"]
        session.pools = data["pools"]
        session.tape = {
            (int(key[0]), int(key[1])): tuple(tuple(item) for item in row)
            for key, row in data["tape"].items()
        }
        session.clock.since_round = data["since_round"]
        session.clock.since_time = data["since_time"]
        return session


def _trim(
    tape: dict[tuple[int, int], tuple[tuple[int, ...], ...]],
) -> dict[tuple[int, int], tuple[tuple[int, ...], ...]]:
    return {key: tuple(row)[-TRACE_CAP:] for key, row in tape.items()}


def _commit(path: Path, payload: dict[str, object]) -> Path:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def _read(path: Path) -> dict[str, object]:
    import torch

    return torch.load(path, weights_only=False)
