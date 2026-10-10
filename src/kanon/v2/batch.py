"""Paczka schematu. 2000 termów albo 2 minuty, co pierwsze.

Po paczce zapis stanu jest wołany raz. Gdy jakieś piętro jest już
zamknięte, potem jest jedno ćwiczenie. Co 10 paczek dochodzi wiersz
metryki: piętro, kształt, numer. Wiersza na term nie ma.
"""

from __future__ import annotations

from collections.abc import Callable

from kanon.v2.schema import Schema

PACKAGE_TERMS = 2000
PACKAGE_SECONDS = 120

_Clock = Callable[[int], float]
_Save = Callable[[tuple[int, str, int]], None]
_Exercise = Callable[[], None]
_Closed = Callable[[], bool]


def run_package(
    schema: Schema,
    shape: object,
    *,
    save: _Save,
    exercise: _Exercise,
    clock: _Clock,
    closed: _Closed,
    terms: int = PACKAGE_TERMS,
    seconds: float = PACKAGE_SECONDS,
) -> int:
    """Jedna paczka. Zapis woła raz, na końcu."""
    schema.shape = shape.name
    count = 0
    while count < terms and schema.index < shape.count:
        shape.decode(schema.index)
        schema.index += 1
        count += 1
        if clock(count) >= seconds:
            break
    if count == 0:
        return 0
    save(schema.state())
    if closed():
        exercise()
    return count


def run_packages(
    schema: Schema,
    shape: object,
    *,
    save: _Save,
    exercise: _Exercise,
    clock: _Clock,
    closed: _Closed,
    metrics: list[tuple[int, str, int]] | None = None,
) -> list[int]:
    """Paczki aż do końca kształtu. Rozmiary paczek wracają w kolejności."""
    sizes: list[int] = []
    while schema.index < shape.count:
        size = run_package(
            schema,
            shape,
            save=save,
            exercise=exercise,
            clock=clock,
            closed=closed,
        )
        if size == 0:
            break
        sizes.append(size)
        if metrics is not None and len(sizes) % 10 == 0:
            metrics.append(schema.state())
    return sizes
