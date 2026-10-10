from kanon.language import Zero
from kanon.v2.batch import run_package, run_packages
from kanon.v2.schema import Schema


def test_five_thousand_fast_terms_make_three_saves():
    schema = Schema()
    saves: list = []
    sizes = run_packages(
        schema,
        _Fast(5000),
        save=saves.append,
        exercise=lambda: None,
        clock=lambda done: 0.0,
        closed=lambda: False,
    )
    assert sizes == [2000, 2000, 1000]
    assert len(saves) == 3
    assert [item[2] for item in saves] == [2000, 4000, 5000]


def test_clock_stops_the_package_after_two_minutes():
    schema = Schema()
    saves: list = []
    size = run_package(
        schema,
        _Fast(100),
        save=saves.append,
        exercise=lambda: None,
        clock=lambda done: 120.0 if done >= 10 else 0.0,
        closed=lambda: False,
    )
    assert size == 10
    assert len(saves) == 1
    assert schema.index == 10


def test_closed_floor_runs_one_exercise_per_package():
    schema = Schema()
    log: list[str] = []
    sizes = run_packages(
        schema,
        _Fast(5000),
        save=lambda state: log.append("save"),
        exercise=lambda: log.append("exercise"),
        clock=lambda done: 0.0,
        closed=lambda: True,
    )
    assert sizes == [2000, 2000, 1000]
    assert log == ["save", "exercise", "save", "exercise", "save", "exercise"]


def test_open_floors_skip_the_exercise():
    exercise = _Stub()
    run_packages(
        Schema(),
        _Fast(5000),
        save=lambda state: None,
        exercise=exercise,
        clock=lambda done: 0.0,
        closed=lambda: False,
    )
    assert exercise.calls == 0


def test_metrics_row_lands_every_ten_packages():
    metrics: list = []
    schema = Schema()
    run_packages(
        schema,
        _Fast(20_000),
        save=lambda state: None,
        exercise=lambda: None,
        clock=lambda done: 0.0,
        closed=lambda: False,
        metrics=metrics,
    )
    assert metrics == [(2, "R1", 20_000)]
    quiet: list = []
    run_packages(
        Schema(),
        _Fast(18_000),
        save=lambda state: None,
        exercise=lambda: None,
        clock=lambda done: 0.0,
        closed=lambda: False,
        metrics=quiet,
    )
    assert quiet == []


class _Fast:
    name = "R1"

    def __init__(self, count: int) -> None:
        self.count = count

    def decode(self, index: int):
        return Zero()


class _Stub:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1
