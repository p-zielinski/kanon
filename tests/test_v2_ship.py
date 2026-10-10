from datetime import date
from pathlib import Path

from kanon.v2.ship import ShipClock, bundle, send

_DAY = date(2026, 10, 10)
_NAMES = (
    "snapshot.json",
    "metrics.csv",
    "0.bin",
    "1.bin",
    "2.bin",
    "3.bin",
    "solo.bin",
    "frozen.bin",
)


def test_five_hours_do_not_ship_and_six_hours_ship_once(tmp_path: Path):
    _fill(tmp_path)
    clock = ShipClock()
    client = _Stub()
    env = {"KANON_V2_TOKEN": "z-testu"}
    log: list[str] = []
    assert send(tmp_path, now=5, saved=True, client=client, env=env, log=log, day=_DAY, clock=clock) == "kontynuacja"
    assert client.latest == []
    assert send(tmp_path, now=6, saved=False, client=client, env=env, log=log, day=_DAY, clock=clock) == "kontynuacja"
    assert client.latest == []
    assert send(tmp_path, now=6, saved=True, client=client, env=env, log=log, day=_DAY, clock=clock) == "kontynuacja"
    assert client.latest == ["kanon-v2-latest"]
    assert send(tmp_path, now=6, saved=True, client=client, env=env, log=log, day=_DAY, clock=clock) == "kontynuacja"
    assert client.latest == ["kanon-v2-latest"]


def test_the_bundle_is_the_shelf_the_metrics_and_six_weights(tmp_path: Path):
    _fill(tmp_path)
    (tmp_path / "resume.pt").write_bytes(b"wagi")
    (tmp_path / "optimizer").write_text("adam", encoding="utf-8")
    (tmp_path / "tape").write_text("slady", encoding="utf-8")
    names = bundle(tmp_path)
    assert names == _NAMES
    assert "resume.pt" not in names
    assert "optimizer" not in names
    assert "tape" not in names


def test_the_same_day_clobbers_latest_and_keeps_one_daily_tag(tmp_path: Path):
    _fill(tmp_path)
    clock = ShipClock()
    client = _Stub()
    env = {"KANON_V2_TOKEN": "z-testu"}
    log: list[str] = []
    send(tmp_path, now=6, saved=True, client=client, env=env, log=log, day=_DAY, clock=clock)
    send(tmp_path, now=12, saved=True, client=client, env=env, log=log, day=_DAY, clock=clock)
    assert client.latest == ["kanon-v2-latest", "kanon-v2-latest"]
    assert client.tags == ["kanon-v2-2026-10-10"]
    assert client.files == [_NAMES, _NAMES, _NAMES, _NAMES]


def test_a_missing_token_is_logged_and_the_loop_continues(tmp_path: Path):
    _fill(tmp_path)
    clock = ShipClock()
    client = _Stub()
    log: list[str] = []
    assert send(tmp_path, now=6, saved=True, client=client, env={}, log=log, day=_DAY, clock=clock) == "kontynuacja"
    assert send(tmp_path, now=6, saved=True, client=client, env={}, log=log, day=_DAY, clock=clock) == "kontynuacja"
    assert client.latest == []
    assert client.tags == []
    assert log
    assert "token" in log[0].lower()
    source = Path("src/kanon/v2/model.py").read_text(encoding="utf-8")
    assert "download" not in source
    import kanon.v2.ship as ship

    assert Path(ship.__file__).name == "ship.py"


class _Stub:
    def __init__(self) -> None:
        self.latest: list[str] = []
        self.tags: list[str] = []
        self.files: list[tuple[str, ...]] = []

    def publish(self, name: str, *, clobber: bool, files: tuple[str, ...]) -> None:
        self.files.append(tuple(files))
        if name == "kanon-v2-latest":
            self.latest.append(name)
            return
        if clobber:
            self.tags.append(f"podmiana {name}")
            return
        if name not in self.tags:
            self.tags.append(name)


def _fill(directory: Path) -> None:
    canon = directory / "canon"
    canon.mkdir()
    (canon / "snapshot.json").write_text("{}\n", encoding="utf-8")
    (directory / "metrics.csv").write_text("numer\n", encoding="utf-8")
    weights = directory / "weights"
    weights.mkdir()
    for name in ("0", "1", "2", "3", "solo", "frozen"):
        (weights / f"{name}.bin").write_bytes(b"w")
