from pathlib import Path

from kanon.v2.resume import DEFAULT_DIR, Session

_V1 = Path("runs/0/resume.pt")
_V2 = Path("runs/v2-seed0/resume.pt")


def test_fiftieth_round_saves_before_two_minutes(tmp_path: Path):
    before, default = _sizes()
    session = Session(tmp_path)
    assert session.workshop(49, 119) is False
    assert not (tmp_path / "resume.pt").exists()
    assert session.workshop(50, 10) is True
    assert (tmp_path / "resume.pt").is_file()
    assert session.workshop(99, 129) is False
    assert session.workshop(100, 129) is True
    assert _sizes() == (before, default)


def test_two_minutes_save_before_the_fiftieth_round(tmp_path: Path):
    before, default = _sizes()
    session = Session(tmp_path)
    assert session.workshop(1, 119) is False
    assert not (tmp_path / "resume.pt").exists()
    assert session.workshop(7, 120) is True
    assert session.workshop(56, 239) is False
    assert session.workshop(7, 240) is True
    assert _sizes() == (before, default)


def test_checkpoint_replaces_one_temporary_file(tmp_path: Path, monkeypatch):
    before, default = _sizes()
    seen: list[tuple[str, str]] = []
    original = Path.replace

    def spy(self: Path, target):
        seen.append((self.name, Path(target).name))
        return original(self, target)

    monkeypatch.setattr(Path, "replace", spy)
    session = Session(tmp_path)
    session.schema.floor = 6
    session.schema.shape = "G"
    session.schema.index = 11
    session.level = 3
    session.weights = {"0": [1], "5": [2]}
    session.optimizers = {}
    session.pools = {"wspolna": ("e0",), "grupa": ("lokalny",), "solo": ()}
    session.tape = {(0, 1): tuple((index,) for index in range(300))}
    session.write()
    session.write()
    assert seen == [("resume.pt.tmp", "resume.pt"), ("resume.pt.tmp", "resume.pt")]
    assert [path.name for path in tmp_path.iterdir()] == ["resume.pt"]
    loaded = Session.open(tmp_path)
    assert loaded.schema.state() == (6, "G", 11)
    assert loaded.level == 3
    assert loaded.weights == {"0": [1], "5": [2]}
    assert loaded.optimizers == {}
    assert loaded.pools == {"wspolna": ("e0",), "grupa": ("lokalny",), "solo": ()}
    assert loaded.tape[(0, 1)] == tuple((index,) for index in range(44, 300))
    assert DEFAULT_DIR == Path("runs/v2-seed0")
    assert _sizes() == (before, default)


def test_a_killed_package_resumes_at_the_last_full_one(tmp_path: Path):
    before, default = _sizes()
    session = Session(tmp_path)
    shape = _Shape()
    session.package(shape, _stop_at_two)
    session.package(shape, _stop_at_two)
    assert Session.open(tmp_path).schema.state() == (2, "R1", 4)
    try:
        session.package(shape, _die_midway)
    except SystemExit:
        pass
    assert session.schema.index == 5
    assert Session.open(tmp_path).schema.state() == (2, "R1", 4)
    assert _sizes() == (before, default)


class _Shape:
    name = "R1"
    count = 20

    def decode(self, index: int):
        return None


def _stop_at_two(count: int) -> float:
    return 120 if count >= 2 else 0


def _die_midway(count: int) -> float:
    if count >= 1:
        raise SystemExit(1)
    return 0


def _sizes() -> tuple[int, int | None]:
    default = _V2.stat().st_size if _V2.is_file() else None
    return _V1.stat().st_size, default
