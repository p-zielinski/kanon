import hashlib
from pathlib import Path

from kanon.language import parse_def
from kanon.v2.ask import ask
from kanon.v2.record import write_snapshot
from kanon.v2.resume import Session
from kanon.v2.shelf import Shelf

_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_ONCE = "(step (eq (call f0 (var 0) (zero)) (var 0)) (rewrite f0) (reduce) (ih))"
_TWICE = "(step (eq (call f0 (var 0) (zero)) (var 0)) (rewrite f0) (reduce) (ih) (ih))"


def test_read_keeps_the_shelf_and_adds_two_and_three(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("kanon.v2.learn.step", lambda *args, **kwargs: calls.append("krok"))
    _save(tmp_path)
    shelf = _digest(tmp_path / "canon" / "snapshot.json")
    resume = _digest(tmp_path / "resume.pt")
    opened = Session.open(tmp_path)
    assert ask(tmp_path, "(call f0 (num 2) (num 3))") == "5"
    assert _digest(tmp_path / "canon" / "snapshot.json") == shelf
    assert _digest(tmp_path / "resume.pt") == resume
    again = Session.open(tmp_path)
    assert again.schema.state() == opened.schema.state() == (2, "R1", 17)
    assert again.level == opened.level == 2
    assert again.pools == opened.pools
    assert calls == []


def test_a_sentence_that_uses_ih_twice_is_rejected(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("kanon.v2.learn.step", lambda *args, **kwargs: calls.append("krok"))
    _save(tmp_path)
    assert ask(tmp_path, _ONCE) == "przyjęte"
    assert ask(tmp_path, _TWICE) == "odrzucone"
    assert calls == []
    assert Session.open(tmp_path).schema.index == 17
    assert Session.open(tmp_path).level == 2


def _save(directory: Path) -> None:
    group = Shelf()
    ident = group.write(parse_def(_ADD))
    group.bind((ident,))
    write_snapshot(
        directory,
        group=group,
        solo=Shelf(),
        local_group=(),
        local_solo=(),
        term=17,
        level=2,
    )
    session = Session(directory)
    session.schema.index = 17
    session.level = 2
    session.pools = {"wspolna": (ident,), "grupa": ("lokalny",), "solo": ()}
    session.write()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
