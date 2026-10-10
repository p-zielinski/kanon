import hashlib
import json
from pathlib import Path

from kanon.language import parse_def
from kanon.v2.measure import build_set, score_set, set_seed, write_metric
from kanon.v2.seed import round_seed
from kanon.v2.shelf import Shelf

_IDENT = "(def f0 1 (var 0))"
_ADD = "(def f0 2 (if0 (var 0) (var 1) (succ (self (pred (var 0)) (var 1)))))"
_SUCC = "(def f0 1 (succ (var 0)))"


def test_frontier_seven_has_no_set_and_leaving_eight_keeps_the_terms(tmp_path: Path):
    group, solo = _shelves()
    early = tmp_path / "early"
    early.mkdir()
    assert build_set(early, floor=7, group=group, solo=solo, seed=0) is None
    assert build_set(early, floor=7, group=group, solo=solo, seed=set_seed(0)) is None
    assert build_set(early, floor=8, group=group, solo=solo, seed=0) is None
    assert list(early.iterdir()) == []
    assert set_seed(0) == 1_000_003
    assert all(set_seed(0) != round_seed(0, n, attempt) for n in range(60) for attempt in range(4))

    path = build_set(tmp_path, floor=9, group=group, solo=solo, seed=0)
    assert path is not None and path.is_file()
    first = path.read_bytes()
    other = tmp_path / "again"
    other.mkdir()
    twin = build_set(other, floor=9, group=group, solo=solo, seed=0)
    assert twin is not None and twin.read_bytes() == first
    group.write(parse_def(_SUCC))
    assert build_set(tmp_path, floor=9, group=group, solo=solo, seed=1) == path
    assert path.read_bytes() == first
    data = json.loads(first)
    assert data["seed"] == set_seed(0)
    assert len(data["tasks"]) == 50


def test_shared_cards_are_ingredients_without_a_family_mark(tmp_path: Path):
    group, solo = _shelves()
    only_group = group.write(parse_def(_SUCC))
    path = build_set(tmp_path, floor=9, group=group, solo=solo, seed=0)
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "rodzina" not in text
    assert "family" not in text
    assert "identyczność" not in text
    assert "dodawanie" not in text
    assert "następnik" not in text
    data = json.loads(text)
    shared = {card.id for card in group.entries()} & {card.id for card in solo.entries()}
    assert only_group not in shared
    for task in data["tasks"]:
        assert "rodzina" not in task
        assert "family" not in task
        assert task["cards"]
        assert set(task["cards"]) <= shared
        assert only_group not in task["term"]


def test_fifty_round_score_keeps_the_weight_hash_and_six_networks(tmp_path: Path):
    path = tmp_path / "metrics.csv"
    write_metric(path, round_no=10, floor=7, gap=20)
    write_metric(path, round_no=20, floor=8, gap=2)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",")[-1] == "luka"
    assert lines[1].split(",")[-1] == ""
    assert lines[2].split(",")[-1] == ""

    blob = bytearray(b"weights-v2")
    digest = hashlib.sha256(blob).hexdigest()
    tasks = tuple(range(50))

    def step() -> None:
        blob[:] = b"stepped"

    def answers(group: int, solo: int, frozen: int):
        return (
            lambda task, limit=group: task < limit,
            lambda task, limit=solo: task < limit,
            lambda task, limit=frozen: task < limit,
        )

    quiet = score_set(49, tasks, floor=9, answers=answers(50, 0, 0), step=step)
    assert quiet.counted is False
    early = score_set(50, (), floor=7, answers=( _boom, _boom, _boom), step=step)
    assert early.gap == ""
    assert early.counted is False
    lead = score_set(50, tasks, floor=9, answers=answers(30, 10, 28), step=step)
    noise = score_set(100, tasks, floor=9, answers=answers(40, 20, 15), step=step)
    assert lead.gap == 2
    assert noise.gap == 20
    assert lead.seeds == (0, 1, 2, 3, 4, 5)
    assert noise.seeds == (0, 1, 2, 3, 4, 5)
    assert hashlib.sha256(blob).hexdigest() == digest
    write_metric(path, round_no=50, floor=9, gap=lead.gap)
    write_metric(path, round_no=100, floor=9, gap=noise.gap)
    stored = path.read_text(encoding="utf-8").splitlines()
    assert int(stored[3].split(",")[-1]) == 2
    assert int(stored[4].split(",")[-1]) == 20


def _boom(_task: object) -> bool:
    raise AssertionError("zbiór")


def _shelves() -> tuple[Shelf, Shelf]:
    group, solo = Shelf(), Shelf()
    for shelf in (group, solo):
        shelf.write(parse_def(_IDENT))
        shelf.write(parse_def(_ADD))
    return group, solo
