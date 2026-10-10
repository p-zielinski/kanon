import tempfile
from pathlib import Path

import pytest

from kanon.canon import Canon
from kanon.language import Break, parse_def, parse_expr, parse_message
from kanon.reduce import reduce_expr
from kanon.roles import Role, assign, format_expr, observe
from kanon.sample import speak
from kanon.tasks import IOPair, generate_task
from kanon.v2.seed import round_seed

RUNS = Path("runs/0")


def test_round_number_enters_the_task_seed_and_the_breaker_seed():
    with pytest.raises(TypeError):
        round_seed(0)
    assert round_seed(0, 0, 0) != round_seed(0, 1, 0)
    assert round_seed(0, 0, 0) != round_seed(0, 0, 1)
    assert round_seed(0, 5, 2) == round_seed(0, 5, 2)


def test_stage0_varies_goals_breaks_once_and_leaves_runs_0(monkeypatch):
    runs = RUNS.resolve()
    before = _bytes(runs)
    with tempfile.TemporaryDirectory(prefix="kanon-v2-stage0-") as directory:
        monkeypatch.chdir(directory)
        goals = [
            format_expr(generate_task(round_seed(0, round_no, 0), 1).target)
            for round_no in range(20)
        ]
        assert len(set(goals)) > 1

        observation = _breaker_observation()
        messages = [
            speak(
                "[BREAK]",
                observation,
                seed=round_seed(0, round_no, 0),
                epsilon=1,
            )
            for round_no in range(40)
        ]
        assert any(isinstance(parse_message(text, arity=1), Break) for text in messages)

        assert reduce_expr(parse_expr("(if0 (zero) (succ (zero)) (zero))")) == 1
        assert reduce_expr(parse_expr("(if0 (succ (zero)) (succ (zero)) (zero))")) == 0
        roles = assign(0)
        assert (roles.author, roles.breaker, roles.user, roles.outsider) == (0, 1, 2, 3)
        held = directory
    assert not Path(held).exists()
    assert _bytes(runs) == before


def _breaker_observation() -> str:
    pairs = tuple(IOPair((value,), value) for value in range(5))
    return observe(
        Role.BREAKER,
        canon=Canon(),
        pairs=pairs,
        candidate=parse_def("(def f0 1 (zero))"),
    )


def _bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink() or not item.is_file():
            continue
        total += item.stat().st_size
    return total
