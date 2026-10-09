import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kanon.canon import Canon
from kanon.roles import format_expr
from kanon.round import RoundEvent, task_seed
from kanon.run import (
    DEV_PER_DEPTH,
    EPSILON_FLOOR,
    METRICS,
    advance_depth,
    attempt_task,
    decay_epsilon,
    dev_seed,
    holdout_seed,
    score_tasks,
)
from kanon.tasks import build_holdout, generate_task

ROOT = Path(__file__).resolve().parents[1]


def test_depth_grows_only_after_two_evals_above_the_threshold_and_epsilon_floors():
    depth, streak = advance_depth(1, 0, 0.61)
    assert (depth, streak) == (1, 1)
    depth, streak = advance_depth(depth, streak, 0.6)
    assert (depth, streak) == (1, 0)
    depth, streak = advance_depth(1, 0, 0.61)
    depth, streak = advance_depth(depth, streak, 0.7)
    assert (depth, streak) == (2, 0)
    depth, streak = advance_depth(4, 1, 0.99)
    assert depth == 4

    epsilon = 1.0
    for _ in range(800):
        epsilon = decay_epsilon(epsilon)
    assert epsilon == EPSILON_FLOOR
    assert decay_epsilon(EPSILON_FLOOR) == EPSILON_FLOOR


def test_holdout_seed_is_independent_of_the_round_seed():
    assert holdout_seed(0) != 0
    assert dev_seed(0) not in (0, holdout_seed(0))
    first = build_holdout(holdout_seed(0))
    second = build_holdout(holdout_seed(0))
    assert [task.seed for task in first.tasks] == [task.seed for task in second.tasks]
    assert len(first.by_depth(1)) == 50
    assert first.tasks[0].seed != build_holdout(0).tasks[0].seed
    assert DEV_PER_DEPTH == 10


def test_measurement_needs_sixteen_audit_pairs_and_does_not_step(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("krok uczenia")

    monkeypatch.setattr("kanon.run.step_all", boom)
    monkeypatch.setattr("kanon.baselines.step", boom)
    canon = Canon()
    task = generate_task(task_seed(0, 2), 1)
    body = format_expr(task.target)
    wrong = "(succ (succ (succ (zero))))"
    missed = attempt_task(
        canon,
        task,
        seed=0,
        messages=(f"(draft {wrong})", f"(solve {wrong})"),
    )
    assert missed.drafts == 1
    assert missed.public_checks == 5
    assert missed.audit_pairs == 16
    assert missed.solved is False
    hit = attempt_task(
        canon,
        task,
        seed=0,
        messages=(f"(draft {body})",) * 5 + (f"(solve {body})",),
    )
    assert hit.drafts == 4
    assert hit.public_checks == 20
    assert hit.audit_pairs == 16
    assert hit.solved is True
    assert attempt_task(canon, task, seed=0, messages=(f"(solve {body})",)).solved is False
    assert canon.replay() == ()
    assert canon.graves == ()
    assert score_tasks(canon, (task,), seed=1) in (0.0, 1.0)


def test_measurement_runs_before_the_learning_step(monkeypatch, tmp_path):
    order: list[str] = []

    def play(*_args, **_kwargs):
        order.append("play")
        return RoundEvent(
            round_no=0,
            depth=1,
            seed=0,
            roles=(0, 1, 2, 3),
            admitted=None,
            audit_ran=False,
            broken=False,
            graves_added=0,
            outsider_ok=False,
            observations={"author": "", "breaker": "", "user": "", "outsider": ""},
            steps=(),
        )

    def solo(*_args, **kwargs):
        order.append("solo")
        assert kwargs["learn"] is False

    def score(*_args, **_kwargs):
        order.append("score")
        return 0.0

    def group_step(*_args, **_kwargs):
        order.append("step")
        return (False, False, False, False)

    def solo_step(*_args, **_kwargs):
        order.append("solo-step")
        return False

    class Dummy:
        def named_parameters(self):
            return []

    monkeypatch.setattr("kanon.run.play", play)
    monkeypatch.setattr("kanon.run.solo_round", solo)
    monkeypatch.setattr("kanon.run.score_tasks", score)
    monkeypatch.setattr("kanon.run.step_all", group_step)
    monkeypatch.setattr("kanon.run.step", solo_step)
    monkeypatch.setattr("kanon.run.build_agents", lambda: (Dummy(), Dummy(), Dummy(), Dummy()))
    monkeypatch.setattr("kanon.run.build_agent", lambda _seed: Dummy())
    monkeypatch.setattr(
        "kanon.run.start_frozen",
        lambda seed: type("Frozen", (), {"seed": seed, "agent": Dummy()})(),
    )

    def end_round(self, round_no):
        del self, round_no
        order.append("end")

    monkeypatch.setattr("kanon.run.Canon.end_round", end_round)
    from kanon.run import run

    run(seed=0, rounds=10, root=tmp_path)
    assert order[:8] == [
        "play",
        "solo",
        "step",
        "solo-step",
        "solo-step",
        "solo-step",
        "solo-step",
        "end",
    ]
    score_at = order.index("score")
    step_after = score_at + order[score_at:].index("step")
    assert all(item == "score" for item in order[score_at:step_after])
    assert order[step_after + 1] == "solo-step"
    assert order[step_after:].index("end") > order[step_after:].index("solo-step")
    holdout = json.loads((tmp_path / "0" / "holdout.json").read_text(encoding="utf-8"))
    assert len(holdout["tasks"]) == 200
    check = json.loads((tmp_path / "0" / "weight_check.json").read_text(encoding="utf-8"))
    assert check["tape_empty"] is True
    assert check["weights_changed"] is False


def test_nonempty_tape_with_unchanged_weights_fails(monkeypatch, tmp_path):
    def play(*_args, **_kwargs):
        return RoundEvent(
            round_no=0,
            depth=1,
            seed=0,
            roles=(0, 1, 2, 3),
            admitted=None,
            audit_ran=False,
            broken=False,
            graves_added=0,
            outsider_ok=False,
            observations={"author": "", "breaker": "", "user": "", "outsider": ""},
            steps=(("author", "(propose (def f0 1 (zero)))", "accepted"),),
        )

    class Dummy:
        def named_parameters(self):
            return []

    monkeypatch.setattr("kanon.run.play", play)
    monkeypatch.setattr("kanon.run.solo_round", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("kanon.run.step_all", lambda *_args, **_kwargs: (False,) * 4)
    monkeypatch.setattr("kanon.run.step", lambda *_args, **_kwargs: False)
    monkeypatch.setattr("kanon.run.build_agents", lambda: (Dummy(), Dummy(), Dummy(), Dummy()))
    monkeypatch.setattr("kanon.run.build_agent", lambda _seed: Dummy())
    monkeypatch.setattr(
        "kanon.run.start_frozen",
        lambda seed: type("Frozen", (), {"seed": seed, "agent": Dummy()})(),
    )
    from kanon.run import run

    with pytest.raises(RuntimeError, match="group step"):
        run(seed=0, rounds=1, root=tmp_path)


def test_a_later_invocation_continues_the_saved_round(tmp_path):
    from kanon.run import run

    straight = tmp_path / "straight"
    split = tmp_path / "split"
    run(seed=1, rounds=2, root=straight)
    run(seed=1, rounds=1, root=split)
    run(seed=1, rounds=1, root=split)
    left = (straight / "1" / "events.jsonl").read_text(encoding="utf-8")
    right = (split / "1" / "events.jsonl").read_text(encoding="utf-8")
    assert left == right
    rounds = [json.loads(line)["round"] for line in right.splitlines()]
    assert rounds == [0, 1]


def test_fresh_removes_the_saved_run(tmp_path):
    from kanon.run import run

    run(seed=1, rounds=2, root=tmp_path)
    run(seed=1, rounds=1, root=tmp_path, fresh=True)
    lines = (tmp_path / "1" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["round"] for line in lines] == [0]


def test_cli_adds_one_round_unless_fresh_is_asked(monkeypatch, tmp_path):
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return tmp_path

    monkeypatch.setattr("kanon.run.run", fake_run)
    from kanon.run import main

    assert main(["--seed", "3", "--root", str(tmp_path)]) == 0
    assert seen["rounds"] == 1
    assert seen["fresh"] is False
    assert main(["--seed", "3", "--root", str(tmp_path), "--fresh"]) == 0
    assert seen["fresh"] is True


def test_smoke_exits_zero_and_logs_have_no_hidden_term(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "kanon.run",
            "--rounds",
            "20",
            "--seed",
            "0",
            "--root",
            str(tmp_path),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    folder = tmp_path / "0"
    events_path = folder / "events.jsonl"
    metrics_path = folder / "metrics.csv"
    snapshot = folder / "canon" / "snapshot.json"
    assert events_path.is_file()
    assert metrics_path.is_file()
    assert snapshot.is_file()
    lines = [line for line in events_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 20
    epsilon = 1.0
    for line in lines:
        event = json.loads(line)
        observations = event["observations"]
        assert "hidden" not in json.dumps(observations)
        assert event["epsilon"] == pytest.approx(epsilon)
        if event["depth"] <= 2:
            for index, role in ((0, "author"), (1, "user"), (2, "outsider")):
                term = format_expr(generate_task(task_seed(event["seed"], index), event["depth"]).target)
                tail = observations[role].split("[PAIRS]", 1)[1]
                assert term not in tail
                assert event["targets"][role] == term
        epsilon = decay_epsilon(epsilon)
    holdout = json.loads((folder / "holdout.json").read_text(encoding="utf-8"))
    assert holdout["seed"] == holdout_seed(0)
    assert len(holdout["tasks"]) == 200
    check = json.loads((folder / "weight_check.json").read_text(encoding="utf-8"))
    assert check["weights_changed"] or check["tape_empty"]
    with metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == list(METRICS)
    assert [row["round"] for row in rows] == ["10", "20"]
    assert "hidden" not in snapshot.read_text(encoding="utf-8")
