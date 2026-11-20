import csv
import os
import subprocess
import sys
from pathlib import Path

from kanon.v2.batch import PACKAGE_SECONDS, PACKAGE_TERMS
from kanon.v2.command import configure, main
from kanon.v2.resume import DEFAULT_DIR, Session

_ROOT = Path(__file__).resolve().parents[1]


def test_a_call_without_the_limit_keeps_the_full_package():
    plan = configure(["--seed", "0"])
    assert plan.seed == 0
    assert plan.directory == DEFAULT_DIR
    assert plan.directory == Path("runs/v2-seed0")
    assert plan.package_terms == PACKAGE_TERMS == 2000
    assert plan.package_seconds == PACKAGE_SECONDS == 120
    assert plan.max_packages is None
    assert plan.shape_count is None
    assert plan.metric_every == 10
    assert plan.survey_limit == 16


def test_smoke_writes_resume_for_two_short_packages(tmp_path: Path):
    before = Path("runs/0/resume.pt").stat().st_size
    code = main(
        [
            "--seed",
            "0",
            "--directory",
            str(tmp_path),
            "--package-terms",
            "5",
            "--max-packages",
            "2",
            "--shape-count",
            "40",
        ]
    )
    assert code == 0
    assert (tmp_path / "resume.pt").is_file()
    assert not (tmp_path / "metrics.csv").exists()
    opened = Session.open(tmp_path)
    assert opened.schema.index == 10
    assert opened.schema.shape == "T"
    assert set(opened.weights) == {"0", "1", "2", "3", "4", "5"}
    assert Path("runs/0/resume.pt").stat().st_size == before


def test_one_package_writes_a_metric_when_the_threshold_is_one(tmp_path: Path):
    code = main(
        [
            "--seed",
            "0",
            "--directory",
            str(tmp_path),
            "--package-terms",
            "5",
            "--max-packages",
            "1",
            "--shape-count",
            "40",
            "--metric-every",
            "1",
        ]
    )
    assert code == 0
    opened = Session.open(tmp_path)
    assert opened.schema.index == 5
    assert opened.schema.shape == "T"
    rows = list(csv.reader((tmp_path / "metrics.csv").open(encoding="utf-8")))
    assert rows[1][0] == "1"
    assert rows[1][1] == "paczka"


def test_survey_ceiling_one_stops_before_a_package(tmp_path: Path):
    before = Path("runs/0/resume.pt").stat().st_size
    code = main(
        [
            "--seed",
            "0",
            "--directory",
            str(tmp_path),
            "--survey-limit",
            "1",
            "--package-terms",
            "5",
            "--max-packages",
            "2",
            "--shape-count",
            "40",
        ]
    )
    assert code == 1
    assert not (tmp_path / "resume.pt").exists()
    assert list(tmp_path.iterdir()) == []
    assert Path("runs/0/resume.pt").stat().st_size == before
    module = subprocess.run(
        [
            sys.executable,
            "-m",
            "kanon.v2",
            "--seed",
            "0",
            "--directory",
            str(tmp_path / "mod"),
            "--survey-limit",
            "1",
            "--package-terms",
            "5",
            "--max-packages",
            "2",
            "--shape-count",
            "40",
        ],
        cwd=_ROOT,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert module.returncode == 1
    assert not (tmp_path / "mod" / "resume.pt").exists()


def test_readme_keeps_the_v1_loop_and_describes_one_v2_process():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "while true; do PYTHONPATH=src python -m kanon.run --seed 0 || exit; done" in text
    assert "python -m kanon.v2 --seed 0" in text
    assert "runs/v2-seed0/" in text
    assert "A GPU is not required." in text
    assert "There is no fifth network." in text
    assert "That loop describes v1." in text
