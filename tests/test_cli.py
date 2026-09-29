"""Small end-to-end CLI tests."""

import json
from pathlib import Path

import pytest

from tsp_learning.cli import main


def test_solve_command_emits_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["solve", "--cities", "6", "--seed", "4", "--solver", "nearest-neighbor-2opt"]) == 0
    output = capsys.readouterr().out
    payload = json.loads(output)
    assert payload["instance"] == "uniform-n6-seed4"
    assert payload["solver"] == "nearest-neighbor-2opt"
    assert sorted(payload["tour"]) == list(range(6))


def test_benchmark_command_writes_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "benchmark.csv"
    assert main(
        [
            "benchmark",
            "--sizes",
            "4",
            "--seeds",
            "1,2",
            "--exact-max-cities",
            "4",
            "--output",
            str(output),
        ]
    ) == 0
    assert output.exists()
    assert "wrote 6 records" in capsys.readouterr().out


def test_summarize_command_writes_aggregate_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    benchmark = tmp_path / "benchmark.csv"
    summary = tmp_path / "summary.csv"
    assert main(
        [
            "benchmark",
            "--sizes",
            "4",
            "--seeds",
            "1,2",
            "--exact-max-cities",
            "4",
            "--output",
            str(benchmark),
        ]
    ) == 0
    capsys.readouterr()

    assert main(["summarize", "--input", str(benchmark), "--output", str(summary)]) == 0
    assert summary.exists()
    assert "wrote 3 summary rows" in capsys.readouterr().out
