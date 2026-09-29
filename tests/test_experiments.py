"""Tests for benchmark records and CSV round trips."""

from pathlib import Path

import pytest

from tsp_learning.experiments import (
    BenchmarkRecord,
    read_benchmark_csv,
    read_summary_csv,
    relative_optimality_gap,
    run_benchmark,
    summarize_benchmark,
    write_benchmark_csv,
    write_summary_csv,
)
from tsp_learning.solvers.heuristics import NearestNeighborSolver, TwoOptSolver


def test_relative_optimality_gap() -> None:
    assert relative_optimality_gap(11.0, 10.0) == pytest.approx(0.1)
    assert relative_optimality_gap(10.0 - 1e-12, 10.0) == 0.0
    assert relative_optimality_gap(10.0, None) is None
    with pytest.raises(ValueError):
        relative_optimality_gap(9.0, 10.0)


def test_benchmark_is_reproducible_except_for_runtime() -> None:
    solvers = [NearestNeighborSolver(), TwoOptSolver()]
    first = run_benchmark(solvers, sizes=[5], seeds=[3], exact_max_cities=5)
    second = run_benchmark(solvers, sizes=[5], seeds=[3], exact_max_cities=5)

    assert len(first) == 3
    assert [record.solver for record in first] == [
        "held-karp",
        "nearest-neighbor",
        "nearest-neighbor-2opt",
    ]
    assert [record.tour_length for record in first] == pytest.approx(
        [record.tour_length for record in second]
    )
    assert all(record.optimality_gap is not None for record in first)


def test_benchmark_omits_unknown_optima_and_round_trips_csv(tmp_path: Path) -> None:
    records = run_benchmark(
        [NearestNeighborSolver()], sizes=[6], seeds=[2], exact_max_cities=5
    )
    assert records[0].optimal_tour_length is None
    assert records[0].optimality_gap is None

    output = write_benchmark_csv(records, tmp_path / "results.csv")
    assert read_benchmark_csv(output) == records


def test_benchmark_summary_aggregates_and_round_trips(tmp_path: Path) -> None:
    records = [
        BenchmarkRecord(5, 0, "a", "solver-a", 10.0, 1.0, 8.0, 0.25),
        BenchmarkRecord(5, 1, "b", "solver-a", 14.0, 3.0, 8.0, 0.5),
        BenchmarkRecord(20, 0, "c", "solver-b", 30.0, 2.0, None, None),
    ]

    summaries = summarize_benchmark(records)

    assert [(summary.n_cities, summary.solver) for summary in summaries] == [
        (5, "solver-a"),
        (20, "solver-b"),
    ]
    first = summaries[0]
    assert first.n_instances == 2
    assert first.mean_tour_length == pytest.approx(12.0)
    assert first.median_runtime_seconds == pytest.approx(2.0)
    assert first.runtime_stddev_seconds == pytest.approx(1.0)
    assert first.mean_optimality_gap == pytest.approx(0.375)
    assert first.median_optimality_gap == pytest.approx(0.375)
    assert first.optimality_gap_stddev == pytest.approx(0.125)
    assert summaries[1].n_optimality_gaps == 0
    assert summaries[1].mean_optimality_gap is None

    output = write_summary_csv(summaries, tmp_path / "summary.csv")
    assert read_summary_csv(output) == summaries
