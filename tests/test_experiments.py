"""Tests for benchmark records and CSV round trips."""

import pytest

from tsp_learning.experiments import (
    read_benchmark_csv,
    relative_optimality_gap,
    run_benchmark,
    write_benchmark_csv,
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


def test_benchmark_omits_unknown_optima_and_round_trips_csv(tmp_path: object) -> None:
    from pathlib import Path

    directory = Path(str(tmp_path))
    records = run_benchmark(
        [NearestNeighborSolver()], sizes=[6], seeds=[2], exact_max_cities=5
    )
    assert records[0].optimal_tour_length is None
    assert records[0].optimality_gap is None

    output = write_benchmark_csv(records, directory / "results.csv")
    assert read_benchmark_csv(output) == records
