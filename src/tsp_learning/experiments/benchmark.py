"""Benchmark exact and approximate solvers on shared generated instances."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from tsp_learning.instances import generate_random_euclidean
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.route import tour_length, validate_tour
from tsp_learning.solvers.base import Solver
from tsp_learning.solvers.exact import HeldKarpSolver


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One solver's result on one benchmark instance."""

    n_cities: int
    seed: int
    instance_id: str
    solver: str
    tour_length: float
    runtime_seconds: float
    optimal_tour_length: float | None
    optimality_gap: float | None


def relative_optimality_gap(
    solver_length: float,
    optimal_length: float | None,
    *,
    tolerance: float = 1e-9,
) -> float | None:
    """Compute ``(solver_length - optimal_length) / optimal_length`` when known."""

    if optimal_length is None:
        return None
    if optimal_length <= 0:
        raise ValueError("optimal_length must be positive")
    gap = (solver_length - optimal_length) / optimal_length
    if gap < -tolerance:
        raise ValueError("solver length is below the supplied optimum")
    return max(0.0, gap)


def _validate_result(instance: TSPInstance, result: SolveResult) -> None:
    validate_tour(result.tour, instance.n_cities)
    measured = tour_length(instance, result.tour)
    if abs(measured - result.length) > 1e-8:
        raise ValueError(
            f"solver {result.solver_name!r} reported {result.length}, "
            f"but its tour has length {measured}"
        )


def run_benchmark(
    solvers: Sequence[Solver],
    *,
    sizes: Sequence[int],
    seeds: Sequence[int],
    exact_max_cities: int = 12,
    reference_solver: Solver | None = None,
) -> list[BenchmarkRecord]:
    """Run solvers on identical deterministic instances.

    The exact reference is recorded as another solver for eligible sizes. For
    larger sizes, optimum-dependent fields are left empty rather than estimated.
    """

    if not solvers:
        raise ValueError("at least one benchmark solver is required")
    if not sizes or any(size < 2 for size in sizes):
        raise ValueError("sizes must contain values of at least two")
    if not seeds:
        raise ValueError("at least one seed is required")
    if exact_max_cities < 2:
        raise ValueError("exact_max_cities must be at least two")
    names = [solver.name for solver in solvers]
    if len(names) != len(set(names)):
        raise ValueError("benchmark solver names must be unique")

    reference = reference_solver or HeldKarpSolver(max_cities=exact_max_cities)
    records: list[BenchmarkRecord] = []
    for n_cities in sizes:
        for seed in seeds:
            instance = generate_random_euclidean(n_cities, seed=seed)
            instance_id = instance.name or f"n{n_cities}-seed{seed}"
            optimum: SolveResult | None = None
            if n_cities <= exact_max_cities:
                optimum = reference.solve(instance)
                _validate_result(instance, optimum)

            ordered_results: list[SolveResult] = []
            if optimum is not None and reference.name not in names:
                ordered_results.append(optimum)
            for solver in solvers:
                result = (
                    optimum
                    if optimum is not None and solver.name == reference.name
                    else solver.solve(instance)
                )
                if result is None:
                    raise RuntimeError("internal error: missing benchmark result")
                _validate_result(instance, result)
                ordered_results.append(result)

            optimal_length = None if optimum is None else optimum.length
            for result in ordered_results:
                records.append(
                    BenchmarkRecord(
                        n_cities=n_cities,
                        seed=seed,
                        instance_id=instance_id,
                        solver=result.solver_name,
                        tour_length=result.length,
                        runtime_seconds=result.runtime_seconds,
                        optimal_tour_length=optimal_length,
                        optimality_gap=relative_optimality_gap(result.length, optimal_length),
                    )
                )
    return records


def write_benchmark_csv(records: Iterable[BenchmarkRecord], output_path: str | Path) -> Path:
    """Write benchmark records to a stable, machine-readable CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(BenchmarkRecord)]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    return path


def read_benchmark_csv(input_path: str | Path) -> list[BenchmarkRecord]:
    """Load records produced by :func:`write_benchmark_csv`."""

    path = Path(input_path)
    records: list[BenchmarkRecord] = []
    with path.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            optimal_text = row["optimal_tour_length"]
            gap_text = row["optimality_gap"]
            records.append(
                BenchmarkRecord(
                    n_cities=int(row["n_cities"]),
                    seed=int(row["seed"]),
                    instance_id=row["instance_id"],
                    solver=row["solver"],
                    tour_length=float(row["tour_length"]),
                    runtime_seconds=float(row["runtime_seconds"]),
                    optimal_tour_length=float(optimal_text) if optimal_text else None,
                    optimality_gap=float(gap_text) if gap_text else None,
                )
            )
    return records
