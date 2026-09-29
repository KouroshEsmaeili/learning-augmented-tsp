"""Aggregate benchmark measurements into compact, reproducible summaries."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean, median, pstdev

from tsp_learning.experiments.benchmark import BenchmarkRecord


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Descriptive statistics for one solver and instance size."""

    n_cities: int
    solver: str
    n_instances: int
    n_optimality_gaps: int
    mean_tour_length: float
    median_tour_length: float
    mean_runtime_seconds: float
    median_runtime_seconds: float
    runtime_stddev_seconds: float
    mean_optimality_gap: float | None
    median_optimality_gap: float | None
    optimality_gap_stddev: float | None


def _optional_statistics(
    values: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    return fmean(values), median(values), pstdev(values)


def summarize_benchmark(records: Sequence[BenchmarkRecord]) -> list[BenchmarkSummary]:
    """Group records by size and solver and compute population statistics."""

    if not records:
        raise ValueError("at least one benchmark record is required")
    grouped: dict[tuple[int, str], list[BenchmarkRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.n_cities, record.solver)].append(record)

    summaries: list[BenchmarkSummary] = []
    for (n_cities, solver), group in sorted(grouped.items()):
        lengths = [record.tour_length for record in group]
        runtimes = [record.runtime_seconds for record in group]
        gaps = [
            record.optimality_gap
            for record in group
            if record.optimality_gap is not None
        ]
        mean_gap, median_gap, gap_stddev = _optional_statistics(gaps)
        summaries.append(
            BenchmarkSummary(
                n_cities=n_cities,
                solver=solver,
                n_instances=len(group),
                n_optimality_gaps=len(gaps),
                mean_tour_length=fmean(lengths),
                median_tour_length=median(lengths),
                mean_runtime_seconds=fmean(runtimes),
                median_runtime_seconds=median(runtimes),
                runtime_stddev_seconds=pstdev(runtimes),
                mean_optimality_gap=mean_gap,
                median_optimality_gap=median_gap,
                optimality_gap_stddev=gap_stddev,
            )
        )
    return summaries


def write_summary_csv(
    summaries: Iterable[BenchmarkSummary], output_path: str | Path
) -> Path:
    """Write aggregate benchmark statistics to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(BenchmarkSummary)]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(summary) for summary in summaries)
    return path


def read_summary_csv(input_path: str | Path) -> list[BenchmarkSummary]:
    """Read aggregate benchmark statistics produced by :func:`write_summary_csv`."""

    summaries: list[BenchmarkSummary] = []
    with Path(input_path).open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            summaries.append(
                BenchmarkSummary(
                    n_cities=int(row["n_cities"]),
                    solver=row["solver"],
                    n_instances=int(row["n_instances"]),
                    n_optimality_gaps=int(row["n_optimality_gaps"]),
                    mean_tour_length=float(row["mean_tour_length"]),
                    median_tour_length=float(row["median_tour_length"]),
                    mean_runtime_seconds=float(row["mean_runtime_seconds"]),
                    median_runtime_seconds=float(row["median_runtime_seconds"]),
                    runtime_stddev_seconds=float(row["runtime_stddev_seconds"]),
                    mean_optimality_gap=(
                        float(row["mean_optimality_gap"])
                        if row["mean_optimality_gap"]
                        else None
                    ),
                    median_optimality_gap=(
                        float(row["median_optimality_gap"])
                        if row["median_optimality_gap"]
                        else None
                    ),
                    optimality_gap_stddev=(
                        float(row["optimality_gap_stddev"])
                        if row["optimality_gap_stddev"]
                        else None
                    ),
                )
            )
    return summaries
