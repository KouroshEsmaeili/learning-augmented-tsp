"""Plot benchmark results without making plotting a core runtime concern."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

from tsp_learning.experiments import BenchmarkRecord


def _plot_metric(
    records: Sequence[BenchmarkRecord],
    *,
    value: Callable[[BenchmarkRecord], float | None],
    ylabel: str,
    output_path: str | Path,
    logarithmic_y: bool = False,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from error

    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        metric = value(record)
        if metric is not None:
            grouped[record.solver][record.n_cities].append(metric)
    if not grouped:
        raise ValueError("no records contain the requested metric")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for solver, by_size in sorted(grouped.items()):
        sizes = sorted(by_size)
        means = [sum(by_size[size]) / len(by_size[size]) for size in sizes]
        axis.plot(sizes, means, marker="o", label=solver)
    axis.set_xlabel("Number of cities")
    axis.set_ylabel(ylabel)
    if logarithmic_y:
        axis.set_yscale("log")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_optimality_gap(
    records: Sequence[BenchmarkRecord], output_path: str | Path
) -> Path:
    """Plot mean relative optimality gap by instance size."""

    return _plot_metric(
        records,
        value=lambda record: record.optimality_gap,
        ylabel="Mean relative optimality gap",
        output_path=output_path,
    )


def plot_runtime(records: Sequence[BenchmarkRecord], output_path: str | Path) -> Path:
    """Plot mean runtime by instance size on a logarithmic y-axis."""

    path = _plot_metric(
        records,
        value=lambda record: record.runtime_seconds,
        ylabel="Mean runtime (seconds)",
        output_path=output_path,
        logarithmic_y=True,
    )
    return path
