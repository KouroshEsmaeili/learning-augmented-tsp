"""Plot benchmark results without making plotting a core runtime concern."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from statistics import fmean, pstdev

import numpy as np

from tsp_learning.experiments import BenchmarkRecord, OnlineRecord
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult

_SOLVER_LABELS = {
    "held-karp": "Held–Karp",
    "nearest-neighbor": "Nearest neighbor",
    "nearest-neighbor-2opt": "Nearest neighbor + 2-opt",
    "nearest-neighbor-best-start": "Nearest neighbor (best start)",
    "learned-edge": "Learned edge",
    "learned-edge-2opt": "Learned edge + 2-opt",
}


def _solver_label(name: str) -> str:
    return _SOLVER_LABELS.get(name, name)


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
        means = [fmean(by_size[size]) for size in sizes]
        deviations = [pstdev(by_size[size]) for size in sizes]
        error_bars = np.asarray(
            (
                [min(mean, deviation) for mean, deviation in zip(means, deviations, strict=True)],
                deviations,
            )
        )
        axis.errorbar(
            sizes,
            means,
            yerr=error_bars,
            marker="o",
            capsize=3,
            label=_solver_label(solver),
        )
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
    """Plot mean relative optimality gap with population-standard-deviation bars."""

    return _plot_metric(
        records,
        value=lambda record: record.optimality_gap,
        ylabel="Mean relative optimality gap",
        output_path=output_path,
    )


def plot_runtime(records: Sequence[BenchmarkRecord], output_path: str | Path) -> Path:
    """Plot mean runtime with standard-deviation bars on a logarithmic axis."""

    path = _plot_metric(
        records,
        value=lambda record: record.runtime_seconds,
        ylabel="Mean runtime (seconds)",
        output_path=output_path,
        logarithmic_y=True,
    )
    return path


def plot_learned_comparison(
    records: Sequence[BenchmarkRecord], output_path: str | Path
) -> Path:
    """Plot paired tour-length change relative to nearest neighbor."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from error

    by_instance: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for record in records:
        by_instance[(record.n_cities, record.instance_id)][record.solver] = record.tour_length
    comparisons: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    compared_solvers = ("nearest-neighbor-2opt", "learned-edge-2opt")
    for (n_cities, _), lengths in by_instance.items():
        baseline = lengths.get("nearest-neighbor")
        if baseline is None:
            continue
        for solver in compared_solvers:
            if solver in lengths:
                comparisons[solver][n_cities].append(100.0 * (lengths[solver] / baseline - 1.0))
    if not comparisons:
        raise ValueError("records do not contain learned and classical comparison data")

    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.axhline(0.0, color="black", linewidth=1, linestyle="--", label="Nearest neighbor")
    for solver, by_size in sorted(comparisons.items()):
        sizes = sorted(by_size)
        axis.errorbar(
            sizes,
            [fmean(by_size[size]) for size in sizes],
            yerr=[pstdev(by_size[size]) for size in sizes],
            marker="o",
            capsize=3,
            label=_solver_label(solver),
        )
    axis.set_xlabel("Number of cities")
    axis.set_ylabel("Mean tour-length change vs nearest neighbor (%)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_hedge_adaptation(
    records: Sequence[OnlineRecord], output_path: str | Path
) -> Path:
    """Plot each expert's probability after every online round."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from error
    if not records:
        raise ValueError("at least one online record is required")
    grouped: dict[str, list[OnlineRecord]] = defaultdict(list)
    for record in records:
        grouped[record.expert].append(record)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    for expert, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda record: record.round_index)
        axis.plot(
            [record.round_index + 1 for record in ordered],
            [record.probability_after for record in ordered],
            marker="o",
            label=_solver_label(expert),
        )
    axis.set_xlabel("Online round")
    axis.set_ylabel("Expert probability after update")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_tour_comparison(
    instance: TSPInstance,
    results: Sequence[SolveResult],
    output_path: str | Path,
) -> Path:
    """Plot several solver tours on the same Euclidean instance."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("plotting requires the 'plot' optional dependency") from error
    if not results:
        raise ValueError("at least one solver result is required")

    n_columns = 2
    n_rows = (len(results) + n_columns - 1) // n_columns
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(9, 4.2 * n_rows), squeeze=False)
    for axis, result in zip(axes.flat, results, strict=False):
        indices = np.asarray((*result.tour, result.tour[0]), dtype=np.intp)
        points = instance.coordinates[indices]
        axis.plot(points[:, 0], points[:, 1], marker="o", linewidth=1.4)
        for city, (x_coordinate, y_coordinate) in enumerate(instance.coordinates):
            axis.annotate(
                str(city),
                (x_coordinate, y_coordinate),
                xytext=(4, 4),
                textcoords="offset points",
            )
        axis.set_title(f"{_solver_label(result.solver_name)}\nlength={result.length:.4f}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(results) :]:
        axis.set_visible(False)
    figure.suptitle(f"Representative instance: {instance.name}")
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
