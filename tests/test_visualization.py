"""Integration tests for the reproducible figure pipeline."""

from pathlib import Path

import numpy as np
import pytest

from tsp_learning.experiments import BenchmarkRecord, OnlineRecord
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.visualization import (
    plot_hedge_adaptation,
    plot_learned_comparison,
    plot_optimality_gap,
    plot_runtime,
    plot_tour_comparison,
)


def test_all_research_plots_are_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MPLBACKEND", "Agg")
    benchmark_records: list[BenchmarkRecord] = []
    for seed, optimum in ((0, 4.0), (1, 5.0)):
        instance_id = f"example-{seed}"
        for solver, length, runtime in (
            ("held-karp", optimum, 0.01),
            ("nearest-neighbor", optimum * 1.2, 0.001),
            ("nearest-neighbor-2opt", optimum * 1.05, 0.002),
            ("learned-edge-2opt", optimum * 1.03, 0.003),
        ):
            benchmark_records.append(
                BenchmarkRecord(
                    n_cities=4,
                    seed=seed,
                    instance_id=instance_id,
                    solver=solver,
                    tour_length=length,
                    runtime_seconds=runtime,
                    optimal_tour_length=optimum,
                    optimality_gap=(length - optimum) / optimum,
                )
            )

    online_records = [
        OnlineRecord(
            round_index=round_index,
            seed=seed,
            instance_id=f"online-{seed}",
            selected_expert="nearest-neighbor-2opt",
            selected_length=4.0,
            expert=expert,
            expert_length=length,
            loss=loss,
            probability_before=before,
            probability_after=after,
        )
        for round_index, seed, expert, length, loss, before, after in (
            (0, 10, "nearest-neighbor", 5.0, 1.0, 0.5, 0.4),
            (0, 10, "nearest-neighbor-2opt", 4.0, 0.0, 0.5, 0.6),
            (1, 11, "nearest-neighbor", 5.0, 1.0, 0.4, 0.3),
            (1, 11, "nearest-neighbor-2opt", 4.0, 0.0, 0.6, 0.7),
        )
    ]
    instance = TSPInstance(
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        name="square",
    )
    tour_results = [
        SolveResult((0, 1, 2, 3), 4.0, 0.01, "held-karp"),
        SolveResult((0, 2, 1, 3), 2.0 + 2.0 * 2.0**0.5, 0.001, "nearest-neighbor"),
    ]

    outputs = (
        plot_optimality_gap(benchmark_records, tmp_path / "gap.png"),
        plot_runtime(benchmark_records, tmp_path / "runtime.png"),
        plot_learned_comparison(benchmark_records, tmp_path / "learned.png"),
        plot_hedge_adaptation(online_records, tmp_path / "hedge.png"),
        plot_tour_comparison(instance, tour_results, tmp_path / "tours.png"),
    )

    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
