"""Command-line entry point for small, reproducible TSP experiments."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from tsp_learning.experiments import (
    read_benchmark_csv,
    read_online_csv,
    run_benchmark,
    run_online_experiment,
    summarize_benchmark,
    summarize_online,
    write_benchmark_csv,
    write_online_csv,
    write_online_summary_csv,
    write_summary_csv,
)
from tsp_learning.instances import generate_random_euclidean
from tsp_learning.solvers.base import Solver
from tsp_learning.solvers.exact import HeldKarpSolver
from tsp_learning.solvers.heuristics import NearestNeighborSolver, TwoOptSolver
from tsp_learning.solvers.learning import (
    LearnedEdgeSolver,
    TrainingConfig,
    load_edge_model,
    save_edge_model,
    train_edge_model,
)
from tsp_learning.visualization import (
    plot_hedge_adaptation,
    plot_learned_comparison,
    plot_optimality_gap,
    plot_runtime,
    plot_tour_comparison,
)


def _integer_list(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def _add_instance_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cities", type=int, default=10, help="number of cities (default: 10)")
    parser.add_argument("--seed", type=int, default=0, help="instance seed (default: 0)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tsp-learning",
        description="Experiments for the symmetric Euclidean traveling salesperson problem.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    solve = subparsers.add_parser("solve", help="solve one generated Euclidean instance")
    _add_instance_arguments(solve)
    solve.add_argument(
        "--solver",
        choices=("held-karp", "nearest-neighbor", "nearest-neighbor-2opt", "learned-edge"),
        default="nearest-neighbor-2opt",
    )
    solve.add_argument("--start-city", type=int, default=0)
    solve.add_argument("--model", type=Path, help="model artifact required by learned-edge")
    solve.add_argument(
        "--no-two-opt",
        action="store_true",
        help="skip learned-tour post-processing",
    )

    benchmark = subparsers.add_parser(
        "benchmark", help="benchmark baselines across sizes and seeds"
    )
    benchmark.add_argument("--sizes", type=_integer_list, default=[5, 8, 10])
    benchmark.add_argument("--seeds", type=_integer_list, default=[0, 1, 2])
    benchmark.add_argument("--exact-max-cities", type=int, default=12)
    benchmark.add_argument("--model", type=Path, help="also evaluate this learned model")
    benchmark.add_argument("--output", type=Path, default=Path("artifacts/benchmark.csv"))

    summarize = subparsers.add_parser(
        "summarize", help="aggregate a benchmark CSV by size and solver"
    )
    summarize.add_argument("--input", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="train a small edge-scoring model")
    train.add_argument("--sizes", type=_integer_list, default=[5, 6, 7])
    train.add_argument("--seeds", type=_integer_list, default=[100, 101, 102])
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--learning-rate", type=float, default=0.01)
    train.add_argument("--hidden-dim", type=int, default=32)
    train.add_argument("--training-seed", type=int, default=0)
    train.add_argument("--output", type=Path, default=Path("artifacts/edge-model.pt"))

    plot = subparsers.add_parser("plot", help="plot a benchmark CSV")
    plot.add_argument("--input", type=Path, default=Path("artifacts/benchmark.csv"))
    plot.add_argument("--output-dir", type=Path, default=Path("artifacts/plots"))
    plot.add_argument(
        "--classical-input",
        type=Path,
        help="optional classical benchmark used for the runtime-scaling plot",
    )
    plot.add_argument("--online-input", type=Path, help="optional long-form Hedge CSV")
    plot.add_argument("--model", type=Path, help="optional model for the tour comparison")
    plot.add_argument("--tour-cities", type=int, default=10)
    plot.add_argument("--tour-seed", type=int, default=200)

    online = subparsers.add_parser("online", help="run Hedge over classical solver experts")
    online.add_argument("--cities", type=int, default=12)
    online.add_argument("--seeds", type=_integer_list, default=[0, 1, 2, 3, 4])
    online.add_argument("--learning-rate", type=float, default=0.5)
    online.add_argument("--output", type=Path, help="optional long-form CSV output")

    summarize_online_parser = subparsers.add_parser(
        "summarize-online", help="aggregate a long-form Hedge CSV by expert"
    )
    summarize_online_parser.add_argument("--input", type=Path, required=True)
    summarize_online_parser.add_argument("--output", type=Path, required=True)
    return parser


def _solver_for_name(
    name: str,
    *,
    start_city: int,
    model_path: Path | None,
    learned_two_opt: bool,
    exact_max_cities: int,
) -> Solver:
    if name == "held-karp":
        return HeldKarpSolver(start_city=start_city, max_cities=exact_max_cities)
    if name == "nearest-neighbor":
        return NearestNeighborSolver(start_city=start_city)
    if name == "nearest-neighbor-2opt":
        return TwoOptSolver(initial_solver=NearestNeighborSolver(start_city=start_city))
    if model_path is None:
        raise ValueError("--model is required when --solver learned-edge is selected")
    model, training_names = load_edge_model(model_path)
    return LearnedEdgeSolver(
        model=model,
        start_city=start_city,
        apply_two_opt=learned_two_opt,
        training_instance_names=training_names,
    )


def _run_solve(arguments: argparse.Namespace) -> int:
    n_cities = int(arguments.cities)
    solver_name = str(arguments.solver)
    solver = _solver_for_name(
        solver_name,
        start_city=int(arguments.start_city),
        model_path=arguments.model,
        learned_two_opt=not bool(arguments.no_two_opt),
        exact_max_cities=20,
    )
    instance = generate_random_euclidean(n_cities, seed=int(arguments.seed))
    result = solver.solve(instance)
    print(
        json.dumps(
            {
                "instance": instance.name,
                "solver": result.solver_name,
                "tour": result.tour,
                "length": result.length,
                "runtime_seconds": result.runtime_seconds,
                "metadata": result.metadata,
            },
            indent=2,
        )
    )
    return 0


def _run_benchmark(arguments: argparse.Namespace) -> int:
    solvers: list[Solver] = [
        NearestNeighborSolver(),
        TwoOptSolver(),
    ]
    model_path: Path | None = arguments.model
    if model_path is not None:
        model, training_names = load_edge_model(model_path)
        solvers.append(LearnedEdgeSolver(model=model, training_instance_names=training_names))
    records = run_benchmark(
        solvers,
        sizes=arguments.sizes,
        seeds=arguments.seeds,
        exact_max_cities=int(arguments.exact_max_cities),
    )
    output = write_benchmark_csv(records, arguments.output)
    print(f"wrote {len(records)} records to {output}")
    return 0


def _run_summarize(arguments: argparse.Namespace) -> int:
    records = read_benchmark_csv(arguments.input)
    summaries = summarize_benchmark(records)
    output = write_summary_csv(summaries, arguments.output)
    print(f"wrote {len(summaries)} summary rows to {output}")
    return 0


def _run_train(arguments: argparse.Namespace) -> int:
    instances = [
        generate_random_euclidean(n_cities, seed=seed)
        for n_cities in arguments.sizes
        for seed in arguments.seeds
    ]
    config = TrainingConfig(
        epochs=int(arguments.epochs),
        learning_rate=float(arguments.learning_rate),
        hidden_dim=int(arguments.hidden_dim),
        seed=int(arguments.training_seed),
    )
    result = train_edge_model(instances, config=config)
    output = save_edge_model(result, arguments.output)
    print(
        f"trained on {len(instances)} instances; "
        f"final loss={result.losses[-1]:.6f}; saved {output}"
    )
    return 0


def _run_plot(arguments: argparse.Namespace) -> int:
    records = read_benchmark_csv(arguments.input)
    output_dir: Path = arguments.output_dir
    outputs = [plot_optimality_gap(records, output_dir / "optimality-gap.png")]
    runtime_records = (
        read_benchmark_csv(arguments.classical_input)
        if arguments.classical_input is not None
        else records
    )
    outputs.append(plot_runtime(runtime_records, output_dir / "runtime-scaling.png"))
    if any(record.solver.startswith("learned-edge") for record in records):
        outputs.append(
            plot_learned_comparison(records, output_dir / "learned-vs-classical.png")
        )
    if arguments.online_input is not None:
        outputs.append(
            plot_hedge_adaptation(
                read_online_csv(arguments.online_input),
                output_dir / "hedge-adaptation.png",
            )
        )
    if arguments.model is not None:
        model, training_names = load_edge_model(arguments.model)
        instance = generate_random_euclidean(
            int(arguments.tour_cities), seed=int(arguments.tour_seed)
        )
        tour_solvers: tuple[Solver, ...] = (
            HeldKarpSolver(max_cities=20),
            NearestNeighborSolver(),
            TwoOptSolver(),
            LearnedEdgeSolver(model=model, training_instance_names=training_names),
        )
        outputs.append(
            plot_tour_comparison(
                instance,
                [solver.solve(instance) for solver in tour_solvers],
                output_dir / "tour-comparison.png",
            )
        )
    print("wrote " + ", ".join(str(output) for output in outputs))
    return 0


def _run_online(arguments: argparse.Namespace) -> int:
    records = run_online_experiment(
        (
            NearestNeighborSolver(),
            NearestNeighborSolver(best_of_all_starts=True),
            TwoOptSolver(),
        ),
        n_cities=int(arguments.cities),
        seeds=arguments.seeds,
        learning_rate=float(arguments.learning_rate),
    )
    for round_index in sorted({record.round_index for record in records}):
        round_records = [record for record in records if record.round_index == round_index]
        representative = round_records[0]
        print(
            json.dumps(
                {
                    "round": round_index,
                    "seed": representative.seed,
                    "instance": representative.instance_id,
                    "selected_expert": representative.selected_expert,
                    "selected_length": representative.selected_length,
                    "losses": {record.expert: record.loss for record in round_records},
                    "probabilities_before": {
                        record.expert: record.probability_before for record in round_records
                    },
                    "probabilities_after": {
                        record.expert: record.probability_after for record in round_records
                    },
                }
            )
        )
    if arguments.output is not None:
        output = write_online_csv(records, arguments.output)
        print(f"wrote {len(records)} expert-round records to {output}")
    return 0


def _run_summarize_online(arguments: argparse.Namespace) -> int:
    summaries = summarize_online(read_online_csv(arguments.input))
    output = write_online_summary_csv(summaries, arguments.output)
    print(f"wrote {len(summaries)} online summary rows to {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the requested command."""

    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        handlers = {
            "solve": _run_solve,
            "benchmark": _run_benchmark,
            "summarize": _run_summarize,
            "train": _run_train,
            "plot": _run_plot,
            "online": _run_online,
            "summarize-online": _run_summarize_online,
        }
        return handlers[str(arguments.command)](arguments)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
