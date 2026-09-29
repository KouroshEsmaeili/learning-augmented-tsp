"""Serializable records for full-information online solver selection."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import fmean

from tsp_learning.instances import generate_random_euclidean
from tsp_learning.online import HedgeSelector
from tsp_learning.solvers.base import Solver


@dataclass(frozen=True, slots=True)
class OnlineRecord:
    """One expert's result and probability for one online round."""

    round_index: int
    seed: int
    instance_id: str
    selected_expert: str
    selected_length: float
    expert: str
    expert_length: float
    loss: float
    probability_before: float
    probability_after: float


@dataclass(frozen=True, slots=True)
class OnlineSummary:
    """Aggregate behavior of one expert over an online run."""

    expert: str
    n_rounds: int
    selected_rounds: int
    mean_tour_length: float
    mean_loss: float
    cumulative_loss: float
    final_probability: float


def run_online_experiment(
    experts: Sequence[Solver],
    *,
    n_cities: int,
    seeds: Sequence[int],
    learning_rate: float,
) -> list[OnlineRecord]:
    """Run Hedge on generated instances and return long-form round records."""

    if n_cities < 2:
        raise ValueError("n_cities must be at least two")
    if not seeds:
        raise ValueError("at least one seed is required")
    selector = HedgeSelector(experts=experts, learning_rate=learning_rate)
    records: list[OnlineRecord] = []
    for seed in seeds:
        instance = generate_random_euclidean(n_cities, seed=seed)
        result = selector.run_round(instance)
        instance_id = instance.name or f"n{n_cities}-seed{seed}"
        for expert, expert_result in result.expert_results.items():
            records.append(
                OnlineRecord(
                    round_index=result.round_index,
                    seed=seed,
                    instance_id=instance_id,
                    selected_expert=result.selected_expert,
                    selected_length=result.selected_result.length,
                    expert=expert,
                    expert_length=expert_result.length,
                    loss=result.losses[expert],
                    probability_before=result.probabilities_before[expert],
                    probability_after=result.probabilities_after[expert],
                )
            )
    return records


def summarize_online(records: Sequence[OnlineRecord]) -> list[OnlineSummary]:
    """Summarize expert losses, selection counts, and final probabilities."""

    if not records:
        raise ValueError("at least one online record is required")
    grouped: dict[str, list[OnlineRecord]] = defaultdict(list)
    for record in records:
        grouped[record.expert].append(record)
    selections_by_round = {
        record.round_index: record.selected_expert for record in records
    }
    selected_counts = Counter(selections_by_round.values())
    summaries: list[OnlineSummary] = []
    for expert, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda record: record.round_index)
        summaries.append(
            OnlineSummary(
                expert=expert,
                n_rounds=len(ordered),
                selected_rounds=selected_counts[expert],
                mean_tour_length=fmean(record.expert_length for record in ordered),
                mean_loss=fmean(record.loss for record in ordered),
                cumulative_loss=sum(record.loss for record in ordered),
                final_probability=ordered[-1].probability_after,
            )
        )
    return summaries


def _write_dataclass_csv(
    rows: Iterable[OnlineRecord] | Iterable[OnlineSummary],
    output_path: str | Path,
    row_type: type[OnlineRecord] | type[OnlineSummary],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(row_type)]
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return path


def write_online_csv(records: Iterable[OnlineRecord], output_path: str | Path) -> Path:
    """Write long-form online round records to CSV."""

    return _write_dataclass_csv(records, output_path, OnlineRecord)


def write_online_summary_csv(
    summaries: Iterable[OnlineSummary], output_path: str | Path
) -> Path:
    """Write compact online expert summaries to CSV."""

    return _write_dataclass_csv(summaries, output_path, OnlineSummary)


def read_online_csv(input_path: str | Path) -> list[OnlineRecord]:
    """Read long-form online records produced by :func:`write_online_csv`."""

    records: list[OnlineRecord] = []
    with Path(input_path).open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            records.append(
                OnlineRecord(
                    round_index=int(row["round_index"]),
                    seed=int(row["seed"]),
                    instance_id=row["instance_id"],
                    selected_expert=row["selected_expert"],
                    selected_length=float(row["selected_length"]),
                    expert=row["expert"],
                    expert_length=float(row["expert_length"]),
                    loss=float(row["loss"]),
                    probability_before=float(row["probability_before"]),
                    probability_after=float(row["probability_after"]),
                )
            )
    return records
