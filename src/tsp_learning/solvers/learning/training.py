"""Deterministic supervised training from exact-tour edge labels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from tsp_learning.problem import TSPInstance
from tsp_learning.solvers.base import Solver
from tsp_learning.solvers.exact import HeldKarpSolver
from tsp_learning.solvers.learning.features import EDGE_FEATURE_DIM, edge_feature_tensor
from tsp_learning.solvers.learning.model import EdgeScoringModel

MODEL_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Hyperparameters for small CPU edge-classification experiments."""

    epochs: int = 100
    learning_rate: float = 0.01
    hidden_dim: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """A trained model and the observed full-batch loss history."""

    model: EdgeScoringModel
    losses: tuple[float, ...]
    training_instance_names: tuple[str, ...]
    config: TrainingConfig


def _optimal_edge_set(tour: tuple[int, ...]) -> set[tuple[int, int]]:
    return {
        (
            min(tour[index], tour[(index + 1) % len(tour)]),
            max(tour[index], tour[(index + 1) % len(tour)]),
        )
        for index in range(len(tour))
    }


def build_edge_training_data(
    instances: Sequence[TSPInstance],
    *,
    reference_solver: Solver | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create pair features and binary labels from exact reference tours."""

    if not instances:
        raise ValueError("at least one training instance is required")
    largest = max(instance.n_cities for instance in instances)
    reference = reference_solver or HeldKarpSolver(max_cities=largest)
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    for instance in instances:
        optimal = reference.solve(instance)
        optimal_edges = _optimal_edge_set(optimal.tour)
        row_indices, column_indices = torch.triu_indices(
            instance.n_cities, instance.n_cities, offset=1
        )
        features = edge_feature_tensor(instance)[row_indices, column_indices]
        labels = torch.tensor(
            [
                1.0 if (int(row), int(column)) in optimal_edges else 0.0
                for row, column in zip(row_indices, column_indices, strict=True)
            ],
            dtype=torch.float32,
        )
        feature_batches.append(features)
        label_batches.append(labels)
    return torch.cat(feature_batches), torch.cat(label_batches)


def train_edge_model(
    instances: Sequence[TSPInstance],
    *,
    config: TrainingConfig | None = None,
    reference_solver: Solver | None = None,
) -> TrainingResult:
    """Train a small edge classifier using exact tours as supervision."""

    settings = config or TrainingConfig()
    torch.manual_seed(settings.seed)
    features, labels = build_edge_training_data(instances, reference_solver=reference_solver)
    model = EdgeScoringModel(hidden_dim=settings.hidden_dim)
    positives = float(labels.sum().item())
    negatives = float(labels.numel()) - positives
    positive_weight = torch.tensor(negatives / positives if positives else 1.0)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    losses: list[float] = []
    model.train()
    for _ in range(settings.epochs):
        optimizer.zero_grad()
        loss = loss_function(model(features), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    model.eval()
    names = tuple(instance.name or f"anonymous-{index}" for index, instance in enumerate(instances))
    return TrainingResult(
        model=model,
        losses=tuple(losses),
        training_instance_names=names,
        config=settings,
    )


def save_edge_model(result: TrainingResult, output_path: str | Path) -> Path:
    """Save model weights and enough metadata to reproduce and audit training."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact: dict[str, object] = {
        "format_version": MODEL_FORMAT_VERSION,
        "feature_dim": EDGE_FEATURE_DIM,
        "hidden_dim": result.config.hidden_dim,
        "training_seed": result.config.seed,
        "training_epochs": result.config.epochs,
        "learning_rate": result.config.learning_rate,
        "training_instance_names": result.training_instance_names,
        "state_dict": result.model.state_dict(),
    }
    torch.save(artifact, path)
    return path


def load_edge_model(path: str | Path) -> tuple[EdgeScoringModel, frozenset[str]]:
    """Load a CPU model and the instance names used during training."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("model artifact must contain a dictionary")
    artifact = cast(dict[str, object], raw)
    if artifact.get("format_version") != MODEL_FORMAT_VERSION:
        raise ValueError("unsupported model artifact version")
    if artifact.get("feature_dim") != EDGE_FEATURE_DIM:
        raise ValueError("model artifact uses incompatible edge features")
    hidden_dim = artifact.get("hidden_dim")
    state_dict = artifact.get("state_dict")
    training_names = artifact.get("training_instance_names")
    if not isinstance(hidden_dim, int) or not isinstance(state_dict, Mapping):
        raise ValueError("model artifact is missing architecture or weights")
    if not isinstance(training_names, (tuple, list)) or not all(
        isinstance(name, str) for name in training_names
    ):
        raise ValueError("model artifact has invalid training-instance metadata")
    model = EdgeScoringModel(hidden_dim=hidden_dim)
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state_dict))
    model.eval()
    return model, frozenset(cast(Sequence[str], training_names))
