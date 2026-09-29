"""Deterministic tests for learned features, training, and inference."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tsp_learning.instances import generate_random_euclidean
from tsp_learning.problem import TSPInstance
from tsp_learning.route import validate_tour
from tsp_learning.solvers.learning import (
    LearnedEdgeSolver,
    TrainingConfig,
    load_edge_model,
    save_edge_model,
    train_edge_model,
)
from tsp_learning.solvers.learning.features import edge_feature_array


def test_edge_features_are_translation_and_scale_invariant() -> None:
    original = TSPInstance(np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]]))
    transformed = TSPInstance(original.coordinates * 7.0 + np.array([11.0, -4.0]))

    np.testing.assert_allclose(edge_feature_array(original), edge_feature_array(transformed))


def test_training_is_reproducible_and_artifact_prevents_overlap(tmp_path: Path) -> None:
    instances = [generate_random_euclidean(5, seed=seed) for seed in (10, 11)]
    config = TrainingConfig(epochs=4, hidden_dim=8, seed=9)
    first = train_edge_model(instances, config=config)
    second = train_edge_model(instances, config=config)

    assert first.losses == pytest.approx(second.losses)
    for left, right in zip(first.model.parameters(), second.model.parameters(), strict=True):
        assert torch.equal(left, right)

    artifact = save_edge_model(first, tmp_path / "model.pt")
    model, training_names = load_edge_model(artifact)
    solver = LearnedEdgeSolver(model=model, training_instance_names=training_names)
    with pytest.raises(ValueError, match="used to train"):
        solver.solve(instances[0])

    evaluation = generate_random_euclidean(6, seed=99)
    result = solver.solve(evaluation)
    validate_tour(result.tour, evaluation.n_cities)
    assert result.metadata["two_opt"] is True
