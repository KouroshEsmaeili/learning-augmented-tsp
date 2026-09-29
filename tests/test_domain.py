"""Tests for instances, distances, tours, and synthetic generation."""

import numpy as np
import pytest

from tsp_learning.distance import euclidean_distance_matrix
from tsp_learning.instances import generate_random_euclidean
from tsp_learning.problem import TSPInstance
from tsp_learning.route import tour_length, validate_tour


def test_instance_copies_and_freezes_coordinates() -> None:
    coordinates = np.array([[0.0, 0.0], [1.0, 0.0]])
    instance = TSPInstance(coordinates)
    coordinates[0, 0] = 99.0

    assert instance.n_cities == 2
    assert instance.coordinates[0, 0] == 0.0
    with pytest.raises(ValueError):
        instance.coordinates[0, 0] = 2.0


@pytest.mark.parametrize(
    "coordinates",
    [
        np.array([0.0, 1.0]),
        np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]),
        np.array([[0.0, 0.0]]),
        np.array([[0.0, 0.0], [np.inf, 1.0]]),
    ],
)
def test_instance_rejects_invalid_coordinates(coordinates: np.ndarray) -> None:
    with pytest.raises(ValueError):
        TSPInstance(coordinates)


def test_euclidean_distance_matrix_is_symmetric() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 0.0]]))
    distances = euclidean_distance_matrix(instance)

    np.testing.assert_allclose(
        distances,
        np.array([[0.0, 5.0, 3.0], [5.0, 0.0, 4.0], [3.0, 4.0, 0.0]]),
    )
    np.testing.assert_allclose(distances, distances.T)


def test_tour_validation_uses_implicit_closing_edge() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))

    assert validate_tour([0, 1, 2, 3], 4) == (0, 1, 2, 3)
    assert tour_length(instance, [0, 1, 2, 3]) == pytest.approx(4.0)
    with pytest.raises(ValueError):
        validate_tour([0, 1, 2, 3, 0], 4)
    with pytest.raises(ValueError):
        validate_tour([0, 1, 1, 3], 4)
    with pytest.raises(TypeError):
        validate_tour([0, 1, 2, 3.0], 4)


def test_generation_is_seeded_and_uses_the_unit_square() -> None:
    first = generate_random_euclidean(7, seed=42)
    second = generate_random_euclidean(7, seed=42)
    different = generate_random_euclidean(7, seed=43)

    np.testing.assert_array_equal(first.coordinates, second.coordinates)
    assert not np.array_equal(first.coordinates, different.coordinates)
    assert np.all((first.coordinates >= 0.0) & (first.coordinates < 1.0))
    assert first.name == "uniform-n7-seed42"
