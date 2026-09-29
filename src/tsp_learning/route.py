"""Tour validation and measurement."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from tsp_learning.distance import euclidean_distance_matrix
from tsp_learning.problem import TSPInstance

Tour = tuple[int, ...]


def validate_tour(tour: Sequence[int], n_cities: int) -> Tour:
    """Validate and normalize a tour.

    Tours contain every city exactly once. The return edge from the last city
    to the first is implicit; repeating the starting city is therefore invalid.
    """

    if n_cities < 2:
        raise ValueError("n_cities must be at least two")
    normalized = tuple(tour)
    if len(normalized) != n_cities:
        raise ValueError(f"tour must contain exactly {n_cities} cities")
    if any(not isinstance(city, int) or isinstance(city, bool) for city in normalized):
        raise TypeError("tour entries must be integers")
    if set(normalized) != set(range(n_cities)):
        raise ValueError("tour must contain each city exactly once")
    return normalized


def tour_length(
    instance: TSPInstance,
    tour: Sequence[int],
    distances: NDArray[np.float64] | None = None,
) -> float:
    """Return the closed-tour length, including the implicit final edge."""

    normalized = validate_tour(tour, instance.n_cities)
    matrix = euclidean_distance_matrix(instance) if distances is None else distances
    expected_shape = (instance.n_cities, instance.n_cities)
    if matrix.shape != expected_shape:
        raise ValueError(f"distance matrix must have shape {expected_shape}")
    indices = np.asarray(normalized, dtype=np.intp)
    successors = np.roll(indices, -1)
    return float(matrix[indices, successors].sum())
