"""Synthetic Euclidean TSP generators."""

from __future__ import annotations

import numpy as np

from tsp_learning.problem import TSPInstance


def generate_random_euclidean(
    n_cities: int,
    *,
    seed: int | None = None,
) -> TSPInstance:
    """Generate points uniformly in the unit square with NumPy's Generator API."""

    if n_cities < 2:
        raise ValueError("n_cities must be at least two")
    rng = np.random.default_rng(seed)
    coordinates = rng.random((n_cities, 2), dtype=np.float64)
    suffix = "unseeded" if seed is None else str(seed)
    return TSPInstance(coordinates=coordinates, name=f"uniform-n{n_cities}-seed{suffix}")
