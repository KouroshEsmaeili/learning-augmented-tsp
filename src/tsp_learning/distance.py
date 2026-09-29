"""Distance utilities for symmetric Euclidean TSP instances."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tsp_learning.problem import TSPInstance


def euclidean_distance_matrix(instance: TSPInstance) -> NDArray[np.float64]:
    """Compute the dense pairwise Euclidean distance matrix."""

    offsets = instance.coordinates[:, np.newaxis, :] - instance.coordinates[np.newaxis, :, :]
    distances = np.linalg.norm(offsets, axis=2)
    return np.asarray(distances, dtype=np.float64)
