"""Core representation of a Euclidean traveling-salesperson instance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TSPInstance:
    """A symmetric Euclidean TSP instance with two-dimensional coordinates.

    Coordinates are copied into an immutable ``float64`` array so callers cannot
    accidentally change an instance after a distance matrix has been computed.
    """

    coordinates: NDArray[np.float64]
    name: str | None = None

    def __post_init__(self) -> None:
        coordinates = np.array(self.coordinates, dtype=np.float64, copy=True)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates must have shape (n_cities, 2)")
        if coordinates.shape[0] < 2:
            raise ValueError("an instance must contain at least two cities")
        if not np.isfinite(coordinates).all():
            raise ValueError("coordinates must contain only finite values")
        coordinates.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def n_cities(self) -> int:
        """Return the number of cities."""

        return int(self.coordinates.shape[0])
