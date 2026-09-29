"""Deterministic nearest-neighbor construction."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from tsp_learning.distance import euclidean_distance_matrix
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.route import Tour, tour_length


def _tour_from_start(
    distances: NDArray[np.float64],
    start_city: int,
) -> Tour:
    n_cities = int(distances.shape[0])
    unvisited = set(range(n_cities))
    unvisited.remove(start_city)
    tour = [start_city]
    while unvisited:
        current = tour[-1]
        next_city = min(unvisited, key=lambda city: (float(distances[current, city]), city))
        tour.append(next_city)
        unvisited.remove(next_city)
    return tuple(tour)


@dataclass(slots=True)
class NearestNeighborSolver:
    """Build a tour by repeatedly choosing the closest unvisited city."""

    start_city: int = 0
    best_of_all_starts: bool = False

    @property
    def name(self) -> str:
        """Return a stable experiment identifier."""

        return "nearest-neighbor-best-start" if self.best_of_all_starts else "nearest-neighbor"

    def solve(self, instance: TSPInstance) -> SolveResult:
        """Construct a deterministic nearest-neighbor tour."""

        if not 0 <= self.start_city < instance.n_cities:
            raise ValueError("start_city is outside the instance")
        started = perf_counter()
        distances = euclidean_distance_matrix(instance)
        starts = range(instance.n_cities) if self.best_of_all_starts else (self.start_city,)
        candidates = (
            (tour_length(instance, candidate, distances), candidate)
            for start in starts
            for candidate in (_tour_from_start(distances, start),)
        )
        length, tour = min(candidates)
        runtime = perf_counter() - started
        return SolveResult(
            tour=tour,
            length=length,
            runtime_seconds=runtime,
            solver_name=self.name,
            metadata={"start_city": tour[0], "best_of_all_starts": self.best_of_all_starts},
        )
