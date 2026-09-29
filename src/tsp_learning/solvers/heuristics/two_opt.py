"""2-opt local search for symmetric TSP tours."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from tsp_learning.distance import euclidean_distance_matrix
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.route import Tour, tour_length, validate_tour
from tsp_learning.solvers.base import Solver
from tsp_learning.solvers.heuristics.nearest_neighbor import NearestNeighborSolver


def _two_opt_search(
    initial_tour: Tour,
    distances: NDArray[np.float64],
    tolerance: float,
    max_passes: int | None,
) -> tuple[Tour, int, int]:
    route = list(initial_tour)
    n_cities = len(route)
    passes = 0
    moves = 0
    improved = True
    while improved and (max_passes is None or passes < max_passes):
        improved = False
        passes += 1
        for first in range(n_cities - 1):
            for second in range(first + 2, n_cities):
                if first == 0 and second == n_cities - 1:
                    continue
                a = route[first]
                b = route[first + 1]
                c = route[second]
                d = route[(second + 1) % n_cities]
                delta = (
                    float(distances[a, c])
                    + float(distances[b, d])
                    - float(distances[a, b])
                    - float(distances[c, d])
                )
                if delta < -tolerance:
                    route[first + 1 : second + 1] = reversed(route[first + 1 : second + 1])
                    moves += 1
                    improved = True
                    break
            if improved:
                break
    return tuple(route), passes, moves


def two_opt(
    instance: TSPInstance,
    initial_tour: Tour,
    *,
    tolerance: float = 1e-12,
    max_passes: int | None = None,
) -> Tour:
    """Return a 2-opt local optimum no worse than ``initial_tour``."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if max_passes is not None and max_passes < 1:
        raise ValueError("max_passes must be positive when provided")
    normalized = validate_tour(initial_tour, instance.n_cities)
    improved, _, _ = _two_opt_search(
        normalized,
        euclidean_distance_matrix(instance),
        tolerance,
        max_passes,
    )
    return improved


@dataclass(slots=True)
class TwoOptSolver:
    """Improve the tour from another solver using first-improvement 2-opt."""

    initial_solver: Solver = field(default_factory=NearestNeighborSolver)
    tolerance: float = 1e-12
    max_passes: int | None = None

    @property
    def name(self) -> str:
        """Return a stable experiment identifier."""

        return f"{self.initial_solver.name}-2opt"

    def solve(self, instance: TSPInstance) -> SolveResult:
        """Run the initial solver and improve its tour."""

        if self.tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if self.max_passes is not None and self.max_passes < 1:
            raise ValueError("max_passes must be positive when provided")
        started = perf_counter()
        initial = self.initial_solver.solve(instance)
        distances = euclidean_distance_matrix(instance)
        tour, passes, moves = _two_opt_search(
            validate_tour(initial.tour, instance.n_cities),
            distances,
            self.tolerance,
            self.max_passes,
        )
        length = tour_length(instance, tour, distances)
        if length > initial.length + self.tolerance:
            raise RuntimeError("2-opt produced a tour worse than its input")
        return SolveResult(
            tour=tour,
            length=length,
            runtime_seconds=perf_counter() - started,
            solver_name=self.name,
            metadata={
                "initial_length": initial.length,
                "initial_solver": initial.solver_name,
                "passes": passes,
                "improving_moves": moves,
            },
        )
