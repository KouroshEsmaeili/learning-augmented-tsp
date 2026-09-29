"""Held--Karp dynamic programming for small symmetric TSP instances."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from time import perf_counter

from tsp_learning.distance import euclidean_distance_matrix
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.route import tour_length


@dataclass(slots=True)
class HeldKarpSolver:
    """Solve small TSP instances exactly in O(n^2 2^n) time and O(n 2^n) space.

    ``max_cities`` is an explicit safety limit: dynamic programming is useful for
    reference labels and validation, not for large instances.
    """

    start_city: int = 0
    max_cities: int = 20

    @property
    def name(self) -> str:
        """Return the solver identifier used in experiment output."""

        return "held-karp"

    def solve(self, instance: TSPInstance) -> SolveResult:
        """Return an optimal tour and its length."""

        n_cities = instance.n_cities
        if not 0 <= self.start_city < n_cities:
            raise ValueError("start_city is outside the instance")
        if n_cities > self.max_cities:
            raise ValueError(
                f"Held--Karp is limited to {self.max_cities} cities; received {n_cities}"
            )

        started = perf_counter()
        distances = euclidean_distance_matrix(instance)
        remaining = tuple(city for city in range(n_cities) if city != self.start_city)

        # costs[(mask, j)] is the shortest path from the fixed start through
        # exactly the cities in mask, ending at j. The start is excluded from masks.
        costs: dict[tuple[int, int], float] = {}
        parents: dict[tuple[int, int], int] = {}
        for city in remaining:
            costs[(1 << city, city)] = float(distances[self.start_city, city])

        for subset_size in range(2, len(remaining) + 1):
            for subset in combinations(remaining, subset_size):
                mask = sum(1 << city for city in subset)
                for last in subset:
                    previous_mask = mask ^ (1 << last)
                    candidates = (
                        (
                            costs[(previous_mask, previous)]
                            + float(distances[previous, last]),
                            previous,
                        )
                        for previous in subset
                        if previous != last
                    )
                    best_cost, best_previous = min(candidates)
                    costs[(mask, last)] = best_cost
                    parents[(mask, last)] = best_previous

        full_mask = sum(1 << city for city in remaining)
        optimal_length, last_city = min(
            (
                costs[(full_mask, city)] + float(distances[city, self.start_city]),
                city,
            )
            for city in remaining
        )

        reversed_path = [last_city]
        mask = full_mask
        while len(reversed_path) < len(remaining):
            current = reversed_path[-1]
            previous = parents[(mask, current)]
            mask ^= 1 << current
            reversed_path.append(previous)
        tour = (self.start_city, *reversed(reversed_path))

        measured_length = tour_length(instance, tour, distances)
        if abs(measured_length - optimal_length) > 1e-9:
            raise RuntimeError("Held--Karp reconstruction did not match the optimal cost")
        runtime = perf_counter() - started
        return SolveResult(
            tour=tour,
            length=measured_length,
            runtime_seconds=runtime,
            solver_name=self.name,
            metadata={
                "optimal": True,
                "start_city": self.start_city,
                "dynamic_programming_states": len(costs),
            },
        )
