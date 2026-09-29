"""Tests for exact and classical solvers."""

from itertools import permutations

import numpy as np
import pytest

from tsp_learning.instances import generate_random_euclidean
from tsp_learning.problem import TSPInstance
from tsp_learning.route import tour_length, validate_tour
from tsp_learning.solvers.base import Solver
from tsp_learning.solvers.exact import HeldKarpSolver
from tsp_learning.solvers.heuristics import NearestNeighborSolver, TwoOptSolver, two_opt


def _brute_force_optimum(instance: TSPInstance) -> float:
    return min(
        tour_length(instance, (0, *tail))
        for tail in permutations(range(1, instance.n_cities))
    )


def test_nearest_neighbor_is_deterministic_and_configurable() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]]))

    result = NearestNeighborSolver(start_city=0).solve(instance)
    assert result.tour == (0, 1, 2)
    assert result.length == pytest.approx(6.0)
    assert NearestNeighborSolver(start_city=2).solve(instance).tour == (2, 1, 0)


def test_best_start_nearest_neighbor_checks_every_start() -> None:
    instance = generate_random_euclidean(8, seed=5)
    individual = [NearestNeighborSolver(start_city=start).solve(instance) for start in range(8)]
    best = NearestNeighborSolver(best_of_all_starts=True).solve(instance)

    assert best.length == pytest.approx(min(result.length for result in individual))


def test_two_opt_removes_a_crossing_without_worsening() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    crossing = (0, 2, 1, 3)

    improved = two_opt(instance, crossing)
    validate_tour(improved, 4)
    assert tour_length(instance, improved) == pytest.approx(4.0)
    assert tour_length(instance, improved) <= tour_length(instance, crossing) + 1e-12


def test_two_opt_handles_an_improvement_using_the_closing_edge() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    crossing = (0, 1, 3, 2)

    improved = two_opt(instance, crossing)

    assert tour_length(instance, improved) == pytest.approx(4.0)
    assert tour_length(instance, improved) < tour_length(instance, crossing)


def test_two_opt_solver_implements_common_interface() -> None:
    solver = TwoOptSolver()
    assert isinstance(solver, Solver)
    result = solver.solve(generate_random_euclidean(10, seed=8))
    assert result.length <= float(result.metadata["initial_length"]) + 1e-12


@pytest.mark.parametrize("n_cities", range(2, 8))
@pytest.mark.parametrize("seed", [0, 17])
def test_held_karp_matches_brute_force(n_cities: int, seed: int) -> None:
    instance = generate_random_euclidean(n_cities, seed=seed)
    result = HeldKarpSolver().solve(instance)

    validate_tour(result.tour, n_cities)
    assert result.tour[0] == 0
    assert result.length == pytest.approx(_brute_force_optimum(instance))
    assert result.metadata["optimal"] is True


def test_held_karp_enforces_complexity_limit() -> None:
    with pytest.raises(ValueError, match="limited"):
        HeldKarpSolver(max_cities=5).solve(generate_random_euclidean(6, seed=0))


def test_held_karp_respects_a_nonzero_start_city() -> None:
    instance = generate_random_euclidean(7, seed=23)
    result = HeldKarpSolver(start_city=4).solve(instance)

    assert result.tour[0] == 4
    assert result.length == pytest.approx(_brute_force_optimum(instance))
