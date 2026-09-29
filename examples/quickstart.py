"""Compare an exact optimum with two classical baselines on one small instance."""

from tsp_learning.instances import generate_random_euclidean
from tsp_learning.solvers.exact import HeldKarpSolver
from tsp_learning.solvers.heuristics import NearestNeighborSolver, TwoOptSolver


def main() -> None:
    """Run a deterministic in-memory comparison."""

    instance = generate_random_euclidean(9, seed=7)
    for solver in (HeldKarpSolver(), NearestNeighborSolver(), TwoOptSolver()):
        result = solver.solve(instance)
        print(f"{result.solver_name:28s} length={result.length:.6f}")


if __name__ == "__main__":
    main()
