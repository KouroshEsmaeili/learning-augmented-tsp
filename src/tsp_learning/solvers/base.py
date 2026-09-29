"""Common interface implemented by all TSP solvers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult


@runtime_checkable
class Solver(Protocol):
    """Structural interface for exact, heuristic, and learned solvers."""

    @property
    def name(self) -> str:
        """Return a stable name for reports and experiment records."""

        ...

    def solve(self, instance: TSPInstance) -> SolveResult:
        """Solve an instance and return a valid closed tour."""

        ...
