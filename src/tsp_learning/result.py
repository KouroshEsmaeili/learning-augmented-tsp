"""Shared solver result type."""

from __future__ import annotations

from dataclasses import dataclass, field

from tsp_learning.route import Tour

type MetadataValue = str | int | float | bool | None


@dataclass(frozen=True, slots=True)
class SolveResult:
    """A solver's tour, objective value, timing, and diagnostic metadata."""

    tour: Tour
    length: float
    runtime_seconds: float
    solver_name: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.length < 0:
            raise ValueError("length must be non-negative")
        if self.runtime_seconds < 0:
            raise ValueError("runtime_seconds must be non-negative")
        if not self.solver_name:
            raise ValueError("solver_name must not be empty")
