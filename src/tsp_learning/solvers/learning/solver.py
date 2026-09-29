"""Construct tours greedily from learned edge scores."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch

from tsp_learning.distance import euclidean_distance_matrix
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.route import tour_length
from tsp_learning.solvers.heuristics import two_opt
from tsp_learning.solvers.learning.features import edge_feature_tensor
from tsp_learning.solvers.learning.model import EdgeScoringModel


@dataclass(slots=True)
class LearnedEdgeSolver:
    """Use neural edge logits in a deterministic greedy construction."""

    model: EdgeScoringModel
    start_city: int = 0
    apply_two_opt: bool = True
    training_instance_names: frozenset[str] = field(default_factory=frozenset)

    @property
    def name(self) -> str:
        """Return a stable experiment identifier."""

        suffix = "-2opt" if self.apply_two_opt else ""
        return f"learned-edge{suffix}"

    def solve(self, instance: TSPInstance) -> SolveResult:
        """Construct a tour, rejecting known train/evaluation overlap."""

        if not 0 <= self.start_city < instance.n_cities:
            raise ValueError("start_city is outside the instance")
        if instance.name is not None and instance.name in self.training_instance_names:
            raise ValueError(f"instance {instance.name!r} was used to train this model")

        started = perf_counter()
        with torch.no_grad():
            scores = np.asarray(self.model(edge_feature_tensor(instance)).cpu().numpy())
        distances = euclidean_distance_matrix(instance)
        unvisited = set(range(instance.n_cities))
        unvisited.remove(self.start_city)
        route = [self.start_city]
        while unvisited:
            current = route[-1]
            next_city = min(
                unvisited,
                key=lambda city: (
                    -float(scores[current, city]),
                    float(distances[current, city]),
                    city,
                ),
            )
            route.append(next_city)
            unvisited.remove(next_city)
        tour = tuple(route)
        constructive_length = tour_length(instance, tour, distances)
        if self.apply_two_opt:
            tour = two_opt(instance, tour)
        length = tour_length(instance, tour, distances)
        return SolveResult(
            tour=tour,
            length=length,
            runtime_seconds=perf_counter() - started,
            solver_name=self.name,
            metadata={
                "start_city": self.start_city,
                "constructive_length": constructive_length,
                "two_opt": self.apply_two_opt,
            },
        )
