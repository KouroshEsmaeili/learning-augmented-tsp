"""Full-information Hedge over a fixed collection of TSP solvers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult
from tsp_learning.solvers.base import Solver


@dataclass(frozen=True, slots=True)
class OnlineRoundResult:
    """Decision, observed expert losses, and posterior weights for one round."""

    round_index: int
    selected_expert: str
    selected_result: SolveResult
    expert_results: dict[str, SolveResult]
    losses: dict[str, float]
    probabilities_before: dict[str, float]
    probabilities_after: dict[str, float]


@dataclass(slots=True)
class HedgeSelector:
    """Adapt solver weights using the multiplicative-weights/Hedge update.

    A round is one fully observed TSP instance. Experts are TSP solvers. Their
    tour lengths are rescaled within the round to losses in [0, 1], and weights
    follow ``w_i <- w_i * exp(-learning_rate * loss_i)``.
    """

    experts: Sequence[Solver]
    learning_rate: float = 0.5
    _weights: dict[str, float] = field(init=False, repr=False)
    _cumulative_losses: dict[str, float] = field(init=False, repr=False)
    _round_index: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        if len(self.experts) < 2:
            raise ValueError("Hedge requires at least two experts")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite value")
        names = [expert.name for expert in self.experts]
        if len(names) != len(set(names)):
            raise ValueError("expert names must be unique")
        initial = 1.0 / len(names)
        self._weights = dict.fromkeys(names, initial)
        self._cumulative_losses = dict.fromkeys(names, 0.0)

    @property
    def probabilities(self) -> dict[str, float]:
        """Return a copy of the current expert distribution."""

        return self._weights.copy()

    @property
    def cumulative_losses(self) -> dict[str, float]:
        """Return each expert's cumulative normalized loss."""

        return self._cumulative_losses.copy()

    def select_expert(self) -> str:
        """Select the highest-weight expert, breaking ties by declaration order."""

        return max(self.experts, key=lambda expert: self._weights[expert.name]).name

    def update(self, losses: Mapping[str, float]) -> None:
        """Apply one Hedge update from losses in the closed interval [0, 1]."""

        expected = set(self._weights)
        if set(losses) != expected:
            raise ValueError("losses must contain exactly one value for every expert")
        for name, loss in losses.items():
            if not math.isfinite(loss) or not 0 <= loss <= 1:
                raise ValueError("expert losses must be finite values in [0, 1]")
            self._weights[name] *= math.exp(-self.learning_rate * loss)
            self._cumulative_losses[name] += loss
        normalizer = sum(self._weights.values())
        self._weights = {name: weight / normalizer for name, weight in self._weights.items()}

    def run_round(self, instance: TSPInstance) -> OnlineRoundResult:
        """Choose an expert, observe all costs, and update the distribution."""

        probabilities_before = self.probabilities
        selected = self.select_expert()
        expert_results = {expert.name: expert.solve(instance) for expert in self.experts}
        lengths = [result.length for result in expert_results.values()]
        minimum = min(lengths)
        maximum = max(lengths)
        if math.isclose(minimum, maximum, rel_tol=1e-12, abs_tol=1e-12):
            losses = dict.fromkeys(expert_results, 0.0)
        else:
            span = maximum - minimum
            losses = {
                name: (result.length - minimum) / span
                for name, result in expert_results.items()
            }
        self.update(losses)
        result = OnlineRoundResult(
            round_index=self._round_index,
            selected_expert=selected,
            selected_result=expert_results[selected],
            expert_results=expert_results,
            losses=losses,
            probabilities_before=probabilities_before,
            probabilities_after=self.probabilities,
        )
        self._round_index += 1
        return result
