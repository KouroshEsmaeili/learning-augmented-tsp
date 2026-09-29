"""Tests for the online Hedge expert selector."""

from dataclasses import dataclass

import numpy as np
import pytest

from tsp_learning.online import HedgeSelector
from tsp_learning.problem import TSPInstance
from tsp_learning.result import SolveResult


@dataclass
class FixedTourSolver:
    """Small deterministic expert used to isolate Hedge behavior."""

    name: str
    tour: tuple[int, ...]

    def solve(self, instance: TSPInstance) -> SolveResult:
        from tsp_learning.route import tour_length

        length = tour_length(instance, self.tour)
        return SolveResult(self.tour, length, 0.0, self.name)


def test_hedge_shifts_probability_toward_lower_loss() -> None:
    selector = HedgeSelector(
        experts=(FixedTourSolver("first", (0, 1, 2, 3)), FixedTourSolver("second", (0, 2, 1, 3))),
        learning_rate=1.0,
    )
    selector.update({"first": 0.0, "second": 1.0})

    assert selector.probabilities["first"] > selector.probabilities["second"]
    assert sum(selector.probabilities.values()) == pytest.approx(1.0)
    assert selector.select_expert() == "first"


def test_online_round_selects_before_observing_and_updates() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]))
    selector = HedgeSelector(
        experts=(FixedTourSolver("short", (0, 1, 2, 3)), FixedTourSolver("crossing", (0, 2, 1, 3)))
    )

    result = selector.run_round(instance)
    assert result.selected_expert == "short"
    assert result.losses == {"short": 0.0, "crossing": 1.0}
    assert result.probabilities_after["short"] > 0.5


def test_online_round_keeps_uniform_weights_when_expert_costs_match() -> None:
    instance = TSPInstance(np.array([[0.0, 0.0], [1.0, 0.0]]))
    selector = HedgeSelector(
        experts=(FixedTourSolver("forward", (0, 1)), FixedTourSolver("reverse", (1, 0)))
    )

    result = selector.run_round(instance)

    assert result.losses == {"forward": 0.0, "reverse": 0.0}
    assert result.probabilities_after == pytest.approx({"forward": 0.5, "reverse": 0.5})


def test_hedge_validates_losses() -> None:
    selector = HedgeSelector(
        experts=(FixedTourSolver("a", (0, 1)), FixedTourSolver("b", (1, 0)))
    )
    with pytest.raises(ValueError):
        selector.update({"a": 0.0})
    with pytest.raises(ValueError):
        selector.update({"a": -0.1, "b": 0.0})
