"""Minimal interface for pluggable world models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from world_model.schema import ActionVector, StateVector


@dataclass(frozen=True)
class WorldModelPrediction:
    next_state_mean: StateVector
    next_state_std: StateVector | None = None


class WorldModel(Protocol):
    def predict_next(
        self,
        state_t: StateVector,
        action_t: ActionVector,
        context: dict[str, float],
    ) -> WorldModelPrediction:
        """Predict next state from current state/action/context."""
        raise NotImplementedError
