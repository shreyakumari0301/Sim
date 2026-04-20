"""World-model package: state/action schemas, adapters, and dataset builders."""

from world_model.adapter import infer_action, patient_context, state_to_vector
from world_model.dataset import build_transition_dataset, transitions_to_rows
from world_model.interface import WorldModel, WorldModelPrediction
from world_model.schema import ActionVector, StateVector, TransitionRecord

__all__ = [
    "ActionVector",
    "StateVector",
    "TransitionRecord",
    "WorldModel",
    "WorldModelPrediction",
    "state_to_vector",
    "infer_action",
    "patient_context",
    "build_transition_dataset",
    "transitions_to_rows",
]
