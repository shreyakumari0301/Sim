"""World model: core types, data generation, prediction, metrics."""

from world_model.core import (
    ActionVector,
    StateVector,
    TransitionRecord,
    WorldModel,
    WorldModelPrediction,
    infer_action,
    patient_context,
    state_to_vector,
)
from world_model.data import build_transition_dataset, transitions_to_rows
from world_model.predict import (
    LoadedWorldModel,
    SklearnWorldModel,
    default_artifacts_dir,
    grounded_rollout_from_history,
    load_world_model,
    require_grounded_rollout,
)

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
    "LoadedWorldModel",
    "SklearnWorldModel",
    "load_world_model",
    "default_artifacts_dir",
    "grounded_rollout_from_history",
    "require_grounded_rollout",
]
