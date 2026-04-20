"""Schemas for world-model state/action transition data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateVector:
    """Flattened simulator state used as model input/output."""

    plasma_conc: float
    tissue_conc: float
    cumulative_auc: float
    time_above_mic: float
    target_biomarker: float
    inflammatory_score: float
    receptor_occupancy: float
    pathway_activity: float
    clinical_response: float
    symptom_score: float
    disease_state_id: float
    response_flag: float
    ae_severity: float
    cumulative_tox: float
    tolerance_level: float
    resistance_flag: float
    receptor_density: float
    rebound_clock: float
    dose_level: float
    drug_active: float
    response_ema: float
    toxicity_ema: float
    state_transition_count: float


@dataclass(frozen=True)
class ActionVector:
    """Control/action signal derived from treatment-policy changes."""

    action_code: float
    dose_delta: float
    target_dose: float
    was_active: float
    is_active: float
    is_dosing_day: float


@dataclass(frozen=True)
class TransitionRecord:
    """Single supervised learning row for next-state prediction."""

    run_id: str
    drug_name: str
    timestep: int
    state_t: StateVector
    action_t: ActionVector
    state_t1: StateVector
    age: float
    renal_function: float
    hepatic_function: float
    cyp450_metaboliser_id: float
