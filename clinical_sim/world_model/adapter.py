"""Adapters between WorldState and world-model vectors."""

from __future__ import annotations

from state import WorldState
from world_model.schema import ActionVector, StateVector

_DISEASE_TO_ID = {
    "naive": 0.0,
    "responding": 1.0,
    "stable": 2.0,
    "progressing": 3.0,
    "remission": 4.0,
}

_CYP_TO_ID = {
    "poor": 0.0,
    "normal": 1.0,
    "rapid": 2.0,
}


def state_to_vector(state: WorldState) -> StateVector:
    ds = str(state.effects.disease_state.value)
    return StateVector(
        plasma_conc=float(state.drug.plasma_conc),
        tissue_conc=float(state.drug.tissue_conc),
        cumulative_auc=float(state.drug.cumulative_auc),
        time_above_mic=float(state.drug.time_above_mic),
        target_biomarker=float(state.biomarkers.target_biomarker),
        inflammatory_score=float(state.biomarkers.inflammatory_score),
        receptor_occupancy=float(state.biomarkers.receptor_occupancy),
        pathway_activity=float(state.biomarkers.pathway_activity),
        clinical_response=float(state.effects.clinical_response),
        symptom_score=float(state.effects.symptom_score),
        disease_state_id=float(_DISEASE_TO_ID.get(ds, 0.0)),
        response_flag=1.0 if state.effects.response_flag else 0.0,
        ae_severity=float(state.toxicity.ae_severity),
        cumulative_tox=float(state.toxicity.cumulative_tox),
        tolerance_level=float(state.tolerance.tolerance_level),
        resistance_flag=1.0 if state.tolerance.resistance_flag else 0.0,
        receptor_density=float(state.tolerance.receptor_density),
        rebound_clock=float(state.tolerance.rebound_clock),
        dose_level=float(state.treatment.dose_level),
        drug_active=1.0 if state.treatment.drug_active else 0.0,
        response_ema=float(state.meta.response_ema),
        toxicity_ema=float(state.meta.toxicity_ema),
        state_transition_count=float(state.meta.state_transition_count),
    )


def infer_action(prev_state: WorldState, next_state: WorldState) -> ActionVector:
    """
    Derive a compact action from two consecutive states.

    action_code:
      0 = hold/none
      1 = start
      2 = escalate
      3 = stop
      4 = de-escalate
    """
    prev_active = bool(prev_state.treatment.drug_active)
    next_active = bool(next_state.treatment.drug_active)
    dose_delta = float(next_state.treatment.dose_level - prev_state.treatment.dose_level)

    action_code = 0.0
    if (not prev_active) and next_active:
        action_code = 1.0
    elif prev_active and (not next_active):
        action_code = 3.0
    elif dose_delta > 0.0:
        action_code = 2.0
    elif dose_delta < 0.0:
        action_code = 4.0

    is_dosing_day = 1.0 if prev_state.meta.t in prev_state.treatment.schedule else 0.0
    return ActionVector(
        action_code=action_code,
        dose_delta=dose_delta,
        target_dose=float(next_state.treatment.dose_level),
        was_active=1.0 if prev_active else 0.0,
        is_active=1.0 if next_active else 0.0,
        is_dosing_day=is_dosing_day,
    )


def patient_context(state: WorldState) -> dict[str, float]:
    cyp = str(state.patient.genotype.get("cyp450_metaboliser", "normal")).lower()
    return {
        "age": float(state.patient.age),
        "renal_function": float(state.patient.renal_function),
        "hepatic_function": float(state.patient.hepatic_function),
        "cyp450_metaboliser_id": float(_CYP_TO_ID.get(cyp, 1.0)),
    }
