"""Layer 3: Control / policy engine."""

from __future__ import annotations

from state import Tolerance, Treatment, WorldState


def apply_layer3(state: WorldState, rule_tables: dict) -> WorldState:
    rt = rule_tables
    tx = state.treatment.model_copy()
    tol = state.tolerance.model_copy()
    day = state.meta.trial_day
    disc = rt["discontinuation_rules"]

    # ── 3a. Hard safety stop — grade 4 toxicity ───────────────────
    if state.toxicity.ae_severity >= 4 and disc.get("grade4_auto_stop", True):
        tx.drug_active = False
        tol = _start_rebound(tol, state)
        return state.copy_updated(treatment=tx, tolerance=tol)

    # ── 3b. Toxicity halt (grade >= tox_halt_grade) ───────────────
    if state.toxicity.ae_severity >= rt["tox_halt_grade"]:
        tx.drug_active = False
        tol = _start_rebound(tol, state)
        return state.copy_updated(treatment=tx, tolerance=tol)

    # ── 3c. Non-response discontinuation ─────────────────────────
    if (
        day >= disc["non_response_day"]
        and not state.effects.response_flag
        and state.effects.clinical_response < disc["non_response_cutoff"]
    ):
        tx.drug_active = False
        tol = _start_rebound(tol, state)
        return state.copy_updated(treatment=tx, tolerance=tol)

    # ── 3d. Dose escalation ───────────────────────────────────────
    if (
        day >= rt["escalation_day"]
        and state.effects.clinical_response < rt["escalation_threshold"]
        and tx.drug_active
    ):
        new_dose = min(tx.dose_level + rt["dose_step"], rt["max_dose"])
        tx.dose_level = new_dose
        tx.drug_active = True

    # ── 3e. De-escalation on moderate toxicity ───────────────────
    if (
        state.toxicity.ae_severity >= rt["de_escalation_grade"]
        and tx.drug_active
        and tx.dose_level > rt["dose_step"]
    ):
        tx.dose_level = max(rt["dose_step"], tx.dose_level - rt["dose_step"])

    return state.copy_updated(treatment=tx, tolerance=tol)


def _start_rebound(tol: Tolerance, state: WorldState) -> Tolerance:
    """Start rebound clock only if drug was previously active."""
    if state.treatment.drug_active and tol.rebound_clock == -1:
        return tol.model_copy(update={"rebound_clock": 0})
    return tol
