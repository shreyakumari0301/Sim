"""Layer 1: Mechanistic (deterministic) engine."""

from __future__ import annotations

import math

from state import Biomarkers, DiseaseState, DrugConcentration, Effects, Toxicity, Tolerance, WorldState

TIMESTEP_HOURS = 24.0  # each t = 1 day

# Default rule table for tests that import RT from this module
RT = {
    "half_life": 12.0,
    "kd": 50.0,
    "emax": 0.9,
    "mic": 10.0,
    "pathway_suppression": 0.3,
    "tox_rate": 0.05,
    "tolerance_rate": 0.002,
    "tolerance_threshold": 500.0,
    "receptor_recovery": 0.01,
    "rebound_magnitude": 0.4,
    "rebound_decay": 0.15,
    "biomarker_sensitivity": 2.0,
    "response_threshold": 0.5,
}


def apply_layer1(state: WorldState, rule_tables: dict) -> WorldState:
    rt = rule_tables
    # Rebound phase: do not clamp pathway_activity back to 1.0 via drug-off recovery
    rebound_at_start = state.tolerance.rebound_clock >= 0
    drug = state.drug.model_copy()
    bm = state.biomarkers.model_copy()
    eff = state.effects.model_copy()
    tox = state.toxicity.model_copy()
    tol = state.tolerance.model_copy()

    # ── 1a. PK decay ──────────────────────────────────────────────
    # Renal function modulates effective elimination for mostly renal-cleared drugs.
    # 1.0 -> baseline elimination; lower renal function slows clearance.
    renal_scale = max(0.2, min(1.5, 0.5 + 0.5 * state.patient.renal_function))
    k_elim = (math.log(2) / rt["half_life"]) * renal_scale
    drug.plasma_conc = drug.plasma_conc * math.exp(-k_elim * TIMESTEP_HOURS)
    drug.tissue_conc = drug.plasma_conc * 0.6  # simplified tissue distribution

    # Dose absorption: if today is a dosing day, add dose to plasma
    if state.meta.t in state.treatment.schedule and state.treatment.drug_active:
        absorbed = state.treatment.dose_level * _bioavailability(state.patient)
        drug.plasma_conc += absorbed

    # AUC accumulation
    drug.cumulative_auc += drug.plasma_conc * TIMESTEP_HOURS
    drug.time_above_mic = (
        drug.time_above_mic + 1 if drug.plasma_conc > rt["mic"] else drug.time_above_mic
    )

    # ── 1b. Receptor occupancy (Hill equation) ────────────────────
    bm.receptor_occupancy = _hill(drug.plasma_conc, rt["kd"], rt.get("hill_n", 1.0)) * tol.receptor_density

    # ── 1c. Pathway activity suppression ─────────────────────────
    suppression = bm.receptor_occupancy * rt["pathway_suppression"]
    bm.pathway_activity = max(0.0, bm.pathway_activity - suppression)

    # ── 1d. Rebound mechanism (biological feedback) ───────────────
    if tol.rebound_clock >= 0:
        overshoot = rt["rebound_magnitude"] * math.exp(-rt["rebound_decay"] * tol.rebound_clock)
        # Allow transient pathway activity above baseline during rebound (spec test expects > 1.0)
        bm.pathway_activity = min(2.0, 1.0 + overshoot)
        tol.rebound_clock += 1
        if overshoot < 0.01:
            tol.rebound_clock = -1

    # ── 1e. Target biomarker (causal outcome) ─────────────────────
    net_effect = rt["emax"] * bm.receptor_occupancy * (1.0 - tol.tolerance_level)
    bm.target_biomarker = max(0.0, bm.target_biomarker - net_effect * rt["biomarker_sensitivity"])

    # ── 1f. Clinical response ─────────────────────────────────────
    eff.clinical_response = min(1.0, net_effect)
    eff.symptom_score = max(0.0, 10.0 - eff.clinical_response * 10.0)
    eff.response_flag = eff.clinical_response >= rt["response_threshold"]

    # ── 1g. Toxicity accumulation ─────────────────────────────────
    if state.treatment.drug_active:
        # Reduced renal function increases drug accumulation-related toxicity pressure.
        renal_tox_factor = 1.0 + max(0.0, 1.0 - state.patient.renal_function) * 0.6
        tox_increment = rt["tox_rate"] * bm.receptor_occupancy * renal_tox_factor
        tox.cumulative_tox += tox_increment
        # Baseline severity from cumulative toxicity; stochastic layer can transiently raise this.
        tox.ae_severity = max(tox.ae_severity, _tox_grade(tox.cumulative_tox))

    # ── 1h. Tolerance accumulation ────────────────────────────────
    if drug.cumulative_auc > rt["tolerance_threshold"] and state.treatment.drug_active:
        tol.tolerance_level = min(1.0, tol.tolerance_level + rt["tolerance_rate"])
        tol.receptor_density = max(0.0, tol.receptor_density - rt["tolerance_rate"] * 0.5)

    # Receptor recovery when drug stopped (pathway clamp only if not in rebound overshoot)
    if not state.treatment.drug_active:
        tol.receptor_density = min(1.0, tol.receptor_density + rt["receptor_recovery"])
        if not rebound_at_start:
            bm.pathway_activity = min(1.0, bm.pathway_activity + rt["receptor_recovery"])

    # Disease state transition
    eff.disease_state = _disease_transition(eff, bm)

    return state.copy_updated(drug=drug, biomarkers=bm, effects=eff, toxicity=tox, tolerance=tol)


def _hill(conc: float, kd: float, hill_n: float = 1.0) -> float:
    """Hill equation: fraction saturation = C^n / (C^n + Kd^n)."""
    if conc <= 0:
        return 0.0
    n = max(0.5, float(hill_n))
    c_n = conc**n
    kd_n = max(1e-9, kd**n)
    return c_n / (c_n + kd_n)


def _bioavailability(patient) -> float:
    """Scale dose absorption by organ function."""
    return patient.hepatic_function * 0.8  # simplified F


def _tox_grade(cumulative_tox: float) -> int:
    """Map cumulative toxicity to CTCAE grade 0-4."""
    if cumulative_tox < 2.0:
        return 0
    if cumulative_tox < 5.0:
        return 1
    if cumulative_tox < 9.0:
        return 2
    if cumulative_tox < 14.0:
        return 3
    return 4


def _disease_transition(eff: Effects, bm: Biomarkers) -> DiseaseState:
    if eff.clinical_response >= 0.8:
        return DiseaseState.REMISSION
    if eff.clinical_response >= 0.5:
        return DiseaseState.RESPONDING
    if eff.clinical_response >= 0.2:
        return DiseaseState.STABLE
    return DiseaseState.PROGRESSING
