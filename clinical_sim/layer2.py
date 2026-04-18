"""Layer 2: Stochastic variation engine."""

from __future__ import annotations

import numpy as np

from state import Biomarkers, Effects, Patient, Tolerance, Toxicity, WorldState


def apply_layer2(state: WorldState, rule_tables: dict) -> WorldState:
    rt = rule_tables
    rng = np.random.default_rng(state.meta.rng_seed + state.meta.t)
    bm = state.biomarkers.model_copy()
    eff = state.effects.model_copy()
    tox = state.toxicity.model_copy()

    # ── 2a. Patient modifier — scale effect by covariates ─────────
    patient_modifier = _patient_modifier(state.patient)

    # ── 2b. Dose-response noise on clinical_response ──────────────
    noise = rng.normal(0.0, rt["noise_sd"])
    eff.clinical_response = float(
        np.clip(eff.clinical_response * patient_modifier + noise, 0.0, 1.0)
    )

    # ── 2c. Probabilistic AE sampling ────────────────────────────
    if state.treatment.drug_active:
        ae_prob_adjusted = rt["ae_probability"] * (1.0 + state.toxicity.cumulative_tox * 0.05)
        if rng.random() < ae_prob_adjusted:
            grade = int(rng.choice([0, 1, 2, 3, 4], p=_normalise(rt["ae_severity_weights"])))
            tox.ae_severity = max(tox.ae_severity, grade)
            if grade >= 2 and "sampled_ae" not in tox.ae_active:
                tox.ae_active = tox.ae_active + ["sampled_ae"]

    # ── 2d. Biomarker noise ───────────────────────────────────────
    bm.target_biomarker = float(max(0.0, bm.target_biomarker + rng.normal(0, rt["noise_sd"])))
    bm.inflammatory_score = float(
        np.clip(bm.inflammatory_score + rng.normal(0, rt["noise_sd"] * 0.5), 0.0, 10.0)
    )

    # ── 2e. Prior exposure modifier ───────────────────────────────
    if len(state.patient.exposure_history) > 0:
        prior_auc = sum(state.patient.exposure_history)
        tol_penalty = min(0.3, prior_auc / 10000.0)
        tol = state.tolerance.model_copy(
            update={
                "tolerance_level": min(1.0, state.tolerance.tolerance_level + tol_penalty),
            }
        )
        return state.copy_updated(biomarkers=bm, effects=eff, toxicity=tox, tolerance=tol)

    return state.copy_updated(biomarkers=bm, effects=eff, toxicity=tox)


def _patient_modifier(patient: Patient) -> float:
    """
    Combine patient covariates into a single response scale factor.
    Values > 1.0 = better response, < 1.0 = worse response.
    """
    mod = 1.0
    mod *= 0.7 + 0.3 * patient.renal_function
    if patient.age > 70:
        mod *= 0.85
    cyp = patient.genotype.get("cyp450_metaboliser", "normal")
    if cyp == "poor":
        mod *= 1.2
    elif cyp == "rapid":
        mod *= 0.75
    return float(np.clip(mod, 0.1, 2.0))


def _normalise(weights: list) -> list:
    total = sum(weights)
    return [w / total for w in weights]
