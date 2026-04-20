"""Cohort-level simulation and subgroup summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from loop import run_simulation
from state import Patient, WorldState


def _age_bucket(age: float) -> str:
    if age < 50:
        return "lt50"
    if age < 65:
        return "50_64"
    return "65plus"


def _renal_bucket(renal_function: float) -> str:
    if renal_function < 0.6:
        return "impaired"
    if renal_function < 0.85:
        return "mild"
    return "normal"


def _genotype_bucket(patient: Patient) -> str:
    return str(patient.genotype.get("cyp450_metaboliser", "unknown"))


def _sample_patient(base: Patient, rng: np.random.Generator) -> Patient:
    age = float(np.clip(rng.normal(base.age, 8.0), 18.0, 90.0))
    weight = float(np.clip(rng.normal(base.weight, 12.0), 35.0, 180.0))
    renal = float(np.clip(rng.normal(base.renal_function, 0.12), 0.2, 1.2))
    hepatic = float(np.clip(rng.normal(base.hepatic_function, 0.1), 0.2, 1.2))

    cyp_choices = ["poor", "normal", "rapid"]
    cyp_probs = [0.15, 0.7, 0.15]
    genotype = dict(base.genotype)
    genotype["cyp450_metaboliser"] = str(rng.choice(cyp_choices, p=cyp_probs))
    genotype["hla_risk"] = bool(rng.random() < 0.1)

    return base.model_copy(
        update={
            "age": age,
            "weight": weight,
            "renal_function": renal,
            "hepatic_function": hepatic,
            "genotype": genotype,
        }
    )


def run_cohort_simulation(
    initial_state: WorldState,
    rule_tables: dict[str, Any],
    *,
    n_patients: int,
    n_timesteps: int,
    cohort_seed: int = 7,
) -> dict[str, Any]:
    """
    Run the simulator for a patient cohort and return aggregate + subgroup summaries.
    """
    if n_patients < 1:
        raise ValueError("n_patients must be at least 1")

    rng = np.random.default_rng(cohort_seed)
    final_rows: list[dict[str, Any]] = []
    subgroup = defaultdict(list)

    for idx in range(n_patients):
        sampled_patient = _sample_patient(initial_state.patient, rng)
        patient_state = initial_state.copy_updated(
            patient=sampled_patient,
            meta=initial_state.meta.model_copy(update={"rng_seed": int(initial_state.meta.rng_seed + idx * 97)}),
        )
        history = run_simulation(
            initial_state=patient_state,
            rule_tables=rule_tables,
            n_timesteps=n_timesteps,
            verbose=False,
        )
        final = history[-1]
        row = {
            "patient_id": idx,
            "age_bucket": _age_bucket(sampled_patient.age),
            "renal_bucket": _renal_bucket(sampled_patient.renal_function),
            "genotype_bucket": _genotype_bucket(sampled_patient),
            "response": float(final.effects.clinical_response),
            "ae_severity": int(final.toxicity.ae_severity),
            "drug_active": bool(final.treatment.drug_active),
            "state_transitions": int(final.meta.state_transition_count),
            "response_ema": float(final.meta.response_ema),
            "toxicity_ema": float(final.meta.toxicity_ema),
        }
        final_rows.append(row)
        key = (row["age_bucket"], row["renal_bucket"], row["genotype_bucket"])
        subgroup[key].append(row)

    responses = np.array([r["response"] for r in final_rows], dtype=float)
    severe_ae_rate = sum(1 for r in final_rows if r["ae_severity"] >= 3) / len(final_rows)
    active_rate = sum(1 for r in final_rows if r["drug_active"]) / len(final_rows)

    subgroup_summary: dict[str, dict[str, float]] = {}
    for (age_b, renal_b, geno_b), rows in subgroup.items():
        rr = np.array([r["response"] for r in rows], dtype=float)
        key = f"{age_b}|{renal_b}|{geno_b}"
        subgroup_summary[key] = {
            "n": float(len(rows)),
            "mean_response": float(rr.mean()),
            "severe_ae_rate": float(sum(1 for r in rows if r["ae_severity"] >= 3) / len(rows)),
            "active_rate": float(sum(1 for r in rows if r["drug_active"]) / len(rows)),
        }

    return {
        "n_patients": n_patients,
        "n_timesteps": n_timesteps,
        "mean_response": float(responses.mean()),
        "std_response": float(responses.std()),
        "severe_ae_rate": float(severe_ae_rate),
        "drug_active_rate": float(active_rate),
        "subgroup_summary": subgroup_summary,
        "patients": final_rows,
    }
