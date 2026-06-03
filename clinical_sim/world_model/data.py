"""Simulator transitions and drug profiles for world-model training."""

from __future__ import annotations

import hashlib
from typing import Any

from loop import run_simulation
from rule_tables import RuleTable
from state import WorldState
from world_model.core import (
    TransitionRecord,
    infer_action,
    patient_context,
    state_to_vector,
)

DEFAULT_DRUGS = ("metformin", "amoxicillin", "ibuprofen")


def drug_rule_table(drug: str) -> dict[str, Any]:
    d = drug.strip().casefold()
    base = RuleTable().to_dict()
    if d == "metformin":
        base.update(
            {
                "half_life": 6.0,
                "kd": 80.0,
                "emax": 0.35,
                "pathway_suppression": 0.4,
                "tox_rate": 0.002,
                "ae_probability": 0.03,
                "response_threshold": 0.04,
                "max_dose": 2000.0,
                "dose_step": 500.0,
            }
        )
    elif d == "amoxicillin":
        base.update(
            {
                "half_life": 1.2,
                "kd": 20.0,
                "emax": 0.75,
                "pathway_suppression": 0.7,
                "tox_rate": 0.01,
                "ae_probability": 0.05,
                "response_threshold": 0.12,
                "max_dose": 1000.0,
                "dose_step": 250.0,
            }
        )
    elif d == "ibuprofen":
        base.update(
            {
                "half_life": 2.0,
                "kd": 40.0,
                "emax": 0.55,
                "pathway_suppression": 0.5,
                "tox_rate": 0.02,
                "ae_probability": 0.07,
                "response_threshold": 0.1,
                "max_dose": 800.0,
                "dose_step": 200.0,
            }
        )
    return base


def drug_id_for_name(drug: str) -> float:
    key = drug.strip().casefold()
    if not key:
        return 0.0
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") % (2**52)
    return float(u)


def build_transition_dataset(
    *,
    initial_state: WorldState,
    rule_tables: dict[str, Any],
    drug_name: str,
    n_timesteps: int,
    run_id: str = "run_0",
) -> list[TransitionRecord]:
    history = run_simulation(
        initial_state=initial_state,
        rule_tables=rule_tables,
        n_timesteps=n_timesteps,
        verbose=False,
    )
    rows: list[TransitionRecord] = []
    for t in range(len(history) - 1):
        s_t = history[t]
        s_t1 = history[t + 1]
        ctx = patient_context(s_t)
        rows.append(
            TransitionRecord(
                run_id=run_id,
                drug_name=drug_name,
                timestep=t,
                state_t=state_to_vector(s_t),
                action_t=infer_action(s_t, s_t1),
                state_t1=state_to_vector(s_t1),
                age=ctx["age"],
                renal_function=ctx["renal_function"],
                hepatic_function=ctx["hepatic_function"],
                cyp450_metaboliser_id=ctx["cyp450_metaboliser_id"],
            )
        )
    return rows


def transitions_to_rows(records: list[TransitionRecord]) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for r in records:
        row: dict[str, float | int | str] = {
            "run_id": r.run_id,
            "drug_name": r.drug_name,
            "drug_id": drug_id_for_name(r.drug_name),
            "timestep": r.timestep,
            "age": r.age,
            "renal_function": r.renal_function,
            "hepatic_function": r.hepatic_function,
            "cyp450_metaboliser_id": r.cyp450_metaboliser_id,
        }
        for k, v in r.state_t.__dict__.items():
            row[f"s_{k}"] = float(v)
        for k, v in r.action_t.__dict__.items():
            row[f"a_{k}"] = float(v)
        for k, v in r.state_t1.__dict__.items():
            row[f"y_{k}"] = float(v)
        out.append(row)
    return out
