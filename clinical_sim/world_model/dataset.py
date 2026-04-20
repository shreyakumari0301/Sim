"""Dataset generation utilities for world-model training."""

from __future__ import annotations

from typing import Any

from loop import run_simulation
from state import WorldState
from world_model.adapter import infer_action, patient_context, state_to_vector
from world_model.schema import TransitionRecord


def build_transition_dataset(
    *,
    initial_state: WorldState,
    rule_tables: dict[str, Any],
    drug_name: str,
    n_timesteps: int,
    run_id: str = "run_0",
) -> list[TransitionRecord]:
    """
    Simulate one trajectory and emit supervised transition rows.

    Row i corresponds to transition from history[i] -> history[i+1].
    """
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
    """
    Flatten transition records to dict rows for CSV/Parquet writers.

    Naming convention:
      - state features:   s_<name>
      - action features:  a_<name>
      - target features:  y_<name>
    """
    out: list[dict[str, float | int | str]] = []
    for r in records:
        row: dict[str, float | int | str] = {
            "run_id": r.run_id,
            "drug_name": r.drug_name,
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
