"""Generate scaled transition datasets for world-model training (CSV)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from cohort import _sample_patient
from rule_tables import RuleTable
from state import Patient, Treatment, WorldState
from world_model.dataset import build_transition_dataset, transitions_to_rows
from world_model.drug_rules import DEFAULT_DRUGS, drug_rule_table


def _make_initial_state(
    *,
    timesteps: int,
    dose: float,
    rng: np.random.Generator,
) -> WorldState:
    base_patient = Patient(
        age=58.0,
        weight=75.0,
        renal_function=0.9,
        hepatic_function=0.95,
        genotype={"cyp450_metaboliser": "normal", "hla_risk": False},
        exposure_history=[],
    )
    patient = _sample_patient(base_patient, rng)
    treatment = Treatment(
        drug_active=True,
        dose_level=dose,
        schedule=list(range(0, max(timesteps, 1), 7)),
        arm_assignment="treatment",
    )
    st = WorldState()
    return st.copy_updated(patient=patient, treatment=treatment, meta=st.meta.model_copy(update={"rng_seed": int(rng.integers(1, 1_000_000))}))


def generate_scaled_dataset(
    *,
    drugs: list[str],
    runs_per_drug: int,
    timesteps: int,
    dose: float,
    base_seed: int,
    use_default_rules: bool,
) -> list[dict[str, float | int | str]]:
    """Run many trajectories and concatenate flattened rows."""
    all_rows: list[dict[str, float | int | str]] = []
    rng_master = np.random.default_rng(base_seed)

    for drug in drugs:
        for r in range(runs_per_drug):
            run_id = f"{drug}_seed{base_seed}_r{r}"
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            state = _make_initial_state(timesteps=timesteps, dose=dose, rng=rng)
            rules: dict[str, Any] = (
                RuleTable().to_dict() if use_default_rules else drug_rule_table(drug)
            )
            records = build_transition_dataset(
                initial_state=state,
                rule_tables=rules,
                drug_name=drug,
                n_timesteps=timesteps,
                run_id=run_id,
            )
            all_rows.extend(transitions_to_rows(records))
    return all_rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate scaled world-model transition CSV for training.")
    p.add_argument(
        "--drugs",
        default=",".join(DEFAULT_DRUGS),
        help="Comma-separated drug names (built-in profiles): metformin,amoxicillin,ibuprofen",
    )
    p.add_argument("--runs-per-drug", type=int, default=20, help="Simulated trajectories per drug.")
    p.add_argument("--timesteps", type=int, default=40, help="Timesteps per trajectory.")
    p.add_argument("--dose", type=float, default=200.0, help="Starting dose (mg).")
    p.add_argument("--base-seed", type=int, default=42, help="Master RNG seed.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/processed/wm_transitions.csv)",
    )
    p.add_argument(
        "--use-default-rules",
        action="store_true",
        help="Ignore drug profiles; use one RuleTable for all (debug).",
    )
    args = p.parse_args(argv)

    drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    rows = generate_scaled_dataset(
        drugs=drugs,
        runs_per_drug=args.runs_per_drug,
        timesteps=args.timesteps,
        dose=args.dose,
        base_seed=args.base_seed,
        use_default_rules=args.use_default_rules,
    )

    out = args.out or Path(__file__).resolve().parents[2] / "data" / "processed" / "wm_transitions.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print("No rows generated.")
        return 1
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")
    print(f"drugs={drugs} runs_per_drug={args.runs_per_drug} timesteps={args.timesteps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
