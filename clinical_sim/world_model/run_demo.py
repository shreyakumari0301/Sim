"""CLI demo for generating world-model transitions for one drug."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rule_tables import RuleTable
from state import Treatment, WorldState
from world_model.dataset import build_transition_dataset, transitions_to_rows
from world_model.drug_rules import drug_rule_table


def _default_out(drug: str) -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed" / f"wm_{drug}_demo.csv"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate and inspect world-model transition rows for one drug.")
    p.add_argument("--drug", required=True, help="Drug name label for this generated dataset.")
    p.add_argument("--timesteps", type=int, default=10, help="Number of timesteps to simulate.")
    p.add_argument("--dose", type=float, default=200.0, help="Starting dose level (mg).")
    p.add_argument("--print-rows", type=int, default=2, help="How many rows to print to stdout.")
    p.add_argument("--save-csv", action="store_true", help="Save full transition table to CSV.")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path.")
    p.add_argument(
        "--use-default-rules",
        action="store_true",
        help="Use the same baseline RuleTable for all drugs (debug only).",
    )
    args = p.parse_args(argv)

    state = WorldState().copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=args.dose,
            schedule=list(range(0, max(args.timesteps, 1), 7)),
            arm_assignment="treatment",
        )
    )

    rules = RuleTable().to_dict() if args.use_default_rules else drug_rule_table(args.drug)
    print(
        "rules:",
        {
            "half_life": rules["half_life"],
            "kd": rules["kd"],
            "emax": rules["emax"],
            "pathway_suppression": rules["pathway_suppression"],
            "tox_rate": rules["tox_rate"],
            "ae_probability": rules["ae_probability"],
            "response_threshold": rules["response_threshold"],
            "max_dose": rules["max_dose"],
            "dose_step": rules["dose_step"],
        },
    )

    records = build_transition_dataset(
        initial_state=state,
        rule_tables=rules,
        drug_name=args.drug,
        n_timesteps=args.timesteps,
        run_id=f"demo_{args.drug}",
    )
    rows = transitions_to_rows(records)

    print(f"drug={args.drug} timesteps={args.timesteps} transitions={len(rows)}")
    if rows:
        print("sample_keys:", list(rows[0].keys())[:14])
    n = max(0, min(args.print_rows, len(rows)))
    for i in range(n):
        print(f"\nrow[{i}]")
        print(json.dumps(rows[i], indent=2))

    if args.save_csv:
        out = args.out_csv or _default_out(args.drug)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsaved_csv={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
