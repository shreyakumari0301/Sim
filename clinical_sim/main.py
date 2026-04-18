"""
Example entry point: read three processed CSVs for one drug, compile rules, run simulation.

From repo root:
  PYTHONPATH=clinical_sim python clinical_sim/main.py --drug silicea
  PYTHONPATH=clinical_sim python clinical_sim/main.py --drug silicea --llm
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from csv_bundle import build_text_bundle
from llm_compiler import compile_rule_tables, load_repo_dotenv
from loop import run_simulation
from state import Patient, Treatment, WorldState


def _default_processed(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed" / name


def main(argv: list[str] | None = None) -> int:
    load_repo_dotenv()

    p = argparse.ArgumentParser(description="Clinical sim: CSV → LLM rules → run_simulation")
    p.add_argument(
        "--drug",
        default="silicea",
        help="Exact match (case-insensitive) on NCBI drug_name, OpenFDA drug_name_clean, DrugBank name tokens.",
    )
    p.add_argument(
        "--openfda-csv",
        type=Path,
        default=_default_processed("openfda_v1.csv"),
        help="Path to openfda_v1.csv",
    )
    p.add_argument(
        "--ncbi-csv",
        type=Path,
        default=_default_processed("ncbi_data.csv"),
        help="Path to ncbi_data.csv (from notebook export)",
    )
    p.add_argument(
        "--drugbank-csv",
        type=Path,
        default=_default_processed("drugbank.csv"),
        help="Path to drugbank.csv",
    )
    p.add_argument(
        "--llm",
        action="store_true",
        help="Call LLM (OPENAI_API_KEY in env or .env). Use OPENAI_BASE_URL for OpenRouter. Default: dry-run.",
    )
    p.add_argument(
        "--no-show-llm-output",
        action="store_true",
        help="With --llm, do not print the model JSON and merged RuleTable (default is to show).",
    )
    p.add_argument("--timesteps", type=int, default=90, help="Simulation length")
    args = p.parse_args(argv)

    pubmed_text, openfda_text, drugbank_text = build_text_bundle(
        args.drug,
        openfda_csv=args.openfda_csv,
        ncbi_csv=args.ncbi_csv,
        drugbank_csv=args.drugbank_csv,
    )

    print(f"Drug: {args.drug!r}")
    print(f"  PubMed blob length:   {len(pubmed_text)} chars (from {args.ncbi_csv})")
    print(f"  OpenFDA blob length:  {len(openfda_text)} chars (from {args.openfda_csv})")
    print(f"  DrugBank blob length: {len(drugbank_text)} chars (from {args.drugbank_csv})")

    if not pubmed_text and args.ncbi_csv.is_file():
        print(
            "  Note: no PubMed rows matched this drug in ncbi_data.csv "
            "(check spelling vs drug_name column).",
            file=sys.stderr,
        )
    if not args.ncbi_csv.is_file():
        print(f"  Warning: NCBI file missing: {args.ncbi_csv}", file=sys.stderr)

    dry_run = not args.llm
    if args.llm and not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set; falling back to dry_run=True", file=sys.stderr)
        dry_run = True

    rules = compile_rule_tables(
        pubmed_text,
        openfda_text,
        drugbank_text,
        dry_run=dry_run,
        drug=args.drug,
        show_llm_output=args.llm and not dry_run and not args.no_show_llm_output,
    )

    print(f"\nRules: v{rules.version} — {rules.source_summary!r}")

    from budget import TokenBudget

    print(f"Budget: {TokenBudget().status()}")

    patient = Patient(
        age=58,
        weight=80.0,
        renal_function=0.9,
        genotype={"cyp450_metaboliser": "normal", "hla_risk": False},
        exposure_history=[],
    )
    treatment = Treatment(
        drug_active=True,
        dose_level=200.0,
        schedule=list(range(0, 90, 7)),
        arm_assignment="treatment",
    )
    state = WorldState()
    state = state.copy_updated(patient=patient, treatment=treatment)

    history = run_simulation(
        initial_state=state,
        rule_tables=rules.to_dict(),
        n_timesteps=args.timesteps,
        verbose=False,
    )
    final = history[-1]
    print(f"\nFinal t={args.timesteps - 1}:")
    print(f"  Response:    {final.effects.clinical_response:.2f}")
    print(f"  Tolerance:   {final.tolerance.tolerance_level:.3f}")
    print(f"  Resistance:  {final.tolerance.resistance_flag}")
    print(f"  AE severity: {final.toxicity.ae_severity}")
    print(f"  Drug active: {final.treatment.drug_active}")
    print(f"  Disease:     {final.effects.disease_state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
