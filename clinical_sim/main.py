"""
Example entry point: read three processed CSVs for one drug, compile rules, run simulation.

From repo root:
  PYTHONPATH=clinical_sim python clinical_sim/main.py --drug ibuprofen
  PYTHONPATH=clinical_sim python clinical_sim/main.py --drug silicea --allow-dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cohort import run_cohort_simulation
from csv_bundle import build_text_bundle
from llm_compiler import compile_rule_tables, get_last_extraction_qc, llm_prompt_char_counts, load_repo_dotenv
from loop import run_simulation
from state import Patient, Treatment, WorldState


def _default_processed(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed" / name


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _boost_llm_context() -> tuple[int, int, int]:
    """
    Increase per-source LLM context windows for retry.

    Defaults: 800/800/800 -> 2000/2000/3000 (PubMed/OpenFDA/DrugBank).
    If env values are already higher, preserve the larger value.
    """
    pm = max(_int_env("LLM_PUBMED_CHARS", 800), 2000)
    of = max(_int_env("LLM_OPENFDA_CHARS", 800), 2000)
    db = max(_int_env("LLM_DRUGBANK_CHARS", 800), 3000)
    os.environ["LLM_PUBMED_CHARS"] = str(pm)
    os.environ["LLM_OPENFDA_CHARS"] = str(of)
    os.environ["LLM_DRUGBANK_CHARS"] = str(db)
    return pm, of, db


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
        "--drugbank-id",
        default=None,
        metavar="DBxxxxx",
        help="Optional DrugBank drug_id for this run (overrides name match). Same as env DRUGBANK_ID.",
    )
    p.add_argument(
        "--allow-dry-run",
        action="store_true",
        help="Allow default-table dry-run when OPENAI_API_KEY is missing. Use only for engine testing, not inference.",
    )
    p.add_argument(
        "--no-show-llm-output",
        action="store_true",
        help="Do not print model JSON and merged RuleTable (default is to show).",
    )
    p.add_argument(
        "--allow-weak-extraction",
        action="store_true",
        help="Allow weak LLM extraction (many null fields) to merge with defaults. Not recommended for inference.",
    )
    p.add_argument(
        "--no-auto-retry-context",
        action="store_true",
        help="Disable automatic one-time retry with larger LLM context after strict weak-extraction failure.",
    )
    p.add_argument("--timesteps", type=int, default=90, help="Simulation length")
    p.add_argument(
        "--cohort-size",
        type=int,
        default=1,
        help="Run a cohort simulation with N patients (default: 1 = single patient mode).",
    )
    p.add_argument(
        "--cohort-seed",
        type=int,
        default=7,
        help="Random seed used for cohort patient sampling (default: 7).",
    )
    args = p.parse_args(argv)

    pubmed_text, openfda_text, drugbank_text = build_text_bundle(
        args.drug,
        openfda_csv=args.openfda_csv,
        ncbi_csv=args.ncbi_csv,
        drugbank_csv=args.drugbank_csv,
        drugbank_id=args.drugbank_id,
    )

    print(f"Drug: {args.drug!r}")
    print(f"  PubMed blob length:   {len(pubmed_text)} chars (from {args.ncbi_csv})")
    print(f"  OpenFDA blob length:  {len(openfda_text)} chars (from {args.openfda_csv})")
    print(f"  DrugBank blob length: {len(drugbank_text)} chars (from {args.drugbank_csv})")
    lp, lo, ld = llm_prompt_char_counts(pubmed_text, openfda_text, drugbank_text)
    print(
        f"  LLM sees (trimmed):   PubMed {lp} + OpenFDA {lo} + DrugBank {ld} chars "
        f"(set LLM_*_CHARS / LLM_TOTAL_SOURCE_CHARS to send more)"
    )

    if not pubmed_text and args.ncbi_csv.is_file():
        print(
            "  Note: no PubMed rows matched this drug in ncbi_data.csv "
            "(check spelling vs drug_name column).",
            file=sys.stderr,
        )
    if not args.ncbi_csv.is_file():
        print(f"  Warning: NCBI file missing: {args.ncbi_csv}", file=sys.stderr)

    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    dry_run = not has_api_key
    if dry_run and not args.allow_dry_run:
        print(
            "OPENAI_API_KEY not set. LLM is required for inference runs; "
            "set OPENAI_API_KEY or use --allow-dry-run for engine testing only.",
            file=sys.stderr,
        )
        return 2
    if dry_run:
        print("Running in explicit dry-run engine-testing mode (--allow-dry-run).", file=sys.stderr)

    reject_weak = not args.allow_weak_extraction
    try:
        rules = compile_rule_tables(
            pubmed_text,
            openfda_text,
            drugbank_text,
            dry_run=dry_run,
            drug=args.drug,
            show_llm_output=not dry_run and not args.no_show_llm_output,
            reject_weak_extraction=reject_weak,
        )
    except ValueError as e:
        weak_msg = "LLM extraction too weak"
        if dry_run or not reject_weak or args.no_auto_retry_context or weak_msg not in str(e):
            raise
        pm, of, db = _boost_llm_context()
        print(
            "Strict extraction failed on first pass; retrying once with increased context "
            f"(PubMed={pm}, OpenFDA={of}, DrugBank={db}).",
            file=sys.stderr,
        )
        rules = compile_rule_tables(
            pubmed_text,
            openfda_text,
            drugbank_text,
            dry_run=False,
            drug=args.drug,
            show_llm_output=not args.no_show_llm_output,
            reject_weak_extraction=True,
        )

    print(f"\nRules: v{rules.version} — {rules.source_summary!r}")
    qc = get_last_extraction_qc()
    if qc:
        print(
            "Extraction QC: "
            f"confidence={qc['confidence']:.2f}, "
            f"nulls={qc['postprocess_null_fields']}/{qc['total_fields']} "
            f"(raw={qc['raw_null_fields']}), "
            f"profile_fallback={qc['profile_fallback_applied']}"
        )

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

    if args.cohort_size > 1:
        out = run_cohort_simulation(
            initial_state=state,
            rule_tables=rules.to_dict(),
            n_patients=args.cohort_size,
            n_timesteps=args.timesteps,
            cohort_seed=args.cohort_seed,
        )
        print("\nCohort summary:")
        print(f"  N:               {out['n_patients']}")
        print(f"  Mean response:   {out['mean_response']:.3f}")
        print(f"  Std response:    {out['std_response']:.3f}")
        print(f"  Severe AE rate:  {out['severe_ae_rate']:.3f}")
        print(f"  Drug active rate:{out['drug_active_rate']:.3f}")
        print("\nSubgroup summary (age|renal|genotype):")
        print(json.dumps(out["subgroup_summary"], indent=2))
    else:
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
        print(f"  Response EMA:{final.meta.response_ema:.3f}")
        print(f"  Toxicity EMA:{final.meta.toxicity_ema:.3f}")
        print(f"  Transitions: {final.meta.state_transition_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
