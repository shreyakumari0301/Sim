"""Generate scaled transition datasets for world-model training (CSV)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from cohort import _sample_patient
from csv_bundle import list_openfda_drugs_with_nonempty_triple_evidence
from rule_tables import RuleTable
from state import Patient, Treatment, WorldState
from world_model.dataset import build_transition_dataset, transitions_to_rows
from world_model.drug_rules import DEFAULT_DRUGS, drug_rule_table


def _repo_processed_csv(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "processed" / name


def load_drug_names_from_file(path: Path, *, max_drugs: int | None = None) -> list[str]:
    """One drug name per line; `#` starts a comment; empty lines skipped; order preserved."""
    text = path.read_text(encoding="utf-8")
    seen: set[str] = set()
    out: list[str] = []
    for line in text.splitlines():
        v = line.split("#", 1)[0].strip()
        if not v:
            continue
        k = v.casefold()
        if k in seen:
            continue
        seen.add(k)
        out.append(v)
        if max_drugs is not None and len(out) >= max_drugs:
            break
    return out


def load_drug_names_from_csv(
    path: Path,
    column: str,
    *,
    max_drugs: int | None = None,
) -> list[str]:
    """Unique non-empty values from `column` (first occurrence order)."""
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if column not in fields:
            raise ValueError(
                f"column {column!r} not in CSV fields: {fields!r} (file {path})"
            )
        seen: set[str] = set()
        out: list[str] = []
        for row in reader:
            raw = (row.get(column) or "").strip()
            if not raw:
                continue
            k = raw.casefold()
            if k in seen:
                continue
            seen.add(k)
            out.append(raw)
            if max_drugs is not None and len(out) >= max_drugs:
                break
    return out


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
        default=None,
        help="Comma-separated drug names. Default: built-in demo list if no other drug source is set.",
    )
    p.add_argument(
        "--drugs-file",
        type=Path,
        default=None,
        help="Text file: one drug name per line (use for long lists).",
    )
    p.add_argument(
        "--drugs-from-csv",
        type=Path,
        default=None,
        help="CSV path; take unique drug names from --drugs-from-csv-column (e.g. openfda_v1.csv).",
    )
    p.add_argument(
        "--drugs-from-csv-column",
        default="drug_name_clean",
        help="Column name when using --drugs-from-csv (default: drug_name_clean).",
    )
    p.add_argument(
        "--max-drugs",
        type=int,
        default=None,
        help="Cap how many distinct drugs to simulate (order: file/CSV order).",
    )
    p.add_argument(
        "--require-triple-evidence",
        action="store_true",
        help=(
            "Use only drugs whose names appear in OpenFDA and also have non-empty "
            "NCBI (PubMed export), OpenFDA label, and DrugBank matches (same rules as csv_bundle / main.py)."
        ),
    )
    p.add_argument(
        "--openfda-csv",
        type=Path,
        default=_repo_processed_csv("openfda_v1.csv"),
        help="OpenFDA table (used with --require-triple-evidence).",
    )
    p.add_argument(
        "--ncbi-csv",
        type=Path,
        default=_repo_processed_csv("ncbi_data.csv"),
        help="NCBI / PubMed export (used with --require-triple-evidence).",
    )
    p.add_argument(
        "--drugbank-csv",
        type=Path,
        default=_repo_processed_csv("drugbank.csv"),
        help="DrugBank export (used with --require-triple-evidence).",
    )
    p.add_argument(
        "--triple-openfda-column",
        default="drug_name_clean",
        help="Column in OpenFDA CSV to enumerate candidate drug names (default: drug_name_clean).",
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
    if args.max_drugs is not None and args.max_drugs < 1:
        p.error("--max-drugs must be >= 1 when set")

    sources = [
        args.require_triple_evidence,
        args.drugs is not None,
        args.drugs_file is not None,
        args.drugs_from_csv is not None,
    ]
    if sum(bool(x) for x in sources) > 1:
        p.error(
            "use only one drug source: --require-triple-evidence OR "
            "--drugs OR --drugs-file OR --drugs-from-csv"
        )

    if args.require_triple_evidence:
        try:
            drugs = list_openfda_drugs_with_nonempty_triple_evidence(
                openfda_csv=args.openfda_csv,
                ncbi_csv=args.ncbi_csv,
                drugbank_csv=args.drugbank_csv,
                openfda_column=args.triple_openfda_column,
                max_drugs=args.max_drugs,
            )
        except ValueError as e:
            p.error(str(e))
    elif args.drugs_from_csv is not None:
        drugs = load_drug_names_from_csv(
            args.drugs_from_csv,
            args.drugs_from_csv_column,
            max_drugs=args.max_drugs,
        )
    elif args.drugs_file is not None:
        drugs = load_drug_names_from_file(args.drugs_file, max_drugs=args.max_drugs)
    elif args.drugs is not None:
        drugs = [d.strip() for d in args.drugs.split(",") if d.strip()]
    else:
        drugs = list(DEFAULT_DRUGS)

    if not drugs:
        if args.require_triple_evidence:
            p.error(
                "no drugs passed triple-evidence filter; verify --openfda-csv, --ncbi-csv, "
                "--drugbank-csv and that each source has matching non-empty rows"
            )
        p.error("no drug names resolved; check --drugs / --drugs-file / --drugs-from-csv")

    if args.require_triple_evidence:
        print(
            f"require-triple-evidence: {len(drugs)} drug(s) with non-empty "
            "OpenFDA + NCBI (PubMed export) + DrugBank matches"
        )

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
