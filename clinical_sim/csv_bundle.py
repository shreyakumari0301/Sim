"""
Build the three text blobs for `compile_rule_tables` from processed CSVs.

Column contracts (your exports):

- **ncbi_data.csv**: pmid, title, abstract, authors, journal, pub_date, doi, drug_name
- **openfda_v1.csv**: full label/OpenFDA column set ending with drug_name_clean (see OPENFDA_COLUMNS)
- **drugbank.csv**: drug_id, secondary_ids, name, type, description, indication, state, cas, status,
  targets, interactions

Matching: drug_name (NCBI), drug_name_clean (OpenFDA), DrugBank `name` tokens (comma/semicolon) or substring.
Truncation before the LLM is applied again inside `compile_rule_tables`.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

# NCBI export columns (order preserved in each citation block)
NCBI_COLUMNS = (
    "pmid",
    "title",
    "abstract",
    "authors",
    "journal",
    "pub_date",
    "doi",
    "drug_name",
)

# OpenFDA v1 — all columns you listed (row filter: drug_name_clean)
OPENFDA_COLUMNS = (
    "effective_time",
    "inactive_ingredient",
    "purpose",
    "keep_out_of_reach_of_children",
    "warnings",
    "questions",
    "spl_product_data_elements",
    "version",
    "dosage_and_administration",
    "pregnancy_or_breast_feeding",
    "stop_use",
    "storage_and_handling",
    "do_not_use",
    "package_label_principal_display_panel",
    "indications_and_usage",
    "set_id",
    "id",
    "active_ingredient",
    "openfda.brand_name",
    "openfda.generic_name",
    "openfda.manufacturer_name",
    "openfda.product_ndc",
    "openfda.product_type",
    "openfda.route",
    "openfda.substance_name",
    "openfda.spl_id",
    "openfda.spl_set_id",
    "openfda.package_ndc",
    "openfda.is_original_packager",
    "openfda.upc",
    "openfda.unii",
    "when_using",
    "ask_doctor",
    "openfda.application_number",
    "openfda.rxcui",
    "openfda.nui",
    "openfda.pharm_class_epc",
    "openfda.pharm_class_cs",
    "spl_unclassified_section",
    "description",
    "clinical_pharmacology",
    "pharmacokinetics",
    "microbiology",
    "clinical_studies",
    "contraindications",
    "precautions",
    "general_precautions",
    "information_for_patients",
    "drug_interactions",
    "carcinogenesis_and_mutagenesis_and_impairment_of_fertility",
    "pregnancy",
    "nonteratogenic_effects",
    "nursing_mothers",
    "pediatric_use",
    "geriatric_use",
    "adverse_reactions",
    "how_supplied",
    "recent_major_changes",
    "boxed_warning",
    "dosage_forms_and_strengths",
    "warnings_and_cautions",
    "use_in_specific_populations",
    "overdosage",
    "mechanism_of_action",
    "pharmacodynamics",
    "nonclinical_toxicology",
    "spl_medguide",
    "openfda.original_packager_product_ndc",
    "openfda.pharm_class_moa",
    "spl_patient_package_insert",
    "instructions_for_use",
    "drug_abuse_and_dependence",
    "drug_and_or_laboratory_test_interactions",
    "laboratory_tests",
    "references",
    "teratogenic_effects",
    "animal_pharmacology_and_or_toxicology",
    "openfda.pharm_class_pe",
    "labor_and_delivery",
    "controlled_substance",
    "dependence",
    "ask_doctor_or_pharmacist",
    "other_safety_information",
    "abuse",
    "drug_name_clean",
)

# DrugBank export columns (order preserved in output text)
DRUGBANK_COLUMNS = (
    "drug_id",
    "secondary_ids",
    "name",
    "type",
    "description",
    "indication",
    "state",
    "cas",
    "status",
    "targets",
    "interactions",
)


def _norm(s: str) -> str:
    return str(s).strip().casefold()


def _split_drugbank_names(name: str) -> set[str]:
    return {_norm(p) for p in re.split(r"[,;]", str(name)) if p.strip()}


def _row_get(row: dict[str, str], key: str) -> str:
    v = row.get(key)
    if v is None:
        return ""
    return str(v).strip()


def _is_empty_cell(val: str) -> bool:
    if not val:
        return True
    if val in ("[]", "nan", "None", "NaN"):
        return True
    return False


def pubmed_text_for_drug(
    ncbi_csv: Path,
    drug: str,
    *,
    max_rows: int = 80,
    max_chars: int = 24_000,
) -> str:
    """Build PubMed context from ncbi_data.csv (pmid … drug_name)."""
    if not ncbi_csv.is_file():
        return ""

    drug_n = _norm(drug)
    chunks: list[str] = []
    n = 0
    with ncbi_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return ""
        for row in reader:
            if _norm(_row_get(row, "drug_name")) != drug_n:
                continue
            lines: list[str] = []
            for col in NCBI_COLUMNS:
                val = _row_get(row, col)
                if not _is_empty_cell(val):
                    lines.append(f"{col}: {val}")
            block = "\n".join(lines)
            if block:
                chunks.append(block)
            n += 1
            if n >= max_rows:
                break

    return "\n\n---\n\n".join(chunks)[:max_chars]


def openfda_text_for_drug(openfda_csv: Path, drug: str, *, max_chars: int = 24_000) -> str:
    """Join all OpenFDA v1 columns for the first row where drug_name_clean matches."""
    if not openfda_csv.is_file():
        return ""

    drug_n = _norm(drug)
    with openfda_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _norm(_row_get(row, "drug_name_clean")) != drug_n:
                continue
            parts: list[str] = []
            for col in OPENFDA_COLUMNS:
                val = _row_get(row, col)
                if not _is_empty_cell(val):
                    parts.append(f"{col}: {val}")
            return "\n".join(parts)[:max_chars]

    return ""


def drugbank_text_for_drug(drugbank_csv: Path, drug: str, *, max_chars: int = 24_000) -> str:
    """Join all DrugBank columns for the first row whose name matches this drug."""
    if not drugbank_csv.is_file():
        return ""

    drug_n = _norm(drug)
    with drugbank_csv.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _row_get(row, "name")
            if not name:
                continue
            aliases = _split_drugbank_names(name)
            if drug_n not in aliases and drug_n not in _norm(name):
                continue
            parts: list[str] = []
            for col in DRUGBANK_COLUMNS:
                val = _row_get(row, col)
                if not _is_empty_cell(val):
                    parts.append(f"{col}: {val}")
            return "\n".join(parts)[:max_chars]

    return ""


def build_text_bundle(
    drug: str,
    *,
    openfda_csv: Path,
    ncbi_csv: Path,
    drugbank_csv: Path,
) -> tuple[str, str, str]:
    """Return (pubmed_text, openfda_text, drugbank_text) for `compile_rule_tables`."""
    pubmed = pubmed_text_for_drug(ncbi_csv, drug)
    openfda = openfda_text_for_drug(openfda_csv, drug)
    drugbank = drugbank_text_for_drug(drugbank_csv, drug)
    return pubmed, openfda, drugbank
