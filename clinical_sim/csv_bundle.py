"""
Build the three text blobs for `compile_rule_tables` from processed CSVs.

Column contracts (your exports):

- **ncbi_data.csv**: pmid, title, abstract, authors, journal, pub_date, doi, drug_name
- **openfda_v1.csv**: full label/OpenFDA column set ending with drug_name_clean (see OPENFDA_COLUMNS)
- **drugbank.csv**: drug_id, secondary_ids, name, type, description, indication, state, cas, status,
  targets, interactions

Matching: **exact** normalized equality on ``drug_name`` (NCBI), ``drug_name_clean`` (OpenFDA).

DrugBank: first exact token match on ``name`` (comma/semicolon split); else **word-boundary** match of
query tokens against ``name`` or ``description`` (so ``ibuprofen`` matches ``Ibuprofen sodium``; avoids
``aspirin`` matching inside ``nitroaspirin``). Optional ``drugbank_id`` / env ``DRUGBANK_ID`` pins a row.
Add INN↔common pairs to ``_DRUGBANK_EQUIVALENTS`` where needed.
Truncation before the LLM is applied again inside `compile_rule_tables`.
"""

from __future__ import annotations

import csv
import os
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

# High-signal OpenFDA columns for LLM (smaller blob; override with OPENFDA_PRIORITY_ONLY=0 for full row)
OPENFDA_PRIORITY_COLUMNS = (
    "warnings",
    "mechanism_of_action",
    "indications_and_usage",
    "dosage_and_administration",
    "adverse_reactions",
    "contraindications",
    "drug_interactions",
    "pregnancy_or_breast_feeding",
    "boxed_warning",
    "dosage_forms_and_strengths",
    "warnings_and_cautions",
    "clinical_pharmacology",
    "pharmacokinetics",
    "description",
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


# DrugBank often lists INN (e.g. acetylsalicylic acid) without the common name in tokens.
_DRUGBANK_EQUIVALENTS: dict[str, frozenset[str]] = {
    "aspirin": frozenset({"acetylsalicylic acid", "asa"}),
}


def _drugbank_match_tokens(drug: str) -> set[str]:
    """Normalized names/tokens that should match the same row as ``drug``."""
    d = _norm(drug)
    s: set[str] = {d}
    for k, syns in _DRUGBANK_EQUIVALENTS.items():
        if d == k or d in syns:
            s.add(k)
            s |= syns
    return s


def _word_boundary_match(token: str, hay: str) -> bool:
    """
    True if ``token`` appears as a whole token in ``hay`` (case-insensitive).

    Uses non-word-char boundaries so ``aspirin`` does not match inside ``nitroaspirin``,
    but ``ibuprofen`` matches ``Ibuprofen sodium``.
    """
    if len(token) < 2:
        return False
    # (?<!\w) / (?!\w): word chars include letters; works for Unicode in Python 3 re.
    pat = r"(?<!\w)" + re.escape(token) + r"(?!\w)"
    return re.search(pat, hay, flags=re.IGNORECASE) is not None


def _drugbank_row_matches_query(query: set[str], name: str, description: str) -> bool:
    """Exact name-token match, else word-boundary match in name or description."""
    if not name and not description:
        return False

    if name:
        aliases = _split_drugbank_names(name)
        full = _norm(name)
        if aliases.intersection(query) or full in query:
            return True

    hay = f"{_norm(name)} {_norm(description)}".strip()
    if not hay:
        return False
    for token in query:
        if _word_boundary_match(token, hay):
            return True
    return False


def _optional_drugbank_id(explicit: str | None) -> str | None:
    """Use explicit ``drugbank_id`` if set, else ``DRUGBANK_ID`` from the environment."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    env = os.environ.get("DRUGBANK_ID", "").strip()
    return env or None


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
    max_rows: int | None = None,
    max_chars: int = 24_000,
) -> str:
    """Build PubMed context from ncbi_data.csv (exact ``drug_name`` match per row)."""
    if not ncbi_csv.is_file():
        return ""

    if max_rows is None:
        try:
            max_rows = int(os.environ.get("PUBMED_MAX_ROWS", "3"))
        except ValueError:
            max_rows = 3
    drug_n = _norm(drug)
    chunks: list[str] = []
    n = 0
    with ncbi_csv.open(encoding="utf-8-sig", newline="") as f:
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


def openfda_text_for_drug(
    openfda_csv: Path,
    drug: str,
    *,
    max_chars: int = 24_000,
    priority_only: bool | None = None,
) -> str:
    """Join OpenFDA columns for the first row where ``drug_name_clean`` matches exactly."""
    if not openfda_csv.is_file():
        return ""

    if priority_only is None:
        priority_only = os.environ.get("OPENFDA_PRIORITY_ONLY", "1").strip().lower() in (
            "1",
            "true",
            "yes",
        )

    drug_n = _norm(drug)
    cols = OPENFDA_PRIORITY_COLUMNS if priority_only else OPENFDA_COLUMNS
    with openfda_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if _norm(_row_get(row, "drug_name_clean")) != drug_n:
                continue
            parts: list[str] = []
            for col in cols:
                val = _row_get(row, col)
                if not _is_empty_cell(val):
                    parts.append(f"{col}: {val}")
            return "\n".join(parts)[:max_chars]

    return ""


def _drugbank_row_to_text(row: dict[str, str]) -> str:
    parts: list[str] = []
    for col in DRUGBANK_COLUMNS:
        val = _row_get(row, col)
        if not _is_empty_cell(val):
            parts.append(f"{col}: {val}")
    return "\n".join(parts)


def drugbank_text_for_drug(
    drugbank_csv: Path,
    drug: str,
    *,
    drugbank_id: str | None = None,
    max_chars: int = 24_000,
) -> str:
    """Join DrugBank columns for the best row: prefer ``drugbank_id`` when set, else richest ``name`` match."""
    if not drugbank_csv.is_file():
        return ""

    want_id = _optional_drugbank_id(drugbank_id)
    if want_id:
        best_text = ""
        best_len = -1
        with drugbank_csv.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if _row_get(row, "drug_id").strip() != want_id:
                    continue
                block = _drugbank_row_to_text(row)
                if len(block) > best_len:
                    best_len = len(block)
                    best_text = block
        if best_text:
            return best_text[:max_chars]

    query = _drugbank_match_tokens(drug)
    best_text = ""
    best_len = -1
    with drugbank_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = _row_get(row, "name")
            desc = _row_get(row, "description")
            if not _drugbank_row_matches_query(query, name, desc):
                continue
            block = _drugbank_row_to_text(row)
            if len(block) > best_len:
                best_len = len(block)
                best_text = block

    return best_text[:max_chars] if best_text else ""


def build_text_bundle(
    drug: str,
    *,
    openfda_csv: Path,
    ncbi_csv: Path,
    drugbank_csv: Path,
    pubmed_max_rows: int | None = None,
    openfda_priority_only: bool | None = None,
    drugbank_id: str | None = None,
) -> tuple[str, str, str]:
    """Return (pubmed_text, openfda_text, drugbank_text) for `compile_rule_tables`."""
    pubmed = pubmed_text_for_drug(ncbi_csv, drug, max_rows=pubmed_max_rows)
    openfda = openfda_text_for_drug(openfda_csv, drug, priority_only=openfda_priority_only)
    drugbank = drugbank_text_for_drug(drugbank_csv, drug, drugbank_id=drugbank_id)
    return pubmed, openfda, drugbank
