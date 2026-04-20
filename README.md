# Sim: A Multi-Source Drug Information Pipeline

## Abstract

This repository implements an end-to-end data engineering workflow that combines **FDA structured product labeling (via OpenFDA)**, **biomedical literature (via NCBI PubMed / Entrez)**, and **curated drug knowledge (via DrugBank XML)**. Raw API responses and large dumps are stored under a configurable data root; analytical work is documented in Jupyter notebooks under `data/processed/`. The goal is to produce analysis-ready tables—especially a normalized **`drug_name_clean`** field—linkable to PubMed abstracts for pharmacology-, toxicity-, and mechanism-oriented retrieval within a fixed publication window (**2020–2026** in the current design).

---

## 1. Introduction

Modern drug informatics rarely relies on a single source. Regulatory labels describe what is approved and how products are presented to consumers; biomedical literature captures mechanisms, toxicology, and clinical nuance; curated databases such as DrugBank organize targets and interactions. This project **ingests** these layers separately, **cleans** OpenFDA records into a consistent drug-name column, and **queries PubMed per drug** using Entrez `esearch` + `efetch`, with optional parsing into structured citation records.

The implementation is packaged as a Python project (`sim`, see `pyproject.toml`) with a CLI entrypoint `sim` that orchestrates downloads and manifests under `DATA_DIR`.

---

## 2. Materials: Data Sources

| Source | Role | Access |
|--------|------|--------|
| **openFDA** (`drug/label`) | Product labels, nested `openfda.*` identifiers, text sections (warnings, clinical pharmacology, etc.) | HTTPS JSON API (paginated) |
| **NCBI Entrez** | PubMed IDs and article XML/text for literature queries | E-utilities (`esearch`, `efetch`) |
| **DrugBank** | Rich XML: drug identity, indication, targets, drug–drug interactions | Full database XML (local file; not redistributed here) |

**Important:** DrugBank content is subject to **DrugBank’s license**. This repository documents *how* it was parsed; it does **not** commit the multi-gigabyte XML file. OpenFDA and PubMed are public; always respect **NCBI’s usage guidelines** (identifying `tool` / `email`, rate limits, optional API key in `.env`).

---

## 3. Methods: Repository Layout and Pipeline Stages

### 3.1 Directory layout (`DATA_DIR`)

| Path | Purpose |
|------|---------|
| `data/raw/openfda/...` | Paginated JSONL from `sim ingest openfda` (one JSON object per line with `results[]`) |
| `data/raw/ncbi/...` | PubMed ingest: `pages.jsonl` (raw `esearch` + `efetch` payloads), optionally `records.jsonl` (parsed articles) |
| `data/raw/drugbank/` | Local `full_database.xml` (when obtained under license) |
| `data/manifests/` | Run manifests (hashes, parameters, status) |
| `data/processed/` | Notebooks and derived tables (e.g. `openfda_v1.csv`; large binaries are gitignored—see below) |

### 3.2 Stage A — OpenFDA ingest

**CLI:** `sim ingest openfda` with parameters such as `--endpoint drug/label`, `--limit-per-page`, `--max-pages`, `--search`.

**Notebook:** `data/processed/openfda.ipynb`

**What the notebook does (in order of the analysis):**

1. **Load** `pages.jsonl` from a raw OpenFDA run; each line contains a `results` array of label objects.
2. **Flatten** nested JSON with `pandas.json_normalize` (initially on the order of **130+ columns** per page).
3. **Concatenate** multiple pages (e.g. two pages of **1,000** records → **2,000** rows in the exploratory workflow captured in the notebook).
4. **Reduce sparsity:** compute column-wise null rates; drop extremely sparse fields (e.g. ≥99% null); drop redundant `*_table` columns when a non-table twin exists (e.g. `clinical_pharmacology` vs `clinical_pharmacology_table`), yielding on the order of **~84** columns before name explosion.
5. **Build `drug_name_clean`:** prefer `openfda.generic_name`, then brand, substance, or `active_ingredient`; normalize to lowercase strings; apply text cleaning (strip marketing phrases like “active ingredient”, regex cleanup, optional “advanced” and noise-removal passes).
6. **Explode** comma-separated multi-ingredient names into **one row per ingredient token** (row count grows from 2,000 to **~2,709** rows in the executed notebook output).
7. **Filter** empty or junk tokens (length, stop-list style exclusions).
8. **Export** `openfda_v1.pkl` (joblib) and `openfda_v1.csv` for downstream use.

The **canonical** column for PubMed queries in the NCBI stage is **`drug_name_clean`** (unique drug counts are inspected in the notebook after cleaning).

### 3.3 Stage B — PubMed (NCBI) ingest linked to OpenFDA

**Code:** `src/sim/ingest/ncbi.py`  
**CLI:** `sim ingest ncbi` with either:

- **Ad-hoc query:** `--term '<Entrez query>'`, or  
- **OpenFDA-driven batch:** `--openfda-csv path/to/openfda_v1.csv --drug-column drug_name_clean` plus **`--start-year` / `--end-year`** (defaults **2020** and **2026**).

**Query construction (OpenFDA mode):** For each distinct cleaned drug name, the pipeline builds a PubMed query of the form:

```text
(<drug>[Title/Abstract]) AND (pharmacology OR toxicity OR "adverse effects" OR mechanism) AND ("2020/01/01"[Date - Publication] : "2026/12/31"[Date - Publication])
```

(Exact year formatting follows the implementation in `_pubmed_term_for_drug`.)

**Retrieval:**

- `esearch`: JSON, paginated with `retmax` / `max_pages`.
- `efetch`: batched by article IDs; `retmode` typically **`xml`** for structured abstracts, with optional `--also-extract` to write parsed **`records.jsonl`** next to **`pages.jsonl`**.

**Why two files:** `pages.jsonl` preserves **raw** API responses for audit and re-parsing; `records.jsonl` holds **one parsed article per line** (title, abstract, authors, DOI, etc.).

**Notebook:** `data/processed/ncbi.ipynb`

**Observed sizes in one completed run (representative):**

| Artifact | Approximate scale (from notebook outputs / file structure) |
|----------|------------------------------------------------------------|
| `pages.jsonl` (OpenFDA-driven PubMed run) | **1,475** lines (one per drug-query page; includes `drug_name`, `query`, `ids`, `esearch`, `efetch_responses`) |
| `records.jsonl` (parsed) | **7,108** rows loaded in the notebook; columns include `pmid`, `title`, `abstract`, `journal`, `pub_date`, `doi`, `format` (`pubmed_xml`), plus provenance fields `source_jsonl_line`, `source_retstart`, `source_efetch_batch` |

These numbers depend on **`retmax`**, **`max_pages`**, and how many drugs remain after CSV deduplication.

### 3.4 Stage C — DrugBank XML flattening

**Notebook:** `data/processed/drugbank.ipynb`

**Method:** `xml.etree.ElementTree.iterparse` over `full_database.xml` in streaming fashion, clearing elements after each `<drug>` to limit memory. For each drug, the notebook extracts identifiers, name, type, description, indication, approval state, CAS, group/status, flattened **target** names, and **drug–drug interaction** partner names, writing chunked CSV (`drugbank_flat.csv` in the notebook).

**Scale:** The notebook enumerates drugs sequentially (e.g. `DB00001`, `DB00002`, …) for sanity checks; the full XML is **very large** (multi-GB) and is **not** committed to git.

---

## 4. Engineering Details

- **HTTP client:** `httpx` with retries/backoff (shared helpers with OpenFDA ingest).
- **Configuration:** `pydantic-settings` (`.env` for `DATA_DIR`, `NCBI_*`, timeouts, optional `HTTP_FORCE_IPV4`).
- **Manifests:** Each ingest writes JSON manifests under `data/manifests/` with hashes and parameters.
- **Tests:** `tests/test_ncbi.py` covers URL construction, ingest/extract smoke tests with mocked HTTP.
- **Progress:** OpenFDA-driven NCBI ingest prints periodic progress (e.g. every 100 drugs) and totals.

---

## 5. Reproducibility

### 5.1 Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 5.2 Configure

Copy `.env.example` to `.env` and set at least `DATA_DIR`, and for NCBI, `NCBI_EMAIL`, `NCBI_TOOL`, and optionally `NCBI_API_KEY`.

### 5.3 Commands (illustrative)

```bash
sim init
sim info

# OpenFDA labels
sim ingest openfda --endpoint drug/label --limit-per-page 1000 --max-pages 2

# PubMed from cleaned drug list (after building openfda_v1.csv)
sim ingest ncbi \
  --db pubmed \
  --openfda-csv data/processed/openfda_v1.csv \
  --drug-column drug_name_clean \
  --start-year 2020 \
  --end-year 2026 \
  --retmax 10 \
  --max-pages 1 \
  --efetch-retmode xml \
  --efetch-rettype "" \
  --also-extract
```

Then open `data/processed/openfda.ipynb`, `ncbi.ipynb`, and `drugbank.ipynb` and **re-point file paths** to your machine’s `DATA_DIR` if paths differ from the notebook author’s WSL/Windows path.

---

## 6. Version control policy (what this repo keeps on GitHub)

- **Tracked:** Application source (`src/sim/`), tests, `pyproject.toml`, and **notebooks** under `data/processed/*.ipynb` (per `.gitignore`).
- **Not tracked:** Raw downloads (`data/raw/**`), manifests, large CSV/PKL/XML/JSONL outputs. Large files were removed from **history** where needed using `git filter-repo` (see `scripts/strip_data_from_git_history.sh`).

This keeps the repository suitable for collaboration without exceeding GitHub’s file-size limits.

---

## 7. Limitations and responsible use

- **PubMed queries** are sensitive to spelling, synonyms, and field tags; recall/precision per drug varies.
- **OpenFDA** reflects label text as published; it is not a substitute for clinical decision-making.
- **DrugBank** requires a valid license for the XML; do not commit proprietary dumps.
- **Publication window** (2020–2026) is a design parameter; widen or narrow via CLI flags when re-running.

---

## 8. References (external)

- OpenFDA API: https://open.fda.gov/
- NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/
- DrugBank: https://go.drugbank.com/ (licensing applies)

---

## 9. Document history

This README summarizes the pipeline and the notebooks `data/processed/openfda.ipynb`, `data/processed/ncbi.ipynb`, and `data/processed/drugbank.ipynb` as they were used to build **`openfda_v1`**, **PubMed extracts**, and **DrugBank flattening** experiments. Exact row counts for your machine may differ slightly from the figures quoted above depending on ingest parameters and API results.

---

## 10. Clinical trial simulation engine (`clinical_sim/`)

A separate, self-contained module implements a **multi-layer trial simulator** (mechanistic PK/PD → stochastic variation → policy/control), plus an **offline LLM rule compiler** with a token budget. It now also includes:

- timestep-level world-state progression metrics (`response_ema`, `toxicity_ema`, `state_transition_count`);
- cohort simulation with stratified subgroup summaries (`age|renal|genotype`).

Dependencies: `pip install -e ".[dev]"` (includes `numpy` and `openai`). Run phase tests:

```bash
pytest clinical_sim/tests -v
```

Example driver (reads your processed CSVs by drug name):

```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug ibuprofen

# Cohort run (stratified summary output)
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --cohort-size 50 --cohort-seed 7

# Engine-only dry run (explicit opt-in, no inference validity)
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug silicea --allow-dry-run
```

Paths default to `data/processed/openfda_v1.csv`, `data/processed/ncbi_data.csv`, and `data/processed/drugbank.csv` (override with `--openfda-csv`, `--ncbi-csv`, `--drugbank-csv`). Text is built in `clinical_sim/csv_bundle.py`: **OpenFDA** and **NCBI** use exact match first, then word-boundary fallback (`metformin` matches `metformin hydrochloride`); **DrugBank** matches tokens in the `name` field. Inference runs now require `OPENAI_API_KEY` (LLM-enabled rule compilation), and weak extractions are rejected by default. Use `--allow-dry-run` only for engine testing and `--allow-weak-extraction` only for debugging.

The simulation loop does **not** call the LLM directly; `compile_rule_tables` does before simulation.

Recent test status in this repo for `clinical_sim/tests`:
- `32 passed, 1 skipped`
- `test_phase7.py` covers world-state progression metrics and cohort simulation outputs.
