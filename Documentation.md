# SIM: A Multi-Source Data Pipeline and Rule-Driven Engine for Clinical Trial Simulation

## Abstract
Clinical trial planning is expensive, slow, and difficult to optimize without large-scale experimentation. This work presents **SIM**, an integrated platform designed with the ultimate objective of simulating clinical trial behavior using multi-source biomedical evidence. The implemented system combines OpenFDA regulatory data, NCBI PubMed literature, and DrugBank drug knowledge into a reproducible data pipeline, then uses these sources to parameterize a three-layer simulation engine. The engine models deterministic pharmacology-inspired effects, stochastic patient-level variation, and adaptive trial policy control. The implementation includes provenance manifests, resilient ingestion, structured extraction, command-line orchestration, and test coverage for ingest and simulation components. Results from implementation-level validation show that the platform is operational end-to-end and suitable as a foundation for iterative trial-design experimentation.

## Index Terms
Clinical trial simulation, OpenFDA, NCBI PubMed, DrugBank, pharmacology modeling, stochastic simulation, decision policy, biomedical data engineering.

## I. Introduction
Clinical trial development requires balancing efficacy, safety, dose strategy, and patient heterogeneity under uncertainty. Traditional workflows depend heavily on costly real-world trial iterations and fragmented evidence sources. The primary objective of this project is to build a practical simulation framework that can approximate trial dynamics before expensive clinical execution.

To achieve this objective, we implemented a unified platform that (1) ingests and standardizes evidence from regulatory, literature, and curated drug datasets, and (2) translates this evidence into a modular clinical simulation engine. The system is designed to support repeatable experimentation, transparent data provenance, and controlled extension for future model refinement.

## II. System Objective and Scope
The ultimate project goal is:

**To simulate clinical trials using real biomedical evidence, enabling early testing of treatment behavior, adverse event dynamics, and policy decisions such as escalation, de-escalation, and discontinuation.**

Current scope includes:
- Multi-source ingest and processing pipeline.
- Rule-table compilation from evidence text.
- Multi-layer trial simulation over timesteps.
- Quality checks through tests and run manifests.

## II-A. Clinical Simulation + World Model — System Overview

This project implements a **hybrid clinical simulation framework** that combines LLM-based knowledge extraction, a mechanistic simulator, and (optionally) a **learned world model** trained on simulator-generated data. The overall goal is to model clinical progression as a **state transition problem**: the next patient state depends on the current state, treatment action, and context (drug parameters, patient covariates). Formally, the system targets approximations of

\[
s_{t+1} = f(s_t, a_t, \text{context}),
\]

so that both **faithful simulation** and **faster predictive rollouts** are possible.

**LLM extraction.** At the front of the pipeline, an LLM converts unstructured biomedical evidence (PubMed, OpenFDA labeling context, DrugBank) into structured parameters. Sources are truncated to controlled lengths (`LLM_*_CHARS` / `LLM_TOTAL_SOURCE_CHARS`) before prompting. The model returns JSON aligned with the `RuleTable` schema (half-life, efficacy bounds, toxicity, response thresholds, trial-control fields, and a short `source_summary`). The implementation rejects weak extractions by default, can retry once with larger context, applies pharmacology priors and calibration guardrails, and uses sparse-extraction profile fallbacks when needed—so runs aim for **high-confidence parameters**, not silent generic defaults.

**RuleTable and drug-specific behavior.** Extracted parameters consolidate into a **RuleTable** that defines PK/PD-style behavior inside the simulator: persistence (half-life), binding proxy (`kd`), maximum effect (`emax`), pathway suppression, toxicity accumulation, adverse-event sampling, and policy thresholds. Different drugs (and different successful extractions) yield different RuleTables and therefore different trajectories.

**Mechanistic simulator as ground truth.** The simulator advances a typed **world state** over discrete timesteps: concentrations, AUC, biomarkers, receptor occupancy, pathway activity, clinical response, symptoms, toxicity, tolerance, treatment status, and meta signals (`response_ema`, `toxicity_ema`, transitions). Actions are inferred from policy effects (e.g., dose change, stop). Given `RuleTable` parameters, the layers produce trajectories used both for cohort metrics and for supervised learning.

**Transition dataset for learning.** Code under `clinical_sim/world_model/` builds **supervised transition rows** from simulator rollouts: state features (`s_*`), inferred actions (`a_*`), and next-state labels (`y_*`). This dataset can be exported (e.g., CSV) to train a world model that approximates the simulator’s transition function—in practice often via **state deltas** \(\Delta s = s_{t+1} - s_t\) for stability. A trained model can support faster rollouts, counterfactuals, and policy search; uncertainty estimates can trigger **fallback to the full simulator** when predictions are unreliable.

**World-model modules.** The current `clinical_sim/world_model/` package includes:
- `schema.py` (typed transition records),
- `interface.py` (world-model API/protocol),
- `adapter.py` (state/action vectorization),
- `dataset.py` (transition row builders),
- `drug_rules.py` (drug-specific RuleTable profiles),
- `run_demo.py` (single-drug transition demo),
- `generate_dataset.py` (scaled multi-drug dataset generation),
- `train_baseline.py` (baseline model training),
- `eval_rollout.py` (one-step and rollout evaluation).

**Outputs.** The stack currently exposes cohort-level summaries (mean response, severe AE rate, drug activity rate, subgroup strata) and, separately, transition tables for ML. The design is **synthetic-ground-truth today** (simulator-as-teacher), with a clear path to incorporate real-world outcomes for calibration and validation later.

Overall, this is a **layered AI–mechanistic** design: knowledge extraction → structured rules → ground-truth dynamics → learned transition model—usable as an early **digital twin–style** foundation for trial strategy experimentation, subject to the limitations in Section VII.

## III. Implemented Architecture
The architecture has two integrated blocks: a data pipeline and a simulation engine.

### A. Data Pipeline (`src/sim`)
1) **Configuration and Paths:** Environment-backed settings and canonical data layout (`raw`, `processed`, `manifests`).
2) **OpenFDA Ingest:** Paginated API fetch with retries, backoff, and JSONL output.
3) **NCBI Ingest and Extraction:** `esearch` + batched `efetch`, with XML/MEDLINE parsing to structured records.
4) **DrugBank Integration:** Licensed XML import and authenticated download support.
5) **Provenance Manifests:** Per-run metadata, status, checksums, and timestamps for reproducibility.
6) **CLI Orchestration:** Commands for initialization, ingest, extraction, and environment-aware diagnostics.

### B. Clinical Simulation Engine (`clinical_sim`)
1) **Typed World State:** Patient, treatment, concentration, biomarkers, effects, toxicity, tolerance, and simulation metadata.
2) **Layer 1 (Mechanistic):** PK decay, dosing absorption, receptor occupancy, pathway suppression, rebound behavior, response and toxicity accumulation.
3) **Layer 2 (Stochastic):** Noise-driven response variation, adverse-event sampling, and covariate-based patient modifiers.
4) **Layer 3 (Policy/Control):** Dose escalation/de-escalation, toxicity halts, and non-response discontinuation logic.
5) **World-State Dynamics Refinement:** timestep-level simulation metadata now tracks `response_ema`, `toxicity_ema`, and `state_transition_count`.
6) **Cohort and Stratified Simulation:** cohort-level simulation is implemented with subgroup summaries across age, renal function, and genotype buckets.
7) **Rule Compilation:** LLM-assisted parameter extraction with token budget controls, weak-extraction rejection, automatic context retry, and extraction-QC reporting (confidence + null-field diagnostics).

## IV. Methods
### A. Evidence Integration Strategy
The pipeline aligns data sources around drug identifiers and cleaned drug-name representations. OpenFDA contributes product and labeling context, NCBI contributes literature evidence, and DrugBank contributes structured drug metadata.

### B. Simulation Flow
At each timestep:
1) The mechanistic layer updates concentration, occupancy, biomarker, response, and tolerance state.
2) The stochastic layer perturbs outcomes based on patient covariates and probabilistic adverse events.
3) The control layer updates treatment policy based on safety and efficacy thresholds.

This layered design separates biological effect modeling from uncertainty and operational policy.

### C. Reproducibility and Validation
Each ingest run writes manifests and checksums. Tests validate URL construction, parser behavior, ingest output persistence, state transitions, and budget controls. This ensures the platform is not only functional but auditable.

## V. Implementation Status and Current Outcomes
The following are fully implemented:
- End-to-end ingest paths for OpenFDA, NCBI, and DrugBank.
- JSONL-first raw retention and structured extraction pipeline.
- Rule-table schema and compiler integration for simulation parameters with extraction quality controls.
- Multi-layer simulation loop with safety and response decision logic.
- World-state progression metrics in the simulation meta layer (`response_ema`, `toxicity_ema`, transitions).
- Cohort simulation runner with subgroup-level aggregated outputs.
- CLI-driven workflow with extraction diagnostics (`confidence`, null-field counts, profile fallback status) and project-level test setup.

These outcomes confirm the platform is ready for iterative calibration and scenario studies aimed at trial strategy optimization.

## VI. Discussion
The most important design decision is the decomposition into deterministic, stochastic, and policy layers. This allows independent tuning of mechanistic assumptions, noise models, and intervention logic. It also improves interpretability compared with monolithic black-box simulation.

Another key implementation choice is keeping LLM usage outside the simulation loop. The loop remains deterministic given a fixed rule table, while LLM extraction is treated as an offline compilation step with explicit budget, weak-extraction rejection, one-shot context retry, and profile-based fallback safeguards.

## VII. Limitations
1) Source quality and naming variability can impact evidence linking.
2) Drug-specific rule extraction quality depends on source relevance and extraction robustness, not just context size.
3) Current rules still rely on profile priors/fallback when extraction is sparse and need broader drug-class calibration for stronger scientific validity.
4) Simulated outcomes are research-oriented and not clinical decision support.

## VIII. Future Work
1) I am implementing richer synonym and ontology normalization (e.g., RxNorm/UMLS/DrugBank alias mapping) to improve cross-source drug entity resolution and prevent fallback to unmatched default-only runs.
2) I am implementing retrospective-outcome calibration (with held-out validation) to improve rule-parameter realism and reduce synthetic response/toxicity drift.
3) I am enforcing LLM-assisted rule compilation for all inference-targeted simulations; dry-run/default-table execution is restricted to engine-testing mode only.
4) I am extending profile-based fallback beyond metformin (e.g., NSAID and oncology-class templates) to reduce unstable default dominance.
5) I will expand cohort analysis with additional subgroup dimensions and outcome metrics.
6) I will expand policy search (adaptive schedules and multi-arm comparison).
7) I will add sensitivity and uncertainty quantification dashboards.

## IX. Conclusion
This project successfully implements a complete foundation for evidence-driven clinical trial simulation. The central objective, simulating trial behavior from integrated biomedical evidence, is already operational at a system level. The current platform supports reproducible ingest, structured evidence extraction, and multi-layer trial dynamics, providing a practical base for higher-fidelity validation and optimization studies.

## X. LLM Format and Execution
### A. LLM output format (strict JSON contract)
The rule compiler (`clinical_sim/llm_compiler.py`) expects one JSON object that maps to the `RuleTable` fields used by the simulator. The model output is parsed, validated, then merged into defaults for any missing fields.

How the LLM works internally (runtime flow):
1) `build_text_bundle` collects drug-scoped text from PubMed, OpenFDA, and DrugBank.
2) Source text is trimmed by `LLM_*_CHARS` (or `LLM_TOTAL_SOURCE_CHARS`) before prompting.
3) `compile_rule_tables` sends a strict JSON-only prompt to the model.
4) The raw response is JSON-parsed; weak extraction is measured by null-field count.
5) If extraction is weak and strict mode is active, one automatic retry runs with larger context.
6) Drug-specific priors/guardrails are applied (including metformin sparse-profile fallback).
7) Final values are validated as `RuleTable`, versioned, and passed into simulation.

Expected JSON shape:
```json
{
  "half_life": 6.0,
  "kd": 80.0,
  "emax": 0.35,
  "mic": 10.0,
  "pathway_suppression": 0.4,
  "tox_rate": 0.002,
  "tolerance_rate": 0.001,
  "tolerance_threshold": 300.0,
  "receptor_recovery": 0.02,
  "rebound_magnitude": 0.15,
  "rebound_decay": 0.1,
  "biomarker_sensitivity": 0.8,
  "response_threshold": 0.04,
  "response_rate_alpha": 2.5,
  "response_rate_beta": 1.5,
  "ae_probability": 0.03,
  "ae_severity_weights": [0.6, 0.25, 0.1, 0.04, 0.01],
  "noise_sd": 0.02,
  "hill_n": 1.2,
  "tox_halt_grade": 3,
  "escalation_threshold": 0.3,
  "escalation_day": 14,
  "response_eval_day": 45,
  "max_dose": 2000.0,
  "dose_step": 500.0,
  "de_escalation_grade": 2,
  "non_response_day": 60,
  "non_response_cutoff": 0.15,
  "grade4_auto_stop": false,
  "source_summary": "Short source-grounded clinical summary."
}
```

Runtime safeguards:
- weak extraction can be rejected (strict behavior by default);
- one automatic retry uses larger context windows;
- extraction QC is printed (`confidence`, null counts, profile fallback);
- sparse metformin extraction can be repaired with profile fallback and guardrails.

### B. How to run
From repo root:

```bash
# 1) Install dependencies
pip install -e ".[dev]"

# 2) Run tests
python -m pytest clinical_sim/tests -q
```

Inference run (LLM-enabled):
```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin
```

Cohort run:
```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --cohort-size 50 --cohort-seed 7
```

Engine-only dry run (no inference validity):
```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug silicea --allow-dry-run
```

Recommended LLM context settings:
```bash
LLM_PUBMED_CHARS=2000
LLM_OPENFDA_CHARS=2000
LLM_DRUGBANK_CHARS=3000
LLM_MAX_NULL_FIELDS=10
LLM_PROFILE_FALLBACK_NULL_FIELDS=10
```

Debugging weak extraction (for inspection only):
```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --allow-weak-extraction
```

## References
[1] OpenFDA, "OpenFDA API," [Online]. Available: https://open.fda.gov/

[2] National Center for Biotechnology Information, "Entrez Programming Utilities Help," [Online]. Available: https://www.ncbi.nlm.nih.gov/books/NBK25501/

[3] DrugBank, "DrugBank Online," [Online]. Available: https://go.drugbank.com/
