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
5) **Rule Compilation:** LLM-assisted parameter extraction with strict token budget controls and deterministic dry-run fallback.

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
- Rule-table schema and compiler integration for simulation parameters.
- Multi-layer simulation loop with safety and response decision logic.
- CLI-driven workflow and project-level test setup.

These outcomes confirm the platform is ready for iterative calibration and scenario studies aimed at trial strategy optimization.

## VI. Discussion
The most important design decision is the decomposition into deterministic, stochastic, and policy layers. This allows independent tuning of mechanistic assumptions, noise models, and intervention logic. It also improves interpretability compared with monolithic black-box simulation.

Another key implementation choice is keeping LLM usage outside the simulation loop. The loop remains deterministic given a fixed rule table, while LLM extraction is treated as an offline compilation step with explicit budget and quality safeguards.

## VII. Limitations
1) Source quality and naming variability can impact evidence linking.
2) Drug-specific rule extraction quality depends on available context.
3) Current rules are generalized defaults and require calibration for domain-specific deployment.
4) Simulated outcomes are research-oriented and not clinical decision support.

## VIII. Future Work
1) I am currently working on richer synonym and ontology normalization for stronger cross-source drug matching.
2) My next implementation will focus on world state models and simulation behavior refinement across timesteps.
3) I will extend the engine toward cohort-level simulation and stratified subgroup analysis.
4) I plan to add calibration pipelines against retrospective outcomes.
5) I will expand policy search (adaptive schedules and multi-arm comparison) and add sensitivity and uncertainty quantification dashboards.

## IX. Conclusion
This project successfully implements a complete foundation for evidence-driven clinical trial simulation. The central objective, simulating trial behavior from integrated biomedical evidence, is already operational at a system level. The current platform supports reproducible ingest, structured evidence extraction, and multi-layer trial dynamics, providing a practical base for higher-fidelity validation and optimization studies.

## References
[1] OpenFDA, "OpenFDA API," [Online]. Available: https://open.fda.gov/

[2] National Center for Biotechnology Information, "Entrez Programming Utilities Help," [Online]. Available: https://www.ncbi.nlm.nih.gov/books/NBK25501/

[3] DrugBank, "DrugBank Online," [Online]. Available: https://go.drugbank.com/
