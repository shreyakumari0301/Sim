# Clinical simulation & world model — command reference

Run these from the **repository root** unless noted. For modules under `clinical_sim/`, set `PYTHONPATH=clinical_sim` or run from the `clinical_sim/` directory as indicated.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

`scikit-learn` is included in `[dev]` for world-model training.

---

## 2. Tests

```bash
python -m pytest clinical_sim/tests -q
```

---

## 3. Main simulator (`clinical_sim/main.py`)

From repo root:

```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug ibuprofen
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --cohort-size 50 --cohort-seed 7
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug silicea --allow-dry-run
```

---

## 4. LLM context (optional)

```bash
export LLM_PUBMED_CHARS=2000
export LLM_OPENFDA_CHARS=2000
export LLM_DRUGBANK_CHARS=3000
export LLM_MAX_NULL_FIELDS=10
export LLM_PROFILE_FALLBACK_NULL_FIELDS=10
```

Debug weak extraction (not for inference):

```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --allow-weak-extraction
```

---

## 5. World model — quick demo (one drug, printed rows)

From `clinical_sim/`:

```bash
cd clinical_sim
python -m world_model.run_demo --drug metformin --timesteps 10 --print-rows 2
python -m world_model.run_demo --drug metformin --timesteps 10 --save-csv
```

Same baseline rules for all drugs (debug):

```bash
python -m world_model.run_demo --drug metformin --use-default-rules --print-rows 1
```

---

## 6. World model — scaled dataset (CSV)

From `clinical_sim/`:

```bash
cd clinical_sim
python -m world_model.generate_dataset --drugs metformin,amoxicillin,ibuprofen --runs-per-drug 20 --timesteps 40 --out ../data/processed/wm_transitions.csv
```

Custom seed and dose:

```bash
python -m world_model.generate_dataset --drugs metformin --runs-per-drug 50 --timesteps 60 --base-seed 42 --dose 200 --out ../data/processed/wm_metformin.csv
```

---

## 7. World model — train baseline (Random Forest)

From `clinical_sim/` (after generating a CSV):

```bash
cd clinical_sim
python -m world_model.train_baseline --csv ../data/processed/wm_transitions.csv --out-dir world_model/artifacts
```

With options:

```bash
python -m world_model.train_baseline --csv ../data/processed/wm_transitions.csv --out-dir world_model/artifacts --n-estimators 120 --max-depth 16 --test-fraction 0.2 --seed 42
```

Artifacts written:

- `world_model/artifacts/world_model_rf.joblib` (or `.pkl` if `joblib` unavailable)
- `world_model/artifacts/world_model_meta.json`

---

## 8. World model — evaluate rollout

Uses the same CSV and trained `artifacts/` directory:

```bash
cd clinical_sim
python -m world_model.eval_rollout --csv ../data/processed/wm_transitions.csv --model-dir world_model/artifacts --horizon 15
```

**Note:** one-step MAE on the **full** CSV is optimistic if that CSV was used for training; for honest metrics, evaluate on a **held-out** CSV generated with a different `--base-seed` or held-out drugs.

---

## 9. One-liner: tiny pipeline (generate → train → eval)

From `clinical_sim/`:

```bash
cd clinical_sim
python -m world_model.generate_dataset --drugs metformin,ibuprofen --runs-per-drug 8 --timesteps 25 --out ../data/processed/wm_small.csv
python -m world_model.train_baseline --csv ../data/processed/wm_small.csv --out-dir world_model/artifacts --n-estimators 40 --max-depth 12
python -m world_model.eval_rollout --csv ../data/processed/wm_small.csv --model-dir world_model/artifacts --horizon 10
```
