# Clinical simulation & world model — command reference

Run from **repository root** unless noted. For `clinical_sim/` modules: `PYTHONPATH=clinical_sim` or `cd clinical_sim`.

See `clinical_sim/world_model/README.md` for world-model layout.

---

## 1. Environment

```bash
python -m venv .venv
source .venv/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

---

## 2. Tests

```bash
python -m pytest clinical_sim/tests -q
```

---

## 3. Main simulator (grounded world model required)

Single-patient runs **always** print WM vs simulator at horizon. Train artifacts first (`world_model/artifacts/`).

```bash
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --allow-dry-run
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug ibuprofen --wm-horizon 20
PYTHONPATH=clinical_sim python clinical_sim/main.py --drug metformin --cohort-size 50
```

Cohort mode: simulator summary only (WM is single-patient).

---

## 4. LLM context

```bash
export LLM_PUBMED_CHARS=2000
export LLM_OPENFDA_CHARS=2000
export LLM_DRUGBANK_CHARS=3000
```

---

## 5. World model — dataset, train, eval

From `clinical_sim/`:

```bash
cd clinical_sim
python3 -m world_model.generate_dataset --drugs metformin,ibuprofen,amoxicillin --runs-per-drug 30 --timesteps 40 --base-seed 42 --out ../data/processed/wm_train.csv
python3 -m world_model.generate_dataset --drugs metformin,ibuprofen,amoxicillin --runs-per-drug 10 --timesteps 40 --base-seed 999 --out ../data/processed/wm_eval.csv
python3 -m world_model.train_baseline --csv ../data/processed/wm_train.csv --out-dir world_model/artifacts
python3 -m world_model.eval_compare --csv ../data/processed/wm_eval.csv --model-dir world_model/artifacts --horizon 15
python3 -m world_model.eval_compare --ladder --csv ../data/processed/wm_eval.csv --horizon 15
```

Artifacts: `world_model/artifacts/world_model_rf.joblib`, `world_model_meta.json`.

**Primary metric:** stable normalized MAE (`metrics_exclude_targets` in meta).

---

## 6. UI

```bash
PYTHONPATH=clinical_sim streamlit run streamlit_app.py
python frontend/server.py
```

Both require trained WM artifacts and show grounded WM vs simulator.

---

## 7. One-liner smoke pipeline

```bash
cd clinical_sim
python3 -m world_model.generate_dataset --drugs metformin,ibuprofen --runs-per-drug 8 --timesteps 25 --out ../data/processed/wm_small.csv
python3 -m world_model.train_baseline --csv ../data/processed/wm_small.csv --out-dir world_model/artifacts --n-estimators 40 --max-depth 12
python3 -m world_model.eval_compare --csv ../data/processed/wm_small.csv --model-dir world_model/artifacts --horizon 10 --ladder
```
