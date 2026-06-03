# World model

Simulator = **ground truth**. World model learns next-day state from simulator CSVs; runtime always compares WM to simulator.

## Files (7 total)

| File | Purpose |
|------|---------|
| `core.py` | Schemas, adapters, CSV/matrix helpers, bounds |
| `data.py` | Drug rules + transition dataset from simulator |
| `predict.py` | Load RF, baselines B0–B2, grounded rollout |
| `metrics.py` | One-step and rollout eval vs labels |
| `generate_dataset.py` | CLI: build training CSV |
| `train_baseline.py` | CLI: train Random Forest |
| `eval_compare.py` | CLI: compare B0–B3; `--ladder` report |

## Workflow

```bash
cd clinical_sim
python3 -m world_model.generate_dataset --drugs metformin,ibuprofen,amoxicillin --runs-per-drug 30 --timesteps 40 --out ../data/processed/wm_train.csv
python3 -m world_model.train_baseline --csv ../data/processed/wm_train.csv --out-dir world_model/artifacts
python3 -m world_model.eval_compare --ladder --csv ../data/processed/wm_eval.csv
```

`main.py` requires `world_model/artifacts/` and prints grounded WM vs simulator after each single-patient run.
