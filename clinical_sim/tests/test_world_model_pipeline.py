"""World-model dataset generation + optional sklearn train/eval smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from world_model.generate_dataset import generate_scaled_dataset
from world_model.train_baseline import train_and_save


def test_generate_scaled_dataset_small() -> None:
    rows = generate_scaled_dataset(
        drugs=["metformin"],
        runs_per_drug=2,
        timesteps=6,
        dose=200.0,
        base_seed=99,
        use_default_rules=False,
    )
    assert len(rows) == 2 * 6
    assert rows[0]["drug_name"] == "metformin"
    assert "drug_id" in rows[0]
    assert "y_clinical_response" in rows[0]


def test_train_and_eval_smoke(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rows = generate_scaled_dataset(
        drugs=["metformin", "ibuprofen"],
        runs_per_drug=4,
        timesteps=8,
        dose=200.0,
        base_seed=1,
        use_default_rules=False,
    )
    import csv

    csv_path = tmp_path / "wm.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    model_dir = tmp_path / "model"
    meta = train_and_save(
        csv_path=csv_path,
        out_dir=model_dir,
        test_fraction=0.25,
        val_fraction=0.25,
        seed=7,
        n_estimators=20,
        max_depth=8,
    )
    assert meta["test_mae_mean"] >= 0.0
    assert "test_normalized_mae_mean" in meta
    assert (model_dir / "world_model_meta.json").is_file()

    from world_model.eval_rollout import evaluate

    ev = evaluate(csv_path=csv_path, model_dir=model_dir, max_horizon=5)
    assert "one_step_mean_mae" in ev
    assert "one_step_normalized_mae" in ev
    assert ev["n_runs_evaluated"] >= 1

    raw = json.loads((model_dir / "world_model_meta.json").read_text(encoding="utf-8"))
    assert raw["n_train"] > 0
