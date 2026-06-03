"""Baseline B0–B2 vs B3 (RF) on toy transition data."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from world_model.core import infer_columns, load_csv, rows_for_drugs, split_drugs, target_scales_from_meta
from world_model.generate_dataset import generate_scaled_dataset
from world_model.metrics import evaluate_one_step
from world_model.predict import fit_baselines, load_world_model, make_predict_fn
from world_model.train_baseline import train_and_save


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_b3_beats_b0_on_toy_data(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rows = generate_scaled_dataset(
        drugs=["metformin", "ibuprofen", "amoxicillin"],
        runs_per_drug=6,
        timesteps=10,
        dose=200.0,
        base_seed=3,
        use_default_rules=False,
    )
    csv_path = tmp_path / "wm.csv"
    _write_csv(csv_path, rows)

    model_dir = tmp_path / "model"
    meta = train_and_save(
        csv_path=csv_path,
        out_dir=model_dir,
        test_fraction=0.34,
        val_fraction=0.0,
        seed=11,
        n_estimators=30,
        max_depth=10,
    )

    all_rows = load_csv(csv_path)
    feature_cols, target_cols = infer_columns(all_rows)
    train_drugs, _, test_drugs = split_drugs(all_rows, test_fraction=0.34, val_fraction=0.0, seed=11)
    train_rows = rows_for_drugs(all_rows, train_drugs)
    test_rows = rows_for_drugs(all_rows, test_drugs)
    scales = target_scales_from_meta(meta, target_cols)

    bundle = fit_baselines(
        train_rows=train_rows,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seed=11,
    )
    b0 = evaluate_one_step(
        eval_rows=test_rows,
        feature_cols=feature_cols,
        target_cols=target_cols,
        scales=scales,
        predict=make_predict_fn(bundle, "B0_persistence"),
    )

    loaded = load_world_model(model_dir)

    def rf_predict(x: np.ndarray) -> np.ndarray:
        return loaded.predict_delta_array(x)

    b3 = evaluate_one_step(
        eval_rows=test_rows,
        feature_cols=feature_cols,
        target_cols=target_cols,
        scales=scales,
        predict=rf_predict,
    )

    assert b3["one_step_normalized_mae"] <= b0["one_step_normalized_mae"]


def test_runtime_grounded_rollout(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    rows = generate_scaled_dataset(
        drugs=["metformin"],
        runs_per_drug=3,
        timesteps=8,
        dose=200.0,
        base_seed=5,
        use_default_rules=False,
    )
    csv_path = tmp_path / "wm.csv"
    _write_csv(csv_path, rows)
    model_dir = tmp_path / "model"
    train_and_save(
        csv_path=csv_path,
        out_dir=model_dir,
        test_fraction=0.0,
        val_fraction=0.0,
        seed=1,
        n_estimators=15,
        max_depth=6,
    )

    from loop import run_simulation
    from state import Treatment, WorldState
    from world_model.data import drug_rule_table
    from world_model.predict import (
        grounded_rollout_from_history,
        history_to_transition_rows,
        load_world_model,
        rollout_predict,
    )

    st = WorldState().copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=200.0,
            schedule=list(range(0, 8, 7)),
            arm_assignment="treatment",
        )
    )
    history = run_simulation(
        initial_state=st,
        rule_tables=drug_rule_table("metformin"),
        n_timesteps=8,
        verbose=False,
    )
    loaded = load_world_model(model_dir)
    replay = history_to_transition_rows(history, drug_name="metformin")
    traj = rollout_predict(replay, loaded=loaded, horizon=5)
    assert len(traj) >= 2

    block = grounded_rollout_from_history(
        history, drug_name="metformin", artifacts_dir=model_dir, horizon=5
    )
    assert "clinical_compare" in block
    assert block["wm_trajectory_steps"] >= 2
