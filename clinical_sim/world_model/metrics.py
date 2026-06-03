"""One-step and rollout evaluation vs simulator labels in CSV."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from world_model.core import (
    build_x_row,
    delta_from_row,
    delta_to_state_features,
    load_csv,
    normalized_mae,
    normalized_mae_report,
    stable_target_mask_from_meta,
    state_from_row,
    target_scales_from_meta,
    to_matrix,
)

PredictFn = Callable[[np.ndarray], np.ndarray]


def rows_by_run(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rid in by_run:
        by_run[rid].sort(key=lambda x: int(x["timestep"]))
    return dict(by_run)


def evaluate_one_step(
    *,
    eval_rows: list[dict[str, str]],
    feature_cols: list[str],
    target_cols: list[str],
    scales: np.ndarray,
    predict: PredictFn,
    stable_mask: np.ndarray | None = None,
) -> dict[str, float]:
    if not eval_rows:
        return {
            "one_step_mean_mae": 0.0,
            "one_step_normalized_mae": 0.0,
            "one_step_normalized_mae_stable": 0.0,
        }
    x = to_matrix(eval_rows, feature_cols)
    y_delta = np.array([delta_from_row(r, target_cols) for r in eval_rows], dtype=np.float64)
    pred = predict(x)
    one_step_mean_mae = float(np.mean(np.abs(pred - y_delta)))
    one_step_normalized_mae, _ = normalized_mae(y_delta, pred, scale=scales, stable_mask=None)
    stable_norm, _ = normalized_mae(
        y_delta, pred, scale=scales, stable_mask=stable_mask
    )
    return {
        "one_step_mean_mae": one_step_mean_mae,
        "one_step_normalized_mae": one_step_normalized_mae,
        "one_step_normalized_mae_stable": stable_norm,
    }


def evaluate_rollout(
    *,
    csv_path: Path,
    feature_cols: list[str],
    target_cols: list[str],
    state_s_cols: list[str],
    scales: np.ndarray,
    predict: PredictFn,
    max_horizon: int,
    test_drugs: set[str],
    stable_mask: np.ndarray | None = None,
    n_seed_runs_per_drug: int = 10,
) -> dict[str, Any]:
    by_run = rows_by_run(csv_path)
    if test_drugs:
        by_run = {
            rid: rows
            for rid, rows in by_run.items()
            if rows and rows[0].get("drug_id", "") in test_drugs
        }
    by_drug_runs: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for traj in by_run.values():
        if traj:
            by_drug_runs[traj[0]["drug_id"]].append(traj)
    rollout_norm_errors: list[list[float]] = []
    seed_rollout_means: list[float] = []
    for _drug_id, trajectories in by_drug_runs.items():
        for traj in trajectories[:n_seed_runs_per_drug]:
            if len(traj) < 2:
                continue
            ro_e: list[float] = []
            prev_state = state_from_row(traj[0], state_s_cols)
            for k, row in enumerate(traj):
                if k >= max_horizon:
                    break
                x = np.array([build_x_row(row, prev_state, feature_cols)], dtype=np.float64)
                y_hat_delta = predict(x)[0]
                y_true_delta = delta_from_row(row, target_cols)
                per_dim_err = np.abs(y_hat_delta - y_true_delta) / scales
                if stable_mask is not None and np.any(stable_mask):
                    err_norm = float(np.mean(per_dim_err[stable_mask]))
                else:
                    err_norm = float(np.mean(per_dim_err))
                ro_e.append(err_norm)
                prev_state = delta_to_state_features(
                    prev_state, y_hat_delta, target_cols, state_s_cols
                )
            if ro_e:
                rollout_norm_errors.append(ro_e)
                seed_rollout_means.append(float(np.mean(ro_e)))
    step_means = [
        float(np.mean([r[i] for r in rollout_norm_errors if len(r) > i]))
        for i in range(max_horizon)
        if any(len(r) > i for r in rollout_norm_errors)
    ]
    rollout_mean = (
        float(np.mean([e for run in rollout_norm_errors for e in run]))
        if rollout_norm_errors
        else 0.0
    )
    return {
        "rollout_mean_normalized_mae": rollout_mean,
        "rollout_mean_normalized_mae_by_step": step_means,
        "rollout_seed_averaged_normalized_mae": (
            float(np.mean(seed_rollout_means)) if seed_rollout_means else 0.0
        ),
        "n_runs_evaluated": len(rollout_norm_errors),
        "n_seed_runs_per_drug": n_seed_runs_per_drug,
    }


def evaluate_predictor_on_csv(
    *,
    csv_path: Path,
    meta: dict[str, Any],
    predict: PredictFn,
    max_horizon: int,
    model_name: str = "model",
) -> dict[str, Any]:
    feature_cols: list[str] = meta["feature_cols"]
    target_cols: list[str] = meta["target_cols"]
    state_s_cols = [c for c in feature_cols if c.startswith("s_")]
    test_drugs = set(meta.get("test_drugs", []))
    scales = target_scales_from_meta(meta, target_cols)
    stable_mask = stable_target_mask_from_meta(meta, target_cols)
    all_rows = load_csv(csv_path)
    eval_rows = [r for r in all_rows if not test_drugs or r["drug_id"] in test_drugs]
    one_step = evaluate_one_step(
        eval_rows=eval_rows,
        feature_cols=feature_cols,
        target_cols=target_cols,
        scales=scales,
        predict=predict,
        stable_mask=stable_mask,
    )
    rollout = evaluate_rollout(
        csv_path=csv_path,
        feature_cols=feature_cols,
        target_cols=target_cols,
        state_s_cols=state_s_cols,
        scales=scales,
        predict=predict,
        max_horizon=max_horizon,
        test_drugs=test_drugs,
        stable_mask=stable_mask,
    )
    out: dict[str, Any] = {
        "model": model_name,
        **one_step,
        **rollout,
        "drug_level_held_out_evaluation": bool(test_drugs),
        "held_out_drug_ids": sorted(test_drugs),
        "metrics_exclude_targets": list(meta.get("metrics_exclude_targets") or []),
    }
    out["one_step_normalized_mae"] = one_step.get(
        "one_step_normalized_mae_stable", one_step["one_step_normalized_mae"]
    )
    if eval_rows:
        x = to_matrix(eval_rows, feature_cols)
        y_delta = np.array([delta_from_row(r, target_cols) for r in eval_rows], dtype=np.float64)
        pred = predict(x)
        report = normalized_mae_report(
            y_delta, pred, target_cols=target_cols, scale=scales, stable_mask=stable_mask
        )
        out["normalized_mae_per_target"] = report["normalized_mae_per_target"]
        out["one_step_normalized_mae_all_targets"] = report["normalized_mae_mean_all_targets"]
    return out
