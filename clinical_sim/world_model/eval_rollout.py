"""Evaluate one-step and multi-step rollout error for trained world model."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[misc, assignment]


def _load_model(model_dir: Path) -> Any:
    j = model_dir / "world_model_rf.joblib"
    p = model_dir / "world_model_rf.pkl"
    if j.is_file() and joblib is not None:
        return joblib.load(j)
    if p.is_file():
        import pickle

        return pickle.loads(p.read_bytes())
    raise FileNotFoundError(f"No model in {model_dir}")


def _load_meta(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "world_model_meta.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rows_by_run(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_run: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_run[r["run_id"]].append(r)
    for rid in by_run:
        by_run[rid].sort(key=lambda x: int(x["timestep"]))
    return dict(by_run)


def _target_scales(meta: dict[str, Any], target_cols: list[str]) -> np.ndarray:
    raw = meta.get("target_scale_std", {})
    vals = [float(raw.get(c, 1.0)) for c in target_cols]
    arr = np.array(vals, dtype=np.float64)
    return np.where(arr > 1e-8, arr, 1.0)


def _state_from_row(row: dict[str, str], state_feature_cols: list[str]) -> dict[str, float]:
    return {c: float(row[c]) for c in state_feature_cols}


def _delta_truth_from_row(row: dict[str, str], target_cols: list[str]) -> np.ndarray:
    vals = []
    for y_col in target_cols:
        suffix = y_col[2:]
        vals.append(float(row[y_col]) - float(row[f"s_{suffix}"]))
    return np.array(vals, dtype=np.float64)


def _delta_to_state_features(
    prev_state: dict[str, float],
    delta_vec: np.ndarray,
    target_cols: list[str],
    state_feature_cols: list[str],
) -> dict[str, float]:
    suffix_map = {c[2:]: float(v) for c, v in zip(target_cols, delta_vec)}
    out: dict[str, float] = {}
    for c in state_feature_cols:
        suffix = c[2:]
        out[c] = prev_state.get(c, 0.0) + suffix_map.get(suffix, 0.0)
    return out


def _build_X_row(
    row: dict[str, str],
    state_override: dict[str, float] | None,
    feature_cols: list[str],
) -> list[float]:
    vals: list[float] = []
    for c in feature_cols:
        if state_override is not None and c.startswith("s_") and c in state_override:
            vals.append(float(state_override[c]))
        else:
            vals.append(float(row[c]))
    return vals


def evaluate(
    *,
    csv_path: Path,
    model_dir: Path,
    max_horizon: int,
) -> dict[str, Any]:
    model = _load_model(model_dir)
    meta = _load_meta(model_dir)
    feature_cols: list[str] = meta["feature_cols"]
    target_cols: list[str] = meta["target_cols"]
    state_s_cols = [c for c in feature_cols if c.startswith("s_")]
    test_drugs = set(meta.get("test_drugs", []))
    scales = _target_scales(meta, target_cols)

    with csv_path.open(encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    eval_rows = [r for r in all_rows if not test_drugs or r["drug_id"] in test_drugs]
    if eval_rows:
        X_all = np.array(
            [[float(r[c]) for c in feature_cols] for r in eval_rows], dtype=np.float64
        )
        y_delta_all = np.array(
            [_delta_truth_from_row(r, target_cols) for r in eval_rows], dtype=np.float64
        )
        pred_all = model.predict(X_all)
        one_step_mean_mae = float(np.mean(np.abs(pred_all - y_delta_all)))
        one_step_normalized_mae = float(np.mean(np.abs(pred_all - y_delta_all) / scales))
    else:
        one_step_mean_mae = 0.0
        one_step_normalized_mae = 0.0

    by_run = _rows_by_run(csv_path)
    if test_drugs:
        by_run = {
            rid: rows for rid, rows in by_run.items() if rows and rows[0].get("drug_id", "") in test_drugs
        }

    by_drug_runs: dict[str, list[list[dict[str, str]]]] = defaultdict(list)
    for traj in by_run.values():
        if traj:
            by_drug_runs[traj[0]["drug_id"]].append(traj)

    rollout_norm_errors: list[list[float]] = []
    seed_rollout_means: list[float] = []
    n_seed_runs = 10

    for _drug_id, trajectories in by_drug_runs.items():
        selected = trajectories[:n_seed_runs]
        for traj in selected:
            if len(traj) < 2:
                continue
            ro_e: list[float] = []
            prev_state = _state_from_row(traj[0], state_s_cols)
            for k, row in enumerate(traj):
                if k >= max_horizon:
                    break
                x = np.array([_build_X_row(row, prev_state, feature_cols)], dtype=np.float64)
                y_hat_delta = model.predict(x)[0]
                y_true_delta = _delta_truth_from_row(row, target_cols)
                err_norm = float(np.mean(np.abs(y_hat_delta - y_true_delta) / scales))
                ro_e.append(err_norm)
                prev_state = _delta_to_state_features(prev_state, y_hat_delta, target_cols, state_s_cols)
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
    rollout_seed_avg = float(np.mean(seed_rollout_means)) if seed_rollout_means else 0.0

    return {
        "one_step_mean_mae": one_step_mean_mae,
        "one_step_normalized_mae": one_step_normalized_mae,
        "rollout_mean_normalized_mae": rollout_mean,
        "rollout_mean_normalized_mae_by_step": step_means,
        "rollout_seed_averaged_normalized_mae": rollout_seed_avg,
        "n_runs_evaluated": len(rollout_norm_errors),
        "n_seed_runs_per_drug": n_seed_runs,
        "drug_level_held_out_evaluation": bool(test_drugs),
        "held_out_drug_ids": sorted(test_drugs),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate world model rollout vs simulator labels in CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Same transition CSV used for training.")
    p.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory with world_model_rf.* and world_model_meta.json",
    )
    p.add_argument("--horizon", type=int, default=15, help="Max timesteps per run for rollout.")
    args = p.parse_args(argv)

    model_dir = args.model_dir or Path(__file__).resolve().parent / "artifacts"
    out = evaluate(csv_path=args.csv, model_dir=model_dir, max_horizon=args.horizon)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
