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


def _y_to_state_features(
    y_vec: np.ndarray,
    target_cols: list[str],
    state_feature_cols: list[str],
) -> dict[str, float]:
    """Map predicted y_* vector back to s_* keys (same suffix after prefix)."""
    suffix_map = {c[2:]: float(v) for c, v in zip(target_cols, y_vec)}
    out: dict[str, float] = {}
    for c in state_feature_cols:
        suffix = c[2:]  # strip s_
        out[c] = suffix_map.get(suffix, 0.0)
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

    with csv_path.open(encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    if all_rows:
        X_all = np.array(
            [[float(r[c]) for c in feature_cols] for r in all_rows], dtype=np.float64
        )
        y_all = np.array(
            [[float(r[c]) for c in target_cols] for r in all_rows], dtype=np.float64
        )
        pred_all = model.predict(X_all)
        one_step_mean_mae = float(np.mean(np.abs(pred_all - y_all)))
    else:
        one_step_mean_mae = 0.0

    by_run = _rows_by_run(csv_path)
    rollout_errors: list[list[float]] = []

    for _run_id, traj in by_run.items():
        if len(traj) < 2:
            continue
        ro_e: list[float] = []
        y_prev_vec: np.ndarray | None = None
        for k, row in enumerate(traj):
            if k >= max_horizon:
                break
            if k == 0:
                state_override = None
            else:
                assert y_prev_vec is not None
                st = _y_to_state_features(y_prev_vec, target_cols, state_s_cols)
                state_override = st
            x = np.array([_build_X_row(row, state_override, feature_cols)], dtype=np.float64)
            y_hat = model.predict(x)[0]
            y_true = np.array([float(row[c]) for c in target_cols], dtype=np.float64)
            err = float(np.mean(np.abs(y_hat - y_true)))
            ro_e.append(err)
            y_prev_vec = y_hat
        if ro_e:
            rollout_errors.append(ro_e)

    step_means = [
        float(np.mean([r[i] for r in rollout_errors if len(r) > i]))
        for i in range(max_horizon)
        if any(len(r) > i for r in rollout_errors)
    ]
    rollout_mean = float(np.mean([e for run in rollout_errors for e in run])) if rollout_errors else 0.0

    return {
        "one_step_mean_mae": one_step_mean_mae,
        "rollout_mean_mae": rollout_mean,
        "rollout_mean_mae_by_step": step_means,
        "n_runs_evaluated": len(rollout_errors),
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
