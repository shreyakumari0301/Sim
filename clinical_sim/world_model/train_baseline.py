"""Train a baseline multi-output regressor on transition CSV (world model v0)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from world_model.core import (
    compute_target_scales,
    delta_matrix,
    infer_columns,
    load_csv,
    normalized_mae_report,
    rows_for_drugs,
    split_drugs,
    to_matrix,
)

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore[misc, assignment]
    _sklearn_import_error = e
else:
    _sklearn_import_error = None


def train_and_save(
    *,
    csv_path: Path,
    out_dir: Path,
    test_fraction: float,
    val_fraction: float,
    seed: int,
    n_estimators: int,
    max_depth: int | None,
) -> dict[str, Any]:
    if RandomForestRegressor is None:
        raise RuntimeError(
            "scikit-learn is required. Install with: pip install scikit-learn"
        ) from _sklearn_import_error

    rows = load_csv(csv_path)
    feature_cols, target_cols = infer_columns(rows)
    train_drugs, val_drugs, test_drugs = split_drugs(
        rows, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
    )
    train_rows = rows_for_drugs(rows, train_drugs)
    val_rows = rows_for_drugs(rows, val_drugs)
    test_rows = rows_for_drugs(rows, test_drugs)

    if not train_rows:
        raise ValueError("train split is empty; need at least one drug in train_drugs")

    X_train = to_matrix(train_rows, feature_cols)
    y_train_delta = delta_matrix(train_rows, target_cols)
    n_feat = len(feature_cols)
    n_tgt = len(target_cols)
    X_test = to_matrix(test_rows, feature_cols) if test_rows else np.empty((0, n_feat))
    y_test_delta = (
        delta_matrix(test_rows, target_cols) if test_rows else np.empty((0, n_tgt))
    )
    X_val = to_matrix(val_rows, feature_cols) if val_rows else np.empty((0, n_feat))
    y_val_delta = (
        delta_matrix(val_rows, target_cols) if val_rows else np.empty((0, n_tgt))
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_delta)

    scale, stable_mask, excluded_targets = compute_target_scales(
        y_train_delta, target_cols
    )

    if len(test_rows):
        pred_test = model.predict(X_test)
        test_mae = float(np.mean(np.abs(pred_test - y_test_delta)))
        test_report = normalized_mae_report(
            y_test_delta,
            pred_test,
            target_cols=target_cols,
            scale=scale,
            stable_mask=stable_mask,
        )
        test_norm_mae = float(test_report["normalized_mae_mean_stable_targets"])
        test_norm_mae_all = float(test_report["normalized_mae_mean_all_targets"])
        test_norm_per_dim = {
            target_cols[i]: test_report["normalized_mae_per_target"][target_cols[i]]
            for i in range(len(target_cols))
        }
        test_norm_per_dim_arr = np.array(
            [test_report["normalized_mae_per_target"][c] for c in target_cols]
        )
        per_dim_mae = np.mean(np.abs(pred_test - y_test_delta), axis=0)
    else:
        test_mae = 0.0
        test_norm_mae = 0.0
        test_norm_mae_all = 0.0
        test_norm_per_dim_arr = np.zeros(n_tgt)
        per_dim_mae = np.zeros(n_tgt)

    if len(val_rows):
        pred_val = model.predict(X_val)
        val_mae = float(np.mean(np.abs(pred_val - y_val_delta)))
        val_report = normalized_mae_report(
            y_val_delta,
            pred_val,
            target_cols=target_cols,
            scale=scale,
            stable_mask=stable_mask,
        )
        val_norm_mae = float(val_report["normalized_mae_mean_stable_targets"])
    else:
        val_mae = 0.0
        val_norm_mae = 0.0

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "target_mode": "delta",
        "state_feature_cols": [c for c in feature_cols if c.startswith("s_")],
        "train_drugs": sorted(train_drugs),
        "val_drugs": sorted(val_drugs),
        "test_drugs": sorted(test_drugs),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "test_mae_mean": test_mae,
        "val_mae_mean": val_mae,
        "test_normalized_mae_mean": test_norm_mae,
        "test_normalized_mae_mean_all_targets": test_norm_mae_all if len(test_rows) else 0.0,
        "val_normalized_mae_mean": val_norm_mae,
        "test_mae_per_target": {
            target_cols[i]: float(per_dim_mae[i]) for i in range(len(target_cols))
        },
        "test_normalized_mae_per_target": {
            target_cols[i]: float(test_norm_per_dim_arr[i]) for i in range(len(target_cols))
        },
        "target_scale_std": {
            target_cols[i]: float(np.std(y_train_delta[:, i])) for i in range(len(target_cols))
        },
        "target_scale_effective": {target_cols[i]: float(scale[i]) for i in range(len(target_cols))},
        "metrics_exclude_targets": excluded_targets,
        "csv_path": str(csv_path),
    }
    try:
        import joblib

        joblib.dump(model, out_dir / "world_model_rf.joblib")
    except ImportError:
        import pickle

        (out_dir / "world_model_rf.pkl").write_bytes(pickle.dumps(model))
    (out_dir / "world_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train baseline RF world model on transition CSV.")
    p.add_argument("--csv", type=Path, required=True, help="Transition CSV from generate_dataset.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for model + meta (default: clinical_sim/world_model/artifacts)",
    )
    p.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of drugs held out for test.",
    )
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of drugs held out for validation.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    out_dir = args.out_dir or Path(__file__).resolve().parent / "artifacts"
    meta = train_and_save(
        csv_path=args.csv,
        out_dir=out_dir,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "test_mae_per_target"}, indent=2))
    print("test_mae_mean:", meta["test_mae_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
