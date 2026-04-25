"""Train a baseline multi-output regressor on transition CSV (world model v0)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError as e:  # pragma: no cover
    RandomForestRegressor = None  # type: ignore[misc, assignment]
    _sklearn_import_error = e
else:
    _sklearn_import_error = None


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _infer_columns(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    if not rows:
        raise ValueError("empty dataset")
    keys = set(rows[0].keys())
    feature_cols = sorted(
        c
        for c in keys
        if c.startswith("s_")
        or c.startswith("a_")
        or c
        in (
            "age",
            "renal_function",
            "hepatic_function",
            "cyp450_metaboliser_id",
            "drug_id",
        )
    )
    target_cols = sorted(c for c in keys if c.startswith("y_"))
    if not feature_cols or not target_cols:
        raise ValueError("could not infer feature or target columns")
    return feature_cols, target_cols


def _split_drugs(
    rows: list[dict[str, str]],
    *,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    drugs = sorted({r["drug_id"] for r in rows})
    if not drugs:
        raise ValueError("no drug_id values found in dataset")
    rng = np.random.default_rng(seed)
    rng.shuffle(drugs)

    n = len(drugs)
    # Drug-level splits need disjoint drug_ids. With very few drugs, a strict
    # train/val/test partition is impossible; degrade gracefully.
    if n == 1:
        return {drugs[0]}, set(), set()
    if n == 2:
        # Only one holdout split is possible without emptying train.
        n_test = 1 if test_fraction > 0.0 else 0
        n_val = 0
        if n_test == 0:
            # Caller asked for no test holdout; keep both drugs in train.
            return set(drugs), set(), set()
        test_drugs = set(drugs[:n_test])
        train_drugs = set(drugs[n_test:])
        return train_drugs, set(), test_drugs

    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0.0 else 0
    n_train = n - n_test - n_val

    if n_train <= 0:
        # Shrink holdouts until at least one train drug remains.
        n_train = 1
        remaining = n - n_train
        if val_fraction > 0.0:
            n_val = max(1, min(n_val, remaining - 1)) if remaining > 1 else 0
            n_test = max(1, remaining - n_val) if remaining - n_val > 0 else remaining
        else:
            n_val = 0
            n_test = max(1, min(n_test, remaining))

        if n_test + n_val >= n:
            n_val = max(0, min(n_val, n - 2))
            n_test = n - 1 - n_val
        if n_test <= 0 or n_test + n_val >= n:
            # Last resort: single-drug test, rest train (val may be empty).
            n_test = 1
            n_val = max(0, min(n_val, n - n_test - 1))

    test_drugs = set(drugs[:n_test])
    val_drugs = set(drugs[n_test : n_test + n_val])
    train_drugs = set(drugs[n_test + n_val :])
    if not train_drugs:
        train_drugs = set(drugs) - test_drugs - val_drugs
    if not train_drugs:
        raise ValueError("unable to create non-empty train split")
    return train_drugs, val_drugs, test_drugs


def _rows_for_drugs(rows: list[dict[str, str]], drug_ids: set[str]) -> list[dict[str, str]]:
    return [r for r in rows if r["drug_id"] in drug_ids]


def _to_matrix(
    rows: list[dict[str, str]], cols: list[str]
) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)


def _to_delta_matrix(rows: list[dict[str, str]], target_cols: list[str]) -> np.ndarray:
    out: list[list[float]] = []
    for r in rows:
        row_vals: list[float] = []
        for y_col in target_cols:
            suffix = y_col[2:]
            s_col = f"s_{suffix}"
            row_vals.append(float(r[y_col]) - float(r[s_col]))
        out.append(row_vals)
    return np.array(out, dtype=np.float64)


def _normalized_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    scale: np.ndarray,
) -> tuple[float, np.ndarray]:
    abs_err = np.abs(y_pred - y_true)
    norm_err = abs_err / scale
    per_dim = np.mean(norm_err, axis=0)
    return float(np.mean(per_dim)), per_dim


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

    rows = _load_csv(csv_path)
    feature_cols, target_cols = _infer_columns(rows)
    train_drugs, val_drugs, test_drugs = _split_drugs(
        rows, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
    )
    train_rows = _rows_for_drugs(rows, train_drugs)
    val_rows = _rows_for_drugs(rows, val_drugs)
    test_rows = _rows_for_drugs(rows, test_drugs)

    X_train = _to_matrix(train_rows, feature_cols)
    y_train_delta = _to_delta_matrix(train_rows, target_cols)
    X_test = _to_matrix(test_rows, feature_cols)
    y_test_delta = _to_delta_matrix(test_rows, target_cols)
    X_val = _to_matrix(val_rows, feature_cols) if val_rows else np.empty((0, len(feature_cols)))
    y_val_delta = (
        _to_delta_matrix(val_rows, target_cols) if val_rows else np.empty((0, len(target_cols)))
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_delta)

    scale = np.std(y_train_delta, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)

    pred_test = model.predict(X_test)
    test_mae = float(np.mean(np.abs(pred_test - y_test_delta)))
    test_norm_mae, test_norm_per_dim = _normalized_mae(y_test_delta, pred_test, scale=scale)
    per_dim_mae = np.mean(np.abs(pred_test - y_test_delta), axis=0)

    if len(X_val):
        pred_val = model.predict(X_val)
        val_mae = float(np.mean(np.abs(pred_val - y_val_delta)))
        val_norm_mae, _ = _normalized_mae(y_val_delta, pred_val, scale=scale)
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
        "val_normalized_mae_mean": val_norm_mae,
        "test_mae_per_target": {
            target_cols[i]: float(per_dim_mae[i]) for i in range(len(target_cols))
        },
        "test_normalized_mae_per_target": {
            target_cols[i]: float(test_norm_per_dim[i]) for i in range(len(target_cols))
        },
        "target_scale_std": {target_cols[i]: float(scale[i]) for i in range(len(target_cols))},
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
