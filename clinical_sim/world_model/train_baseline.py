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


def _split_by_run(
    rows: list[dict[str, str]], test_fraction: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    run_ids = sorted({r["run_id"] for r in rows})
    rng = np.random.default_rng(seed)
    rng.shuffle(run_ids)
    n_test = max(1, int(len(run_ids) * test_fraction))
    test_ids = set(run_ids[:n_test])
    train = [r for r in rows if r["run_id"] not in test_ids]
    test = [r for r in rows if r["run_id"] in test_ids]
    if not train:
        train, test = test, train
    return train, test


def _to_matrix(
    rows: list[dict[str, str]], cols: list[str]
) -> np.ndarray:
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)


def train_and_save(
    *,
    csv_path: Path,
    out_dir: Path,
    test_fraction: float,
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
    train_rows, test_rows = _split_by_run(rows, test_fraction=test_fraction, seed=seed)

    X_train = _to_matrix(train_rows, feature_cols)
    y_train = _to_matrix(train_rows, target_cols)
    X_test = _to_matrix(test_rows, feature_cols)
    y_test = _to_matrix(test_rows, target_cols)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = float(np.mean(np.abs(pred - y_test)))
    per_dim_mae = np.mean(np.abs(pred - y_test), axis=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "test_mae_mean": mae,
        "test_mae_per_target": {target_cols[i]: float(per_dim_mae[i]) for i in range(len(target_cols))},
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
    p.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of runs held out for test.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=80)
    p.add_argument("--max-depth", type=int, default=None)
    args = p.parse_args(argv)

    out_dir = args.out_dir or Path(__file__).resolve().parent / "artifacts"
    meta = train_and_save(
        csv_path=args.csv,
        out_dir=out_dir,
        test_fraction=args.test_fraction,
        seed=args.seed,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    print(json.dumps({k: v for k, v in meta.items() if k != "test_mae_per_target"}, indent=2))
    print("test_mae_mean:", meta["test_mae_mean"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
