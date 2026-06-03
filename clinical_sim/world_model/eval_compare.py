"""Compare RF (B3) and baselines B0–B2 on the same held-out drug metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[misc, assignment]

from world_model.core import infer_columns, load_csv, rows_for_drugs, split_drugs
from world_model.metrics import evaluate_predictor_on_csv
from world_model.predict import fit_baselines, load_world_model, make_predict_fn

LADDER = [
    {
        "tier": "baselines",
        "id": "B0_persistence",
        "name": "Persistence",
        "predicts": "Δstate = 0 (next state = current)",
    },
    {
        "tier": "baselines",
        "id": "B1_mean_delta",
        "name": "Mean train delta",
        "predicts": "Δstate = average Δ from training drugs",
    },
    {
        "tier": "baselines",
        "id": "B2_ridge",
        "name": "Ridge regression",
        "predicts": "Linear map from (state, action, context) → Δstate",
    },
    {
        "tier": "world_model_v0",
        "id": "B3_random_forest",
        "name": "Random Forest (world model)",
        "predicts": "Nonlinear ensemble; artifact: world_model_rf.joblib",
    },
    {
        "tier": "world_model_v1_planned",
        "id": "B4_lstm",
        "name": "LSTM / neural (not implemented)",
        "predicts": "Sequence model — future work",
    },
]


def _load_rf(model_dir: Path) -> Any:
    j = model_dir / "world_model_rf.joblib"
    p = model_dir / "world_model_rf.pkl"
    if j.is_file() and joblib is not None:
        return joblib.load(j)
    if p.is_file():
        import pickle

        return pickle.loads(p.read_bytes())
    raise FileNotFoundError(f"No RF model in {model_dir}")


def _load_meta(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "world_model_meta.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_all(
    *,
    csv_path: Path,
    model_dir: Path,
    max_horizon: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Score B0–B3 on eval CSV using drug splits from RF meta (or refit splits if missing)."""
    meta = _load_meta(model_dir)
    rf = _load_rf(model_dir)

    rows = load_csv(csv_path)
    train_drugs = set(meta.get("train_drugs", [])) | set(meta.get("val_drugs", []))
    if not train_drugs:
        train_drugs, _, _ = split_drugs(rows, test_fraction=0.6, val_fraction=0.2, seed=seed)
    feature_cols, target_cols = infer_columns(rows)
    train_rows = rows_for_drugs(rows, train_drugs)
    bundle = fit_baselines(
        train_rows=train_rows,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seed=seed,
    )

    eval_meta = dict(meta)
    results: dict[str, Any] = {
        "csv_path": str(csv_path),
        "model_dir": str(model_dir),
        "train_drugs": sorted(train_drugs),
        "test_drugs": meta.get("test_drugs", []),
        "models": {},
    }

    for bid in ("B0_persistence", "B1_mean_delta", "B2_ridge"):
        try:
            predict = make_predict_fn(bundle, bid)
            results["models"][bid] = evaluate_predictor_on_csv(
                csv_path=csv_path,
                meta=eval_meta,
                predict=predict,
                max_horizon=max_horizon,
                model_name=bid,
            )
        except RuntimeError as e:
            results["models"][bid] = {"model": bid, "error": str(e)}

    def rf_predict(x):
        return rf.predict(x)

    results["models"]["B3_random_forest"] = evaluate_predictor_on_csv(
        csv_path=csv_path,
        meta=eval_meta,
        predict=rf_predict,
        max_horizon=max_horizon,
        model_name="B3_random_forest",
    )

    return results


def evaluate_rf(
    *,
    csv_path: Path,
    model_dir: Path,
    max_horizon: int,
) -> dict[str, Any]:
    """Evaluate trained RF (B3) on a CSV of simulator transitions."""
    loaded = load_world_model(model_dir)

    def predict(x):
        return loaded.predict_delta_array(x)

    return evaluate_predictor_on_csv(
        csv_path=csv_path,
        meta=loaded.meta,
        predict=predict,
        max_horizon=max_horizon,
        model_name="B3_random_forest",
    )


def _fmt_metric(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.4f}"


def format_ladder_report(
    *,
    train_meta: dict[str, Any],
    compare_report: dict[str, Any] | None,
) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("WORLD MODEL LADDER (simulator labels = ground truth)")
    lines.append("=" * 72)
    excluded = train_meta.get("metrics_exclude_targets") or []
    if excluded:
        lines.append(f"Stable metric excludes: {', '.join(excluded)}")
    lines.append("")
    lines.append("--- Training (world_model_meta.json) ---")
    lines.append(f"  Train rows: {train_meta.get('n_train')}  Test rows: {train_meta.get('n_test')}")
    lines.append(f"  Test norm MAE (stable): {_fmt_metric(train_meta.get('test_normalized_mae_mean'))}")
    lines.append("")

    if compare_report:
        lines.append("--- Held-out eval (eval_compare) ---")
        header = f"{'ID':<22} {'1-step stable':>14} {'Rollout stable':>14}"
        lines.append(header)
        lines.append("-" * len(header))
        models = compare_report.get("models") or {}
        for entry in LADDER:
            mid = entry["id"]
            if mid not in models and entry["tier"] == "world_model_v1_planned":
                lines.append(f"{mid:<22} {'—':>14} {'—':>14}")
                continue
            m = models.get(mid, {})
            if "error" in m:
                lines.append(f"{mid:<22} {str(m['error'])[:30]}")
                continue
            one = m.get("one_step_normalized_mae_stable") or m.get(
                "one_step_normalized_mae"
            )
            roll = m.get("rollout_mean_normalized_mae")
            lines.append(f"{mid:<22} {_fmt_metric(one):>14} {_fmt_metric(roll):>14}")
    else:
        lines.append("(Pass --csv to print B0–B3 comparison.)")

    lines.append("")
    lines.append("=" * 72)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate and compare world-model baselines B0–B3.")
    p.add_argument("--csv", type=Path, default=None, help="Eval CSV (simulator transitions).")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--out", type=Path, default=None, help="Write JSON report to this path.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ladder",
        action="store_true",
        help="Print human-readable baseline → RF ladder (needs --csv for B0–B3 table).",
    )
    p.add_argument("--json-out", type=Path, default=None, help="With --ladder, also write JSON.")
    args = p.parse_args(argv)

    model_dir = args.model_dir or Path(__file__).resolve().parent / "artifacts"

    if args.ladder:
        meta_path = model_dir / "world_model_meta.json"
        if not meta_path.is_file():
            print(f"Missing {meta_path}. Run train_baseline first.")
            return 1
        train_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        compare_report = None
        if args.csv and args.csv.is_file():
            compare_report = compare_all(
                csv_path=args.csv,
                model_dir=model_dir,
                max_horizon=args.horizon,
                seed=args.seed,
            )
        print(format_ladder_report(train_meta=train_meta, compare_report=compare_report))
        if args.json_out:
            payload = {
                "ladder": LADDER,
                "train_meta": train_meta,
                "eval_compare": compare_report,
            }
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return 0

    if not args.csv:
        print("Provide --csv for comparison, or use --ladder for training summary.")
        return 2

    report = compare_all(
        csv_path=args.csv,
        model_dir=model_dir,
        max_horizon=args.horizon,
        seed=args.seed,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
