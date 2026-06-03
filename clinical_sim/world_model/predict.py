"""Load trained model, baselines B0–B2, and grounded rollout vs simulator."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from world_model.core import (
    ActionVector,
    StateVector,
    WorldModel,
    WorldModelPrediction,
    build_x_row,
    clip_delta_vector,
    clip_state_features,
    delta_from_row,
    delta_matrix,
    delta_to_state_features,
    infer_action,
    infer_columns,
    load_csv,
    patient_context,
    rows_for_drugs,
    split_drugs,
    state_from_row,
    state_to_vector,
    to_matrix,
)
from world_model.data import drug_id_for_name

try:
    import joblib
except ImportError:
    joblib = None  # type: ignore[misc, assignment]

try:
    from sklearn.linear_model import Ridge
except ImportError:
    Ridge = None  # type: ignore[misc, assignment]

BASELINE_IDS = ("B0_persistence", "B1_mean_delta", "B2_ridge", "B3_random_forest")


# --- Baselines ---


@dataclass
class BaselineBundle:
    feature_cols: list[str]
    target_cols: list[str]
    global_mean_delta: np.ndarray
    per_drug_mean_delta: dict[str, np.ndarray]
    ridge: Any | None = None


def fit_baselines(
    *,
    train_rows: list[dict[str, str]],
    feature_cols: list[str],
    target_cols: list[str],
    ridge_alpha: float = 1.0,
    seed: int = 42,
) -> BaselineBundle:
    y_train = delta_matrix(train_rows, target_cols)
    global_mean = np.mean(y_train, axis=0) if len(y_train) else np.zeros(len(target_cols))
    per_drug: dict[str, np.ndarray] = {}
    by_drug: dict[str, list[dict[str, str]]] = {}
    for r in train_rows:
        by_drug.setdefault(r["drug_id"], []).append(r)
    for did, rows in by_drug.items():
        per_drug[did] = np.mean(delta_matrix(rows, target_cols), axis=0)
    ridge_model = None
    if Ridge is not None and train_rows:
        x_train = to_matrix(train_rows, feature_cols)
        ridge_model = Ridge(alpha=ridge_alpha, random_state=seed)
        ridge_model.fit(x_train, y_train)
    return BaselineBundle(
        feature_cols=feature_cols,
        target_cols=target_cols,
        global_mean_delta=global_mean,
        per_drug_mean_delta=per_drug,
        ridge=ridge_model,
    )


def make_predict_fn(bundle: BaselineBundle, baseline_id: str):
    if baseline_id == "B0_persistence":

        def predict(x: np.ndarray) -> np.ndarray:
            return np.zeros((x.shape[0], len(bundle.target_cols)))

        return predict
    if baseline_id == "B1_mean_delta":

        def predict(x: np.ndarray) -> np.ndarray:
            return np.tile(bundle.global_mean_delta, (x.shape[0], 1))

        return predict
    if baseline_id == "B2_ridge":
        if bundle.ridge is None:
            raise RuntimeError("Ridge baseline unavailable (install scikit-learn)")

        def predict(x: np.ndarray) -> np.ndarray:
            return bundle.ridge.predict(x)

        return predict
    raise ValueError(f"unknown baseline_id: {baseline_id}")


def fit_baselines_from_csv(
    csv_path,
    *,
    test_fraction: float = 0.2,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[BaselineBundle, dict[str, Any]]:
    path = Path(csv_path)
    rows = load_csv(path)
    feature_cols, target_cols = infer_columns(rows)
    train_drugs, val_drugs, test_drugs = split_drugs(
        rows, test_fraction=test_fraction, val_fraction=val_fraction, seed=seed
    )
    train_rows = rows_for_drugs(rows, train_drugs)
    bundle = fit_baselines(
        train_rows=train_rows,
        feature_cols=feature_cols,
        target_cols=target_cols,
        seed=seed,
    )
    split_info = {
        "feature_cols": feature_cols,
        "target_cols": target_cols,
        "train_drugs": sorted(train_drugs),
        "val_drugs": sorted(val_drugs),
        "test_drugs": sorted(test_drugs),
    }
    return bundle, split_info


# --- Trained model + grounded rollout ---


def _load_sklearn_model(model_dir: Path) -> Any:
    j = model_dir / "world_model_rf.joblib"
    p = model_dir / "world_model_rf.pkl"
    if j.is_file() and joblib is not None:
        return joblib.load(j)
    if p.is_file():
        import pickle

        return pickle.loads(p.read_bytes())
    raise FileNotFoundError(f"No model in {model_dir}")


@dataclass
class LoadedWorldModel:
    model: Any
    meta: dict[str, Any]
    artifacts_dir: Path

    @property
    def feature_cols(self) -> list[str]:
        return list(self.meta["feature_cols"])

    @property
    def target_cols(self) -> list[str]:
        return list(self.meta["target_cols"])

    @property
    def state_feature_cols(self) -> list[str]:
        return [c for c in self.feature_cols if c.startswith("s_")]

    def predict_delta_array(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x)

    def predict_delta_row(
        self,
        row: dict[str, float | int | str],
        state_override: dict[str, float] | None = None,
    ) -> np.ndarray:
        x = np.array(
            [build_x_row({k: str(v) for k, v in row.items()}, state_override, self.feature_cols)],
            dtype=np.float64,
        )
        return self.predict_delta_array(x)[0]


def default_artifacts_dir() -> Path:
    return Path(__file__).resolve().parent / "artifacts"


def load_world_model(artifacts_dir: Path | None = None) -> LoadedWorldModel:
    root = artifacts_dir or default_artifacts_dir()
    meta_path = root / "world_model_meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model = _load_sklearn_model(root)
    return LoadedWorldModel(model=model, meta=meta, artifacts_dir=root)


class SklearnWorldModel:
    def __init__(self, loaded: LoadedWorldModel) -> None:
        self._loaded = loaded

    def predict_next(
        self,
        state_t: StateVector,
        action_t: ActionVector,
        context: dict[str, float],
    ) -> WorldModelPrediction:
        row: dict[str, float | int | str] = {}
        for k, v in state_t.__dict__.items():
            row[f"s_{k}"] = float(v)
        for k, v in action_t.__dict__.items():
            row[f"a_{k}"] = float(v)
        row["age"] = context.get("age", 0.0)
        row["renal_function"] = context.get("renal_function", 0.0)
        row["hepatic_function"] = context.get("hepatic_function", 0.0)
        row["cyp450_metaboliser_id"] = context.get("cyp450_metaboliser_id", 1.0)
        row["drug_id"] = context.get("drug_id", 0.0)
        delta = self._loaded.predict_delta_row(row)
        target_cols = self._loaded.target_cols
        state_cols = self._loaded.state_feature_cols
        prev = {c: float(row[c]) for c in state_cols if c in row}
        updated = delta_to_state_features(prev, delta, target_cols, state_cols)
        next_d = {c[2:]: updated[c] for c in state_cols}
        return WorldModelPrediction(next_state_mean=StateVector(**next_d))


def history_to_transition_rows(
    history: list,
    *,
    drug_name: str,
    run_id: str = "main",
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for t in range(len(history) - 1):
        s0, s1 = history[t], history[t + 1]
        st = state_to_vector(s0)
        st1 = state_to_vector(s1)
        act = infer_action(s0, s1)
        ctx = patient_context(s0)
        row: dict[str, float | int | str] = {
            "run_id": run_id,
            "drug_name": drug_name,
            "drug_id": drug_id_for_name(drug_name),
            "timestep": t,
            "age": ctx["age"],
            "renal_function": ctx["renal_function"],
            "hepatic_function": ctx["hepatic_function"],
            "cyp450_metaboliser_id": ctx["cyp450_metaboliser_id"],
        }
        for k, v in st.__dict__.items():
            row[f"s_{k}"] = float(v)
        for k, v in act.__dict__.items():
            row[f"a_{k}"] = float(v)
        for k, v in st1.__dict__.items():
            row[f"y_{k}"] = float(v)
        rows.append(row)
    return rows


def rollout_predict(
    replay_rows: list[dict[str, float | int | str]],
    *,
    loaded: LoadedWorldModel,
    horizon: int | None = None,
) -> list[dict[str, float]]:
    if not replay_rows:
        return []
    max_h = min(horizon if horizon is not None else len(replay_rows), len(replay_rows))
    state_cols = loaded.state_feature_cols
    target_cols = loaded.target_cols
    prev_state = state_from_row({k: str(v) for k, v in replay_rows[0].items()}, state_cols)
    trajectory: list[dict[str, float]] = [dict(prev_state)]
    for k in range(max_h):
        row = replay_rows[k]
        delta = clip_delta_vector(
            loaded.predict_delta_row(row, state_override=prev_state),
            target_cols,
            max_abs_delta=None,
        )
        prev_state = clip_state_features(
            delta_to_state_features(prev_state, delta, target_cols, state_cols)
        )
        trajectory.append(dict(prev_state))
    return trajectory


def compare_clinical_at_horizon(
    *,
    wm_trajectory: list[dict[str, float]],
    sim_history: list,
    horizon: int,
) -> dict[str, Any]:
    if not wm_trajectory or not sim_history:
        return {"status": "too_short"}
    h = min(horizon, len(wm_trajectory) - 1, len(sim_history) - 1)
    wm = wm_trajectory[h]
    sim = sim_history[h]
    wm_resp = float(wm.get("s_clinical_response", wm.get("clinical_response", 0.0)))
    sim_resp = float(sim.effects.clinical_response)
    wm_ae = float(wm.get("s_ae_severity", wm.get("ae_severity", 0.0)))
    sim_ae = float(sim.toxicity.ae_severity)
    wm_tox_ema = float(wm.get("s_toxicity_ema", wm.get("toxicity_ema", 0.0)))
    sim_tox_ema = float(sim.meta.toxicity_ema)
    wm_active = float(wm.get("s_drug_active", wm.get("drug_active", 0.0))) >= 0.5
    sim_active = bool(sim.treatment.drug_active)
    return {
        "horizon": h,
        "final_response_error": abs(wm_resp - sim_resp),
        "final_ae_severity_error": abs(wm_ae - sim_ae),
        "final_toxicity_ema_error": abs(wm_tox_ema - sim_tox_ema),
        "drug_active_match": wm_active == sim_active,
        "wm": {
            "clinical_response": wm_resp,
            "ae_severity": wm_ae,
            "toxicity_ema": wm_tox_ema,
            "drug_active": wm_active,
        },
        "simulator": {
            "clinical_response": sim_resp,
            "ae_severity": sim_ae,
            "toxicity_ema": sim_tox_ema,
            "drug_active": sim_active,
        },
    }


def grounded_rollout_from_history(
    history: list,
    *,
    drug_name: str,
    artifacts_dir: Path | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    loaded = load_world_model(artifacts_dir)
    rows = history_to_transition_rows(history, drug_name=drug_name)
    traj = rollout_predict(rows, loaded=loaded, horizon=horizon)
    h = horizon if horizon is not None else len(history) - 1
    clinical = compare_clinical_at_horizon(
        wm_trajectory=traj, sim_history=history, horizon=h
    )
    return {
        "artifacts_dir": str(loaded.artifacts_dir),
        "n_replay_rows": len(rows),
        "wm_trajectory_steps": len(traj),
        "wm_trajectory": traj,
        "clinical_compare": clinical,
        "ground_truth": "simulator",
    }


def require_grounded_rollout(
    history: list,
    *,
    drug_name: str,
    artifacts_dir: Path | None = None,
    horizon: int | None = None,
) -> dict[str, Any]:
    root = artifacts_dir or default_artifacts_dir()
    if not (root / "world_model_meta.json").is_file():
        raise FileNotFoundError(
            f"World model required but missing artifacts under {root}. "
            "Train first: python -m world_model.train_baseline --csv <wm_train.csv>"
        )
    return grounded_rollout_from_history(
        history,
        drug_name=drug_name,
        artifacts_dir=root,
        horizon=horizon,
    )
