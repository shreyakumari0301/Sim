"""Schemas, adapters, CSV/matrix helpers, bounds, and rollout state math."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from state import WorldState

# --- Schemas ---


@dataclass(frozen=True)
class StateVector:
    plasma_conc: float
    tissue_conc: float
    cumulative_auc: float
    time_above_mic: float
    target_biomarker: float
    inflammatory_score: float
    receptor_occupancy: float
    pathway_activity: float
    clinical_response: float
    symptom_score: float
    disease_state_id: float
    response_flag: float
    ae_severity: float
    cumulative_tox: float
    tolerance_level: float
    resistance_flag: float
    receptor_density: float
    rebound_clock: float
    dose_level: float
    drug_active: float
    response_ema: float
    toxicity_ema: float
    state_transition_count: float


@dataclass(frozen=True)
class ActionVector:
    action_code: float
    dose_delta: float
    target_dose: float
    was_active: float
    is_active: float
    is_dosing_day: float


@dataclass(frozen=True)
class TransitionRecord:
    run_id: str
    drug_name: str
    timestep: int
    state_t: StateVector
    action_t: ActionVector
    state_t1: StateVector
    age: float
    renal_function: float
    hepatic_function: float
    cyp450_metaboliser_id: float


@dataclass(frozen=True)
class WorldModelPrediction:
    next_state_mean: StateVector
    next_state_std: StateVector | None = None


class WorldModel(Protocol):
    def predict_next(
        self,
        state_t: StateVector,
        action_t: ActionVector,
        context: dict[str, float],
    ) -> WorldModelPrediction:
        raise NotImplementedError


# --- Adapters ---

_DISEASE_TO_ID = {
    "naive": 0.0,
    "responding": 1.0,
    "stable": 2.0,
    "progressing": 3.0,
    "remission": 4.0,
}
_CYP_TO_ID = {"poor": 0.0, "normal": 1.0, "rapid": 2.0}


def state_to_vector(state: WorldState) -> StateVector:
    ds = str(state.effects.disease_state.value)
    return StateVector(
        plasma_conc=float(state.drug.plasma_conc),
        tissue_conc=float(state.drug.tissue_conc),
        cumulative_auc=float(state.drug.cumulative_auc),
        time_above_mic=float(state.drug.time_above_mic),
        target_biomarker=float(state.biomarkers.target_biomarker),
        inflammatory_score=float(state.biomarkers.inflammatory_score),
        receptor_occupancy=float(state.biomarkers.receptor_occupancy),
        pathway_activity=float(state.biomarkers.pathway_activity),
        clinical_response=float(state.effects.clinical_response),
        symptom_score=float(state.effects.symptom_score),
        disease_state_id=float(_DISEASE_TO_ID.get(ds, 0.0)),
        response_flag=1.0 if state.effects.response_flag else 0.0,
        ae_severity=float(state.toxicity.ae_severity),
        cumulative_tox=float(state.toxicity.cumulative_tox),
        tolerance_level=float(state.tolerance.tolerance_level),
        resistance_flag=1.0 if state.tolerance.resistance_flag else 0.0,
        receptor_density=float(state.tolerance.receptor_density),
        rebound_clock=float(state.tolerance.rebound_clock),
        dose_level=float(state.treatment.dose_level),
        drug_active=1.0 if state.treatment.drug_active else 0.0,
        response_ema=float(state.meta.response_ema),
        toxicity_ema=float(state.meta.toxicity_ema),
        state_transition_count=float(state.meta.state_transition_count),
    )


def infer_action(prev_state: WorldState, next_state: WorldState) -> ActionVector:
    prev_active = bool(prev_state.treatment.drug_active)
    next_active = bool(next_state.treatment.drug_active)
    dose_delta = float(next_state.treatment.dose_level - prev_state.treatment.dose_level)
    action_code = 0.0
    if (not prev_active) and next_active:
        action_code = 1.0
    elif prev_active and (not next_active):
        action_code = 3.0
    elif dose_delta > 0.0:
        action_code = 2.0
    elif dose_delta < 0.0:
        action_code = 4.0
    is_dosing_day = 1.0 if prev_state.meta.t in prev_state.treatment.schedule else 0.0
    return ActionVector(
        action_code=action_code,
        dose_delta=dose_delta,
        target_dose=float(next_state.treatment.dose_level),
        was_active=1.0 if prev_active else 0.0,
        is_active=1.0 if next_active else 0.0,
        is_dosing_day=is_dosing_day,
    )


def patient_context(state: WorldState) -> dict[str, float]:
    cyp = str(state.patient.genotype.get("cyp450_metaboliser", "normal")).lower()
    return {
        "age": float(state.patient.age),
        "renal_function": float(state.patient.renal_function),
        "hepatic_function": float(state.patient.hepatic_function),
        "cyp450_metaboliser_id": float(_CYP_TO_ID.get(cyp, 1.0)),
    }


# --- State bounds ---

MIN_STD_FOR_NORM = 1e-3
SCALE_EPS = 1e-4

_BOUNDS: dict[str, tuple[float, float | None]] = {
    "plasma_conc": (0.0, None),
    "tissue_conc": (0.0, None),
    "cumulative_auc": (0.0, None),
    "time_above_mic": (0.0, None),
    "target_biomarker": (0.0, None),
    "inflammatory_score": (0.0, None),
    "receptor_occupancy": (0.0, 1.0),
    "pathway_activity": (0.0, 2.0),
    "clinical_response": (0.0, 1.0),
    "symptom_score": (0.0, 10.0),
    "disease_state_id": (0.0, 4.0),
    "response_flag": (0.0, 1.0),
    "ae_severity": (0.0, 4.0),
    "cumulative_tox": (0.0, None),
    "tolerance_level": (0.0, 1.0),
    "resistance_flag": (0.0, 1.0),
    "receptor_density": (0.0, 1.0),
    "rebound_clock": (-1.0, None),
    "dose_level": (0.0, None),
    "drug_active": (0.0, 1.0),
    "response_ema": (0.0, 1.0),
    "toxicity_ema": (0.0, None),
    "state_transition_count": (0.0, None),
}


def compute_target_scales(
    y_train_delta: np.ndarray,
    target_cols: list[str],
    *,
    min_std_for_norm: float = MIN_STD_FOR_NORM,
    scale_eps: float = SCALE_EPS,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw_std = np.std(y_train_delta, axis=0) if len(y_train_delta) else np.zeros(len(target_cols))
    scale = np.maximum(raw_std, scale_eps)
    stable_mask = raw_std >= min_std_for_norm
    if not np.any(stable_mask) and len(stable_mask):
        stable_mask = np.ones(len(target_cols), dtype=bool)
    excluded = [target_cols[i] for i in range(len(target_cols)) if not stable_mask[i]]
    return scale, stable_mask, excluded


def clip_state_features(state: dict[str, float]) -> dict[str, float]:
    out = dict(state)
    for key, val in list(out.items()):
        suffix = key[2:] if key.startswith("s_") else key
        bounds = _BOUNDS.get(suffix)
        if bounds is None:
            continue
        lo, hi = bounds
        v = float(val)
        if hi is not None:
            v = min(hi, max(lo, v))
        else:
            v = max(lo, v)
        out[key] = v
    return out


def clip_delta_vector(
    delta_vec: np.ndarray,
    target_cols: list[str],
    *,
    max_abs_delta: float | None = None,
) -> np.ndarray:
    out = np.array(delta_vec, dtype=np.float64, copy=True)
    if max_abs_delta is not None and max_abs_delta > 0:
        out = np.clip(out, -max_abs_delta, max_abs_delta)
    return out


# --- CSV / matrices ---


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def infer_columns(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
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


def split_drugs(
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
    if n == 1:
        return {drugs[0]}, set(), set()
    if n == 2:
        n_test = 1 if test_fraction > 0.0 else 0
        if n_test == 0:
            return set(drugs), set(), set()
        return set(drugs[n_test:]), set(), set(drugs[:n_test])
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction))) if val_fraction > 0.0 else 0
    n_train = n - n_test - n_val
    if n_train <= 0:
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


def rows_for_drugs(rows: list[dict[str, str]], drug_ids: set[str]) -> list[dict[str, str]]:
    return [r for r in rows if r["drug_id"] in drug_ids]


def to_matrix(rows: list[dict[str, str]], cols: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(cols)), dtype=np.float64)
    return np.array([[float(r[c]) for c in cols] for r in rows], dtype=np.float64)


def delta_matrix(rows: list[dict[str, str]], target_cols: list[str]) -> np.ndarray:
    if not rows:
        return np.empty((0, len(target_cols)), dtype=np.float64)
    out: list[list[float]] = []
    for r in rows:
        row_vals: list[float] = []
        for y_col in target_cols:
            suffix = y_col[2:]
            row_vals.append(float(r[y_col]) - float(r[f"s_{suffix}"]))
        out.append(row_vals)
    return np.array(out, dtype=np.float64)


def delta_from_row(row: dict[str, str], target_cols: list[str]) -> np.ndarray:
    vals = []
    for y_col in target_cols:
        suffix = y_col[2:]
        vals.append(float(row[y_col]) - float(row[f"s_{suffix}"]))
    return np.array(vals, dtype=np.float64)


def target_scales_from_meta(meta: dict, target_cols: list[str]) -> np.ndarray:
    raw = meta.get("target_scale_effective") or meta.get("target_scale_std") or {}
    vals = [float(raw.get(c, 1.0)) for c in target_cols]
    return np.array(vals, dtype=np.float64)


def stable_target_mask_from_meta(meta: dict, target_cols: list[str]) -> np.ndarray:
    excluded = set(meta.get("metrics_exclude_targets") or [])
    if excluded:
        return np.array([c not in excluded for c in target_cols], dtype=bool)
    return np.ones(len(target_cols), dtype=bool)


def normalized_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    scale: np.ndarray,
    stable_mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    abs_err = np.abs(y_pred - y_true)
    norm_err = abs_err / scale
    per_dim = np.mean(norm_err, axis=0)
    if stable_mask is not None and np.any(stable_mask):
        agg = float(np.mean(per_dim[stable_mask]))
    else:
        agg = float(np.mean(per_dim))
    return agg, per_dim


def normalized_mae_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    target_cols: list[str],
    scale: np.ndarray,
    stable_mask: np.ndarray,
) -> dict[str, float | dict[str, float] | list[str]]:
    _, per_dim = normalized_mae(y_true, y_pred, scale=scale, stable_mask=None)
    stable_agg, _ = normalized_mae(y_true, y_pred, scale=scale, stable_mask=stable_mask)
    excluded = [target_cols[i] for i in range(len(target_cols)) if not stable_mask[i]]
    per_target = {target_cols[i]: float(per_dim[i]) for i in range(len(target_cols))}
    return {
        "normalized_mae_mean_all_targets": float(np.mean(per_dim)),
        "normalized_mae_mean_stable_targets": stable_agg,
        "normalized_mae_per_target": per_target,
        "metrics_exclude_targets": excluded,
    }


# --- Rollout state math ---


def state_from_row(row: dict[str, str], state_feature_cols: list[str]) -> dict[str, float]:
    return {c: float(row[c]) for c in state_feature_cols}


def delta_to_state_features(
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
    return clip_state_features(out)


def build_x_row(
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
