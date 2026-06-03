"""Stable normalized MAE excludes near-constant target dimensions."""

from __future__ import annotations

import numpy as np

from world_model.core import clip_state_features, compute_target_scales, normalized_mae_report


def test_near_constant_target_excluded_from_stable_aggregate() -> None:
    y = np.array([[1.0, 100.0], [1.0, 102.0], [1.0, 98.0]])
    pred = np.array([[1.1, 110.0], [0.9, 90.0], [1.0, 100.0]])
    cols = ["y_a", "y_b"]
    scale, stable_mask, excluded = compute_target_scales(y, cols, min_std_for_norm=0.5)
    assert "y_a" in excluded
    report = normalized_mae_report(y, pred, target_cols=cols, scale=scale, stable_mask=stable_mask)
    assert report["normalized_mae_mean_stable_targets"] < report["normalized_mae_mean_all_targets"]


def test_clip_state_features_bounds() -> None:
    s = clip_state_features({"s_clinical_response": 2.5, "s_ae_severity": 9.0, "s_plasma_conc": -1.0})
    assert s["s_clinical_response"] == 1.0
    assert s["s_ae_severity"] == 4.0
    assert s["s_plasma_conc"] == 0.0
