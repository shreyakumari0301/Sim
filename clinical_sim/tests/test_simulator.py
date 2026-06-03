"""Simulator core: state, layers, loop, cohort, LLM budget."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from budget import TokenBudget
from cohort import run_cohort_simulation
from layer1 import RT, apply_layer1
from layer2 import apply_layer2
from layer3 import apply_layer3
from llm_compiler import _validate_llm_extraction, compile_rule_tables
from loop import run_simulation
from rule_tables import RuleTable
from state import DiseaseState, Effects, Patient, Tolerance, Treatment, Toxicity, WorldState

RT2 = {
    "response_rate_alpha": 3.0,
    "response_rate_beta": 2.0,
    "ae_probability": 0.1,
    "ae_severity_weights": [0.5, 0.25, 0.15, 0.07, 0.03],
    "noise_sd": 0.05,
    "hill_n": 1.5,
}

RT3 = {
    "tox_halt_grade": 3,
    "escalation_threshold": 0.4,
    "escalation_day": 14,
    "response_eval_day": 30,
    "max_dose": 400.0,
    "dose_step": 50.0,
    "de_escalation_grade": 2,
    "discontinuation_rules": {
        "non_response_day": 60,
        "non_response_cutoff": 0.2,
        "grade4_auto_stop": True,
    },
}

ALL_RT = {
    "half_life": 12.0,
    "kd": 50.0,
    "emax": 0.9,
    "mic": 10.0,
    "pathway_suppression": 0.3,
    "tox_rate": 0.001,
    "tolerance_rate": 0.002,
    "tolerance_threshold": 500.0,
    "receptor_recovery": 0.01,
    "rebound_magnitude": 0.3,
    "rebound_decay": 0.15,
    "biomarker_sensitivity": 1.0,
    "response_threshold": 0.5,
    "response_rate_alpha": 3.0,
    "response_rate_beta": 2.0,
    "ae_probability": 0.0,
    "ae_severity_weights": [0.6, 0.2, 0.1, 0.07, 0.03],
    "noise_sd": 0.02,
    "hill_n": 1.5,
    "tox_halt_grade": 3,
    "escalation_threshold": 0.4,
    "escalation_day": 14,
    "response_eval_day": 30,
    "max_dose": 400.0,
    "dose_step": 50.0,
    "de_escalation_grade": 2,
    "discontinuation_rules": {
        "non_response_day": 999,
        "non_response_cutoff": 0.2,
        "grade4_auto_stop": True,
    },
}


def test_default_state():
    s = WorldState()
    assert s.biomarkers.receptor_occupancy == 0.0
    assert s.tolerance.tolerance_level == 0.0
    assert s.meta.t == 0


def test_copy_updated():
    s = WorldState()
    t = Tolerance(tolerance_level=0.5)
    s2 = s.copy_updated(tolerance=t)
    assert s2.tolerance.tolerance_level == 0.5
    assert s.tolerance.tolerance_level == 0.0


def test_disease_state_enum():
    assert DiseaseState.RESPONDING == "responding"


def test_plasma_decays():
    s = WorldState()
    s = s.copy_updated(drug=s.drug.model_copy(update={"plasma_conc": 100.0}))
    s2 = apply_layer1(s, RT)
    assert s2.drug.plasma_conc < 100.0


def test_dose_absorption():
    s = WorldState()
    s = s.copy_updated(treatment=Treatment(drug_active=True, dose_level=200.0, schedule=[0]))
    s2 = apply_layer1(s, RT)
    assert s2.drug.plasma_conc > 0.0


def test_tolerance_accrues():
    s = WorldState()
    s = s.copy_updated(
        drug=s.drug.model_copy(update={"plasma_conc": 200.0, "cumulative_auc": 600.0}),
        treatment=Treatment(drug_active=True, dose_level=200.0, schedule=[]),
    )
    s2 = apply_layer1(s, RT)
    assert s2.tolerance.tolerance_level > 0.0


def test_rebound_fires_when_drug_stopped():
    s = WorldState()
    s = s.copy_updated(tolerance=s.tolerance.model_copy(update={"rebound_clock": 0}))
    s2 = apply_layer1(s, RT)
    assert s2.biomarkers.pathway_activity > 1.0


def test_reproducible_with_seed():
    s = WorldState()
    s1 = apply_layer2(s, RT2)
    s2 = apply_layer2(s, RT2)
    assert abs(s1.biomarkers.target_biomarker - s2.biomarkers.target_biomarker) < 1e-9


def test_prior_exposure_adds_tolerance():
    p = Patient(exposure_history=[3000.0, 3000.0])
    s = WorldState()
    s = s.copy_updated(patient=p)
    s2 = apply_layer2(s, RT2)
    assert s2.tolerance.tolerance_level > 0.0


def test_poor_metaboliser_higher_effect():
    p_poor = Patient(genotype={"cyp450_metaboliser": "poor", "hla_risk": False})
    p_rapid = Patient(genotype={"cyp450_metaboliser": "rapid", "hla_risk": False})
    s_poor = WorldState()
    s_poor = s_poor.copy_updated(
        patient=p_poor,
        effects=s_poor.effects.model_copy(update={"clinical_response": 0.5}),
        treatment=Treatment(drug_active=True, dose_level=100.0, schedule=[]),
    )
    s_rapid = s_poor.copy_updated(patient=p_rapid)
    r_poor = apply_layer2(s_poor, RT2).effects.clinical_response
    r_rapid = apply_layer2(s_rapid, RT2).effects.clinical_response
    assert r_poor > r_rapid


def test_grade4_stops_drug():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(drug_active=True, dose_level=200.0, schedule=[]),
        toxicity=Toxicity(ae_severity=4, cumulative_tox=12.0),
    )
    s2 = apply_layer3(s, RT3)
    assert s2.treatment.drug_active is False


def test_escalation_on_low_response():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(drug_active=True, dose_level=100.0, schedule=[]),
        effects=Effects(clinical_response=0.2),
        meta=s.meta.model_copy(update={"trial_day": 20}),
    )
    s2 = apply_layer3(s, RT3)
    assert s2.treatment.dose_level > 100.0


def test_rebound_clock_starts_on_halt():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(drug_active=True, dose_level=200.0, schedule=[]),
        toxicity=Toxicity(ae_severity=3, cumulative_tox=7.0),
    )
    s2 = apply_layer3(s, RT3)
    assert s2.tolerance.rebound_clock == 0


def test_history_length():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=200.0,
            schedule=list(range(90)),
        )
    )
    history = run_simulation(s, ALL_RT, n_timesteps=30)
    assert len(history) == 31


def test_reproducible():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=200.0,
            schedule=list(range(90)),
        )
    )
    h1 = run_simulation(s, ALL_RT, n_timesteps=20)
    h2 = run_simulation(s, ALL_RT, n_timesteps=20)
    assert h1[-1].drug.plasma_conc == h2[-1].drug.plasma_conc


def test_tolerance_increases_over_time():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=300.0,
            schedule=list(range(90)),
        )
    )
    history = run_simulation(s, ALL_RT, n_timesteps=90)
    assert history[-1].tolerance.tolerance_level > history[10].tolerance.tolerance_level


def test_budget_blocks_when_exhausted():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        b = TokenBudget(tmp)
        b.configure(daily_token_limit=100, min_call_interval_s=0)
        b.record(95)
        allowed, reason = b.can_call(50)
        assert not allowed
        assert "budget" in reason.lower()
    finally:
        tmp.unlink(missing_ok=True)


def test_budget_records_correctly():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        b = TokenBudget(tmp)
        b.record(300)
        assert b.status()["used_today"] == 300
        assert b.status()["total_calls"] == 1
    finally:
        tmp.unlink(missing_ok=True)


def test_dry_run_returns_defaults():
    rt = compile_rule_tables("", "", "", dry_run=True)
    assert isinstance(rt, RuleTable)
    assert rt.version == "dry_run"
    assert rt.half_life > 0


def test_rule_table_to_dict_flat():
    rt = RuleTable()
    d = rt.to_dict()
    assert "half_life" in d
    assert "non_response_day" in d["discontinuation_rules"]


def test_weak_llm_extraction_raises(monkeypatch):
    monkeypatch.setenv("LLM_MAX_NULL_FIELDS", "2")
    bad = {f"k{i}": None for i in range(10)}
    bad["source_summary"] = "ok"
    with pytest.raises(ValueError, match="too weak"):
        _validate_llm_extraction(bad, total_source_chars=1000)


def test_meta_progression_metrics_are_updated():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=200.0,
            schedule=list(range(40)),
        )
    )
    history = run_simulation(s, ALL_RT, n_timesteps=20)
    final = history[-1]
    assert final.meta.response_ema > 0.0
    assert final.meta.toxicity_ema >= 0.0
    assert final.meta.state_transition_count >= 0


def test_cohort_simulation_returns_subgroups():
    s = WorldState()
    s = s.copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=200.0,
            schedule=list(range(40)),
        )
    )
    out = run_cohort_simulation(
        initial_state=s,
        rule_tables=ALL_RT,
        n_patients=8,
        n_timesteps=20,
        cohort_seed=11,
    )
    assert out["n_patients"] == 8
    assert "subgroup_summary" in out
    assert len(out["patients"]) == 8
