from cohort import run_cohort_simulation
from loop import run_simulation
from state import Treatment, WorldState

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
