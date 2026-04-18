from state import Treatment, WorldState
from loop import run_simulation

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
