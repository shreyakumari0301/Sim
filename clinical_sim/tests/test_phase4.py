from state import Effects, Toxicity, Treatment, WorldState
from layer3 import apply_layer3

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
