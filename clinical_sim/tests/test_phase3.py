from state import Patient, Treatment, WorldState
from layer1 import RT, apply_layer1
from layer2 import apply_layer2

RT2 = {
    "response_rate_alpha": 3.0,
    "response_rate_beta": 2.0,
    "ae_probability": 0.1,
    "ae_severity_weights": [0.5, 0.25, 0.15, 0.07, 0.03],
    "noise_sd": 0.05,
    "hill_n": 1.5,
}


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
