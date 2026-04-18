from state import Treatment, WorldState
from layer1 import RT, apply_layer1


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
