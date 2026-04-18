from state import DiseaseState, Patient, Tolerance, WorldState


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
