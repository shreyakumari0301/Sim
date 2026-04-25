from world_model.adapter import infer_action, state_to_vector
from world_model.dataset import build_transition_dataset, transitions_to_rows
from world_model.drug_rules import drug_id_for_name


def test_drug_id_for_name_stable_and_distinct() -> None:
    a = drug_id_for_name("metformin")
    assert drug_id_for_name("metformin") == a
    assert drug_id_for_name(" Metformin ") == a
    assert drug_id_for_name("ibuprofen") != a


def test_state_to_vector_smoke() -> None:
    from state import WorldState

    s = WorldState()
    v = state_to_vector(s)
    assert v.clinical_response == 0.0
    assert v.dose_level == 0.0
    assert v.drug_active == 0.0


def test_infer_action_detects_stop() -> None:
    from state import Treatment, WorldState

    s0 = WorldState().copy_updated(treatment=Treatment(drug_active=True, dose_level=200.0, schedule=[0]))
    s1 = WorldState().copy_updated(treatment=Treatment(drug_active=False, dose_level=200.0, schedule=[0]))
    a = infer_action(s0, s1)
    assert a.action_code == 3.0
    assert a.was_active == 1.0
    assert a.is_active == 0.0


def test_build_transition_dataset_rows_match_timesteps() -> None:
    from rule_tables import RuleTable
    from state import Treatment, WorldState

    initial = WorldState().copy_updated(
        treatment=Treatment(
            drug_active=True,
            dose_level=200.0,
            schedule=list(range(0, 30, 7)),
            arm_assignment="treatment",
        )
    )
    rt = RuleTable().to_dict()
    records = build_transition_dataset(
        initial_state=initial,
        rule_tables=rt,
        drug_name="metformin",
        n_timesteps=5,
        run_id="wm_test",
    )
    assert len(records) == 5
    flat = transitions_to_rows(records)
    assert len(flat) == 5
    assert "s_clinical_response" in flat[0]
    assert "a_action_code" in flat[0]
    assert "y_clinical_response" in flat[0]
    assert flat[0]["drug_id"] == drug_id_for_name("metformin")
