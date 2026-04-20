from llm_compiler import _count_null_fields, _maybe_apply_drug_profile_fallback


def test_metformin_profile_fallback_applies_on_sparse_extraction() -> None:
    parsed = {
        "half_life": None,
        "kd": None,
        "emax": None,
        "pathway_suppression": None,
        "tox_rate": None,
        "ae_probability": None,
        "hill_n": None,
        "max_dose": None,
        "dose_step": None,
        "response_threshold": None,
        "tox_halt_grade": None,
        "source_summary": "sparse",
    }
    assert _count_null_fields(parsed) > 10
    applied = _maybe_apply_drug_profile_fallback(parsed, drug="metformin")
    assert applied is True
    assert parsed["tox_rate"] == 0.005
    assert parsed["max_dose"] == 2000.0
    assert parsed["tox_halt_grade"] == 3


def test_profile_fallback_does_not_apply_for_non_target_drug() -> None:
    parsed = {
        "half_life": None,
        "kd": None,
        "emax": None,
        "pathway_suppression": None,
        "tox_rate": None,
        "ae_probability": None,
        "hill_n": None,
        "max_dose": None,
        "source_summary": "sparse",
    }
    applied = _maybe_apply_drug_profile_fallback(parsed, drug="ibuprofen")
    assert applied is False

