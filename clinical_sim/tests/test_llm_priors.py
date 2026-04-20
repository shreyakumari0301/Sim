from llm_compiler import _apply_calibration_guardrails, _apply_pharmacology_priors


def test_low_toxicity_class_clamps_metformin() -> None:
    parsed = {"tox_rate": 0.1, "tox_halt_grade": 1}
    _apply_pharmacology_priors(
        parsed,
        drug="metformin",
        pubmed_text="",
        openfda_text="",
        drugbank_text="",
    )
    assert parsed["tox_rate"] == 0.02
    assert parsed["tox_halt_grade"] == 3


def test_nsaid_toxicity_class_clamps_upper_bound() -> None:
    parsed = {"tox_rate": 0.2, "tox_halt_grade": 2}
    _apply_pharmacology_priors(
        parsed,
        drug="ibuprofen",
        pubmed_text="",
        openfda_text="",
        drugbank_text="",
    )
    assert parsed["tox_rate"] == 0.05
    assert parsed["tox_halt_grade"] == 3


def test_oncology_text_sets_high_toxicity_band() -> None:
    parsed = {"tox_rate": 0.02, "tox_halt_grade": 4}
    _apply_pharmacology_priors(
        parsed,
        drug="exampledrug",
        pubmed_text="Antineoplastic agent used in cancer treatment.",
        openfda_text="",
        drugbank_text="",
    )
    assert parsed["tox_rate"] == 0.1
    assert parsed["tox_halt_grade"] == 4


def test_calibration_guardrails_clamp_unstable_values() -> None:
    parsed = {
        "tolerance_rate": 0.02,
        "noise_sd": 0.2,
        "receptor_recovery": 0.3,
        "emax": 0.35,
        "response_threshold": 0.2,
        "tox_rate": 0.005,
        "ae_probability": 0.2,
    }
    _apply_calibration_guardrails(parsed)
    assert parsed["tolerance_rate"] == 0.005
    assert parsed["noise_sd"] == 0.05
    assert parsed["receptor_recovery"] == 0.1
    assert parsed["response_threshold"] == 0.08
    assert parsed["ae_probability"] == 0.08

