"""Drug-conditioned RuleTable profiles for simulation and dataset generation."""

from __future__ import annotations

from typing import Any

from rule_tables import RuleTable

DEFAULT_DRUGS = ("metformin", "amoxicillin", "ibuprofen")


def drug_rule_table(drug: str) -> dict[str, Any]:
    """
    Minimal drug-conditioned profiles for demo/data generation.
    Falls back to default RuleTable when no profile exists.
    """
    d = drug.strip().casefold()
    base = RuleTable().to_dict()

    if d == "metformin":
        base.update(
            {
                "half_life": 6.0,
                "kd": 80.0,
                "emax": 0.35,
                "pathway_suppression": 0.4,
                "tox_rate": 0.002,
                "ae_probability": 0.03,
                "response_threshold": 0.04,
                "max_dose": 2000.0,
                "dose_step": 500.0,
            }
        )
    elif d == "amoxicillin":
        base.update(
            {
                "half_life": 1.2,
                "kd": 20.0,
                "emax": 0.75,
                "pathway_suppression": 0.7,
                "tox_rate": 0.01,
                "ae_probability": 0.05,
                "response_threshold": 0.12,
                "max_dose": 1000.0,
                "dose_step": 250.0,
            }
        )
    elif d == "ibuprofen":
        base.update(
            {
                "half_life": 2.0,
                "kd": 40.0,
                "emax": 0.55,
                "pathway_suppression": 0.5,
                "tox_rate": 0.02,
                "ae_probability": 0.07,
                "response_threshold": 0.1,
                "max_dose": 800.0,
                "dose_step": 200.0,
            }
        )
    return base


def drug_id_for_name(drug: str) -> float:
    """Stable numeric id for one-hot-style features in ML."""
    m = {name: float(i) for i, name in enumerate(DEFAULT_DRUGS)}
    key = drug.strip().casefold()
    return m.get(key, -1.0)
