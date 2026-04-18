"""Typed rule table for simulation layers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DiscontinuationRules(BaseModel):
    non_response_day: int = 60
    non_response_cutoff: float = 0.2
    grade4_auto_stop: bool = True


class RuleTable(BaseModel):
    # Provenance
    version: str = "0.0"
    source_summary: str = ""

    # Layer 1
    half_life: float = 12.0
    kd: float = 50.0
    emax: float = 0.9
    mic: float = 10.0
    pathway_suppression: float = 0.3
    tox_rate: float = 0.02
    tolerance_rate: float = 0.001
    tolerance_threshold: float = 500.0
    receptor_recovery: float = 0.01
    rebound_magnitude: float = 0.3
    rebound_decay: float = 0.15
    biomarker_sensitivity: float = 1.0
    response_threshold: float = 0.5

    # Layer 2
    response_rate_alpha: float = 3.0
    response_rate_beta: float = 2.0
    ae_probability: float = 0.05
    ae_severity_weights: list = Field(default_factory=lambda: [0.6, 0.2, 0.1, 0.07, 0.03])
    noise_sd: float = 0.02
    hill_n: float = 1.5

    # Layer 3
    tox_halt_grade: int = 3
    escalation_threshold: float = 0.4
    escalation_day: int = 14
    response_eval_day: int = 30
    max_dose: float = 400.0
    dose_step: float = 50.0
    de_escalation_grade: int = 2
    discontinuation_rules: DiscontinuationRules = Field(default_factory=DiscontinuationRules)

    def to_dict(self) -> dict[str, Any]:
        """Flat dict for layer functions (discontinuation_rules nested)."""
        d = self.model_dump()
        d["discontinuation_rules"] = self.discontinuation_rules.model_dump()
        return d
