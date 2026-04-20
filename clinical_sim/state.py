"""World state: Pydantic models for the clinical trial simulation engine."""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DiseaseState(str, Enum):
    NAIVE = "naive"
    RESPONDING = "responding"
    STABLE = "stable"
    PROGRESSING = "progressing"
    REMISSION = "remission"


class DrugConcentration(BaseModel):
    plasma_conc: float = 0.0  # ng/mL
    tissue_conc: float = 0.0  # ng/mL
    cumulative_auc: float = 0.0  # ng/mL·h — area under curve so far
    time_above_mic: int = 0  # timesteps spent above MIC threshold


class Biomarkers(BaseModel):
    target_biomarker: float = 0.0  # domain units (e.g. ng/mL PSA). >= 0
    inflammatory_score: float = 0.0  # normalised [0.0, 10.0]
    receptor_occupancy: float = 0.0  # fraction bound [0.0, 1.0]
    pathway_activity: float = 1.0  # suppression level [0.0, 1.0]. 1.0 = fully active


class Effects(BaseModel):
    clinical_response: float = 0.0  # [0.0, 1.0] composite response score
    symptom_score: float = 10.0  # lower = better [0.0, 10.0]
    disease_state: DiseaseState = DiseaseState.NAIVE
    response_flag: bool = False  # True once clinical_response >= response_threshold


class Toxicity(BaseModel):
    ae_severity: int = 0  # CTCAE grade 0-4
    organ_stress: dict = Field(
        default_factory=lambda: {
            "liver": 0.0,
            "kidney": 0.0,
            "cardiac": 0.0,
        }
    )  # each [0.0, 1.0]
    cumulative_tox: float = 0.0  # running sum of daily tox exposure
    ae_active: List[str] = Field(default_factory=list)  # list of active AE names


class Tolerance(BaseModel):
    tolerance_level: float = 0.0  # [0.0, 1.0]. 0 = no tolerance, 1 = full
    resistance_flag: bool = False  # True if acquired resistance detected
    receptor_density: float = 1.0  # [0.0, 1.0]. Decreases with tolerance
    rebound_clock: int = -1  # -1 = inactive. >= 0 = timesteps since drug stopped


class Patient(BaseModel):
    age: float = 60.0
    weight: float = 75.0  # kg
    renal_function: float = 1.0  # [0.0, 1.0] eGFR normalised
    hepatic_function: float = 1.0  # [0.0, 1.0]
    genotype: dict = Field(
        default_factory=lambda: {
            "cyp450_metaboliser": "normal",  # poor|normal|rapid
            "hla_risk": False,
        }
    )
    exposure_history: List[float] = Field(default_factory=list)  # cumulative_auc per prior cycle
    current_dose: float = 0.0  # mg


class Treatment(BaseModel):
    drug_active: bool = False
    dose_level: float = 0.0  # mg
    schedule: List[int] = Field(default_factory=list)  # timesteps when dose is given
    arm_assignment: str = "control"  # "control"|"treatment"|"combo"


class SimulationMeta(BaseModel):
    t: int = 0  # current timestep
    trial_day: int = 0  # calendar day in trial
    rule_version: str = "0.0"  # version tag of current rule_tables
    rng_seed: int = 42
    response_ema: float = 0.0  # smoothed response across timesteps
    toxicity_ema: float = 0.0  # smoothed toxicity across timesteps
    state_transition_count: int = 0  # number of disease-state transitions observed


class WorldState(BaseModel):
    drug: DrugConcentration = Field(default_factory=DrugConcentration)
    biomarkers: Biomarkers = Field(default_factory=Biomarkers)
    effects: Effects = Field(default_factory=Effects)
    toxicity: Toxicity = Field(default_factory=Toxicity)
    tolerance: Tolerance = Field(default_factory=Tolerance)
    patient: Patient = Field(default_factory=Patient)
    treatment: Treatment = Field(default_factory=Treatment)
    meta: SimulationMeta = Field(default_factory=SimulationMeta)

    def copy_updated(self, **kwargs) -> "WorldState":
        """Return a new WorldState with one or more namespace fields replaced."""
        data = self.model_dump()
        for key, val in kwargs.items():
            if hasattr(val, "model_dump"):
                data[key] = val.model_dump()
            else:
                data[key] = val
        return WorldState(**data)
