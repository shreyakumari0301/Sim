from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
CLINICAL_SIM_DIR = ROOT / "clinical_sim"
if str(CLINICAL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(CLINICAL_SIM_DIR))

from csv_bundle import build_text_bundle
from llm_compiler import compile_rule_tables, load_repo_dotenv
from loop import run_simulation
from state import Patient, Treatment, WorldState


def _default_processed(name: str) -> Path:
    return ROOT / "data" / "processed" / name


def _build_initial_state(
    *,
    age: int,
    weight: float,
    renal_function: float,
    metaboliser: str,
    initial_dose: float,
    horizon: int,
) -> WorldState:
    patient = Patient(
        age=age,
        weight=weight,
        renal_function=renal_function,
        genotype={"cyp450_metaboliser": metaboliser, "hla_risk": False},
        exposure_history=[],
    )
    treatment = Treatment(
        drug_active=True,
        dose_level=initial_dose,
        schedule=list(range(0, horizon, 7)),
        arm_assignment="treatment",
    )
    state = WorldState()
    return state.copy_updated(patient=patient, treatment=treatment)


def _run_single_scenario(
    *,
    rules_dict: dict,
    timesteps: int,
    age: int,
    weight: float,
    renal_function: float,
    metaboliser: str,
    initial_dose: float,
) -> tuple[dict, list[dict], dict]:
    initial_state = _build_initial_state(
        age=age,
        weight=weight,
        renal_function=renal_function,
        metaboliser=metaboliser,
        initial_dose=initial_dose,
        horizon=timesteps,
    )
    history = run_simulation(
        initial_state=initial_state,
        rule_tables=rules_dict,
        n_timesteps=timesteps,
        verbose=False,
    )
    final = history[-1]
    final_metrics = {
        "response": round(float(final.effects.clinical_response), 4),
        "ae_severity": int(final.toxicity.ae_severity),
        "toxicity_ema": round(float(final.meta.toxicity_ema), 4),
        "drug_active": bool(final.treatment.drug_active),
    }
    trajectory_rows = []
    for idx, step in enumerate(history):
        trajectory_rows.append(
            {
                "timestep": idx,
                "transition_count": int(step.meta.state_transition_count),
                "action_dose_level": float(step.treatment.dose_level),
                "action_drug_active": bool(step.treatment.drug_active),
                "state_response": round(float(step.effects.clinical_response), 4),
                "state_toxicity_ema": round(float(step.meta.toxicity_ema), 4),
                "state_ae_severity": int(step.toxicity.ae_severity),
            }
        )
    initial_state_row = {
        "age": age,
        "weight": weight,
        "renal_function": renal_function,
        "metaboliser": metaboliser,
        "initial_dose": initial_dose,
    }
    return final_metrics, trajectory_rows, initial_state_row


def main() -> None:
    load_repo_dotenv()
    st.set_page_config(page_title="Scenario A/B Clinical Sim", page_icon=":test_tube:")
    st.title("A/B Scenario Compare (Fixed Drug)")
    st.caption("Run exactly two scenarios and compare: response, ae_severity, toxicity_ema, drug_active.")

    if not os.environ.get("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is required.")
        st.stop()

    drug = st.text_input("Drug (fixed for A and B)", value="metformin").strip()
    timesteps = st.number_input("Timesteps", min_value=7, max_value=365, value=90, step=1)

    with st.expander("Data sources", expanded=False):
        openfda_csv = st.text_input("OpenFDA CSV", value=str(_default_processed("openfda_v1.csv")))
        ncbi_csv = st.text_input("NCBI CSV", value=str(_default_processed("ncbi_data.csv")))
        drugbank_csv = st.text_input("DrugBank CSV", value=str(_default_processed("drugbank.csv")))

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Scenario A")
        dose_a = st.number_input("A: Initial dose", min_value=0.0, max_value=5000.0, value=100.0, step=50.0)
        renal_a = st.slider("A: Renal function", min_value=0.1, max_value=1.5, value=0.5, step=0.05)
        metab_a = st.selectbox("A: CYP450 metaboliser", options=["poor", "normal", "rapid"], index=0)
        age_a = st.number_input("A: Age", min_value=18, max_value=100, value=58, step=1)
        weight_a = st.number_input("A: Weight (kg)", min_value=30.0, max_value=250.0, value=80.0, step=1.0)

    with c2:
        st.subheader("Scenario B")
        dose_b = st.number_input("B: Initial dose", min_value=0.0, max_value=5000.0, value=1000.0, step=50.0)
        renal_b = st.slider("B: Renal function", min_value=0.1, max_value=1.5, value=1.0, step=0.05)
        metab_b = st.selectbox("B: CYP450 metaboliser", options=["poor", "normal", "rapid"], index=2)
        age_b = st.number_input("B: Age", min_value=18, max_value=100, value=58, step=1)
        weight_b = st.number_input("B: Weight (kg)", min_value=30.0, max_value=250.0, value=80.0, step=1.0)

    if st.button("Run A/B comparison", type="primary", use_container_width=True):
        if not drug:
            st.error("Drug is required.")
            st.stop()

        with st.spinner("Compiling rules from evidence once for fixed drug..."):
            pubmed_text, openfda_text, drugbank_text = build_text_bundle(
                drug,
                openfda_csv=Path(openfda_csv),
                ncbi_csv=Path(ncbi_csv),
                drugbank_csv=Path(drugbank_csv),
                drugbank_id=None,
            )
            rules = compile_rule_tables(
                pubmed_text,
                openfda_text,
                drugbank_text,
                dry_run=False,
                drug=drug,
                show_llm_output=False,
                reject_weak_extraction=True,
            )

        with st.spinner("Running scenario A and B..."):
            out_a, traj_a, init_a = _run_single_scenario(
                rules_dict=rules.to_dict(),
                timesteps=int(timesteps),
                age=int(age_a),
                weight=float(weight_a),
                renal_function=float(renal_a),
                metaboliser=metab_a,
                initial_dose=float(dose_a),
            )
            out_b, traj_b, init_b = _run_single_scenario(
                rules_dict=rules.to_dict(),
                timesteps=int(timesteps),
                age=int(age_b),
                weight=float(weight_b),
                renal_function=float(renal_b),
                metaboliser=metab_b,
                initial_dose=float(dose_b),
            )

        st.subheader("LLM Response Table")
        st.table([rules.to_dict()])

        st.subheader("Initial State Table")
        st.table(
            [
                {"scenario": "A", **init_a},
                {"scenario": "B", **init_b},
            ]
        )

        st.subheader("Transition + Action + State Table")
        st.markdown("**Scenario A**")
        st.dataframe(traj_a, use_container_width=True)
        st.markdown("**Scenario B**")
        st.dataframe(traj_b, use_container_width=True)

        st.subheader("Final Output Comparison")
        st.table(
            [
                {"metric": "response", "scenario_a": out_a["response"], "scenario_b": out_b["response"]},
                {"metric": "ae_severity", "scenario_a": out_a["ae_severity"], "scenario_b": out_b["ae_severity"]},
                {"metric": "toxicity_ema", "scenario_a": out_a["toxicity_ema"], "scenario_b": out_b["toxicity_ema"]},
                {"metric": "drug_active", "scenario_a": out_a["drug_active"], "scenario_b": out_b["drug_active"]},
            ]
        )


if __name__ == "__main__":
    main()
