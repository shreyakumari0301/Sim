"""LLM-assisted compilation of RuleTable from text sources (offline from simulation loop)."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from budget import TokenBudget
from rule_tables import DiscontinuationRules, RuleTable

PARAM_SCHEMA = """
{
  "half_life": <float, hours>,
  "kd": <float, ng/mL — dissociation constant>,
  "emax": <float, 0.0-1.0 — maximum effect fraction>,
  "mic": <float, ng/mL — minimum inhibitory concentration>,
  "pathway_suppression": <float, 0.0-1.0 per unit occupancy>,
  "tox_rate": <float, toxicity units per timestep at full dose>,
  "tolerance_rate": <float, 0.0-0.01, daily tolerance increment>,
  "tolerance_threshold": <float, cumulative AUC at which tolerance starts>,
  "receptor_recovery": <float, 0.0-0.05, daily recovery rate>,
  "rebound_magnitude": <float, 0.0-1.0, overshoot fraction>,
  "rebound_decay": <float, daily decay of rebound>,
  "biomarker_sensitivity": <float, biomarker drop per unit effect>,
  "response_threshold": <float, 0.0-1.0>,
  "response_rate_alpha": <float, Beta dist alpha>,
  "response_rate_beta": <float, Beta dist beta>,
  "ae_probability": <float, 0.0-1.0, per-day AE probability>,
  "ae_severity_weights": <list of 5 floats summing to 1.0, grades 0-4>,
  "noise_sd": <float, standard deviation for biomarker noise>,
  "hill_n": <float, Hill coefficient, usually 1.0-3.0>,
  "tox_halt_grade": <int, 2-4>,
  "escalation_threshold": <float, 0.0-1.0>,
  "escalation_day": <int, trial day for escalation evaluation>,
  "response_eval_day": <int, trial day for response evaluation>,
  "max_dose": <float, mg>,
  "dose_step": <float, mg>,
  "de_escalation_grade": <int, 1-3>,
  "non_response_day": <int>,
  "non_response_cutoff": <float>,
  "grade4_auto_stop": <bool>,
  "source_summary": <string, ≤80 chars describing data used>
}
"""

SYSTEM_PROMPT = """
You are a clinical pharmacology expert and simulation engineer.
You will be given structured data extracted from PubMed abstracts,
OpenFDA drug labels, and DrugBank entries for a specific drug.

Your task: extract numerical parameters for a pharmacological simulation.
Respond ONLY with a valid JSON object. No prose. No markdown fences.
Use null for any parameter you cannot determine from the data.
All floats must be JSON numbers, not strings.
"""


def compile_rule_tables(
    pubmed_text: str,
    openfda_text: str,
    drugbank_text: str,
    prior_version: Optional[RuleTable] = None,
    dry_run: bool = False,
) -> RuleTable:
    """
    Call the LLM to extract rule_tables from raw data.
    Budget-gated: will raise RuntimeError if budget exhausted.
    """
    if dry_run:
        return RuleTable(version="dry_run", source_summary="dry run — no API call")

    from openai import OpenAI

    budget = TokenBudget()
    estimated_tokens = _estimate_tokens(pubmed_text, openfda_text, drugbank_text)
    allowed, reason = budget.can_call(estimated_tokens)
    if not allowed:
        raise RuntimeError(f"LLM call blocked by budget: {reason}")

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    patch_note = ""
    if prior_version:
        patch_note = (
            f"\nExisting parameters (version {prior_version.version}): "
            f"{json.dumps(prior_version.to_dict(), indent=2)[:600]}\n"
            f"Update only parameters where the new data gives better evidence. "
            f"Return ALL fields regardless.\n"
        )

    user_prompt = f"""
DATA SOURCES:

=== PubMed ===
{pubmed_text[:800]}

=== OpenFDA ===
{openfda_text[:600]}

=== DrugBank ===
{drugbank_text[:600]}

{patch_note}

Extract parameters matching this schema exactly:
{PARAM_SCHEMA}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=900,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    tokens_used = response.usage.total_tokens if response.usage else 0
    budget.record(tokens_used)

    raw = response.choices[0].message.content or ""
    raw = raw.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}") from e

    return _build_rule_table(parsed, prior_version, budget)


def _build_rule_table(
    parsed: dict[str, Any], prior_version: Optional[RuleTable], budget: TokenBudget
) -> RuleTable:
    base_rt = prior_version or RuleTable()
    base = base_rt.model_dump()
    disc = dict(base_rt.discontinuation_rules.model_dump())

    for k in ("non_response_day", "non_response_cutoff", "grade4_auto_stop"):
        if k in parsed and parsed[k] is not None:
            disc[k] = parsed.pop(k)
    parsed.pop("discontinuation_rules", None)

    source_summary = parsed.pop("source_summary", None)
    if source_summary is None:
        source_summary = base_rt.source_summary
    parsed.pop("version", None)

    field_keys = set(RuleTable.model_fields) - {"discontinuation_rules"}
    for k, v in list(parsed.items()):
        if v is None or k not in field_keys:
            continue
        base[k] = v

    base["discontinuation_rules"] = DiscontinuationRules(**disc)
    base["source_summary"] = str(source_summary)
    base["version"] = f"1.{budget.status()['total_calls']}"
    return RuleTable.model_validate(base)


def update_rules_at_milestone(
    current_rules: RuleTable,
    state_history: list,
    new_data: dict,
    force: bool = False,
) -> RuleTable:
    """Optional milestone update; returns current_rules if LLM call blocked."""
    budget = TokenBudget()
    if not force:
        allowed, reason = budget.can_call(500)
        if not allowed:
            print(f"[milestone] LLM update skipped: {reason}")
            return current_rules

    if not _data_has_changed(new_data):
        print("[milestone] No significant data change — skipping LLM update")
        return current_rules

    return compile_rule_tables(
        pubmed_text=new_data.get("pubmed", ""),
        openfda_text=new_data.get("openfda", ""),
        drugbank_text=new_data.get("drugbank", ""),
        prior_version=current_rules,
    )


def _estimate_tokens(pubmed: str, openfda: str, drugbank: str) -> int:
    prompt_chars = len(pubmed[:800]) + len(openfda[:600]) + len(drugbank[:600]) + len(PARAM_SCHEMA)
    return int(prompt_chars / 4) + 200


def _data_has_changed(new_data: dict) -> bool:
    return bool(new_data.get("pubmed") or new_data.get("openfda") or new_data.get("drugbank"))
