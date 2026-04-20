"""LLM-assisted compilation of RuleTable from text sources (offline from simulation loop)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

from budget import TokenBudget
from rule_tables import DiscontinuationRules, RuleTable

# Compact schema for the prompt (saves input tokens). Parsing still expects the same keys.
PARAM_SCHEMA = (
    '{"half_life":float,"kd":float,"emax":float,"mic":float,"pathway_suppression":float,'
    '"tox_rate":float,"tolerance_rate":float,"tolerance_threshold":float,"receptor_recovery":float,'
    '"rebound_magnitude":float,"rebound_decay":float,"biomarker_sensitivity":float,'
    '"response_threshold":float,"response_rate_alpha":float,"response_rate_beta":float,'
    '"ae_probability":float,"ae_severity_weights":[5 floats summing to 1],'
    '"noise_sd":float,"hill_n":float,"tox_halt_grade":int,"escalation_threshold":float,'
    '"escalation_day":int,"response_eval_day":int,"max_dose":float,"dose_step":float,'
    '"de_escalation_grade":int,"non_response_day":int,"non_response_cutoff":float,'
    '"grade4_auto_stop":bool,"source_summary":string}'
)

SYSTEM_PROMPT = (
    "You are a pharmacology model that MUST output a complete JSON object.\n"
    "TASK:\n"
    "Extract or estimate pharmacokinetic (PK), pharmacodynamic (PD), toxicity, and trial-control parameters "
    "for the DRUG OF INTEREST from the provided sources.\n"
    "CRITICAL RULES:\n"
    "- Fill all fields in the schema.\n"
    "- Do not return null unless absolutely impossible.\n"
    "- If a value is missing, estimate a clinically plausible value using pharmacological knowledge.\n"
    "- Keep values internally consistent (e.g., high tox_rate implies higher ae_probability).\n"
    "- Output ONLY valid JSON; no markdown, no explanation.\n"
    "RANGE GUIDANCE:\n"
    "- half_life: 1-48 hours.\n"
    "- kd: 50-200 for weak/non-receptor metabolic drugs (e.g., metformin), 1-20 for potent targeted drugs.\n"
    "- emax: 0.2-0.5 for mild metabolic drugs, 0.6-0.9 for strong drugs.\n"
    "- pathway_suppression: 0.2-0.5 metabolic, 0.5-0.9 strong inhibitors.\n"
    "- tox_rate: 0.001-0.01 low toxicity, >0.05 high toxicity.\n"
    "- ae_probability: 0.01-0.1.\n"
    "- hill_n: 1.0-2.0.\n"
    "- tox_halt_grade: 2-4.\n"
    "- response_threshold: 0.05-0.3.\n"
    "- max_dose: use realistic clinical max (metformin about 2000 mg).\n"
    "- ae_severity_weights must sum to 1."
)


def load_repo_dotenv() -> None:
    """Load ``<repo>/.env`` (``KEY=value`` lines). Does not override existing env vars."""
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        val = val.strip().strip('"').strip("'")
        os.environ[key] = val


def _llm_source_limits() -> tuple[int, int, int]:
    """
    Chars per source sent to the model.

    - ``LLM_TOTAL_SOURCE_CHARS`` — if set, split evenly three ways (default **2400** → 800 each).
    - Else ``LLM_PUBMED_CHARS`` / ``LLM_OPENFDA_CHARS`` / ``LLM_DRUGBANK_CHARS`` (defaults **800** each).
    """
    total_env = os.environ.get("LLM_TOTAL_SOURCE_CHARS")
    if total_env is not None and total_env.strip() != "":
        try:
            n = max(0, int(total_env)) // 3
            return (n, n, n)
        except ValueError:
            pass

    def _one(name: str, default: str) -> int:
        try:
            return max(0, int(os.environ.get(name, default)))
        except ValueError:
            return int(default)

    return (
        _one("LLM_PUBMED_CHARS", "800"),
        _one("LLM_OPENFDA_CHARS", "800"),
        _one("LLM_DRUGBANK_CHARS", "800"),
    )


def _trim_sources(pubmed_text: str, openfda_text: str, drugbank_text: str) -> tuple[str, str, str]:
    pm, of, db = _llm_source_limits()
    return pubmed_text[:pm], openfda_text[:of], drugbank_text[:db]


def llm_prompt_char_counts(pubmed_text: str, openfda_text: str, drugbank_text: str) -> tuple[int, int, int]:
    """Lengths actually sent to the model after ``LLM_*_CHARS`` / ``LLM_TOTAL_SOURCE_CHARS`` trimming."""
    pm, of, db = _trim_sources(pubmed_text, openfda_text, drugbank_text)
    return len(pm), len(of), len(db)


def _count_null_fields(parsed: dict[str, Any]) -> int:
    """Count null values except ``source_summary`` (allowed to be sparse)."""
    skip = {"source_summary"}
    return sum(1 for k, v in parsed.items() if k not in skip and v is None)


def _validate_llm_extraction(parsed: dict[str, Any], *, total_source_chars: int) -> None:
    """Fail fast if the model returned mostly nulls (avoids silent default merge)."""
    if total_source_chars < 300:
        return
    try:
        threshold = int(os.environ.get("LLM_MAX_NULL_FIELDS", "10"))
    except ValueError:
        threshold = 10
    n = _count_null_fields(parsed)
    if n > threshold:
        raise ValueError(
            f"LLM extraction too weak ({n} null fields, max allowed {threshold}). "
            "Tighten CSV filters / drug spelling, increase PUBMED_MAX_ROWS or context, then retry."
        )


def _grounding_block(drug: str) -> str:
    d = drug.strip()
    if not d:
        return ""
    return (
        f"\nDRUG OF INTEREST: {d}\n"
        "STRICT RULES:\n"
        f"- Extract parameters only for {d}. Ignore other drugs, unrelated indications, and "
        "modified forms (e.g. nitroaspirin, ester prodrugs) unless the text clearly refers to the "
        "same active moiety as this drug.\n"
        "- Prefer explicit values in source text; otherwise infer from pharmacological class.\n"
        "- Toxicity prior by class: antihyperglycemics low, NSAIDs moderate, oncology high.\n"
        "- Avoid nulls; use null only if value is absolutely impossible to estimate.\n"
        "DRUG CLASS PRIORS:\n"
        "- Antihyperglycemics (e.g., metformin): tox_rate 0.001-0.02 (very low).\n"
        "- NSAIDs: tox_rate 0.01-0.05 (moderate).\n"
        "- Oncology drugs: tox_rate 0.1-0.3 (high).\n"
        "- Widely used first-line therapies should bias toward lower toxicity.\n"
        "CALIBRATION CONSTRAINTS:\n"
        "- Aim for realistic activation behavior; weak/metabolic drugs should use lower response thresholds.\n"
        "- weak drugs: response_threshold 0.02-0.08; strong drugs: 0.1-0.3.\n"
        "- Keep tolerance_rate low (<0.005) unless strong evidence.\n"
        "- Keep noise_sd <= 0.05 and receptor_recovery <= 0.1.\n"
        "- Keep internal consistency: low tox_rate should align with low ae_probability.\n"
    )


_LOW_TOX_DRUGS = {"metformin"}
_NSAID_DRUGS = {"ibuprofen", "naproxen", "diclofenac", "aspirin", "celecoxib", "indomethacin"}
_ONCOLOGY_TOKENS = (
    "oncology",
    "antineoplastic",
    "chemotherapy",
    "tumor",
    "cancer",
)

_LAST_EXTRACTION_QC: dict[str, Any] | None = None


def get_last_extraction_qc() -> dict[str, Any] | None:
    """Return extraction QC for the most recent compile call in this process."""
    return dict(_LAST_EXTRACTION_QC) if _LAST_EXTRACTION_QC else None


def _infer_toxicity_class(drug: str, pubmed_text: str, openfda_text: str, drugbank_text: str) -> str:
    d = (drug or "").strip().casefold()
    if d in _LOW_TOX_DRUGS:
        return "low"
    if d in _NSAID_DRUGS:
        return "moderate"
    source = f"{pubmed_text}\n{openfda_text}\n{drugbank_text}".casefold()
    if any(tok in source for tok in _ONCOLOGY_TOKENS):
        return "high"
    return "unknown"


def _apply_pharmacology_priors(
    parsed: dict[str, Any], *, drug: str, pubmed_text: str, openfda_text: str, drugbank_text: str
) -> None:
    """
    Clamp high-impact toxicity controls to pharmacologically plausible ranges.
    """
    tox_class = _infer_toxicity_class(drug, pubmed_text, openfda_text, drugbank_text)

    tox = parsed.get("tox_rate")
    if isinstance(tox, (int, float)):
        tox_v = float(tox)
        if tox_class == "low":
            parsed["tox_rate"] = min(max(tox_v, 0.001), 0.02)
        elif tox_class == "moderate":
            parsed["tox_rate"] = min(max(tox_v, 0.01), 0.05)
        elif tox_class == "high":
            parsed["tox_rate"] = min(max(tox_v, 0.1), 0.3)

    halt = parsed.get("tox_halt_grade")
    if isinstance(halt, (int, float)):
        parsed["tox_halt_grade"] = max(int(halt), 3)


def _metformin_profile_defaults() -> dict[str, Any]:
    """Drug-sensitive fallback profile for metformin-like antihyperglycemics."""
    return {
        "half_life": 6.0,
        "kd": 80.0,
        "emax": 0.35,
        "pathway_suppression": 0.4,
        "tox_rate": 0.002,
        "ae_probability": 0.03,
        "tolerance_rate": 0.001,
        "tolerance_threshold": 300.0,
        "receptor_recovery": 0.02,
        "rebound_magnitude": 0.15,
        "rebound_decay": 0.1,
        "biomarker_sensitivity": 0.8,
        "hill_n": 1.2,
        "max_dose": 2000.0,
        "dose_step": 500.0,
        "response_threshold": 0.04,
        "response_rate_alpha": 2.5,
        "response_rate_beta": 1.5,
        "ae_severity_weights": [0.6, 0.25, 0.1, 0.04, 0.01],
        "noise_sd": 0.02,
        "tox_halt_grade": 3,
        "escalation_threshold": 0.3,
        "escalation_day": 14,
        "response_eval_day": 45,
        "de_escalation_grade": 2,
        "non_response_day": 60,
        "non_response_cutoff": 0.15,
        "grade4_auto_stop": False,
    }


def _maybe_apply_drug_profile_fallback(parsed: dict[str, Any], *, drug: str) -> bool:
    """
    Fill key PK/PD fields for known drugs when extraction is too sparse.
    Returns True when a profile fallback was applied.
    """
    d = (drug or "").strip().casefold()
    if not d:
        return False
    try:
        null_threshold = int(os.environ.get("LLM_PROFILE_FALLBACK_NULL_FIELDS", "10"))
    except ValueError:
        null_threshold = 12
    if _count_null_fields(parsed) <= null_threshold:
        return False

    profile: dict[str, Any] | None = None
    if d == "metformin":
        profile = _metformin_profile_defaults()
    if not profile:
        return False

    for k, v in profile.items():
        if parsed.get(k) is None:
            parsed[k] = v
    return True


def _apply_calibration_guardrails(parsed: dict[str, Any]) -> None:
    """Clamp unstable outputs into clinically plausible operating ranges."""
    tol = parsed.get("tolerance_rate")
    if isinstance(tol, (int, float)):
        parsed["tolerance_rate"] = min(float(tol), 0.005)

    noise = parsed.get("noise_sd")
    if isinstance(noise, (int, float)):
        parsed["noise_sd"] = min(float(noise), 0.05)

    rec = parsed.get("receptor_recovery")
    if isinstance(rec, (int, float)):
        parsed["receptor_recovery"] = min(float(rec), 0.1)

    emax = parsed.get("emax")
    thr = parsed.get("response_threshold")
    if isinstance(emax, (int, float)) and isinstance(thr, (int, float)) and float(emax) < 0.5:
        parsed["response_threshold"] = min(float(thr), 0.08)

    tox = parsed.get("tox_rate")
    ae = parsed.get("ae_probability")
    if isinstance(tox, (int, float)) and isinstance(ae, (int, float)) and float(tox) <= 0.01:
        parsed["ae_probability"] = min(float(ae), 0.08)


def compile_rule_tables(
    pubmed_text: str,
    openfda_text: str,
    drugbank_text: str,
    prior_version: Optional[RuleTable] = None,
    dry_run: bool = False,
    *,
    drug: str | None = None,
    budget_file: Path | None = None,
    show_llm_output: bool = False,
    reject_weak_extraction: bool = False,
) -> RuleTable:
    """
    Call the LLM to extract rule_tables from raw data.
    Budget-gated: will raise RuntimeError if budget exhausted.

    ``budget_file`` — optional path for TokenBudget state (tests use a temp file).
    ``show_llm_output`` — when True and not dry_run, print raw model JSON and merged RuleTable.
    ``drug`` — grounds the prompt (recommended).
    ``reject_weak_extraction`` — if True, raise when the model returns too many JSON nulls (strict QC).
        Default False: always merge non-null fields into defaults (extractor + fallback), like earlier behavior.
    """
    if dry_run:
        return RuleTable(version="dry_run", source_summary="dry run — no API call")

    load_repo_dotenv()

    from openai import OpenAI

    budget = TokenBudget(budget_file)
    # Default 0s between LLM compiles (day-to-day). Set LLM_MIN_CALL_INTERVAL_S=300 to throttle.
    try:
        _rl = int(os.environ.get("LLM_MIN_CALL_INTERVAL_S", "0"))
    except ValueError:
        _rl = 0
    budget.configure(min_call_interval_s=max(0, _rl))
    # Older .sim_budget.json may still have per_call_max_tokens=2000; LLM estimate needs more headroom.
    budget.ensure_min_per_call_tokens()

    estimated_tokens = _estimate_tokens(*_trim_sources(pubmed_text, openfda_text, drugbank_text))
    allowed, reason = budget.can_call(estimated_tokens)
    if not allowed:
        raise RuntimeError(f"LLM call blocked by budget: {reason}")

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
    # Default OpenAI; set OPENAI_BASE_URL=https://openrouter.ai/api/v1 for OpenRouter keys (sk-or-v1-...).
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    # OpenRouter example: OPENAI_MODEL=openai/gpt-4o-mini

    patch_note = ""
    if prior_version:
        patch_note = (
            f"\nExisting parameters (version {prior_version.version}): "
            f"{json.dumps(prior_version.to_dict(), indent=2)[:600]}\n"
            f"Update only parameters where the new data gives better evidence. "
            f"Return ALL fields regardless.\n"
        )

    pm, of, db = _trim_sources(pubmed_text, openfda_text, drugbank_text)
    drug_s = (drug or "").strip()
    ground = _grounding_block(drug_s) if drug_s else ""

    user_prompt = f"""{ground}DATA:
PubMed:{pm}
OpenFDA:{of}
DrugBank:{db}
{patch_note}
Schema (same keys, JSON values): {PARAM_SCHEMA}
"""

    try:
        max_out = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "1200"))
    except ValueError:
        max_out = 1200
    max_out = max(256, min(max_out, 4096))

    response = client.chat.completions.create(
        model=model,
        max_tokens=max_out,
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

    global _LAST_EXTRACTION_QC
    src_total = len(pubmed_text) + len(openfda_text) + len(drugbank_text)
    raw_null_n = _count_null_fields(parsed)
    profile_fallback_applied = False
    if (drug or "").strip():
        profile_fallback_applied = _maybe_apply_drug_profile_fallback(parsed, drug=(drug or ""))
    if (drug or "").strip():
        _apply_pharmacology_priors(
            parsed,
            drug=(drug or ""),
            pubmed_text=pubmed_text,
            openfda_text=openfda_text,
            drugbank_text=drugbank_text,
        )
    _apply_calibration_guardrails(parsed)
    null_n = _count_null_fields(parsed)
    total_fields = len(RuleTable.model_fields) - 1  # exclude discontinuation_rules (nested)
    confidence = max(0.0, min(1.0, 1.0 - (null_n / float(total_fields))))
    _LAST_EXTRACTION_QC = {
        "drug": (drug or "").strip(),
        "raw_null_fields": raw_null_n,
        "postprocess_null_fields": null_n,
        "total_fields": total_fields,
        "confidence": confidence,
        "profile_fallback_applied": profile_fallback_applied,
    }
    if reject_weak_extraction:
        _validate_llm_extraction(parsed, total_source_chars=src_total)
    elif null_n > int(os.environ.get("LLM_WARN_NULL_FIELDS", "15")):
        print(
            f"  Note: LLM returned {null_n} null fields (merge uses defaults for those). "
            f"Use --strict-llm or LLM_MAX_NULL_FIELDS to enforce, or increase context (LLM_PUBMED_CHARS, …).",
            file=sys.stderr,
        )

    rules = _build_rule_table(parsed, prior_version, budget)
    if show_llm_output:
        print("\n=== LLM raw JSON (model output) ===\n", file=sys.stdout)
        print(raw, file=sys.stdout)
        print("\n=== RuleTable after merge with defaults ===\n", file=sys.stdout)
        print(json.dumps(rules.to_dict(), indent=2), file=sys.stdout)
        print(file=sys.stdout)
    return rules


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
    try:
        _rl = int(os.environ.get("LLM_MIN_CALL_INTERVAL_S", "0"))
    except ValueError:
        _rl = 0
    budget.configure(min_call_interval_s=max(0, _rl))
    budget.ensure_min_per_call_tokens()
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
        drug=new_data.get("drug"),
    )


def _estimate_tokens(pubmed: str, openfda: str, drugbank: str) -> int:
    """Rough input-token estimate (chars/4) + system + completion reserve for budget checks."""
    try:
        max_out = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "1200"))
    except ValueError:
        max_out = 1200
    max_out = max(256, min(max_out, 4096))
    prompt_chars = (
        len(SYSTEM_PROMPT)
        + len(pubmed)
        + len(openfda)
        + len(drugbank)
        + len(PARAM_SCHEMA)
        + 120
    )
    return int(prompt_chars / 4) + max_out + 80


def _data_has_changed(new_data: dict) -> bool:
    return bool(new_data.get("pubmed") or new_data.get("openfda") or new_data.get("drugbank"))
