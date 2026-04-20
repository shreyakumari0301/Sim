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
    "Clinical pharmacology expert. Output ONE JSON object matching the schema for the DRUG OF "
    "INTEREST only. No prose or markdown. Floats as JSON numbers. Use null when evidence for that "
    "field is missing in the excerpts — do not fill from unrelated drug classes."
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
        "- Ignore antibiotic/oncology-only concepts unless they clearly apply to this drug.\n"
        "- If the excerpts lack evidence for a field, use JSON null for that field.\n"
    )


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

    src_total = len(pubmed_text) + len(openfda_text) + len(drugbank_text)
    null_n = _count_null_fields(parsed)
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
