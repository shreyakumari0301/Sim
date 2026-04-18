"""
Live OpenAI calls — skipped unless explicitly enabled.

Run (from repo root, with clinical_sim on PYTHONPATH):

  RUN_LLM_INTEGRATION=1 OPENAI_API_KEY=sk-... pytest clinical_sim/tests/test_llm_integration.py -v

Default ``pytest clinical_sim/tests`` does not call the API.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from budget import TokenBudget
from llm_compiler import compile_rule_tables, load_repo_dotenv

load_repo_dotenv()


def _llm_integration_enabled() -> bool:
    return (
        os.environ.get("RUN_LLM_INTEGRATION") == "1"
        and bool(os.environ.get("OPENAI_API_KEY", "").strip())
    )


@pytest.mark.skipif(not _llm_integration_enabled(), reason="Set RUN_LLM_INTEGRATION=1 and OPENAI_API_KEY")
def test_compile_rule_tables_live_openai(tmp_path: Path) -> None:
    """One real API call; uses an isolated budget file and no rate-limit wait."""
    budget_path = tmp_path / "budget.json"
    b = TokenBudget(budget_path)
    b.configure(daily_token_limit=100_000, min_call_interval_s=0, per_call_max_tokens=4_000)

    pubmed = "pmid: 1\ntitle: Aspirin PK\nabstract: Short half-life, platelet COX-1."
    openfda = "warnings: bleeding risk\nmechanism_of_action: COX inhibition"
    drugbank = "half_life: Acetylsalicylic acid has plasma half-life ~15-20 min at low dose."

    rules = compile_rule_tables(
        pubmed,
        openfda,
        drugbank,
        dry_run=False,
        budget_file=budget_path,
        show_llm_output=False,
        reject_weak_extraction=False,
    )

    assert rules.version != "dry_run"
    assert rules.source_summary or rules.half_life > 0
