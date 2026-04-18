import tempfile
from pathlib import Path

import pytest

from budget import TokenBudget
from llm_compiler import _validate_llm_extraction, compile_rule_tables
from rule_tables import RuleTable


def test_budget_blocks_when_exhausted():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        b = TokenBudget(tmp)
        b.configure(daily_token_limit=100, min_call_interval_s=0)
        b.record(95)
        allowed, reason = b.can_call(50)
        assert not allowed
        assert "budget" in reason.lower()
    finally:
        tmp.unlink(missing_ok=True)


def test_budget_records_correctly():
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = Path(f.name)
    try:
        b = TokenBudget(tmp)
        b.record(300)
        assert b.status()["used_today"] == 300
        assert b.status()["total_calls"] == 1
    finally:
        tmp.unlink(missing_ok=True)


def test_dry_run_returns_defaults():
    rt = compile_rule_tables("", "", "", dry_run=True)
    assert isinstance(rt, RuleTable)
    assert rt.version == "dry_run"
    assert rt.half_life > 0


def test_rule_table_to_dict_flat():
    rt = RuleTable()
    d = rt.to_dict()
    assert "half_life" in d
    assert "non_response_day" in d["discontinuation_rules"]


def test_weak_llm_extraction_raises(monkeypatch):
    monkeypatch.setenv("LLM_MAX_NULL_FIELDS", "2")
    bad = {f"k{i}": None for i in range(10)}
    bad["source_summary"] = "ok"
    with pytest.raises(ValueError, match="too weak"):
        _validate_llm_extraction(bad, total_source_chars=1000)
