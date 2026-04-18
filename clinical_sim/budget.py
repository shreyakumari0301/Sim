"""Token budget controller for LLM rule compilation."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

BUDGET_FILE = Path(".sim_budget.json")

DEFAULT_BUDGET: dict[str, Any] = {
    "daily_token_limit": 50_000,
    "per_call_max_tokens": 2_000,
    "min_call_interval_s": 300,
    "used_today": 0,
    "last_call_ts": 0.0,
    "last_reset_date": "",
    "total_calls": 0,
    "total_tokens_used": 0,
}


class TokenBudget:
    """Guards all LLM calls. Call .can_call() before every API request."""

    def __init__(self, budget_file: Path | None = None):
        self.path = budget_file or BUDGET_FILE
        self._state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return DEFAULT_BUDGET.copy()
        try:
            with open(self.path, encoding="utf-8") as f:
                raw = f.read().strip()
            if not raw:
                return DEFAULT_BUDGET.copy()
            return json.loads(raw)
        except json.JSONDecodeError:
            return DEFAULT_BUDGET.copy()

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def _reset_if_new_day(self) -> None:
        today = time.strftime("%Y-%m-%d")
        lr = self._state.get("last_reset_date")
        # Fresh budget file: anchor to today without wiping counters
        if lr in ("", None):
            self._state["last_reset_date"] = today
            self._save()
            return
        if lr != today:
            self._state["used_today"] = 0
            self._state["last_reset_date"] = today
            self._save()

    def can_call(self, estimated_tokens: int = 500) -> tuple[bool, str]:
        """Returns (allowed, reason)."""
        self._reset_if_new_day()
        now = time.time()

        elapsed = now - self._state["last_call_ts"]
        if self._state["last_call_ts"] > 0 and elapsed < self._state["min_call_interval_s"]:
            wait = int(self._state["min_call_interval_s"] - elapsed)
            return False, f"Rate limit: wait {wait}s before next call"

        if self._state["used_today"] + estimated_tokens > self._state["daily_token_limit"]:
            remaining = self._state["daily_token_limit"] - self._state["used_today"]
            return False, f"Daily budget exhausted. {remaining} tokens remaining today"

        if estimated_tokens > self._state["per_call_max_tokens"]:
            return (
                False,
                f"Requested {estimated_tokens} > per-call max {self._state['per_call_max_tokens']}",
            )

        return True, "ok"

    def record(self, tokens_used: int) -> None:
        self._reset_if_new_day()
        self._state["used_today"] += tokens_used
        self._state["total_tokens_used"] += tokens_used
        self._state["total_calls"] += 1
        self._state["last_call_ts"] = time.time()
        self._save()

    def status(self) -> dict[str, Any]:
        self._reset_if_new_day()
        return {
            "used_today": self._state["used_today"],
            "daily_limit": self._state["daily_token_limit"],
            "remaining_today": self._state["daily_token_limit"] - self._state["used_today"],
            "total_calls": self._state["total_calls"],
            "total_tokens": self._state["total_tokens_used"],
        }

    def configure(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if k in self._state:
                self._state[k] = v
        self._save()
