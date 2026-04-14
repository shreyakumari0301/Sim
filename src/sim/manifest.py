"""Run manifests for provenance (Phase 2)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_head_short() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RunManifest(BaseModel):
    """One ingest or pipeline run; written as JSON next to outputs."""

    manifest_version: str = Field(default="1", description="Schema version for this file.")
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(description="Logical source name, e.g. openfda, ncbi, drugbank.")
    started_at_utc: str = Field(default_factory=utc_now_iso)
    status: str = Field(default="started", description="started | completed | failed")
    git_commit: str | None = Field(default_factory=git_head_short)
    notes: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path
