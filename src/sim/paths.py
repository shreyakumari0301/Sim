"""Resolved paths under `DATA_DIR` (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sim.settings import Settings


@dataclass(frozen=True)
class DataLayout:
    """Canonical directories for raw ingests, processed tables, and run manifests."""

    root: Path
    raw: Path
    processed: Path
    manifests: Path

    @staticmethod
    def from_settings(settings: Settings) -> "DataLayout":
        root = settings.resolved_data_dir()
        return DataLayout(
            root=root,
            raw=root / "raw",
            processed=root / "processed",
            manifests=root / "manifests",
        )

    def ensure(self) -> "DataLayout":
        for p in (self.raw, self.processed, self.manifests):
            p.mkdir(parents=True, exist_ok=True)
        return self
