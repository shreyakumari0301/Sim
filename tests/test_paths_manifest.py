"""Phase 2: layout and manifests."""

from pathlib import Path

import pytest

from sim.manifest import RunManifest, file_sha256
from sim.paths import DataLayout
from sim.settings import Settings, get_settings


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "OPEN_FDA_SITE",
        "NCBI_SITE",
        "OPENFDA_API_BASE",
        "OPENFDA_API_KEY",
        "OPEN_FDA_API_KEY",
        "NCBI_EUTILS_BASE",
        "NCBI_API_KEY",
        "NCBI_TOOL",
        "NCBI_EMAIL",
        "DATA_DIR",
        "LOG_LEVEL",
        "HTTP_MAX_RETRIES",
        "HTTP_BACKOFF_BASE_SECONDS",
        "HTTP_CONNECT_TIMEOUT",
        "HTTP_READ_TIMEOUT",
        "HTTP_FORCE_IPV4",
    ):
        monkeypatch.delenv(key, raising=False)


def test_data_layout_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    layout = DataLayout.from_settings(s)
    assert layout.root == (tmp_path / "d").resolve()
    assert layout.raw.name == "raw"
    assert layout.processed.name == "processed"
    layout.ensure()
    assert layout.raw.is_dir()


def test_run_manifest_write_roundtrip(tmp_path: Path) -> None:
    m = RunManifest(source="test", status="completed", notes="n")
    path = tmp_path / "m.json"
    m.write(path)
    text = path.read_text(encoding="utf-8")
    assert "test" in text
    assert m.run_id in text


def test_file_sha256(tmp_path: Path) -> None:
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    assert len(file_sha256(p)) == 64
