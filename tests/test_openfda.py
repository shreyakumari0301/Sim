"""OpenFDA ingest (mocked HTTP)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from sim.ingest.openfda import (
    _build_url,
    _force_ipv4_socket,
    _normalize_endpoint,
    run_openfda_ingest,
)
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
        "DATA_DIR",
        "LOG_LEVEL",
        "HTTP_MAX_RETRIES",
        "HTTP_BACKOFF_BASE_SECONDS",
        "HTTP_CONNECT_TIMEOUT",
        "HTTP_READ_TIMEOUT",
        "HTTP_FORCE_IPV4",
    ):
        monkeypatch.delenv(key, raising=False)


def test_force_ipv4_restores_getaddrinfo() -> None:
    import socket

    orig = socket.getaddrinfo
    with _force_ipv4_socket():
        assert socket.getaddrinfo is not orig
    assert socket.getaddrinfo is orig


def test_normalize_endpoint() -> None:
    assert _normalize_endpoint("drug/label") == "drug/label.json"
    assert _normalize_endpoint("drug/label.json") == "drug/label.json"


def test_build_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    u = _build_url(s, "drug/label", {"limit": 1, "skip": 0})
    assert u.startswith("https://api.fda.gov/")
    assert "drug/label.json" in u
    assert "limit=1" in u


def test_run_openfda_ingest_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HTTP_MAX_RETRIES", "0")
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    layout = DataLayout.from_settings(s)

    page = {
        "meta": {"results": {"total": 1}},
        "results": [{"id": "test"}],
    }

    class FakeResp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return page

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> FakeResp:
            assert "api.fda.gov" in url
            return FakeResp()

    with patch("sim.ingest.openfda.httpx.Client", return_value=FakeClient()):
        jsonl_path, manifest_path = run_openfda_ingest(
            s,
            layout,
            endpoint="drug/label",
            search=None,
            limit_per_page=10,
            max_pages=1,
        )

    assert jsonl_path.exists()
    assert manifest_path.exists()
    text = jsonl_path.read_text(encoding="utf-8").strip()
    assert '"results"' in text
    man = manifest_path.read_text(encoding="utf-8")
    assert "completed" in man
