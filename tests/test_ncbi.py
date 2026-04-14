"""NCBI ingest (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sim.ingest.ncbi import (
    _efetch_url,
    _esearch_url,
    run_ncbi_ingest,
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


def test_esearch_url_contains_params(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    u = _esearch_url(s, db="pubmed", term="test", retstart=0, retmax=10)
    assert "esearch.fcgi" in u
    assert "db=pubmed" in u
    assert "term=test" in u


def test_efetch_url_joins_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    s = Settings(_env_file=None)
    u = _efetch_url(s, db="pubmed", ids=["1", "2"], rettype="medline", retmode="text")
    assert "efetch.fcgi" in u
    assert "id=1%2C2" in u or "id=1,2" in u


def test_run_ncbi_ingest_writes_jsonl(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HTTP_MAX_RETRIES", "0")
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    layout = DataLayout.from_settings(s)

    esearch_body = {
        "esearchresult": {
            "count": "1",
            "idlist": ["123"],
        }
    }

    calls: list[str] = []

    class FakeResp:
        def __init__(self, text: str, json_body: dict | None = None) -> None:
            self.status_code = 200
            self._text = text
            self._json = json_body

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            assert self._json is not None
            return self._json

        @property
        def text(self) -> str:
            return self._text

    class FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeClient:
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def get(self, url: str) -> FakeResp:
            calls.append(url)
            if "esearch.fcgi" in url:
                return FakeResp("", json_body=esearch_body)
            if "efetch.fcgi" in url:
                return FakeResp("PMID-123\nTITLE- x\n", json_body=None)
            raise AssertionError(url)

    with patch("sim.ingest.ncbi.httpx.Client", return_value=FakeClient()):
        jsonl_path, manifest_path = run_ncbi_ingest(
            s,
            layout,
            db="pubmed",
            term="test",
            retmax=10,
            max_pages=1,
            efetch_batch_size=200,
            efetch_rettype="medline",
            efetch_retmode="text",
        )

    assert jsonl_path.exists()
    assert manifest_path.exists()
    assert calls and "eutils.ncbi.nlm.nih.gov" in calls[0]
    text = jsonl_path.read_text(encoding="utf-8").strip()
    assert "efetch_responses" in text
    assert "123" in text
    man = manifest_path.read_text(encoding="utf-8")
    assert "completed" in man
    assert "ncbi" in man.lower()
