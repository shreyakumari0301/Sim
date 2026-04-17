"""NCBI ingest (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from sim.ingest.ncbi import (
    _efetch_url,
    _esearch_url,
    parse_efetch_payload,
    parse_pubmed_efetch_xml,
    run_ncbi_extract,
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


def test_parse_medline_extracts_fields() -> None:
    text = """PMID- 12345678
TI  - Example title
AB  - First paragraph.
FAU - Doe, Jane
AU  - Doe J
DP  - 2020 Jan 1
"""
    recs = parse_efetch_payload(text)
    assert len(recs) == 1
    r = recs[0]
    assert r["pmid"] == "12345678"
    assert r["title"] == "Example title"
    assert r["abstract"] == "First paragraph."
    assert "Doe, Jane" in r["authors"]
    assert r["pub_date"] == "2020 Jan 1"
    assert r["format"] == "medline"


def test_parse_pubmed_xml_minimal() -> None:
    xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">999</PMID>
      <Article>
        <ArticleTitle>XML title</ArticleTitle>
        <Abstract>
          <AbstractText>Abstract body.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>Ann</ForeName>
          </Author>
        </AuthorList>
        <Journal>
          <Title>Test Journal</Title>
          <JournalIssue>
            <PubDate>
              <Year>2021</Year>
              <Month>Jun</Month>
            </PubDate>
          </JournalIssue>
        </Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""
    recs = parse_pubmed_efetch_xml(xml)
    assert len(recs) == 1
    r = recs[0]
    assert r["pmid"] == "999"
    assert r["title"] == "XML title"
    assert r["abstract"] == "Abstract body."
    assert r["authors"] == ["Smith, Ann"]
    assert r["journal"] == "Test Journal"
    assert r["pub_date"] == "2021 Jun"
    assert r["format"] == "pubmed_xml"


def test_run_ncbi_extract_from_pages_jsonl(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    s = Settings(_env_file=None)
    layout = DataLayout.from_settings(s)
    layout.ensure()

    pages = tmp_path / "pages.jsonl"
    page = {
        "retstart": 0,
        "ids": ["123"],
        "esearch": {},
        "efetch_responses": ["PMID- 123\nTI  - T\nAB  - A\n"],
    }
    pages.write_text(__import__("json").dumps(page) + "\n", encoding="utf-8")

    out, man = run_ncbi_extract(layout, pages)
    assert out.exists()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = __import__("json").loads(lines[0])
    assert row["pmid"] == "123"
    assert row["title"] == "T"
    assert man.exists()
