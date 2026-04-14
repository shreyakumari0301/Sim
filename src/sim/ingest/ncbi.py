"""NCBI Entrez E-utilities: esearch + batched efetch (Phase 3)."""

from __future__ import annotations

import json
import random
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import httpx

from sim.ingest.openfda import _force_ipv4_socket, _httpx_timeout, _request_json
from sim.manifest import RunManifest, file_sha256, utc_now_iso
from sim.paths import DataLayout
from sim.settings import Settings


def _eutils_url(settings: Settings, path: str) -> str:
    base = str(settings.ncbi_eutils_base).rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _request_text(client: httpx.Client, settings: Settings, url: str) -> str:
    """GET with the same retry semantics as OpenFDA `_request_json`, but return body text."""
    last_err: Exception | None = None
    for attempt in range(settings.http_max_retries + 1):
        try:
            r = client.get(url)
            if r.status_code == 429 or 500 <= r.status_code < 600:
                if attempt >= settings.http_max_retries:
                    r.raise_for_status()
                delay = settings.http_backoff_base_seconds * (2**attempt)
                delay += random.uniform(0, 0.25 * delay)
                time.sleep(delay)
                continue
            r.raise_for_status()
            return r.text
        except httpx.RequestError as e:
            last_err = e
            if attempt >= settings.http_max_retries:
                break
            delay = settings.http_backoff_base_seconds * (2**attempt)
            delay += random.uniform(0, 0.25 * delay)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def _entrez_common_params(settings: Settings) -> dict[str, Any]:
    p: dict[str, Any] = {"tool": settings.ncbi_tool}
    if settings.ncbi_email:
        p["email"] = settings.ncbi_email
    if settings.ncbi_api_key:
        p["api_key"] = settings.ncbi_api_key
    return p


def _esearch_url(
    settings: Settings,
    *,
    db: str,
    term: str,
    retstart: int,
    retmax: int,
) -> str:
    q = {
        **_entrez_common_params(settings),
        "db": db,
        "term": term,
        "retstart": retstart,
        "retmax": retmax,
        "retmode": "json",
    }
    q = {k: v for k, v in q.items() if v is not None and v != ""}
    return _eutils_url(settings, "esearch.fcgi") + "?" + urlencode(q)


def _efetch_url(
    settings: Settings,
    *,
    db: str,
    ids: list[str],
    rettype: str | None,
    retmode: str,
) -> str:
    q = {
        **_entrez_common_params(settings),
        "db": db,
        "id": ",".join(ids),
        "retmode": retmode,
    }
    if rettype:
        q["rettype"] = rettype
    q = {k: v for k, v in q.items() if v is not None and v != ""}
    return _eutils_url(settings, "efetch.fcgi") + "?" + urlencode(q)


def _id_batches(ids: list[str], batch_size: int) -> Iterator[list[str]]:
    for i in range(0, len(ids), batch_size):
        yield ids[i : i + batch_size]


def iter_ncbi_esearch_efetch_pages(
    settings: Settings,
    *,
    db: str,
    term: str,
    retmax: int,
    max_pages: int,
    efetch_batch_size: int,
    efetch_rettype: str | None,
    efetch_retmode: str,
) -> Iterator[dict[str, Any]]:
    """Yield one dict per esearch page: esearch JSON plus efetch payloads for ID batches."""
    if retmax < 1:
        raise ValueError("retmax must be at least 1")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if efetch_batch_size < 1:
        raise ValueError("efetch_batch_size must be at least 1")

    timeout = _httpx_timeout(settings)
    stack = _force_ipv4_socket() if settings.http_force_ipv4 else nullcontext()
    with stack:
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            http2=False,
        ) as client:
            for page in range(max_pages):
                retstart = page * retmax
                url = _esearch_url(
                    settings,
                    db=db,
                    term=term,
                    retstart=retstart,
                    retmax=retmax,
                )
                esearch = _request_json(client, settings, url)
                idlist: list[str] = []
                er = esearch.get("esearchresult")
                if isinstance(er, dict):
                    raw = er.get("idlist")
                    if isinstance(raw, list):
                        idlist = [str(x) for x in raw]

                efetch_responses: list[str] = []
                for batch in _id_batches(idlist, efetch_batch_size):
                    furl = _efetch_url(
                        settings,
                        db=db,
                        ids=batch,
                        rettype=efetch_rettype,
                        retmode=efetch_retmode,
                    )
                    efetch_responses.append(_request_text(client, settings, furl))

                yield {
                    "retstart": retstart,
                    "retmax": retmax,
                    "ids": idlist,
                    "esearch": esearch,
                    "efetch_responses": efetch_responses,
                }

                if not idlist:
                    break
                if len(idlist) < retmax:
                    break

                total: int | None = None
                if isinstance(er, dict) and er.get("count") is not None:
                    try:
                        total = int(str(er.get("count")))
                    except ValueError:
                        total = None
                if total is not None and retstart + len(idlist) >= total:
                    break


def run_ncbi_ingest(
    settings: Settings,
    layout: DataLayout,
    *,
    db: str,
    term: str,
    retmax: int,
    max_pages: int,
    efetch_batch_size: int,
    efetch_rettype: str | None,
    efetch_retmode: str,
) -> tuple[Path, Path]:
    """Write esearch+efetch pages to JSONL under raw/ncbi/<db>/..., manifest; return paths."""
    layout.ensure()
    safe_db = db.strip().replace("/", "_")
    run = RunManifest(
        source="ncbi",
        status="started",
        notes="NCBI Entrez esearch + efetch.",
        extra={
            "db": db,
            "term": term,
            "retmax": retmax,
            "max_pages": max_pages,
            "efetch_batch_size": efetch_batch_size,
            "efetch_rettype": efetch_rettype,
            "efetch_retmode": efetch_retmode,
        },
    )

    out_dir = layout.raw / "ncbi" / safe_db / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pages.jsonl"

    manifest_path = layout.manifests / f"ncbi_{safe_db}_{utc_now_iso().replace(':', '')}.json"

    pages_written = 0
    total_ids = 0
    try:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for page in iter_ncbi_esearch_efetch_pages(
                settings,
                db=db,
                term=term,
                retmax=retmax,
                max_pages=max_pages,
                efetch_batch_size=efetch_batch_size,
                efetch_rettype=efetch_rettype,
                efetch_retmode=efetch_retmode,
            ):
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                pages_written += 1
                total_ids += len(page.get("ids") or [])

        run.status = "completed"
        run.extra["pages_written"] = pages_written
        run.extra["total_ids_across_pages"] = total_ids
        run.extra["output_dir"] = str(out_dir)
        run.extra["jsonl_path"] = str(jsonl_path)
        run.extra["pages_jsonl_sha256"] = file_sha256(jsonl_path)
    except Exception as e:
        run.status = "failed"
        run.notes = (run.notes or "") + f" | error: {e!s}"
        raise
    finally:
        if run.status == "started":
            run.status = "interrupted"
            run.notes = (run.notes or "") + " | run ended before completion"
        run.write(manifest_path)

    return jsonl_path, manifest_path
