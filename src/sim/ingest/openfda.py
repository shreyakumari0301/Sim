"""OpenFDA API fetch with pagination (Phase 3)."""

from __future__ import annotations

import json
import random
import socket
import time
from contextlib import contextmanager, nullcontext
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import httpx

from sim.manifest import RunManifest, file_sha256, utc_now_iso
from sim.paths import DataLayout
from sim.settings import Settings


@contextmanager
def _force_ipv4_socket() -> Iterator[None]:
    """Temporarily force ``socket.getaddrinfo`` to IPv4 only (thread-local side effect)."""
    orig = socket.getaddrinfo

    def getaddrinfo(
        host: str | bytes,
        port: int | str | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple]:
        if family == 0:
            family = socket.AF_INET
        return orig(host, port, family, type, proto, flags)

    socket.getaddrinfo = getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = orig


def _httpx_timeout(settings: Settings) -> httpx.Timeout:
    # Default timeout applies to read/write/pool; connect is often the bottleneck (TLS).
    return httpx.Timeout(
        settings.http_read_timeout_seconds,
        connect=settings.http_connect_timeout_seconds,
    )


def _normalize_endpoint(endpoint: str) -> str:
    e = endpoint.strip().strip("/")
    if not e.endswith(".json"):
        e = f"{e}.json"
    return e


def _build_url(settings: Settings, endpoint: str, params: dict[str, Any]) -> str:
    base = str(settings.openfda_api_base).rstrip("/") + "/"
    path = _normalize_endpoint(endpoint)
    q = {k: v for k, v in params.items() if v is not None and v != ""}
    return urljoin(base, path) + ("?" + urlencode(q) if q else "")


def _request_json(
    client: httpx.Client,
    settings: Settings,
    url: str,
) -> dict[str, Any]:
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
            return r.json()
        except httpx.RequestError as e:
            # Includes ConnectTimeout, TLS handshake failures, DNS, etc. (not HTTPStatusError).
            last_err = e
            if attempt >= settings.http_max_retries:
                break
            delay = settings.http_backoff_base_seconds * (2**attempt)
            delay += random.uniform(0, 0.25 * delay)
            time.sleep(delay)
        except json.JSONDecodeError as e:
            last_err = e
            if attempt >= settings.http_max_retries:
                break
            delay = settings.http_backoff_base_seconds * (2**attempt)
            delay += random.uniform(0, 0.25 * delay)
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def iter_openfda_pages(
    settings: Settings,
    *,
    endpoint: str,
    search: str | None,
    limit_per_page: int,
    max_pages: int,
    initial_skip: int = 0,
) -> Iterator[dict[str, Any]]:
    """Yield full OpenFDA JSON objects, one per HTTP response (page)."""
    if limit_per_page < 1 or limit_per_page > 1000:
        raise ValueError("limit_per_page must be between 1 and 1000")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if initial_skip < 0:
        raise ValueError("initial_skip must be non-negative")

    params: dict[str, Any] = {"limit": limit_per_page}
    if search:
        params["search"] = search
    if settings.openfda_api_key:
        params["api_key"] = settings.openfda_api_key

    timeout = _httpx_timeout(settings)
    stack = _force_ipv4_socket() if settings.http_force_ipv4 else nullcontext()
    with stack:
        # HTTP/1.1 avoids HTTP/2 stalls on some WSL setups.
        with httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            http2=False,
        ) as client:
            skip = initial_skip
            for _ in range(max_pages):
                page_params = {**params, "skip": skip}
                url = _build_url(settings, endpoint, page_params)
                data = _request_json(client, settings, url)
                yield data
                results = data.get("results")
                if not isinstance(results, list) or len(results) < limit_per_page:
                    break
                skip += limit_per_page


def run_openfda_ingest(
    settings: Settings,
    layout: DataLayout,
    *,
    endpoint: str,
    search: str | None,
    limit_per_page: int,
    max_pages: int,
    initial_skip: int = 0,
) -> tuple[Path, Path]:
    """Write pages to JSONL under raw/openfda, write run manifest; return (jsonl_path, manifest_path)."""
    layout.ensure()
    run = RunManifest(
        source="openfda",
        status="started",
        notes="OpenFDA paginated download.",
        extra={
            "endpoint": endpoint,
            "search": search,
            "limit_per_page": limit_per_page,
            "max_pages": max_pages,
            "initial_skip": initial_skip,
        },
    )

    safe_ep = endpoint.strip("/").replace("/", "_")
    out_dir = layout.raw / "openfda" / safe_ep / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pages.jsonl"

    manifest_path = layout.manifests / f"openfda_{safe_ep}_{utc_now_iso().replace(':', '')}.json"

    total_records = 0
    pages_written = 0
    try:
        with jsonl_path.open("w", encoding="utf-8") as f:
            for page in iter_openfda_pages(
                settings,
                endpoint=endpoint,
                search=search,
                limit_per_page=limit_per_page,
                max_pages=max_pages,
                initial_skip=initial_skip,
            ):
                f.write(json.dumps(page, ensure_ascii=False) + "\n")
                pages_written += 1
                res = page.get("results")
                if isinstance(res, list):
                    total_records += len(res)

        run.status = "completed"
        run.extra["pages_written"] = pages_written
        run.extra["total_records_in_pages"] = total_records
        run.extra["output_dir"] = str(out_dir)
        run.extra["jsonl_path"] = str(jsonl_path)
        run.extra["pages_jsonl_sha256"] = file_sha256(jsonl_path)
    except Exception as e:
        run.status = "failed"
        run.notes = (run.notes or "") + f" | error: {e!s}"
        raise
    finally:
        # Ctrl+C / SystemExit leave status as "started" unless we mark it here (not subclasses of Exception).
        if run.status == "started":
            run.status = "interrupted"
            run.notes = (run.notes or "") + " | run ended before completion"
        run.write(manifest_path)

    return jsonl_path, manifest_path
