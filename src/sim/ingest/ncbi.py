"""NCBI Entrez E-utilities: esearch + batched efetch and local extraction."""

from __future__ import annotations

import json
import random
import re
import time
import xml.etree.ElementTree as ET
import csv
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


def _pubmed_term_for_drug(drug_name: str, *, start_year: int, end_year: int) -> str:
    safe = drug_name.replace('"', " ").strip()
    return (
        f"({safe}[Title/Abstract]) AND "
        '(pharmacology OR toxicity OR "adverse effects" OR mechanism) AND '
        f'("{start_year}/01/01"[Date - Publication] : "{end_year}/12/31"[Date - Publication])'
    )


def _drug_names_from_openfda_csv(csv_path: Path, *, column: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise KeyError(f"Column not found in CSV: {column}")
        for row in reader:
            raw = row.get(column)
            if raw is None:
                continue
            name = str(raw).strip()
            if not name:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
    return names


def _eutils_url(settings: Settings, path: str) -> str:
    base = str(settings.ncbi_eutils_base).rstrip("/") + "/"
    return urljoin(base, path.lstrip("/"))


def _request_text(client: httpx.Client, settings: Settings, url: str) -> str:
    """GET with OpenFDA-like retries, returning response text."""
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
    params: dict[str, Any] = {"tool": settings.ncbi_tool}
    if settings.ncbi_email:
        params["email"] = settings.ncbi_email
    if settings.ncbi_api_key:
        params["api_key"] = settings.ncbi_api_key
    return params


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


def _xml_local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def parse_pubmed_efetch_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch XML into article dicts."""
    xml_text = xml_text.strip()
    if not xml_text:
        return []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    out: list[dict[str, Any]] = []
    for article in root.iter():
        if _xml_local_tag(article.tag) != "PubmedArticle":
            continue

        pmid: str | None = None
        title: str | None = None
        abstract_parts: list[str] = []
        authors: list[str] = []
        journal: str | None = None
        pub_date: str | None = None
        doi: str | None = None

        medline = next((x for x in article.iter() if _xml_local_tag(x.tag) == "MedlineCitation"), None)
        if medline is not None:
            for el in medline.iter():
                if _xml_local_tag(el.tag) == "PMID" and el.text:
                    pmid = el.text.strip()
                    break

        article_node = next((x for x in article.iter() if _xml_local_tag(x.tag) == "Article"), None)
        if article_node is not None:
            for el in article_node.iter():
                t = _xml_local_tag(el.tag)
                if t == "ArticleTitle" and el.text:
                    title = el.text.strip()
                elif t == "Abstract":
                    for ab in el.iter():
                        if _xml_local_tag(ab.tag) == "AbstractText" and ab.text:
                            label = ab.get("Label")
                            chunk = ab.text.strip()
                            abstract_parts.append(f"{label}: {chunk}" if label else chunk)
                elif t == "Author":
                    collective: str | None = None
                    last = fore = initials = None
                    for ch in el:
                        ct = _xml_local_tag(ch.tag)
                        if ct == "LastName" and ch.text:
                            last = ch.text.strip()
                        elif ct == "ForeName" and ch.text:
                            fore = ch.text.strip()
                        elif ct == "Initials" and ch.text:
                            initials = ch.text.strip()
                        elif ct == "CollectiveName" and ch.text:
                            collective = ch.text.strip()
                    if collective:
                        authors.append(collective)
                    elif last:
                        authors.append(", ".join(x for x in [last, fore or initials] if x))
                elif t == "Journal":
                    for jel in el.iter():
                        if _xml_local_tag(jel.tag) == "Title" and jel.text:
                            journal = jel.text.strip()
                            break
                    for jel in el.iter():
                        if _xml_local_tag(jel.tag) == "PubDate":
                            year = next(
                                (c.text for c in jel if _xml_local_tag(c.tag) == "Year" and c.text),
                                None,
                            )
                            month = next(
                                (c.text for c in jel if _xml_local_tag(c.tag) == "Month" and c.text),
                                None,
                            )
                            day = next(
                                (c.text for c in jel if _xml_local_tag(c.tag) == "Day" and c.text),
                                None,
                            )
                            if year:
                                pub_date = " ".join(x for x in [year, month, day] if x)
                            break
                elif t == "ELocationID" and el.get("EIdType") == "doi" and el.text:
                    doi = el.text.strip()

        if doi is None:
            for el in article.iter():
                if _xml_local_tag(el.tag) == "ArticleId" and el.get("IdType") == "doi" and el.text:
                    doi = el.text.strip()
                    break

        out.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": "\n\n".join(abstract_parts) if abstract_parts else None,
                "authors": authors,
                "journal": journal,
                "pub_date": pub_date,
                "doi": doi,
                "format": "pubmed_xml",
            }
        )

    return out


def _parse_medline_record_block(block: str) -> dict[str, Any]:
    data: dict[str, list[str]] = {}
    field: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal field, buf
        if field and buf:
            data.setdefault(field, []).append(" ".join(buf).strip())
        buf = []

    for line in block.splitlines():
        if not line.strip():
            continue
        if line.startswith("      ") and field:
            buf.append(line.strip())
            continue
        flush()
        if line.startswith("PMID-"):
            field = "PMID"
            buf = [line[5:].strip()]
        else:
            m = re.match(r"^([A-Z]{2,4})\s*-\s*(.*)$", line)
            if m:
                field = m.group(1)
                buf = [m.group(2).strip()]
            else:
                field = None
    flush()

    faus = data.get("FAU") or []
    aus = data.get("AU") or []

    return {
        "pmid": (data.get("PMID") or [None])[0],
        "title": (data.get("TI") or [None])[0],
        "abstract": "\n\n".join(data.get("AB") or []) or None,
        "authors": list(faus) if faus else list(aus),
        "journal": (data.get("JT") or data.get("TA") or [None])[0],
        "pub_date": (data.get("DP") or [None])[0],
        "doi": None,
        "format": "medline",
    }


def parse_medline_efetch_text(text: str) -> list[dict[str, Any]]:
    """Parse PubMed efetch text/medline into article dicts."""
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?m)^(?=PMID- )", text)
    out: list[dict[str, Any]] = []
    for part in parts:
        part = part.strip()
        if part:
            out.append(_parse_medline_record_block(part))
    return out


def parse_efetch_payload(text: str) -> list[dict[str, Any]]:
    t = text.lstrip()
    if t.startswith("<?xml") or t.startswith("<PubmedArticleSet") or "<PubmedArticle" in text[:2000]:
        parsed = parse_pubmed_efetch_xml(text)
        if parsed:
            return parsed
    return parse_medline_efetch_text(text)


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
    """Yield one dict per esearch page with matching efetch payloads."""
    if retmax < 1:
        raise ValueError("retmax must be at least 1")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if efetch_batch_size < 1:
        raise ValueError("efetch_batch_size must be at least 1")

    timeout = _httpx_timeout(settings)
    stack = _force_ipv4_socket() if settings.http_force_ipv4 else nullcontext()
    with stack:
        with httpx.Client(timeout=timeout, follow_redirects=True, http2=False) as client:
            for page in range(max_pages):
                retstart = page * retmax
                esearch = _request_json(
                    client,
                    settings,
                    _esearch_url(settings, db=db, term=term, retstart=retstart, retmax=retmax),
                )
                idlist: list[str] = []
                er = esearch.get("esearchresult")
                if isinstance(er, dict):
                    raw = er.get("idlist")
                    if isinstance(raw, list):
                        idlist = [str(x) for x in raw]

                efetch_responses: list[str] = []
                for batch in _id_batches(idlist, efetch_batch_size):
                    efetch_responses.append(
                        _request_text(
                            client,
                            settings,
                            _efetch_url(
                                settings,
                                db=db,
                                ids=batch,
                                rettype=efetch_rettype,
                                retmode=efetch_retmode,
                            ),
                        )
                    )

                yield {
                    "retstart": retstart,
                    "retmax": retmax,
                    "ids": idlist,
                    "esearch": esearch,
                    "efetch_responses": efetch_responses,
                }

                if not idlist or len(idlist) < retmax:
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
    """Write esearch+efetch pages to JSONL under raw/ncbi/<db>/... and a manifest."""
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


def iter_records_from_ncbi_pages_jsonl(jsonl_path: Path) -> Iterator[dict[str, Any]]:
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            page = json.loads(line)
            retstart = page.get("retstart")
            responses = page.get("efetch_responses") or []
            if not isinstance(responses, list):
                continue
            for batch_idx, payload in enumerate(responses):
                if not isinstance(payload, str):
                    continue
                for rec in parse_efetch_payload(payload):
                    rec["source_jsonl_line"] = line_no
                    rec["source_retstart"] = retstart
                    rec["source_efetch_batch"] = batch_idx
                    yield rec


def run_ncbi_extract(
    layout: DataLayout,
    jsonl_path: Path,
    *,
    out_name: str = "records.jsonl",
) -> tuple[Path, Path]:
    """Parse NCBI pages.jsonl and write one structured record per article."""
    layout.ensure()
    jsonl_path = jsonl_path.resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(str(jsonl_path))

    out_path = jsonl_path.parent / out_name
    run = RunManifest(
        source="ncbi_extract",
        status="started",
        notes="Parse efetch payloads from pages.jsonl into structured records.",
        extra={
            "input_jsonl": str(jsonl_path),
            "output_jsonl": str(out_path),
        },
    )
    manifest_path = layout.manifests / f"ncbi_extract_{utc_now_iso().replace(':', '')}.json"

    n_records = 0
    try:
        with out_path.open("w", encoding="utf-8") as out:
            for rec in iter_records_from_ncbi_pages_jsonl(jsonl_path):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_records += 1
        run.status = "completed"
        run.extra["records_written"] = n_records
        run.extra["records_jsonl_sha256"] = file_sha256(out_path)
    except Exception as e:
        run.status = "failed"
        run.notes = (run.notes or "") + f" | error: {e!s}"
        raise
    finally:
        if run.status == "started":
            run.status = "interrupted"
            run.notes = (run.notes or "") + " | run ended before completion"
        run.write(manifest_path)

    return out_path, manifest_path


def run_ncbi_ingest_from_openfda(
    settings: Settings,
    layout: DataLayout,
    *,
    db: str,
    openfda_csv_path: Path,
    drug_column: str,
    start_year: int,
    end_year: int,
    retmax: int,
    max_pages: int,
    efetch_batch_size: int,
    efetch_rettype: str | None,
    efetch_retmode: str,
) -> tuple[Path, Path]:
    """
    Read drug names from OpenFDA processed CSV and query PubMed per drug.
    Writes one JSONL line per (drug, page) with same page payload plus `drug_name` and `query`.
    """
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year")
    if not openfda_csv_path.is_file():
        raise FileNotFoundError(str(openfda_csv_path))

    layout.ensure()
    safe_db = db.strip().replace("/", "_")
    run = RunManifest(
        source="ncbi",
        status="started",
        notes="NCBI Entrez from OpenFDA drug_name_clean list with publication date window.",
        extra={
            "db": db,
            "openfda_csv_path": str(openfda_csv_path),
            "drug_column": drug_column,
            "start_year": start_year,
            "end_year": end_year,
            "retmax": retmax,
            "max_pages": max_pages,
            "efetch_batch_size": efetch_batch_size,
            "efetch_rettype": efetch_rettype,
            "efetch_retmode": efetch_retmode,
        },
    )

    out_dir = layout.raw / "ncbi" / f"{safe_db}_openfda" / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "pages.jsonl"
    manifest_path = layout.manifests / f"ncbi_{safe_db}_openfda_{utc_now_iso().replace(':', '')}.json"

    drugs = _drug_names_from_openfda_csv(openfda_csv_path, column=drug_column)
    lines_written = 0
    total_ids = 0
    try:
        print(f"Starting NCBI ingest from OpenFDA list: total drugs = {len(drugs)}")
        with jsonl_path.open("w", encoding="utf-8") as out:
            for idx, drug_name in enumerate(drugs, start=1):
                term = _pubmed_term_for_drug(
                    drug_name,
                    start_year=start_year,
                    end_year=end_year,
                )
                pages_for_drug = 0
                ids_for_drug = 0
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
                    row = {
                        "drug_name": drug_name,
                        "query": term,
                        **page,
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    lines_written += 1
                    n_ids = len(page.get("ids") or [])
                    total_ids += n_ids
                    ids_for_drug += n_ids
                    pages_for_drug += 1

                if idx % 100 == 0 or idx == len(drugs):
                    print(
                        "Progress: "
                        f"{idx}/{len(drugs)} drugs processed | "
                        f"last_drug_pages={pages_for_drug} | "
                        f"last_drug_ids={ids_for_drug} | "
                        f"total_pages={lines_written} | "
                        f"total_ids={total_ids}"
                    )

        run.status = "completed"
        run.extra["drugs_total"] = len(drugs)
        run.extra["pages_written"] = lines_written
        run.extra["total_ids_across_pages"] = total_ids
        run.extra["output_dir"] = str(out_dir)
        run.extra["jsonl_path"] = str(jsonl_path)
        run.extra["pages_jsonl_sha256"] = file_sha256(jsonl_path)
        print(
            "Completed NCBI ingest from OpenFDA list: "
            f"drugs={len(drugs)}, pages={lines_written}, total_ids={total_ids}"
        )
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