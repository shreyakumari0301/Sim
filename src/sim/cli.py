"""CLI entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from sim.manifest import RunManifest, utc_now_iso
from sim.paths import DataLayout
from sim.settings import get_settings


def _cmd_init(_args: argparse.Namespace) -> int:
    s = get_settings()
    layout = DataLayout.from_settings(s).ensure()
    manifest_path = layout.manifests / f"init_{utc_now_iso().replace(':', '')}.json"
    RunManifest(source="init", status="completed", notes="Data directories created.").write(
        manifest_path
    )
    print(f"Layout ready under {layout.root}")
    print(f"  raw:       {layout.raw}")
    print(f"  processed: {layout.processed}")
    print(f"  manifests: {layout.manifests}")
    print(f"Wrote {manifest_path}")
    return 0


def _cmd_info(_args: argparse.Namespace) -> int:
    s = get_settings()
    layout = DataLayout.from_settings(s)
    print(f"DATA_DIR:  {layout.root}")
    print(f"raw:       {layout.raw}")
    print(f"processed: {layout.processed}")
    print(f"manifests: {layout.manifests}")
    return 0


def _cmd_ingest_openfda(args: argparse.Namespace) -> int:
    from sim.ingest.openfda import run_openfda_ingest

    s = get_settings()
    layout = DataLayout.from_settings(s)
    try:
        jsonl_path, manifest_path = run_openfda_ingest(
            s,
            layout,
            endpoint=args.endpoint,
            search=(args.search or None),
            limit_per_page=args.limit_per_page,
            max_pages=args.max_pages,
        )
    except httpx.RequestError as e:
        print(
            "OpenFDA request failed (network). "
            "Try curl -vI https://api.fda.gov/drug/label.json?limit=1; "
            "on WSL set HTTP_FORCE_IPV4=1 or raise HTTP_CONNECT_TIMEOUT/HTTP_READ_TIMEOUT in .env.",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1
    print(f"JSONL:    {jsonl_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def _cmd_ingest_ncbi(args: argparse.Namespace) -> int:
    from sim.ingest.ncbi import run_ncbi_ingest

    s = get_settings()
    layout = DataLayout.from_settings(s)
    try:
        jsonl_path, manifest_path = run_ncbi_ingest(
            s,
            layout,
            db=args.db,
            term=args.term,
            retmax=args.retmax,
            max_pages=args.max_pages,
            efetch_batch_size=args.efetch_batch_size,
            efetch_rettype=(args.efetch_rettype or None),
            efetch_retmode=args.efetch_retmode,
        )
    except httpx.RequestError as e:
        print(
            "NCBI E-utilities request failed (network). "
            "Try curl -vI https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=cancer&retmode=json; "
            "on WSL set HTTP_FORCE_IPV4=1 or raise timeouts in .env.",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1
    print(f"JSONL:    {jsonl_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def _cmd_ingest_drugbank_import(args: argparse.Namespace) -> int:
    from sim.ingest.drugbank import run_drugbank_import_xml

    s = get_settings()
    layout = DataLayout.from_settings(s)
    xml_path = Path(args.xml_path)
    out_xml_path, manifest_path = run_drugbank_import_xml(
        s,
        layout,
        xml_path=xml_path,
    )
    print(f"XML:      {out_xml_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def _cmd_ingest_drugbank_download(_args: argparse.Namespace) -> int:
    from sim.ingest.drugbank import run_drugbank_download

    s = get_settings()
    layout = DataLayout.from_settings(s)
    try:
        out_xml_path, manifest_path = run_drugbank_download(s, layout)
    except httpx.RequestError as e:
        print(
            "DrugBank download failed (network/auth). "
            "Verify DRUGBANK_DOWNLOAD_URL, DRUGBANK_API_KEY, and DRUGBANK_AUTH_SCHEME in .env.",
            file=sys.stderr,
        )
        print(e, file=sys.stderr)
        return 1
    print(f"XML:      {out_xml_path}")
    print(f"Manifest: {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sim", description="sim data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create data directories and a sample run manifest.")
    p_init.set_defaults(_fn=_cmd_init)

    p_info = sub.add_parser("info", help="Print resolved data paths from settings.")
    p_info.set_defaults(_fn=_cmd_info)

    p_ingest = sub.add_parser("ingest", help="Download external data into data/raw.")
    ingest_sub = p_ingest.add_subparsers(dest="ingest_target", required=True)

    p_openfda = ingest_sub.add_parser(
        "openfda",
        help="Fetch OpenFDA JSON (paginated) into data/raw/openfda/...",
    )
    p_openfda.add_argument(
        "--endpoint",
        default="drug/label",
        help="API path without host, e.g. drug/label or drug/event (default: drug/label).",
    )
    p_openfda.add_argument(
        "--search",
        default=None,
        help='Optional search clause, e.g. openfda.generic_name:"aspirin"',
    )
    p_openfda.add_argument(
        "--limit-per-page",
        type=int,
        default=100,
        help="Records per request (1–1000, default: 100).",
    )
    p_openfda.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Safety cap on pagination (default: 1).",
    )
    p_openfda.set_defaults(_fn=_cmd_ingest_openfda)

    p_ncbi = ingest_sub.add_parser(
        "ncbi",
        help="NCBI Entrez esearch + efetch into data/raw/ncbi/...",
    )
    p_ncbi.add_argument(
        "--db",
        default="pubmed",
        help="Entrez database, e.g. pubmed, pmc, mesh (default: pubmed).",
    )
    p_ncbi.add_argument(
        "--term",
        required=True,
        help="Entrez query, e.g. aspirin[Title] AND review[PT].",
    )
    p_ncbi.add_argument(
        "--retmax",
        type=int,
        default=20,
        help="Max IDs per esearch page (default: 20).",
    )
    p_ncbi.add_argument(
        "--max-pages",
        type=int,
        default=1,
        help="Safety cap on esearch pagination (default: 1).",
    )
    p_ncbi.add_argument(
        "--efetch-batch-size",
        type=int,
        default=200,
        help="Max IDs per efetch request (default: 200; NCBI guidance).",
    )
    p_ncbi.add_argument(
        "--efetch-rettype",
        default="medline",
        help="efetch rettype (default: medline; use empty string for DB default).",
    )
    p_ncbi.add_argument(
        "--efetch-retmode",
        default="text",
        help="efetch retmode (default: text).",
    )
    p_ncbi.set_defaults(_fn=_cmd_ingest_ncbi)

    p_drugbank_import = ingest_sub.add_parser(
        "drugbank-import",
        help="Import an existing DrugBank XML file into data/raw/drugbank/...",
    )
    p_drugbank_import.add_argument(
        "--xml-path",
        default="full database.xml",
        help='Path to existing DrugBank XML file (default: "full database.xml").',
    )
    p_drugbank_import.set_defaults(_fn=_cmd_ingest_drugbank_import)

    p_drugbank_download = ingest_sub.add_parser(
        "drugbank-download",
        help="Download DrugBank full database XML using DRUGBANK_* settings.",
    )
    p_drugbank_download.set_defaults(_fn=_cmd_ingest_drugbank_download)

    args = parser.parse_args(argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    sys.exit(main())
