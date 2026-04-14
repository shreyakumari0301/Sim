"""CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

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
        print("Could not reach OpenFDA (network error).", file=sys.stderr)
        print(
            "  Try: curl -vI https://api.fda.gov/drug/label.json?limit=1",
            file=sys.stderr,
        )
        print(
            "  If that hangs: check VPN, firewall, corporate proxy, or WSL networking.",
            file=sys.stderr,
        )
        print(
            "  If curl shows IPv6 'Network is unreachable', set HTTP_FORCE_IPV4=1 in .env",
            file=sys.stderr,
        )
        print(
            "  You can raise timeouts in .env: HTTP_CONNECT_TIMEOUT, HTTP_READ_TIMEOUT",
            file=sys.stderr,
        )
        print(f"  Detail: {e}", file=sys.stderr)
        return 1
    print(f"JSONL:    {jsonl_path}")
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

    args = parser.parse_args(argv)
    fn = getattr(args, "_fn", None)
    if fn is None:
        parser.print_help()
        return 2
    return int(fn(args))


if __name__ == "__main__":
    sys.exit(main())
