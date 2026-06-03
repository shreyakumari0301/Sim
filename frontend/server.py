"""
Local dashboard server: serves `frontend/index.html` and POST /api/run.

Run from repo root:
  python frontend/server.py

Then open http://127.0.0.1:8765/
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
CLINICAL = REPO_ROOT / "clinical_sim"
FRONTEND = Path(__file__).resolve().parent


def _json_handler(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    raise TypeError(type(obj))


def _ensure_clinical_path() -> None:
    if str(CLINICAL) not in sys.path:
        sys.path.insert(0, str(CLINICAL))


def _grounded_world_model_block(
    *,
    history: list,
    drug: str,
    artifacts_dir: Path,
    horizon: int = 15,
) -> dict:
    from world_model.predict import require_grounded_rollout

    if len(history) < 2:
        return {"status": "too_short", "detail": "Need at least two states for WM compare."}

    h = min(horizon, len(history) - 1)
    try:
        block = require_grounded_rollout(
            history,
            drug_name=drug,
            artifacts_dir=artifacts_dir,
            horizon=h,
        )
    except FileNotFoundError as e:
        return {"status": "no_artifact", "detail": str(e)}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

    clinical = block.get("clinical_compare", {})
    return {
        "status": "ok",
        "ground_truth": "simulator",
        "horizon": h,
        "clinical_compare": clinical,
        "artifacts_dir": block.get("artifacts_dir"),
        "wm_trajectory_steps": block.get("wm_trajectory_steps"),
    }


def _run_payload(body: dict) -> dict:
    _ensure_clinical_path()
    from csv_bundle import build_text_bundle
    from llm_compiler import compile_rule_tables, get_last_extraction_qc, load_repo_dotenv
    from loop import run_simulation
    from rule_tables import RuleTable
    from state import Patient, SimulationMeta, Treatment, WorldState

    load_repo_dotenv()

    drug = str(body.get("drug") or "metformin").strip() or "metformin"
    timesteps = max(5, min(500, int(body.get("timesteps") or 60)))
    rng_seed = int(body.get("rng_seed") or 42)
    dose_level = float(body.get("dose_level") or 200.0)
    age = float(body.get("age") or 58.0)
    renal = float(body.get("renal_function") or 0.9)
    use_llm = bool(body.get("use_llm"))

    def _default_processed(name: str) -> Path:
        return REPO_ROOT / "data" / "processed" / name

    openfda_csv = _default_processed("openfda_v1.csv")
    ncbi_csv = _default_processed("ncbi_data.csv")
    drugbank_csv = _default_processed("drugbank.csv")

    pubmed_text, openfda_text, drugbank_text = build_text_bundle(
        drug,
        openfda_csv=openfda_csv,
        ncbi_csv=ncbi_csv,
        drugbank_csv=drugbank_csv,
        drugbank_id=os.environ.get("DRUGBANK_ID"),
    )

    has_key = bool(os.environ.get("OPENAI_API_KEY"))
    dry_run = not (use_llm and has_key)
    llm_block: dict = {
        "dry_run": dry_run,
        "use_llm_requested": use_llm,
        "openai_configured": has_key,
        "source_chars": {
            "pubmed": len(pubmed_text),
            "openfda": len(openfda_text),
            "drugbank": len(drugbank_text),
        },
    }

    if dry_run:
        rules = compile_rule_tables(
            pubmed_text,
            openfda_text,
            drugbank_text,
            dry_run=True,
            drug=drug,
            show_llm_output=False,
            reject_weak_extraction=False,
        )
        llm_block["note"] = (
            "Engine-testing path: merged parameters are defaults "
            "(see RuleTable in clinical_sim/rule_tables.py). "
            "Set use_llm + OPENAI_API_KEY + CSV bundles for live extraction."
        )
    else:
        rules = compile_rule_tables(
            pubmed_text,
            openfda_text,
            drugbank_text,
            dry_run=False,
            drug=drug,
            show_llm_output=False,
            reject_weak_extraction=False,
        )
        qc = get_last_extraction_qc()
        if qc:
            llm_block["extraction_qc"] = qc

    llm_block["merged_rule_table"] = rules.model_dump(mode="json")

    patient = Patient(
        age=age,
        weight=80.0,
        renal_function=renal,
        genotype={"cyp450_metaboliser": "normal", "hla_risk": False},
        exposure_history=[],
    )
    treatment = Treatment(
        drug_active=True,
        dose_level=dose_level,
        schedule=list(range(0, timesteps, 7)),
        arm_assignment="treatment",
    )
    meta = SimulationMeta(rng_seed=rng_seed)
    state0 = WorldState().copy_updated(patient=patient, treatment=treatment, meta=meta)

    history = run_simulation(
        initial_state=state0,
        rule_tables=rules.to_dict(),
        n_timesteps=timesteps,
        verbose=False,
    )

    def _compact(ws):
        return ws.model_dump(mode="json")

    mid_idx = len(history) // 2
    world_state_block = {
        "schema_note": "Pydantic WorldState: drug, biomarkers, effects, toxicity, tolerance, patient, treatment, meta.",
        "initial": _compact(history[0]),
        "mid": _compact(history[mid_idx]),
        "final": _compact(history[-1]),
    }

    trajectory = []
    for ws in history:
        trajectory.append(
            {
                "t": int(ws.meta.t),
                "clinical_response": float(ws.effects.clinical_response),
                "plasma_conc": float(ws.drug.plasma_conc),
                "ae_severity": int(ws.toxicity.ae_severity),
                "dose_level": float(ws.treatment.dose_level),
                "response_ema": float(ws.meta.response_ema),
                "toxicity_ema": float(ws.meta.toxicity_ema),
                "disease_state": str(ws.effects.disease_state.value),
                "drug_active": bool(ws.treatment.drug_active),
            }
        )

    artifacts_dir = CLINICAL / "world_model" / "artifacts"
    wm_horizon = max(3, min(60, int(body.get("wm_horizon") or 15)))
    wm_block: dict = {"artifacts_dir": str(artifacts_dir), "wm_horizon": wm_horizon}
    wm_block.update(
        _grounded_world_model_block(
            history=history,
            drug=drug,
            artifacts_dir=artifacts_dir,
            horizon=wm_horizon,
        )
    )

    return {
        "llm": llm_block,
        "world_state": world_state_block,
        "world_model": wm_block,
        "trajectory": trajectory,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            data = (FRONTEND / "index.html").read_bytes()
            self._send(200, data, "text/html; charset=utf-8")
            return
        self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/run":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send(400, b'{"detail":"invalid json"}', "application/json; charset=utf-8")
            return
        try:
            out = _run_payload(body)
            payload = json.dumps(out, default=_json_handler).encode("utf-8")
            self._send(200, payload, "application/json; charset=utf-8")
        except Exception as e:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            err = json.dumps({"detail": str(e), "traceback": tb}).encode("utf-8")
            self._send(500, err, "application/json; charset=utf-8")


def main() -> int:
    port = int(os.environ.get("SIM_DASHBOARD_PORT", "8765"))
    host = os.environ.get("SIM_DASHBOARD_HOST", "127.0.0.1")
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"SIM dashboard: http://{host}:{port}/", file=sys.stderr)
    httpd.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
