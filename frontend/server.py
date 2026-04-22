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


def _transition_feature_row(
    *,
    s0,
    s1,
    drug_name: str,
    run_id: str,
    timestep: int,
) -> dict[str, float | int | str]:
    from world_model.adapter import infer_action, patient_context, state_to_vector
    from world_model.drug_rules import drug_id_for_name

    st = state_to_vector(s0)
    st1 = state_to_vector(s1)
    act = infer_action(s0, s1)
    ctx = patient_context(s0)
    row: dict[str, float | int | str] = {
        "run_id": run_id,
        "drug_name": drug_name,
        "drug_id": drug_id_for_name(drug_name),
        "timestep": timestep,
        "age": ctx["age"],
        "renal_function": ctx["renal_function"],
        "hepatic_function": ctx["hepatic_function"],
        "cyp450_metaboliser_id": ctx["cyp450_metaboliser_id"],
    }
    for k, v in st.__dict__.items():
        row[f"s_{k}"] = float(v)
    for k, v in act.__dict__.items():
        row[f"a_{k}"] = float(v)
    for k, v in st1.__dict__.items():
        row[f"y_{k}"] = float(v)
    return row


def _maybe_world_model_block(
    *,
    history: list,
    drug: str,
    feature_cols: list[str],
    target_cols: list[str],
    model,
) -> dict:
    if len(history) < 2:
        return {"status": "too_short", "detail": "Need at least two states for one-step compare."}
    t = min(5, len(history) - 2)
    s0, s1 = history[t], history[t + 1]
    row = _transition_feature_row(
        s0=s0,
        s1=s1,
        drug_name=drug,
        run_id="dashboard",
        timestep=t,
    )
    try:
        x = [[float(row[c]) for c in feature_cols]]
    except KeyError as e:
        return {
            "status": "feature_mismatch",
            "detail": f"Row missing column expected by trained model: {e}",
        }
    pred = model.predict(x)[0]
    pred_map = {target_cols[i]: float(pred[i]) for i in range(len(target_cols))}
    actual_map = {c: float(row[c]) for c in target_cols if c in row}
    err = None
    if actual_map:
        diffs = {k: pred_map[k] - actual_map[k] for k in target_cols if k in actual_map}
        err = {
            "mae_selected": sum(abs(v) for v in diffs.values()) / max(len(diffs), 1),
            "per_feature_abs": {k: abs(v) for k, v in list(diffs.items())[:12]},
        }
    return {
        "status": "ok",
        "timestep": t,
        "state_vector_t": {k: row[k] for k in row if k.startswith("s_")},
        "action_vector": {k: row[k] for k in row if k.startswith("a_")},
        "predicted_next_features": pred_map,
        "simulator_next_features": actual_map,
        "error_vs_simulator": err,
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
    meta_path = artifacts_dir / "world_model_meta.json"
    model_path = artifacts_dir / "world_model_rf.joblib"
    alt_pkl = artifacts_dir / "world_model_rf.pkl"
    wm_block: dict = {"artifacts_dir": str(artifacts_dir)}
    if not meta_path.is_file():
        wm_block["status"] = "no_artifact"
        wm_block["detail"] = "Train a model (see CLINICAL_SIM_COMMANDS.md) to enable RF predictions."
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = meta.get("feature_cols") or []
        target_cols = meta.get("target_cols") or []
        path = model_path if model_path.is_file() else alt_pkl
        if not path.is_file():
            wm_block["status"] = "no_model_file"
            wm_block["detail"] = f"Missing {model_path.name} / {alt_pkl.name}"
        else:
            try:
                import joblib

                model = joblib.load(path)
            except Exception:
                import pickle

                model = pickle.loads(path.read_bytes())
            wm_block.update(
                _maybe_world_model_block(
                    history=history,
                    drug=drug,
                    feature_cols=feature_cols,
                    target_cols=target_cols,
                    model=model,
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
