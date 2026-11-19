"""Does the exported service reproduce the holdout score the engine recorded? An end-to-end check on practice data.

One practice case (``planning_gate_v2/practice``, invented tables) is built on DCLab's own unattended path with the rules
arm and no model, its final stage approved as a person does, and exported as a service. The service is then started in a
fresh Python process (cwd the export folder, only the keys its README names) and scores the project's real holdout rows,
taken from the project's own data file as CSV text. The project's primary metric on those predictions is compared with the
final stage's recorded holdout metric; the same rows are posted as JSON (the two answers must agree); and a few edge
rows ("NA", an empty value, a date in another format, an unseen category) show what the service does with them.

    DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.agent_eval.export_parity_live --case pr10 --case pr02

No model is called. Everything goes to ``agent_runs/live_tests/<timestamp>-export-parity/``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SEED = 11

# the service side, run in a fresh process with cwd = the export folder: argv[1] the request file, argv[2] the answer file
CLIENT = r"""
import io, json, sys
sys.path.insert(0, ".")
job = json.load(open(sys.argv[1]))
try:
    from app import app
    client = app.test_client()
    transport = "flask test_client"
except ModuleNotFoundError as missing:
    if missing.name not in ("flask", "werkzeug"):
        raise
    # Flask is not installed here: the app's own route bodies, called as app.py calls them (read_csv, then Log.predict)
    from scorer import Scorer, read_csv
    from serve_log import Log, Refused
    scorer = Scorer(); log = Log(scorer)
    transport = "direct (flask not installed): app.py's route bodies"
    class R:
        def __init__(self, status, body): self.status_code, self.json, self.is_json = status, json.loads(json.dumps(body, default=str)), True
        def get_data(self, as_text=True): return json.dumps(self.json)
    class Client:
        def get(self, path):
            return R(200, {k: v for k, v in scorer.meta.items() if k != "input_columns"} | {"inputs": scorer.meta["input_columns"]})
        def post(self, path, data=None, json=None):
            try:
                if path == "/predict/csv":
                    frame = read_csv(data["file"][0].read(), scorer.meta, None)
                    return R(200, log.predict(frame, None))
                return R(200, log.predict(json, None))
            except Refused as error:
                return R(400, {"error": str(error)})
            except Exception as error:  # Flask would answer 500
                return R(500, {"error": type(error).__name__ + ": " + str(error)[:300]})
    client = Client()
out = {"transport": transport, "model": client.get("/model").json}
csv_bytes = open(job["csv"], "rb").read()
r = client.post("/predict/csv", data={"file": (io.BytesIO(csv_bytes), "holdout.csv")})
out["csv"] = {"status": r.status_code, "body": r.json}
r = client.post("/predict", json=job["json_rows"])
out["json"] = {"status": r.status_code, "body": r.json}
edges = {}
for name, text in job["edge_csv"].items():
    try:
        r = client.post("/predict/csv", data={"file": (io.BytesIO(text.encode()), name + ".csv")})
        edges["csv:" + name] = {"status": r.status_code, "body": r.json if r.is_json else r.get_data(as_text=True)[:400]}
    except Exception as e:
        edges["csv:" + name] = {"exception": type(e).__name__ + ": " + str(e)[:300]}
for name, row in job["edge_json"].items():
    try:
        r = client.post("/predict", json=row)
        edges["json:" + name] = {"status": r.status_code, "body": r.json if r.is_json else r.get_data(as_text=True)[:400]}
    except Exception as e:
        edges["json:" + name] = {"exception": type(e).__name__ + ": " + str(e)[:300]}
out["edges"] = edges
json.dump(out, open(sys.argv[2], "w"), default=str)
"""


def _scores(answers: list[dict[str, Any]], task: str, labels: list[Any] | None) -> Any:
    import numpy as np

    if task == "multiclass":
        return np.asarray([[a["probabilities"][str(l) if str(l) in a["probabilities"] else l] for l in labels] for a in answers], dtype=float)
    if task == "binary":
        return np.asarray([a["score"] for a in answers], dtype=float)
    return np.asarray([a["prediction"] for a in answers], dtype=float)


def _edge_rows(raw: Any, meta: dict[str, Any], first: dict[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    """Edge rows from one real holdout row (its CSV text): "NA" and an empty value in a numeric and a text column, the
    time column in another date format, and an unseen category."""
    import pandas as pd

    types = meta.get("input_types") or {}
    cols = [c for c in raw.columns if c in types]
    numeric = next((c for c in cols if str(types[c]).startswith(("int", "Int", "float"))), None)
    texts = [c for c in cols if str(types[c]) in ("object", "str", "string", "category") and c != meta.get("time_column")
             and not raw[c].str.match(r"^\d{4}-\d{2}-\d{2}").all()]
    # a category the model uses: few levels (an identifier such as booking_ref is dropped by the recipe and proves nothing)
    text = min(texts, key=lambda c: raw[c].nunique()) if texts else None
    time_col = meta.get("time_column")
    variants: dict[str, dict[str, str]] = {"unchanged": dict(first)}
    if numeric:
        variants[f"NA_in_{numeric}"] = {**first, numeric: "NA"}
        variants[f"empty_{numeric}"] = {**first, numeric: ""}
    if text:
        variants[f"NA_in_{text}"] = {**first, text: "NA"}
        variants[f"empty_{text}"] = {**first, text: ""}
        variants[f"unseen_{text}"] = {**first, text: "zzz_never_seen_level"}
    if time_col and time_col in first and first[time_col]:
        try:
            when = pd.to_datetime(first[time_col])
            variants[f"date_dmy_{time_col}"] = {**first, time_col: when.strftime("%d/%m/%Y")}
            variants[f"date_words_{time_col}"] = {**first, time_col: when.strftime("%B %d, %Y")}
            variants[f"date_garbage_{time_col}"] = {**first, time_col: "not a date"}
        except (ValueError, TypeError):
            pass
    else:  # no time column: a date-looking column, if any
        date_like = next((c for c in cols if "date" in c.lower() or "time" in c.lower() or c.lower().endswith("_at")), None)
        if date_like and first.get(date_like):
            variants[f"date_other_{date_like}"] = {**first, date_like: "31/12/2021"}
    csvs = {name: pd.DataFrame([row]).to_csv(index=False) for name, row in variants.items()}
    jsons: dict[str, Any] = {name: row for name, row in variants.items()}
    if numeric:  # JSON gives real types: a null and the string "NA"
        jsons[f"json_null_{numeric}"] = {**first, numeric: None}
    return csvs, jsons


def run_case(case_id: str, run_dir: Path) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    from dclab_rnd.agent_eval.planning_gate_v2.practice import FOLDER
    from dclab_rnd.agent_eval.planning_gate_v2.run import build, load_held
    from dclab_rnd.expansion import runner as rn
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import engine, service
    from dclab_rnd.studio.data import data_path, load_table

    cases, _, tables = load_held(FOLDER)
    case = next(c for c in cases if c["id"] == case_id)
    frames = getattr(tables, f"make_{case_id}")(SEED)
    folder = run_dir / "projects" / f"{case_id}-{SEED}-rules"
    row = build(case, frames, "rules", folder)
    store = open_stores(folder)[0]
    pid = row["project_id"]
    out: dict[str, Any] = {"case": case_id, "seed": SEED, "project_id": pid, "session_status": row.get("session_status"),
                           "build_split": row.get("holdout_split")}
    record = store.read_stage(pid, "final")
    if not record:
        return {**out, "error": "no final stage was built"}
    metric = record.get("primary_metric")
    recorded = ((record.get("evidence") or {}).get("holdout_metrics") or {}).get(metric)
    out.update(metric=metric, recorded=recorded, task_type=record.get("task_type"),
               holdout_rows_recorded=(record.get("evidence") or {}).get("holdout_rows"),
               artifact=((record.get("evidence") or {}).get("service_artifact") or {}).get("complete"))
    engine.approve(store, pid, "final", actor="human")
    exported = service.export(store, pid, run_dir / "exports" / case_id)
    out["export"] = {"folder": exported["folder"], "checked": exported["checked"], "model_version": exported["meta"].get("model_version"),
                     "recipe": exported["meta"].get("recipe"), "model": exported["meta"].get("model")}
    meta = exported["meta"]

    # the holdout rows, as the data file holds them (CSV text), in the bundle's test order
    project = store.get(pid)
    prepared = engine.prepare(store, project)
    bundle = prepared.bundle
    rows = np.asarray(prepared.source_rows)[np.asarray(bundle.test_idx)]
    path = data_path(store, project)
    loaded = load_table(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    if len(raw) != len(loaded):
        return {**out, "error": f"raw text rows {len(raw)} != loaded rows {len(loaded)}: cannot map positions"}
    target = prepared.solution.target
    holdout_raw = raw.iloc[rows].reset_index(drop=True)
    features_raw = holdout_raw.drop(columns=[target])
    csv_path = run_dir / f"{case_id}_holdout.csv"
    features_raw.to_csv(csv_path, index=False)
    y_test = bundle.y_test.to_numpy()
    # the raw target beside the bundle's: the rows are the right ones only if they agree
    raw_y = loaded.iloc[rows][target].reset_index(drop=True)
    if prepared.class_labels:
        mapped = raw_y.map({lab: i for i, lab in enumerate(prepared.class_labels)})
        mapped = mapped.where(mapped.notna(), raw_y.astype(str).map({str(lab): i for i, lab in enumerate(prepared.class_labels)}))
        target_agrees = bool(np.array_equal(mapped.to_numpy(dtype=float), y_test.astype(float)))
    else:
        target_agrees = bool(np.allclose(pd.to_numeric(raw_y).to_numpy(dtype=float), y_test.astype(float)))
    out["holdout_rows_sent"] = int(len(features_raw))
    out["raw_target_matches_bundle"] = target_agrees

    # JSON rows: the same rows with real types (as load_table reads them; dates as text)
    json_rows = json.loads(loaded.iloc[rows].drop(columns=[target]).to_json(orient="records", date_format="iso"))
    first = {k: str(v) for k, v in features_raw.iloc[0].to_dict().items()}
    edge_csv, edge_json = _edge_rows(features_raw, meta, first)
    job = {"csv": str(csv_path), "json_rows": json_rows, "edge_csv": edge_csv, "edge_json": edge_json}
    job_file, answer_file = run_dir / f"{case_id}_job.json", run_dir / f"{case_id}_answers.json"
    job_file.write_text(json.dumps(job, default=str), encoding="utf-8")
    logs = run_dir / "service_logs" / case_id
    logs.mkdir(parents=True, exist_ok=True)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "TMPDIR", "LANG") if k in os.environ}
    env.update(DCLAB_SERVICE_KEY=exported["key"], DCLAB_SERVICE_LOG_DIR=str(logs), DCLAB_NO_LIVE_MODELS="1")
    if exported.get("log_key"):
        env["DCLAB_LOG_KEY"] = exported["log_key"]
    done = subprocess.run([sys.executable, "-c", CLIENT, str(job_file), str(answer_file)], cwd=exported["folder"], env=env,
                          capture_output=True, text=True, timeout=1800)
    if done.returncode != 0:
        return {**out, "error": "service process failed: " + done.stderr[-1500:]}
    answers = json.loads(answer_file.read_text(encoding="utf-8"))
    out["transport"] = answers.get("transport")
    if answers["csv"]["status"] != 200:
        return {**out, "error": f"CSV predict failed: {answers['csv']}"[:1500]}
    task = meta["task"]
    labels = meta.get("class_labels")
    csv_scores = _scores(answers["csv"]["body"]["predictions"], task, labels)
    spec = bundle.spec
    computed = rn.compute_metrics(bundle.task_type, bundle.y_test, csv_scores, n_classes=rn._n_classes(bundle), detail=True,
                                  precision_target=float(spec.settings.get("precision_target", 0.9)), alert_budget=spec.settings.get("alert_budget"))
    service_value = computed.get(metric)
    out.update(service=service_value, difference=(None if service_value is None or recorded is None else float(service_value) - float(recorded)))
    if answers["json"]["status"] == 200:
        json_scores = _scores(answers["json"]["body"]["predictions"], task, labels)
        out["csv_vs_json_max_abs_diff"] = float(np.max(np.abs(json_scores - csv_scores))) if json_scores.shape == csv_scores.shape else "shape differs"
    else:
        out["csv_vs_json_max_abs_diff"] = f"JSON predict failed: {answers['json']}"[:600]
    edges = {}
    for name, ans in answers["edges"].items():
        body = ans.get("body")
        if isinstance(body, dict) and "predictions" in body:
            edges[name] = {"status": ans["status"], "prediction": body["predictions"][0]}
        else:
            edges[name] = ans
    out["edges"] = edges
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.export_parity_live", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", action="append", default=None)
    args = parser.parse_args(argv)
    os.environ["DCLAB_NO_LIVE_MODELS"] = "1"
    run_dir = ROOT / "agent_runs" / "live_tests" / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-export-parity"
    run_dir.mkdir(parents=True)
    results = []
    for case_id in args.case or ["pr10", "pr02"]:  # one at a time
        try:
            result = run_case(case_id, run_dir)
        except Exception as error:  # noqa: BLE001 - a crash is a finding
            import traceback

            result = {"case": case_id, "error": f"{type(error).__name__}: {error}"[:1500], "trace": traceback.format_exc()[-3000:]}
        results.append(result)
        print(json.dumps(result, indent=1, default=str), flush=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    print(f"kept {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
