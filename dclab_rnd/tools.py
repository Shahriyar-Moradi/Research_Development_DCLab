"""Agent tool surface over DCLab R&D evidence.

An LLM agent (the DCLab product agent, a LangGraph node, Claude, GPT, a local SLM)
should not *remember* DCLab's evidence; it should *call* it. Each tool here is a
plain Python function with a JSON schema, so the same surface works for:

* OpenAI / Anthropic style tool calling (``tool_schemas()``),
* a Model Context Protocol server (``python -m dclab_rnd.tools mcp``, needs ``pip install mcp``),
* the command line (``python -m dclab_rnd.tools call search_evidence '{"query": "..."}'``).

Tools are deterministic and read-only, except none: they never execute user code
(notebook review is static) and never train models.

Verification: ``python -m dclab_rnd.tools verify-auditor`` runs the column auditor
BLIND (no catalog, no precedents) on datasets whose leakage is already known and
reports how many known leaks it finds. See ``evidence/campaigns/agent_verification_v1``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dclab_rnd.evidence_index import ROOT, STAGE_CARDS, EvidenceIndex

_INDEX: EvidenceIndex | None = None
STAGE_ORDER = ["data_understanding", "leakage_audit", "feature_engineering", "model_selection", "optimization_reliability"]
POST_OUTCOME_NAME = re.compile(
    r"(^|[_\s-])(final|after|post|outcome|result|resolved|resolution|closed|refund(ed)?|cancel(l)?ed|cancellation|"
    r"churn(ed)?_date|end_date|duration|settled|paid|chargeback|returned|rating|review_score|registered|casual)([_\s-]|$)",
    re.IGNORECASE,
)


def _index() -> EvidenceIndex:
    """The repository's evidence index, with the workspace's accepted lessons when a workspace installed them (A5.3)."""
    global _INDEX
    if _INDEX is None:
        _INDEX = EvidenceIndex.load(ROOT)
    from . import lessons

    return lessons.merged(_INDEX)


def _brief(record: dict[str, Any], chars: int = 700) -> dict[str, Any]:
    return {
        "record_id": record["record_id"],
        "type": record["type"],
        "title": record["title"],
        "text": record["text"][:chars] + ("…" if len(record["text"]) > chars else ""),
        "citations": [c for c in record["citations"] if c],
        **({"score": record["score"]} if "score" in record else {}),
    }


# --------------------------------------------------------------------------- tools


def search_evidence(query: str, type: str | None = None, dataset: str | None = None, stage: str | None = None,
                    task_type: str | None = None, k: int = 5) -> dict[str, Any]:
    """Filter-then-rank search over rules, workflows, datasets, experiments, leakage precedents, pitfalls and findings."""
    hits = _index().search(query, k=k, type=type, dataset=dataset, stage=stage, task_type=task_type)
    return {"query": query, "filters": {"type": type, "dataset": dataset, "stage": stage, "task_type": task_type},
            "results": [_brief(h) for h in hits]}


def get_record(record_id: str) -> dict[str, Any]:
    """Return one full evidence record by ID (e.g. DCLAB-R04, WF-05, EXP-007, LEAK-bank_marketing, PIT-003)."""
    record = _index().get(record_id)
    if record is None:
        return {"error": f"No record {record_id!r}. Use search_evidence to find IDs."}
    return record


def get_rules(category: str | None = None) -> dict[str, Any]:
    """List DCLab model-building rules, optionally for one category (e.g. leakage, splitting, evaluation)."""
    rules = [r for r in _index().records if r["type"] == "rule" and (category is None or r["metadata"].get("category") == category)]
    return {"category": category, "rules": [{"rule_id": r["record_id"], "category": r["metadata"].get("category"),
                                             "statement": r["title"], "text": r["text"]} for r in rules]}


def plan_next_stage(dataset: str | None = None, completed_stages: list[str] | None = None) -> dict[str, Any]:
    """Return the next workflow stage, its method, the rules that govern it, and precedents for this dataset if any."""
    done = set(completed_stages or [])
    next_stage = next((s for s in STAGE_ORDER if s not in done), None)
    if next_stage is None:
        return {"next_stage": None, "message": "All five stages are complete. Next: independent confirmation and promotion review (DCLAB-R22)."}
    card = STAGE_CARDS[next_stage]
    index = _index()
    stage_rules = {
        "data_understanding": ["DCLAB-R01", "DCLAB-R02", "DCLAB-R03", "DCLAB-R12"],
        "leakage_audit": ["DCLAB-R04", "DCLAB-R05", "DCLAB-R06", "DCLAB-R01"],
        "feature_engineering": ["DCLAB-R07", "DCLAB-R08", "DCLAB-R09", "DCLAB-R11"],
        "model_selection": ["DCLAB-R13", "DCLAB-R14", "DCLAB-R16"],
        "optimization_reliability": ["DCLAB-R15", "DCLAB-R16", "DCLAB-R17", "DCLAB-R18", "DCLAB-R10"],
    }[next_stage]
    precedents = index.search(next_stage.replace("_", " "), k=3, type="experiment", dataset=dataset, stage=next_stage) if dataset else []
    workflow = [r for r in index.records if r["type"] == "workflow" and r["record_id"] in card["workflow"].split("/")]
    return {
        "next_stage": next_stage,
        "title": card["title"],
        "method": card["method"],
        "workflow_blocks": [_brief(w) for w in workflow],
        "rules": [_brief(index.get(r)) for r in stage_rules if index.get(r)],
        "dataset_precedents": [_brief(p) for p in precedents],
        "deterministic_owner": "Splits, metrics, artifacts and selection rules are executed by code; the agent proposes and critiques.",
    }


def review_notebook(path: str) -> dict[str, Any]:
    """Static methodology review of a Jupyter notebook with evidence-backed findings (never executes code)."""
    from dclab_rnd.copilot import review_notebook as _review

    target = Path(path)
    if not target.is_absolute():
        target = (Path.cwd() / target).resolve()
    if target.suffix != ".ipynb" or not target.exists():
        return {"error": f"{path} is not an existing .ipynb file"}
    report = _review(target, _index())
    report.pop("cells", None)
    for f in report["findings"]:
        f["proof"] = [{"record_id": p["record_id"], "type": p["type"], "title": p["title"]} for p in f["proof"]]
    return report


def review_code(source: str) -> dict[str, Any]:
    """Static methodology review of a code snippet (one notebook cell or script)."""
    from dclab_rnd.copilot import review_source

    report = review_source(source, _index())
    for f in report["findings"]:
        f["proof"] = [{"record_id": p["record_id"], "type": p["type"], "title": p["title"]} for p in f["proof"]]
    return report


def audit_columns(path: str, target: str, task: str = "auto", blind: bool = False, max_rows: int = 20000) -> dict[str, Any]:
    """Rank columns of a CSV/parquet file by leakage risk using deterministic heuristics.

    Signals: suspicious post-outcome names, identifier-like uniqueness, single-column predictive power
    that is too good, exact arithmetic reconstruction of a numeric target, and (unless ``blind``) known
    DCLab leakage precedents. Signals propose a review; only decision-time semantics confirm leakage (DCLAB-R05).
    """
    import numpy as np
    import pandas as pd

    file = Path(path)
    if not file.is_absolute():
        file = (Path.cwd() / file).resolve()
    if not file.exists():
        return {"error": f"{path} not found"}
    frame = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_csv(file, low_memory=False)
    if target not in frame.columns:
        return {"error": f"target {target!r} not in columns"}
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=42)
    return _audit_frame(frame.drop(columns=[target]), frame[target], task=task, blind=blind, source=str(path))


def _audit_frame(X: Any, y: Any, task: str = "auto", blind: bool = False, source: str = "") -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    if task == "auto":
        task = "classification" if (y.dtype == object or y.nunique() <= 20) else "regression"
    if task == "classification":
        classes = y.astype(str)
        positive = classes.value_counts().index[-1] if classes.nunique() == 2 else None
        y_bin = (classes == positive).astype(int) if positive is not None else None
    known = {} if blind else {
        c.lower(): r["record_id"]
        for r in _index().records if r["type"] == "leakage_precedent" for c in r["metadata"].get("columns", [])
    }
    rows = []
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    for col in X.columns:
        s = X[col]
        signals, score = [], 0.0
        if POST_OUTCOME_NAME.search(str(col)):
            signals.append("name suggests a post-outcome value"); score += 1.0
        uniq = s.nunique(dropna=True) / max(len(s.dropna()), 1)
        if uniq > 0.95 and (not pd.api.types.is_float_dtype(s) or (s.dropna() % 1 == 0).all()):
            signals.append(f"identifier-like ({uniq:.0%} unique)"); score += 1.5
        strength = None
        if task == "classification" and y_bin is not None:
            values = s
            if not pd.api.types.is_numeric_dtype(s):
                means = y_bin.groupby(s.astype(str)).mean()
                values = s.astype(str).map(means)
            values = pd.to_numeric(values, errors="coerce")
            mask = values.notna()
            if mask.sum() > 20 and values[mask].nunique() > 1 and y_bin[mask].nunique() == 2:
                auc = float(roc_auc_score(y_bin[mask], values[mask]))
                strength = max(auc, 1 - auc)
                if strength >= 0.80 and uniq <= 0.95:
                    signals.append(f"single column separates the classes (AUC {strength:.3f})"); score += 1.0 + 4 * (strength - 0.8)
        elif task == "regression" and pd.api.types.is_numeric_dtype(s):
            corr = pd.concat([s, y], axis=1).dropna().corr(method="spearman").iloc[0, 1]
            if pd.notna(corr):
                strength = abs(float(corr))
                if strength >= 0.90:
                    signals.append(f"single column tracks the target (|Spearman| {strength:.3f})"); score += 1.0 + 4 * (strength - 0.9)
        if col.lower() in known:
            signals.append(f"known DCLab leakage precedent {known[col.lower()]}"); score += 3.0
        rows.append({"column": str(col), "risk_score": round(score, 3), "signals": signals,
                     "single_column_strength": None if strength is None else round(strength, 4), "unique_ratio": round(float(uniq), 4)})
    if task == "regression" and pd.api.types.is_numeric_dtype(y):
        target = y.to_numpy(dtype=float)
        tol = 1e-6 * (np.abs(target).max() + 1)
        # as floats: yes/no columns count as numeric here and numpy refuses to subtract booleans; nullable columns keep NaN
        arrays = {c: X[c].to_numpy(dtype=float, na_value=np.nan) for c in numeric_cols}
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1:]:
                for op, values in (("+", arrays[a] + arrays[b]), ("-", arrays[a] - arrays[b]), ("-", arrays[b] - arrays[a])):
                    match = float((np.abs(values - target) <= tol).mean())
                    if match >= 0.99:
                        for row in rows:
                            if row["column"] in (a, b):
                                row["signals"].append(f"target = {a} {op} {b} on {match:.0%} of rows"); row["risk_score"] += 3.0
                        break
    rows.sort(key=lambda r: (-r["risk_score"], r["column"]))
    flagged = [r for r in rows if r["risk_score"] >= 1.0]
    return {
        "source": source, "task": task, "rows": int(len(X)), "blind": blind,
        "flagged": flagged, "all_columns": rows,
        "note": "Signals start a semantic review; only the time at which a value exists relative to the decision confirms leakage (DCLAB-R05).",
    }


TOOLS: dict[str, tuple[Callable[..., Any], dict[str, Any]]] = {
    "search_evidence": (search_evidence, {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language question or keywords."},
            "type": {"type": "string", "enum": ["rule", "workflow", "dataset", "experiment", "leakage_precedent", "pitfall", "finding"]},
            "dataset": {"type": "string", "description": "Dataset key, e.g. bank_marketing, hyperack, telco_churn."},
            "stage": {"type": "string", "enum": STAGE_ORDER},
            "task_type": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 20, "default": 5},
        },
        "required": ["query"],
    }),
    "get_record": (get_record, {"type": "object", "properties": {"record_id": {"type": "string"}}, "required": ["record_id"]}),
    "get_rules": (get_rules, {"type": "object", "properties": {"category": {"type": "string"}}}),
    "plan_next_stage": (plan_next_stage, {
        "type": "object",
        "properties": {"dataset": {"type": "string"}, "completed_stages": {"type": "array", "items": {"type": "string", "enum": STAGE_ORDER}}},
    }),
    "review_notebook": (review_notebook, {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
    "review_code": (review_code, {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]}),
    "audit_columns": (audit_columns, {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "CSV or parquet file containing features and the target."},
            "target": {"type": "string"},
            "task": {"type": "string", "enum": ["auto", "classification", "regression"], "default": "auto"},
            "blind": {"type": "boolean", "default": False, "description": "Ignore known DCLab precedents (for verification)."},
            "max_rows": {"type": "integer", "default": 20000},
        },
        "required": ["path", "target"],
    }),
}


def tool_schemas(style: str = "openai") -> list[dict[str, Any]]:
    """Tool definitions for function calling. ``style`` is ``openai`` or ``anthropic``."""
    out = []
    for name, (fn, schema) in TOOLS.items():
        description = (fn.__doc__ or "").strip().split("\n\n")[0].replace("\n", " ")
        if style == "anthropic":
            out.append({"name": name, "description": description, "input_schema": schema})
        else:
            out.append({"type": "function", "function": {"name": name, "description": description, "parameters": schema}})
    return out


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> Any:
    if name not in TOOLS:
        return {"error": f"Unknown tool {name!r}. Available: {', '.join(TOOLS)}"}
    fn, schema = TOOLS[name]
    arguments = arguments or {}
    allowed = set(schema.get("properties", {}))
    unknown = set(arguments) - allowed
    if unknown:
        return {"error": f"Unknown argument(s) for {name}: {', '.join(sorted(unknown))}"}
    missing = [r for r in schema.get("required", []) if r not in arguments]
    if missing:
        return {"error": f"Missing required argument(s) for {name}: {', '.join(missing)}"}
    return fn(**arguments)


# --------------------------------------------------------------------------- auditor verification


def _verification_cases() -> list[dict[str, Any]]:
    import pandas as pd

    cases = []
    for key, known in (("bank_marketing", ["duration"]),
                       ("online_shoppers", ["PageValues", "Administrative", "Administrative_Duration", "Informational",
                                            "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
                                            "BounceRates", "ExitRates"])):
        base = ROOT / "data/public" / key
        if (base / "X.parquet").exists():
            cases.append({"dataset": key, "X": pd.read_parquet(base / "X.parquet"),
                          "y": pd.read_parquet(base / "y.parquet").iloc[:, 0], "known": known, "task": "classification"})
    hyper = ROOT / "data" / "project" / "hyperack" / "hyper_ackt-dataset.csv"
    if hyper.exists():
        df = pd.read_csv(hyper)
        cases.append({"dataset": "hyperack", "X": df.drop(columns=["hyper_ack"]), "y": df["hyper_ack"],
                      "known": ["final_customer_fare", "final_biker_fare"], "task": "classification"})
    telco = ROOT / "data" / "project" / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if telco.exists():
        df = pd.read_csv(telco)
        cases.append({"dataset": "telco_churn", "X": df.drop(columns=["Churn"]), "y": df["Churn"],
                      "known": ["customerID"], "task": "classification"})
    try:
        from dclab_rnd.expansion import datasets as expansion
    except Exception:
        expansion = None
    if expansion is not None:
        for key, spec in expansion.SPECS.items():
            if not spec.blocked_features or not expansion.raw_path(ROOT, spec).exists():
                continue  # only datasets with confirmed leaks that are already downloaded (never fetch here)
            bundle = expansion.load_bundle(ROOT, key)
            X, y = bundle.X_train, bundle.y_train
            if not all(c in X.columns for c in spec.blocked_features):
                continue
            task = "regression" if spec.task_type.endswith("regression") else "classification"
            cases.append({"dataset": key, "X": X, "y": y, "known": list(spec.blocked_features), "task": task})
    return cases


def verify_auditor(write: bool = True) -> dict[str, Any]:
    cases = []
    for case in _verification_cases():
        X = case["X"]
        if len(X) > 20000:
            X = X.sample(20000, random_state=42)
        y = case["y"].loc[X.index]
        report = _audit_frame(X, y, task=case["task"], blind=True, source=case["dataset"])
        flagged = [r["column"] for r in report["flagged"]]
        known = case["known"]
        found = [c for c in known if c in flagged]
        ranks = {c: next((i + 1 for i, r in enumerate(report["all_columns"]) if r["column"] == c), None) for c in known}
        cases.append({
            "dataset": case["dataset"], "known_leaks": known, "found": found, "missed": [c for c in known if c not in flagged],
            "recall": round(len(found) / len(known), 4) if known else None,
            "flagged_total": len(flagged), "false_alarms": [c for c in flagged if c not in known],
            "rank_of_known": ranks,
            "top_flags": [{k: r[k] for k in ("column", "risk_score", "signals")} for r in report["flagged"][:6]],
        })
    total_known = sum(len(c["known_leaks"]) for c in cases)
    total_found = sum(len(c["found"]) for c in cases)
    summary = {
        "campaign_id": "agent_verification_v1",
        "experiment_id": "VER-001",
        "kind": "auditor_blind_replay",
        "question": "Without being told, does the deterministic column auditor flag the leakage columns the R&D already confirmed?",
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": len(cases),
        "known_leaks": total_known,
        "found": total_found,
        "recall": round(total_found / total_known, 4) if total_known else None,
        "datasets_with_every_leak_found": sum(not c["missed"] for c in cases),
        "datasets_with_at_least_one_leak_found": sum(bool(c["found"]) for c in cases),
        "cases": cases,
    }
    if write:
        out = ROOT / "evidence/campaigns" / "agent_verification_v1"
        (out / "results").mkdir(parents=True, exist_ok=True)
        (out / "results" / "VER-001_auditor_blind_replay.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        lines = ["# Agent verification: blind leakage-auditor replay", "",
                 summary["question"], "",
                 f"The auditor ran with `blind=True` (no catalog, no precedents) on {summary['datasets']} datasets with "
                 f"{summary['known_leaks']} confirmed leakage columns. It flagged {summary['found']} of them "
                 f"(recall {summary['recall']:.0%}).", "",
                 "| Dataset | Known leaks | Found | Missed | Other flags |", "|---|---|---|---|---|"]
        for c in cases:
            lines.append(f"| {c['dataset']} | {len(c['known_leaks'])} | {', '.join(c['found']) or '—'} | "
                         f"{', '.join(c['missed']) or '—'} | {', '.join(c['false_alarms'][:5]) or '—'} |")
        lines += ["", "## How to read this", "",
                  "- A **miss** is a leak whose numbers look ordinary. Only the decision-time contract (when the value is written) can catch it, "
                  "which is why the agent must ask for that contract and never treat a clean heuristic scan as proof of safety.",
                  "- An **other flag** is a review request, not an accusation. A strong legitimate predictor is flagged too (DCLAB-R05).",
                  "- This replay is the regression test for the auditor: any change to its heuristics must not lower recall.", "",
                  "Re-run: `python -m dclab_rnd.tools verify-auditor`.", ""]
        (out / "VERIFICATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- MCP + CLI


def serve_mcp() -> None:  # pragma: no cover - requires the optional `mcp` package
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise SystemExit("The MCP server needs the optional package: pip install mcp") from exc
    server = FastMCP("dclab-rnd-evidence")
    for name, (fn, _) in TOOLS.items():
        server.tool(name=name)(fn)
    server.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.tools", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print tool names and descriptions")
    schemas = sub.add_parser("schemas", help="print tool schemas for function calling")
    schemas.add_argument("--style", choices=["openai", "anthropic"], default="openai")
    call = sub.add_parser("call", help="call one tool with JSON arguments")
    call.add_argument("name")
    call.add_argument("arguments", nargs="?", default="{}")
    sub.add_parser("verify-auditor", help="blind replay of the column auditor on known leaks")
    sub.add_parser("mcp", help="serve the tools over MCP stdio (needs `pip install mcp`)")
    args = parser.parse_args(argv)
    if args.command == "list":
        for spec in tool_schemas():
            print(f"{spec['function']['name']:<18} {spec['function']['description']}")
    elif args.command == "schemas":
        print(json.dumps(tool_schemas(args.style), indent=2))
    elif args.command == "call":
        print(json.dumps(call_tool(args.name, json.loads(args.arguments)), indent=2, ensure_ascii=False, default=str))
    elif args.command == "verify-auditor":
        summary = verify_auditor()
        print(json.dumps({k: summary[k] for k in ("datasets", "known_leaks", "found", "recall")}, indent=2))
        for c in summary["cases"]:
            print(f"  {c['dataset']:<28} found {c['found']}  missed {c['missed']}  other flags {c['false_alarms'][:4]}")
    elif args.command == "mcp":
        serve_mcp()
    return 0


if __name__ == "__main__":
    sys.exit(main())
