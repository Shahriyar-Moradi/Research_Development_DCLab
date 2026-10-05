#!/usr/bin/env python3
"""The research map: one structured record per research idea, from the files in the repository.

Each track record answers: the idea · its champion(s) · its experiments · its notebooks ·
its evaluation · its reports and research notes · related evidence and related tracks.
The same records feed two views, so they can never disagree:

- research/<track>/INDEX.md and the track graph in research/README.md (``render_all``), and
- the Research map page of the Research Studio (``research_map`` behind ``/api/research``).

    python -m dclab_rnd.research_map           # write every INDEX.md   (make research-index)
    python -m dclab_rnd.research_map --check   # exit 1 if any is stale (make rd-check, CI)
    python -m dclab_rnd.research_map --json    # print the map as JSON

Standard library only. Files are listed with `git ls-files`, so the output is the
same on every machine and in CI (untracked local files never appear).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
RESEARCH = ROOT / "research"
EXP50 = "evidence/campaigns/model_building_50_v1"
EXPANSION = "evidence/campaigns/expansion_v1"
PITFALLS = "evidence/campaigns/pitfalls_v1"
VERIFY = "evidence/campaigns/agent_verification_v1"
CODES = "evidence/campaigns/category_codes_v1"

# Evidence that lives outside a track folder but belongs to it.
RELATED = {
    "tabular-classification": {"campaigns": [EXP50, PITFALLS, CODES], "tracks": ["churn-prediction", "tabular-foundation-models", "ml-methodology"]},
    "churn-prediction": {"campaigns": [PITFALLS], "tracks": ["tabular-classification", "tabular-foundation-models"]},
    "tabular-foundation-models": {"campaigns": [], "tracks": ["tabular-classification", "churn-prediction"]},
    "llm-fine-tuning": {"campaigns": [EXP50, EXPANSION], "tracks": ["workflow-model", "agentic-ml-copilot", "evaluation-and-trust"]},
    "ml-methodology": {"campaigns": [EXP50, EXPANSION, PITFALLS, VERIFY, CODES], "tracks": ["tabular-classification", "evaluation-and-trust"]},
    "agentic-ml-copilot": {"campaigns": [PITFALLS, VERIFY], "tracks": ["evaluation-and-trust", "llm-fine-tuning", "workflow-actions"]},
    "evaluation-and-trust": {"campaigns": [VERIFY, PITFALLS], "tracks": ["agentic-ml-copilot", "ml-methodology"]},
    "anomaly-and-fraud-detection": {"campaigns": [EXPANSION, PITFALLS], "tracks": ["tabular-classification"]},
    "time-series-forecasting": {"campaigns": [EXPANSION], "tracks": ["mlops-and-deployment"]},
    "nlp-and-text": {"campaigns": [EXPANSION], "tracks": ["llm-fine-tuning"]},
    "graph-neural-networks": {"campaigns": [], "tracks": ["temporal-gnn"]},
    "temporal-gnn": {"campaigns": [], "tracks": ["graph-neural-networks", "vision-scene-graphs"]},
    "vision-scene-graphs": {"campaigns": [], "tracks": ["computer-vision", "temporal-gnn", "driving-maps"]},
    "computer-vision": {"campaigns": [], "tracks": ["vision-scene-graphs"]},
    "driving-maps": {"campaigns": [], "tracks": ["vision-scene-graphs", "temporal-gnn"]},
    "workflow-model": {"campaigns": [], "tracks": ["llm-fine-tuning", "workflow-actions"]},
    "workflow-actions": {"campaigns": [], "tracks": ["workflow-model", "agentic-ml-copilot"]},
    "cross-industry-workflows": {"campaigns": [], "tracks": ["workflow-model"]},
    "data-science-foundations": {"campaigns": [EXP50], "tracks": ["tabular-classification", "churn-prediction"]},
    "mlops-and-deployment": {"campaigns": [], "tracks": ["time-series-forecasting", "evaluation-and-trust"]},
    "recommender-systems": {"campaigns": [], "tracks": []},
    "causal-inference-and-experimentation": {"campaigns": [], "tracks": []},
}
EXPANSION_TRACK = {"credit_card_fraud": "anomaly-and-fraud-detection", "bike_sharing_daily": "time-series-forecasting",
                   "ecommerce_clothing_reviews": "nlp-and-text", "letter_recognition": "ml-methodology"}
METRIC = {"roc_auc": "ROC-AUC", "average_precision": "average precision", "macro_f1": "macro-F1", "mae": "MAE"}
THEMES = {
    "Tabular modeling": ["tabular-classification", "churn-prediction", "tabular-foundation-models", "data-science-foundations"],
    "Methodology and trust": ["ml-methodology", "evaluation-and-trust", "mlops-and-deployment", "causal-inference-and-experimentation"],
    "Agents and language models": ["agentic-ml-copilot", "llm-fine-tuning", "workflow-model", "workflow-actions", "cross-industry-workflows"],
    "Other data types": ["time-series-forecasting", "nlp-and-text", "anomaly-and-fraud-detection", "recommender-systems"],
    "Graphs and vision": ["graph-neural-networks", "temporal-gnn", "computer-vision", "vision-scene-graphs", "driving-maps"],
}


# --------------------------------------------------------------------------- helpers


@lru_cache(maxsize=1)
def _tracked_files() -> tuple[str, ...]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return tuple(sorted(f for f in out.splitlines() if f))
    except (OSError, subprocess.CalledProcessError):
        return tuple(sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts))


def tracked_files(fresh: bool = False) -> list[str]:
    if fresh:
        _tracked_files.cache_clear()
    return list(_tracked_files())


def read_json(rel: str):
    path = ROOT / rel
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def link(text: str, target: str, base: str) -> str:
    return f"[{text}]({os.path.relpath(target, base)})"


def readme_field(track_dir: Path) -> tuple[str, str, str]:
    text = (track_dir / "README.md").read_text(encoding="utf-8") if (track_dir / "README.md").exists() else ""
    title = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), track_dir.name)
    status = ""
    m = re.search(r"\*\*Status:?\*\*:?\s*(.+)", text)
    if m:
        status = m.group(1).strip()
    question = ""
    m = re.search(r"## Research question\s+(.+?)(?:\n\n|\Z)", text, re.S)
    if m:
        question = " ".join(m.group(1).split())
    elif text:
        paras = [p for p in text.split("\n\n") if p.strip() and not p.startswith("#")]
        question = " ".join(re.sub(r"\*\*[^*]+\*\*", "", paras[0]).split()) if paras else ""
    return title, status, question


def fmt(v, digits=4):
    return f"{v:,.1f}" if isinstance(v, (int, float)) and abs(v) >= 100 else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


# --------------------------------------------------------------------------- champions


def holdout_row(result: dict, rel: str) -> dict:
    ev = result.get("evidence", {})
    metric = result.get("primary_metric", "roc_auc")
    hold = ev.get("holdout_metrics", {}) or {}
    ci = ev.get("holdout_primary_metric_ci") or ev.get("holdout_roc_auc_bootstrap_ci") or {}
    value = hold.get(metric)
    interval = f"{fmt(ci.get('low'))}–{fmt(ci.get('high'))}" if ci.get("low") is not None else ""
    model = f"{ev.get('model', result.get('model_name', '?'))} ({ev.get('selected_optimization', 'baseline')})"
    return {"dataset": result.get("dataset"), "model": model, "metric": METRIC.get(metric, metric),
            "value": fmt(value) if value is not None else "?", "interval": interval, "source": rel}


def champions(track: str) -> tuple[list[dict], str]:
    """Rows and a one-line definition of what 'champion' means for this track."""
    rows: list[dict] = []
    if track == "tabular-classification":
        evidence = read_json("evidence/knowledge/evidence.json") or {}
        for c in sorted(evidence.get("champions", []), key=lambda c: c["dataset"]):
            rows.append({"dataset": c["dataset"], "model": c["model"], "metric": "ROC-AUC", "value": fmt(c["roc_auc"]),
                         "interval": "", "source": c["source_path"]})
        return rows, ("Best deployment-eligible result per dataset from the evidence registry "
                      "(leakage-unsafe runs are excluded). Rebuilt by `make rd-sync`.")
    if track == "churn-prediction":
        summary = read_json("research/churn-prediction/experiments/churn_exp/results/summary.json") or {}
        best = summary.get("best") or {}
        if best:
            m = best.get("metrics", {})
            rows.append({"dataset": "telco_churn", "model": f"{best.get('id')} {best.get('title')} ({best.get('model')})",
                         "metric": "ROC-AUC", "value": fmt(m.get("roc_auc")), "interval": f"AP {fmt(m.get('average_precision'))}",
                         "source": "research/churn-prediction/experiments/churn_exp/results/summary.json"})
        path = ROOT / "research/churn-prediction/experiments/tabular_transformers/outputs/tabular_transformer_metrics.csv"
        if path.exists():
            best_tt = max(csv.DictReader(path.read_text(encoding="utf-8").splitlines()), key=lambda r: float(r["roc_auc"]))
            rows.append({"dataset": "telco_churn (deep-model comparison)", "model": best_tt["model"], "metric": "ROC-AUC",
                         "value": fmt(float(best_tt["roc_auc"])), "interval": "single holdout",
                         "source": path.relative_to(ROOT).as_posix()})
        return rows, "Best result of each churn study; development CV, not independent confirmation."
    if track == "tabular-foundation-models":
        path = ROOT / "research/tabular-foundation-models/experiments/tabpfn/results/benchmark.csv"
        if path.exists():
            best: dict[str, dict] = {}
            for r in csv.DictReader(path.read_text(encoding="utf-8").splitlines()):
                if r.get("roc_auc") and (r["dataset"] not in best or float(r["roc_auc"]) > float(best[r["dataset"]]["roc_auc"])):
                    best[r["dataset"]] = r
            for ds, r in sorted(best.items()):
                warn = " ⚠ includes leaky final fares" if "hyper" in ds.lower() else ""
                rows.append({"dataset": ds + warn, "model": r["model"], "metric": "ROC-AUC", "value": fmt(float(r["roc_auc"])),
                             "interval": "", "source": path.relative_to(ROOT).as_posix()})
        return rows, "Best model per dataset in the TabPFN benchmark."
    if track == "ml-methodology":
        for path in sorted((ROOT / EXP50 / "results").glob("EXP-*_optimization_reliability.json")):
            rows.append(holdout_row(json.loads(path.read_text()), path.relative_to(ROOT).as_posix()))
        return rows, ("Each dataset's preselected recipe scored once on its locked holdout in the 50-experiment campaign "
                      "(95% bootstrap interval).")
    datasets = [ds for ds, t in EXPANSION_TRACK.items() if t == track]
    for path in sorted((ROOT / EXPANSION / "results").glob("EXP-*_optimization_reliability.json")):
        result = json.loads(path.read_text())
        if result.get("dataset") in datasets:
            rows.append(holdout_row(result, path.relative_to(ROOT).as_posix()))
    if rows:
        return rows, "Seed result from the task-type expansion campaign: one locked holdout, used once (95% interval)."
    return [], ""


# --------------------------------------------------------------------------- track records


def theme_of(track: str) -> str:
    return next((theme for theme, members in THEMES.items() if track in members), "Other ideas")


def _group(paths: list[str], base: str, threshold: int) -> list[dict]:
    """Group files by folder; a suite with more than `threshold` files across sub-folders collapses into one group."""
    by_folder: dict[str, list[str]] = defaultdict(list)
    for f in paths:
        parts = os.path.relpath(f, base).split("/")
        suite = "/".join(parts[:2]) if parts[0] == "experiments" else parts[0]
        group = [n for n in paths if os.path.relpath(n, base).startswith(suite + "/")]
        key = f"{base}/{suite}" if len(group) > threshold and len({str(Path(n).parent) for n in group}) > 1 else str(Path(f).parent)
        by_folder[key].append(f)
    return [{"label": os.path.relpath(folder, base) + "/", "path": folder, "files": fs,
             "folders": len({str(Path(f).parent) for f in fs})} for folder, fs in sorted(by_folder.items())]


def track_data(track_dir: Path, files: list[str]) -> dict:
    """Everything known about one research idea, as plain JSON-ready data."""
    track = track_dir.name
    base = f"research/{track}"
    mine = [f for f in files if f.startswith(base + "/")]
    title, status, question = readme_field(track_dir)
    rel = RELATED.get(track, {"campaigns": [], "tracks": []})
    rows, definition = champions(track)

    exp_root = f"{base}/experiments/"
    groups: dict[str, list[str]] = defaultdict(list)
    for f in mine:
        if f.startswith(exp_root):
            groups[f[len(exp_root):].split("/")[0]].append(f)
    experiments = []
    for name, fs in sorted(groups.items()):
        entry = next((f for f in fs if f == f"{exp_root}{name}/README.md"), None) or next(
            (f for f in fs if f.endswith(".md") and f.count("/") == exp_root.count("/") + 1), f"{exp_root}{name}")
        subprojects = sorted({f[len(exp_root + name) + 1:].split("/")[0] for f in fs
                              if f[len(exp_root + name) + 1:].split("/")[0].endswith("_exp")})
        experiments.append({"name": name, "path": exp_root + name, "entry": entry, "files": len(fs),
                            "results": sum(1 for f in fs if "/results/" in f and f.endswith((".json", ".csv"))),
                            "notebooks": sum(1 for f in fs if f.endswith(".ipynb")),
                            "scripts": sum(1 for f in fs if f.endswith(".py")),
                            "subprojects": [{"name": s, "path": f"{exp_root}{name}/{s}"} for s in subprojects]})

    campaigns = []
    for c in rel["campaigns"]:
        report = next((f for f in files if f.startswith(c + "/") and f.count("/") == c.count("/") + 1
                       and f.endswith(".md") and "REPORT" in f.upper()), f"{c}/README.md")
        campaigns.append({"name": Path(c).name, "path": c, "report": report,
                          "results": sum(1 for f in files if f.startswith(c + "/results/") and f.endswith(".json"))})

    eval_files = [f for f in mine if f.startswith(f"{base}/evaluation/")]
    by_eval: dict[str, list[str]] = defaultdict(list)
    for f in eval_files:
        by_eval[f[len(base + "/evaluation/"):].split("/")[0]].append(f)
    evaluation = [{"name": sub, "path": f"{base}/evaluation/{sub}", "files": fs} for sub, fs in sorted(by_eval.items())]
    shared_evaluation = []
    for c in rel["campaigns"]:
        if c in (VERIFY, PITFALLS):
            report = next((f for f in files if f.startswith(c + "/") and f.endswith(".md") and "REPORT" in f.upper()), None)
            if report:
                shared_evaluation.append(report)

    docs = [f for f in mine if f.endswith((".md", ".pdf")) and f not in (f"{base}/README.md", f"{base}/INDEX.md")
            and (f.startswith(f"{base}/reports/") or re.search(r"(REPORT|BENCHMARK|ROADMAP|PLAN|GUIDE|SPEC|MODEL)", Path(f).name.upper()))]
    notebooks = _group([f for f in mine if f.endswith(".ipynb")], base, 12)
    reports = _group(docs, base, 12)
    return {
        "name": track, "title": title, "status": status, "idea": question, "theme": theme_of(track),
        "path": base, "readme": f"{base}/README.md", "index": f"{base}/INDEX.md",
        "champion": {"definition": definition, "rows": rows},
        "experiments": experiments, "campaigns": campaigns,
        "notebooks": notebooks, "evaluation": evaluation, "shared_evaluation": shared_evaluation,
        "reports": reports, "related": [t for t in rel["tracks"]],
        "counts": {"champions": len(rows), "experiments": len(experiments), "campaigns": len(campaigns),
                   "notebooks": sum(len(g["files"]) for g in notebooks), "evaluation": len(eval_files) + len(shared_evaluation),
                   "reports": len(docs), "files": len(mine)},
    }


def track_dirs() -> list[Path]:
    return sorted(p for p in RESEARCH.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))


def research_map(fresh: bool = True) -> dict:
    """The whole map: themes → tracks, each track's record, and the relation edges between tracks."""
    files = tracked_files(fresh)
    tracks = {t.name: track_data(t, files) for t in track_dirs()}
    themes = [{"name": theme, "tracks": [m for m in members if m in tracks]} for theme, members in THEMES.items()]
    loose = [t for t in tracks if not any(t in th["tracks"] for th in themes)]
    if loose:
        themes.append({"name": "Other ideas", "tracks": loose})
    edges = sorted({tuple(sorted((t, r))) for t in tracks for r in RELATED.get(t, {}).get("tracks", []) if r in tracks})
    totals = {k: sum(t["counts"][k] for t in tracks.values()) for k in ("experiments", "notebooks", "champions", "reports")}
    totals["tracks"] = len(tracks)
    totals["campaigns"] = len({c["path"] for t in tracks.values() for c in t["campaigns"]})
    return {"themes": themes, "tracks": tracks, "edges": [list(e) for e in edges], "totals": totals,
            "layout": ["README.md", "INDEX.md", "experiments/", "notebooks/", "evaluation/", "reports/"]}


# --------------------------------------------------------------------------- read-only file preview for the UI

PREVIEW_ROOTS = ("research/", "evidence/campaigns/", "docs/")
PREVIEW_SUFFIXES = {".md", ".txt", ".json", ".jsonl", ".csv", ".py", ".ipynb", ".yaml", ".yml", ".toml"}
PREVIEW_LIMIT = 200_000


def preview(rel: str) -> dict:
    """Text of one tracked file (or the listing of a tracked folder) under research/, evidence/campaigns/ or docs/.

    Raises ValueError for anything else: absolute paths, `..`, untracked or binary files.
    """
    rel = (rel or "").strip().strip("/")
    if not rel or rel.startswith("/") or ".." in rel.split("/") or not rel.startswith(PREVIEW_ROOTS):
        raise ValueError("Only files under research/, evidence/campaigns/ or docs/ can be previewed")
    files = tracked_files()
    if rel not in files:
        inside = [f for f in files if f.startswith(rel + "/")]
        if not inside:
            raise ValueError("Not a tracked file or folder")
        entries = sorted({rel + "/" + f[len(rel) + 1:].split("/")[0] + ("/" if "/" in f[len(rel) + 1:] else "") for f in inside})
        return {"path": rel, "kind": "folder", "entries": entries[:500], "files": len(inside), "truncated": len(entries) > 500}
    path = (ROOT / rel).resolve()
    if ROOT.resolve() not in path.parents or Path(rel).suffix.lower() not in PREVIEW_SUFFIXES:
        raise ValueError("This file type has no text preview")
    text = path.read_text(encoding="utf-8", errors="replace")
    kind = Path(rel).suffix.lower().lstrip(".")
    if kind == "ipynb":
        try:
            cells = json.loads(text).get("cells", [])
            text = "\n\n".join(f"── {c.get('cell_type', 'cell')} cell {i + 1} ──\n" + "".join(c.get("source", []))
                               for i, c in enumerate(cells))
        except (json.JSONDecodeError, AttributeError):
            pass
    return {"path": rel, "kind": kind, "text": text[:PREVIEW_LIMIT], "truncated": len(text) > PREVIEW_LIMIT}


# --------------------------------------------------------------------------- INDEX.md


def render_index(t: dict) -> str:
    base = t["path"]
    lines = [f"# {t['title']} · index", "",
             "> Generated by `make research-index` from the files in this folder and the evidence registry. "
             "Edit `README.md`, not this file.", "",
             f"**Idea.** {t['idea'] or 'See README.md.'}", "",
             f"**Status.** {t['status'] or 'see README.md'} · **Read first:** [README.md](README.md)", ""]

    lines += ["## Champion", ""]
    if t["champion"]["rows"]:
        lines += [t["champion"]["definition"], "", "| Dataset | Champion | Metric | Score | Interval / note | Source |", "|---|---|---|---:|---|---|"]
        for r in t["champion"]["rows"]:
            lines.append(f"| {r['dataset']} | {r['model']} | {r['metric']} | {r['value']} | {r['interval']} | "
                         f"{link(Path(r['source']).name, r['source'], base)} |")
    else:
        lines.append("No champion yet. The first measured result recorded in this track becomes the baseline to beat.")
    lines.append("")

    lines += ["## Experiments", ""]
    if t["experiments"]:
        lines += ["| Experiment | Result files | Notebooks | Scripts | Entry point |", "|---|---:|---:|---:|---|"]
        for e in t["experiments"]:
            lines.append(f"| {link(e['name'], e['path'], base)} | {e['results']} | {e['notebooks']} | {e['scripts']} | "
                         f"{link(Path(e['entry']).name, e['entry'], base)} |")
            if e["subprojects"]:
                lines.append(f"| ↳ {len(e['subprojects'])} sub-projects: " + ", ".join(
                    link(s["name"], s["path"], base) for s in e["subprojects"]) + " | | | | |")
    else:
        lines.append("No experiments in this folder yet.")
    if t["campaigns"]:
        lines += ["", "**Related evidence campaigns** (shared across tracks, in `evidence/campaigns/`):", ""]
        for c in t["campaigns"]:
            lines.append(f"- {link(c['name'], c['path'], base)}: {c['results']} result files · "
                         f"{link(Path(c['report']).name, c['report'], base)}")
    lines.append("")

    lines += ["## Notebooks", ""]
    for g in t["notebooks"]:
        if len(g["files"]) > 12:
            where = f" across {g['folders']} folders" if g["folders"] > 1 else ""
            lines.append(f"- {link(g['label'], g['path'], base)}: {len(g['files'])} notebooks{where}")
        else:
            lines.append(f"- {link(g['label'], g['path'], base)}: " + ", ".join(link(Path(f).stem, f, base) for f in g["files"]))
    if not t["notebooks"]:
        lines.append("No notebooks yet.")
    lines.append("")

    lines += ["## Evaluation", ""]
    for e in t["evaluation"]:
        lines.append(f"- {link(e['name'], e['path'], base)}: {len(e['files'])} file(s)"
                     if len(e["files"]) > 1 or e["files"][0] != e["path"] else f"- {link(e['name'], e['path'], base)}")
    for report in t["shared_evaluation"]:
        lines.append(f"- Shared evaluation evidence: {link(Path(report).name, report, base)}")
    if not t["evaluation"] and not [c for c in t["campaigns"] if c["path"] in (VERIFY, PITFALLS)]:
        lines.append("No evaluation artifacts yet.")
    lines.append("")

    lines += ["## Reports and research notes", ""]
    for g in t["reports"]:
        if len(g["files"]) > 8:
            where = f" across {g['folders']} folders (one set per sub-project)" if g["folders"] > 1 else ""
            lines.append(f"- {link(g['label'], g['path'], base)}: {len(g['files'])} documents{where}")
        else:
            lines.append(f"- {link(g['label'], g['path'], base)}: " + ", ".join(link(Path(f).name, f, base) for f in g["files"]))
    if not t["reports"]:
        lines.append("Research notes live in [README.md](README.md) until there is enough material for `reports/`.")
    lines.append("")

    if t["related"]:
        lines += ["## Related tracks", "", " · ".join(link(r, f"research/{r}/INDEX.md", base) for r in t["related"]), ""]
    return "\n".join(lines)


def build_index(track_dir: Path, files: list[str]) -> str:
    return render_index(track_data(track_dir, files))

def build_overview(tracks: list[Path]) -> str:
    names = {t.name for t in tracks}
    lines = ["```mermaid", "flowchart LR"]
    for i, (theme, members) in enumerate(THEMES.items()):
        lines.append(f'  subgraph T{i}["{theme}"]')
        for m in members:
            if m in names:
                lines.append(f'    {m.replace("-", "_")}["{m}"]')
        lines.append("  end")
    seen = set()
    for t in sorted(names):
        for r in RELATED.get(t, {}).get("tracks", []):
            edge = tuple(sorted((t, r)))
            if r in names and edge not in seen:
                seen.add(edge)
                lines.append(f'  {edge[0].replace("-", "_")} --- {edge[1].replace("-", "_")}')
    lines.append("```")
    return "\n".join(lines)


def render_all() -> dict[Path, str]:
    files = tracked_files()
    tracks = track_dirs()
    out = {t / "INDEX.md": build_index(t, files) + "\n" for t in tracks}
    readme = RESEARCH / "README.md"
    text = readme.read_text(encoding="utf-8")
    start, end = "<!-- track-graph:start -->", "<!-- track-graph:end -->"
    if start in text and end in text:
        new = text.split(start)[0] + start + "\n" + build_overview(tracks) + "\n" + end + text.split(end)[1]
        out[readme] = new
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if any generated file is stale")
    parser.add_argument("--json", action="store_true", help="print the research map as JSON")
    args = parser.parse_args(argv)
    if args.json:
        print(json.dumps(research_map(), indent=2, ensure_ascii=False))
        return 0
    outputs = render_all()
    stale = [p for p, text in outputs.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    if args.check:
        if stale:
            print("Stale research index: " + ", ".join(str(p.relative_to(ROOT)) for p in stale)
                  + ". Run: make research-index")
            return 1
        print("Research index is current.")
        return 0
    for p in stale:
        p.write_text(outputs[p], encoding="utf-8")
    print(f"Wrote {len(stale)} file(s); {len(outputs)} checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
