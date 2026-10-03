#!/usr/bin/env python3
"""Generate research/<track>/INDEX.md: one map per research idea.

Each INDEX answers, from the files actually in the repository:
    the idea · its champion(s) · its experiments · its notebooks · its evaluation ·
    its reports and research notes · related evidence and related tracks.

    python scripts/build_research_index.py          # write every INDEX.md
    python scripts/build_research_index.py --check  # exit 1 if any INDEX.md is stale (CI)

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
EXP50 = "evidence/campaigns/model_building_50_v1"
EXPANSION = "evidence/campaigns/expansion_v1"
PITFALLS = "evidence/campaigns/pitfalls_v1"
VERIFY = "evidence/campaigns/agent_verification_v1"

# Evidence that lives outside a track folder but belongs to it.
RELATED = {
    "tabular-classification": {"campaigns": [EXP50, PITFALLS], "tracks": ["churn-prediction", "tabular-foundation-models", "ml-methodology"]},
    "churn-prediction": {"campaigns": [PITFALLS], "tracks": ["tabular-classification", "tabular-foundation-models"]},
    "tabular-foundation-models": {"campaigns": [], "tracks": ["tabular-classification", "churn-prediction"]},
    "llm-fine-tuning": {"campaigns": [EXP50, EXPANSION], "tracks": ["workflow-model", "agentic-ml-copilot", "evaluation-and-trust"]},
    "ml-methodology": {"campaigns": [EXP50, EXPANSION, PITFALLS, VERIFY], "tracks": ["tabular-classification", "evaluation-and-trust"]},
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


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return sorted(f for f in out.splitlines() if f)
    except (OSError, subprocess.CalledProcessError):
        return sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts)


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
            best_tt = max(csv.DictReader(path.open()), key=lambda r: float(r["roc_auc"]))
            rows.append({"dataset": "telco_churn (deep-model comparison)", "model": best_tt["model"], "metric": "ROC-AUC",
                         "value": fmt(float(best_tt["roc_auc"])), "interval": "single holdout",
                         "source": path.relative_to(ROOT).as_posix()})
        return rows, "Best result of each churn study; development CV, not independent confirmation."
    if track == "tabular-foundation-models":
        path = ROOT / "research/tabular-foundation-models/experiments/tabpfn/results/benchmark.csv"
        if path.exists():
            best: dict[str, dict] = {}
            for r in csv.DictReader(path.open()):
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


# --------------------------------------------------------------------------- index


def build_index(track_dir: Path, files: list[str]) -> str:
    track = track_dir.name
    base = f"research/{track}"
    mine = [f for f in files if f.startswith(base + "/")]
    title, status, question = readme_field(track_dir)
    rel = RELATED.get(track, {"campaigns": [], "tracks": []})
    lines = [f"# {title} · index", "",
             "> Generated by `make research-index` from the files in this folder and the evidence registry. "
             "Edit `README.md`, not this file.", "",
             f"**Idea.** {question or 'See README.md.'}", "",
             f"**Status.** {status or 'see README.md'} · **Read first:** [README.md](README.md)", ""]

    rows, definition = champions(track)
    lines += ["## Champion", ""]
    if rows:
        lines += [definition, "", "| Dataset | Champion | Metric | Score | Interval / note | Source |", "|---|---|---|---:|---|---|"]
        for r in rows:
            lines.append(f"| {r['dataset']} | {r['model']} | {r['metric']} | {r['value']} | {r['interval']} | "
                         f"{link(Path(r['source']).name, r['source'], base)} |")
    else:
        lines.append("No champion yet. The first measured result recorded in this track becomes the baseline to beat.")
    lines.append("")

    # experiments
    exp_root = f"{base}/experiments/"
    groups: dict[str, list[str]] = defaultdict(list)
    for f in mine:
        if f.startswith(exp_root):
            groups[f[len(exp_root):].split("/")[0]].append(f)
    lines += ["## Experiments", ""]
    if groups:
        lines += ["| Experiment | Result files | Notebooks | Scripts | Entry point |", "|---|---:|---:|---:|---|"]
        for name, fs in sorted(groups.items()):
            results = sum(1 for f in fs if "/results/" in f and f.endswith((".json", ".csv")))
            notebooks = sum(1 for f in fs if f.endswith(".ipynb"))
            scripts = sum(1 for f in fs if f.endswith(".py"))
            entry = next((f for f in fs if f == f"{exp_root}{name}/README.md"), None) or next(
                (f for f in fs if f.endswith(".md") and f.count("/") == exp_root.count("/") + 1), f"{exp_root}{name}")
            lines.append(f"| {link(name, exp_root + name, base)} | {results} | {notebooks} | {scripts} | "
                         f"{link(Path(entry).name, entry, base)} |")
            subprojects = sorted({f[len(exp_root + name) + 1:].split('/')[0] for f in fs
                                  if f[len(exp_root + name) + 1:].split('/')[0].endswith('_exp')})
            if subprojects:
                lines.append(f"| ↳ {len(subprojects)} sub-projects: " + ", ".join(
                    link(s, f'{exp_root}{name}/{s}', base) for s in subprojects) + " | | | | |")
    else:
        lines.append("No experiments in this folder yet.")
    if rel["campaigns"]:
        lines += ["", "**Related evidence campaigns** (shared across tracks, in `evidence/campaigns/`):", ""]
        for c in rel["campaigns"]:
            n = sum(1 for f in files if f.startswith(c + "/results/") and f.endswith(".json"))
            report = next((f for f in files if f.startswith(c + "/") and f.count("/") == c.count("/") + 1
                           and f.endswith(".md") and "REPORT" in f.upper()), f"{c}/README.md")
            lines.append(f"- {link(Path(c).name, c, base)}: {n} result files · {link(Path(report).name, report, base)}")
    lines.append("")

    # notebooks
    notebooks = [f for f in mine if f.endswith(".ipynb")]
    lines += ["## Notebooks", ""]
    if notebooks:
        by_folder: dict[str, list[str]] = defaultdict(list)
        for f in notebooks:
            parts = os.path.relpath(f, base).split("/")
            suite = "/".join(parts[:2]) if parts[0] == "experiments" else parts[0]
            group = [n for n in notebooks if os.path.relpath(n, base).startswith(suite + "/")]
            key = f"{base}/{suite}" if len(group) > 12 and len({str(Path(n).parent) for n in group}) > 1 else str(Path(f).parent)
            by_folder[key].append(f)
        for folder, fs in sorted(by_folder.items()):
            label = os.path.relpath(folder, base)
            if len(fs) > 12:
                subs = len({str(Path(f).parent) for f in fs})
                where = f" across {subs} folders" if subs > 1 else ""
                lines.append(f"- {link(label + '/', folder, base)}: {len(fs)} notebooks{where}")
            else:
                lines.append(f"- {link(label + '/', folder, base)}: " + ", ".join(link(Path(f).stem, f, base) for f in fs))
    else:
        lines.append("No notebooks yet.")
    lines.append("")

    # evaluation
    eval_files = [f for f in mine if f.startswith(f"{base}/evaluation/")]
    lines += ["## Evaluation", ""]
    if eval_files:
        by_folder = defaultdict(list)
        for f in eval_files:
            sub = f[len(base + "/evaluation/"):].split("/")[0]
            by_folder[sub].append(f)
        for sub, fs in sorted(by_folder.items()):
            target = f"{base}/evaluation/{sub}"
            lines.append(f"- {link(sub, target, base)}: {len(fs)} file(s)" if len(fs) > 1 or fs[0] != target
                         else f"- {link(sub, target, base)}")
    extra = [c for c in rel["campaigns"] if c in (VERIFY, PITFALLS)]
    for c in extra:
        report = next((f for f in files if f.startswith(c + "/") and f.endswith(".md") and "REPORT" in f.upper()), None)
        if report:
            lines.append(f"- Shared evaluation evidence: {link(Path(report).name, report, base)}")
    if not eval_files and not extra:
        lines.append("No evaluation artifacts yet.")
    lines.append("")

    # reports and notes
    docs = [f for f in mine if f.endswith((".md", ".pdf")) and f != f"{base}/README.md" and f != f"{base}/INDEX.md"
            and (f.startswith(f"{base}/reports/") or re.search(r"(REPORT|BENCHMARK|ROADMAP|PLAN|GUIDE|SPEC|MODEL)", Path(f).name.upper()))]
    lines += ["## Reports and research notes", ""]
    if docs:
        by_folder = defaultdict(list)
        for f in docs:
            parts = os.path.relpath(f, base).split("/")
            suite = "/".join(parts[:2]) if parts[0] == "experiments" else parts[0]
            group = [d for d in docs if os.path.relpath(d, base).startswith(suite + "/")]
            folders = {str(Path(d).parent) for d in group}
            by_folder[f"{base}/{suite}" if len(group) > 12 and len(folders) > 1 else str(Path(f).parent)].append(f)
        for folder, fs in sorted(by_folder.items()):
            label = os.path.relpath(folder, base)
            if len(fs) > 8:
                subs = len({str(Path(f).parent) for f in fs})
                where = f" across {subs} folders (one set per sub-project)" if subs > 1 else ""
                lines.append(f"- {link(label + '/', folder, base)}: {len(fs)} documents{where}")
            else:
                lines.append(f"- {link(label + '/', folder, base)}: " + ", ".join(link(Path(f).name, f, base) for f in fs))
    else:
        lines.append("Research notes live in [README.md](README.md) until there is enough material for `reports/`.")
    lines.append("")

    if rel["tracks"]:
        lines += ["## Related tracks", "", " · ".join(link(t, f"research/{t}/INDEX.md", base) for t in rel["tracks"]), ""]
    return "\n".join(lines)


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
    tracks = sorted(p for p in RESEARCH.iterdir() if p.is_dir() and not p.name.startswith(("_", ".")))
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
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
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
