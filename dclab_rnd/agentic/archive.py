"""Persist every Research Studio artifact as on-disk files.

Keeps the existing SQLite ledger + trial-* worker outputs, and also materializes
every process step into a clean, named folder tree so nothing lives only in the DB.

Layout per run (canonical path stays agent_runs/<run_id>/ for API compatibility):

  <run_id>/
    README.md
    LABEL.txt                 human-readable short name
    run_config.json
    trajectory.json
    manifest.json
    process/                  numbered copies of each research stage
      01_profiles.json
      02_agenda.json
      03_data_review.json
      04_proposals/
      05_trials_index.jsonl
      06_critiques/
      07_synthesis.json
      99_failures.jsonl
    events/                   every event as 00001_<kind>.json
    llm_transcripts/          llm_request payloads (review before sharing)
    snapshots/                latest convenient named phase outputs
    trial-001/                deterministic worker outputs (+ META.json)
    trial-002/
    ...

Studio home also gets:
  README.md
  STUDIO_ARCHIVE_INDEX.json / .md
  catalog/by_{project,dataset,status}/<label>__<shortid> -> ../../<run_id>
  studio_assets/              UI screenshots and other studio files
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROCESS_FILES = {
    "profiles": "01_profiles.json",
    "agenda": "02_agenda.json",
    "data_review": "03_data_review.json",
    "synthesis": "07_synthesis.json",
}

SNAPSHOT_KINDS = {
    "profiles": "profiles.json",
    "agenda": "agenda.json",
    "data_review": "data_review.json",
    "synthesis": "synthesis.json",
}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def _slug(text: str, *, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", (text or "run").strip().lower()).strip("-")
    return (cleaned or "run")[:max_len]


def run_label(run: dict[str, Any]) -> str:
    """Stable human label: YYYYMMDD_<project>_<datasets>_<shortid>."""
    cfg = run.get("config") or {}
    created = (run.get("created") or _now())[:10].replace("-", "")
    project = _slug(str(cfg.get("project") or "general"), max_len=20)
    datasets = _slug("-".join(cfg.get("datasets") or ["unknown"]), max_len=40)
    short = (run.get("id") or "unknown")[:8]
    return f"{created}_{project}_{datasets}_{short}"


def run_dir(store, run_id: str) -> Path:
    path = store.home / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_run_config(store, run_id: str) -> Path:
    run = store.get(run_id)
    label = run_label(run)
    root = run_dir(store, run_id)
    (root / "LABEL.txt").write_text(label + "\n", encoding="utf-8")
    path = root / "run_config.json"
    _write_json(
        path,
        {
            "schema_version": 2,
            "run_id": run_id,
            "label": label,
            "created": run.get("created"),
            "updated": run.get("updated"),
            "status": run.get("status"),
            "phase": run.get("phase"),
            "config": run.get("config"),
            "llm_calls": run.get("llm_calls"),
            "usage": run.get("usage"),
            "active_seconds": run.get("active_seconds"),
            "error": run.get("error"),
        },
    )
    return path


def write_trial_meta(directory: Path, trial: dict[str, Any]) -> Path:
    """Sidecar metadata next to worker outputs (does not change worker filenames)."""
    plan = trial.get("plan") or {}
    result = trial.get("result") or {}
    metrics = result.get("metrics") or {}
    meta = {
        "schema_version": 1,
        "trial_id": trial.get("id"),
        "status": trial.get("status"),
        "title": plan.get("title"),
        "dataset": plan.get("dataset"),
        "model": plan.get("model"),
        "hypothesis": plan.get("hypothesis"),
        "metrics": metrics,
        "paired_comparison": trial.get("paired_comparison"),
        "error": trial.get("error"),
        "artifact_directory": trial.get("artifact_directory") or str(directory),
        "slug": _slug(f"{plan.get('model', 'model')}-{plan.get('title', 'trial')}", max_len=60),
    }
    path = directory / "META.json"
    _write_json(path, meta)
    return path


def append_event_file(store, run_id: str, seq: int, kind: str, created: str, payload: Any) -> Path:
    """Write one event as a numbered JSON file under events/ or llm_transcripts/."""
    root = run_dir(store, run_id)
    folder = "llm_transcripts" if kind == "llm_request" else "events"
    name = f"{seq:05d}_{kind}.json"
    path = root / folder / name
    _write_json(
        path,
        {
            "seq": seq,
            "run_id": run_id,
            "kind": kind,
            "created": created,
            "payload": payload,
        },
    )

    process = root / "process"
    process.mkdir(parents=True, exist_ok=True)

    snap_name = SNAPSHOT_KINDS.get(kind)
    if snap_name:
        _write_json(root / "snapshots" / snap_name, payload)
    process_name = PROCESS_FILES.get(kind)
    if process_name:
        _write_json(process / process_name, payload)

    if kind == "proposal":
        for rel in (
            root / "snapshots" / "proposals.jsonl",
            process / "04_proposals" / f"{seq:05d}_proposal.json",
        ):
            if rel.suffix == ".jsonl":
                rel.parent.mkdir(parents=True, exist_ok=True)
                with rel.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"seq": seq, "created": created, **payload}, ensure_ascii=False)
                        + "\n"
                    )
            else:
                _write_json(rel, {"seq": seq, "created": created, **payload})

    if kind == "critique":
        for rel in (
            root / "snapshots" / "critiques.jsonl",
            process / "06_critiques" / f"{seq:05d}_critique.json",
        ):
            if rel.suffix == ".jsonl":
                rel.parent.mkdir(parents=True, exist_ok=True)
                with rel.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps({"seq": seq, "created": created, **payload}, ensure_ascii=False)
                        + "\n"
                    )
            else:
                _write_json(rel, {"seq": seq, "created": created, **payload})

    if kind == "trial":
        compact = {
            "seq": seq,
            "created": created,
            "id": payload.get("id"),
            "status": payload.get("status"),
            "plan": payload.get("plan"),
            "error": payload.get("error"),
            "artifact_directory": payload.get("artifact_directory"),
            "metrics": (payload.get("result") or {}).get("metrics"),
            "paired_comparison": payload.get("paired_comparison"),
        }
        for rel in (
            root / "snapshots" / "trials_index.jsonl",
            process / "05_trials_index.jsonl",
        ):
            rel.parent.mkdir(parents=True, exist_ok=True)
            with rel.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(compact, ensure_ascii=False, default=str) + "\n")
        # Prefer writing META beside worker artifacts when the trial folder exists.
        artifact = payload.get("artifact_directory")
        if artifact:
            trial_path = Path(artifact)
            if trial_path.exists():
                write_trial_meta(trial_path, payload)
        else:
            # Fallback: infer trial-00N from evidence id suffix.
            evidence_id = str(payload.get("id") or "")
            if ":trial-" in evidence_id:
                trial_name = evidence_id.split(":")[-1]
                trial_path = root / trial_name
                if trial_path.exists():
                    write_trial_meta(trial_path, payload)

    if kind == "failure":
        fail_path = process / "99_failures.jsonl"
        fail_path.parent.mkdir(parents=True, exist_ok=True)
        with fail_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"seq": seq, "created": created, **payload}, ensure_ascii=False, default=str)
                + "\n"
            )

    return path


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_studio_run_report(store, run_id: str) -> Path:
    """Write a human-readable report that mirrors the Studio UI text for one run."""
    root = run_dir(store, run_id)
    run = store.get(run_id)
    cfg = run.get("config") or {}
    label = run_label(run)
    events = store.events(run_id)
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(event["kind"], []).append(event)

    reports = root / "reports"
    if reports.exists():
        shutil.rmtree(reports)
    reports.mkdir(parents=True, exist_ok=True)

    agenda = (by_kind.get("agenda") or [{}])[-1].get("payload") or {}
    data_review = (by_kind.get("data_review") or [{}])[-1].get("payload") or {}
    synthesis = (by_kind.get("synthesis") or [{}])[-1].get("payload") or {}
    profiles = (by_kind.get("profiles") or [{}])[-1].get("payload") or []
    critiques = [e["payload"] for e in by_kind.get("critique", [])]
    proposals = [e["payload"] for e in by_kind.get("proposal", [])]
    trials = [e["payload"] for e in by_kind.get("trial", [])]
    phases = [e for e in by_kind.get("phase", [])]
    completed = [t for t in trials if t.get("status") == "completed"]
    failed = [t for t in trials if t.get("status") != "completed"]
    latest_auc = None
    if completed:
        latest = completed[-1]
        latest_auc = ((latest.get("result") or {}).get("metrics") or {}).get("roc_auc")

    # --- split readable text files ---
    sections: dict[str, str] = {}

    header_lines = [
        f"# {(cfg.get('project') or 'general').replace('_', ' ').title()} · RESEARCH IN MOTION",
        "",
        f"**Datasets:** {', '.join(cfg.get('datasets') or [])}",
        f"**Status:** {run.get('status')} · **Phase:** {run.get('phase')}",
        f"**LLM:** {cfg.get('model')} via NOOA + OpenAI Responses · "
        f"{cfg.get('repeats')} × 3-fold group CV · {(cfg.get('max_rows') or 0):,} row cap",
        f"**Run ID:** `{run_id}`",
        f"**Label:** `{label}`",
        "",
        "## Research goal",
        "",
        cfg.get("goal") or "—",
        "",
    ]
    sections["00_header.md"] = "\n".join(header_lines)

    if agenda:
        sections["01_agenda.md"] = "\n".join(
            [
                "# Research agenda",
                "",
                "## Goal interpretation",
                "",
                agenda.get("goal_interpretation") or "—",
                "",
                "## Research questions",
                "",
                *[f"- {q}" for q in agenda.get("research_questions") or ["—"]],
                "",
                "## Sequence",
                "",
                *[f"- {q}" for q in agenda.get("sequence") or ["—"]],
                "",
                "## Success criteria",
                "",
                *[f"- {q}" for q in agenda.get("success_criteria") or ["—"]],
                "",
                "## Limitations",
                "",
                *[f"- {q}" for q in agenda.get("limitations") or ["—"]],
                "",
            ]
        )

    if data_review:
        sections["02_data_review.md"] = "\n".join(
            [
                "# Data & leakage review",
                "",
                "## Dataset findings",
                "",
                *[f"- {q}" for q in data_review.get("dataset_findings") or ["—"]],
                "",
                "## Leakage risks",
                "",
                *[f"- {q}" for q in data_review.get("leakage_risks") or ["—"]],
                "",
                "## Availability assumptions",
                "",
                *[f"- {q}" for q in data_review.get("availability_assumptions") or ["—"]],
                "",
                "## Recommended tests",
                "",
                *[f"- {q}" for q in data_review.get("recommended_tests") or ["—"]],
                "",
                "## External-data hypotheses",
                "",
                *[f"- {q}" for q in data_review.get("external_data_hypotheses") or ["—"]],
                "",
            ]
        )

    if profiles:
        profile_bits = ["# Dataset profiles", ""]
        for profile in profiles if isinstance(profiles, list) else [profiles]:
            profile_bits.append(f"## {profile.get('dataset', 'dataset')}")
            profile_bits.append("")
            profile_bits.append("```json")
            profile_bits.append(json.dumps(profile, indent=2, ensure_ascii=False, default=str))
            profile_bits.append("```")
            profile_bits.append("")
        sections["03_profiles.md"] = "\n".join(profile_bits)

    exp_lines = [
        "# Experiment evidence",
        "",
        "Adaptive development scores. No untouched test set or production approval.",
        "",
        f"- Experiments completed: **{len(completed)}** of {cfg.get('max_experiments')} limit",
        f"- Rejected or failed: **{len(failed)}**",
        f"- Latest CV AUC: **{_fmt(latest_auc)}**",
        f"- Learned lessons: **{len(synthesis.get('lessons') or [])}**",
        f"- LLM calls: **{run.get('llm_calls')}** · tokens: "
        f"**{(run.get('usage') or {}).get('total_tokens', (run.get('usage') or {}).get('input_tokens', '—'))}**",
        "",
        "| # | Experiment | Model | CV AUC | Log loss | Paired Δ AUC |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, trial in enumerate(trials, start=1):
        plan = trial.get("plan") or {}
        metrics = (trial.get("result") or {}).get("metrics") or {}
        paired = ((trial.get("paired_comparison") or {}).get("mean_deltas") or {}).get("roc_auc")
        exp_lines.append(
            f"| {idx} | {plan.get('title', trial.get('id'))} | {plan.get('model')} | "
            f"{_fmt(metrics.get('roc_auc'))} | {_fmt(metrics.get('log_loss'))} | "
            f"{_fmt(paired) if paired is not None else '—'} |"
        )
    exp_lines.append("")

    for idx, trial in enumerate(trials, start=1):
        plan = trial.get("plan") or {}
        result = trial.get("result") or {}
        metrics = result.get("metrics") or {}
        spread = result.get("fold_standard_deviation") or {}
        features = plan.get("features") or []
        feat_txt = (
            "; ".join(
                f"{f.get('name')} = {f.get('operation')}({', '.join(f.get('inputs') or [])})"
                for f in features
            )
            if features
            else "None — baseline inputs"
        )
        sensitivity = result.get("input_sensitivity") or []
        sens_txt = ", ".join(
            f"{s.get('column')} ({_fmt(s.get('auc_drop'))} AUC drop)" for s in sensitivity[:5]
        ) or "—"
        stresses = result.get("stress_tests") or []
        stress_txt = "; ".join(
            f"{s.get('column')}: {_fmt(((s.get('metric_deltas_clean_minus_stressed') or {}).get('roc_auc')))} AUC drop"
            for s in stresses[:3]
        ) or "Not requested"
        exp_lines += [
            f"## {idx}. {plan.get('title', trial.get('id'))}",
            "",
            f"- **Status:** {trial.get('status')}",
            f"- **Model / dataset:** {plan.get('model')} · {plan.get('dataset')}",
            f"- **Hypothesis:** {plan.get('hypothesis')}",
            f"- **Engineered features:** {feat_txt}",
            f"- **Policy exclusions:** {', '.join(result.get('excluded_by_policy') or []) or 'None declared; availability remains an assumption'}",
            f"- **Most sensitive inputs:** {sens_txt}",
            f"- **Missing-input stress:** {stress_txt}",
            f"- **Output reliability:** Brier {_fmt(metrics.get('brier'))} · Average precision {_fmt(metrics.get('average_precision'))}",
            f"- **Uncertainty:** AUC fold SD {_fmt(spread.get('roc_auc'))}; correlated folds, not a confidence interval.",
            f"- **Artifacts:** `trial-{idx:03d}/result.json`, `recipe.json`, `oof_predictions.jsonl`, `META.json`",
            "",
        ]
        if trial.get("error"):
            exp_lines += ["### Error", "", trial["error"], ""]
        sections[f"04_experiment_{idx:02d}.md"] = "\n".join(
            [
                f"# Experiment {idx}: {plan.get('title', trial.get('id'))}",
                "",
                *exp_lines[-20:],
            ]
        )
    sections["04_experiments.md"] = "\n".join(exp_lines)

    if critiques:
        crit_lines = ["# Critiques", ""]
        for idx, critique in enumerate(critiques, start=1):
            block = [
                f"## Critique {idx}",
                "",
                "### Observation",
                "",
                critique.get("observation") or "—",
                "",
                "### Interpretation",
                "",
                critique.get("interpretation") or "—",
                "",
                "### Next question",
                "",
                critique.get("next_question") or "—",
                "",
                f"**Continue research:** {critique.get('continue_research')}",
                "",
                "### Limitations & counterevidence",
                "",
                *[f"- {item}" for item in critique.get("limitations") or ["—"]],
                "",
                f"**Evidence IDs:** {', '.join(critique.get('evidence_ids') or [])}",
                "",
            ]
            crit_lines.extend(block)
            sections[f"05_critique_{idx:02d}.md"] = "\n".join(
                [f"# Critique {idx}", ""] + block[2:]
            )
        sections["05_critiques.md"] = "\n".join(crit_lines)
        # Convenience: latest critique alone (matches UI "Latest scientific critique")
        latest = critiques[-1]
        sections["05_latest_critique.md"] = "\n".join(
            [
                "# Latest scientific critique",
                "",
                latest.get("observation") or "—",
                "",
                latest.get("interpretation") or "—",
                "",
                "## Next question",
                "",
                latest.get("next_question") or "—",
                "",
                "## Limitations & counterevidence",
                "",
                *[f"- {item}" for item in latest.get("limitations") or ["—"]],
                "",
            ]
        )

    if synthesis:
        syn_lines = [
            "# Research synthesis",
            "",
            synthesis.get("summary") or "—",
            "",
            "## Learned lessons",
            "",
        ]
        for idx, lesson in enumerate(synthesis.get("lessons") or [], start=1):
            syn_lines += [
                f"### Lesson {idx}",
                "",
                lesson.get("claim") or "—",
                "",
                f"**Scope:** {lesson.get('scope')}",
                "",
                f"**Counterevidence:** {lesson.get('counterevidence')}",
                "",
                f"**Follow-up:** {lesson.get('follow_up')}",
                "",
                f"**Evidence IDs:** {', '.join(lesson.get('evidence_ids') or [])}",
                "",
            ]
        syn_lines += [
            "## Theoretical principles",
            "",
            *[f"- {item}" for item in synthesis.get("theoretical_principles") or ["—"]],
            "",
            "## Reusable workflow blocks",
            "",
            *[f"- {item}" for item in synthesis.get("workflow_blocks") or ["—"]],
            "",
            "## Still unanswered",
            "",
            *[f"- {item}" for item in synthesis.get("unanswered_questions") or ["—"]],
            "",
        ]
        sections["06_synthesis.md"] = "\n".join(syn_lines)

    if proposals:
        prop_lines = ["# Experiment proposals", ""]
        for idx, proposal in enumerate(proposals, start=1):
            prop_lines += [
                f"## Proposal {idx}: {proposal.get('title')}",
                "",
                f"- Model: {proposal.get('model')}",
                f"- Dataset: {proposal.get('dataset')}",
                f"- Hypothesis: {proposal.get('hypothesis')}",
                f"- Expected learning: {proposal.get('expected_learning')}",
                "",
            ]
        sections["07_proposals.md"] = "\n".join(prop_lines)

    activity = ["# Agent activity", ""]
    for event in phases:
        activity.append(
            f"- **{event['payload'].get('name')}** · {event.get('created')}"
        )
    activity.append("")
    sections["08_agent_activity.md"] = "\n".join(activity)

    # Master report concatenating UI-facing sections
    master = [
        sections["00_header.md"],
        "## Summary metrics",
        "",
        f"- Research complete / status: **{run.get('status')}**",
        f"- Experiments: **{len(completed)}** of {cfg.get('max_experiments')} limit",
        f"- Rejected or failed: **{len(failed)}**",
        f"- Latest CV AUC: **{_fmt(latest_auc)}**",
        f"- Learned lessons: **{len(synthesis.get('lessons') or [])}** (provisional, evidence-linked)",
        f"- LLM calls: **{run.get('llm_calls')}**",
        f"- Usage: `{json.dumps(run.get('usage') or {}, ensure_ascii=False)}`",
        "",
        "---",
        "",
        sections.get("04_experiments.md", ""),
        "",
        "---",
        "",
        sections.get("05_latest_critique.md", sections.get("05_critiques.md", "")),
        "",
        "---",
        "",
        sections.get("06_synthesis.md", ""),
        "",
        "---",
        "",
        sections.get("02_data_review.md", ""),
        "",
        "---",
        "",
        sections.get("01_agenda.md", ""),
        "",
        "---",
        "",
        sections.get("08_agent_activity.md", ""),
        "",
        "---",
        "",
        "## Machine-readable sources",
        "",
        "- `trajectory.json`",
        "- `process/`",
        "- `events/`",
        "- `llm_transcripts/`",
        "- `trial-*/`",
        "- `manifest.json`",
        "",
    ]
    master_path = reports / "STUDIO_RUN_REPORT.md"
    master_path.write_text("\n".join(master), encoding="utf-8")

    for name, text in sections.items():
        (reports / name).write_text(text, encoding="utf-8")

    index_lines = [
        "# Reports index",
        "",
        f"Run `{run_id}` · label `{label}`",
        "",
        "These markdown files store **all generated Studio text** in a clean structure.",
        "",
    ]
    for path in sorted(reports.glob("*.md")):
        index_lines.append(f"- [`{path.name}`]({path.name})")
    index_lines.append("")
    (reports / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    return master_path


def _list_files(root: Path) -> list[dict[str, Any]]:
    files = []
    if not root.exists():
        return files
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "manifest.json":
            continue
        rel = path.relative_to(root).as_posix()
        files.append({"path": rel, "bytes": path.stat().st_size, "suffix": path.suffix})
    return files


def write_manifest(store, run_id: str) -> Path:
    root = run_dir(store, run_id)
    run = store.get(run_id)
    label = run_label(run)
    events = store.events(run_id)
    trials = sorted(
        [
            {
                "trial": p.name,
                "files": [c.name for c in sorted(p.iterdir()) if c.is_file()],
                "meta": json.loads((p / "META.json").read_text(encoding="utf-8"))
                if (p / "META.json").exists()
                else None,
            }
            for p in root.glob("trial-*")
            if p.is_dir()
        ],
        key=lambda item: item["trial"],
    )
    files = _list_files(root)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "label": label,
        "archived_at": _now(),
        "status": run.get("status"),
        "phase": run.get("phase"),
        "project": (run.get("config") or {}).get("project"),
        "datasets": (run.get("config") or {}).get("datasets"),
        "model": (run.get("config") or {}).get("model"),
        "event_count": len(events),
        "event_kinds": sorted({e["kind"] for e in events}),
        "llm_request_count": sum(1 for e in events if e["kind"] == "llm_request"),
        "trials": trials,
        "file_count": len(files),
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "layout": {
            "run_config.json": "Launch settings and status",
            "trajectory.json": "Reviewable decision trail (llm_request omitted)",
            "process/": "Numbered stage outputs from the research loop",
            "events/": "Every studio step payload",
            "llm_transcripts/": "LLM request payloads — review before sharing",
            "snapshots/": "Latest convenient named phase copies",
            "trial-*/": "Worker metrics, recipe, OOF predictions, META.json",
            "reports/": "Human-readable Studio texts (full UI narrative as Markdown)",
            "manifest.json": "Inventory of all archived files",
            "LABEL.txt": "Human-readable run label",
            "README.md": "This run's guide",
        },
    }
    path = root / "manifest.json"
    _write_json(path, manifest)
    readme = root / "README.md"
    readme.write_text(
        "\n".join(
            [
                f"# Research run `{run_id}`",
                "",
                f"**Label:** `{label}`",
                "",
                f"- Status: **{run.get('status')}**",
                f"- Phase: {run.get('phase')}",
                f"- Project: {(run.get('config') or {}).get('project')}",
                f"- Datasets: {', '.join((run.get('config') or {}).get('datasets') or [])}",
                f"- LLM: {(run.get('config') or {}).get('model')}",
                f"- Events: {len(events)}",
                f"- Files archived: {len(files)}",
                "",
                "## Layout",
                "",
                "- `run_config.json` — launch settings",
                "- `LABEL.txt` — human-readable name",
                "- `trajectory.json` — reviewable decision trail",
                "- `process/` — numbered stage outputs (profiles → synthesis)",
                "- `events/` — every process step as JSON",
                "- `llm_transcripts/` — LLM request payloads",
                "- `snapshots/` — latest named phase outputs",
                "- `trial-*/` — executed experiment metrics, recipes, META.json",
                "- `reports/STUDIO_RUN_REPORT.md` — **full UI text** (critiques, lessons, experiments)",
                "- `reports/*.md` — split readable sections",
                "- `manifest.json` — inventory of all files",
                "",
                "SQLite ledgers at the studio home (`research.sqlite3`, `checkpoints.sqlite3`)",
                "remain the live index; this folder is the durable file archive.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _reset_derived(root: Path) -> None:
    for name in ("snapshots", "events", "llm_transcripts", "process"):
        path = root / name
        if path.exists():
            shutil.rmtree(path)


def archive_run(store, run_id: str, *, reset_snapshots: bool = True) -> dict[str, Any]:
    """Rebuild the full on-disk archive for one run from the SQLite ledger + trial dirs."""
    root = run_dir(store, run_id)
    write_run_config(store, run_id)

    if reset_snapshots:
        _reset_derived(root)

    for event in store.events(run_id):
        append_event_file(
            store,
            run_id,
            event["seq"],
            event["kind"],
            event["created"],
            event["payload"],
        )

    # Ensure META exists for any trial folders even if trial events were sparse.
    for trial_path in sorted(root.glob("trial-*")):
        if not trial_path.is_dir():
            continue
        if (trial_path / "META.json").exists():
            continue
        result_path = trial_path / "result.json"
        if not result_path.exists():
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        write_trial_meta(
            trial_path,
            {
                "id": f"{run_id}:{trial_path.name}",
                "status": "completed",
                "plan": result.get("plan"),
                "result": result,
                "artifact_directory": str(trial_path),
            },
        )

    trajectory = store.export(run_id)
    _write_json(root / "trajectory.json", trajectory)
    write_studio_run_report(store, run_id)
    manifest_path = write_manifest(store, run_id)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _safe_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        # Fallback when symlinks are unavailable: write a pointer file.
        link.write_text(f"{target}\n", encoding="utf-8")


def write_catalog_links(store) -> Path:
    """Human-navigable catalog of symlinks into canonical run folders."""
    catalog = store.home / "catalog"
    if catalog.exists():
        shutil.rmtree(catalog)
    catalog.mkdir(parents=True, exist_ok=True)

    for run in store.list():
        run_id = run["id"]
        root = store.home / run_id
        if not root.exists():
            continue
        label = run_label(run)
        cfg = run.get("config") or {}
        project = _slug(str(cfg.get("project") or "general"), max_len=24)
        status = _slug(str(run.get("status") or "unknown"), max_len=24)
        datasets = cfg.get("datasets") or ["unknown"]
        rel_target = Path("../..") / run_id

        _safe_symlink(catalog / "by_run_id" / run_id, Path("..") / ".." / run_id)
        _safe_symlink(catalog / "by_label" / label, rel_target)
        _safe_symlink(catalog / "by_project" / project / label, Path("../../..") / run_id)
        _safe_symlink(catalog / "by_status" / status / label, Path("../../..") / run_id)
        for dataset in datasets:
            ds = _slug(str(dataset), max_len=40)
            _safe_symlink(catalog / "by_dataset" / ds / label, Path("../../..") / run_id)

    (catalog / "README.md").write_text(
        "\n".join(
            [
                "# Research Studio catalog",
                "",
                "Symlinks into canonical `agent_runs/<run_id>/` folders.",
                "",
                "- `by_label/` — date_project_datasets_shortid",
                "- `by_project/` — general / hyperack / telco_churn",
                "- `by_dataset/` — adult, bank_marketing, …",
                "- `by_status/` — completed, failed, paused, …",
                "- `by_run_id/` — raw UUID lookup",
                "",
                "Do not delete these links instead of the real run folder.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return catalog


def organize_studio_assets(store) -> Path:
    """Move loose studio screenshots into studio_assets/ (keeps originals if move fails)."""
    assets = store.home / "studio_assets"
    assets.mkdir(parents=True, exist_ok=True)
    for pattern in ("ui-*.png", "ui-*.jpg", "*.screenshot.*"):
        for path in store.home.glob(pattern):
            if not path.is_file():
                continue
            dest = assets / path.name
            if dest.exists():
                continue
            try:
                shutil.move(str(path), str(dest))
            except OSError:
                shutil.copy2(path, dest)
    (assets / "README.md").write_text(
        "# Studio assets\n\nUI screenshots and other non-run Research Studio files.\n",
        encoding="utf-8",
    )
    return assets


def write_studio_readme(store) -> Path:
    path = store.home / "README.md"
    path.write_text(
        "\n".join(
            [
                "# DCLab Research Studio — local archive",
                "",
                "This directory stores **every** Research Studio research run.",
                "",
                "## Always present (live index)",
                "",
                "- `research.sqlite3` — run index + append-only event ledger",
                "- `checkpoints.sqlite3` — LangGraph durable checkpoints (pause/resume)",
                "",
                "## Per-run folders",
                "",
                "Canonical path: `<run_id>/` (API and worker use this).",
                "Each folder contains config, trajectory, process stage files, events,",
                "LLM transcripts, snapshots, and `trial-*/` worker artifacts.",
                "",
                "## Human navigation",
                "",
                "- `STUDIO_ARCHIVE_INDEX.json` / `.md` — inventory of all runs",
                "- `catalog/` — symlinks by label, project, dataset, status",
                "- `studio_assets/` — UI screenshots and studio files",
                "",
                "## Refresh archives",
                "",
                "```bash",
                ".venv-agent/bin/python -m dclab_rnd.agentic archive --all",
                "```",
                "",
                "Existing storage methods are kept. Archiving only **adds** readable files.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_studio_index(store) -> Path:
    """Studio-wide index of every archived research run."""
    organize_studio_assets(store)
    write_studio_readme(store)
    write_catalog_links(store)

    runs = []
    for run in store.list():
        root = store.home / run["id"]
        manifest_path = root / "manifest.json"
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        )
        trials = sorted(p.name for p in root.glob("trial-*") if p.is_dir())
        label = run_label(run)
        runs.append(
            {
                "run_id": run["id"],
                "label": label,
                "status": run.get("status"),
                "created": run.get("created"),
                "updated": run.get("updated"),
                "project": (run.get("config") or {}).get("project"),
                "datasets": (run.get("config") or {}).get("datasets"),
                "model": (run.get("config") or {}).get("model"),
                "goal": ((run.get("config") or {}).get("goal") or "")[:180],
                "llm_calls": run.get("llm_calls"),
                "trials": trials,
                "file_count": manifest.get("file_count"),
                "total_bytes": manifest.get("total_bytes"),
                "path": run["id"],
                "catalog_label_path": f"catalog/by_label/{label}",
            }
        )
    index = {
        "schema_version": 2,
        "studio": "DCLab Research Studio",
        "home": str(store.home),
        "generated_at": _now(),
        "run_count": len(runs),
        "runs": runs,
    }
    path = store.home / "STUDIO_ARCHIVE_INDEX.json"
    _write_json(path, index)
    md = store.home / "STUDIO_ARCHIVE_INDEX.md"
    lines = [
        "# DCLab Research Studio — archive index",
        "",
        f"Generated: `{index['generated_at']}`  ",
        f"Home: `{store.home}`  ",
        f"Runs: **{len(runs)}**",
        "",
        "| Label | Status | Project | Datasets | Model | Trials | Files |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for item in runs:
        lines.append(
            f"| `{item['label']}` | {item['status']} | {item.get('project') or '—'} | "
            f"{', '.join(item.get('datasets') or [])} | {item.get('model')} | "
            f"{len(item.get('trials') or [])} | {item.get('file_count') or 0} |"
        )
    lines += [
        "",
        "Canonical folders: `agent_runs/<run_id>/`  ",
        "Friendly links: `agent_runs/catalog/by_label/`",
        "",
        "Re-archive everything:",
        "",
        "```bash",
        ".venv-agent/bin/python -m dclab_rnd.agentic archive --all",
        "```",
        "",
    ]
    md.write_text("\n".join(lines), encoding="utf-8")
    return path


def archive_all(store) -> list[dict[str, Any]]:
    results = []
    for run in store.list():
        results.append(archive_run(store, run["id"], reset_snapshots=True))
    write_studio_index(store)
    return results
