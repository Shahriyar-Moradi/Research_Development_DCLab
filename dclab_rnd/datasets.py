"""Training and evaluation sets from real use, built for a workspace (package 19.12): ``make datasets``.

For each workspace, what its consent allows (``consent.py``) goes through the builders of ``dclab_core.datasets``:
the decision sets (one per point, the sealed partition in a file of its own), Home conversations, preference pairs,
agent trajectories and workflow examples under **train**; flagged sessions as candidate benchmark cases under
**improve**. Nothing is built for a use the workspace has not switched on, and a project (or the draft it came from)
takes part only when its own switch is on too. The development records (``devlog.py``) are the owner's own and are
listed beside them.

Each set is written to ``<workspace>/datasets/<builder>/<name>/``: ``rows.jsonl`` (train and dev), ``sealed.jsonl``
(a decision set's sealed partition, never used for tuning) and ``manifest.json`` (a ``DatasetManifest``: every source
by id, the counts, the consent basis). A row that still holds an e-mail address or an absolute path is refused and
counted. A reviewer accepts a candidate case in ``datasets/failures/reviews.json`` (``{id: {"verdict": "accept",
"by": ...}}``); the accepted ones are written to ``accepted.jsonl`` for the next benchmark version.

    python -m dclab_rnd.datasets [--home DIR]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any
from dclab_core import consent as core

FOLDER = "datasets"
REFUSED = (re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"), re.compile(r"(?<![\w.])/(?:Users|home|private|var|tmp)/[^\s\"']+"))
LIMIT = 10_000_000


def _clean(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    kept, refused = [], 0
    for row in rows:
        text = json.dumps(row, ensure_ascii=False, default=str)
        if any(p.search(text) for p in REFUSED):
            refused += 1
        else:
            kept.append(row)
    return kept, refused


def _write(root: Path, builder: str, name: str, rows: list[dict[str, Any]], manifest: Any) -> dict[str, Any]:
    rows, refused = _clean(rows)
    folder = root / builder / name
    folder.mkdir(parents=True, exist_ok=True)
    sealed = [r for r in rows if r.get("split") == "sealed"]
    open_rows = [r for r in rows if r.get("split") != "sealed"]
    for file, items in (("rows.jsonl", open_rows), ("sealed.jsonl", sealed)):
        path = folder / file
        if items:
            path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in items), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    if refused:  # the manifest of what was kept: rows, splits and sources recounted
        from dclab_core.datasets import refit

        manifest = refit(manifest, rows, f"{refused} rows refused: an e-mail address or a path")
    (folder / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return {"builder": builder, "name": name, "rows": len(rows), "sealed": len(sealed), "refused": refused, "folder": str(folder), "kept": rows}


def anonymous_card(card: dict[str, Any] | None) -> dict[str, Any] | None:
    """A data card as it may leave: its tables named ``table_1.csv``, ``labels``… (a file's name can be a person's),
    and every string through the export's redaction."""
    from . import consent

    if not card:
        return card
    names = {t.get("name"): (t.get("name") if t.get("name") == "labels" else f"table_{i + 1}{Path(str(t.get('name'))).suffix[:10]}")
             for i, t in enumerate(card.get("tables") or [])}

    from dclab_core.datacard import _short

    def rename(text: Any) -> Any:
        if not isinstance(text, str):
            return text
        for old, new in sorted(names.items(), key=lambda kv: -len(str(kv[0]))):
            if old:  # the name in full, and as a statement shortens a long one
                text = text.replace(str(old), new).replace(_short(old), new)
        return text

    out = {**card, "name": "tables", "tables": [{**t, "name": names.get(t.get("name"), "table")} for t in card.get("tables") or []],
           "relations": [{**r, "table": names.get(r.get("table"), r.get("table")), "other_table": names.get(r.get("other_table"), r.get("other_table"))}
                         for r in card.get("relations") or []],
           "problems": [{**pr, "tables": [names.get(t, rename(t)) for t in pr.get("tables") or []], "statement": rename(pr.get("statement"))}
                        for pr in card.get("problems") or []]}
    return consent._scrub(out, {})


def _workflow_items(services: Any, allowed_projects: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    from . import consent
    from .draft.card import card_of

    items = []
    for draft in services.drafts.list(limit=100_000):
        project_id = draft.get("project_id")
        if not project_id or project_id not in allowed_projects:
            continue
        project = allowed_projects[project_id]
        try:
            card = card_of(draft).model_dump(mode="json")
            records = services.projects.records(project_id)
        except Exception:  # noqa: BLE001 - a draft the card cannot read yet: left out
            continue
        card = consent._scrub(card, {})  # its facts quote the person's words: redacted as every exported text
        pipeline = []
        for stage, record in (records or {}).items():
            outputs = ((record or {}).get("run") or {}).get("outputs") or {}
            pipeline.append({"stage": stage, "status": (record or {}).get("status"), "metric": outputs.get("metric"), "value": outputs.get("value"),
                             "chosen": outputs.get("chosen"), "run": ((record or {}).get("run") or {}).get("run")})
        solution = project.get("solution") or {}
        items.append({"draft": draft.get("id"), "project": project_id, "problem_card": card, "data_card": anonymous_card(draft.get("data_card")),
                      "workflow": consent._scrub(draft.get("workflow"), {}),  # drawn from the person's words, or a model's
                      "solution": {k: solution.get(k) for k in ("target", "task", "kind", "time_column", "group_column", "metric")} if solution else None,
                      "forbidden": [f.get("column") if isinstance(f, dict) else f for f in solution.get("forbidden") or []] if solution else None,
                      "pipeline": pipeline})
    return items


def _sessions(services: Any, exported: list[dict[str, Any]], drafts: dict[str, dict[str, Any]], switches: dict[str, bool],
              projects: dict[str, dict[str, Any]], draft_map: dict[str, str]) -> list[dict[str, Any]]:
    """Each visit with an exported event, its flags (``signals``, on the visit's events the consent lets out: a visit
    can touch a project that did not consent) and what the export says of it."""
    from . import consent, experience

    visits = sorted({(e.get("ids") or {}).get("visit") for e in exported} - {None})
    by_visit: dict[str, list[dict[str, Any]]] = {}
    for e in exported:
        by_visit.setdefault((e.get("ids") or {}).get("visit"), []).append(e)
    raw = experience.visit_events(services.experience, visits) if visits else {}
    out = []
    for visit in visits:
        allowed = [e for e in raw.get(visit) or []
                   if consent._allowed_for(consent._project_of(e.ids.project, e.ids.conversation, draft_map), "improve", switches, projects)
                   and core.exportable(e.consent, "improve")]  # 23.1: an event written at operate shapes no signal
        flags = [f.model_dump() for f in experience.flags(allowed)]
        mine = by_visit.get(visit) or []
        created = next((e for e in mine if e["kind"] == "draft.created"), None)
        conversation = (created or {}).get("ids", {}).get("conversation")
        draft = drafts.get(conversation or "") or {}
        card = None
        try:
            from .draft.card import card_of

            card = card_of(draft).family.value if draft else None
        except Exception:  # noqa: BLE001
            pass
        answers: dict[str, str] = {}  # 21.5: what the person answered to each question (exported, so redacted), to replay the session
        asked = None
        for e in sorted(mine, key=lambda e: (e.get("at") or "", e.get("seq") or 0)):
            if e["kind"] == "chat.agent" and (e.get("payload") or {}).get("question_field"):
                asked = e["payload"]["question_field"]
            elif e["kind"] == "chat.person" and asked and (e.get("payload") or {}).get("text"):
                answers.setdefault(str(asked), str(e["payload"]["text"])[:1000])
                asked = None
        out.append({"visit": visit, "flags": flags, "problem": (created or {}).get("payload", {}).get("problem"), "family": card, "answers": answers,
                    "outcome": "built" if any(e["kind"] == "draft.built" for e in mine) else "open", "data_card": anonymous_card(draft.get("data_card")),
                    "draft": conversation if draft else None})
    return out


def build(services: Any, root: Path | None = None) -> dict[str, Any]:
    """Every set this workspace's consent allows, written under its datasets folder; what was built, and what not. A
    set this build does not write (its use switched off, its project withdrawn or deleted, its point gone) is removed:
    the folder holds what the consent allows today, never what it allowed once."""
    import shutil
    import sys

    from dclab_core import datasets as builders
    from dclab_core.ledger import POINTS

    from . import consent

    home = Path(services.store.home)
    root = Path(root) if root is not None else home / FOLDER
    switches = consent.workspace_consent(home)
    workspace = getattr(services.experience, "workspace", None)
    out: dict[str, Any] = {"workspace": workspace, "consent": switches, "built": [], "not_built": []}
    projects = {p["id"]: p for p in services.projects.list()}
    all_drafts = {d["id"]: d for d in services.drafts.list(limit=100_000) if d.get("id")}
    draft_map = {i: str(d["project_id"]) for i, d in all_drafts.items() if d.get("project_id")}
    written: set[Path] = set()

    def keep(builder: str, name: str, make: Any) -> list[dict[str, Any]]:
        try:
            rows, manifest = make()
        except Exception as exc:  # noqa: BLE001 - one set that cannot be built never costs the others
            out["not_built"].append({"use": builder, "why": f"{name}: {type(exc).__name__}"})
            print(f"datasets: {builder}/{name} not built: {type(exc).__name__}: {exc}"[:400], file=sys.stderr, flush=True)
            return []
        done = _write(root, builder, name, rows, manifest)
        written.add((root / builder / name).resolve())
        out["built"].append({k: v for k, v in done.items() if k != "kept"})
        return done["kept"]

    if switches.get("train"):
        basis = "train: the workspace's switch and each project's own"
        events = consent.export_events(services.experience.read(limit=LIMIT), "train", switches, projects, draft_map)
        records = consent.export_decisions(services.ledger.records(), services.ledger.labels(), switches, projects, draft_map)
        for point in sorted({d["point"] for d in records if d.get("point") in POINTS}):
            keep("decisions", point, lambda point=point: builders.decisions(records, point, workspace=workspace, consent=basis, drafts=draft_map))
        allowed = {i: p for i, p in projects.items() if consent.allowed(switches, p, "train")}
        keep("conversations", "problem_understanding", lambda: builders.conversations(events, workspace=workspace, consent=basis, drafts=draft_map))
        keep("preferences", "corrections", lambda: builders.preferences(records, workspace=workspace, consent=basis, drafts=draft_map))
        keep("trajectories", "agent_runs", lambda: builders.trajectories(events, workspace=workspace, consent=basis, drafts=draft_map))
        keep("workflows", "problem_to_pipeline", lambda: builders.workflows(_workflow_items(services, allowed), workspace=workspace, consent=basis))
    else:
        out["not_built"].append({"use": "train", "why": "the workspace's switch is off"})
    if switches.get("improve"):
        basis = "improve: the workspace's switch and each project's own"
        events = consent.export_events(services.experience.read(limit=LIMIT), "improve", switches, projects, draft_map)
        try:
            reviews = json.loads((root / "failures" / "reviews.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            reviews = {}
        kept = keep("failures", "benchmark_candidates", lambda: builders.failures(_sessions(services, events, all_drafts, switches, projects, draft_map),
                                                                                     reviews, workspace=workspace, consent=basis))
        accepted = [r for r in kept if r.get("review") == "accept"]
        path = root / "failures" / "benchmark_candidates" / "accepted.jsonl"
        if accepted:
            path.write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in accepted), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    else:
        out["not_built"].append({"use": "improve", "why": "the workspace's switch is off"})
    for folder in sorted(root.glob("*/*")) if root.exists() else []:  # what this build did not write goes
        if folder.is_dir() and folder.parent.name != "development" and folder.resolve() not in written:
            shutil.rmtree(folder, ignore_errors=True)
            out.setdefault("removed", []).append(folder.relative_to(root).as_posix())
    return out


def development(root: Path) -> dict[str, Any]:
    """The development records (``devlog.py``): the owner's own, listed with a manifest of their own."""
    from dclab_core import datasets as builders

    from . import devlog

    trajectories = devlog.load()
    rows = [{"package": t.package, "split": "train"} for t in trajectories]
    manifest = builders.manifest("development", "development.records", rows, [("package", t.package) for t in trajectories], use=None,
                                 consent="the owner's own development records", policy="As devlog.py splits them: whole packages, by hash.",
                                 limits=["The examples themselves are research/llm-fine-tuning/experiments/sft/out_dev (make sft-dev)."])
    folder = Path(root) / "development"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(manifest.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return {"builder": "development", "name": "records", "rows": len(rows), "sealed": 0, "refused": 0, "folder": str(folder)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.datasets", description=__doc__.split("\n")[0])
    parser.add_argument("--home", type=Path, default=None, help="the workspace folder (default: DCLAB_AGENT_HOME)")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    from .agentic.pool import Pool
    from .settings import ROOT, Settings

    load_dotenv(ROOT / ".env", override=False)
    pool = Pool(Settings.load(args.home))
    for services in [pool.default] + [pool.get(w) for w in (pool.workspace_ids()[1:] if pool.settings.auth != "none" else [])]:
        done = build(services)
        print(f"workspace {done['workspace']}: consent {done['consent']}")
        for item in done["built"]:
            print(f"  {item['builder']:<14} {item['name']:<36} {item['rows']:>7} rows ({item['sealed']} sealed, {item['refused']} refused)")
        for item in done["not_built"]:
            print(f"  not built for {item['use']}: {item['why']}")
    dev = development(Path(pool.default.store.home) / FOLDER)
    print(f"  development    records                              {dev['rows']:>7} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
