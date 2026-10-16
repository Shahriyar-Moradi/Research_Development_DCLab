"""The data catalog (package 23.9): what this studio records, where, in what shape, for how long and under which consent,
generated from the contracts and the code so it cannot drift.

``python -m dclab_rnd.data_catalog`` writes ``docs/DATA_CATALOG.md`` and ``docs/data_catalog.json``; ``--check`` fails
when the committed files differ from what the code generates now (``make rd-gates`` runs it), so adding an event kind, a
ledger point, an audit kind, a dataset builder or a table without describing it fails the gate. Nothing here is edited
by hand: a description lives next to what it describes (a table's comment in ``storage/models.py``) or in ``NOTES`` below.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, get_args

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

from dclab_core import catalog as core

CLASS_NOTES = core.CLASS_NOTES
LEAVES = core.LEAVES

NOTES: dict[str, str] = {
    # tables without a comment of their own in storage/models.py
    "table:workspace": "A workspace: the unit that owns every other row.",
    "table:app_user": "A person who can sign in (accounts on); the password hash is the only secret kept.",
    "table:membership": "A person's role in a workspace.",
    "table:project": "A project: its solution, settings and stage state, as one document.",
    "table:stage_record": "A stage's result record (evidence, claims, notes, its run).",
    "table:activity": "A project's log of what happened, one line per move.",
    "table:transition": "Every move the workflow validator checked, allowed or not.",
    "table:draft": "A draft before it is a project: the problem, the data analysis and the understanding.",
    "table:draft_event": "A draft's chat, pipeline and analysis events, in order.",
    "table:intern_session": "An intern session: its goal, steps, budget and final report.",
    # dataset builders (dclab_core/datasets.py)
    "builder:decisions": "One set per decision point: the state in bands, every answer, the label by source and the outcome (23.5).",
    "builder:conversations": "Home conversations that ended in a build, as chat examples; an edited reply is the reply (23.6).",
    "builder:preferences": "A person's correction of a decision, as a preferred and a rejected answer.",
    "builder:trajectories": "An agent's moves in a run: the tool, its verdict, the step; no cell value or free text.",
    "builder:workflows": "Solution workflows drawn from problem and data, with the validator's verdict.",
    "builder:failures": "Flagged sessions as candidate benchmark cases for a reviewer.",
    "builder:development": "How DCLab itself was built: one record per package, kept for the owner's SFT.",
    "builder:outcomes": "Each decision with the number that followed it, judged against its reference (23.5).",
    "builder:episodes": "One row per turn of a conversation: the message, the steps, the model calls, the moves, the decisions, the feedback and the outcomes, joined by ids (23.10); no free text under improve.",
    "builder:problem_maps": "A person's words as a problem map of parts: the map first drawn, the one built from, the person's edits as the correction, and what the programme did against its best leaf (24.7).",
    "builder:routes": "A sub-problem routed to an approach: its state, the candidates with the codes that refuted them, the one chosen and what became of it; codes and counts only (24.7).",
    "builder:twin": "What the system and Claude, as the AI-first expert, answered on the same inputs: every twin step, the preference pairs a truth decided, the expert's records with what became of them, and Claude's answers as a teacher's beside the rule's at the ledger's points, one set per use (workflow, technical); Claude's words redacted (25.7).",
    "builder:model_calls": "Each model call whose prompt and reply were kept (train only; never a call that was shown cell values), with the check's verdict and the feedback (23.10).",
    # where the files live when there is no database
    "files:experience_event": "experience/YYYY/MM/DD/<visit>.jsonl",
    "files:decision_record": "ledger/decisions/YYYY/MM/DD.jsonl",
    "files:decision_label": "ledger/labels/YYYY/MM/DD.jsonl",
    "files:decision_outcome": "ledger/outcomes/YYYY/MM/DD.jsonl",
    "files:audit_event": "audit.jsonl",
    "files:model_request": "model_requests.jsonl",
    "files:model_output_check": "model_requests_checks.jsonl",
}
BUILDER_NEEDS = {"failures": "improve", "episodes": "improve (free text only at train)", "development": "nothing (the owner's own records of how DCLab is built)"}


def _tables() -> list[dict[str, Any]]:
    from .storage.models import metadata

    source = (Path(__file__).parent / "storage" / "models.py").read_text(encoding="utf-8")
    comments = {name: (com or "").lstrip("# ").strip() for _, com, name in re.findall(r'^(\w+) = sa\.Table\(\s*(#[^\n]*)?\n\s*"(\w+)"', source, re.M)}
    out = []
    for name, table in sorted(metadata.tables.items()):
        note = comments.get(name) or NOTES.get(f"table:{name}")
        if not note:
            raise SystemExit(f"data_catalog: table {name} has no description: give it a comment in storage/models.py or a NOTES entry")
        out.append({"table": name, "what": note, "columns": [c.name for c in table.columns], "files": NOTES.get(f"files:{name}"),
                    "append_only": "append-only" in note})
    return out


# modules that read or list kinds rather than write them: the library (the contract, the adapters, the queries), the finders,
# the retention sweep, the catalog, the Admin page that shows them
READERS = ("dclab_core/", "dclab_rnd/data_catalog.py", "dclab_rnd/experience/__init__.py", "dclab_rnd/experience/finder.py", "dclab_rnd/experience/queries.py",
           "dclab_rnd/experience/retention.py", "dclab_rnd/experience/catalog.py", "dclab_rnd/experience/__main__.py", "dclab_rnd/datasets.py",
           "dclab_rnd/consent.py", "dclab_rnd/agentic/pages/sessions.py", "dclab_rnd/agentic/web/src/views/admin.html", "dclab_rnd/agentic/web/src/views/sessions.html")


def _owners(kinds: list[str]) -> dict[str, list[str]]:
    """The modules of the studio whose source names each kind in quotes, readers left out: where it is emitted (the web code
    included). A kind no module names is emitted by the host (an adapter, the browser's generic sender)."""
    sources: dict[str, str] = {}
    for path in (ROOT / "dclab_rnd").rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        if "/static/" in rel or "/__pycache__/" in rel or "/schemas/" in rel or path.suffix not in (".py", ".js", ".html") or rel.endswith("/api.js"):
            continue
        if rel.startswith(READERS):
            continue
        sources[rel] = path.read_text(encoding="utf-8", errors="ignore")
    return {k: sorted(rel for rel, text in sources.items() if f'"{k}"' in text or f"'{k}'" in text) for k in kinds}


def _kinds() -> list[dict[str, Any]]:
    from dclab_core import experience as x

    owners = _owners(sorted(x.KINDS))
    out = core.event_kinds(owners)
    for k in out:
        k["example"] = f"python -m dclab_rnd.experience search --kind {k['kind']}"
    return out


def _points() -> list[dict[str, Any]]:
    from .ledger import CAPTURED

    out = core.decision_points()
    for p in out:
        p["decided_in"] = CAPTURED.get(p["point"])  # where in the studio it is decided
    return out


def _audit() -> list[str]:
    from . import audit

    return list(audit.KINDS)


def _builders() -> list[dict[str, Any]]:
    from dclab_core import datasets

    out = []
    for name in get_args(datasets.Builder):
        note = NOTES.get(f"builder:{name}")
        if not note:
            raise SystemExit(f"data_catalog: dataset builder {name} has no description: add NOTES['builder:{name}']")
        out.append({"builder": name, "what": note, "needs": BUILDER_NEEDS.get(name, "train")})
    return out


def _contracts() -> list[dict[str, Any]]:
    return core.contracts()


def _retention() -> list[dict[str, Any]]:
    from .experience import retention

    return [{"what": "events (the session record)", "setting": "DCLAB_RETENTION_EVENT_DAYS", "default_days": retention.DEFAULT_DAYS["DCLAB_RETENTION_EVENT_DAYS"]},
            {"what": "prompts and replies inside events", "setting": "DCLAB_RETENTION_MODEL_TEXT_DAYS", "default_days": retention.DEFAULT_DAYS["DCLAB_RETENTION_MODEL_TEXT_DAYS"]},
            {"what": "a draft's raw upload", "setting": "DCLAB_RETENTION_RAW_DAYS", "default_days": None},
            {"what": "a deleted project, draft or session", "setting": "(immediate)", "default_days": 0}]


def build() -> dict[str, Any]:
    """The whole catalog as a document: sorted, with nothing that depends on the clock or the machine."""
    from dclab_core import __version__

    return {"about": "Generated by python -m dclab_rnd.data_catalog from the contracts and the code; never edited by hand.", "dclab_core": __version__,
            "classes": core.classes(),
            "event_kinds": _kinds(), "decision_points": _points(), "audit_kinds": _audit(), "dataset_builders": _builders(),
            "tables": _tables(), "contracts": _contracts(), "retention": _retention(),
            "consent": {"operate": "kept to run the product; never leaves", "improve": "may be read and turned into evaluation cases",
                        "train": "may become training data", "rule": "a record leaves only if it was written at that level (its stamp) and the switch is on now"}}


def render(doc: dict[str, Any]) -> str:
    lines = ["# Data catalog", "", f"_{doc['about']}_ dclab-core {doc['dclab_core']}.", "",
             "## Consent and retention", ""]
    lines += [f"- **{k}** — {v}" for k, v in doc["consent"].items()]
    lines += [""] + [f"- {r['what']}: `{r['setting']}`" + (f", default {r['default_days']} days" if r["default_days"] else "") for r in doc["retention"]]
    lines += ["", "## Field classes", "", "| Class | Letter | Meaning |", "|---|---|---|"]
    lines += [f"| {k} | {v['letter']} | {v['meaning']} |" for k, v in doc["classes"].items()]
    lines += ["", f"## Event kinds ({len(doc['event_kinds'])})", "", "| Kind | Fields (class) | Never leaves | Only at train | Export needs | Emitted by | Example |", "|---|---|---|---|---|---|---|"]
    for k in doc["event_kinds"]:
        fields = ", ".join(f"{f} ({CLASS_NOTES_LETTER[c]})" for f, c in k["fields"].items())
        owners = ", ".join(f"`{o}`" for o in k["emitted_by"][:3]) + (f" +{len(k['emitted_by']) - 3}" if len(k["emitted_by"]) > 3 else "") or "the host (an adapter, the browser's sender)"
        lines.append(f"| `{k['kind']}` | {fields or '—'} | {', '.join(k['never_leaves']) or '—'} | {', '.join(k['train_only']) or '—'} | {k['leaves_at']} | {owners} | `{k['example']}` |")
    lines += ["", f"## Decision points ({len(doc['decision_points'])})", "", "Every record of a point leaves under the consent of the event kinds' rule: written at `train`, "
              "train switch on, project allowing it; its example is `python -m dclab_rnd.ledger.shadow recorded` or the Decisions tab.", "",
              "| Point | Kind | Answer | Subjects | About | Decided in |", "|---|---|---|---|---|---|"]
    lines += [f"| `{p['point']}` | {p['kind']} | {p['answer']} | {', '.join(p['subjects'])} | {p['about']} | {p.get('decided_in') or '—'} |" for p in doc["decision_points"]]
    lines += ["", f"## Audit kinds ({len(doc['audit_kinds'])})", "", ", ".join(f"`{k}`" for k in doc["audit_kinds"]),
              "", f"## Dataset builders ({len(doc['dataset_builders'])})", "", "| Builder | What | Needs |", "|---|---|---|"]
    lines += [f"| `{b['builder']}` | {b['what']} | {b['needs']} |" for b in doc["dataset_builders"]]
    lines += ["", f"## Tables ({len(doc['tables'])})", "", "| Table | What | Files without a database |", "|---|---|---|"]
    lines += [f"| `{t['table']}` | {t['what']} | {('`' + t['files'] + '`') if t['files'] else '—'} |" for t in doc["tables"]]
    lines += ["", f"## Contracts ({len(doc['contracts'])})", "", "| Contract | Version | Also reads |", "|---|---|---|"]
    lines += [f"| {c['contract']} | `{c['version']}` | {', '.join('`' + r.rsplit('.', 1)[-1] + '`' for r in c['reads']) or '—'} |" for c in doc["contracts"]]
    return "\n".join(lines) + "\n"


CLASS_NOTES_LETTER = {k: v[0] for k, v in CLASS_NOTES.items()}


def outputs() -> dict[Path, str]:
    doc = build()
    return {DOCS / "DATA_CATALOG.md": render(doc), DOCS / "data_catalog.json": json.dumps(doc, indent=1, ensure_ascii=False, sort_keys=True) + "\n"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.data_catalog", description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail when the committed files differ from what the code generates")
    args = parser.parse_args(argv)
    made = outputs()
    if args.check:
        stale = [p.name for p, text in made.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
        if stale:
            print(f"data catalog is stale: {', '.join(stale)}; run: make catalog", file=sys.stderr)
            return 1
        print("data catalog is fresh")
        return 0
    for path, text in made.items():
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
