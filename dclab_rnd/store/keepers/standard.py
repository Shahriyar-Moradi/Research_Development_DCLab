"""The file keepers (32.3, 32.4): every file of every workspace, each kept by exactly one keeper under exactly one kind.

One inventory per run (:func:`inventory`) walks ``agent_runs`` with no depth limit: every file belongs to the deepest
workspace holding it (a live test's arm, a twin lane, a tester's folder), and its path inside that workspace decides its
kind by :data:`CLASSES`, first match wins; a file no class names is kept as ``other``, so nothing is left out. Each kind's
keeper keeps its files: an append-only log (``*.jsonl``) by its new tail, anything else whole when it changed.

What is never kept as content, only counted (each file's name, size and hash):

- a table's rows (any ``data/`` folder, and exports of tables): they are kept only where a person said yes (32.16);
- a file whose name says secret (``.env``, a key, a token: ``testers.NEVER``);
- everything of a workspace whose consent is ``operate``.

Column previews and the other keys that quote a table's cells (``testers.QUOTES_CELLS``) are removed from the files that hold
them (``testers.SCRUBBED``: projects, stages, activity, drafts, events, intern sessions) before keeping, as the tester share
does; the entry says so. The consent of a file is its own workspace's (``workspaces.discover``).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from ..layout import ROOT
from . import Claim, Keeper, REGISTRY, Source, register
from .workspaces import AGENT_RUNS, Workspace, discover

TEST_RUNS = ROOT / "training_records" / "test_runs"
SKIP_NAMES = (".lock", ".part", ".tmp", ".rc", ".DS_Store")
TABLE_DIRS = re.compile(r"(^|/)(data|clean_exports|studio_assets|embeddings)/")
TABLE_FILES = re.compile(r"\.(csv|tsv|parquet|feather|xlsx|xls|arrow|npy|npz|pkl|pickle|sqlite3?)(\.gz)?$", re.I)  # a table wherever it sits
NOT_TABLES = re.compile(r"(^|/)checkpoints\.sqlite3$")  # the teams' checkpoints: their state, not a table of anyone's rows

# (kind, top prefix, pattern on the path inside its workspace), first match wins; the prefixes are the keepers' claims
CLASSES: tuple[tuple[str, str, re.Pattern], ...] = tuple((k, p, re.compile(rx)) for k, p, rx in (
    ("sessions", "experience/", r"^experience/"),
    ("decisions", "ledger/", r"^ledger/"),
    ("actions", "agent_steps", r"^agent_steps\.jsonl$"),
    ("audit", "audit", r"^audit\.jsonl$"),
    ("model_requests", "model_", r"^model_(requests|requests_checks|shadow)\.jsonl$|^model_exchanges/"),
    ("actions", "searches/", r"^searches/"),
    ("conversations", "drafts/", r"^drafts/"),
    ("conversations", "intern/", r"^intern/"),
    ("conversations", "workspace_chat/", r"^workspace_chat/"),
    ("conversations", "twin/", r"^twin/"),
    ("conversations", "mcp/", r"^mcp/"),
    ("agents", "claude_code/", r"^claude_code/"),  # a tester's own sessions, as their share brought them (kept by the agents keeper)
    ("workflows", "projects/", r"^projects/"),
    ("workflows", "orchestra/", r"^orchestra/"),
    ("workflows", "checkpoints", r"^checkpoints\.sqlite3$"),
    ("tasks", "jobs", r"^jobs(_history)?\.json(l)?$"),
    ("students", "students/", r"^students/"),
    ("students", "own_models/", r"^own_models/"),
    ("teachers", "judgments/", r"^judgments/"),
    ("tests", "live_tests/", r"^live_tests/"),
    ("tests", "code_runs/", r"^code_runs/"),
    ("other", "", r""),
))
KEEPER_OF = {"sessions": "sessions", "decisions": "decisions", "actions": "actions", "audit": "audit", "model_requests": "model_requests",
             "conversations": "conversations", "workflows": "projects", "tasks": "tasks", "students": "students", "teachers": "judgments",
             "tests": "live_tests", "other": "other", "agents": "agents"}  # kind -> the keeper of its files (a capture rule's finding names it)
FEEDS = {"sessions": ("m2_workflow_decisions", "m4_workflow_llm", "m6_stepwise_decisions"), "actions": ("m5_solver_llm", "m6_stepwise_decisions", "m7_judges"),
         "decisions": ("m1_students", "m2_workflow_decisions", "m3_ml_decisions"), "conversations": ("m4_workflow_llm", "m2_workflow_decisions", "m7_judges"),
         "model_requests": ("m4_workflow_llm", "m5_solver_llm", "m7_judges"), "audit": ("m2_workflow_decisions",),
         "workflows": ("m2_workflow_decisions", "m3_ml_decisions", "m5_solver_llm", "m6_stepwise_decisions"), "tasks": ("m2_workflow_decisions", "m6_stepwise_decisions"),
         "students": ("m1_students",), "teachers": ("m1_students", "m7_judges"), "tests": ("m4_workflow_llm", "m5_solver_llm", "m7_judges"),
         "other": (), "agents": ("m4_workflow_llm", "m5_solver_llm", "m7_judges")}


def classify(rel: str) -> tuple[str, str]:
    """The kind of a file by its path inside its workspace, and the claim prefix it falls under."""
    for kind, prefix, pattern in CLASSES:
        if pattern.search(rel):
            return kind, prefix
    return "other", ""


def _secret(rel: str) -> bool:
    from ...testers import NEVER

    return bool(NEVER.search(rel))


def _scrubbed(rel: str) -> bool:
    from ...testers import SCRUBBED

    return bool(SCRUBBED.match(rel) or rel.startswith("mcp/"))  # an MCP client's call carries a tool's result: scrubbed the same way


def scrub(rel: str, raw: bytes) -> bytes:
    """A file that may quote a table's cells, without the keys that quote them (each JSON line of a log on its own)."""
    from ...testers import _scrub

    if rel.endswith(".jsonl"):
        out = []
        for line in raw.decode("utf-8", errors="replace").splitlines(keepends=True):
            try:
                out.append(json.dumps(_scrub(json.loads(line)), ensure_ascii=False, sort_keys=True) + "\n")
            except ValueError:
                out.append(line)
        return "".join(out).encode()
    try:
        return (json.dumps(_scrub(json.loads(raw)), ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode()
    except ValueError:
        return raw


def notebook(raw: bytes) -> bytes:
    """A notebook without the outputs that show a table (a displayed frame, a plot): its code, its text, its errors and what it
    printed kept, a printed table in it replaced by its size."""
    from ..shapers import _tables_out

    try:
        doc = json.loads(raw)
        for cell in doc.get("cells") or []:
            if "outputs" in cell:
                kept = []
                for o in cell["outputs"] or []:
                    if not isinstance(o, dict):
                        continue
                    if o.get("output_type") == "error":
                        kept.append({k: o.get(k) for k in ("output_type", "ename", "evalue", "traceback")})
                    elif o.get("output_type") == "stream":
                        text = "".join(o.get("text") or []) if isinstance(o.get("text"), list) else str(o.get("text") or "")
                        kept.append({"output_type": "stream", "name": o.get("name"), "text": _tables_out(text)})
                cell["outputs"] = kept
            if "execution_count" in cell:
                cell["execution_count"] = None
        return (json.dumps(doc, ensure_ascii=False, indent=1) + "\n").encode()
    except (ValueError, AttributeError):
        return raw


class Item:
    __slots__ = ("ws", "rel", "path", "kind", "prefix", "counted")

    def __init__(self, ws: Workspace, rel: str, path: Path, kind: str, prefix: str, counted: Optional[str]):
        self.ws, self.rel, self.path, self.kind, self.prefix, self.counted = ws, rel, path, kind, prefix, counted


def inventory(env: Any) -> list[Item]:
    """Every file under the workspaces' folder, each in its deepest workspace, classified once (cached on ``env`` for the run)."""
    if isinstance(env, dict) and "_inventory" in env:
        return env["_inventory"]
    base = Path(env["agent_runs"]) if isinstance(env, dict) and env.get("agent_runs") else AGENT_RUNS
    only = Path(env["only"]).resolve() if isinstance(env, dict) and env.get("only") else None
    roots = {w.root.resolve(): w for w in discover(base)}
    items: list[Item] = []
    if base.exists():
        for path in sorted(base.rglob("*")):
            if path.is_symlink() or not path.is_file() or path.name.endswith(SKIP_NAMES):
                continue
            resolved = path.resolve()
            root = next((p for p in resolved.parents if p in roots), None)  # the deepest workspace holding it
            if root is None:
                continue
            ws = roots[root]
            if only is not None and root != only:
                continue
            rel = resolved.relative_to(root).as_posix()
            kind, prefix = classify(rel)
            table = TABLE_DIRS.search(rel) or (TABLE_FILES.search(rel) and not NOT_TABLES.search(rel))
            counted = ("a table's rows: kept only where a person said yes" if table else
                       "a secret's file name" if _secret(rel) else
                       "this workspace's consent does not allow keeping its content" if ws.consent == "operate" else None)
            items.append(Item(ws, rel, path, kind, prefix, counted))
    if isinstance(env, dict):
        env["_inventory"] = items
    return items


def _counted(group: list[Item]) -> bytes:
    rows = []
    for item in group:
        try:
            rows.append({"source": item.rel, "bytes": item.path.stat().st_size, "sha256": hashlib.sha256(item.path.read_bytes()).hexdigest(),
                         "why": item.counted})
        except OSError as exc:
            rows.append({"source": item.rel, "error": f"{type(exc).__name__}: {exc}"})
    return json.dumps(rows, indent=0).encode()


def _secrets(raw: bytes) -> bytes:
    """Anything shaped like a credential replaced by a marker, in every file every keeper keeps (the store is pushed)."""
    from .agents import redact

    return redact(raw)


def _gz_secrets(raw: bytes) -> bytes:
    """A gzip file with the credentials inside it replaced (opened, cleaned, written again with no time inside)."""
    import gzip
    import io

    try:
        inner = gzip.decompress(raw)
    except OSError:
        return _secrets(raw)
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as fh:
        fh.write(_secrets(inner))
    return buf.getvalue()


def code_runs(raw: bytes) -> bytes:
    """The workspace's code runs, a notebook cell's as a notebook is kept: what it printed without a printed table, its
    pictures (a plot of the rows) counted, not kept. The workspace's own file stays whole."""
    from ..shapers import _tables_out

    out = []
    for line in raw.decode("utf-8", errors="replace").splitlines(keepends=True):
        try:
            doc = json.loads(line)
        except ValueError:
            out.append(line)
            continue
        if isinstance(doc, dict) and doc.get("kind") == "notebook_cell":
            doc["stdout"] = _tables_out(str(doc.get("stdout") or ""))
            pictures = doc.get("pictures")
            doc["pictures"] = len(pictures) if isinstance(pictures, list) else pictures
            charts = doc.get("charts")  # a chart's points may be a column's own values: counted, not kept
            doc["charts"] = len(charts) if isinstance(charts, list) else charts
        out.append(json.dumps(doc, ensure_ascii=False) + "\n")
    return "".join(out).encode()


def _for_kind(kind: str) -> Any:
    def sources(env: Any) -> Iterator[Source]:
        counts: dict[tuple[str, str], list[Item]] = {}
        for item in inventory(env):
            if item.kind != kind:
                continue
            if item.counted:
                counts.setdefault((item.ws.name, item.prefix), []).append(item)
                continue
            log = item.rel.endswith(".jsonl")
            if item.rel.endswith(".ipynb"):
                yield Source(item.ws.name, item.rel, item.path, consent=item.ws.consent, transform=lambda raw: _secrets(notebook(raw)),
                             note="its cells' outputs removed (a printed table is a table's rows); its code and text kept")
            elif item.rel.startswith("code_runs/"):
                yield Source(item.ws.name, item.rel, item.path, consent=item.ws.consent, log=log, transform=lambda raw: _secrets(code_runs(raw)),
                             note="a notebook cell's printed tables removed and its pictures counted, as a notebook's outputs")
            elif _scrubbed(item.rel):
                yield Source(item.ws.name, item.rel, item.path, consent=item.ws.consent, log=log, transform=lambda raw, r=item.rel: _secrets(scrub(r, raw)),
                             note="scrubbed: the keys that quote a table's cells removed (testers.QUOTES_CELLS)")
            else:
                yield Source(item.ws.name, item.rel, item.path, consent=item.ws.consent, log=log, transform=_secrets)
        for (ws_name, prefix), group in sorted(counts.items(), key=lambda kv: kv[0]):
            yield Source(ws_name, f"{prefix}~counts.json", lambda g=group: _counted(g), consent=group[0].ws.consent,
                         note="counts only: " + "; ".join(sorted({i.counted for i in group})))
    return sources


def _repo_files(base: Path, prefix: str, workspace: str = "rnd") -> Any:
    def sources(env: Any) -> Iterator[Source]:
        if isinstance(env, dict) and env.get("only"):
            return  # before one workspace's removal: the repository's own records are not that workspace's
        root = Path(env["test_runs"]) if isinstance(env, dict) and env.get("test_runs") else base
        for path in sorted(root.rglob("*")) if root.exists() else []:
            if path.is_file() and not path.name.endswith(SKIP_NAMES):
                yield Source(workspace, f"{prefix}{path.relative_to(root).as_posix()}", path, consent="train", log=path.suffix == ".jsonl",
                             transform=_gz_secrets if path.suffix == ".gz" else _secrets)  # a run's own gzip: opened, cleaned, closed again
    return sources


def _claims(kind: str) -> tuple[Claim, ...]:
    return tuple(Claim(kind, p) for k, p, _ in CLASSES if k == kind)


ABOUT = {"sessions": "the experience log's events", "actions": "agent steps and tool calls", "decisions": "the ledger: records, labels, outcomes",
         "conversations": "Home, the intern (every saved version), the workspace chat, the twin", "model_requests": "every model request with its shadows and checks",
         "audit": "the audit trail", "workflows": "the projects' records, the teams' boards, walks and checkpoints", "tasks": "every job's state changes",
         "students": "every student: registers, cards, measurements and fitted files", "teachers": "every judgment a teacher made, whole",
         "tests": "the live tests' own files", "other": "anything else a workspace holds, so nothing is left out",
         "agents": "a tester's own Claude Code sessions, as their share brought them (with their yes)"}
KEEPERS = tuple(Keeper(KEEPER_OF[kind], _claims(kind), _for_kind(kind), FEEDS[kind], ABOUT[kind]) for kind in KEEPER_OF if kind != "agents")
TEST_RUNS_KEEPER = Keeper("test_runs", (Claim("tests", "test_runs/"),), _repo_files(TEST_RUNS, "test_runs/"), ("m2_workflow_decisions", "m7_judges"),
                          "every test run and gate of this repository, whole")


def install(registry: Optional[dict[str, Keeper]] = None) -> dict[str, Keeper]:
    from .. import fixes, workflows
    from .agents import KEEPER as AGENTS
    from .commits import KEEPER as COMMITS
    from .database import KEEPER as DATABASE
    from .people import KEEPER as PEOPLE

    reg = REGISTRY if registry is None else registry
    walks = Keeper("walks", (Claim("workflows", "walks/"),), workflows.keeper_sources, ("m2_workflow_decisions", "m6_stepwise_decisions", "m4_workflow_llm"),
                   "each walk from a draft to a report as one workflow record (32.5)")
    fix = Keeper("fixes", (Claim("fixes", "fixes/"),), fixes.keeper_sources, ("m4_workflow_llm", "m5_solver_llm", "m7_judges"),
                 "a failing test, the run that passed it next, and the commits between (32.6)")
    for keeper in KEEPERS + (TEST_RUNS_KEEPER, DATABASE, PEOPLE, AGENTS, COMMITS, walks, fix):
        if keeper.name not in reg:
            register(keeper, reg)
    return reg


__all__ = ["CLASSES", "KEEPER_OF", "classify", "inventory", "scrub", "KEEPERS", "install"]
