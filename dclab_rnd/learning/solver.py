"""The intern as the solver of a challenge pack (H3 of the harness-learning plan): a local model does the homework.

``intern_attempt(folder, client, playbook)`` opens the pack's public table as a project, gives the intern the person's
request (the outcome, the moment, the task: nothing of the key), lets it work with its tools under a budget, and returns
the finished project as an acceptance ``Run`` for the grader, with the intern's session. Where the project is kept:

* a **learning** variant's attempt is a workspace of its own under ``<home>/learning_attempts/<date>/<id>/`` (its
  project with every stage record, the intern's whole session), so the store keeps it as a scored solver run;
* a **check or sealed** variant's attempt runs in a temporary folder and leaves nothing but its grade: an exam problem
  never becomes training data (no event, no decision, a model exchange's shape only; ``_recording``).

``local_gateway(model)`` serves the intern from a model on this machine only (Ollama by default), refused otherwise:
a learning loop repeats, and only a local model repeats for free.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dclab_core.challenge_pack import ChallengeKey

from .packs import read_public

OLLAMA = "http://127.0.0.1:11434/v1"
BUDGET = {"max_steps": 24, "max_minutes": 15}
MIN_CONTEXT = 16384  # the intern's first message is about 5,000 tokens (its rules and 28 tools) and grows every step


class ContextTooSmall(RuntimeError):
    """The server cuts every prompt to a window smaller than the intern's conversation: the model would never see its task."""


def served_context(model: str, base_url: str = OLLAMA) -> int | None:
    """The context window the server loaded the model with (Ollama's /api/ps, after a first request), or None when the
    server does not say. Measured 2026-10-23: Ollama loads a model with 4,096 tokens unless told otherwise, and silently
    cuts each prompt to about half of it (every pilot request carried 2,050 tokens), so the intern lost its task."""
    import urllib.request

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        with urllib.request.urlopen(f"{root}/api/ps", timeout=10) as r:
            models = json.loads(r.read().decode()).get("models") or []
    except Exception:  # noqa: BLE001 - not Ollama, or no such route: unknown
        return None
    for m in models:
        if m.get("name") == model or m.get("model") == model:
            value = m.get("context_length")
            return int(value) if isinstance(value, (int, float)) else None
    return None


def check_context(model: str, base_url: str = OLLAMA, need: int = MIN_CONTEXT) -> int | None:
    """Refuse a model served with a window under ``need`` (a model variant with ``PARAMETER num_ctx`` fixes it)."""
    window = served_context(model, base_url)
    if window is not None and window < need:
        raise ContextTooSmall(f"{model} is served with a {window}-token window; the intern needs at least {need}: "
                              f"make a variant with 'PARAMETER num_ctx 32768' (ollama create NAME -f Modelfile) and use it")
    return window


class NotLocal(RuntimeError):
    """The intern would be served by a model that is not on this machine."""


def local_gateway(model: str, base_url: str = OLLAMA, home: Path | None = None, reasoning_field: str | None = "reasoning") -> Any:
    """A gateway whose every tier is the local model, for this process; the intern's purpose checked to be served by it.
    ``reasoning_field``: the field the server returns the model's reasoning in (Ollama's /v1: ``reasoning``, measured
    2026-10-23, research_notes/open_models_2026-10/servers.md), read and sent back on every turn."""
    from dclab_rnd.models import settings
    from dclab_rnd.models.gateway import Gateway, for_workspace

    if not settings.is_local(base_url):
        raise NotLocal(f"{base_url} is not on this machine: a learning loop runs on a local model only")
    os.environ.update({"DCLAB_LLM_PROVIDER": "", "OPENAI_API_KEY": "ollama", "OPENAI_BASE_URL": base_url, "DCLAB_LLM_BASE_URL": base_url,
                       "OPENAI_MODEL": model})
    os.environ.update({f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.ROUTABLE for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")})
    os.environ.update({f"DCLAB_TIER_{t.upper()}_REASONING_FIELD": reasoning_field or "" for t in settings.ROUTABLE})
    os.environ.pop("DCLAB_INTERN_MODEL", None)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    os.environ.setdefault("DCLAB_LOG_PROMPTS", "1")  # a local model's whole exchanges are ours to keep (the owner's capture rule)
    gateway = for_workspace(home) if home is not None else Gateway()
    served = gateway.route("intern")[0]
    tier = settings.tier(served)
    if not settings.is_local(tier.get("base_url") or ""):
        raise NotLocal(f"the intern would be served by the {served} tier at {tier.get('base_url')}: refused")
    return gateway


def task_of(pack: Any) -> str:
    """The person's request as the intern reads it: the outcome, the task and the moment, never a trap."""
    b = pack.brief
    return (f"{b.request}\nThe outcome column is `{b.target}` ({b.task}). The prediction is made: {b.moment}.\n"
            "Work in the existing project: look at the data, save the solution (the columns to keep out with their reasons, "
            "identifiers, and any time or group column the split must respect), run every stage, and report the holdout score.")


def _empty() -> dict[str, Any]:
    return {"evidence": {}}


@contextlib.contextmanager
def _recording(root: Path, keep: bool) -> Iterator[None]:
    """Where an attempt's records go. A kept (learning) attempt records into its own folder, beside its projects: every
    event, every decision, so the store finds one whole workspace. An exam attempt records nothing (the events quiet, no
    ledger) and its model exchanges keep their shape only, never a message: an exam problem never becomes training data.
    The previous log, ledger and switch come back after. (A server request's own services would win over the installed
    ones; the learning loop runs from the command line.)"""
    from dclab_rnd import experience, ledger

    log, book, prompts = experience._INSTALLED.get("log"), ledger._INSTALLED.get("ledger"), os.environ.get("DCLAB_LOG_PROMPTS")
    try:
        if keep:
            experience.install(experience.open_log(root))
            ledger.install(ledger.open_ledger(root))
            yield
        else:
            ledger.install(None)
            os.environ["DCLAB_LOG_PROMPTS"] = "0"
            with experience.quiet():
                yield
    finally:
        experience.install(log)
        ledger.install(book)
        if prompts is None:
            os.environ.pop("DCLAB_LOG_PROMPTS", None)
        else:
            os.environ["DCLAB_LOG_PROMPTS"] = prompts


def intern_attempt(folder: Path, client: Any, key: ChallengeKey, playbook: dict[str, Any] | None = None, home: Path | None = None,
                   budget: dict[str, Any] | None = None, label: str = "") -> dict[str, Any]:
    """One homework by the intern: the finished project as a ``Run`` (stages not reached are empty, so the grader marks
    them), the session, whether the final stage ran, and where the attempt was kept (None for an exam problem)."""
    from dclab_rnd.intern import Intern, SessionStore
    from dclab_rnd.intern.tools import Toolbox
    from dclab_rnd.kinds.acceptance import Run
    from dclab_rnd.studio import ProjectStore, data as studio_data
    from dclab_rnd.studio.store import STAGE_KEYS

    pack, table = read_public(folder)
    keep = key.partition == "learning" and home is not None
    if keep:
        at = datetime.now(timezone.utc)
        root = Path(home) / "learning_attempts" / f"{at:%Y-%m-%d}" / f"{at:%H%M%S}-{pack.id}{('-' + label) if label else ''}"
        root.mkdir(parents=True, exist_ok=True)
        scratch = None
    else:
        scratch = tempfile.TemporaryDirectory()
        root = Path(scratch.name)
    started = time.monotonic()
    try:
        with _recording(root, keep):
            store = ProjectStore(root / "projects")
            project = store.create("Challenge", "general", pack.brief.request[:300])
            pid = project["id"]
            shutil.copyfile(Path(folder) / "table.csv", store.data_dir(pid) / "table.csv")
            studio_data.attach_data(store, pid, "table.csv")
            toolbox = Toolbox(store)
            if keep:  # the tools offered, whole, beside the session: a trainer shows the model what it could call
                _keep_tools(root, toolbox)
            intern = Intern(SessionStore(root / "intern"), toolbox, client)
            session = intern.run(intern.start(task_of(pack), {**BUDGET, **(budget or {})}, project_id=pid, playbook=playbook, by="system")["id"])
            project = store.get(pid)
            records = {}
            for stage in STAGE_KEYS:
                try:
                    records[stage] = store.read_stage(pid, stage) or _empty()
                except (KeyError, FileNotFoundError, ValueError):
                    records[stage] = _empty()
            finished = (records.get("final") or {}).get("status") == "completed"
            run = Run(case=None, solution=project.get("solution") or {}, records=records)  # type: ignore[arg-type]
            return {"run": run, "session": session, "finished": finished, "kept": str(root) if keep else None,
                    "seconds": round(time.monotonic() - started, 1), "table": len(table)}
    finally:
        if scratch is not None:
            scratch.cleanup()


def _keep_tools(root: Path, toolbox: Any) -> None:
    """``tools.json``: the schemas the intern was offered, exactly as the loop sends them (the project's tools and the
    session's own ``write_plan`` and ``finish``), their hash as the run records it, and the hash of its rules (D2's open
    item: an intern session keeps its messages but not the tools it could call)."""
    import hashlib

    from dclab_rnd.agents.runtime import tools_hash
    from dclab_rnd.intern.loop import POLICY, SESSION_TOOLS

    tools = toolbox.registry.schemas(names=list(toolbox.names()) + list(SESSION_TOOLS))
    (Path(root) / "tools.json").write_text(json.dumps({"tools": tools, "tools_sha256": tools_hash(tools),
                                                        "policy_sha256": hashlib.sha256(POLICY.encode()).hexdigest()[:16]}, indent=1) + "\n",
                                            encoding="utf-8")


def summary_of(session: dict[str, Any]) -> dict[str, Any]:
    """What a session did, without its words: steps, tool errors, how it ended, the thinking slots filled."""
    steps = session.get("steps") or []
    return {"status": session.get("status"), "ended": session.get("ended"), "steps": len(steps),
            "tool_errors": sum(1 for s in steps if s.get("ok") is False), "tools": [s.get("tool") for s in steps],
            "thoughts": {src: sum(1 for s in steps if s.get("thought_source") == src) for src in ("recorded", "visible", "from_record")},
            "error": (session.get("error") or "")[:300] or None, "playbook": {k: (session.get("playbook") or {}).get(k) for k in ("sha256", "shown", "used")}}


def keep_grade(root: str | None, grade: Any, observed: Any) -> None:
    """The grade beside a kept attempt (codes and checks; the key never)."""
    if root:
        (Path(root) / "grade.json").write_text(json.dumps({"grade": grade.model_dump(), "observed": observed.model_dump()}, indent=1) + "\n",
                                                encoding="utf-8")


__all__ = ["OLLAMA", "BUDGET", "MIN_CONTEXT", "ContextTooSmall", "served_context", "check_context", "NotLocal", "local_gateway", "task_of", "intern_attempt", "summary_of", "keep_grade"]
