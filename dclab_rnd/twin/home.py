"""The twin on Home, live (package 25.4): while DCLab's Home agent talks with a person, Claude, as the AI-first expert,
reads the same conversation and the same data and answers every Home step its own way; both answers, step by step,
are kept in the workspace's twin folder and shown side by side.

The owner's switch, ``<workspace>/twin.json`` (all off by default; changed by the owner, audited):

    on            run Claude beside every Home conversation
    sample        Claude may also see up to 20 redacted rows of the table (its columns that hold a person's data left out)
    session_eur   what one draft's Claude may spend (1.5: the owner's cap, 2026-10-16)
    monthly_eur   what every draft's Claude may spend together this month

After each turn of the Home agent (a draft's start, each message), and on its own for a message the person wrote to
Claude alone, :func:`after_turn` starts the AI-first turn in a thread of its own, so the system's turn never waits for
it and never fails because of it. The turn asks Claude once (purpose ``ai_first``, through the gateway: routed, capped
by ``run_limit_eur``, counted), with the expert's prompt and Home's (``prompts/ai_first_expert.md`` and
``ai_first_home.md``); reads its answer into an ``ExpertTrace`` with the Home keys (the reply, the next question, the
action, the pack, whether it is ready, what blocks it); sets each Home step beside the system's state after the same
turn as a ``dclab_core.twin`` step; and tells the page (an SSE event ``twin``). The system never sees Claude's answer
nor a message written to Claude alone.

The folder (``<workspace>/twin/``; a ``README.md`` there says the same):

    index.jsonl                                   one line per session (a draft), with where its folder is
    sessions/YYYY/MM/DD/<session>/session.json    the session: draft, settings, spend, turns, steps
    sessions/…/<session>/conversation.jsonl       the person's messages to Claude alone, and Claude's replies
    sessions/…/<session>/ai_first.jsonl           each Claude turn: what it was given (names and counts), its answer, the record read from it, cost
    sessions/…/<session>/system.jsonl             the system's state after each turn, in the twin steps' shapes
    sessions/…/<session>/thinking.jsonl           Claude's thinking, when the transport returns it
    sessions/…/<session>/steps.jsonl              the twin steps: both answers per Home step, their differences, the verdict once a truth is known
    sessions/…/<session>/picks.jsonl              the person's picks between the two answers (each also a step with its truth)
    sessions/…/<session>/budget.jsonl             each time a person added to this draft's Claude budget, by whom (25.10)
    runs/<run>/                                   the twin runs of whole problems (25.2, 25.3)

The sample's rows are never written to the folder: only their count and the columns left out.
"""

from __future__ import annotations

import contextvars
import json
import random
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

SETTINGS_FILE = "twin.json"
DEFAULTS: dict[str, Any] = {"on": False, "sample": False, "session_eur": 1.5, "monthly_eur": 30.0}
LIMITS = {"session_eur": (0.01, 15.0), "monthly_eur": (0.01, 500.0)}
RAISE_STEP = 1.5  # what one press of "add budget" gives a draft's Claude (25.10)
SAMPLE_ROWS = 20
CELL_CHARS = 60
MAX_TOKENS = 4000
HOME_STEPS = ("home.understanding", "home.next_question", "home.pack", "home.map", "home.workflow", "home.ready")
README = """# The twin folder

DCLab's own system and Claude, as an AI-first machine-learning expert, on the same problems (phase 3,
docs/guides/PHASE3_AI_FIRST_PLAN.md).

| Path | What it holds |
|---|---|
| `index.jsonl` | one line per Home session (a draft): its id, the draft, when it started, where its folder is |
| `sessions/YYYY/MM/DD/<session>/session.json` | the session: the draft, the settings it ran under, what Claude spent, its turns and steps, the people (account ids) it was for |
| `sessions/.../conversation.jsonl` | the person's messages to Claude alone, and Claude's replies |
| `sessions/.../ai_first.jsonl` | each Claude turn: whose (the account id), what it was given (names and counts), its answer, the expert record read from it, tokens and cost |
| `sessions/.../system.jsonl` | the system's state after each turn, in the twin steps' shapes |
| `sessions/.../thinking.jsonl` | Claude's thinking, when the transport returns it |
| `sessions/.../steps.jsonl` | the twin steps (`dclab_core.twin`): both answers per Home step, the differences, the verdict once a truth is known |
| `sessions/.../picks.jsonl` | the person's picks between the two answers; each pick is also a step with its truth (`person_pick`) |
| `sessions/.../budget.jsonl` | each time a person added to this draft's Claude budget: before, after, by whom |
| `runs/<run>/` | twin runs of whole problems: two lanes, the comparison, Claude's expert record and how it was followed (docs/guides/TWIN_RUNS.md) |

The table's rows never enter this folder. Training data is built from it only under the workspace's train consent.
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def person() -> Optional[str]:
    """The signed-in person this turn is for (their account id), or None without accounts: every row says whose it is."""
    try:
        from ..accounts.principal import current

        who = current()
        return who.user_id if who is not None and who.user_id else None
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------- the owner's switch


def settings(home: Path) -> dict[str, Any]:
    path = Path(home) / SETTINGS_FILE
    try:
        stored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        stored = {}
    return {k: stored.get(k, v) if isinstance(stored.get(k, v), type(v)) or (isinstance(v, float) and isinstance(stored.get(k), int)) else v
            for k, v in DEFAULTS.items()}


def set_settings(home: Path, changes: dict[str, Any]) -> dict[str, Any]:
    """The switch changed: only the known keys, booleans as booleans, caps within their bounds (a ValueError otherwise)."""
    current = settings(home)
    for key, value in changes.items():
        if key not in DEFAULTS:
            raise ValueError(f"{key} is not a twin setting; one of {', '.join(DEFAULTS)}")
        if isinstance(DEFAULTS[key], bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} is true or false")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{key} is a number of euros")
            low, high = LIMITS[key]
            if not low <= float(value) <= high:
                raise ValueError(f"{key} is between {low} and {high}")
            value = float(value)
        current[key] = value
    path = Path(home) / SETTINGS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
    tmp.replace(path)
    return current


# ------------------------------------------------------------------------------------------------- the folder


class Sessions:
    """The twin folder's Home sessions: one per draft, by day, with an index."""

    _locks: dict[str, threading.RLock] = {}
    _guard = threading.Lock()

    def __init__(self, home: Path):
        from . import twin_home

        self.root = twin_home(Path(home))
        self._lock = self.folder_lock(self.root)

    @classmethod
    def folder_lock(cls, path: Path) -> threading.RLock:
        """One lock per twin folder for every instance in the process: a turn and a raise never write session.json at once."""
        with cls._guard:
            return cls._locks.setdefault(str(Path(path).resolve()), threading.RLock())

    @staticmethod
    def _write_json(path: Path, doc: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(path)  # never read half-written

    def _index(self) -> list[dict[str, Any]]:
        path = self.root / "index.jsonl"
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def find(self, session: str) -> Optional[Path]:
        row = next((r for r in self._index() if r.get("session") == session), None)
        return self.root / row["path"] if row else None

    def open(self, draft_id: str, settings_now: dict[str, Any]) -> tuple[str, Path]:
        """The draft's session, made on its first turn."""
        session = f"draft-{draft_id}"
        with self._lock:
            folder = self.find(session)
            if folder is not None and folder.is_dir():
                return session, folder
            day = datetime.now(timezone.utc).strftime("%Y/%m/%d")
            folder = self.root / "sessions" / day / session
            folder.mkdir(parents=True, exist_ok=True)
            self._write_json(folder / "session.json", {"session": session, "kind": "draft", "draft_id": draft_id, "started": now(),
                                                       "settings": settings_now, "spent_eur": 0.0, "turns": 0, "steps": 0})
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "index.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"session": session, "kind": "draft", "draft_id": draft_id, "started": now(),
                                    "path": str(folder.relative_to(self.root))}) + "\n")
            (self.root / "README.md").write_text(README, encoding="utf-8")
        return session, folder

    def append(self, folder: Path, name: str, row: dict[str, Any]) -> None:
        with self._lock, (folder / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def rows(folder: Path, name: str) -> list[dict[str, Any]]:
        path = folder / name
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def manifest(self, folder: Path) -> dict[str, Any]:
        return json.loads((folder / "session.json").read_text(encoding="utf-8"))

    def update(self, folder: Path, **changes: Any) -> dict[str, Any]:
        with self._lock:
            m = self.manifest(folder)
            for key, value in changes.items():
                m[key] = (m.get(key) or 0) + value if key in ("spent_eur", "turns", "steps") else value  # session_eur is set, never added
            self._write_json(folder / "session.json", m)
            return m

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        out = []
        for row in reversed(self._index()[-limit:]):
            folder = self.root / row["path"]
            if (folder / "session.json").is_file():
                out.append({**row, **{k: v for k, v in self.manifest(folder).items() if k in ("spent_eur", "turns", "steps", "last")}})
        return out

    def month_spent(self) -> float:
        month = datetime.now(timezone.utc).strftime("%Y-%m")
        total = 0.0
        for row in self._index():
            folder = self.root / row["path"]
            if (folder / "session.json").is_file():
                for turn in self.rows(folder, "ai_first.jsonl"):
                    if str(turn.get("at", "")).startswith(month):
                        total += float(turn.get("cost_eur") or 0.0)
        return total

    def forget(self, draft_id: str) -> int:
        """A deleted draft's twin session goes with it (its folder and its line in the index)."""
        session = f"draft-{draft_id}"
        with self._lock:
            rows = self._index()
            keep = [r for r in rows if r.get("session") != session]
            gone = [r for r in rows if r.get("session") == session]
            for r in gone:
                shutil.rmtree(self.root / r["path"], ignore_errors=True)
            if gone:
                (self.root / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in keep), encoding="utf-8")
        return len(gone)


# ------------------------------------------------------------------------------------------------- the system's side


def system_view(drafts: Any, draft: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The system's answer to each Home step after its turn, in the twin steps' shapes (the card, the open question, the
    pack, the map, the drawn workflow, the build's blockers)."""
    from ..draft import api, card as dcard, map as dmap

    out: dict[str, dict[str, Any]] = {}
    try:
        c = dcard.card_of(draft)
        b = c.brief
        out["home.understanding"] = {"outcome": c.outcome or b.target_definition.value, "moment": b.prediction_moment.value, "action": b.action.value,
                                     "unit": b.unit.value, "family": c.family.value, "labels": c.labels.value, "target_column": b.target_column.value}
    except Exception:  # noqa: BLE001 — a card that cannot be drawn yet is no answer
        out["home.understanding"] = {}
    open_q = next((q for q in reversed(draft.get("questions") or []) if not q.get("answered") and not q.get("superseded")), None)
    out["home.next_question"] = {"field": open_q.get("field"), "options": list(open_q.get("options") or [])} if open_q else {"field": "none"}
    pack = draft.get("pack")
    pack = pack.get("key") if isinstance(pack, dict) else pack  # the draft keeps the pack with its source and reason
    out["home.pack"] = {"pack": pack} if pack else {}
    try:
        m = dmap.current(draft)
        out["home.map"] = {"size": m.size, "families": sorted({sp.family for sp in m.sub_problems if sp.family}), "parts": [p.title for p in m.parts]} if m else {}
    except Exception:  # noqa: BLE001
        out["home.map"] = {}
    wf = draft.get("workflow") or {}
    out["home.workflow"] = {"steps": [n.get("wf") for n in wf.get("nodes") or [] if n.get("wf")]} if wf.get("nodes") else {}
    try:
        blockers = api.build_blockers(drafts, draft["id"])
        out["home.ready"] = {"ready": not blockers, "blockers": sorted({getattr(r, "rule", "blocked") for r in blockers})}
    except Exception:  # noqa: BLE001
        out["home.ready"] = {}
    return {k: {f: v for f, v in d.items() if v not in (None, "")} for k, d in out.items()}


# ------------------------------------------------------------------------------------------------- what Claude is given


def redacted_sample(drafts: Any, draft: dict[str, Any], rows: int = SAMPLE_ROWS) -> tuple[list[dict[str, Any]], list[str]]:
    """Up to ``rows`` rows of the draft's cleaned table for Claude: a column that holds a person's data (dclab_core.sensitivity)
    is left out, text is redacted (dclab_core.redact) and cut. Returns the rows and the columns left out."""
    import pandas as pd

    from dclab_core import redact, sensitivity

    asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset")), None) or next(iter(draft.get("assets") or []), None)
    if not asset or not asset.get("clean_file", "clean.parquet"):
        return [], []
    path = drafts.data_dir(draft["id"]) / asset.get("clean_file", "clean.parquet")
    if not path.is_file():
        return [], []
    frame = pd.read_parquet(path)
    if frame.empty:
        return [], []
    left_out = [c for c in frame.columns if sensitivity.classify(c, frame[c].dropna().astype(str).head(50).tolist())[0] != "none"]
    keep = [c for c in frame.columns if c not in left_out][:40]  # the purpose's limit: 20 rows of at most 40 columns
    picked = frame[keep].sample(n=min(rows, len(frame)), random_state=random.Random(draft["id"]).randint(0, 2**31 - 1))
    out = []
    for record in picked.to_dict(orient="records"):
        row = {}
        for key, value in record.items():
            if isinstance(value, str):
                text, _ = redact.redact_text(value)
                value = (text or "")[:CELL_CHARS]
            elif hasattr(value, "isoformat"):
                value = value.isoformat()
            elif value is not None and not isinstance(value, (int, float, bool)):
                value = str(value)[:CELL_CHARS]
            row[str(key)] = value
        out.append(row)
    return out, [str(c) for c in left_out]


def given(drafts: Any, draft: dict[str, Any], conversation: list[dict[str, Any]], sample_on: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """What Claude is given for this turn, and a summary of it in names and counts (what the folder keeps)."""
    from ..draft import map as dmap
    from ..draft import pack as packs

    messages = [{"role": m.get("role"), "text": (m.get("text") or "")[:2000], "kind": m.get("kind")} for m in draft.get("messages") or [] if m.get("text")]
    data: dict[str, Any] = {}
    try:
        card = dmap.data_card_of(draft)
        data = card.model_dump(mode="json") if card is not None else {}
    except Exception:  # noqa: BLE001
        data = {}
    analysis = draft.get("analysis") or {}
    findings = [h.get("title") for h in analysis.get("highlights") or [] if isinstance(h, dict) and h.get("title")][:20]
    rows, left_out = redacted_sample(drafts, draft) if sample_on else ([], [])
    payload = {"problem": (draft.get("problem") or "")[:4000], "conversation": messages[-40:], "to_you_alone": conversation[-20:],
               "data_card": data, "findings": findings, "packs": list(getattr(packs, "KEYS", ())), "sample_rows": rows,
               "sample_left_out": left_out}
    summary = {"messages": len(messages), "to_you_alone": len(conversation), "tables": len((data or {}).get("tables") or []),
               "findings": len(findings), "sample_rows": len(rows), "sample_left_out": len(left_out)}
    return payload, summary


# ------------------------------------------------------------------------------------------------- Claude's turn


def ai_values(trace: Any, take: dict[str, Any]) -> dict[str, dict[str, Any]]:
    from dclab_core import expert

    values = expert.step_values(trace)
    under = dict(values.get("home.understanding") or {})
    if take.get("action"):
        under["action"] = str(take["action"])[:400]
    nq = take.get("next_question") if isinstance(take.get("next_question"), dict) else {}
    out = {"home.understanding": under,
           "home.next_question": {k: v for k, v in {"field": str(nq.get("field") or "none")[:40],
                                                    "options": [str(o)[:120] for o in (nq.get("options") or [])[:4]] or None}.items() if v is not None},
           "home.pack": {"pack": str(take["pack"])[:60]} if take.get("pack") else {},
           "home.map": dict(values.get("home.map") or {}),
           "home.workflow": dict(values.get("home.workflow") or {}),
           "home.ready": {"ready": bool(take.get("ready"))} | ({"blockers": [str(b)[:120] for b in take.get("blockers") or []][:20]} if take.get("blockers") is not None else {})}
    if trace.problem_workflow.parts:
        out["home.map"]["parts"] = [p.title for p in trace.problem_workflow.parts]
    return out


class HomeTwin:
    """Claude beside one workspace's Home conversations."""

    _locks: dict[str, threading.Lock] = {}
    _guard = threading.Lock()

    def __init__(self, drafts: Any, gateway: Any, home: Path):
        self.drafts, self.gateway, self.home = drafts, gateway, Path(home)
        self.sessions = Sessions(self.home)

    def lock(self, draft_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(f"{self.home}:{draft_id}", threading.Lock())

    def cap(self, folder: Optional[Path], s: dict[str, Any]) -> float:
        """This draft's cap: the workspace's, or what a person raised it to (25.10)."""
        own = self.sessions.manifest(folder).get("session_eur") if folder is not None and (folder / "session.json").is_file() else None
        own = float(own) if isinstance(own, (int, float)) and not isinstance(own, bool) else 0.0
        return max(own, float(s["session_eur"]))  # a raised draft still follows a higher workspace default

    def refusal(self, folder: Path, s: dict[str, Any]) -> Optional[str]:
        spent = float(self.sessions.manifest(folder).get("spent_eur") or 0.0)
        cap = self.cap(folder, s)
        if spent >= cap:
            return f"this draft's Claude has spent its cap (€{spent:.2f} of €{cap:.2f}); add budget to go on"
        if self.sessions.month_spent() >= s["monthly_eur"]:
            return f"the workspace's Claude has spent this month's cap (€{s['monthly_eur']:.2f})"
        return None

    def turn(self, draft_id: str, event: str, text: Optional[str] = None, to: str = "both") -> Optional[dict[str, Any]]:
        """One AI-first turn after the system's (``to="both"``) or for a message to Claude alone (``to="claude"``)."""
        s = settings(self.home)
        if not s["on"]:
            return None
        with self.lock(draft_id):
            return self._turn(draft_id, event, text, to, s)

    def _turn(self, draft_id: str, event: str, text: Optional[str], to: str, s: dict[str, Any]) -> Optional[dict[str, Any]]:
        from dclab_core import expert, twin

        from .. import prompts

        draft = self.drafts.get(draft_id)
        session, folder = self.sessions.open(draft_id, s)
        if to == "claude" and text:
            self.sessions.append(folder, "conversation.jsonl", {"at": now(), "role": "user", "user": person(), "text": text[:4000]})
        refused = self.refusal(folder, s)
        if refused:
            self._status(folder, draft_id, session, "capped", refused)
            return {"status": "capped", "note": refused}
        conversation = [{"role": r.get("role"), "text": r.get("text")} for r in self.sessions.rows(folder, "conversation.jsonl")]
        payload, summary = given(self.drafts, draft, conversation, s["sample"])
        client = self.gateway.client("ai_first", draft_id=draft_id) if self.gateway is not None else None
        if client is None:
            note = ("Claude is not reachable on this server: no model is configured for it (set ANTHROPIC_API_KEY and "
                    "DCLAB_LLM_PROVIDER=anthropic in .env and restart), or live models are switched off (DCLAB_NO_LIVE_MODELS)")
            self._status(folder, draft_id, session, "unavailable", note)
            return {"status": "unavailable", "note": note}
        self._status(folder, draft_id, session, "thinking", "Claude is reading the conversation and the data")
        spent = float(self.sessions.manifest(folder).get("spent_eur") or 0.0)
        client.run_limit_eur = max(0.0, self.cap(folder, s) - spent)
        system_prompt = prompts.text("ai_first_expert") + "\n\n" + prompts.text("ai_first_home")
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"HOME TURN ({event})\n\n" + json.dumps(payload, ensure_ascii=False, default=str)}]
        options = {"thinking": "adaptive"} if (getattr(client, "tier", {}) or {}).get("provider") == "anthropic" else {}
        before = float(getattr(client, "spent_eur", 0.0) or 0.0)
        at = now()
        try:
            out = client.complete(messages, None, MAX_TOKENS, **options)
        except Exception as exc:  # noqa: BLE001 — Claude failing never fails the system's turn
            row = {"at": at, "user": person(), "event": event, "to": to, "given": summary, "status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            self.sessions.append(folder, "ai_first.jsonl", row)
            self._status(folder, draft_id, session, "failed", "Claude's request failed: " + row["error"])
            return row
        cost = max(0.0, float(getattr(client, "spent_eur", 0.0) or 0.0) - before)
        if out.get("finish_reason") == "content_filter":  # the model declined (Anthropic's stop reason "refusal"): said, never hidden
            row = {"at": at, "user": person(), "event": event, "to": to, "given": summary, "status": "declined", "cost_eur": round(cost, 6)}
            self.sessions.append(folder, "ai_first.jsonl", row)
            self.sessions.update(folder, spent_eur=round(cost, 6))
            self._status(folder, draft_id, session, "declined", "Claude declined to answer this request (the provider's refusal); DCLab's own answer stands")
            return row
        raw_answer = str(out.get("content") or "")
        raw, problems = expert.extract(raw_answer)
        take = raw or {}
        trace = expert.read(take, version=int(self.sessions.manifest(folder).get("turns") or 0) + 1, checkpoint=event, model=getattr(client, "model", None))
        usage = out.get("usage") or {}
        row = {"at": at, "user": person(), "event": event, "to": to, "given": summary, "status": "answered" if raw else "unreadable", "answer": raw_answer[:20000],
               "problems": problems + list(trace.problems), "trace": trace.model_dump(mode="json"), "reply": str(take.get("reply") or "")[:2000],
               "cost_eur": round(cost, 6), "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "model": getattr(client, "model", None)}
        self.sessions.append(folder, "ai_first.jsonl", row)
        if out.get("thinking"):
            self.sessions.append(folder, "thinking.jsonl", {"at": at, "event": event, "thinking": [t.get("text") for t in out["thinking"]]})
        if row["reply"]:
            self.sessions.append(folder, "conversation.jsonl", {"at": now(), "role": "claude", "text": row["reply"], "event": event})
        system = system_view(self.drafts, self.drafts.get(draft_id))
        self.sessions.append(folder, "system.jsonl", {"at": now(), "event": event, "values": system})
        mine = ai_values(trace, take) if raw else {}
        shared = {"draft": draft_id, "messages": summary["messages"], "event": event, "at": at}
        written = []
        for step in HOME_STEPS:
            a, b = system.get(step) or {}, mine.get(step) or {}
            a = {k: v for k, v in a.items() if k in twin.spec(step).fields}
            b = {k: v for k, v in b.items() if k in twin.spec(step).fields}
            if not a and not b:
                continue
            try:
                record = twin.step(f"{session}:{row['at']}:{step}", session, step, at, shared,
                                   twin.ArmAnswer(arm="system", answerer="home (rules and roles)", value=a, status="answered" if a else "abstained"),
                                   twin.ArmAnswer(arm="ai_first", answerer="ai_first@home", model=row["model"], value=b,
                                                  status="answered" if b else "abstained", said=row["reply"] or None, cost_eur=cost if step == HOME_STEPS[0] else 0.0,
                                                  input_tokens=int(usage.get("input_tokens") or 0) if step == HOME_STEPS[0] else 0,
                                                  output_tokens=int(usage.get("output_tokens") or 0) if step == HOME_STEPS[0] else 0),
                                   subject=event, input_summary={"draft": draft_id, "messages": summary["messages"], "sample_rows": summary["sample_rows"]})
            except ValueError as exc:  # an answer that cannot be kept (too long, a row) is said, never stored
                self.sessions.append(folder, "ai_first.jsonl", {"at": now(), "event": event, "status": "step_refused", "step": step, "error": str(exc)[:300]})
                continue
            self.sessions.append(folder, "steps.jsonl", json.loads(record.model_dump_json()))
            written.append({"step": step, "differs": [d.field for d in record.differences if d.kind not in ("same", "neither")]})
        people = sorted({*(self.sessions.manifest(folder).get("people") or []), *([row["user"]] if row["user"] else [])})
        self.sessions.update(folder, spent_eur=round(cost, 6), turns=1, steps=len(written), last=now(), people=people,
                             status={"state": row["status"], "note": None if raw else "Claude answered, but not as a readable record", "at": now()})
        self._tell(draft_id, {"session": session, "status": row["status"], "reply": row["reply"], "steps": written, "cost_eur": round(cost, 6),
                              "question": (take.get("next_question") or {}) if isinstance(take.get("next_question"), dict) else {}})
        return {**row, "steps": written}

    def _status(self, folder: Path, draft_id: str, session: str, state: str, note: str) -> None:
        """Where Claude's lane stands (thinking, answered, capped, unavailable, failed), kept on the session so the page shows
        it after a reload, and told to the page now."""
        self.sessions.update(folder, status={"state": state, "note": note[:600], "at": now()})
        self._tell(draft_id, {"session": session, "status": state, "note": note[:600]})

    def _tell(self, draft_id: str, payload: dict[str, Any]) -> None:
        try:
            self.drafts.emit(draft_id, "twin", payload)
        except Exception:  # noqa: BLE001 — the page learns on its next read
            pass

    def raise_budget(self, draft_id: str, add_eur: float, who: Optional[str] = None, room: Optional[float] = None) -> dict[str, Any]:
        """A person adds to this draft's Claude budget: by ``add_eur`` (at most the session bound, 15, in all), never past
        ``room`` (what is left of the person's own monthly limit, when accounts are on). Kept in the session's
        ``budget.jsonl`` with who and when; the caller audits it and records it in the session record."""
        if isinstance(add_eur, bool) or not isinstance(add_eur, (int, float)) or not 0 < float(add_eur) <= LIMITS["session_eur"][1]:
            raise ValueError(f"add between 0 and {LIMITS['session_eur'][1]:.2f} euros")
        s = settings(self.home)
        if not s["on"]:
            raise ValueError("Claude beside the system is off in this workspace (the owner's switch)")
        session, folder = self.sessions.open(draft_id, s)
        with self.sessions._lock:  # read, decide and write as one: two raises at once are two raises
            before = self.cap(folder, s)
            after = min(LIMITS["session_eur"][1], before + float(add_eur))
            if room is not None:
                spent = float(self.sessions.manifest(folder).get("spent_eur") or 0.0)
                after = min(after, spent + max(0.0, room))
            if after <= before:
                raise ValueError("no room left: this draft is at its bound, or your monthly model budget is used up")
            after = round(after, 2)
            self.sessions.update(folder, session_eur=after)
            self.sessions.append(folder, "budget.jsonl", {"at": now(), "by": who or "person", "before": round(before, 2), "after": after, "asked": float(add_eur)})
        return {"session": session, "before": round(before, 2), "session_eur": after}

    def pick(self, draft_id: str, step: str, arm: str) -> dict[str, Any]:
        """The person's pick: the chosen arm's answer, on the fields where the two differ, is the step's truth (``person_pick``).
        A new line is appended (the step's latest is the one shown); the old one stays as it was."""
        from dclab_core import twin

        if arm not in twin.ARMS:
            raise ValueError("arm is system or ai_first")
        folder = self.sessions.find(f"draft-{draft_id}")
        rows = [r for r in self.sessions.rows(folder, "steps.jsonl") if r.get("step") == step] if folder else []
        if not rows:
            raise KeyError(f"no twin step {step} on this draft yet")
        last = twin.TwinStep.read(rows[-1])
        chosen = getattr(last, arm)
        fields = [d.field for d in last.differences if d.kind not in ("same", "neither") and chosen.value.get(d.field) is not None]
        if not fields:
            raise ValueError("the two answers do not differ on anything the chosen one said")
        picked = twin.with_truth(last, twin.Truth(value={f: chosen.value[f] for f in fields}, source="person_pick", note=f"the person chose {arm}"))
        self.sessions.append(folder, "steps.jsonl", json.loads(picked.model_dump_json()))
        self.sessions.append(folder, "picks.jsonl", {"at": now(), "user": person(), "step": step, "arm": arm, "fields": fields, "id": last.id})
        return {"step": step, "arm": arm, "fields": fields, "winner": picked.verdict.winner}

    def view(self, draft_id: str) -> dict[str, Any]:
        """What the page shows: the settings, Claude's conversation, its last record, and each step's last pair."""
        s = settings(self.home)
        folder = self.sessions.find(f"draft-{draft_id}")
        if folder is None or not folder.is_dir():
            reachable = self.gateway is not None and self.gateway.client("ai_first", draft_id=draft_id) is not None
            status = None if reachable or not s["on"] else {"state": "unavailable", "note": "Claude is not reachable on this server: no model is configured "
                                                            "for it (ANTHROPIC_API_KEY and DCLAB_LLM_PROVIDER=anthropic in .env), or live models are switched off"}
            return {"settings": s, "session": None, "conversation": [], "steps": [], "record": None, "status": status, "cap_eur": s["session_eur"]}
        steps: dict[str, dict[str, Any]] = {}
        for r in self.sessions.rows(folder, "steps.jsonl"):
            steps[r["step"]] = r
        turns = [r for r in self.sessions.rows(folder, "ai_first.jsonl") if r.get("trace")]
        thoughts = self.sessions.rows(folder, "thinking.jsonl")
        manifest = self.sessions.manifest(folder)
        return {"settings": s, "session": manifest, "status": manifest.get("status"), "cap_eur": self.cap(folder, s), "conversation": self.sessions.rows(folder, "conversation.jsonl")[-60:],
                "steps": [steps[k] for k in HOME_STEPS if k in steps], "record": turns[-1]["trace"] if turns else None,
                "thinking": [t for t in (thoughts[-1].get("thinking") if thoughts else []) if t][:6]}


def after_turn(drafts: Any, gateway: Any, home: Path, draft_id: str, event: str, text: Optional[str] = None, to: str = "both",
               wait: bool = False) -> Optional[threading.Thread]:
    """Start Claude's turn in a thread of its own (the request's workspace carried into it), unless the switch is off.
    ``wait`` runs it here (tests, a worker)."""
    if not settings(Path(home))["on"]:
        return None
    twin = HomeTwin(drafts, gateway, Path(home))
    if wait:
        twin.turn(draft_id, event, text, to)
        return None
    ctx = contextvars.copy_context()

    def go() -> None:
        try:
            ctx.run(twin.turn, draft_id, event, text, to)
        except Exception:  # noqa: BLE001 — a failing twin never reaches the person's turn
            pass

    thread = threading.Thread(target=go, name=f"twin-{draft_id}", daemon=True)
    thread.start()
    return thread
