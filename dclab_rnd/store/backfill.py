"""Thinking and reasoning for the past (the owner, 2026-10-28 night: "make sure all previous data has thinking and
reasoning parts, it's crucial"; plan ~/.claude/plans/deep-meandering-hippo.md, Part C).

Thinking that was never saved cannot be recovered; this module never pretends it can. ``plan`` counts the kept calls that
could be asked again with thinking on (whole, with a reply, no thinking text, consent to train, never exam material) and
what it would cost. The paid replay itself waits for its own package (the review of 2026-10-28): it must re-send the
original inputs (the shaped records have their tables removed), keep the original's split and consent, skip sealed
originals, and label its reasoning "replayed" in the contract, never "recorded".

``python -m dclab_rnd.store backfill``
"""

from __future__ import annotations

from typing import Any, Iterable

from dclab_core.store import StoreRecord

from . import shapers
from .layout import Layout

PRICE_IN, PRICE_OUT = 2.0 / 1_000_000, 10.0 / 1_000_000  # USD per token, claude-sonnet-5-5 (the product's model)
# never replayed: the purposes that write or grade the evaluation people (their outputs are exam material, some sealed; a
# replay would put exam-like text into training data) and the teacher's own calls
NOT_REPLAYED = {"journey_writer": "writes the evaluation people (dev and sealed)", "journey_checker": "grades the evaluation people",
                "reasoning_teacher": "a teacher's call is not an agent's work"}
THINKING_OUT = 800  # the output tokens a replay is planned at: its answer and its thinking
REMOVED = "[a table of"  # the shaper's mark where it removed a printed table: those inputs are not the originals


def _thinking_text(body: dict[str, Any]) -> str:
    return "\n\n".join(str(t.get("text") or t.get("thinking") or "") for t in body.get("thinking") or [] if isinstance(t, dict)).strip()


def replayable(records: Iterable[StoreRecord]) -> list[StoreRecord]:
    """Kept calls that could be asked again: whole, with a reply, no thinking text, consent to train, not sealed, not exam
    material, and with their inputs as sent (no table removed by the shaper)."""
    out = []
    for r in records:
        b = r.body
        if not (r.provenance.source.startswith("model_exchanges/") and b.get("whole") and isinstance(b.get("messages"), list)):
            continue
        if (not b.get("reply") and not b.get("tool_calls")) or _thinking_text(b) or r.provenance.consent != "train":
            continue
        if r.split == "sealed" or str(b.get("purpose")) in NOT_REPLAYED or REMOVED in str(b.get("messages")):
            continue
        out.append(r)
    return out


def estimate(records: list[StoreRecord]) -> float:
    tokens_in = sum(int(((r.body.get("usage") or {}).get("input_tokens")) or 2000) for r in records)
    return round(tokens_in * PRICE_IN + len(records) * THINKING_OUT * PRICE_OUT, 2)


def plan(layout: Layout) -> dict[str, Any]:
    todo = replayable(shapers.records(layout, "model_requests"))
    by_purpose: dict[str, int] = {}
    for r in todo:
        by_purpose[str(r.body.get("purpose"))] = by_purpose.get(str(r.body.get("purpose")), 0) + 1
    return {"replayable": len(todo), "replay_usd": estimate(todo), "by_purpose": dict(sorted(by_purpose.items(), key=lambda kv: -kv[1])),
            "note": "the replay itself waits for its own package (inputs as sent, the original's split and consent, a replayed source)"}


def _canon(reply: str, calls: list[Any]) -> str:
    import json
    import re

    norm = lambda c: {"name": (c.get("function") or c).get("name"), "arguments": (c.get("function") or c).get("arguments")}  # noqa: E731
    args = []
    for c in calls or []:
        if isinstance(c, dict):
            n = norm(c)
            a = n["arguments"]
            if isinstance(a, str):
                try:
                    a = json.loads(a or "{}")
                except ValueError:
                    pass
            args.append({"name": n["name"], "arguments": a})
    return json.dumps({"reply": re.sub(r"\s+", " ", str(reply or "")).strip().lower(), "calls": args}, sort_keys=True, default=str)


def _words(text: Any) -> set[str]:
    import re

    return set(re.findall(r"[a-z0-9\u0600-\u06ff]+", str(text or "").lower()))


def _close(a: Any, b: Any) -> bool:
    """Two answer values are the same decision: equal when short (a field, a status, a number), the same keys for objects,
    and for longer text the same wording at least half over (the same words, said a little differently)."""
    import json

    if isinstance(a, str) and isinstance(b, str):
        for x in (a, b):
            if x.strip().startswith("{"):
                try:
                    a2, b2 = json.loads(a), json.loads(b)
                    return _close(a2, b2)
                except ValueError:
                    break
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, str) and isinstance(b, str) and max(len(a), len(b)) > 30:
        wa, wb = _words(a), _words(b)
        return bool(wa | wb) and len(wa & wb) / len(wa | wb) >= 0.5
    return a == b


def same_decision(original: dict[str, Any], again: dict[str, Any]) -> bool:
    """A replay reached the same decision as its original: the same tool calls in order with close arguments, and a close
    reply (the review of the first replays: the same decisions in other words never matched exactly)."""
    import json

    def calls(x: dict[str, Any]) -> list[tuple[Any, Any]]:
        out = []
        for c in x.get("tool_calls") or []:
            if isinstance(c, dict):
                f = c.get("function") or c
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except ValueError:
                        pass
                out.append((f.get("name"), args))
        return out
    oc, nc = calls(original), calls(again)
    if [n for n, _ in oc] != [n for n, _ in nc]:
        return False
    if not all(_close(a, b) for (_, a), (_, b) in zip(oc, nc)):
        return False
    o, n = str(original.get("reply") or ""), str(again.get("reply") or again.get("content") or "")
    return (not o.strip() and not n.strip()) or _close(o, n)


def originals(base, wanted: set[str]) -> dict[str, dict[str, Any]]:
    """The kept exchanges as they were sent (the workspace's own files, never the shaped copies: those have tables
    removed), by their request id, for the ids the store found replayable."""
    import json
    from pathlib import Path

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(base).rglob("model_exchanges/**/*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            rid = str(d.get("model_request") or "")
            if rid in wanted and rid not in out and d.get("whole") and isinstance(d.get("messages"), list):
                out[rid] = d
    return out


def replay(layout: Layout, *, cap_usd: float, yes: bool = False, base=None, gateway: Any = None, limit: int = 1000,
           log: Any = print) -> dict[str, Any]:
    """Ask each replayable call again, with its original messages and tools and thinking on, within one cap for the run.
    Each answer is kept in ``raw/replays/`` beside the original's id, split and consent, with whether it agrees with the
    original (the same reply and tool calls); only an agreeing replay's thinking explains the original (llm.from_exchange
    takes it as "replayed", weight 0.5). The replay's own call is never kept as a new conversation."""
    import json
    import os
    from datetime import datetime, timezone

    todo = {r.body.get("model_request"): r for r in replayable(shapers.records(layout, "model_requests")) if r.body.get("model_request")}
    folder = layout.replays()
    done = {json.loads(x)["model_request"] for p in folder.glob("*.jsonl") for x in p.read_text(encoding="utf-8").splitlines() if x.strip()} \
        if folder.is_dir() else set()
    todo = {k: v for k, v in todo.items() if k not in done}
    from .keepers.workspaces import AGENT_RUNS

    found = originals(base or AGENT_RUNS, set(todo))
    plan_ = {"to_replay": len(found), "already": len(done), "without_original": len(todo) - len(found),
             "estimate_usd": estimate([todo[k] for k in found]), "cap_usd": cap_usd}
    if not yes:
        return {**plan_, "sent": 0, "note": "the plan only: add --yes"}
    from ..models import for_tools

    gateway = gateway or for_tools()
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
    spent, sent, agreed = 0.0, 0, 0
    saved = os.environ.get("DCLAB_LOG_PROMPTS")
    os.environ["DCLAB_LOG_PROMPTS"] = "0"  # the replay's own call keeps its shape only: never a second conversation of the same inputs
    try:
        for rid, original in list(found.items())[:limit]:
            if spent >= cap_usd:
                break
            client = gateway.client(str(original.get("purpose")))
            if client is None:
                continue
            client.run_limit_eur = max(0.0, (cap_usd - spent) * 0.92)  # USD cap to EUR, rounded down
            try:
                reply = client.complete(original["messages"], original.get("tools") or None)
            except Exception as error:  # noqa: BLE001 - a failed replay is recorded, the run goes on
                log(f"replay {rid}: {type(error).__name__}")
                continue
            cost = float(getattr(client, "spent_eur", 0.0) or 0.0) / 0.92
            spent += cost
            sent += 1
            thinking = [t for t in (reply.get("thinking") or []) if isinstance(t, dict)]
            agree = same_decision(original, {"reply": reply.get("content") or "", "tool_calls": reply.get("tool_calls") or []})
            agreed += agree
            rec = todo[rid]
            line = {"model_request": rid, "purpose": original.get("purpose"), "original_at": original.get("at"),
                    "replayed_at": datetime.now(timezone.utc).isoformat(), "model": reply.get("model") or original.get("model"),
                    "thinking": thinking, "reply": reply.get("content") or "", "tool_calls": reply.get("tool_calls") or [], "agree": bool(agree),
                    "original_reply": original.get("reply") or "", "original_tool_calls": original.get("tool_calls") or [],  # agreement can be judged again
                    "split": rec.split, "consent": rec.provenance.consent, "cost_usd": round(cost, 5)}
            with out_path.open("a", encoding="utf-8") as fh:  # appended at once: a stopped run keeps what it paid for
                fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    finally:
        if saved is None:
            os.environ.pop("DCLAB_LOG_PROMPTS", None)
        else:
            os.environ["DCLAB_LOG_PROMPTS"] = saved
    return {**plan_, "sent": sent, "agreed": agreed, "spent_usd": round(spent, 4), "kept": str(out_path)}


def replays(layout: Layout) -> dict[str, dict[str, Any]]:
    """The agreeing replays' thinking by the original's request id (llm.from_exchange reads it as "replayed")."""
    import json
    from pathlib import Path

    folder = layout.replays()
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.jsonl")) if folder.is_dir() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            text = "\n\n".join(str(t.get("text") or t.get("thinking") or "") for t in d.get("thinking") or []).strip()
            agree = same_decision({"reply": d.get("original_reply"), "tool_calls": d.get("original_tool_calls")}, d) \
                if "original_reply" in d else d.get("agree")  # judged again with today's rule when the original is kept beside it
            if agree and text and d.get("split") != "sealed":
                out[str(d["model_request"])] = {"text": text, "model": d.get("model")}
    return out


__all__ = ["replayable", "estimate", "plan", "replay", "replays", "originals", "NOT_REPLAYED"]
