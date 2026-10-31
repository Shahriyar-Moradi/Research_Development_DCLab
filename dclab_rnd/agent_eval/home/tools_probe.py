"""A live probe of Home's model tools (packages H7c, H9b, H11b): requests worded so the rules cannot read them, so the
model must use query_data, propose_step or compare_periods. Dev only (never the sealed set), English only (the owner,
2026-10-28), capped, every draft kept whole under ``agent_runs/live_tests/<date>-home-tools/`` and every model call kept by
the gateway with its thinking.

    python -m dclab_rnd.agent_eval.home.tools_probe            # the plan only
    python -m dclab_rnd.agent_eval.home.tools_probe --yes --cap-eur 0.5

What it reports per request: the kind of help read, the model tools called, whether the person got an answer, a prepared
table or a brief, and how many model calls carried thinking text. It measures that the paths work live, not a score.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
KEEP = ROOT / "agent_runs" / "live_tests"

# (id, the request, the table, the track it is, what the person says when Home cannot read it)
PROBES = [
    ("t-a1", "For each plan, what's the typical monthly outlay of our clients?", "customers", "analysis",
     "Please work it out from the table."),
    ("t-a2", "How long do business clients usually stay with us compared with consumers?", "customers", "analysis",
     "Use the tenure in months, per segment."),
    ("t-p1", "Tidy my orders up: get rid of repeated order numbers and throw away the channel column.", "sales", "preparation",
     "Yes, please do those two things."),
    ("t-p2", "I only want orders worth more than 100 left in the file, as Excel.", "sales", "preparation",
     "Keep rows where amount is above 100, then give me Excel."),
    ("t-b1", "Revenue went down from February to March this year - where did it come from?", "sales", "business",
     "This year is 2026."),
    ("t-b2", "Were spring 2026 sales weaker than the winter before? Show me where the difference sits.", "sales", "business",
     "Spring is March to May 2026; winter is December 2025 to February 2026."),
]
WHY = "the Home tools probe: written requests and replies, the kind of help chosen by the probe: a script's acts, never a person's label"
TOOLS = ("query_data", "save_answer", "propose_step", "undo_step", "export_data", "compare_periods", "decision_brief")


def _exchanges(draft_id: str, since: str) -> list[dict[str, Any]]:
    """This draft's model calls as the gateway kept them (with their thinking), made since the probe began."""
    out = []
    for path in sorted((ROOT / "agent_runs" / "model_exchanges").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("draft") == draft_id and str(d.get("at") or "") >= since:
                out.append(d)
    return out


def _thought(d: dict[str, Any]) -> bool:
    return any((t.get("text") or t.get("thinking") or "").strip() for t in (d.get("thinking") or d.get("thinking_redacted") or [])
               if isinstance(t, dict))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.home.tools_probe")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    print(f"{len(PROBES)} dev requests (English), each a draft with Home's model clients; capped at €{args.cap_eur or 0:.2f}")
    if not args.yes or not args.cap_eur or args.cap_eur <= 0:
        print("nothing was sent: add --yes and --cap-eur")
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd.agent_eval.home import tables
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    from dclab_rnd.models.gateway import mark_invented_tables
    mark_invented_tables(gateway, "home tools probe")  # every table here is invented by code
    gateway.shadows = None
    clients: list[Any] = []

    def client(purpose: str, draft_id: str) -> Any:
        c = gateway.client(purpose, draft_id=draft_id)
        if c is not None:
            c.run_limit_eur = max(0.0, args.cap_eur - sum(float(getattr(x, "spent_eur", 0.0) or 0.0) for x in clients))
            clients.append(c)
        return c

    keep = KEEP / f"{datetime.now(timezone.utc):%Y-%m-%d}-home-tools" / f"{datetime.now(timezone.utc):%H%M%S}"
    from dclab_rnd.agent_eval.home import mark_scripted

    mark_scripted(keep, WHY)  # the probe picks the kind of help itself: never a person's label
    from dclab_rnd import ledger

    before = ledger.current_ledger()
    ledger.install(ledger.FileLedger(keep / "ledger", keep.name))  # every decision Home makes in the probe, kept for the decision models
    rows = []
    since = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="dclab-home-tools-") as tmp:
        store = DraftStore(Path(tmp) / "drafts")
        for pid, text, table, track, reply in PROBES:
            d = store.create(text)
            agent = HomeAgent(store, client("home_agent", d["id"]), on_request=lambda *a: None,
                              intent_client=client("home_intent", d["id"]), query_client=client("home_query", d["id"]),
                              business_client=client("home_business", d["id"]))
            try:
                agent.start(d["id"])
                if agent.track(store.get(d["id"])) != track:
                    agent.choose_intent(d["id"], track)  # the probe is about the tools, not the reading of the kind
                name = tables.write(table, store.data_dir(d["id"]))
                asset = pipeline.new_asset(store, d["id"], "upload", name, name)
                pipeline.run(store, d["id"], asset["id"], agent, None)
                for _ in range(3):  # the brief's questions: the person answers with their own words
                    q = agent.pending(store.get(d["id"]))
                    if q is None:
                        break
                    agent.reply(d["id"], reply if q.get("field") != "confirm_step" else "Yes, apply it")
                done = store.get(d["id"])
                if not (done.get("answers") or (done.get("recipe") or {}).get("done") or (done.get("business") or {}).get("result")):
                    agent.reply(d["id"], reply)
                    q = agent.pending(store.get(d["id"]))
                    if q is not None and q.get("field") == "confirm_step":
                        agent.reply(d["id"], "Yes, apply it")
                error = None
            except Exception as exc:  # noqa: BLE001 - a probe that fails is reported, the others go on
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
            done = store.get(d["id"])
            calls = _exchanges(d["id"], since)
            rows.append({"id": pid, "track": agent.track(done), "error": error,
                         "tools": [c.get("name") for x in calls for c in x.get("tool_calls") or [] if c.get("name") in TOOLS],
                         "model_calls": {p: sum(1 for x in calls if x.get("purpose") == p) for p in sorted({x.get("purpose") for x in calls})},
                         "with_thinking": sum(1 for x in calls if _thought(x)),
                         "answered": bool(done.get("answers")), "prepared": bool((done.get("recipe") or {}).get("done")),
                         "brief": bool((done.get("business") or {}).get("result")),
                         "brief_by": ((done.get("business") or {}).get("worded") or {}).get("by"),
                         "answer_by": [a.get("by") for a in done.get("answers") or []]})
        shutil.copytree(Path(tmp) / "drafts", keep / "drafts", dirs_exist_ok=True)
    ledger.install(before)
    spent = sum(float(getattr(x, "spent_eur", 0.0) or 0.0) for x in clients)
    report = {"at": datetime.now(timezone.utc).isoformat(), "spent_eur": round(spent, 4), "rows": rows, "kept": str(keep.relative_to(ROOT))}
    (keep / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(f"spent €{spent:.4f}; kept {report['kept']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
