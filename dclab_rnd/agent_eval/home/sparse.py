"""36.3: written people for the kinds of help that have no runs yet (analysis, preparation, reporting, business; the
coverage report of 2026-10-28 counted 0 runs and 0 examples for each), played through Home with its model clients, so
those tracks get whole conversations with the model's thinking kept.

Dev only, English only (the owner, 2026-10-28), never a sealed set. Each person has a request, a table, the answers they
give Home's questions and the follow-up messages they send after the first result. They never click: no save, no
download, no thumb, so no verdict of theirs is written; Home's choices are what is learned. Every draft is kept whole
under ``agent_runs/live_tests/<date>-home-sparse/<time>/`` in a workspace layout (drafts and ledger) with
``scripted.json``: their acts are a script's, never a person's label. The gateway keeps each model call with its thinking.

    python -m dclab_rnd.agent_eval.home.sparse                       # the plan only
    python -m dclab_rnd.agent_eval.home.sparse --yes --cap-eur 0.5

The people were written by the session that wrote Home's rules, from the test tables' columns: they measure nothing (no
score is read from them); they are training conversations, silver and apart.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[3]
KEEP = ROOT / "agent_runs" / "live_tests"
WHY = "36.3: written people played through Home (dev, English): their answers and follow-ups are a script, never a person's label"
NOT_SURE = "I'm not sure"


def _p(id_: str, track: str, table: str, text: str, replies: tuple[str, ...] = (), then: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"id": id_, "track": track, "table": table, "text": text, "replies": list(replies), "then": list(then)}


PEOPLE = [
    # analysis: a question about their own numbers
    _p("s-a1", "analysis", "customers", "Our retention team wants to know which plan loses the biggest share of its customers next month.",
       ("Churn is churned_next_month.",), ("And split by segment as well?",)),
    _p("s-a2", "analysis", "sales", "Which product categories bring in the most money in each region?",
       ("Total amount, only paid orders.",), ("Now just the app channel.",)),
    _p("s-a3", "analysis", "bookings", "How many days ahead do travellers usually book, for each destination?",
       ("The median lead time is fine.",), ("Is it different when the booking has free cancellation?",)),
    _p("s-a4", "analysis", "customers", "Do customers with more support tickets spend more per month?",
       ("Average monthly spend by number of tickets.",)),
    # preparation: a table made ready for them
    _p("s-p1", "preparation", "sales", "Clean up my orders file: drop the cancelled orders and remove repeated order IDs.",
       ("Yes, go ahead.",), ("Can I have it as Excel?",)),
    _p("s-p2", "preparation", "customers", "I need the customer list without the city column, only people on the premium plan.",
       ("Yes.",), ("CSV is fine.",)),
    _p("s-p3", "preparation", "bookings", "Give me only the bookings with a price above 500 and drop the nationality column.",
       ("Yes, apply both.",)),
    _p("s-p4", "preparation", "sales", "Keep just the refunded orders from Dubai so I can send them to the finance team.",
       ("Yes.",), ("Parquet please.",)),
    # reporting: numbers to follow over time
    _p("s-r1", "reporting", "sales", "Set up a monthly dashboard for our finance lead with revenue and number of orders per channel.",
       ("Revenue and orders.", "The finance lead.")),
    _p("s-r2", "reporting", "bookings", "I want a weekly report of bookings and the cancellation rate by destination for the operations manager.",
       ("Bookings and cancellation rate.",)),
    _p("s-r3", "reporting", "customers", "Build a KPI board for the board meeting: churn rate and average monthly spend by segment, refreshed every month.",
       ("Churn rate and average monthly spend.",)),
    _p("s-r4", "reporting", "sales", "Can you make me a daily dashboard of refunds by region?",
       ("Number of refunded orders.",)),
    # business: why a number moved, what to do about it
    _p("s-b1", "business", "sales", "Revenue dropped from January to February 2026. Which regions explain the drop?",
       ("January 2026 against February 2026.",)),
    _p("s-b2", "business", "sales", "Were sales in the first quarter of 2026 weaker than in the last quarter of 2025? Where is the difference?",
       ("Q4 2025 is October to December 2025; Q1 2026 is January to March 2026.",)),
    _p("s-b3", "business", "bookings", "Cancellations went up from summer 2025 to winter. Where is it coming from?",
       ("Summer is June to August 2025; winter is December 2025 to February 2026.",)),
    _p("s-b4", "business", "sales", "Should we move budget from the store channel to the app? Compare March 2026 with March 2025.",
       ("March 2025 against March 2026, total amount.",)),
]
TOOLS = ("query_data", "make_chart", "save_answer", "propose_step", "undo_step", "export_data", "compare_periods", "decision_brief")


def _answer_questions(agent: Any, store: Any, draft_id: str, replies: list[str], limit: int = 4) -> None:
    """Home's pending questions answered in the person's own words, then "I'm not sure"; a guarded step gets a whole yes."""
    for _ in range(limit):
        q = agent.pending(store.get(draft_id))
        if q is None:
            return
        agent.reply(draft_id, "Yes" if q.get("field") == "confirm_step" else (replies.pop(0) if replies else NOT_SURE))


def play(people: list[dict[str, Any]], keep: Path, client: Callable[[str, str], Any]) -> list[dict[str, Any]]:
    """Each person through Home in a throwaway draft store, copied whole into ``keep`` (drafts and ledger, marked
    scripted). ``client(purpose, draft_id)`` gives Home's model clients (None: the rules alone). One person that fails is
    reported and the others go on."""
    from dclab_rnd import ledger
    from dclab_rnd.agent_eval.home import mark_scripted, tables
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore

    keep = Path(keep)
    mark_scripted(keep, WHY)
    before = ledger.current_ledger()
    ledger.install(ledger.FileLedger(keep / "ledger", keep.name))  # every decision Home makes, kept for the decision models
    rows: list[dict[str, Any]] = []
    try:
        with tempfile.TemporaryDirectory(prefix="dclab-home-sparse-") as tmp:
            store = DraftStore(Path(tmp) / "drafts")
            for p in people:
                d = store.create(p["text"])
                agent = HomeAgent(store, client("home_agent", d["id"]), on_request=lambda *a: None,
                                  intent_client=client("home_intent", d["id"]), query_client=client("home_query", d["id"]),
                                  business_client=client("home_business", d["id"]))
                read, error = None, None
                try:
                    agent.start(d["id"])
                    read = agent.track(store.get(d["id"]))
                    if read != p["track"]:
                        agent.choose_intent(d["id"], p["track"])  # the person's own track: what these conversations are for
                    name = tables.write(p["table"], store.data_dir(d["id"]))
                    asset = pipeline.new_asset(store, d["id"], "upload", name, name)
                    pipeline.run(store, d["id"], asset["id"], agent, None)
                    replies = list(p["replies"])
                    _answer_questions(agent, store, d["id"], replies)
                    for message in p["then"]:  # the person's next message after the first result
                        agent.reply(d["id"], message)
                        _answer_questions(agent, store, d["id"], replies)
                except Exception as exc:  # noqa: BLE001 - a person that fails is reported; the others go on
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"
                done = store.get(d["id"])
                rows.append({"id": p["id"], "track": agent.track(done), "read": read, "error": error,
                             "answers": len(done.get("answers") or []), "prepared": bool((done.get("recipe") or {}).get("done")),
                             "brief": bool((done.get("business") or {}).get("result")),
                             "dashboard": bool((done.get("dashboard") or {}).get("board")),
                             "turns": sum(1 for m in done.get("messages") or [] if m.get("role") == "agent")})
            shutil.copytree(Path(tmp) / "drafts", keep / "drafts", dirs_exist_ok=True)
    finally:
        ledger.install(before)
    return rows


def _thinking(since: str) -> tuple[int, int]:
    """Model calls made since ``since`` that the gateway kept, and how many of them carry thinking text."""
    calls = thought = 0
    for path in sorted((ROOT / "agent_runs" / "model_exchanges").rglob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if str(d.get("at") or "") >= since and str(d.get("purpose") or "").startswith("home_"):
                calls += 1
                thought += int(any((t.get("text") or t.get("thinking") or "").strip()
                                   for t in (d.get("thinking") or d.get("thinking_redacted") or []) if isinstance(t, dict)))
    return calls, thought


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.home.sparse")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    tracks = sorted({p["track"] for p in PEOPLE})
    print(f"{len(PEOPLE)} written people (English, dev) on {', '.join(tracks)}; Home with its model clients; "
          f"capped at EUR {args.cap_eur or 0:.2f}")
    if not args.yes or not args.cap_eur or args.cap_eur <= 0:
        print("nothing was sent: add --yes and --cap-eur")
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    from dclab_rnd.models.gateway import mark_invented_tables
    mark_invented_tables(gateway, "home sparse eval")  # every table here is invented by code
    gateway.shadows = None
    clients: list[Any] = []

    def client(purpose: str, draft_id: str) -> Any:
        c = gateway.client(purpose, draft_id=draft_id)
        if c is not None:
            c.run_limit_eur = max(0.0, args.cap_eur - sum(float(getattr(x, "spent_eur", 0.0) or 0.0) for x in clients))
            clients.append(c)
        return c

    now = datetime.now(timezone.utc)
    keep = KEEP / f"{now:%Y-%m-%d}-home-sparse" / f"{now:%H%M%S}"
    since = now.isoformat()
    rows = play(PEOPLE, keep, client)
    spent = sum(float(getattr(x, "spent_eur", 0.0) or 0.0) for x in clients)
    calls, thought = _thinking(since)
    report = {"at": datetime.now(timezone.utc).isoformat(), "spent_eur": round(spent, 4), "model_calls": calls,
              "with_thinking": thought, "rows": rows, "kept": str(keep.relative_to(ROOT))}
    (keep / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    print(f"spent EUR {spent:.4f}; {calls} model calls, {thought} with thinking text; kept {report['kept']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
