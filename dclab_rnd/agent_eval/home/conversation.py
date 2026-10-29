"""Home's understanding conversation, measured (the owner's Telco test, 2026-10-29). The Home eval's person only says
"I'm not sure" to a structured question, so a model-led conversation could not be measured with it. Here each written
person (``personas_dev.json``, by an agent shown only the format) opens vaguely, shares a table or not, and answers Home's
open questions from their hidden facts (situation, goal, data, decision), matched to what Home asked. When Home offers
options, the person picks the one that fits their right route, or says "Something else".

Scored, per person and in total: the right route (and the segmentation tool for new groups), the turns before the options,
whether the right route was among them, silent turns (a message of theirs with no reply), a question repeated word for
word, numbers no one gave (cited.py, as the Home eval reads them) and quotes that are not their words.

    python -m dclab_rnd.agent_eval.home.conversation                          # the plan; nothing is sent
    python -m dclab_rnd.agent_eval.home.conversation --yes --cap-eur 2        # with the configured model (thinking kept)
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

HERE = Path(__file__).resolve().parent
PERSONAS = HERE / "personas_dev.json"
SPLITS = {"dev": PERSONAS, "sealed_v1": HERE / "personas_sealed_v1.json"}  # sealed: written by an agent shown the format alone
SEALED = {"sealed_v1": HERE / "personas_sealed_v1.sha256"}  # pinned before its one run
MAX_TURNS = 10
TOPICS = (  # what Home's question is about -> the fact that answers it
    ("goal", re.compile(r"\b(?:goal|achieve|hoping|hope|want|trying|improve|change|end up|outcome|success|aim)\b", re.I)),
    ("decision", re.compile(r"\b(?:decid\w*|decision|act|action|do with|use (?:it|this|the result)|next step)\b", re.I)),
    ("data", re.compile(r"\b(?:data|table|file|spreadsheet|records?|columns?|information|track|collect|have about)\b", re.I)),
    ("situation", re.compile(r"\b(?:situation|business|what happened|tell me (?:more|a bit)|context|problem|going on)\b", re.I)),
)
DONE = "That's all I know, really."
# the owner's own session (2026-10-29), a permanent case beside the written people: their words, the table they shared (the
# eval's customers table stands for Telco: an outcome-like yes/no column and no date to compare periods on)
OWNER = {"id": "owner-telco", "opening": "how can I analyze my customer", "upload": "customers", "route": "prediction",
         "tool": "segmentation_project", "facts": {"goal": "show my customer segments", "data": "it is my customer churn table",
                                                   "decision": "I want to treat each group of customers differently",
                                                   "situation": "we are a telecom company and customers leave us"}}
# the gate, written 2026-10-29 before any run of the conversation with the fixed code (dev; a sealed set of people is read once)
GATE = {"version": "home_conversation_gate_v1", "route_accuracy": 0.80, "right_route_offered": 0.90, "silent_turns": 0,
        "repeated_questions": 0, "uncited": 0, "quote_invalid": 0, "median_turns_before_options": (2, 3)}


def verdict(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The gate on a run: every check, and pass or fail."""
    people = max(int(summary.get("people") or 0), 1)
    low, high = GATE["median_turns_before_options"]
    checks = {"route_accuracy": summary["route_accuracy"] >= GATE["route_accuracy"],
              "right_route_offered": summary["right_route_offered"] / people >= GATE["right_route_offered"],
              **{k: summary[k] <= GATE[k] for k in ("silent_turns", "repeated_questions", "uncited", "quote_invalid")},
              "median_turns_before_options": summary["median_turns_before_options"] is not None
                                             and low <= summary["median_turns_before_options"] <= high}
    return {"gate": GATE["version"], "status": "pass" if all(checks.values()) else "fail", "checks": checks}


def load(path: Path = PERSONAS, owner: bool = True) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8")) + ([OWNER] if owner else [])


def answer_for(question: str, facts: Mapping[str, str], used: set[str]) -> str:
    """The person's reply to an open question: the fact it asks about, else the next fact they have not said yet."""
    for topic, pattern in TOPICS:
        if topic not in used and facts.get(topic) and pattern.search(question):
            used.add(topic)
            return str(facts[topic])
    for topic in ("situation", "goal", "data", "decision", "other"):
        if topic not in used and facts.get(topic):
            used.add(topic)
            return str(facts[topic])
    return DONE


def pick(options: list[dict[str, Any]], route: str, tool: str) -> Optional[str]:
    """The option a person with this route takes: the right tool first, then the right kind of help."""
    for o in options:
        if o.get("tool") == tool and o.get("track") == route:
            return o["title"]
    if tool != "segmentation_project":
        for o in options:
            if o.get("track") == route and o.get("tool") != "segmentation_project":
                return o["title"]
    return None


def play(persona: Mapping[str, Any], understand_client: Any, intent_client: Any, store: Any) -> dict[str, Any]:
    """One person, start to route; returns the scores and the transcript's shape (never sent anywhere)."""
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft import understand as U
    from dclab_rnd.draft.chat import HomeAgent

    from . import read_draft, tables

    d = store.create(str(persona["opening"]))
    agent = HomeAgent(store, None, on_request=lambda *a: None, intent_client=intent_client, understand_client=understand_client)
    agent.start(d["id"])
    if persona.get("upload"):
        name = tables.write(str(persona["upload"]), store.data_dir(d["id"]))
        asset = pipeline.new_asset(store, d["id"], "upload", name, name)
        pipeline.run(store, d["id"], asset["id"], agent, None)
    used: set[str] = set()
    silent = turns_before = offers = 0
    right_offered = False
    asked: list[str] = []
    for _ in range(MAX_TURNS):
        draft = store.get(d["id"])
        if (draft.get("intent") or {}).get("status") not in ("asked", None) and not agent.pending(draft):
            break  # routed, nothing pending: the conversation's job is done
        q = agent.pending(draft)
        if q is None:
            break
        asked.append(str(q.get("text") or ""))
        if q.get("offer"):
            offers += 1
            choice = pick(q["offer"], str(persona["route"]), str(persona["tool"]))
            right_offered = right_offered or choice is not None
            reply = choice or U.SOMETHING_ELSE
        elif q.get("understand"):
            turns_before += 0 if offers else 1
            reply = answer_for(str(q.get("text") or ""), persona.get("facts") or {}, used)
        elif q.get("field") == "segments_route":
            reply = "Find new groups in the data" if persona["tool"] == "segmentation_project" else "Describe groups I already have"
        else:
            reply = "I'm not sure"
        before = len(store.get(d["id"])["messages"])
        agent.reply(d["id"], reply)
        after = store.get(d["id"])["messages"]
        if len(after) <= before + 1 or after[-1].get("role") == "user":
            silent += 1
    draft = store.get(d["id"])
    intent = draft.get("intent") or {}
    track = None if intent.get("status") == "asked" else intent.get("track")
    family = (draft.get("understanding") or {}).get("family")
    right = track == persona["route"] and (persona["tool"] != "segmentation_project" or family == "segmentation")
    seen = read_draft(draft, set(), set())
    return {"id": persona["id"], "route": persona["route"], "tool": persona["tool"], "upload": persona.get("upload"), "track": track,
            "family": family, "right": right, "turns_before_options": turns_before, "offers": offers, "right_offered": right_offered,
            "silent": silent, "repeated": len(asked) - len(set(asked)), "uncited": seen["uncited"], "quote_invalid": seen["quote_invalid"],
            "fallback": bool((draft.get("understand") or {}).get("failed")), "refused_options": len((draft.get("understand") or {}).get("refused") or [])}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(len(rows), 1)
    return {"people": len(rows), "route_right": sum(r["right"] for r in rows), "route_accuracy": round(sum(r["right"] for r in rows) / n, 3),
            "right_route_offered": sum(r["right_offered"] for r in rows),
            "median_turns_before_options": statistics.median([r["turns_before_options"] for r in rows]) if rows else None,
            "max_turns_before_options": max((r["turns_before_options"] for r in rows), default=None),
            "silent_turns": sum(r["silent"] for r in rows), "repeated_questions": sum(r["repeated"] for r in rows),
            "uncited": sum(len(r["uncited"]) for r in rows), "quote_invalid": sum(len(r["quote_invalid"]) for r in rows),
            "fallbacks": sum(r["fallback"] for r in rows), "refused_options": sum(r["refused_options"] for r in rows)}


def measure(personas: list[dict[str, Any]], understand_for: Callable[[str], Any], intent_for: Callable[[str], Any],
            keep: Optional[Path] = None) -> dict[str, Any]:
    import tempfile

    from dclab_rnd.draft.store import DraftStore

    from . import _isolated, mark_scripted

    rows = []
    with _isolated():
        for persona in personas:
            folder = (keep / str(persona["id"])) if keep else Path(tempfile.mkdtemp())
            store = DraftStore(folder / "drafts")
            rows.append(play(persona, understand_for(str(persona["id"])), intent_for(str(persona["id"])), store))
    if keep:
        mark_scripted(keep, "written people answered by a script from their hidden facts: their acts are a script's, never a person's label")
    return {"rows": rows, "summary": summarize(rows)}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    parser.add_argument("--only", default=None, help="a comma list of person ids (dev only)")
    parser.add_argument("--split", choices=tuple(SPLITS), default="dev")
    args = parser.parse_args(argv)
    root = HERE.parents[2]
    results = root / "evidence" / "campaigns" / "home_v1" / "results"
    if args.split in SEALED:
        import hashlib

        if args.only:
            parser.error("--only is for dev: a sealed set is run once, whole")
        pinned = SEALED[args.split].read_text(encoding="utf-8").strip()
        if hashlib.sha256(SPLITS[args.split].read_bytes()).hexdigest() != pinned:
            print("refused: the sealed set does not match its pinned fingerprint", file=sys.stderr)
            return 2
        if args.yes and list(results.glob(f"HOMECONV-*_{args.split}.json")):
            print(f"refused: {args.split} was already run; a sealed set is run once", file=sys.stderr)
            return 2
        from dclab_rnd import experience as _experience

        _experience.sealed_run(args.split)  # an exam's model calls keep their shape only (the owner's rule)
    people = load(SPLITS[args.split], owner=args.split == "dev")
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        people = [p for p in people if p["id"] in wanted]
    print(json.dumps({"people": len(people), "per_person_calls": "1 intent reading + up to 4 understanding turns",
                      "estimate_eur": round(len(people) * 5 * 0.012, 2)}, indent=1))
    if not args.yes:
        print("nothing was sent: add --yes and --cap-eur")
        return 0
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: --cap-eur above zero is required", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(HERE.parents[2] / ".env", override=False)
    from dclab_rnd.models.gateway import for_workspace

    from .__main__ import Budget

    gateway = for_workspace()
    gateway.shadows = None
    budget = Budget(gateway, args.cap_eur)
    keep = None if args.split in SEALED else \
        root / "agent_runs" / "live_tests" / f"{datetime.now(timezone.utc):%Y-%m-%d}-home-conversation" / f"dev-{datetime.now(timezone.utc):%H%M%S}"  # an exam is never kept
    result = measure(people, lambda pid: budget.client(f"conv-{pid}", "home_understand"), lambda pid: budget.intent(f"conv-{pid}"), keep)
    n = 1 + max([int(m.group(1)) for p in results.glob("HOMECONV-*.json") if (m := re.match(r"HOMECONV-(\d+)", p.name))] or [0])
    record = {"id": f"HOMECONV-{n:03d}", "split": args.split, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model": getattr(budget.clients[0], "model", None) if budget.clients else None, "spent_eur": round(budget.spent_eur(), 4),
              "kept": str(keep.relative_to(root)) if keep else None, **result, "verdict": verdict(result["summary"])}
    path = results / f"HOMECONV-{n:03d}_{args.split}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=1))
    print(json.dumps(verdict(result["summary"]), indent=1))
    print(f"spent €{budget.spent_eur():.4f}; stored {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
