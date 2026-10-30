"""Claude's routing labels (the owner, 2026-10-29: "you can use Claude labels"): Home's routing decisions have almost no
person's label yet, so Claude reads each unlabelled one blind and its answer becomes a silver label of the m7_routing set.

What Claude sees: the person's words, the decision's question, its options with their meanings, and the state's bands. It
never sees what Home, the rules, a model or our own classifier answered. Its answer is written to the store's
``labels/m7_routing_claude.jsonl``, by example id, never into a ledger: it stays apart from every person's label, and the
routing set reads it only where no person's or written label stands (class ``silver``, weight 0.5). Every request goes
through the gateway's ``routing_labeler`` purpose, which keeps the exchange and the API's thinking summary; this file keeps
the answer and the exchange's request id beside it; the label's reason is that exchange's API thinking summary, never a
field asked for in the reply (the owner's rule; the first 305 labels, 2026-10-29, also carry a "reason" field, kept as written).

Exams are never labelled (the sealed split is not read). Scripted examples (a test's click) are labelled only up to
``--scripted`` per point, so a census of test traffic does not spend the budget.

    python -m dclab_rnd.store.routing_labels                       # the plan: how many per point, the estimate
    python -m dclab_rnd.store.routing_labels --yes --cap-eur 3     # send; stops at the cap, safe to run again
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .layout import Layout

MODEL_TYPE = "m7_routing"
POINTS = ("request.intent", "request.exit", "request.suggestion", "request.tool", "problem.pack", "workflow.next_question")
FILE = "m7_routing_claude.jsonl"
SEED = 29
EUR_PER_CALL = 0.012  # Sonnet 5.5 at effort xhigh on a short labelling request, measured on decision_labeler (2026-10-29)

FIELD_MEANINGS = {  # workflow.next_question: what each question asks the person
    "target": "what the model should predict (the outcome)", "prediction_moment": "when each prediction is made",
    "action": "what is done with each prediction", "costs": "the cost of a wrong alarm against a missed case",
    "data_plan": "which data they have or will bring", "entity": "what one row is about (a customer, an order ...)",
    "labels": "whether past outcomes are recorded", "budget": "how many cases can be acted on", "family": "the kind of model problem",
    "kind": "the kind of problem", "none": "no question is needed now"}
EXIT_MEANINGS = {"ml_project": "a prediction model built as a project", "analysis": "answers from their data in the chat",
                 "dataset": "a prepared table to download", "dashboard": "a dashboard kept up to date",
                 "decision_brief": "a business decision brief", "answer": "a cited answer, no data work"}
TOOL_MEANINGS = {"ml_project": "a prediction model project", "query": "the query engine: numbers per group from their table",
                 "segmentation_project": "a segmentation project that finds new groups", "recipe": "a preparation recipe",
                 "dashboard": "a dashboard", "business_brief": "a comparison of two periods as a decision brief",
                 "cited_answer": "an answer cited from the evidence"}


def _meanings(point: str, options: list[str]) -> dict[str, str]:
    from dclab_core.routing_example import TRACK_MEANINGS

    if point in ("request.intent", "request.suggestion"):
        table = {**TRACK_MEANINGS, "none": "no other kind of help than the one Home is on"}
    elif point == "request.exit":
        table = EXIT_MEANINGS
    elif point == "request.tool":
        table = TOOL_MEANINGS
    elif point == "workflow.next_question":
        table = FIELD_MEANINGS
    else:
        table = {}
    return {o: table.get(o, "") for o in options}


def _options(point: str, example: dict[str, Any]) -> list[str]:
    options = [str(o) for o in example.get("options") or []]
    if not options and point == "problem.pack":  # the pack point records no options: the packs Home offers
        from ..draft.pack import PACKS

        options = [str(p["key"]) for p in PACKS]
    return options


# What each fact of a decision means (2026-10-30, the next-question study: "labels: unknown" means no separate file of cases
# was uploaded, and Claude read it as "nobody said whether outcomes are recorded")
FACT_MEANINGS = {
    "labels": "file: the cases come as a separate file the person uploaded; unknown: no such file (this says nothing about whether past outcomes are recorded)",
    "open_questions": "how many of the three essentials (outcome, prediction moment, action) are still unanswered",
    "step": "how many questions Home has asked so far", "steps_left": "how many questions Home may still ask in this conversation",
    "outcome_known": "the outcome to predict is settled", "moment_known": "when each prediction is made is settled",
    "action_known": "what is done with each prediction is settled", "costs_known": "the cost of a miss and of a false alarm is settled",
    "data_plan_known": "where the data comes from is settled", "entity_known": "the unit each prediction is about is settled",
    "labels_known": "whether past outcomes are recorded is settled", "budget_known": "how many flags a team can review is settled",
    "family_known": "the kind of problem is settled", "declined": "how many questions the person said they could not answer",
    "pack": "the domain pack Home chose",
}


def view(example: dict[str, Any]) -> dict[str, Any]:
    """What Claude is shown: the words, the question, the options and their meanings, the state's bands. Never a reading."""
    from dclab_core.ledger import POINTS as CORE

    point = str(example["point"])
    options = _options(point, example)
    meanings = _meanings(point, options)
    if point == "problem.pack":
        from ..draft.pack import PACKS

        meanings = {str(p["key"]): str(p.get("title") or "") for p in PACKS}
    return {"request": example.get("text"), "decision": CORE[point].about if point in CORE else point,
            "options": [{"option": o, "means": meanings.get(o, "")} for o in options], "facts": example.get("state") or {},
            "facts_mean": {k: FACT_MEANINGS[k] for k in (example.get("state") or {}) if k in FACT_MEANINGS}}


def pending(layout: Layout, scripted_per_point: int = 40, points: tuple[str, ...] = POINTS) -> list[dict[str, Any]]:
    """The examples to label: train and dev (the sealed split is never read), with words, with no label, not labelled by
    Claude before; every real, written or eval one, and a seeded sample of scripted ones per point."""
    done = labelled(layout)
    out: list[dict[str, Any]] = []
    rng = random.Random(SEED)
    for point in points:
        base = layout.sets(MODEL_TYPE, point)
        versions = sorted(base.glob("v*")) if base.exists() else []
        if not versions:
            continue
        rows = []
        for split in ("train", "dev"):
            path = versions[-1] / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
                ex = json.loads(line)
                if ex.get("label") is None and ex.get("text") and not ex.get("exam") and ex["id"] not in done and _options(point, ex):
                    rows.append(ex)
        scripted = [x for x in rows if x.get("origin") == "scripted"]
        rng.shuffle(scripted)
        out += [x for x in rows if x.get("origin") != "scripted"] + scripted[:scripted_per_point]
    return out


def labelled(layout: Layout) -> set[str]:
    path = layout.labels(FILE)
    return {json.loads(line).get("example") for line in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()


def run(layout: Layout, ask: Callable[[dict[str, Any]], tuple[Optional[dict[str, Any]], dict[str, Any]]], examples: list[dict[str, Any]],
        model: str) -> dict[str, int]:
    """Ask about each example and append its label line. ``ask`` returns (parsed reply, meta: request id, tokens, reply text).
    Stops at the first error that says the cap or the provider stopped it, keeping what was written."""
    path = layout.labels(FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"asked": 0, "labelled": 0, "unsure": 0, "refused": 0}
    for ex in examples:
        shown = view(ex)
        try:
            parsed, meta = ask(shown)
        except RuntimeError as error:
            counts["stopped"] = 1
            counts["why_stopped"] = str(error)[:200]  # type: ignore[assignment]
            break
        counts["asked"] += 1
        answer = (parsed or {}).get("answer") if isinstance(parsed, dict) else None
        options = [o["option"] for o in shown["options"]]
        if answer is None:
            counts["unsure"] += 1
        elif str(answer) not in options:
            counts["refused"] += 1
        value = str(answer) if answer is not None and str(answer) in options else None
        counts["labelled"] += value is not None
        line = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "example": ex["id"], "point": ex["point"],
                "origin": ex.get("origin"), "split": ex.get("split"), "value": value, "answer": answer,
                "confidence": (parsed or {}).get("confidence") if isinstance(parsed, dict) else None,
                "model": model, "purpose": "routing_labeler", **meta}
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
    return counts


def asker(gateway: Any, cap_eur: float) -> tuple[Callable[[dict[str, Any]], tuple[Optional[dict[str, Any]], dict[str, Any]]], Any]:
    from ..prompts import text

    client = gateway.client("routing_labeler", draft_id="routing-labels")
    if client is None:
        raise RuntimeError("no model serves routing_labeler: set the provider's key")
    client.run_limit_eur = cap_eur
    system = text("routing_labeler")

    def ask(shown: dict[str, Any]) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
        reply = client.complete([{"role": "system", "content": system},
                                 {"role": "user", "content": json.dumps(shown, ensure_ascii=False, default=str)}], None, max_tokens=1200)
        raw = (reply.get("content") or "").strip()
        start, end = raw.find("{"), raw.rfind("}")
        try:
            parsed = json.loads(raw[start:end + 1]) if start >= 0 else None
        except ValueError:
            parsed = None
        usage = reply.get("usage") or {}
        return parsed, {"request": getattr(client, "last_request", None), "reply": raw[:2000], "thought": bool(reply.get("thinking") or reply.get("reasoning")),  # the summary itself is in the exchange (request id)
                        "input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}

    return ask, client


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="send (without it, the plan only)")
    parser.add_argument("--cap-eur", type=float, default=None, help="required with --yes")
    parser.add_argument("--scripted", type=int, default=40, help="scripted examples labelled per point at most")
    parser.add_argument("--point", action="append", choices=POINTS, help="only this point (repeatable)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    layout = Layout()
    todo = pending(layout, args.scripted, tuple(args.point or POINTS))
    if args.limit is not None:
        todo = todo[:args.limit]
    by_point: dict[str, int] = {}
    for ex in todo:
        by_point[ex["point"]] = by_point.get(ex["point"], 0) + 1
    print(json.dumps({"to_label": len(todo), "by_point": by_point, "estimate_eur": round(len(todo) * EUR_PER_CALL, 2)}, indent=1))
    if not args.yes:
        print("nothing was sent: add --yes and --cap-eur")
        return 0
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: --cap-eur above zero is required", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    from ..models.gateway import for_workspace

    gateway = for_workspace()
    gateway.shadows = None
    ask, client = asker(gateway, args.cap_eur)
    counts = run(layout, ask, todo, str(getattr(client, "model", "model")))
    print(json.dumps({**counts, "spent_eur": round(float(getattr(client, "spent_eur", 0.0) or 0.0), 4)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
