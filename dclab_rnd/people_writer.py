"""Written people for the understanding students' homework (S8 B1 of the 2026-10-25 plan; the owner: "the student
model must have more ability", with $15 to test and fix).

The understanding students (``problem.family``, ``problem.ready``, ``solution.route``, ``workflow.next_question``,
``task.framing``) failed their exams on tiny sets: 118 written people, on whom the rule is already right 95% of the
time. This writes many more, harder ones through the model gateway (Claude only; every request and reply kept whole):

- **The truth is never the writer's.** Each person gets its kind and family from this module (the 15 replica kinds and
  the family their journeys declare); the writer writes only the person's first words and answers, on the kind's own
  table (its column names).
- **Variety on purpose**: every batch names its own businesses (drawn from a fixed pool by kind and batch), and the
  prompt asks for wrong terms, mistakes, vague lines and few or many answers (``prompts/journey_writer.md``).
- **A checker** (``prompts/journey_checker.md``, a second request per batch) says whether each person fits the problem;
  a person who does not is kept apart (``*_flagged.json``), never deleted, never homework.
- The sealed people of the exam are written apart, by another writer (a fresh agent told only the format, the kinds and
  their columns), pinned by hash, and never read here.

``python -m dclab_rnd.people_writer plan`` prints what it would send; ``write --yes --cap-eur 3.5`` sends it (resumable:
a batch already answered is not asked again); ``status`` counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
JOURNEYS = ROOT / "dclab_rnd" / "agent_eval" / "journeys"
OUT = JOURNEYS / "journeys_homework_v1.json"
FLAGGED = JOURNEYS / "journeys_homework_v1_flagged.json"
HOME = ROOT / "agent_runs" / "live_tests" / "2026-10-25-people"
WRITER, CHECKER = "journey_writer", "journey_checker"  # the gateway's purposes
PROMPT, ID_PREFIX = "journey_writer", "JH"  # writer A; use_writer("b") switches to the second writer (S8 B2 v2)


def use_writer(name: str) -> None:
    """The second writer (a different prompt, its own files and ids), so the homework is not one writer's style."""
    global OUT, FLAGGED, HOME, PROMPT, ID_PREFIX
    if name == "b":
        OUT, FLAGGED = JOURNEYS / "journeys_homework_v2.json", JOURNEYS / "journeys_homework_v2_flagged.json"
        HOME, PROMPT, ID_PREFIX = ROOT / "agent_runs" / "live_tests" / "2026-10-25-people-b", "journey_writer_b", "JK"
    elif name == "d":  # the fourth writer (the owner, 2026-10-26): busy people's short messages, against the student's style gap
        OUT, FLAGGED = JOURNEYS / "journeys_homework_v3.json", JOURNEYS / "journeys_homework_v3_flagged.json"
        HOME, PROMPT, ID_PREFIX = ROOT / "agent_runs" / "live_tests" / "2026-10-26-people-d", "journey_writer_d", "JM"
    elif name != "a":
        raise ValueError(f"no writer {name!r}: a, b or d")
PER_BATCH, BATCHES = 10, 4  # 15 kinds x 4 batches x 10 people = 600
ANSWER_KEYS = ("target", "prediction_moment", "action", "entity", "labels", "budget", "costs", "data_plan", "family")
FAMILIES = {  # each kind's family, as its journeys declare it (the truth; never the writer's)
    "rare_event": "rare_event", "binary": "classification", "multiclass": "classification", "regression": "regression",
    "places": "regression", "text_tabular": ["classification", "text_or_images"], "embedded": ["text_or_images", "classification"],
    "anomaly": "anomaly", "forecasting": "forecasting", "time_to_event": "time_to_event", "streams": "streams",
    "entity_tables": "entity_tables", "segmentation": "segmentation", "uplift": "uplift", "ranking": "ranking",
}
BUSINESSES = ("a grocery chain", "a dental clinic", "a car repair shop", "an online bookstore", "a food delivery app", "a gym",
              "a mobile phone operator", "a bank's card team", "an insurance broker", "a hotel group", "a private school",
              "a university's admissions office", "a hospital ward", "a pharmacy chain", "a logistics company", "a ride-hailing app",
              "a furniture factory", "a solar panel installer", "a water utility", "a city council's call centre", "a fashion web shop",
              "a coffee roaster", "a bike rental", "a cinema chain", "a software company's support desk", "a game studio",
              "a news website", "a charity's donor team", "a real-estate agency", "a farm cooperative", "a wind farm operator",
              "a parcel locker network", "a call-centre outsourcer", "a dating app", "a pet supplies shop", "a travel agency",
              "a recruitment firm", "a printing house", "an e-scooter fleet", "a museum's membership desk")


def _families_text(kind: str) -> str:
    f = FAMILIES[kind]
    return " and ".join(f) if isinstance(f, list) else f


def kind_facts(kind: str) -> dict[str, Any]:
    """What the writer is told about a kind: its family, its table's column names, where the outcome is (never a value)."""
    from . import kinds

    case = kinds.get(kind).case(0)
    table = getattr(case, "table", None)
    facts: dict[str, Any] = {"kind": kind, "family": _families_text(kind), "columns": [str(c) for c in table.columns] if table is not None else []}
    if getattr(case, "target", None):
        facts["outcome_column"] = str(case.target)
    labels = getattr(case, "labels", None)
    if isinstance(labels, (list, tuple)) and labels and hasattr(labels[0], "columns"):
        facts["cases_file_columns"] = [str(c) for c in labels[0].columns]
        facts["note"] = "the known cases come in a second file, apart from the table"
    elif getattr(case, "files", None) is not None:
        facts["note"] = "the person has more than one file"
    return facts


def examples(kind: str, n: int = 2) -> list[dict[str, Any]]:
    """Format examples of the kind from the dev journeys (dev material; never a sealed set)."""
    found = [j for j in json.loads((JOURNEYS / "journeys_dev.json").read_text(encoding="utf-8")) if j.get("kind") == kind]
    return [{"words": j["words"], "answers": j.get("answers") or {}} for j in found[:n]]


def businesses(kind: str, batch: int) -> list[str]:
    rng = random.Random(f"{kind}:{batch}")
    return rng.sample(BUSINESSES, PER_BATCH)


def request(kind: str, batch: int) -> list[dict[str, str]]:
    from . import prompts

    ask = {"problem": kind_facts(kind), "people": PER_BATCH, "their_businesses": businesses(kind, batch), "examples_of_the_format": examples(kind)}
    return [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": json.dumps(ask, ensure_ascii=False)}]


def check_request(kind: str, people: list[dict[str, Any]]) -> list[dict[str, str]]:
    from . import prompts

    ask = {"problem": kind_facts(kind), "people": [{"index": i, "words": p["words"], "answers": p["answers"]} for i, p in enumerate(people)]}
    return [{"role": "system", "content": prompts.text(CHECKER)}, {"role": "user", "content": json.dumps(ask, ensure_ascii=False)}]


def parse_people(content: str) -> list[dict[str, Any]]:
    """The people of a reply that pass the format: words on a few lines, answers under the known keys, short."""
    from .core_advisor import _first_object

    doc = _first_object(content or "") or {}
    found = doc.get("people") if isinstance(doc.get("people"), list) else None
    if found is None:  # a reply whose JSON is broken somewhere (a stray brace): each person's object read on its own
        decoder, found, at = json.JSONDecoder(), [], (content or "").find('{"words"')
        while at >= 0:
            try:
                obj, end = decoder.raw_decode(content, at)
                found.append(obj)
            except ValueError:
                end = at + 1
            at = content.find('{"words"', end)
    out = []
    for p in found:
        if not isinstance(p, dict):
            continue
        words = p.get("words")
        answers = {k: str(v).strip()[:400] for k, v in (p.get("answers") or {}).items() if k in ANSWER_KEYS and isinstance(v, (str, int, float)) and str(v).strip()}
        if isinstance(words, str) and 3 <= len(words.strip()) <= 800:
            out.append({"words": words.strip(), "answers": answers})
    return out


def person(kind: str, batch: int, i: int, p: dict[str, Any], prompt_hash: str) -> dict[str, Any]:
    out = {"id": f"{ID_PREFIX}-{kind}-{batch}{i:02d}", "kind": kind, "family": FAMILIES[kind], "words": p["words"], "answers": p["answers"],
           "written_by": f"{PROMPT}@{prompt_hash}"}
    if kind == "rare_event":
        out["labels"] = "file"  # the replica's found cases come in their own file, as the dev journeys say
    return out


def _prompt_hash() -> str:
    from . import prompts

    return hashlib.sha256((prompts.text(PROMPT) + prompts.text(CHECKER)).encode()).hexdigest()[:12]


def _checker_hash() -> str:
    from . import prompts

    return hashlib.sha256(prompts.text(CHECKER).encode()).hexdigest()[:12]


def _client(purpose: str, cap_eur: float, spent: float) -> Any:
    from .models.gateway import for_workspace
    from .models.prices import price_of

    client = for_workspace().client(purpose)
    model = str(getattr(client, "model", "") or "")
    if client is None or not model.startswith("claude") or price_of(model) is None:
        raise SystemExit(f"{purpose} must be served by a priced Claude model here (it is {model or 'no model'}): nothing sent")
    client.run_limit_eur = max(0.0, cap_eur - spent)
    return client


def _ask(client: Any, messages: list[dict[str, str]], max_tokens: int) -> dict[str, Any]:
    reply = client.complete(messages, None, max_tokens=max_tokens)
    return {"content": reply.get("content") or "", "finish_reason": reply.get("finish_reason"), "usage": reply.get("usage"),
            "model": getattr(client, "model", None)}


def write(cap_eur: float, home: Optional[Path] = None, kinds_: Optional[list[str]] = None) -> dict[str, Any]:
    """Every (kind, batch) not answered yet: the writer's request, then the checker's on its people; each request and
    reply kept whole in ``home/replies``; the people that fit to ``OUT``, the others to ``FLAGGED``."""
    home = Path(home or HOME)
    replies = Path(home) / "replies"
    replies.mkdir(parents=True, exist_ok=True)
    spent = sum(float((json.loads(p.read_text(encoding="utf-8")).get("spent_eur") or 0.0)) for p in replies.glob("*.json"))
    prompt_hash = _prompt_hash()
    for batch in range(BATCHES):  # batch by batch across the kinds: a cap reached early still leaves every kind its people
        for kind in kinds_ or list(FAMILIES):
            path = replies / f"{kind}-{batch}.json"
            if path.is_file():
                continue
            if spent >= cap_eur:
                return {"stopped": f"the cap of EUR {cap_eur} is reached", "spent_eur": round(spent, 4)}
            writer = _client(WRITER, cap_eur, spent)
            messages = request(kind, batch)
            try:
                got = _ask(writer, messages, 6000)
            except RuntimeError as exc:  # the gateway's own cap refuses a request that could pass it (writer B, 2026-10-25: it
                if not str(exc).startswith("BudgetExceeded"):  # ended in a traceback at 58 of 60 and nothing was assembled)
                    raise
                return {"stopped": f"the cap of EUR {cap_eur} is reached", "spent_eur": round(spent, 4), **assemble(home)}
            people = parse_people(got["content"])
            checker = _client(CHECKER, cap_eur, spent + float(writer.spent_eur or 0.0))
            check_messages = check_request(kind, people) if people else None
            checked = _ask(checker, check_messages, 2500) if people else None
            cost = float(writer.spent_eur or 0.0) + float(getattr(checker, "spent_eur", 0.0) or 0.0)
            spent += cost
            path.write_text(json.dumps({"kind": kind, "batch": batch, "prompt": prompt_hash, "checker": _checker_hash(), "request": messages, "reply": got,
                                        "check_request": check_messages, "check_reply": checked, "spent_eur": round(cost, 6)},
                                       ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"spent_eur": round(spent, 4), **assemble(home)}


def _check_of(doc: dict[str, Any]) -> Optional[dict[str, Any]]:
    """A kept batch's latest check: a recheck under a newer checker prompt wins; the older ones stay in the file."""
    rechecks = doc.get("rechecks") or []
    return (rechecks[-1].get("reply") if rechecks else None) or doc.get("check_reply")


def recheck(cap_eur: float, home: Optional[Path] = None) -> dict[str, Any]:
    """Every kept batch whose latest check was made under another checker prompt, checked again (the old check kept)."""
    from . import prompts

    home = Path(home or HOME)
    current = hashlib.sha256(prompts.text(CHECKER).encode()).hexdigest()[:12]
    spent, asked = 0.0, 0
    for path in sorted((Path(home) / "replies").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        last = (doc.get("rechecks") or [{}])[-1].get("checker") or doc.get("checker")
        people = parse_people(doc["reply"]["content"])
        if last == current or not people:
            continue
        if spent >= cap_eur:
            return {"stopped": f"the cap of EUR {cap_eur} is reached", "spent_eur": round(spent, 4), "asked": asked}
        client = _client(CHECKER, cap_eur, spent)
        messages = check_request(doc["kind"], people)
        got = _ask(client, messages, 2500)
        spent += float(client.spent_eur or 0.0)
        asked += 1
        doc.setdefault("rechecks", []).append({"checker": current, "request": messages, "reply": got, "spent_eur": round(float(client.spent_eur or 0.0), 6)})
        doc["spent_eur"] = round(float(doc.get("spent_eur") or 0.0) + float(client.spent_eur or 0.0), 6)
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"spent_eur": round(spent, 4), "asked": asked, **assemble(home)}


def assemble(home: Optional[Path] = None) -> dict[str, int]:
    """The people of every kept reply, split by the checker into homework and kept-apart (rebuilt from the replies)."""
    from .core_advisor import _first_object

    home = Path(home or HOME)
    fit, flagged = [], []
    for path in sorted((Path(home) / "replies").glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        people = parse_people(doc["reply"]["content"])
        checks = {c.get("index"): c for c in ((_first_object((_check_of(doc) or {}).get("content") or "") or {}).get("checks") or [])
                  if isinstance(c, dict)}
        for i, p in enumerate(people):
            row = person(doc["kind"], doc["batch"], i, p, doc["prompt"])
            c = checks.get(i)
            if c is None or c.get("fits") is not True:
                flagged.append({**row, "check": (c or {}).get("why") or "no check answer"})
            else:
                fit.append(row)
    OUT.write_text(json.dumps(fit, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    FLAGGED.write_text(json.dumps(flagged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"people": len(fit), "flagged": len(flagged)}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.people_writer", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "write", "status", "assemble", "recheck"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap-eur", type=float)
    ap.add_argument("--kinds", nargs="*")
    ap.add_argument("--writer", default="a", choices=("a", "b", "d"))
    args = ap.parse_args(argv)
    use_writer(args.writer)
    if args.command == "plan":
        n = len(args.kinds or FAMILIES) * BATCHES
        print(f"Plan: {n} writer requests ({PER_BATCH} people each) and {n} checker requests through {WRITER} and {CHECKER} "
              f"(Claude only), about EUR {round(n * 0.05, 2)}; every request and reply kept whole in {HOME}.")
        print(json.dumps(request((args.kinds or list(FAMILIES))[0], 0)[1], ensure_ascii=False)[:1500])
        return 0
    if args.command == "status":
        replies = list((HOME / "replies").glob("*.json"))
        print({"batches": len(replies), "spent_eur": round(sum(float(json.loads(p.read_text()).get("spent_eur") or 0) for p in replies), 4)})
        return 0
    if args.command == "assemble":
        print(assemble())
        return 0
    if not args.yes or args.cap_eur is None:
        print("Add --yes --cap-eur N to send the requests.")
        return 0
    from . import real_shadow

    real_shadow.load_env()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    print(json.dumps(recheck(args.cap_eur) if args.command == "recheck" else write(args.cap_eur, kinds_=args.kinds), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
