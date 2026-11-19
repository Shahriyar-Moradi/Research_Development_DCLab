"""Does the problem memory help the Lead on its own? The day-2 measurement (2026-11-19).

Eight invented dev stories (``memory_day2_stories.json``). On day 1 a person states three facts in plain words: the
prediction moment, the outcome column, and the unit or horizon. They are stored in the Problem Base the way Home's
hand-off stores a brief's lines (``draft/memory.py``: "Prediction made: ...", "Outcome predicted: ...", one line per
fact, through ``memory.bases.upsert_base_fact`` on a host-resolved scope). On day 2 the person opens a NEW conversation
on the same project and problem with a short request that needs those facts ("Let's build the model now.").

The Lead's first day-2 turn runs twice per story, in two fresh conversations: the memory switch off
(``DCLAB_LEAD_PROBLEM_MEMORY=0``) and on (``=1``); the bounded loop stays off in both. Scored by code, no judge:

* re-asked: for each fact, a question sentence (ending in "?") that names that fact's keywords (the story's own words
  for it, and the general words in ``GENERIC_ASK``);
* questions: how many question sentences the reply holds;
* named: the facts the reply acts on, naming the stored moment, outcome or unit in a sentence that is not a question.

The keywords are written in the stories file and here before any run; their hash is kept in every result.

    python -m dclab_rnd.agent_eval.memory_day2                         # the plan; nothing sent
    python -m dclab_rnd.agent_eval.memory_day2 --dry-run               # every step with a scripted client; nothing sent
    python -m dclab_rnd.agent_eval.memory_day2 --yes --cap-eur 0.3     # the live run (qwen/qwen3.8-flash), at most EUR 0.30
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
STORIES = HERE / "memory_day2_stories.json"
CAMPAIGN = "memory_day2_v1"
CAMPAIGN_DIR = ROOT / "evidence" / "campaigns" / CAMPAIGN
RESULTS = CAMPAIGN_DIR / "results"
KEEP_ROOT = ROOT / "agent_runs" / "live_tests"
MODEL = "qwen/qwen3.8-flash"  # the product's model (the owner's choice, 2026-11-11); never a Claude model in a test
MAX_CAP_EUR = 0.30
ESTIMATE_EUR_PER_TURN = 0.006  # HAB-065: 113 calls of the Lead on this model cost EUR 0.105; a first turn is a few calls
ARMS = ("off", "on")
FACT_KEYS = ("moment", "outcome", "unit")
# Home's hand-off turns a brief's line label into the Problem Base field the same way (draft/memory.py hand_off)
FIELD_MAP = {"prediction_made": "prediction_moment", "outcome_predicted": "target"}
NOTE_MARK = "Problem memory (revision"

# The general words of a question about each fact, whatever the story (case-insensitive regular expressions).
GENERIC_ASK: dict[str, tuple[str, ...]] = {
    "moment": (r"prediction (?:moment|time|point)", r"\bmoment\b", r"point in time", r"at what (?:point|time|stage)",
               r"\bwhen\b[^?]*\bpredict", r"\bcut-?off\b", r"\b(?:known|available) (?:at|when|before|by)\b"),
    "outcome": (r"\btarget\b", r"\boutcome\b", r"\blabel\b", r"\bwhat\b[^?]*\b(?:to|you|we)\b[^?]*\bpredict",
                r"which column[^?]*\b(?:predict|target|outcome)"),
    "unit": (r"\bhorizon\b", r"\bwindow\b", r"\bgrain\b", r"\bgranularity\b", r"\bunit\b", r"one row per", r"\beach row\b",
             r"what does (?:a|each|one) row", r"how far (?:ahead|in advance|out)", r"\btime ?frame\b"),
}


def load_stories(path: Path = STORIES) -> list[dict[str, Any]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(doc["stories"])


def rules_sha256() -> str:
    """The hash of what decides the scores: the stories (their facts and keywords) and the general question words."""
    payload = {"stories": json.loads(STORIES.read_text(encoding="utf-8")), "generic_ask": GENERIC_ASK}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- the invented tables


def frame_of(story: Mapping[str, Any]) -> Any:
    """The story's table, made by code from its seed (never a real file)."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(int(story["seed"]))
    n = int(story["rows"])
    data: dict[str, Any] = {}
    for col in story["columns"]:
        kind, name = col["kind"], col["name"]
        if kind == "id":
            data[name] = [f"{col.get('prefix', '')}{100000 + i}" for i in range(n)]
        elif kind == "int":
            data[name] = rng.integers(int(col["low"]), int(col["high"]) + 1, n)
        elif kind == "float":
            data[name] = np.round(rng.uniform(float(col["low"]), float(col["high"]), n), 2)
        elif kind == "cat":
            data[name] = rng.choice(np.array(col["choices"], dtype=object), n)
        elif kind == "bool":
            data[name] = (rng.random(n) < float(col["p"])).astype(int)
        elif kind == "date":
            start = pd.Timestamp(col["start"])
            data[name] = sorted((start + pd.to_timedelta(rng.integers(0, int(col["days"]), n), unit="D")).strftime("%Y-%m-%d"))
        elif kind == "datetime":
            start = pd.Timestamp(col["start"])
            data[name] = sorted((start + pd.to_timedelta(rng.integers(0, int(col["hours"]) * 60, n), unit="min")).strftime("%Y-%m-%d %H:%M"))
        elif kind == "monthly":
            start = pd.Timestamp(col["start"])
            data[name] = sorted((start + pd.DateOffset(months=int(m))).strftime("%Y-%m-%d") for m in rng.integers(0, int(col["months"]), n))
        elif kind == "weekly":
            start = pd.Timestamp(col["start"])
            data[name] = sorted((start + pd.to_timedelta(7 * rng.integers(0, int(col["weeks"]), n), unit="D")).strftime("%Y-%m-%d"))
        else:
            raise ValueError(f"{story['id']}: unknown column kind {kind!r}")
    return pd.DataFrame(data)


def column_view(story: Mapping[str, Any], frame: Any) -> list[dict[str, Any]]:
    """The column view the Lead gets (names, kinds, what DCLab noticed; never a value), through Home's own upload path."""
    from dclab_rnd.draft import pipeline
    from dclab_rnd.draft import understand as U
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.store import DraftStore

    with tempfile.TemporaryDirectory() as folder:
        store = DraftStore(Path(folder) / "drafts")
        d = store.create("a table")
        agent = HomeAgent(store, None, on_request=lambda *a: None)
        filename = f"{story['table']}.csv"
        frame.to_csv(store.data_dir(d["id"]) / filename, index=False)
        asset = pipeline.new_asset(store, d["id"], "upload", filename, filename)
        pipeline.run(store, d["id"], asset["id"], agent, None)
        return U.columns_view(store.get(d["id"]))


# ---------------------------------------------------------------- day 1: the facts in the Problem Base


def field_of(line: str) -> str:
    label = line.split(":", 1)[0].strip().lower().replace(" ", "_") if ":" in line else "fact"
    return FIELD_MAP.get(label, label)


def day1(home: Path, story: Mapping[str, Any]) -> dict[str, Any]:
    """The project and its problem, and the person's three facts written to the Problem Base on a host-resolved scope."""
    from dclab_rnd.accounts.principal import Principal
    from dclab_rnd.memory import bases, scope as mem_scope
    from dclab_rnd.studio.store import ProjectStore

    projects = ProjectStore(Path(home) / "projects")
    project = projects.create(story["title"], goal=story["day1"][:200])
    person = str(story["person"])
    projects.update(project["id"], workspace_id="local", members=[person])
    problem = mem_scope.ensure_default_problem(projects, project["id"], workspace_id="local", by=person, name=story["title"])
    principal = Principal(role="data_scientist", workspace_id=None, user_id=person, via="token",
                          workspaces=(("local", "Local", "data_scientist"),))

    def scope(session: str) -> Any:
        return mem_scope.resolve_scope(workspace_id="local", principal=principal, projects=projects, project_id=project["id"],
                                       problem_id=problem["problem_id"], session_id=session, require_problem=True)

    day1_scope = scope(f"{story['id']}-day1")
    for fact in story["facts"]:
        bases.upsert_base_fact(Path(home), day1_scope, target="problem", subject="problem", field=field_of(fact["text"]),
                               text=fact["text"], source=f"home brief:{story['id']}-day1")
    return {"project_id": project["id"], "problem_id": problem["problem_id"], "scope": scope}


# ---------------------------------------------------------------- scoring by code


def _compiled(story: Mapping[str, Any]) -> dict[str, dict[str, list[re.Pattern[str]]]]:
    def words(items: Any) -> list[re.Pattern[str]]:
        return [re.compile(r"(?<![\w])" + re.escape(str(w).lower()) + r"(?![\w])") for w in items]

    out = {}
    for fact in story["facts"]:
        key = fact["key"]
        out[key] = {"ask": [re.compile(p) for p in GENERIC_ASK[key]] + words(fact["ask_keywords"]),
                    "named": words(fact["named_keywords"])}
    return out


def sentences(text: str) -> list[str]:
    """The reply's sentences; a question is one ending in "?" (markdown marks and closing quotes set aside)."""
    parts: list[str] = []
    for line in str(text or "").splitlines():
        parts += [p.strip() for p in re.split(r"(?<=[.?!])\s+", line) if p.strip()]
    return parts


def is_question(sentence: str) -> bool:
    return sentence.rstrip(" *_\"')]`").endswith("?")


def score(story: Mapping[str, Any], text: str, extra: str = "") -> dict[str, Any]:
    """``text``: the Lead's message; ``extra``: what else the person saw (the options' words), read for named facts only."""
    patterns = _compiled(story)
    said = sentences(text)
    questions = [s for s in said if is_question(s)]
    statements = [s for s in said + sentences(extra) if not is_question(s)]
    reasked, named, matched = [], [], {}
    for key in (f["key"] for f in story["facts"]):
        hits = [q for q in questions if any(p.search(q.lower()) for p in patterns[key]["ask"])]
        if hits:
            reasked.append(key)
            matched[key] = [h[:240] for h in hits]
        if any(p.search(s.lower()) for s in statements for p in patterns[key]["named"]):
            named.append(key)
    return {"reasked": reasked, "questions": len(questions), "named": named, "matched_questions": matched}


def paired(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """The paired result: re-asked facts off against on, people better, worse and equal (fewer re-asked facts is better)."""
    by: dict[str, dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["story"]), {})[str(r["arm"])] = r["score"]
    both = {s: arms for s, arms in by.items() if "off" in arms and "on" in arms}
    better = sorted(s for s, a in both.items() if len(a["on"]["reasked"]) < len(a["off"]["reasked"]))
    worse = sorted(s for s, a in both.items() if len(a["on"]["reasked"]) > len(a["off"]["reasked"]))
    equal = sorted(s for s in both if s not in better and s not in worse)
    per_fact = {arm: {k: sum(k in a[arm]["reasked"] for a in both.values()) for k in FACT_KEYS} for arm in ARMS}
    return {"stories": len(both), "facts": 3 * len(both),
            "reasked": {arm: sum(len(a[arm]["reasked"]) for a in both.values()) for arm in ARMS},
            "reasked_by_fact": per_fact,
            "questions": {arm: sum(int(a[arm]["questions"]) for a in both.values()) for arm in ARMS},
            "named": {arm: sum(len(a[arm]["named"]) for a in both.values()) for arm in ARMS},
            "better": better, "worse": worse, "equal": equal}


# ---------------------------------------------------------------- day 2: the Lead's first turn, memory off and on


@contextmanager
def _switches(memory_on: bool) -> Iterator[None]:
    from dclab_rnd.agents import decision_loop

    keys = (decision_loop.SWITCH_ENV, decision_loop.MEMORY_ENV)
    before = {k: os.environ.get(k) for k in keys}
    os.environ[decision_loop.SWITCH_ENV] = "0"  # the bounded loop stays off in both arms (lead_loop_live_v1: it hurts)
    os.environ[decision_loop.MEMORY_ENV] = "1" if memory_on else "0"
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Seen:
    """The client as the Lead uses it, noting whether any request carried the memory note (the switch really worked)."""

    def __init__(self, inner: Any):
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "note_sent", False)

    def complete(self, messages: Any, *args: Any, **kwargs: Any) -> Any:
        if any(NOTE_MARK in str(m.get("content") or "") for m in messages or [] if isinstance(m, Mapping)):
            object.__setattr__(self, "note_sent", True)
        return self._inner.complete(messages, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._inner, name, value)


def day2_turn(story: Mapping[str, Any], arm: str, home: Path, bound: Mapping[str, Any], view: list[dict[str, Any]], frame: Any,
              clients: Mapping[str, Any], folder: Path) -> dict[str, Any]:
    from dclab_core import case as C
    from dclab_rnd.agents import case_agent as A

    folder.mkdir(parents=True, exist_ok=True)
    lead = _Seen(clients["case"]) if clients.get("case") is not None else None
    state = A.CaseState(case=C.new(f"{story['id']}-day2-{arm}"), table=A.table_of(f"{story['table']}.csv", view, len(frame)), frame=frame,
                        specialist_client=clients.get("specialist"), route_client=clients.get("route"),
                        checkpoint=folder / "case_checkpoint.json", workspace_home=Path(home), project_id=bound["project_id"],
                        problem_id=bound["problem_id"], authorized_scope=bound["scope"](f"{story['id']}-day2-{arm}"))
    with _switches(arm == "on"):
        out = A.turn(state, lead, story["day2"])
    options = [" ".join(str(x or "") for x in (o.title, o.what, o.gets)) for o in out.get("options") or []]
    text = str(out.get("text") or "")
    got = score(story, text, "\n".join(options))
    row = {"story": story["id"], "arm": arm, "score": got, "note_sent": bool(getattr(lead, "note_sent", False)),
           "asking": bool(out.get("asking")), "options": len(options), "route": getattr(state.case.route, "track", None),
           "spent_eur": round(float(state.spent_eur or 0.0), 5)}
    (folder / "turn.json").write_text(json.dumps({**row, "request": story["day2"], "reply": text, "option_texts": options,
                                                  "case_file": state.case.model_dump(mode="json")}, indent=1, ensure_ascii=False,
                                                 default=str) + "\n", encoding="utf-8")
    return row


def measure(stories: list[Mapping[str, Any]], clients_for: Callable[[str, str], Mapping[str, Any]], keep: Path,
            exhausted: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Every story's day 1 written, then its two day-2 turns (the order alternates by story, so neither arm always goes first)."""
    rows: list[dict[str, Any]] = []
    incomplete = None
    for i, story in enumerate(stories):
        home = Path(keep) / "stories" / story["id"] / "workspace"
        home.mkdir(parents=True, exist_ok=True)
        bound = day1(home, story)
        frame = frame_of(story)
        view = column_view(story, frame)
        for arm in (ARMS if i % 2 == 0 else tuple(reversed(ARMS))):
            if exhausted():
                incomplete = incomplete or "cap reached"
                break
            try:
                rows.append(day2_turn(story, arm, home, bound, view, frame, clients_for(story["id"], arm),
                                      Path(keep) / "stories" / story["id"] / arm))
            except Exception as error:  # noqa: BLE001 - a failed turn is said, never scored as the arm's answer
                from .conversation_exam import _run_stopped

                why = _run_stopped(error)
                rows.append({"story": story["id"], "arm": arm, "error": f"{type(error).__name__}: {error}"[:400]})
                if why:
                    incomplete = why
                    break
        if incomplete:
            break
    scored = [r for r in rows if "score" in r]
    return {"rows": rows, "paired": paired(scored), "incomplete": incomplete,
            "note_check": {arm: [r["story"] for r in scored if r["arm"] == arm and r["note_sent"] != (arm == "on")] for arm in ARMS}}


# ---------------------------------------------------------------- the command


class _Scripted:
    """The dry run's client: one fixed reply, no network."""

    spent_eur = 0.0
    model = "scripted"

    def complete(self, messages: Any, tools: Any = None, max_tokens: Any = None, **kwargs: Any) -> dict[str, Any]:
        text = "I will predict the outcome per order. What is the prediction moment? Which horizon do you need?"
        return {"content": text, "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": text}}


def _code_state() -> dict[str, Any]:
    from .harness_ablation import code_state

    return code_state()


def _print_summary(result: Mapping[str, Any]) -> None:
    print(json.dumps({"paired": result["paired"], "note_check": result["note_check"], "incomplete": result["incomplete"],
                      "errors": [r for r in result["rows"] if "error" in r]}, indent=1))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.memory_day2", description=__doc__.split("\n\n")[0])
    parser.add_argument("--yes", action="store_true", help="send the live run (needs --cap-eur, at most 0.30)")
    parser.add_argument("--cap-eur", type=float, default=None)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dry-run", action="store_true", help="every step with a scripted client in a temporary folder; nothing sent")
    parser.add_argument("--only", default="", help="story ids, comma-separated")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    stories = load_stories()
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        stories = [s for s in stories if s["id"] in wanted]
    estimate = round(len(stories) * len(ARMS) * ESTIMATE_EUR_PER_TURN, 3)
    print(json.dumps({"stories": [s["id"] for s in stories], "split": "dev", "arms": list(ARMS), "turns": len(stories) * len(ARMS),
                      "model": args.model, "estimate_eur": estimate, "max_cap_eur": MAX_CAP_EUR, "rules_sha256": rules_sha256()}, indent=1))
    if args.dry_run:
        with tempfile.TemporaryDirectory() as folder:
            result = measure(stories, lambda story, arm: {"case": _Scripted()}, Path(folder))
        _print_summary(result)
        print("dry run: nothing was sent, nothing stored")
        return 0 if not result["incomplete"] and not any("error" in r for r in result["rows"]) else 1
    if not args.yes:
        print("nothing was sent: add --yes and --cap-eur (at most 0.30), or --dry-run")
        return 0
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: --cap-eur above zero is required", file=sys.stderr)
        return 2
    if args.cap_eur > MAX_CAP_EUR:
        print(f"refused: this measurement's cap is at most EUR {MAX_CAP_EUR:.2f}", file=sys.stderr)
        return 2
    if "claude" in str(args.model).lower():
        print("refused: the owner's rule of 2026-11-15: no Claude model in any test, arm or judge, at any provider", file=sys.stderr)
        return 2
    return _live(stories, args.model, args.cap_eur)


def _live(stories: list[dict[str, Any]], model: str, cap_eur: float) -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd import experience

    from .conversation_exam import EXAM, Budget, practice_effort
    from .harness_ablation import capture, owner_allows, price_snapshot, run_gateway

    if os.environ.get("DCLAB_NO_LIVE_MODELS", "").strip().lower() in ("1", "true", "on", "yes"):
        print("refused: DCLAB_NO_LIVE_MODELS is set", file=sys.stderr)
        return 2
    if not experience.model_text_allowed():
        print("refused: prompts and thinking would not be kept (DCLAB_LOG_PROMPTS is off)", file=sys.stderr)
        return 2
    why = owner_allows(cap_eur)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(m.group(1)) for p in RESULTS.glob("MD2-*.json") if (m := re.match(r"MD2-(\d+)", p.name))] or [0])
    run = f"MD2-{n:03d}"
    run_dir = KEEP_ROOT / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-memory-day2"
    run_dir.mkdir(parents=True, exist_ok=True)
    price_snapshot(run_dir, model)  # before the gateway reads its prices: the cap must be able to count the model
    gateway = run_gateway(run_dir)
    gateway.invented_tables = "memory day2 dev: stories' tables made by code from a seed"
    unserved = sorted(p for p in ("case_agent", "case_specialist", "route_check") if not gateway.available(p))
    if unserved:
        print(f"refused: no model serves {', '.join(unserved)}", file=sys.stderr)
        return 2
    effort = practice_effort()
    budget = Budget(gateway, cap_eur)
    tag = f"{EXAM}{run}:"

    def clients_for(story: str, arm: str) -> dict[str, Any]:
        key = f"{tag}{arm}-{story}"
        return {"case": budget.client(key, "case_agent", model), "specialist": budget.client(key, "case_specialist", model),
                "route": budget.client(key, "route_check", model)}

    manifest = {"run": run, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "dev", "model": model,
                "thinking_effort": effort, "cap_eur": cap_eur, "arms": list(ARMS), "stories": [s["id"] for s in stories],
                "rules_sha256": rules_sha256(), "code": _code_state(),
                "fixed_before_run": "facts, keywords and scoring rules in memory_day2_stories.json and GENERIC_ASK, hashed in rules_sha256"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    result = measure(stories, clients_for, run_dir, exhausted=lambda: budget.spent_eur() >= 0.98 * budget.cap_eur)
    for r in result["rows"]:
        if "score" in r:
            r["spent_eur"] = round(budget.spent_eur(f"{tag}{r['arm']}-{r['story']}"), 5)
    kept = capture(run_dir)
    record = {"id": run, "campaign_id": CAMPAIGN, "split": "dev", "set": "memory_day2_dev_v1", **manifest,
              "spent_eur": round(budget.spent_eur(), 4), "kept": str(run_dir.relative_to(ROOT)), "capture": kept, **result,
              "rule": "Eight invented dev stories, one turn each arm, one repeat: counts with no interval; never a claim about "
                      "other conversations or production readiness."}
    path = RESULTS / f"{run}_dev.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    _print_summary(result)
    print(json.dumps({"capture": kept, "spent_eur": round(budget.spent_eur(), 4)}, indent=1))
    if kept["without_thinking"]:
        print(f"WARNING: answered calls with no thinking kept, by purpose: {kept['without_thinking']}", file=sys.stderr)
    print(f"stored {path.relative_to(ROOT)}; kept {run_dir.relative_to(ROOT)}")
    return 0 if not result["incomplete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
