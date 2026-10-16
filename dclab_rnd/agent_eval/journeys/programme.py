"""Journeys of a problem with several parts (packages 24.6 and 24.8): a written person, from the first sentence to the one read of the holdout.

A journey (``journeys_programme_dev.json``, ``journeys_programme_sealed.json``; the sealed file's SHA-256 pinned in ``sealed_programme.sha256``) is a person, as in
``journeys/__init__.py``: the problem in their words, their answers keyed by the field a question asks, the files they bring (``brings``: the rides always, the confirmed
cases, the table of accounts), what they do when a part cannot work for lack of a file (``when_blocked``: drop it with a reason, or stop), and what a careful expert would
expect (``expect``: the size, the families of the parts, the approach of each, and whether the pieces can be composed and judged). :func:`play` runs the person as the
product does, with no model unless an arm gives the map roles one: a draft with the rides and the cases, the coordinator (21.2) walking it to a built project (it draws the
map of the problem and the written person confirms it), the table of accounts attached by name, the plan drawn again from the tables, the programme run as a job would run
it, a blocked part dropped when the person says so, and the pipeline read at the end. It is scored on four parts, each against the expert's expectation and the replica's
planted truth:

- **card**: the size is the expert's (one problem or several) and, for one problem, its family;
- **workflow**: every part the expert expects is drawn with the expert's approach (a part the files cannot solve has no approach), at most one part is drawn beyond
  them, and no move of the project's graph was refused;
- **pipeline**: every step behind the programme's numbers has its run and none is stale, a replay in a scratch copy makes the same document; a person who stopped at a
  blocked part has no stale step;
- **holdout**: a person who goes on and has the cases gets a composed result that is never worse than its best leaf (above it, with the interval, when all three leaves
  ran) and the holdout read once; a person who stops at a blocked part, or who has no confirmed case, leaves the holdout closed with the reason said.

A part that does not apply (a single problem has no programme to run) is not scored; a journey is right when every part that applies is. The gate is fixed in code
(``GATE``) before any run; the sealed set is run once, both arms (the rules, and the map roles served by the model) on the same people.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "journeys_programme_v1" / "results"
SPLITS = {"dev": HERE / "journeys_programme_dev.json", "sealed": HERE / "journeys_programme_sealed.json"}
SEALED_HASH = HERE / "sealed_programme.sha256"
PARTS = ("card", "workflow", "pipeline", "holdout")
REPLICAS = {"fraud_programme": "dclab_core.replicas.fraud_programme"}
ARMS = ("rules", "roles")
EXTRA_PARTS = 1  # parts drawn beyond the expert's that still count as right
DROP_REASON = "the files I have cannot do this part"

GATE = {
    "version": "journeys_programme_gate_v1",  # fixed on 2026-10-16, before the sealed set was run (and before the harness was written for it)
    "split": "sealed",
    "journey": "right on the card (the size, and for one problem its family), the workflow (every part the expert expects is drawn with the expert's approach, at most one part "
               "beyond them, no graph move refused), the pipeline, and the holdout (a person who goes on and has the confirmed cases gets a composed result never worse than its best "
               "leaf, above it with the interval when all three leaves ran, read once; a person who stops at a blocked part, or who has no confirmed case, leaves it closed with the "
               "reason said); a part that does not apply is not scored",
    "suite": "at least 85% of the sealed journeys right, the lower end of their 95% Wilson interval at least 0.55, no graph move refused, no holdout read twice",
    "roles": "the roles arm (the decomposer, the map reviewer, the data-need agent and the solution router served by the model) is right on at least as many journeys as the rules "
             "arm and on at least as many of each part, every request answered, no graph move refused, within the cap; a role serves the product only then",
    "seed": 5,  # the replicas' seed for the sealed run: not the dev seed (0), the calibration seeds (10 to 15) or the older sets'
    "author": "an agent given only the three files' columns, the journeys' format and the vocabulary of families and approaches; it never read the rules, the dev people or any code",
}
MIN_SHARE, MIN_LOW = 0.85, 0.55


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(split: str = "dev") -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"no programme journeys named {split!r}: one of {', '.join(SPLITS)}")
    if split == "sealed" and fingerprint(SPLITS[split]) != SEALED_HASH.read_text(encoding="utf-8").strip():
        raise ValueError("the sealed programme journeys changed after they were sealed: a sealed set is never edited")
    return json.loads(SPLITS[split].read_text(encoding="utf-8"))


def sealed_runs() -> list[Path]:
    return sorted(RESULTS.glob("JPR-*_sealed.json")) if RESULTS.exists() else []


# ---------------------------------------------------------------------------------------------------------------- one person

def _pairs(families: list[str], approaches: list[Optional[str]]) -> Counter:
    return Counter(zip(families, approaches))


def _workflow_ok(expect: dict[str, Any], families: list[str], approaches: list[Optional[str]]) -> tuple[bool, str]:
    """The expert's parts are all drawn, each with the expert's approach, and at most EXTRA_PARTS are drawn beyond them."""
    wanted, drawn = _pairs(list(expect["families"]), list(expect["approaches"])), _pairs(families, approaches)
    missing = wanted - drawn
    extra = sum((drawn - wanted).values())
    if missing:
        return False, f"parts expected but not drawn with that approach: {sorted(missing.elements())}; drawn {sorted(drawn.elements(), key=str)}"
    if extra > EXTRA_PARTS:
        return False, f"{extra} parts drawn beyond the expert's: {sorted((drawn - wanted).elements(), key=str)}"
    return True, ""


def play(journey: dict[str, Any], seed: int = 0, stores: Any = None, live: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One programme journey from its first sentence; its scores and what happened at each part. ``stores``: the (drafts, projects) it plays in, a workspace's own when a
    test wants to read what the journey left in its record; else temporary ones. ``live``: ``{"map_advisor": ...}``, the map roles served by a model (the roles arm)."""
    import importlib

    from dclab_rnd.agent_eval.journeys import _Person, _upload, refused
    from dclab_rnd.draft.card import card_of
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import ProjectStore, data as sd, pipeline, programme

    started = time.perf_counter()
    advisor = (live or {}).get("map_advisor")
    replica_module = importlib.import_module(REPLICAS[journey["programme"]])
    replica = replica_module.generate(seed=seed)
    expect = journey["expect"]
    brings = list(journey.get("brings") or ["rides", "accounts", "cases"])
    when_blocked = journey.get("when_blocked", "stop")
    complex_ = expect.get("size", "complex") == "complex"
    row: dict[str, Any] = {"id": journey["id"], "programme": journey["programme"], "brings": brings, "parts": {}, "why": []}
    with tempfile.TemporaryDirectory() as tmp:
        home, mine = Path(tmp), Path(tmp) / "person"
        mine.mkdir()
        drafts, projects = stores if stores is not None else (DraftStore(home / "drafts"), ProjectStore(home / "projects"))
        replica_module.save(replica, mine)
        agent = HomeAgent(drafts, None)
        draft_id = drafts.create(journey["words"])["id"]
        agent.start(draft_id)
        for name, key in ((replica_module.RIDE_FILE, "rides"), (replica_module.CASE_FILE, "cases")):  # the table and the cases, as a person brings them
            if key in brings:
                _upload(drafts, agent, draft_id, name, mine / name)
        person = _Person(journey["answers"])
        import contextlib

        with private_scope(True) if advisor is None else contextlib.nullcontext():  # a private scope forbids model requests: the rules arm keeps to it, the roles arm must leave it
            out = Coordinator(drafts, projects, agent, None, map_advisor=advisor).run(draft_id, person.person)
        understood = card_of(drafts.get(draft_id))
        row.update(asked=person.asked, questions=len(person.asked), steps=[s["action"] for s in out["steps"]], state=out["state"], said=out.get("says"))
        held = (drafts.get(draft_id).get("problem_map") or {})
        drawn_map = held.get("map") or {}
        size = drawn_map.get("size")
        row["got"] = {"size": size, "family": understood.family.value, "families": [s.get("family") for s in drawn_map.get("sub_problems") or []]}
        built = out.get("project_id")
        # the card: the size is the expert's, and for one problem its family
        size_ok = size == expect.get("size", "complex")
        family_ok = complex_ or understood.family.value == expect["families"][0] or row["got"]["families"] == list(expect["families"])
        card_ok = size_ok and family_ok
        row["parts"]["card"] = bool(card_ok)
        if not card_ok:
            row["why"].append(f"card: read as {size!r} ({row['got']['families']}), the expert's {expect.get('size')!r} {expect['families']}")
        if built is None:
            row["parts"].update(workflow=False, pipeline=None if not complex_ else False, holdout=None if not complex_ else False)
            row["why"].append(f"not built: stopped at {out['state']}: {out.get('says')}")
            return _end(row, started)
        if not complex_:  # one problem: the family and the approach routed for it are the workflow; the engine's own journeys carry its stages
            routes = ((held.get("routes") or {}).get("routes") or {})
            families = [s.get("family") for s in drawn_map.get("sub_problems") or []]
            approaches = [(routes.get(s.get("id")) or {}).get("chosen") for s in drawn_map.get("sub_problems") or []]
            ok, why = _workflow_ok(expect, families, approaches)
            refused_moves = refused(projects, built)
            row["refused_moves"] = len(refused_moves)
            row["parts"]["workflow"] = ok and not refused_moves
            if not ok or refused_moves:
                row["why"].append(f"workflow: {why or 'a graph move was refused'}")
            row["got"]["approaches"] = approaches
            row["parts"].update(pipeline=None, holdout=None)
            return _end(row, started)
        # several parts: the person attaches the table of accounts when they have it, and the plan is drawn again from the tables
        project = projects.get(built)
        plan = programme.plan_of(project)
        if plan is None:
            row["parts"].update(workflow=False, pipeline=False, holdout=False)
            row["why"].append("no plan of a programme was kept with the project")
            return _end(row, started)
        if "accounts" in brings:
            replica.accounts.to_csv(projects.data_dir(built) / "accounts.csv", index=False)
            sd.attach_table(projects, built, "accounts.csv", "accounts")
        programme.refresh_plan(projects, built, advisor)
        plan = programme.plan_of(projects.get(built))
        families = [s["family"] for s in plan["problem_map"]["sub_problems"]]
        approaches = [(plan["routes"].get(s["id"]) or {}).get("chosen") for s in plan["problem_map"]["sub_problems"]]
        row["got"].update(families=families, approaches=approaches)
        ok, why = _workflow_ok(expect, families, approaches)
        view = programme.run(projects, built, advisor=advisor)
        if view["state"] == "planned" and view.get("blockers") and when_blocked == "drop":  # the person drops each part the files cannot do, with the reason, and goes on
            stored = programme.stored(projects.get(built))
            for leaf in (stored.leaves if stored is not None else []):
                if leaf.status == "blocked":
                    programme.drop(projects, built, leaf.block, DROP_REASON)
            view = programme.run(projects, built, advisor=advisor)
        refused_moves = refused(projects, built)
        row["refused_moves"] = len(refused_moves)
        row["parts"]["workflow"] = ok and not refused_moves
        if not ok or refused_moves:
            row["why"].append(f"workflow: {why or 'a graph move was refused'}; refused moves {[t.get('move') for t in refused_moves[:3]]}")
        project = projects.get(built)
        p = programme.stored(project)
        steps = pipeline.of(projects, built)
        stale = [f"{s.id} ({s.why})" for s in steps.steps if s.status == "stale"]
        read = p is not None and p.holdout is not None
        missing = [f"{s.id} ({s.status})" for s in steps.steps if s.status not in ("done", "stale") and not str(s.why or "").startswith("dropped by")] if read else []
        replay = pipeline.rerun(projects, built, "programme:holdout") if read and not missing else {"same": not read}
        row["parts"]["pipeline"] = not stale and not missing and bool(replay.get("same"))
        if stale or missing or not replay.get("same"):
            row["why"].append(f"pipeline: stale {stale}, no run {missing}, replay same {replay.get('same')}")
        row["holdout_uses"] = int(project.get("holdout_uses", 0) or 0)
        has_cases = "cases" in brings
        blocked_now = bool(p is not None and (p.blockers or any(l.status in ("planned", "blocked") for l in p.leaves)))
        if not has_cases or blocked_now:  # nothing to judge a list on, or a part cannot work and the person stops: the holdout stays closed, and says why
            closed_ok = not read and row["holdout_uses"] == 0 and bool(view.get("blockers") or view.get("issues") or (p is not None and any("No confirmed case" in n for n in p.notes)))
            expected_closed = (not has_cases and not expect.get("composes")) or (blocked_now and when_blocked == "stop")
            row["parts"]["holdout"] = bool(closed_ok and expected_closed)
            if not row["parts"]["holdout"]:
                row["why"].append(f"holdout: {'read although it should stay closed' if read else 'closed, but the expert expected a composed result' if not expected_closed else 'closed without saying why'}")
            return _end(row, started)
        if not read:
            row["parts"]["holdout"] = False
            row["why"].append(f"holdout: the programme is {view.get('state')}: " + "; ".join(view.get("blockers") or view.get("issues") or []))
            return _end(row, started)
        h = p.holdout
        built_leaves = [l for l in p.leaves if l.status == "built"]
        failed = []
        if not expect.get("composes"):
            failed.append("the holdout was read although the expert expected nothing to compose")
        if h.reads != 1 or row["holdout_uses"] != 1:
            failed.append(f"the holdout was read {h.reads} time(s) by the programme and {row['holdout_uses']} by the project")
        if h.difference.value < 0:
            failed.append(f"the composition is worse than its best leaf: {h.difference.value}")
        if len(built_leaves) >= 3 and not h.beats_best_leaf:
            failed.append(f"all {len(built_leaves)} leaves ran and the composition did not beat the best leaf: {h.difference.value} ({h.difference.low} to {h.difference.high})")
        if programme.core.validate(p, programme.maps(project)[1]):
            failed.append("the programme does not validate: " + "; ".join(programme.core.validate(p, programme.maps(project)[1])[:2]))
        row["parts"]["holdout"] = not failed
        row["holdout"] = {"composed": h.composed.value, "best_leaf": h.best_leaf_estimate.value, "difference": h.difference.value, "low": h.difference.low, "high": h.difference.high,
                          "leaves": len(built_leaves), "chosen": h.chosen}
        if failed:
            row["why"].append(f"holdout: {failed}")
    return _end(row, started)


def _end(row: dict[str, Any], started: float) -> dict[str, Any]:
    row["seconds"] = round(time.perf_counter() - started, 1)
    applies = [row["parts"].get(p) for p in PARTS if row["parts"].get(p) is not None]
    row["right"] = bool(applies) and all(applies)
    return row


# ---------------------------------------------------------------------------------------------------------------- the suite

def _play_one(journey: dict[str, Any], seed: int, live: Optional[dict[str, Any]]) -> dict[str, Any]:
    try:
        return play(journey, seed, None, live)
    except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason
        import traceback

        where = traceback.extract_tb(error.__traceback__)[-1]
        return {"id": journey["id"], "programme": journey.get("programme"), "parts": {p: False for p in PARTS}, "right": False,
                "why": [f"broke: {type(error).__name__}: {str(error)[:240]} at {where.name}:{where.lineno}"]}


def measure(split: str, seed: int, live: Optional[dict[str, Any]] = None, only: Optional[list[str]] = None) -> list[dict[str, Any]]:
    return [_play_one(j, seed, live) for j in load(split) if not only or j["id"] in only]


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if not n:
        return 0.0, 1.0
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return round(centre - half, 3), round(centre + half, 3)


def verdict(rows: list[dict[str, Any]]) -> str:
    right, n = sum(bool(r["right"]) for r in rows), len(rows)
    low, high = wilson(right, n)
    refused = sum(r.get("refused_moves") or 0 for r in rows)
    twice = sum(1 for r in rows if (r.get("holdout_uses") or 0) > 1)
    ok = n > 0 and right / n >= MIN_SHARE and low >= MIN_LOW and refused == 0 and twice == 0
    return (f"{'pass' if ok else 'fail'}: {right} of {n} journeys right (95% interval {low} to {high}), {refused} moves refused by the validator, "
            f"{twice} holdouts read twice")


def by_part(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {p: {"right": sum(1 for r in rows if r["parts"].get(p) is True), "applies": sum(1 for r in rows if r["parts"].get(p) is not None)} for p in PARTS}


def compare(by_arm: dict[str, list[dict[str, Any]]], failures: int = 0) -> dict[str, Any]:
    """The roles against the rules on the same people: right on at least as many journeys and as many of each part, no request failed, no graph move refused."""
    rules, roles = by_arm["rules"], by_arm["roles"]
    r_right, o_right = sum(bool(r["right"]) for r in rules), sum(bool(r["right"]) for r in roles)
    parts_rules, parts_roles = by_part(rules), by_part(roles)
    refused = sum(r.get("refused_moves") or 0 for r in roles)
    ok = o_right >= r_right and all(parts_roles[p]["right"] >= parts_rules[p]["right"] for p in PARTS) and failures == 0 and refused == 0
    base = {r["id"]: r for r in rules}
    changed = [{"id": r["id"], "rules": bool(base[r["id"]]["right"]), "roles": bool(r["right"])} for r in roles if r["id"] in base and bool(r["right"]) != bool(base[r["id"]]["right"])]
    return {"rules_right": r_right, "roles_right": o_right, "parts": {"rules": parts_rules, "roles": parts_roles}, "changed": changed, "request_failures": failures,
            "roles_refused_moves": refused, "roles_verdict": verdict(roles), "passed": bool(ok and verdict(roles).startswith("pass")), "rule": GATE["roles"]}


def store(rows: list[dict[str, Any]], split: str, arm: str, seed: int, spent_eur: float = 0.0, extra: Optional[dict[str, Any]] = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("JPR-*.json")] or [0])
    ident = f"JPR-{n:03d}"
    doc = {"id": ident, "campaign_id": "journeys_programme_v1", "kind": "programme_journeys", "split": split, "arm": arm, "journeys": fingerprint(SPLITS[split]),
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gate": GATE, "seed": seed, "spent_eur": spent_eur, "rows": rows,
           "summary": {"journeys": len(rows), "right": sum(bool(r["right"]) for r in rows), "by_part": by_part(rows)}, "verdict": verdict(rows), "status": "completed",
           "experiment_id": ident, "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **(extra or {}),
           "rule": f"Scores on {len(rows)} written journeys over a seeded replica say how DCLab carries these people from their first sentence to a holdout number, or stops them with "
                   "the reason; they are not a claim about any other problem or about production readiness."}
    path = RESULTS / f"{ident}_{split}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def roles_passed() -> bool:
    """Whether the map roles passed their sealed measurement on the programme journeys: the stored roles arm's own comparison with the rules arm."""
    for path in sealed_runs():
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("arm") == "roles" and doc.get("journeys") == fingerprint(SPLITS["sealed"]):
            return bool((doc.get("roles_against_rules") or {}).get("passed"))
    return False


# ---------------------------------------------------------------------------------------------------------------- the command

def main(argv: Optional[list[str]] = None) -> int:
    """``python -m dclab_rnd.agent_eval.journeys.programme --split dev|sealed [--yes --cap-eur 0.5] [--no-store] [--only ID ...]``.

    dev: the rules arm (no model, no cost), and with ``--yes --cap-eur`` the roles arm as well. sealed: both arms, once, on the gate's seed; it needs ``--yes --cap-eur``
    (the roles arm sends requests), is refused if it was run, is stored whole, and a run in which a request failed is not stored (the rules cannot answer for a role)."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.journeys.programme")
    parser.add_argument("--split", choices=tuple(SPLITS), default="dev")
    parser.add_argument("--yes", action="store_true", help="send the roles arm's requests")
    parser.add_argument("--cap-eur", type=float, default=None, help="the roles arm's spend cap for the whole run (required with --yes)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-store", action="store_true")
    args = parser.parse_args(argv)
    sealed = args.split == "sealed"
    if sealed and sealed_runs():
        print(f"refused: the sealed programme journeys were run ({sealed_runs()[0].name}); a sealed set is run once", file=sys.stderr)
        return 2
    if sealed and (args.only or args.no_store):
        print("refused: the sealed journeys are run whole and stored, once", file=sys.stderr)
        return 2
    if sealed and args.seed not in (None, GATE["seed"]):
        print(f"refused: the sealed journeys run on the gate's seed, {GATE['seed']}", file=sys.stderr)
        return 2
    seed = GATE["seed"] if sealed else (0 if args.seed is None else args.seed)
    people = [j for j in load(args.split) if not args.only or j["id"] in args.only]
    if sealed and not args.yes:
        print(f"Plan: {len(people)} people, both arms, once, on seed {seed}. The roles arm sends the map roles' requests (the decomposer, the map reviewer, the data-need agent and "
              "the solution router) for each person; it needs --yes and --cap-eur. Nothing was sent.")
        return 0
    if args.yes and (args.cap_eur is None or args.cap_eur <= 0):
        print("refused: a remote model needs --cap-eur above zero", file=sys.stderr)
        return 2
    by_arm: dict[str, list[dict[str, Any]]] = {"rules": measure(args.split, seed, None, args.only)}
    for r in by_arm["rules"]:
        print(f"{'ok  ' if r['right'] else 'MISS'} rules {r['id']:6} {json.dumps(r['parts'])} {r.get('seconds', '')}s  {'; '.join(r['why'])[:300]}")
    print("rules:", verdict(by_arm["rules"]))
    if not args.yes:
        print("the roles arm was not run: add --yes and --cap-eur to send its requests")
        if not args.no_store and not sealed:
            print(store(by_arm["rules"], args.split, "rules", seed))
        return 0
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)  # the provider and its key, as the server reads them: only now that a euro cap is set
    from dclab_rnd.agent_eval.complex import Counting
    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.models import prices, settings
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    gateway.shadows = None  # measurement traffic stays out of the shadow log
    for purpose in ("decomposer", "map_reviewer", "data_need", "solution_router"):
        if not gateway.available(purpose):
            print(f"refused: no model serves {purpose} here", file=sys.stderr)
            return 2
        tier = gateway.route(purpose)[0] if hasattr(gateway, "route") else settings.purpose(purpose).tier
        public = settings.public(tier)
        if not public.get("local"):
            try:
                prices.estimate(public.get("model"), False, 1000, 1000)
            except ValueError:
                print("refused: the model has no price, so no euro cap can stop it (add it to the price table)", file=sys.stderr)
                return 2
    inner = GatewayAdvisor(gateway, max_tokens=2500, cap_eur=args.cap_eur, memo=True)  # an identical ask is paid for once (the sealed run of 24.8 was made so)
    advisor = Counting(inner)
    by_arm["roles"] = measure(args.split, seed, {"map_advisor": advisor}, args.only)
    spent = round(inner.spent_eur(), 4)
    sent = sum(inner.calls.values())
    if not sent or spent <= 0:  # asks that no model answered (a private scope, no key): the arm is the rules again, and comparing it with them proves nothing
        print(f"refused to score: {advisor.requests} asks, {sent} sent to a model, EUR {spent} spent: no request reached a model, so there is no roles arm", file=sys.stderr)
        return 1
    for r in by_arm["roles"]:
        print(f"{'ok  ' if r['right'] else 'MISS'} roles {r['id']:6} {json.dumps(r['parts'])} {r.get('seconds', '')}s  {'; '.join(r['why'])[:300]}")
    result = compare(by_arm, failures=len(advisor.failures))
    print("roles:", verdict(by_arm["roles"]))
    print(f"spent EUR {spent} (cap {args.cap_eur}); {advisor.requests} asks, {sent} sent to a model ({dict(inner.calls)}; {sum(inner.hits.values())} answered from the advisor's own memory), {len(advisor.failures)} failed; roles against rules: {json.dumps(result, default=str)[:900]}")
    if advisor.failures:
        print("a request failed (the provider, or the cap): the rules answered for it, so this is not a roles arm; nothing is stored", file=sys.stderr)
        return 1
    if not args.no_store:
        for arm, rows in by_arm.items():
            print(store(rows, args.split, arm, seed, spent if arm == "roles" else 0.0, {"roles_against_rules": result, "asks": advisor.requests, "sent": dict(inner.calls)} if arm == "roles" else None))
    return 0


__all__ = ["GATE", "PARTS", "ARMS", "load", "play", "measure", "verdict", "compare", "store", "sealed_runs", "roles_passed", "wilson"]


if __name__ == "__main__":
    import sys

    sys.exit(main())
