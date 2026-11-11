"""``python -m dclab_rnd.agent_eval.journeys --split dev|sealed_v7 [--workers N] [--only ID|KIND ...] [--no-store] [--yes]``.

Every sealed set up to sealed_v6 is spent; sealed_v7 is run once, whole, with --yes (even the free rules arm: a run
spends the set for every arm)."""

from __future__ import annotations

from dclab_rnd.memory.reading import memory_off_for_evaluation

memory_off_for_evaluation()  # 35.4: a graded run never reads a remembered line

import argparse
import json
import os
import sys
from typing import Optional
from pathlib import Path

from . import GATES, PARTS as PARTS_, SEALED, families, measure, sealed_runs, store, store_compare, verdict


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev", *SEALED))
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))  # each one an engine: memory
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--compare", action="store_true", help="22.9: every arm (the rules, each gated decision model, all) on the split")
    parser.add_argument("--seed", type=int, default=None, help="the replicas' seed (the sealed run uses the gate's)")
    parser.add_argument("--reader", action="store_true", help="20.10 v4: the rules and the conversation reader (a model) on the same people")
    parser.add_argument("--yes", action="store_true", help="send the reader's requests (with --reader)")
    parser.add_argument("--cap-eur", type=float, default=0.5, help="the reader's spend cap for the whole run")
    parser.add_argument("--spent", action="store_true", help="read a sealed set already run as dev material: nothing is stored")
    parser.add_argument("--model-rerun", default=None, help="the owner's one-time exception (2026-11-11): a new model reruns a spent "
                                                             "sealed set once; named apart, marked not unseen")
    parser.add_argument("--keep", type=Path, default=None, help="with --live-arms: every live journey's whole record copied into DIR/<arm>/<id>/")
    parser.add_argument("--walk", choices=("loop", "graph"), default="loop", help="28.3: the coordinator's loop, or its graph (the orchestration layer)")
    parser.add_argument("--team-arms", action="store_true", help="28.7: the rules, the single agent (the loop with the conversation reader) and the team "
                        "(the graph with the reader and the understanding critic's model) on the same people; with --yes and --cap-eur")
    parser.add_argument("--live-arms", action="store_true", help="26.5: the rules, the workflow model alone, and the workflow model with the "
                                                                    "decision models, on dev (with --yes and --cap-eur)")
    parser.add_argument("--pattern", choices=("second_opinion", "debate"), help="28.6: a team pattern against its baseline on the dev people and "
                        "the spent fifth sealed set (dev material); with --yes and --cap-eur")
    args = parser.parse_args(argv)
    from dclab_rnd import experience as _experience

    _experience.sealed_run(getattr(args, "split", None))  # an exam's model calls keep their shape only (the owner's rule, 2026-10-28)
    if args.pattern:
        return _pattern(args)
    sealed = args.split in SEALED
    if args.spent:
        if not (sealed and sealed_runs(args.split)):
            print("--spent reads a sealed set that was run already, as dev material")
            return 2
        args.no_store, sealed = True, False  # its people are dev material now: played, never stored
    if args.model_rerun and not (sealed and sealed_runs(args.split) and (args.reader or args.team_arms)):
        print("--model-rerun is for a spent sealed set run with a model's arms (--reader or --team-arms): the rules alone are no model's rerun")
        return 2
    if sealed and sealed_runs(args.split) and not args.spent:
        from .. import sealed as _sealed
        import dclab_rnd.agent_eval.journeys as _jr
        from dclab_rnd.models import settings as _settings

        model = _settings.served_model("conversation_reader")
        first = [p for p in sealed_runs(args.split) if "_rerun_" not in p.name]
        why = _sealed.refusal(args.model_rerun, model, [p for p in sealed_runs(args.split) if p.stem.endswith(_sealed.suffix(model))])
        if why:
            print(f"the sealed journeys were run once already: {why}")
            return 2
        _jr.RERUN = {"suffix": _sealed.suffix(model), **_sealed.stamp(model, [p.stem.split("_")[0] for p in first]),
                     "models": {"model": model}}  # the bench attributes the rerun to its model
    if args.split == "sealed_v4" and not args.spent and not args.reader:
        print("the fourth sealed set judges the reader against the rules: run it with --reader, both arms at once")
        return 2
    if args.split in ("sealed_v5", "sealed_v6") and not args.spent and not args.team_arms:
        print("the fifth sealed set judges the team against the rules and the single agent: run it with --team-arms, the three arms at once")
        return 2
    if sealed and (args.only or args.no_store):
        print("the sealed journeys are run whole and stored, once")
        return 2
    gate = GATES.get(args.split)
    if sealed and args.seed not in (None, gate["seed"]):
        print(f"the sealed journeys run on the gate's seed, {gate['seed']}")
        return 2
    if sealed and not args.yes:  # an exam is spent by its one run, so even the free rules arm asks first
        print(f"Plan: {args.split} is played once, whole, on its gate's seed, and stored; it is then spent for every arm. "
              "Add --yes to run it.")
        return 0
    seed = gate["seed"] if sealed else (0 if args.seed is None else args.seed)
    if args.compare:
        if sealed:
            print("the arms are compared on dev: a sealed set is run once, for its gate")
            return 2
        from . import arms, compare

        by_arm = {arm: measure(args.split, seed, args.workers, args.only, doc) for arm, doc in arms().items()}
        result = compare(by_arm)
        for arm, a in result["arms"].items():
            print(f"{'ok  ' if a['ok'] else 'WORSE'} {arm:30} {a['right']}/{a['journeys']} right  {json.dumps(a['parts'])}  changed by the model {a['decisions_by_model']} (agreed {a['agreed_with_the_rule']}), "
                  f"held for review {a['held_for_review']}, refused by the validator {a['refused_by_validator']}, moves refused {a['refused_moves']}, "
                  f"changed {a['changed']}")
        print(result["verdict"])
        if not args.no_store:
            print(store_compare(by_arm, result, args.split, seed))
        return 0
    if args.reader:
        return _reader_arm(args, seed)
    if args.live_arms:
        return _live_arms(args, seed)
    if args.team_arms:
        return _team_arms(args, seed)
    rows = measure(args.split, seed, args.workers, args.only, walk=args.walk)
    for r in rows:
        mark = "ok  " if r["right"] else "MISS"
        print(f"{mark} {r['id']:6} {r['kind']:14} {json.dumps(r['parts'])} {r.get('seconds', '')}s asked {r.get('asked')} card {r.get('got')}  {'; '.join(r['why'])[:300]}")
    for f in families(rows):
        print(f"  {f['family']:16} {f['right']}/{f['journeys']}  interval {f['interval']}")
    result = verdict(rows)
    print(result)
    if not args.no_store:  # 28.3: a walk through the graph is stored as its own arm, beside the loop's
        print(store(rows, args.split, result, arm="rules" if args.walk == "loop" else "rules_graph", seed=seed, extra={"walk": args.walk}))
    return 0


def _reader_arm(args, seed: int) -> int:
    """The rules and the reader played on the same people, one after the other in this process (the reader's client lives
    here); a sealed set stores both arms, once."""
    from . import load, play, verdict

    people = [j for j in load(args.split) if not args.only or j["id"] in args.only or j["kind"] in args.only]
    if not args.yes:
        print(f"Plan: {len(people)} people, the conversation reader asked about each problem sentence and each answer that settles the "
              f"card (about 3 to 5 requests each), capped at {args.cap_eur} EUR for the run. Add --yes to send them.")
        return 0
    from dotenv import load_dotenv

    from dclab_rnd.draft.reader import reader
    from dclab_rnd.models.gateway import for_workspace
    from dclab_rnd.settings import ROOT

    load_dotenv(ROOT / ".env", override=False)  # the provider and its key, as the server reads them

    from dclab_rnd.models.gateway import mark_invented_tables
    read = reader(mark_invented_tables(for_workspace(), "journeys (replicas)"), cap_eur=args.cap_eur, force=True, prebuild=True)
    if read is None:
        print("No model serves conversation_reader: set the provider's key (and leave DCLAB_NO_LIVE_MODELS unset).")
        return 2
    by_arm = {"rules": measure(args.split, seed, args.workers, args.only)}
    by_arm["reader"] = []
    for j in people:
        try:
            by_arm["reader"].append(play(j, seed, None, {"reader": read}))
        except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason (as in measure)
            by_arm["reader"].append({"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": {}, "right": False,
                                     "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]})
    spent = round(read.advisor.spent_eur(), 4)
    right = {arm: sum(r["right"] for r in rows) for arm, rows in by_arm.items()}
    passed = verdict(by_arm["reader"]).startswith("pass") and right["reader"] >= right["rules"] and read.failures == 0
    against = {"reader_right": right["reader"], "rules_right": right["rules"], "reader_requests": read.asked,
               "reader_failures": read.failures, "passed": passed, "rule": GATES.get(args.split, {}).get("reader")}
    for arm, rows in by_arm.items():
        print(f"{arm:7} {sum(r['right'] for r in rows)} of {len(rows)} right, cards {sum(r['parts']['card'] for r in rows)}")
    rules = {r["id"]: r for r in by_arm["rules"]}
    for r in by_arm["reader"]:
        if r["right"] != rules[r["id"]]["right"]:
            print(f"  {'GAINED' if r['right'] else 'LOST':6} {r['id']} {r['kind']}: {'; '.join(r['why'] or rules[r['id']]['why'])[:200]}")
    print(f"spent EUR {spent} (cap {args.cap_eur}); {read.asked} reader requests, {read.failures} failed; reader against rules: {against}")
    if read.failures:
        print("a reader request failed (the provider, or the cap): the rules answered for it, so this is not a reader arm; nothing is stored")
        return 1
    if not args.no_store:
        for arm, rows in by_arm.items():
            print(store(rows, args.split, verdict(rows), arm=arm, seed=seed, spent_eur=spent if arm == "reader" else 0.0,
                        extra={"reader_against_rules": against} if arm == "reader" else None))
    return 0


class _Counted:
    """An advisor whose failed requests are counted (a role falls back to its rules on one: that is no live arm)."""

    def __init__(self, inner):
        self.inner, self.failures = inner, 0

    def ask(self, purpose, evidence):
        try:
            return self.inner.ask(purpose, evidence)
        except Exception:
            self.failures += 1
            raise

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _Shared(_Counted):
    """28.6: the single designer's answer, asked once for the same evidence and shared by both arms of the debate: the
    rules' conversation gives the planner the same evidence in both, so the debate alone differs between them."""

    def __init__(self, inner):
        super().__init__(inner)
        self.memo: dict[str, object] = {}

    def ask(self, purpose, evidence):
        key = purpose + json.dumps(evidence, sort_keys=True, default=str)
        if key not in self.memo:
            self.memo[key] = super().ask(purpose, evidence)
        return self.memo[key]


def _debate_seats(args):
    """The debate's three seats, their clients made before the journeys' private scope: the planner on the standard tier,
    the challenger and the judge on the cheap tier (Haiku 5.5 unless the workspace names another), each with its share of
    the cap; None when a seat has no model."""
    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.models import settings as model_settings
    from dclab_rnd.models.gateway import for_workspace

    if not model_settings.openrouter_default():  # on Anthropic the cheap seats are Haiku; on OpenRouter the tier's own model serves
        os.environ.setdefault("DCLAB_TIER_CHEAP_MODEL", "claude-haiku-5-5")
    gateway = for_workspace()
    from dclab_rnd.models.gateway import mark_invented_tables
    mark_invented_tables(gateway, "journeys (replicas)")  # every table here is invented by code
    designer = _Shared(GatewayAdvisor(gateway, cap_eur=args.cap_eur * 0.6, max_tokens=1500))
    # the first run (void: four requests failed) cut three challengers at 1,500 tokens and a judge at 900: Haiku writes at length
    challenger = _Counted(GatewayAdvisor(gateway, cap_eur=args.cap_eur * 0.25, max_tokens=4000))
    judge = _Counted(GatewayAdvisor(gateway, cap_eur=args.cap_eur * 0.15, max_tokens=2500))
    if any(seat._client(purpose) is None for seat, purpose in ((designer, "solution_planner"), (challenger, "solution_challenger"), (judge, "debate_judge"))):
        return None
    return designer, challenger, judge


def _pattern(args) -> int:
    """28.6: one team pattern against its baseline, both live, on the same people and seeds, one after the other in this
    process: the second opinion against the single agent (the loop with the conversation reader). The gate
    (``PATTERN_GATES``) is fixed before the run; each arm has half the cap; a run in which a request failed is not stored;
    every live journey's whole record is kept in ``agent_runs/live_tests/<date>_pattern_<name>/<arm>/``."""
    from datetime import date

    from . import PATTERN_GATES, PATTERN_MATERIAL, load, pattern_verdict, play

    gate = PATTERN_GATES[args.pattern]
    people = [(j, seed, split) for split, seed in PATTERN_MATERIAL for j in load(split) if not args.only or j["id"] in args.only]
    if args.only:
        args.no_store = True  # a subset is never stored: the gate is over the whole material (the review of 28.6)
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: a pattern's arms need --cap-eur above zero", file=sys.stderr)
        return 2
    if not args.yes:
        material = f"{len(people)} people ({', '.join(f'{s} seed {d}' for s, d in PATTERN_MATERIAL)}; the spent sealed set read as dev material)"
        if args.pattern == "debate":
            print(f"Plan: {material}, the conversation by the rules in both arms; the single designer (the solution planner, standard tier, its "
                  f"answer asked once a person and shared by both arms) and the debate (the same, with the challenger and the judge on the cheap "
                  f"tier, Haiku 5.5 unless DCLAB_TIER_CHEAP_MODEL names another; the judge only on a dispute); about 1 planner and 1 to 2 cheap "
                  f"requests a person; capped at {args.cap_eur} EUR (the planner 60%, the challenger 25%, the judge 15%). "
                  f"Gate {gate['version']}: {gate['rule']}. Add --yes to send them.")
        else:
            print(f"Plan: {material}, two arms: the single agent (the loop with the conversation reader) and the same with the second "
                  f"opinion on the family (no model request added: the person's choice is read by code), one reader shared by both with its "
                  f"answers remembered (the same answer read once); about 3 to 5 reader requests a person; capped at {args.cap_eur} EUR. "
                  f"Gate {gate['version']}: {gate['rule']}. Add --yes to send them.")
        return 0
    from dotenv import load_dotenv

    from dclab_rnd.draft.reader import reader
    from dclab_rnd.models.gateway import for_workspace
    from dclab_rnd.orchestra import patterns
    from dclab_rnd.settings import ROOT as ROOT_

    load_dotenv(ROOT_ / ".env", override=False)  # the provider and its key, as the server reads them
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    if args.pattern == "debate":
        seats = _debate_seats(args)
        if seats is None:
            print("No model serves solution_planner, solution_challenger or debate_judge: set the provider's key (and leave DCLAB_NO_LIVE_MODELS unset).")
            return 2
        designer, challenger, judge = seats
        lives = {"baseline": {"designer": designer}, "arm": {"designer": designer, "challenger": challenger, "judge": judge}}
    else:
        gateway = for_workspace()
        from dclab_rnd.models.gateway import mark_invented_tables
        mark_invented_tables(gateway, "journeys (replicas)")  # every table here is invented by code
        shared = reader(gateway, cap_eur=args.cap_eur, force=True, prebuild=True)
        if shared is None:
            print("No model serves conversation_reader: set the provider's key (and leave DCLAB_NO_LIVE_MODELS unset).")
            return 2
        # one reader for both arms, its answers remembered: the same answer is read the same way in both, so the arms differ only
        # where the judge acts (the second review of 28.6: two readers' noise decided the gate); the judge adds no request
        shared.advisor.memo = True
        readers = {"baseline": shared, "arm": shared}
        lives = {"baseline": {"reader": shared}, "arm": {"reader": shared, args.pattern: True}}
    keep = ROOT_ / "agent_runs" / "live_tests" / f"{date.today().isoformat()}_pattern_{args.pattern}"
    by_arm: dict[str, list] = {"baseline": [], "arm": []}
    for j, seed, split in people:
        for arm, live in lives.items():
            try:
                row = play(j, seed, None, live, keep / arm / j["id"])
            except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason
                row = {"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": {}, "right": False,
                       "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]}
            by_arm[arm].append({**row, "split": split})
    if args.pattern == "debate":  # the designer's answer is both arms' (asked once): the debate's seats add to it
        spent = {"baseline": round(designer.spent_eur(), 4), "arm": round(designer.spent_eur() + challenger.spent_eur() + judge.spent_eur(), 4)}
        failures = designer.failures + challenger.failures + judge.failures
        models = {"designer": designer.report(), "challenger": challenger.report(), "judge": judge.report()}
    else:  # one shared reader: both arms cost what it spent
        spent = {"baseline": round(shared.advisor.spent_eur(), 4), "arm": round(shared.advisor.spent_eur(), 4)}
        failures = shared.failures
        models = {"reader": shared.advisor.report() if hasattr(shared.advisor, "report") else None}
    against = pattern_verdict(args.pattern, by_arm["baseline"], by_arm["arm"], spent, failures)
    for side in ("baseline", "arm"):
        print(f"{side:8} parts {against['parts'][side]}, {against['questions'][side]} questions a person, asked a field twice: {against['asked_twice'][side]}")
    print(f"gained {against['gained']} (by the pattern {against['gained_by_the_pattern']}), lost {against['lost']}, broke {against['broke']}; "
          f"spent EUR {spent} (cap {args.cap_eur}); {failures} failed requests; {args.pattern} against {gate['version']}: "
          f"{'pass' if against['passed'] else 'fail'}; records kept in {keep}")
    if failures or against["broke"]:
        print("a request failed or a journey broke: these are not the two arms the gate compares; nothing is stored")
        return 1
    if not args.no_store:
        print(patterns.store(args.pattern, [f"{s} seed {d}" for s, d in PATTERN_MATERIAL], by_arm, against,
                             extra={"models": models, "cap_eur": args.cap_eur}))
    return 0


def _team_arms(args, seed: int) -> int:
    """28.7: three arms on the same people, one after the other in this process (the models' clients live here): the rules
    (the loop, no model), the single agent (the loop with the conversation reader) and the team (the graph, the same reader
    and the understanding critic's model). The gate (``TEAM_GATE``) is fixed before the run; each live arm has its own share
    of the cap; a run in which a request failed is not stored; every live journey's whole record is kept in
    ``agent_runs/live_tests/<date>_team_arms/<arm>/``."""
    from datetime import date

    from . import TEAM_GATE, load, play, verdict

    people = [j for j in load(args.split) if not args.only or j["id"] in args.only or j["kind"] in args.only]
    if args.cap_eur is None or args.cap_eur <= 0:
        print("refused: the team arms need --cap-eur above zero", file=sys.stderr)
        return 2
    from dclab_rnd.orchestra import patterns as team_patterns

    joined = {name: True for name in ("second_opinion",) if team_patterns.passed(name)}  # 28.6: a pattern joins the team only where it passed
    if not args.yes:
        print(f"Plan: {len(people)} people, three arms (the rules; the loop with the conversation reader; the graph with the reader and the "
              f"understanding critic's model{', and ' + ', '.join(sorted(joined)) if joined else ''}), about 3 to 5 reader requests and 1 to 3 critic requests a person; capped at {args.cap_eur} EUR "
              f"(the single agent a third, the team two thirds). Gate: {TEAM_GATE['rule']}. Add --yes to send them.")
        return 0
    from dotenv import load_dotenv

    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.draft.reader import reader
    from dclab_rnd.models.gateway import for_workspace
    from dclab_rnd.settings import ROOT as ROOT_

    load_dotenv(ROOT_ / ".env", override=False)  # the provider and its key, as the server reads them
    gateway = for_workspace()
    from dclab_rnd.models.gateway import mark_invented_tables
    mark_invented_tables(gateway, "journeys (replicas)")  # every table here is invented by code
    single = reader(gateway, cap_eur=args.cap_eur / 3, force=True, prebuild=True)
    team_reader = reader(gateway, cap_eur=args.cap_eur / 3, force=True, prebuild=True)
    critic = _Counted(GatewayAdvisor(gateway, cap_eur=args.cap_eur / 3, max_tokens=1200))
    if single is None or team_reader is None or critic._client("understanding_critic") is None:  # made before the journeys' private scope
        print("No model serves conversation_reader or understanding_critic: set the provider's key (and leave DCLAB_NO_LIVE_MODELS unset).")
        return 2
    keep = ROOT_ / "agent_runs" / "live_tests" / f"{date.today().isoformat()}_team_arms"
    by_arm = {"rules": measure(args.split, seed, args.workers, args.only)}
    by_arm["single"], by_arm["team"] = [], []
    for j in people:
        for arm, live, walk in (("single", {"reader": single}, "loop"), ("team", {"reader": team_reader, "critic": critic, **joined}, "graph")):
            try:
                by_arm[arm].append(play(j, seed, None, live, keep / arm / j["id"], walk=walk))
            except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason
                by_arm[arm].append({"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": {}, "right": False,
                                    "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]})
    spent = {"single": round(single.advisor.spent_eur(), 4), "team": round(team_reader.advisor.spent_eur() + critic.spent_eur(), 4)}
    failures = single.failures + team_reader.failures + critic.failures
    parts = {arm: {p: sum(1 for r in rows if r["parts"].get(p) is True) for p in PARTS_} for arm, rows in by_arm.items()}
    sealed = args.split != "dev"
    card_ok = parts["team"]["card"] > parts["single"]["card"] if sealed else parts["team"]["card"] >= parts["single"]["card"]
    per = spent["team"] / max(1, len(people))
    passed = (all(parts["team"][p] >= parts["rules"][p] for p in PARTS_) and card_ok and failures == 0 and per <= TEAM_GATE["cap_per_journey_eur"]
              and not any((r.get("refused_moves") or 0) for r in by_arm["team"]))
    against = {"parts": parts, "spent_eur": spent, "per_journey_eur": round(per, 4), "failures": failures, "passed": passed, "gate": TEAM_GATE,
               "critic_requests": critic.report() if hasattr(critic, "report") else None}
    for arm, rows in by_arm.items():
        print(f"{arm:7} {sum(r['right'] for r in rows)} of {len(rows)} right, parts {parts[arm]}")
    rules = {r["id"]: r for r in by_arm["rules"]}
    for arm in ("single", "team"):
        for r in by_arm[arm]:
            if r["right"] != rules[r["id"]]["right"]:
                print(f"  {arm} {'GAINED' if r['right'] else 'LOST':6} {r['id']} {r['kind']}: {'; '.join(r['why'] or rules[r['id']]['why'])[:200]}")
    print(f"spent EUR {spent} (cap {args.cap_eur}); {failures} failed requests; team against the gate: {'pass' if passed else 'fail'}; records kept in {keep}")
    if failures:
        print("a request failed (the provider, or the cap): the rules answered for it, so these are not live arms; nothing is stored")
        return 1
    if not args.no_store:
        for arm, rows in by_arm.items():
            print(store(rows, args.split, verdict(rows), arm=arm if arm == "rules" else f"{arm}_live", seed=seed, spent_eur=spent.get(arm, 0.0),
                        extra={"team_against": against, "walk": "graph" if arm == "team" else "loop"} if arm != "rules" else None))
    return 0


def _live_arms(args, seed: int) -> int:
    """26.5: the third arm. The rules alone; the workflow model alone (the configured model as the Home agent and the
    roles, every decision point on the rules: its routing is empty); and the workflow model with the decision models
    (the same, with every model that passed its sealed gate routed to act, and the Home agent's decide_point answering
    from them). Played in this process (the clients live here), on dev only, each live arm with half the cap."""
    from . import arms, compare, load, play

    if args.split != "dev":
        print("the live arms are compared on dev: a sealed set is run once, for its own gate")
        return 2
    people = [j for j in load(args.split) if not args.only or j["id"] in args.only or j["kind"] in args.only]
    gated = arms()
    combined = gated.get("all") or next((doc for name, doc in gated.items() if name != "rules" and not name.endswith("@check")), {"decisions": {}})
    routed = {"workflow_model": {"decisions": {}}, "workflow_model+decision_models": combined}
    if not args.yes:
        print(f"Plan: {len(people)} people, three arms (the rules; the configured model as the Home agent and the roles; the same with "
              f"{sorted((combined.get('decisions') or {})) or 'no gated model'} routed to act), capped at {args.cap_eur} EUR for both live arms. "
              "Add --yes to send the requests.")
        return 0
    from dotenv import load_dotenv

    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.models.gateway import Gateway
    from dclab_rnd.settings import ROOT

    load_dotenv(ROOT / ".env", override=False)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    by_arm = {"rules": measure(args.split, seed, args.workers, args.only)}
    spent, models, incomplete = 0.0, {}, []
    for name, doc in routed.items():
        gateway = Gateway()
        from dclab_rnd.models.gateway import mark_invented_tables
        mark_invented_tables(gateway, "journeys (replicas)")  # every table here is invented by code
        client = gateway.client("home_agent")
        if client is None:
            print("no model can serve the Home agent here (no key, or DCLAB_NO_LIVE_MODELS): nothing live to run")
            return 2
        arm_cap = args.cap_eur / 2  # half the cap per live arm, shared by the Home agent and the roles as they spend it
        client.run_limit_eur = arm_cap
        live = {"client": client, "advisor": GatewayAdvisor(gateway, cap_eur=arm_cap)}
        rows, most = [], 0.0
        for j in people:
            before = client.spent_eur + live["advisor"].spent_eur()
            left = arm_cap - client.spent_eur - live["advisor"].spent_eur()
            if rows and left < 1.5 * most:  # the next journey could run out mid-way: it would quietly fall back to the rules
                incomplete.append(f"{name}: stopped before {j['id']} (EUR {left:.3f} left, a journey took up to EUR {most:.3f})")
                break
            try:
                rows.append(play(j, seed, doc, live, keep=args.keep / name.replace("+", "_and_") / j["id"] if args.keep else None))
            except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason
                rows.append({"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": {}, "right": False,
                             "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]})
            most = max(most, client.spent_eur + live["advisor"].spent_eur() - before)
        by_arm[name] = rows
        spent += client.spent_eur + live["advisor"].spent_eur()  # what the clients cost, a broken journey's requests included
        models[name] = sorted({m for r in rows for m in (r.get("models") or {}).values()})
    result = compare(by_arm)
    for arm, a in result["arms"].items():
        print(f"{'ok  ' if a['ok'] else 'WORSE'} {arm:32} {a['right']}/{a['journeys']} right  {json.dumps(a['parts'])}  changed by a decision model "
              f"{a['decisions_by_model']}, held for review {a['held_for_review']}, refused by the validator {a['refused_by_validator']}")
    print(result["verdict"], f"spent EUR {spent:.4f} of {args.cap_eur}")
    if incomplete:  # an arm cut short by its budget is not an arm of the people it did not play: nothing is stored
        print("not stored, an arm was cut short by the cap: " + "; ".join(incomplete))
        return 1
    if not args.no_store and not args.only:
        print(store_compare(by_arm, result, args.split, seed, spent_eur=round(spent, 4), routing={"rules": {"decisions": {}}, **routed},
                            extra={"arm_kind": "live_arms_26_5", "models": models, "cap_eur": args.cap_eur}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
