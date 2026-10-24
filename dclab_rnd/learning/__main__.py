"""``python -m dclab_rnd.learning baseline|refusal|draft|pilot|probe``: the baseline (H0), the refusal test (H1), drafting lessons (H2), the pilot and the probe (H3)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.learning", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("baseline", help="today's DCLab on a family's variants, nothing learned: the zero of every learning curve")
    base.add_argument("--family", default="groups")
    base.add_argument("--split", default="check", choices=("learning", "check", "sealed"))
    base.add_argument("--n", type=int, default=20)
    base.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/learning_v1/results/")
    sub.add_parser("refusal", help="H1: a fixed list of bad moves every guard must still refuse (100% or a candidate is dropped)")
    draft = sub.add_parser("draft", help="H2: lessons drawn from graded learning variants, screened, filed as drafts in the playbook")
    draft.add_argument("--family", default="groups")
    draft.add_argument("--n", type=int, default=5)
    draft.add_argument("--model", action="store_true", help="the reflector purpose's model drafts (needs --yes and --cap-eur); default: the rules")
    draft.add_argument("--yes", action="store_true")
    draft.add_argument("--cap-eur", type=float, default=None, help="the most the reflector may spend on one challenge")
    pilot = sub.add_parser("pilot", help="H3: can a local model do the homework at all? the intern on the first learning variants, no lesson shown")
    pilot.add_argument("--model", default=None, help="the local model's name on the server, e.g. qwen3.5:4b-16k")
    pilot.add_argument("--hosted", action="store_true", help="a one-off comparison on the configured hosted model (Claude), never the learning loop; needs --yes and --cap-eur")
    pilot.add_argument("--yes", action="store_true")
    pilot.add_argument("--cap-eur", type=float, default=None, help="with --hosted: the most this whole run may spend")
    pilot.add_argument("--cap-eur-each", type=float, default=0.5, help="with --hosted: the most one homework may spend")
    pilot.add_argument("--limit", type=int, default=None, help="at most this many new homeworks in this process, then exit 3 (loop it: a fresh process a homework)")
    pilot.add_argument("--pause", action="store_true", help="ask a running pilot to stop after the homework it is on; the same command again goes on")
    pilot.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    pilot.add_argument("--family", default="groups")
    pilot.add_argument("--n", type=int, default=5)
    pilot.add_argument("--max-minutes", type=float, default=None, help="each homework's time budget (default the solver's 15); a laptop's speed, recorded in the result")
    pilot.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/learning_v1/results/")
    probe = sub.add_parser("probe", help="H3: does one lesson help a local model? the same learning variants without and with it, paired, graded")
    probe.add_argument("--model", default=None, help="the local model's name on the server, e.g. qwen3.5:4b-16k")
    probe.add_argument("--bullet", default=None, help="the lesson's id in the family's playbook (B-...)")
    probe.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    probe.add_argument("--family", default="groups")
    probe.add_argument("--lineage", default="main")
    probe.add_argument("--n", type=int, default=3, help="how many learning variants (the first ones)")
    probe.add_argument("--repeats", type=int, default=1, help="pairs per variant")
    probe.add_argument("--max-minutes", type=float, default=None, help="each homework's time budget (default the solver's 15)")
    probe.add_argument("--no-reuse-pilot", action="store_true", help="never take the pilot's attempts as the arm without the lesson")
    probe.add_argument("--hosted", action="store_true", help="a one-off comparison on the configured hosted model (Claude), never the learning loop; needs --yes and --cap-eur")
    probe.add_argument("--yes", action="store_true")
    probe.add_argument("--cap-eur", type=float, default=None, help="with --hosted: the most this whole run may spend")
    probe.add_argument("--cap-eur-each", type=float, default=0.5, help="with --hosted: the most one homework may spend")
    probe.add_argument("--limit", type=int, default=None, help="at most this many new homeworks in this process, then exit 3 (loop it: a fresh process a homework)")
    probe.add_argument("--pause", action="store_true", help="ask a running probe to stop after the homework it is on")
    probe.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/learning_v1/results/")
    args = parser.parse_args(argv)
    if args.command == "probe":
        return _probe(args)
    if args.command == "pilot":
        return _pilot(args)
    if args.command == "draft":
        return _draft(args)
    if args.command == "refusal":
        from .refusal import run

        report = run()
        _keep(report)  # every gate run is kept (the owner's rule), beside the test runs
        for row in report["cases"]:
            print(f"{'refused' if row['refused'] else 'ALLOWED':8} {row['case']}: {row['detail']}")
        print(f"{report['refused']} of {report['of']} refused")
        return 0 if report["passed"] else 1
    from .. import experience
    from ..settings import Settings
    from .scoreboard import SealedSpent, baseline

    before = experience.current_log()  # every attempt is a run of the learning agent in the workspace's record
    if before is None:
        experience.install(experience.open_log(Settings.load().agent_home))
    try:
        result = baseline(args.family, args.split, args.n, write=args.write)
    except SealedSpent as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if before is None:
            experience.install(None)
    print(json.dumps({"id": result["id"], "path": result.get("path"), "verdict": result["verdict"], "summary": result["summary"]}, indent=1))
    return 1 if result["verdict"].startswith("fail") else 0


def _pilot(args) -> int:
    from .. import experience
    from ..settings import Settings
    from .pilot import pause_file, run_pilot
    from .scoreboard import _store

    home = Path(Settings.load().agent_home)
    if args.pause:
        pause_file(home).parent.mkdir(parents=True, exist_ok=True)
        pause_file(home).write_text("pause\n", encoding="utf-8")
        print("the pilot stops after the homework it is on; run the same pilot command again to go on")
        return 0
    if not args.model and not args.hosted:
        print("--model is needed", file=sys.stderr)
        return 2
    pause_file(home).unlink(missing_ok=True)  # a start means go: a pause asked while nothing ran is not this run's
    got = _local(args)
    if isinstance(got, int):
        return got
    client, window = got
    before = experience.current_log()
    if before is None:
        experience.install(experience.open_log(home))
    try:
        budget = {**({"max_minutes": args.max_minutes} if args.max_minutes else {}), **({"max_eur": args.cap_eur_each} if args.hosted else {})} or None
        result = run_pilot(home, client, args.model, args.family, args.n, log=lambda *a: print(*a, flush=True), budget=budget, limit=args.limit,
                           cap_eur=args.cap_eur if args.hosted else None)
        result["context_window"] = window
    finally:
        if before is None:
            experience.install(None)
    if args.write and result.get("status") != "paused":
        path = _store(result, args.family, "pilot")
        result["path"] = str(path)
    print(json.dumps({k: result.get(k) for k in ("id", "path", "model", "verdict", "summary")}, indent=1))
    if result.get("status") == "paused":
        return 3  # more to do: the same command again goes on (a loop runs one homework a process)
    return 0 if result["verdict"].startswith("ok") else 2 if result["verdict"].startswith("invalid") else 1


def _hosted(args) -> tuple[object, None] | int:
    """The intern's client on the configured hosted model, for a one-off comparison only (the learning loop's solver is
    local): .env loaded as the server loads it, a priced model named before any spend, --yes and a cap. Its runs are a
    frontier model's: the store keeps them silver, never in a training set."""
    import os

    from dotenv import load_dotenv

    from ..models import prices, settings
    from ..models.gateway import for_workspace
    from ..settings import Settings

    if not args.yes or not args.cap_eur or args.cap_eur <= 0:
        print("a hosted model runs only with --yes and --cap-eur > 0 (the whole run)", file=sys.stderr)
        return 2
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    gateway = for_workspace(Path(Settings.load().agent_home))
    client = gateway.client("intern")
    tier = settings.tier(gateway.route("intern")[0])
    if client is None:
        print("no model serves the intern", file=sys.stderr)
        return 2
    if settings.is_local(tier.get("base_url") or ""):
        print("the intern is served locally: run without --hosted", file=sys.stderr)
        return 2
    if prices.price_of(tier.get("model") or "") is None:
        print(f"{tier.get('model')} has no price: a cap cannot hold", file=sys.stderr)
        return 2
    if args.model and args.model != tier.get("model"):
        print(f"--model {args.model} is not the served model {tier.get('model')}", file=sys.stderr)
        return 2
    args.model = tier.get("model")
    print(f"hosted: {args.model} at {settings._host(tier.get('base_url') or '')}; at most EUR {args.cap_eur:.2f} this run, "
          f"EUR {args.cap_eur_each:.2f} a homework", file=sys.stderr, flush=True)
    return client, None


def _local(args) -> tuple[object, int | None] | int:
    """The intern's client on a local model, checked: on this machine, a window that holds its prompt, answering. An exit
    code when it is not. With --hosted, the capped hosted client instead."""
    if getattr(args, "hosted", False):
        return _hosted(args)
    from .solver import ContextTooSmall, NotLocal, check_context, local_gateway

    from ..settings import Settings

    try:
        gateway = local_gateway(args.model, args.base_url, home=Path(Settings.load().agent_home))
    except NotLocal as exc:
        print(str(exc), file=sys.stderr)
        return 2
    client = gateway.client("intern")
    if client is None:
        print("no model serves the intern: is the local server running with the model pulled?", file=sys.stderr)
        return 2
    try:
        client.complete([{"role": "user", "content": "Reply with the word ready."}], [], max_tokens=200)
    except Exception as exc:  # noqa: BLE001
        print(f"the local model did not answer ({type(exc).__name__}: {str(exc)[:200]}): start the server and pull the model", file=sys.stderr)
        return 2
    try:
        window = check_context(args.model, args.base_url)
    except ContextTooSmall as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"{args.model}: served with a {window or 'unknown'}-token window", file=sys.stderr, flush=True)
    return client, window


def _probe(args) -> int:
    from .. import experience
    from ..settings import Settings
    from .probe import pause_file, run_probe
    from .scoreboard import _store

    home = Path(Settings.load().agent_home)
    if args.pause:
        pause_file(home).parent.mkdir(parents=True, exist_ok=True)
        pause_file(home).write_text("pause\n", encoding="utf-8")
        print("the probe stops after the homework it is on; run the same probe command again to go on")
        return 0
    if (not args.model and not args.hosted) or not args.bullet:
        print("--model (or --hosted) and --bullet are needed", file=sys.stderr)
        return 2
    pause_file(home).unlink(missing_ok=True)
    got = _local(args)
    if isinstance(got, int):
        return got
    client, window = got
    before = experience.current_log()
    if before is None:
        experience.install(experience.open_log(home))
    try:
        result = run_probe(home, client, args.model, args.bullet, args.family, args.n, args.repeats, lineage=args.lineage,
                           budget={**({"max_minutes": args.max_minutes} if args.max_minutes else {}), **({"max_eur": args.cap_eur_each} if args.hosted else {})} or None,
                           reuse_pilot=not args.no_reuse_pilot, log=lambda *a: print(*a, flush=True), limit=args.limit,
                           cap_eur=args.cap_eur if args.hosted else None)
        result["context_window"] = window
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    finally:
        if before is None:
            experience.install(None)
    if args.write and result.get("status") != "paused":
        result["path"] = str(_store(result, args.family, "probe"))
    print(json.dumps({k: result.get(k) for k in ("id", "path", "model", "verdict", "summary", "bullet")}, indent=1, ensure_ascii=False))
    if result.get("status") == "paused":
        return 3  # more to do: the same command again goes on (a loop runs one homework a process)
    return 0 if not result["verdict"].startswith("invalid") else 2


def _keep(report: dict) -> None:
    from datetime import datetime, timezone

    root = Path(__file__).resolve().parents[2] / "training_records" / "test_runs"
    at = datetime.now(timezone.utc)
    try:
        folder = root / f"{at:%Y-%m-%d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"refusal-{at:%Y%m%dT%H%M%S%fZ}.json").write_text(json.dumps({**report, "at": at.isoformat(timespec="seconds")}, indent=1) + "\n", encoding="utf-8")
    except OSError:
        pass  # a report that cannot be kept never hides the verdict


def _draft(args) -> int:
    from .. import experience
    from ..settings import Settings
    from .drafting import draft_lessons

    home = Path(Settings.load().agent_home)
    client = None
    if args.model:
        if not args.yes or not args.cap_eur or args.cap_eur <= 0:
            print("a model drafts only with --yes and --cap-eur > 0 (per challenge); without --model the rules draft, free", file=sys.stderr)
            return 2
        from dotenv import load_dotenv

        from ..models import prices, settings
        from ..models.gateway import for_workspace

        load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)  # the provider and its key, as the server reads them
        gateway = for_workspace(home)
        client = gateway.client("reflector")
        if client is None:
            print("no model serves the reflector purpose: configure a tier, or draft with the rules", file=sys.stderr)
            return 2
        tier = settings.tier(gateway.route("reflector")[0])
        if not settings.is_local(tier.get("base_url") or "") and prices.price_of(tier.get("model") or "") is None:
            # a cap counts euros: a model with no price spends without counting, so the cap would never stop it
            print(f"the reflector would be served by {tier.get('model')} at {settings._host(tier.get('base_url') or '')}, which has no price: "
                  "a cap cannot hold; add its price or route the purpose to a priced model", file=sys.stderr)
            return 2
        print(f"the reflector is {tier.get('model')} at {settings._host(tier.get('base_url') or '')}", file=sys.stderr)
        print(f"the reflector may spend up to EUR {args.cap_eur:.2f} a challenge, {args.n} challenges", file=sys.stderr)
    before = experience.current_log()
    if before is None:
        experience.install(experience.open_log(home))
    try:
        result = draft_lessons(home, args.family, args.n, client=client, cap_eur=args.cap_eur or 0.10)
    finally:
        if before is None:
            experience.install(None)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
