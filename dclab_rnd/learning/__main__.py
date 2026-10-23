"""``python -m dclab_rnd.learning baseline|refusal|draft``: the baseline (H0), the refusal test (H1), drafting lessons (H2)."""

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
    args = parser.parse_args(argv)
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
        from ..models.gateway import for_workspace

        client = for_workspace(home).client("reflector")
        if client is None:
            print("no model serves the reflector purpose: configure a tier, or draft with the rules", file=sys.stderr)
            return 2
        print(f"the reflector may spend up to EUR {args.cap_eur:.2f} a challenge, {args.n} challenges", file=sys.stderr)
    before = experience.current_log()
    if before is None:
        experience.install(experience.open_log(home))
    try:
        result = draft_lessons(home, args.family, args.n, client=client, cap_eur=args.cap_eur or 0.10)
    finally:
        if before is None:
            experience.install(None)
    if client is not None:
        result["spent_eur"] = round(float(getattr(client, "spent_eur", 0.0) or 0.0), 4)
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
