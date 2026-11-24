"""The team-arms comparison (B4 of the team workspace, 2026-11-24): rules (R), helper (H) and agent (A) brains for the
team roles, on the same written cases, data, tools, permissions and budgets, scored by code from the records.

    python -m dclab_rnd.agent_eval.team_arms plan --set dev --models qwen/qwen3.8-flash          # what would run, and its cost; spends nothing
    python -m dclab_rnd.agent_eval.team_arms pin                                               # pin the gate's terms (before any run)
    python -m dclab_rnd.agent_eval.team_arms run --set quick --yes --cap-eur 0.5 --models qwen/qwen3.8-flash
    python -m dclab_rnd.agent_eval.team_arms run --set dev --roles lead,data_science --families control,paraphrase --yes --cap-eur 0.5
    python -m dclab_rnd.agent_eval.team_arms report --set dev                                  # the comparison, into the campaign
    python -m dclab_rnd.agent_eval.team_arms receipt RUN --snapshot SNAPSHOT                    # after make archive
    python -m dclab_rnd.agent_eval.team_arms schema                                            # the case format's JSON Schema

Parts: :mod:`cases` (the format; ``CASES.md`` is the case author's guide), :mod:`play` (one case, one arm, one model, on a
fresh workspace through the real team runtime), :mod:`score` (code only, from the records), :mod:`gate` (the pinned
winner rule and limits), :mod:`report` (the comparison) and :mod:`judge` (an optional free, non-Claude judge).

The rules that hold whatever the command: no Claude model anywhere (an arm, a judge: ``claude``, ``anthropic`` or
``fable`` in a model id, or Claude's endpoint, is refused); a run needs ``--yes`` and ``--cap-eur`` (at most 0.5) and
stops when the cap is reached, recording what ran; every model request goes through the run's gateway under
``agent_runs/live_tests/<run>/`` with its thinking kept, and the run checks that every answered request kept it; the
sealed set runs once (a committed marker is written before its first play) and whole (no split); the gate must be pinned.
The cases are written by an independent session, never by the builder of the arms: ``cases/quick_example.json`` is a
harness smoke test, never scored.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import cases as C
from . import gate
from . import play as P
from .play import ARM_NAMES, ARMS, Refused, RunBudget, model_refusal, slug

CAMPAIGN = gate.CAMPAIGN.name
ROOT = gate.ROOT
RUN_ROOT = ROOT / "agent_runs" / "live_tests"
DEFAULT_MODELS = ("qwen/qwen3.8-flash",)  # the cheap one; the strong open model is named on the command line
STRONG_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
JUDGE_MODEL = "nvidia/nemotron-3.5-super-vl:free"
MAX_CAP_EUR = gate.LIMITS["max_eur_per_run"]
# the tokens one play of an arm is guessed to use (for the estimate only; the cap is what holds)
GUESS = {"H": {"calls": 4, "in": 2500, "out": 1500}, "A": {"calls": 6, "in": 6000, "out": 1200}}
AGENT_TURNS_PER_ROLE = 3
PROVIDER_STOP = "the provider refused the account: "  # B4.1: what run.json's stopped says (go on with --resume once it is fixed)
# the review of B4.1 (F5): a run with a play failed for the provider's reason has not finished, and says so
PROVIDER_FAILED_STOP = "plays failed for the provider's reason (a rate limit, a provider error, the network): go on with --resume"
PROVIDER_TRIES = 2  # the review of B4.1 (F6): a play the provider failed this often already is final the next time (never a loop)
REQUEST_TOKENS = {"in": 30000, "out": 8000}  # one request's upper bound for the cap's margin (a long agent turn, its thinking)


# ---------------------------------------------------------------------------------------------------------- choosing
def select(case_set: C.CaseSet, roles: Optional[list[str]] = None, families: Optional[list[str]] = None) -> list[C.Case]:
    """The cases of the split: every case naming one of ``roles`` (all when None), of one of ``families`` (all when None)."""
    out = []
    for c in case_set.cases:
        if roles and not set(c.roles) & set(roles):
            continue
        if families and c.family not in families:
            continue
        out.append(c)
    return out


def estimate_eur(cases: list[C.Case], arms: list[str], models: list[str]) -> Optional[float]:
    """A guess of the run's euros from the models' prices (None when a model has no price: the cap then cannot hold)."""
    from dclab_rnd.models import prices

    total = 0.0
    for model in models:
        price = prices.price_of(model)
        if price is None:
            return None
        for arm in arms:
            if arm == "R":
                continue
            g = GUESS[arm]
            for c in cases:
                calls = g["calls"] * (len(c.roles) * AGENT_TURNS_PER_ROLE if arm == "A" else 1)
                total += calls * (g["in"] * price.input + g["out"] * price.output) / 1e6
    return round(total, 4)


def plan(set_name: str, arms: list[str], models: list[str], roles: Optional[list[str]] = None, families: Optional[list[str]] = None,
         folder: Path = C.CASE_DIR) -> dict[str, Any]:
    """What a run would do: the cases (ids, families, roles), the plays, the models and the euro estimate. Spends nothing."""
    out: dict[str, Any] = {"gate": gate.GATE_ID, "pinned": gate.pinned() is None, "pin_problem": gate.pinned(), "set": set_name,
                           "arms": arms, "models": models, "limits": gate.LIMITS}
    why = C.check_pin(set_name, folder)
    if why:
        out["cases"] = why
        return out
    if set_name == "sealed":  # counts only: nobody who changes an arm reads a sealed case
        chosen = C.load(set_name, folder).cases
        out.update(cases=len(chosen), families=dict(sorted({c.family: sum(1 for x in chosen if x.family == c.family) for c in chosen}.items())),
                   sealed_started=gate.SEALED_MARKER.exists())
    else:
        chosen = select(C.load(set_name, folder), roles, families)
        out["cases"] = [{"id": c.id, "family": c.family, "roles": c.roles, "entry": c.entry} for c in chosen]
    model_arms = [a for a in arms if a != "R"]
    out["plays"] = len(chosen) * (("R" in arms) + len(model_arms) * len(models))
    out["estimate_eur"] = estimate_eur(chosen, arms, models) if model_arms else 0.0
    out["refused_models"] = [m for m in models if model_refusal(m)]
    return out


# ---------------------------------------------------------------------------------------------------------- thinking
def _lines(folders: Iterable[Path], pattern: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for folder in folders:
        for path in sorted(Path(folder).glob(pattern)):
            rows += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows


def _thought(row: dict[str, Any]) -> bool:
    return bool(row.get("thinking") or row.get("reasoning") or row.get("reasoning_details") or row.get("thinking_redacted"))


def thinking_check(folders: Iterable[Path], vaults: Iterable[Path] = ()) -> dict[str, Any]:
    """Every answered model request kept whole with its thinking (CLAUDE.md rule 9), read from what the gateway kept:

    - each answered request in a usage log has its exchange line (a request the gateway failed to keep is missing, never
      silently passed: the review of B4, F14);
    - each exchange line kept here is whole and holds its thinking (or its redacted thinking);
    - a sealed run's line here is its shape only: its whole copy, with the thinking, must be in the vault (by request id)."""
    folders, vaults = list(folders), list(vaults)
    rows = _lines(folders, "**/model_exchanges/**/*.jsonl")
    used = [u for u in _lines(folders, "**/usage.jsonl") if u.get("outcome") == "ok"]
    ok = [r for r in rows if r.get("outcome") == "ok" and not r.get("in_vault")]
    shapes = [r for r in rows if r.get("outcome") == "ok" and r.get("in_vault")]
    kept = {str(v.get("model_request")): v for v in _lines(vaults, "**/model_exchanges/**/*.jsonl") if v.get("outcome") == "ok"}
    thought = [r for r in ok if _thought(r)]
    whole = [r for r in ok if r.get("whole") is True]
    vault_missing = [r for r in shapes if not _thought(kept.get(str(r.get("model_request"))) or {})]
    missing = sorted({str(r.get("purpose")) for r in ok if r not in thought} | {str(r.get("purpose")) for r in vault_missing})
    unkept = max(0, len(used) - len(ok) - len(shapes))
    return {"exchanges": len(rows), "ok": len(ok), "with_thinking": len(thought), "whole": len(whole), "vault_shape_lines": len(shapes),
            "vault_without_thinking": len(vault_missing), "answered_in_usage": len(used), "answered_not_kept": unkept,
            "without_thinking": missing, "not_whole": len(ok) - len(whole),
            "verified": len(thought) == len(ok) and len(whole) == len(ok) and not vault_missing and not unkept}


# ---------------------------------------------------------------------------------------------------------- one run
def sealed_refusal(run_root: Optional[Path] = None, marker: Optional[Path] = None) -> Optional[str]:
    run_root, marker = Path(run_root or RUN_ROOT), Path(marker or gate.SEALED_MARKER)
    if Path(marker).exists() or list(Path(run_root).glob("*-team-arms-sealed*")):
        return "the sealed set has run already: it runs once"
    return None


def begin_sealed(run_dir: Path, facts: dict[str, Any], run_root: Optional[Path] = None, marker: Optional[Path] = None) -> None:
    """Write the run-once marker before the sealed set's first play (exclusively: a second start, even at the same moment,
    fails). Refused when the set has run already."""
    why = sealed_refusal(run_root, marker)
    if why:
        raise Refused(why)
    marker = Path(marker or gate.SEALED_MARKER)
    Path(marker).parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "x", encoding="utf-8") as handle:  # "x": created here or not at all
        handle.write(json.dumps({**facts, "run": Path(run_dir).name}, indent=1, default=str) + "\n")


def margin_eur(models: list[str]) -> float:
    """One request's upper bound on the dearest model (as the gateway estimates it): less than this left is the cap reached."""
    from dclab_rnd.models import prices

    worst = 0.0
    for model in models:
        price = prices.price_of(model)
        if price is not None:
            worst = max(worst, (REQUEST_TOKENS["in"] * price.input + REQUEST_TOKENS["out"] * price.output) / 1e6)
    return round(worst, 6)


# What the arms and the scorer run on (the third review's M1): the team runtime, the harness, the team prompts, the core
# contracts the team agents decide with, the agent loop and the gateway. The case files are pinned on their own (cases/).
CODE_PATHS = ("dclab_rnd/teamspace", "dclab_rnd/agent_eval/team_arms", "dclab_rnd/prompts/team_*.md", "dclab_core/team_brain.py",
              "dclab_core/team_work.py", "dclab_core/target_contract.py", "dclab_rnd/agents", "dclab_rnd/models")
NOT_CODE = ("__pycache__", "dclab_rnd/agent_eval/team_arms/cases/")
CODE_SUFFIXES = (".py", ".md", ".json", ".sha256", ".toml", ".yaml", ".yml", ".txt")  # what code is made of: never a .DS_Store, a swap file or
# an editor's backup, which a person's machine rewrites between two sittings (the fourth review's P1)


def code_files(root: Path = ROOT, paths: Iterable[str] = CODE_PATHS) -> list[tuple[str, str]]:
    """Every file under the code paths as it is on disk, tracked or not, as sorted (path, sha256 of its bytes)."""
    import hashlib

    found: set[Path] = set()
    for pattern in paths:
        for hit in Path(root).glob(pattern):
            found |= {hit} if hit.is_file() else {f for f in hit.rglob("*") if f.is_file()}
    out = []
    for f in found:
        rel = f.relative_to(root).as_posix()
        if any(x in rel for x in NOT_CODE) or f.suffix not in CODE_SUFFIXES or any(part.startswith(".") for part in rel.split("/")):
            continue
        out.append((rel, hashlib.sha256(f.read_bytes()).hexdigest()))
    return sorted(out)


def code_state(root: Path = ROOT) -> dict[str, Any]:
    """What the arms and the scorer run on: one hash over the files' contents (the third review's M1: a commit, or a file going
    from untracked to tracked with the same bytes, changes nothing a play does, so it never blocks a resume), and the commit
    for information only."""
    import hashlib
    import subprocess

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=30).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        commit = None
    files = code_files(root)
    return {"content_sha256": hashlib.sha256(json.dumps(files).encode()).hexdigest(), "files": len(files), "paths": list(CODE_PATHS),
            "commit": commit}


def _state(folder: Path) -> str:
    """Where one play stands: ``final`` (scored, and not cut by the cap: complete or not, it is never played again),
    ``unscored`` (played to its end but its score failed: scored again from its records), or ``replay`` (cut by the cap,
    or never ended: a killed process). The re-review's N2: a crash, a failed delivery or a score error is final.
    B4.1: a play the provider refused or failed (``provider_refused``, ``provider_failed``: a turn fell back to the rules for
    the provider's reason) is ``replay`` too: it counts neither for nor against an arm."""
    played = folder / P.PLAY
    if not played.exists():
        return "replay"
    stopped = json.loads(played.read_text(encoding="utf-8")).get("stopped")
    if stopped in P.REPLAYABLE:
        return "replay"
    return "final" if (folder / "score.json").exists() else "unscored"


def _provider_tries(folder: Path) -> int:
    """How often this play was played before and failed for the provider's reason (its ``.cut-N`` folders)."""
    out = 0
    for cut in Path(folder).parent.glob(Path(folder).name + ".cut-*"):
        try:
            out += json.loads((cut / P.PLAY).read_text(encoding="utf-8")).get("stopped") == "provider_failed"
        except (OSError, ValueError):
            continue
    return out


def _row(folder: Path, case: C.Case, arm: str, model: Optional[str]) -> dict[str, Any]:
    row: dict[str, Any] = {"case": case.id, "arm": arm, "model": model}
    played = json.loads((folder / P.PLAY).read_text(encoding="utf-8")) if (folder / P.PLAY).exists() else {}
    row.update(seconds=played.get("seconds"), euros=played.get("euros"))
    if (folder / "score.json").exists():
        scored = json.loads((folder / "score.json").read_text(encoding="utf-8"))
        row.update(main=scored["main"], complete=scored["complete"], hard_zeros=scored["hard_zero_codes"])
        if scored.get("score_error"):
            row["score_error"] = scored["score_error"]
    else:
        row.update(main=0, complete=False, hard_zeros=[], score_error=played.get("score_error") or "not scored")
    return row


def run_cases(case_set: C.CaseSet, cases: list[C.Case], arms: list[str], models: list[str], run_dir: Path, *, cap_eur: float,
              gateway_for: Optional[Callable[[Path], Any]], judge_for: Optional[Callable[[Path], Any]] = None,
              exchanges: Iterable[Path] = (), manifest: Optional[dict[str, Any]] = None, resume: bool = False) -> dict[str, Any]:
    """Play every case under every arm (and model), score each, judge it when asked; stop at the cap. ``gateway_for``
    builds a play's gateway in its folder (the live one, or a scripted one in tests). Writes ``run.json``.

    ``resume``: the same run again, in a new sitting with its own cap (a set larger than one cap: the sealed set), on the
    same code and the same pinned cases as its first sitting (else refused). Only a play cut by the cap, refused or failed by
    the provider (B4.1: the run stops at once when the provider refuses the account), or never ended is played again, from a fresh workspace, the cut one kept aside (``<folder>.cut-N``); every other play is final, complete or
    not (a crash is not a second try). ``finished`` is true once every play is final. A sealed run's ``run.json`` holds
    counts only, never a case's row."""
    from . import judge as J
    from . import score as S

    exchanges = list(exchanges)
    manifest = dict(manifest or {})
    sealed = case_set.set == "sealed"
    run_dir.mkdir(parents=True, exist_ok=True)
    budget = RunBudget(cap_eur, margin_eur(models))
    before = json.loads((run_dir / "run.json").read_text(encoding="utf-8")) if resume and (run_dir / "run.json").exists() else {}
    if resume and before.get("finished"):
        raise Refused("this run finished: every play is final")
    if before and before.get("fingerprint") != gate.fingerprint():
        raise Refused("the gate changed since this run began: a run is scored under one gate")
    if before:  # never hidden by the newer manifest: a difference refuses the sitting
        if "code" in manifest and (before.get("code") or {}).get("content_sha256") != (manifest["code"] or {}).get("content_sha256"):
            raise Refused("the code changed since this run began: a run goes on only on the code it began with")
        if "cases_sha256" in manifest and before.get("cases_sha256") != manifest["cases_sha256"]:
            raise Refused("the pinned cases changed since this run began: a run goes on only on the cases it began with")
        if "split" in manifest and before.get("split") != manifest["split"]:
            raise Refused(f"this run's split is {before.get('split')}: a run goes on with the split it began with")
        if before.get("arms") not in (None, arms) or before.get("models") not in (None, models):
            raise Refused(f"this run plays arms {before.get('arms')} on models {before.get('models')}: a run goes on with them (the third review's M3)")
    sitting = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cap_eur": cap_eur}
    doc = {**before, **manifest, "run": run_dir.name, "set": case_set.set, "arms": arms, "models": models, "cap_eur": cap_eur,
           "fingerprint": gate.fingerprint(), "cases": len(cases) if sealed else [{"id": c.id, "family": c.family, "roles": list(c.roles)} for c in cases],
           "sittings": [*(before.get("sittings") or []), sitting], "stopped": "interrupted", "finished": False}
    plays = [(c, a, m, run_dir / "plays" / c.id / (a if m is None else f"{a}__{slug(m)}")) for c in cases for a in arms
             for m in ([None] if a == "R" else models)]

    def save() -> None:
        rows = [_row(f, c, a, m) for c, a, m, f in plays if f.exists() and _state(f) != "replay"]
        doc["plays"] = ({"final": len(rows), "complete": sum(1 for r in rows if r.get("complete")), "of": len(plays)} if sealed else rows)
        (run_dir / "run.json").write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")

    save()
    stopped: Optional[str] = None
    try:
        for case, arm, model, folder in plays:
            state = _state(folder) if folder.exists() else "replay"
            if state == "final":
                continue
            if state == "unscored":  # played to its end: scored again from its records, never played again
                try:
                    S.score(folder, case, exchanges)
                except Exception as exc:  # noqa: BLE001 - a second failure is final (the third review's M2): incomplete, never blocking
                    S.write_unscorable(folder, case, f"{type(exc).__name__}: {str(exc)[:300]}")
                continue
            if budget.reached():
                stopped = "the cap was reached"
                break
            if budget.provider_refused:
                stopped = PROVIDER_STOP + budget.provider_refused
                break
            if folder.exists():  # cut by the cap (or a killed process) in an earlier sitting: kept aside, played again fresh
                n = 1 + len(list(folder.parent.glob(folder.name + ".cut-*")))
                folder.rename(folder.with_name(f"{folder.name}.cut-{n}"))
            gateway = gateway_for(folder) if (arm != "R" and gateway_for is not None) else None
            if arm != "R" and gateway is None:
                raise Refused("a model arm needs the run's gateway")
            record = P.play(case, arm, model, folder, gateway=gateway, budget=budget if arm != "R" else None, set_name=case_set.set)
            if record.get("stopped") == "provider_failed" and _provider_tries(folder) >= PROVIDER_TRIES:
                # failed for the provider on every try: final now, incomplete (the arm did not play it to the end), and why
                record.update(stopped="provider_failed_again", provider_tries=_provider_tries(folder) + 1)
                (folder / P.PLAY).write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            if record.get("stopped") not in P.REPLAYABLE:  # B4.1: a play the provider refused or failed is never scored
                try:  # a scoring error is the play's record, never the run's end
                    scored = S.score(folder, case, exchanges)
                except Exception as exc:  # noqa: BLE001
                    record["score_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                    (folder / P.PLAY).write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
                else:
                    if judge_for is not None and scored["complete"]:
                        J.judge_play(folder, case, judge_for(folder))
            save()
            print(f"  {sum(1 for *_, f in plays if f.exists() and _state(f) == 'final')} of {len(plays)} plays final", flush=True)
            if record.get("stopped") == "cap":
                stopped = "the cap was reached"
                break
            if record.get("stopped") == "provider_refused":  # B4.1: at once: every later model play would be the rules'
                stopped = PROVIDER_STOP + (budget.provider_refused or "a request was refused for the account")
                break
    except Exception as exc:  # noqa: BLE001
        stopped = f"{type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        sitting.update(stopped=stopped, spent_eur=round(budget.spent_eur(), 6), ended=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if stopped is None and any(f.exists() and _state(f) == "replay" for *_, f in plays):
            stopped = PROVIDER_FAILED_STOP  # F5: never read as a finished run with a case "not played to the end"
            sitting["stopped"] = stopped
        doc["stopped"] = stopped
        doc["spent_eur"] = round(sum(float(x.get("spent_eur") or 0.0) for x in doc["sittings"]), 6)
        doc["finished"] = stopped is None and all(f.exists() and _state(f) == "final" for *_, f in plays)
        doc["thinking"] = thinking_check([run_dir], exchanges)
        doc["ended"] = sitting["ended"]
        save()
    return doc


# ---------------------------------------------------------------------------------------------------------- the live wiring
def _live_gateway(run_dir: Path) -> Callable[[Path], Any]:
    from ..harness_ablation import run_gateway

    def make(folder: Path) -> Any:
        folder.mkdir(parents=True, exist_ok=True)
        gateway = run_gateway(folder)  # its usage log and every exchange, with its thinking, in the play's folder
        gateway.invented_tables = None if os.environ.get("DCLAB_SEALED_RUN") else "team arms: the team workspace's invented fixtures"
        return gateway

    return make


def _preflight(models: list[str], run_dir: Path) -> Optional[str]:
    """Before any play: every arm model priced (the cap holds only on a price), and served by a non-Claude endpoint."""
    from dclab_rnd.models import prices

    from ..harness_ablation import price_snapshot

    for model in models:
        if prices.price_of(model) is None:
            try:
                price_snapshot(run_dir, model)
            except Exception as exc:  # noqa: BLE001
                return f"{model} has no price ({type(exc).__name__}): the cap cannot hold"
        if prices.price_of(model) is None:
            return f"{model} has no price: the cap cannot hold"
        probe = _live_gateway(run_dir)(run_dir / "preflight")
        for purpose in ("team_helper", "team_agent"):
            client = probe.client(purpose, model=model)
            if client is None:
                return f"no model serves {purpose} (no key, or live calls are off)"
            endpoint = str((getattr(client, "tier", None) or {}).get("base_url") or "")
            why = model_refusal(getattr(client, "model", "")) or (f"{purpose} is served by {endpoint}, Claude's endpoint"
                                                                   if any(w in endpoint.lower() for w in ("anthropic", "claude")) else None)
            if why:
                return why
    return None


def _judge_factory(model: str, cap_eur: float) -> Callable[[Path], Any]:
    from dclab_rnd.models import settings as model_settings
    from dclab_rnd.models.gateway import Bound

    from ..harness_ablation import judge_tier, run_gateway

    tier = judge_tier(model)
    if tier is None:
        raise Refused("no key serves the judge's endpoint, or the provider is blocked")

    def make(folder: Path) -> Any:
        gateway = run_gateway(folder / "judge")
        bound = Bound(gateway, "exam_judge", model_settings.purpose("exam_judge"), tier, None, f"team-arms-judge:{folder.name}", fixed=True)
        bound.run_limit_eur = cap_eur
        return bound

    return make


def _cmd_run(args: argparse.Namespace) -> int:
    arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
    models = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    roles = [r.strip() for r in args.roles.split(",")] if args.roles else None
    families = [f.strip() for f in args.families.split(",")] if args.families else None
    bad = [a for a in arms if a not in ARMS]
    if bad:
        print(f"refused: unknown arms {bad}", file=sys.stderr)
        return 2
    for m in [*models, *([args.judge] if args.judge else [])]:
        why = model_refusal(m)
        if why:
            print(f"refused: {why}", file=sys.stderr)
            return 2
    if any(a != "R" for a in arms) and not models:
        print("refused: a model arm needs --models", file=sys.stderr)
        return 2
    if args.cap_eur is None or not 0 < args.cap_eur <= MAX_CAP_EUR:
        print(f"refused: --cap-eur is required, above 0 and at most {MAX_CAP_EUR} euro per run", file=sys.stderr)
        return 2
    why = gate.pinned()
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    sealed = args.set == "sealed"
    if sealed and (roles or families):
        print("refused: the sealed set runs whole, once (no split)", file=sys.stderr)
        return 2
    if sealed and (arms != gate.TERMS["sealed_arms"] or models != gate.TERMS["sealed_models"]):
        print(f"refused: the sealed set runs every arm on the pinned models, whole: --arms {','.join(gate.TERMS['sealed_arms'])} "
              f"--models {','.join(gate.TERMS['sealed_models'])}", file=sys.stderr)
        return 2
    resume = _resumable(args.resume, args.set) if args.resume else None
    if isinstance(resume, str):
        print(f"refused: {resume}", file=sys.stderr)
        return 2
    if sealed and resume is None:
        why = sealed_refusal()
        if why:
            print(f"refused: {why} (a sitting cut by the cap goes on with --resume <its run>)", file=sys.stderr)
            return 2
    why = C.check_pin(args.set)
    if why:
        print(f"refused: {why}", file=sys.stderr)
        return 2
    if args.judge:
        from .judge import judge_refusal

        why = judge_refusal(args.judge, models)
        if why:
            print(f"refused: {why}", file=sys.stderr)
            return 2
    shown = plan(args.set, arms, models, roles, families)
    if not args.yes:
        print(json.dumps({"would_run": shown, "cap_eur": args.cap_eur}, indent=1, default=str))
        return 0
    case_set = C.load(args.set)
    chosen = case_set.cases if sealed else select(case_set, roles, families)
    if not chosen:
        print("refused: no case in this split", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    for name in ("DCLAB_DATABASE_URL", "DCLAB_STUDIO_HOME", "DCLAB_DRAFT_HOME", "DCLAB_INTERN_HOME"):
        os.environ.pop(name, None)  # the plays stay in the run's folder, never in a shared database or workspace
    from ..harness_ablation import owner_allows

    refused = owner_allows(args.cap_eur)
    if refused:
        print(f"refused: {refused}", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc)
    if resume is not None:  # the review of B4.1: a provider paid again within the pause after its refusal is asked again
        from dclab_rnd.models import client as model_client

        model_client.resume()
    run_dir = resume or RUN_ROOT / f"{stamp:%Y-%m-%d-%H%M%S}-team-arms-{args.set}"
    staging = RUN_ROOT / f"{stamp:%Y-%m-%d-%H%M%S}-team-arms-preflight-{args.set}"
    if models:  # before the sealed marker: a refused model never spends the sealed set's one run
        why = _preflight(models, staging)
        if why:
            print(f"refused: {why}", file=sys.stderr)
            return 2
    try:
        judge_for = _judge_factory(args.judge, args.cap_eur) if args.judge else None
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    vault: list[Path] = []
    if sealed and resume is None:
        try:  # the marker first, then the run's folder (the marker refuses a second start, a folder of a first one)
            begin_sealed(run_dir, {"started": stamp.isoformat(timespec="seconds"), "code": code_state(), "fingerprint": gate.fingerprint(),
                                   "cases_sha256": C.pinned_sha("sealed")})
        except (Refused, FileExistsError) as exc:
            print(f"refused: {exc}", file=sys.stderr)
            return 2
    if sealed:
        from dclab_rnd import experience
        from dclab_rnd.settings import Settings

        place = Path(Settings().agent_home) / "exam_vault" / f"team_arms_{run_dir.name}"
        experience.sealed_run("sealed_team_arms", vault=place)
        vault = [place]
    else:
        os.environ["DCLAB_LOG_PROMPTS"] = "1"
    run_dir.mkdir(parents=True, exist_ok=resume is not None)
    if staging.exists():  # each sitting's own preflight folder (the re-review's N7: a resumed run already has one)
        import shutil

        kept = run_dir / f"preflight-{stamp:%Y%m%d-%H%M%S}"
        shutil.move(str(staging), str(kept))
        prices_file = os.environ.get("DCLAB_PRICES_FILE", "")
        if prices_file.startswith(str(staging)):
            os.environ["DCLAB_PRICES_FILE"] = str(kept) + prices_file[len(str(staging)):]
    manifest = {"code": code_state(), "cases_sha256": C.pinned_sha(args.set) if args.set != "quick" else C.sha256(C.set_path("quick")),
                "split": {"roles": roles, "families": families}, "judge": args.judge, "evaluation_only": True,
                "smoke": not case_set.scored}
    try:
        doc = run_cases(case_set, chosen, arms, models, run_dir, cap_eur=args.cap_eur, gateway_for=_live_gateway(run_dir),
                        judge_for=judge_for, exchanges=vault, manifest=manifest, resume=resume is not None)
    except Refused as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run": str(run_dir.relative_to(ROOT)), "plays": doc["plays"] if sealed else len(doc["plays"]), "stopped": doc["stopped"], "spent_eur": doc["spent_eur"],
                      "thinking": doc["thinking"]}, indent=1, default=str))
    if not doc["thinking"]["verified"]:
        print("warning: an answered model request kept no thinking or was not kept whole (CLAUDE.md rule 9): see thinking in run.json",
              file=sys.stderr)
    if not doc["finished"]:
        split = "".join(f" --{k} {','.join(v)}" for k, v in (("roles", roles), ("families", families)) if v)
        print(f"the run stopped before every play ({doc['stopped']}): go on with run --set {args.set} --arms {','.join(arms)} "
              f"--models {','.join(models)}{split} --resume {run_dir.relative_to(ROOT)} --yes --cap-eur 0.5 "
              "(final plays are kept; plays cut by the cap, or refused or failed by the provider, are played again)")
    print(f"next: make archive, then python -m dclab_rnd.agent_eval.team_arms receipt {run_dir.relative_to(ROOT)} --snapshot <the snapshot it printed>")
    return 0 if doc["stopped"] is None and doc["thinking"]["verified"] else 1


def _resumable(name: str, set_name: str) -> Any:
    """The run folder a ``--resume`` names (a Path), or why it may not go on (a str). A sealed run goes on only as the run
    its marker names, until it finished."""
    run = Path(name) if Path(name).is_absolute() else ROOT / name
    doc = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else None
    if doc is None:
        return f"{name} is no team-arms run"
    if doc.get("set") != set_name:
        return f"{name} is a run of the {doc.get('set')} set"
    if doc.get("finished"):
        return f"{name} finished: every play is complete"
    if set_name == "sealed":
        marker = json.loads(gate.SEALED_MARKER.read_text(encoding="utf-8")) if gate.SEALED_MARKER.exists() else {}
        if marker.get("run") != run.name:
            return "only the sealed run the marker names goes on"
    return run


def _cmd_report(args: argparse.Namespace) -> int:
    from . import report as R

    runs = [Path(r) if Path(r).is_absolute() else ROOT / r for r in (args.runs or [])]
    if not runs:
        runs = [p for p in sorted(RUN_ROOT.glob(f"*-team-arms-{args.set}")) if (p / "run.json").exists()
                and json.loads((p / "run.json").read_text(encoding="utf-8")).get("fingerprint") == gate.fingerprint()]
    if not runs:
        print("refused: no run of this set under the current gate", file=sys.stderr)
        return 2
    rows, manifests = R.load_rows(runs)
    sets = {m.get("set") for m in manifests}
    if len(sets) != 1:
        print(f"refused: the runs hold more than one set ({sorted(map(str, sets))})", file=sys.stderr)
        return 2
    set_name = sets.pop()
    if any(m.get("fingerprint") != gate.fingerprint() for m in manifests):
        print("refused: a run was made under another gate: a comparison is read under the gate it ran on", file=sys.stderr)
        return 2
    if set_name == "sealed" and not all(m.get("finished") for m in manifests):
        print("refused: the sealed run has not finished every play (go on with run --resume); it is reported once, whole", file=sys.stderr)
        return 2
    doc = R.build(rows, manifests, set_name)
    if set_name == "quick" or args.dry_run:
        out = runs[0] / "report.json"
        out.write_text(json.dumps(doc, indent=1, default=str) + "\n", encoding="utf-8")
        (runs[0] / "report.md").write_text(R.markdown({**doc, "id": "smoke"}), encoding="utf-8")
        print(json.dumps({"verdict": doc["verdict"], "written": str(out)}, indent=1))
        return 0
    try:
        path = R.write(doc)
    except ValueError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"verdict": doc["verdict"], "result": str(path.relative_to(ROOT))}, indent=1))
    return 0


def _cmd_receipt(args: argparse.Namespace) -> int:
    from ..harness_ablation import receipt

    run = Path(args.run) if Path(args.run).is_absolute() else ROOT / args.run
    out = receipt(run, Path(args.snapshot) if args.snapshot else None)
    print(json.dumps({k: out[k] for k in ("run", "files", "archived", "capture")}, indent=1, default=str))
    return 0 if out.get("archived") in (True, None) else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.team_arms", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        p = sub.add_parser(name)
        p.add_argument("--set", choices=list(C.SET_FILES), default="quick")
        p.add_argument("--arms", default="R,H,A")
        p.add_argument("--models", default=",".join(DEFAULT_MODELS), help=f"OpenRouter ids, comma-separated (the strong one: {STRONG_MODEL})")
        p.add_argument("--roles", default=None, help="split: only cases naming one of these roles")
        p.add_argument("--families", default=None, help="split: only cases of these families")
        if name == "run":
            p.add_argument("--yes", action="store_true")
            p.add_argument("--cap-eur", type=float, default=None)
            p.add_argument("--judge", default=None, help=f"a free, non-Claude judge (e.g. {JUDGE_MODEL}); advisory only")
            p.add_argument("--resume", default=None, help="a run cut by the cap: go on with it (complete plays are kept)")
    rep = sub.add_parser("report")
    rep.add_argument("--set", choices=list(C.SET_FILES), default="dev")
    rep.add_argument("--runs", nargs="*", default=None)
    rep.add_argument("--dry-run", action="store_true", help="write the report in the first run's folder, not the campaign")
    pin = sub.add_parser("pin")
    pin.add_argument("--replace", action="store_true")
    rec = sub.add_parser("receipt")
    rec.add_argument("run")
    rec.add_argument("--snapshot", default=None)
    sub.add_parser("schema")
    args = parser.parse_args(argv)
    if args.command == "plan":
        arms = [a.strip().upper() for a in args.arms.split(",") if a.strip()]
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        print(json.dumps(plan(args.set, arms, models, args.roles.split(",") if args.roles else None,
                              args.families.split(",") if args.families else None), indent=1, default=str))
        return 0
    if args.command == "pin":
        done, said = gate.pin(replace=args.replace)
        print(("pinned " + gate.GATE_ID + ": " if done else "refused: ") + said, file=sys.stdout if done else sys.stderr)
        return 0 if done else 2
    if args.command == "schema":
        print(C.write_schema())
        return 0
    if args.command == "report":
        return _cmd_report(args)
    if args.command == "receipt":
        return _cmd_receipt(args)
    return _cmd_run(args)


__all__ = ["ARM_NAMES", "CAMPAIGN", "DEFAULT_MODELS", "JUDGE_MODEL", "MAX_CAP_EUR", "STRONG_MODEL", "estimate_eur", "main", "plan", "run_cases",
           "begin_sealed", "code_files", "code_state", "sealed_refusal", "select", "thinking_check"]
