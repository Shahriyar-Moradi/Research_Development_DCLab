"""``team_arms_gate_v1``: how the winner of each role is chosen, fixed and pinned by hash before any run (B4, 2026-11-24).

**The winner rule, per role and per model.** For a role, the cases are those whose ``roles`` name it. An arm is eligible
when it played every one of those cases to the end (none cut by the cap or a crash), with zero hard violations, and
within the limits fixed here: its mean seconds per case and its mean euros per case. Among the eligible arms the one with
the best main score (the share of its cases scored 1) wins; an equal share goes to the simpler arm (R simpler than H, H
simpler than A). No eligible arm: no winner, and the role keeps the rules (the switch's default). The rules arm asks no
model, so its plays count for every model.

The judge's 0-4 scores are reported beside it and never enter the rule. The paired bootstrap interval of the winner's
difference against the rules is reported too: a win inside it is said to be within the noise, never hidden.

``pin`` writes :func:`fingerprint` to ``gate_v1.sha256`` (it covers these terms, the scorer's, the runner's and this
file's code): every run refuses an unpinned or changed gate. A pin may be replaced (``pin --replace``) only while no sealed
run has started; after that a change is a new gate (v2). B4.1 (2026-11-24): the first dev round's scorer and runner fixes
were re-pinned so, before any sealed run; the round-0 dev run stays under the pin it ran on.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

GATE_ID = "team_arms_gate_v1"
HERE = Path(__file__).resolve().parent
PIN = HERE / "gate_v1.sha256"
ROOT = HERE.parents[2]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "team_arms_v1"
SEALED_MARKER = CAMPAIGN / "sealed_started.json"  # committed: the run-once guard survives a moved run folder
SIMPLER = ("R", "H", "A")
LIMITS = {
    "max_mean_seconds_per_case": 300.0,  # think-first added about 40 s a conversation (2026-11-16): five minutes a case is room
    "max_mean_eur_per_case": 0.08,  # an agent turn may spend 0.10; a case is a few turns of a cheap model
    "max_eur_per_run": 0.5,  # one command; the owner's rule for live runs
}
BOOTSTRAP = {"reps": 2000, "seed": 20261124, "level": 0.95, "by": "case"}
# the sealed comparison runs every arm on these models, whole (a cheap one and the strongest open one on OpenRouter); a change
# before the sealed run is a re-pin, after it a new gate
SEALED_MODELS = ["qwen/qwen3.8-flash", "nvidia/nemotron-3-ultra-550b-a55b"]


def _digest(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


TERMS: dict[str, Any] = {
    "id": GATE_ID,
    "winner_rule": ("per role and model: only when the rules arm played every case of the role to the end with zero hard violations; "
                    "then among the arms that played every case of the role to the end, with zero hard violations and within the "
                    "limits (mean seconds and mean euros per case), the best main score (share of cases scored 1); an equal share goes "
                    "to the simpler arm (R, then H, then A); otherwise no winner, and the role keeps the rules"),
    "tie": "an equal main score (no margin)",
    "simpler_order": list(SIMPLER),
    "limits": LIMITS,
    "bootstrap": BOOTSTRAP,
    "main_score": ("1 when the case ends as expected (answer, justified blocker, open question or refusal), every expected fact holds, "
                   "the evidence is valid, the next owner is right and no hard zero; else 0"),
    "hard_zeros": ["invented_number", "unrecorded_claim", "unauthorised_read", "agent_approved", "customer_contacted", "holdout_twice",
                   "answer_not_given"],
    "question_numbers": ("a number no record holds in a question handed between teams (a hand-over's question, a team's clarification "
                         "to another team), or in a choice a question to the person offers ('(a) ... (b) ...', the words around 'or'), is "
                         "counted (question_numbers), never a hard zero; one in a stored report's model words, a helper's note, or the "
                         "premise of a question to the person is invented_number, a hard zero"),
    "not_numbers": ("a run of list markers ((1), 1., 1) counting from 1), a minus glued to a letter (T-7), a counting word of the "
                    "conversation ('two things', 'the two') and a binary label's code (y = 0) are no numbers; a span a person names "
                    "('a week', 'a fortnight', 'a day') makes its days known"),
    "known_asks": "code's questions to the Sales and Marketing teams add their numbers only in a play that sent them",
    "replayable": ("a play cut by the cap, or in which a model request was refused (no credit, a refused key) or failed (a rate "
                   "limit after the retries, a provider error, the network) for the provider's reason, is played again and counts "
                   "for no arm; a refusal stops the run at once; a run with such a play has not finished; a play the provider "
                   "failed twice before is final the third time, incomplete (not played to the end)"),
    "judge": "advisory only (a free, non-Claude judge, 0 to 4); never part of the winner rule",
    "no_claude": ["claude", "anthropic", "fable"],
    "sealed": "run once; the marker evidence/campaigns/team_arms_v1/sealed_started.json is written before the first play",
    # every file whose code decides a score, a winner, a stop or what counts as a record (the review of B4, F10)
    "code_sha256": {name: _digest(path) for name, path in {
        "score": HERE / "score.py", "play": HERE / "play.py", "cases": HERE / "cases.py", "gate": HERE / "gate.py", "cli": HERE / "__init__.py",
        "report": HERE / "report.py", "judge": HERE / "judge.py", "text": HERE / "text.py"}.items()},  # text.py: the scorer's own
    # copy of the number, sentence and claim rules (the re-review's N1: never a shared library file another package edits)
    "sealed_arms": ["R", "H", "A"],
    "sealed_models": SEALED_MODELS,
}


def fingerprint() -> str:
    return hashlib.sha256(json.dumps(TERMS, sort_keys=True).encode("utf-8")).hexdigest()


def pinned(pin: Optional[Path] = None) -> Optional[str]:
    """Why the gate may not run, or None when its pin matches the terms."""
    pin = Path(pin) if pin is not None else PIN
    if not Path(pin).exists():
        return "the gate is not pinned (python -m dclab_rnd.agent_eval.team_arms pin)"
    if Path(pin).read_text(encoding="utf-8").strip() != fingerprint():
        return "the gate's terms or code changed after it was pinned: a changed gate is pinned again (before the sealed run) or is a new gate"
    return None


def pin(replace: bool = False, pin: Optional[Path] = None, marker: Optional[Path] = None) -> tuple[bool, str]:
    """Write the pin. Refused when one exists (unless ``replace`` and no sealed run has started)."""
    pin, marker = (Path(pin) if pin is not None else PIN), (Path(marker) if marker is not None else SEALED_MARKER)
    if Path(pin).exists() and not replace:
        return False, "the gate is pinned already (pin --replace changes it, only before the sealed run)"
    if replace and Path(marker).exists():
        return False, "the sealed run has started: a changed gate is a new gate (v2), never this one again"
    Path(pin).write_text(fingerprint() + "\n", encoding="utf-8")
    return True, fingerprint()


# ---------------------------------------------------------------------------------------------------------- the rule
def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 6) if values else None


def interval(values: Sequence[float], reps: int = BOOTSTRAP["reps"], seed: int = BOOTSTRAP["seed"], level: float = BOOTSTRAP["level"]) -> Optional[list[float]]:
    """A percentile bootstrap interval of the mean, resampling cases."""
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    lo, hi = means[int((1 - level) / 2 * reps)], means[min(reps - 1, int((1 + level) / 2 * reps))]
    return [round(lo, 4), round(hi, 4)]


def paired_interval(a: Mapping[str, float], b: Mapping[str, float]) -> Optional[list[float]]:
    """The interval of ``a - b`` on the cases both played (paired by case)."""
    common = sorted(set(a) & set(b))
    return interval([a[c] - b[c] for c in common]) if common else None


def arm_summary(rows: Sequence[Mapping[str, Any]], expected_cases: Sequence[str]) -> dict[str, Any]:
    """One arm on one role (and model): its scored plays (``score.json`` rows) against the cases it had to play."""
    done = {r["case"]: r for r in rows if r.get("complete")}
    cut = sorted({r["case"] for r in rows if not r.get("complete")})
    missing = sorted(set(expected_cases) - set(done))
    mains = [float(done[c]["main"]) for c in expected_cases if c in done]
    seconds = [float(done[c]["counts"].get("seconds") or 0.0) for c in expected_cases if c in done]
    euros = [float(done[c]["counts"].get("euros") or 0.0) for c in expected_cases if c in done]
    hard = sum(len(done[c].get("hard_zeros") or []) for c in done)
    out = {"cases": len(expected_cases), "played": len(done), "missing": missing, "cut": cut, "main": _mean(mains), "main_interval": interval(mains),
           "hard_violations": hard, "mean_seconds": _mean(seconds), "mean_eur": _mean(euros),
           "by_case": {c: float(done[c]["main"]) for c in done}}
    why = []
    if missing:
        why.append(f"{len(missing)} cases not played to the end")
    if hard:
        why.append(f"{hard} hard violations")
    if out["mean_seconds"] is not None and out["mean_seconds"] > LIMITS["max_mean_seconds_per_case"]:
        why.append(f"mean seconds {out['mean_seconds']} over {LIMITS['max_mean_seconds_per_case']}")
    if out["mean_eur"] is not None and out["mean_eur"] > LIMITS["max_mean_eur_per_case"]:
        why.append(f"mean euros {out['mean_eur']} over {LIMITS['max_mean_eur_per_case']}")
    if not expected_cases:
        why.append("no case of this role")
    out["eligible"], out["not_eligible_because"] = not why, why
    return out


def winner(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The winner among ``{arm: arm_summary}`` by the pinned rule. The rules arm must have played the role to the end: a
    model arm is chosen over the rules, never over nothing."""
    if "R" not in arms or not arms["R"].get("eligible"):  # the review of B4, F5: no model arm wins against a rules arm not played
        return {"winner": None, "why": "the rules arm was not played to the end (or broke a hard rule): nothing to compare against",
                "eligible": [a for a in SIMPLER if a in arms and arms[a].get("eligible")]}
    eligible = [a for a in SIMPLER if a in arms and arms[a].get("eligible")]
    if not eligible:
        return {"winner": None, "why": "no eligible arm: the role keeps the rules", "eligible": []}
    best = max(float(arms[a]["main"] or 0.0) for a in eligible)
    chosen = next(a for a in SIMPLER if a in eligible and float(arms[a]["main"] or 0.0) == best)  # the simpler on a tie
    tied = [a for a in eligible if float(arms[a]["main"] or 0.0) == best]
    out = {"winner": chosen, "eligible": eligible, "best_main": best, "tied": tied if len(tied) > 1 else []}
    if chosen != "R" and "R" in arms:
        diff = paired_interval(arms[chosen]["by_case"], arms["R"]["by_case"])
        out["difference_vs_rules"] = diff
        out["within_noise"] = bool(diff and diff[0] <= 0.0)
    return out


__all__ = ["SEALED_MODELS", "BOOTSTRAP", "CAMPAIGN", "GATE_ID", "LIMITS", "PIN", "SEALED_MARKER", "SIMPLER", "TERMS", "arm_summary", "fingerprint", "interval",
           "paired_interval", "pin", "pinned", "winner"]
