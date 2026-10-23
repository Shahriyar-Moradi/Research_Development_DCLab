"""The scoreboard's baseline: today's DCLab on a family's variants, nothing learned (H0; Phase 9's 31.4).

``baseline(family, split, n)`` draws the first ``n`` seeds of a partition, writes each pack apart from its key, and runs
two arms on it: **rules** (today's DCLab: the proposal accepted, nothing stated) and **reference** (the key's
statements: proof the variant is solvable). Both are graded. A variant the reference fails, or whose attempt broke, is
dropped from the score and listed. Every attempt is a run of the ``learning`` agent in the harness, each arm a step
with its verdict and codes (never a column's name); the reference arm, which states the key's answers, writes no
decision to the ledger and no event to the session's record.

The result is the zero of every learning curve: ``evidence/campaigns/learning_v1/results/LRN-NNN_<family>_<split>.json``,
its pass rate with a Wilson interval, its decision score with a bootstrap interval over challenges, the codes and the
knobs it failed on. A sealed partition is read once: its marker is written before the first solve, and a second read
is refused, whether the first was stored, crashed or was never meant to be stored.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dclab_core import challenge_pack as cp
from dclab_core import variants as V

from ..agents import harness
from .grading import grade_run, solve
from .packs import materialize, public_dir, read_key

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "learning_v1"
RESULTS = CAMPAIGN / "results"
AGENT = "learning"
GATE = {"version": "learning_gate_v1", "fixed": "2026-10-23, before the first baseline",
        "grader": cp.GRADER, "generator": V.VERSION, "partition_salt": cp.SALT, "partition_bounds": "learning < 0.5 <= check < 0.8 <= sealed",
        "codes": sorted(cp.CODES), "honest_estimate": f"validation beats the holdout by at most {cp.GUARD_Z} times both estimates' noise together "
        "(the holdout's from its bootstrap interval, validation's from the folds' spread over the root of their count; without an "
        f"interval: within {cp.GAP_LIMIT:.0%} of the validation score; without a validation score: failed)",
        "beats_baseline": "the holdout better than the best simple baseline; without a holdout or a baseline: failed",
        "known_before": "no honest column known before the moment kept out (one check), and each decoy kept in (a check each)",
        "solvable": "a variant counts only when its reference statements pass every check; one that does not, or whose attempt broke, is dropped and listed",
        "sealed": "a sealed partition is read once per promotion of the whole loop: its marker is written before the first solve",
        "pass": "a baseline is a measurement, not a promotion: it passes when every scored variant was solvable"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def fingerprints() -> dict[str, str]:
    """The code a baseline ran with: the generator and the grader, so a later run with other code is never compared blind."""
    import dclab_core.challenge_pack as grader_module
    import dclab_core.variants as generator_module

    return {"generator": _sha(Path(generator_module.__file__)), "grader": _sha(Path(grader_module.__file__))}


def _arm(grade: cp.Grade, observed: cp.Observed) -> dict[str, Any]:
    return {"verdict": grade.verdict, "codes": grade.codes, "decision_score": grade.decision_score, "traps_caught": grade.traps_caught,
            "traps": grade.traps, "checks": [c.model_dump() for c in grade.checks], "metric": observed.metric,
            "validation": observed.validation, "holdout": observed.holdout, "interval": observed.holdout_interval,
            "baseline": observed.baseline, "split": observed.split}


def _codes_only(grade: cp.Grade) -> dict[str, Any]:
    """What a step of the session keeps: the verdict and the codes, never a column's name (a learner reads this record)."""
    return {"verdict": grade.verdict, "codes": grade.codes, "decision_score": grade.decision_score,
            "traps_caught": grade.traps_caught, "traps": grade.traps, "grader": grade.grader}


@contextmanager
def _answers_unrecorded() -> Iterator[None]:
    """The reference arm states the key's answers: they never become a person's decisions in a ledger, nor events of the
    session's record a learner reads later."""
    from .. import experience, ledger

    before = ledger._INSTALLED.get("ledger")
    ledger.install(None)
    try:
        with experience.quiet():
            yield
    finally:
        ledger.install(before)


def attempt(root: Path, family: str, seed: int) -> dict[str, Any]:
    """One variant, both arms, as a run of the learning agent: the pack written, each arm solved from the public folder
    alone, then graded against the key read through the grader's own door."""
    started = time.perf_counter()
    variant = materialize(root, family, seed)
    folder = public_dir(root, family, variant.pack.id)
    arms: dict[str, Any] = {}
    with harness.session(AGENT, params={"family": family, "pack": variant.pack.id, "partition": variant.key.partition}) as current:
        for arm in ("rules", "reference"):
            with current.timed(f"solve:{arm}", {"pack": variant.pack.id}) as out:
                if arm == "rules":
                    run = solve(folder)
                else:
                    with _answers_unrecorded():
                        run = solve(folder, V.reference(read_key(root, family, variant.pack.id)))
                grade, observed = grade_run(read_key(root, family, variant.pack.id), run)  # the grader's door, after the solve
                arms[arm] = _arm(grade, observed)
                out["result"] = _codes_only(grade)
    return {"pack": variant.pack.id, "seed": int(seed), "partition": variant.key.partition, "knobs": variant.key.knobs,
            "arms": arms, "solvable": arms["reference"]["verdict"] == "pass", "seconds": round(time.perf_counter() - started, 2)}


def wilson(hits: int, n: int) -> list[float] | None:
    from ..agent_eval.run import wilson as _wilson

    return _wilson(hits, n)


def bootstrap_mean(values: list[float], draws: int = 2000, seed: int = 0) -> list[float] | None:
    """A 95% interval of the mean, resampling challenges (each variant one challenge)."""
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(draws))
    return [round(means[int(0.025 * draws)], 4), round(means[int(0.975 * draws) - 1], 4)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [r for r in rows if r["solvable"]]
    rules = [r["arms"]["rules"] for r in scored]
    passed = sum(a["verdict"] == "pass" for a in rules)
    scores = [a["decision_score"] for a in rules]
    by_knob: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in scored:
        for knob in ("task", "key_name", "leak_name", "decoy", "timed"):
            cell = by_knob[knob][str(r["knobs"].get(knob))]
            cell[0] += r["arms"]["rules"]["verdict"] == "pass"
            cell[1] += 1
    return {"variants": len(rows), "solvable": len(scored), "unsolvable": [r["pack"] for r in rows if not r["solvable"]],
            "broke": [r["pack"] for r in rows if r.get("error")],
            "passed": passed, "pass_rate": round(passed / len(scored), 4) if scored else None, "pass_rate_interval": wilson(passed, len(scored)),
            "decision_score": round(sum(scores) / len(scores), 4) if scores else None, "decision_score_interval": bootstrap_mean(scores),
            "traps_caught": sum(a["traps_caught"] for a in rules), "traps": sum(a["traps"] for a in rules),
            "codes": dict(Counter(c for a in rules for c in a["codes"]).most_common()),
            "by_knob": {k: {v: {"passed": p, "of": n} for v, (p, n) in sorted(cells.items())} for k, cells in by_knob.items()}}


RESULT = re.compile(r"^LRN-(\d+)_(?P<family>[a-z0-9]+)_(?P<split>learning|check|sealed)\.json$")


def _results(family: str, split: str) -> list[Path]:
    if not RESULTS.is_dir():
        return []
    return sorted(p for p in RESULTS.glob("LRN-*.json") if (m := RESULT.match(p.name)) and m["family"] == family and m["split"] == split)


def _marker(family: str) -> Path:
    return RESULTS / f".sealed_{family}"


class SealedSpent(RuntimeError):
    """A sealed partition already read: a second look would turn the exam into practice."""


def _store(result: dict[str, Any], family: str, split: str) -> Path:
    """The result under the next free id; ``open(..., "x")`` so two runs finishing together never share one."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    while True:
        numbers = [int(m.group(1)) for p in RESULTS.glob("LRN-*.json") if (m := RESULT.match(p.name))]
        rid = f"LRN-{(max(numbers) + 1) if numbers else 1:03d}"
        path = RESULTS / f"{rid}_{family}_{split}.json"
        try:
            with open(path, "x", encoding="utf-8") as fh:
                fh.write(json.dumps({**result, "id": rid, "experiment_id": rid}, indent=1, ensure_ascii=False) + "\n")
            result.update(id=rid, experiment_id=rid)
            return path
        except FileExistsError:
            continue


def baseline(family: str = "groups", split: str = "check", n: int = 20, write: bool = False, log=print) -> dict[str, Any]:
    if split not in cp.PARTITIONS:
        raise ValueError(f"split must be one of {', '.join(cp.PARTITIONS)}")
    if split == "sealed":
        if not write:
            raise SealedSpent("a sealed partition is read only to be stored: run it with --write")
        if _marker(family).exists() or _results(family, "sealed"):
            raise SealedSpent(f"the sealed {family} variants were read already: they are read once per promotion of the whole loop")
        RESULTS.mkdir(parents=True, exist_ok=True)
        _marker(family).write_text(datetime.now(timezone.utc).isoformat(timespec="seconds") + "\n", encoding="utf-8")  # before the first solve
    seeds = V.seeds(family, split, n)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:  # the packs and keys are drawn again from their seeds: nothing kept on disk
        for seed in seeds:
            try:
                row = attempt(Path(tmp), family, seed)
            except Exception as exc:  # noqa: BLE001 - one attempt that breaks is a row with its reason, never the end of the baseline
                row = {"pack": f"seed-{seed}", "seed": int(seed), "partition": split, "knobs": {}, "arms": {}, "solvable": False,
                       "error": f"{type(exc).__name__}: {exc}"}
                log(f"seed {seed} {split}: broke ({row['error'][:160]})")
            else:
                log(f"{row['pack']} {split}: rules {row['arms']['rules']['verdict']} {row['arms']['rules']['codes']}; "
                    f"reference {row['arms']['reference']['verdict']} ({row['seconds']}s)")
            rows.append(row)
    summary = summarize(rows)
    unsolvable = summary["unsolvable"]
    verdict = (f"fail: {len(unsolvable)} variants not solvable or broken ({', '.join(unsolvable)})" if unsolvable else
               f"ok: baseline recorded, today's DCLab passes {summary['passed']} of {summary['solvable']} {split} variants")
    result = {"id": None, "experiment_id": None, "campaign_id": "learning_v1", "kind": "learning_baseline", "status": "completed",
              "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "family": family, "split": split, "seeds": seeds,
              "arms": {"rules": "today's DCLab: the proposal accepted, nothing stated", "reference": "the key's statements (solvability)"},
              "gate": GATE, "fingerprints": fingerprints(), "spent_eur": 0.0, "rows": rows, "summary": summary, "verdict": verdict,
              "rule": "every variant drawn from its seed, written apart from its key, solved from its public folder, graded by code"}
    if write:
        path = _store(result, family, split)
        result["path"] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    return result


__all__ = ["GATE", "AGENT", "RESULTS", "attempt", "summarize", "baseline", "bootstrap_mean", "fingerprints", "SealedSpent"]
