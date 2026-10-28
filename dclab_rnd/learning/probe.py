"""The probe gate (H3): does one lesson help the local solver? Measured in pairs, never asserted.

``run_probe(home, client, model, bullet)`` gives the intern, served by a local model, the same learning variants twice:
once without the lesson and once with it (only that lesson, in its first user message), grades both by code, and pairs
them (the same variant, the same model, the same budget). Its gate was fixed before the first probe (``GATE``):

* every pair moves the lesson's counters by code: shown, helpful where the grader's decision score rose, harmful where it
  fell (the playbook retires a lesson at harmful minus helpful of three);
* a draft becomes **probed** only over at least ``MIN_PAIRS`` pairs, when the mean gain is above zero and no pair loses a
  pass (a homework passed without the lesson and failed with it); with fewer pairs the counters move and the status does not;
* a homework that broke before the model answered makes the probe invalid: nothing is counted.

Only learning variants are read (a check or sealed variant never reaches a learner). Every attempt is kept whole as the
pilot's are, with ``probe_row.json`` beside it, so a paused probe goes on where it stopped and a pair is never counted
twice. The pilot's attempts of the same model and budget may serve as the arm without the lesson (``reuse_pilot``): they
are exactly that, and on this laptop each homework costs half an hour or more.
"""

from __future__ import annotations

import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dclab_core import variants as V
from dclab_core.challenge_pack import partition_of
from dclab_core.playbook import Playbook

from .grading import grade_run
from .inject import cited_in, for_solver
from .packs import materialize, public_dir, read_key
from .pilot import _relative, broke, done_row
from .playbooks import FilePlaybooks
from .solver import BUDGET, intern_attempt, keep_grade, summary_of

GATE = {"version": "probe_gate_v1", "fixed": "2026-10-24, before the first probe",
        "pair": "the same learning variant, the same local model and budget, without and with the one lesson",
        "counts": "every pair: shown +1; helpful +1 where the grader's decision score rose with the lesson; harmful +1 where it fell",
        "promote": "draft -> probed over at least 9 pairs with at least 3 repeats of each variant (the plan's H3), when the mean gain is "
                   "above zero and no pair loses a pass; the status read again after the counts moved (a lesson they retire stays retired)",
        "too_few": "fewer than 9 pairs or fewer than 3 repeats a variant: the counters move, the status does not",
        "reuse": "the pilot's attempt serves as the arm without the lesson only with the same model, budget, tools and rules (their hashes)",
        "invalid": "a homework that broke before the model answered: nothing counted",
        "never": "a check or sealed variant is never read by the probe"}
MIN_PAIRS = 9
MIN_REPEATS = 3
PAUSE = "PAUSE_PROBE"
ARMS = ("without", "with")


def pause_file(home: Path) -> Path:
    return Path(home) / "learning_attempts" / PAUSE


def _rows(home: Path) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for saved in sorted((Path(home) / "learning_attempts").glob("*/*-probe-*/probe_row.json")):
        folder = saved.parent
        if (folder / "invalid.json").exists() or not (folder / "grade.json").exists():
            continue
        out.append((folder, json.loads(saved.read_text(encoding="utf-8"))))
    return out


def done(home: Path, family: str, seed: int, repeat: int, arm: str, bullet: str, model: str, budget: dict[str, Any],
         lineage: str = "main") -> dict[str, Any] | None:
    """A probe homework already done and graded (its row beside it), matched on everything that makes it this one: the
    variant, the repeat, the arm, the model, the budget, and for the arm with the lesson, the lesson and its lineage."""
    for _, row in reversed(_rows(home)):
        same = (row.get("family"), row.get("seed"), row.get("repeat"), row.get("arm"), row.get("model"), row.get("budget")) == \
            (family, seed, repeat, arm, model, budget)
        if same and (arm == "without" or (row.get("bullet"), row.get("lineage")) == (bullet, lineage)):  # an arm without a lesson serves any lesson
            return row
    return None


def _pilot_without(home: Path, family: str, seed: int, model: str, budget: dict[str, Any]) -> dict[str, Any] | None:
    """The pilot's attempt of this variant by this model, when it ran under the same budget: the arm without a lesson."""
    row = done_row(home, family, seed, model)
    if row is None or not row.get("kept"):
        return None
    sessions = sorted((Path(home) / row["kept"] / "intern").glob("*.json"))
    if not sessions:
        return None
    session = json.loads(sessions[0].read_text(encoding="utf-8"))
    ran = session.get("budget") or {}
    if any(ran.get(k) != budget.get(k) for k in ("max_steps", "max_minutes")) or session.get("playbook"):
        return None
    offered = Path(home) / row["kept"] / "tools.json"  # the same tools and rules as today's, or the pair would mix two versions
    if not offered.exists() or {k: json.loads(offered.read_text(encoding="utf-8")).get(k) for k in ("tools_sha256", "policy_sha256")} != _today():
        return None
    return {**row, "arm": "without", "repeat": 0, "budget": budget, "from_pilot": True}


def _today() -> dict[str, Any]:
    """The hashes of the tools and the rules the intern is offered today, as ``tools.json`` records them."""
    import tempfile as _t

    from dclab_rnd.intern.tools import Toolbox
    from dclab_rnd.studio import ProjectStore

    from .solver import _keep_tools

    with _t.TemporaryDirectory() as tmp:
        _keep_tools(Path(tmp), Toolbox(ProjectStore(Path(tmp) / "projects")))
        doc = json.loads((Path(tmp) / "tools.json").read_text(encoding="utf-8"))
    return {k: doc.get(k) for k in ("tools_sha256", "policy_sha256")}


def _attempt(home: Path, tmp: Path, client: Any, model: str, family: str, seed: int, repeat: int, arm: str, bullet: str,
             shown: dict[str, Any] | None, budget: dict[str, Any], log, lineage: str = "main") -> dict[str, Any]:
    variant = materialize(tmp, family, seed)
    key = read_key(tmp, family, variant.pack.id)
    started = time.monotonic()
    base = {"pack": variant.pack.id, "seed": seed, "family": family, "repeat": repeat, "arm": arm, "bullet": bullet if arm == "with" else None,
            "lineage": lineage if arm == "with" else None, "model": model, "budget": budget}
    try:
        out = intern_attempt(public_dir(tmp, family, variant.pack.id), client, key, playbook=shown if arm == "with" else None, home=home,
                             budget=budget, label=f"probe-{bullet[2:]}-{arm}-r{repeat}")
    except Exception as exc:  # noqa: BLE001 - an attempt that breaks is a row with its reason
        log(f"{variant.pack.id} {arm} r{repeat}: broke ({type(exc).__name__})")
        return {**base, "broke": True, "finished": False, "verdict": "broke", "codes": [], "decision_score": None,
                "error": f"{type(exc).__name__}: {exc}"[:300], "seconds": round(time.monotonic() - started, 1)}
    grade, observed = grade_run(key, out["run"])
    keep_grade(out["kept"], grade, observed)
    row = {**base, "broke": broke(out["session"]), "finished": out["finished"], "verdict": grade.verdict, "codes": grade.codes,
           "decision_score": grade.decision_score, "seconds": out["seconds"], "kept": _relative(out["kept"], home),
           "session": summary_of(out["session"]), "cited": cited_in(out["session"]) if arm == "with" else []}
    if out["kept"]:
        (Path(out["kept"]) / "probe_row.json").write_text(json.dumps(row, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if row["broke"]:  # the server or DCLab failed before the model answered: done again next time, never the model's fail in training
            mark_invalid(Path(out["kept"]), "the session failed before the model answered (the server or DCLab broke)")
    log(f"{variant.pack.id} {arm} r{repeat}: finished {out['finished']}, {grade.verdict} {grade.codes} score {grade.decision_score} "
        f"in {out['seconds']}s")
    return row


def run_probe(home: Path, client: Any, model: str, bullet: str, family: str = "groups", n: int = 3, repeats: int = 1,
              seeds: list[int] | None = None, lineage: str = "main", budget: dict[str, Any] | None = None, reuse_pilot: bool = True,
              log=print, limit: int | None = None, cap_eur: float | None = None, count: bool = True) -> dict[str, Any]:
    """``limit``: at most this many new homeworks in this process, then it pauses (a fresh process a homework gives its
    memory back; the pilot's measurement of 2026-10-24). ``count=False`` measures only: a hosted model's pairs never move
    a lesson's counters or status, which describe the local solver (found 2026-10-24: a Claude probe had moved them)."""
    if client is None:  # with no model the intern runs the standard plan: that would measure the rules, not the lesson
        raise ValueError("a probe needs a model client")
    seeds = list(seeds) if seeds is not None else V.seeds(family, "learning", n)
    for seed in seeds:
        if partition_of(family, seed) != "learning":
            raise ValueError(f"seed {seed} is not a learning variant: a probe never reads a check or sealed one")
    books = FilePlaybooks(home)
    book, _ = books.book(family, lineage)
    lesson = next((b for b in book.bullets if b.id == bullet), None)
    if lesson is None:
        raise ValueError(f"no lesson {bullet} in the {family}/{lineage} playbook")
    if lesson.status == "retired":
        raise ValueError(f"{bullet} is retired")
    shown = for_solver(Playbook(family=family, lineage=lineage, version=book.version, bullets=[lesson]), statuses=(lesson.status,))
    budget = {**BUDGET, **(budget or {})}
    pairs, rows, paused, new = [], [], False, 0
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            for repeat in range(repeats):
                pair = {}
                for arm in ARMS:
                    row = done(home, family, seed, repeat, arm, bullet, model, budget, lineage)
                    if row is None and arm == "without" and repeat == 0 and reuse_pilot:
                        row = _pilot_without(home, family, seed, model, budget)
                    if row is not None:
                        log(f"{row['pack']} {arm} r{repeat}: done before ({row['verdict']} {row['codes']}), not repeated")
                    else:
                        spent = sum(float(((r.get("session") or {}).get("eur")) or 0.0) for r in rows if not r.get("from_pilot"))
                        if cap_eur is not None and spent >= cap_eur:  # a hosted model: the cap holds for the whole probe, its earlier homeworks too
                            log(f"the cap of EUR {cap_eur:.2f} is spent (EUR {spent:.4f}): stopped")
                            paused = True
                            break
                        if pause_file(home).exists() or (limit is not None and new >= limit):
                            if pause_file(home).exists():
                                pause_file(home).unlink()
                            paused = True
                            break
                        new += 1
                        row = _attempt(home, Path(tmp), client, model, family, seed, repeat, arm, bullet, shown, budget, log, lineage)
                    pair[arm] = row
                    rows.append(row)
                if paused:
                    break
                pairs.append(pair)
            if paused:
                break
    if paused:
        log("paused: run the same command again to go on")
        return {"campaign_id": "learning_v1", "kind": "learning_probe", "status": "paused", "family": family, "bullet": bullet, "model": model,
                "rows": rows, "verdict": f"paused after {len(pairs)} pairs: run the same command again to go on"}
    return judge(home, books, family, lineage, lesson, model, pairs, rows, budget, log, count=count)


def judge(home: Path, books: FilePlaybooks, family: str, lineage: str, lesson: Any, model: str, pairs: list[dict[str, Any]],
          rows: list[dict[str, Any]], budget: dict[str, Any], log=print, count: bool = True) -> dict[str, Any]:
    from .scoreboard import fingerprints

    broken = [r for r in rows if r.get("broke")]
    gains = []
    for p in pairs:
        a, b = p["without"].get("decision_score"), p["with"].get("decision_score")
        gains.append(None if a is None or b is None else round(float(b) - float(a), 4))
    scored = [g for g in gains if g is not None]
    mean = round(sum(scored) / len(scored), 4) if scored else None
    lost = sum(1 for p in pairs if p["without"].get("verdict") == "pass" and p["with"].get("verdict") != "pass")
    counts = {"shown": len(scored), "helpful": sum(1 for g in scored if g > 0), "harmful": sum(1 for g in scored if g < 0)}
    repeats_of: dict[int, int] = {}
    for p, g in zip(pairs, gains):
        if g is not None:
            repeats_of[p["with"]["seed"]] = repeats_of.get(p["with"]["seed"], 0) + 1
    enough = len(scored) >= MIN_PAIRS and bool(repeats_of) and min(repeats_of.values()) >= MIN_REPEATS
    if broken:
        verdict, counted = f"invalid: {len(broken)} homeworks broke before the model answered; nothing counted", None
    elif not count:
        verdict, counted = (f"measured only ({model}, not the local solver): mean gain {mean} over {len(scored)} pairs, passes lost {lost}; "
                            "the lesson's counters and status are the local solver's and were not moved"), None
    else:
        counted = _count_once(home, books, family, lineage, lesson.id, pairs, gains)
        promote = enough and mean is not None and mean > 0 and lost == 0
        fresh, _ = books.book(family, lineage)  # the counts may have retired it: read its status after them
        status = next((b.status for b in fresh.bullets if b.id == lesson.id), lesson.status)
        promote = promote and status == "draft"
        if promote:
            status = books.status(family, lesson.id, "probed", by="probe", reason=f"probe_gate_v1: mean gain {mean} over {len(scored)} pairs, no pass lost",
                                  lineage=lineage).bullets
            status = next(b.status for b in status if b.id == lesson.id)
        verdict = (f"probed: mean gain {mean} over {len(scored)} pairs, no pass lost" if promote else
                   f"too few pairs ({len(scored)} pairs, {min(repeats_of.values()) if repeats_of else 0} repeats a variant; the gate asks "
                   f"{MIN_PAIRS} and {MIN_REPEATS}): counters moved, status kept ({status}); mean gain {mean}, passes lost {lost}"
                   if not enough else f"not probed ({status}): mean gain {mean}, passes lost {lost} over {len(scored)} pairs")
    book, _ = books.book(family, lineage)
    after = next((b for b in book.bullets if b.id == lesson.id), None)
    return {"campaign_id": "learning_v1", "kind": "learning_probe", "status": "completed", "family": family, "split": "learning",
            "model": model, "budget": budget, "bullet": {"id": lesson.id, "text": lesson.text, "scope": lesson.scope, "by": lesson.by,
                                                          "status_before": lesson.status, "status_after": after.status if after else None,
                                                          "counters_after": {k: getattr(after, k) for k in ("shown", "helpful", "harmful")} if after else None},
            "gate": GATE, "fingerprints": fingerprints(), "rows": rows,
            "pairs": [{"seed": p["with"]["seed"], "repeat": p["with"]["repeat"], "without": p["without"].get("decision_score"),
                       "with": p["with"].get("decision_score"), "gain": g, "without_from_pilot": bool(p["without"].get("from_pilot")),
                       "cited": p["with"].get("cited") or []} for p, g in zip(pairs, gains)],
            "summary": {"pairs": len(pairs), "scored": len(scored), "mean_gain": mean, "passes_lost": lost, "counts": counts,
                        "counted_now": counted, "enough": enough}, "verdict": verdict, "spent_eur": round(sum(float(((r.get("session") or {}).get("eur")) or 0.0) for r in rows if not r.get("from_pilot")), 6),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "paired homeworks on learning variants, graded by code; the lesson's counters by code, its status by the gate"}


def _count_once(home: Path, books: FilePlaybooks, family: str, lineage: str, bullet: str, pairs: list[dict[str, Any]],
                gains: list[float | None]) -> dict[str, int]:
    """The counters of the pairs not counted before (``probe_counted.json`` beside each pair's attempt with the lesson): a
    probe run again after it finished counts nothing twice."""
    counts = {"shown": 0, "helpful": 0, "harmful": 0}
    marks = []
    for p, g in zip(pairs, gains):
        folder = Path(home) / p["with"]["kept"] if p["with"].get("kept") else None
        if g is None or folder is None or (folder / "probe_counted.json").exists():
            continue
        counts["shown"] += 1
        counts["helpful"] += int(g > 0)
        counts["harmful"] += int(g < 0)
        marks.append((folder, g))
    if counts["shown"]:
        books.count(family, bullet, {k: v for k, v in counts.items() if v}, by="probe",
                    reason=f"probe_gate_v1: {counts['shown']} pairs, {counts['helpful']} helped, {counts['harmful']} hurt", lineage=lineage)
        written = books.deltas(family, lineage)
        index, at = len(written) - 1, written[-1].at  # 31.17: the tally names the change it wrote, never a second clock
        for folder, g in marks:
            (folder / "probe_counted.json").write_text(json.dumps({"bullet": bullet, "gain": g, "at": at, "lineage": lineage,
                                                                   "delta": index}) + "\n", encoding="utf-8")
    return counts


def mark_invalid(folder: Path, why: str) -> None:
    """``invalid.json`` beside an attempt: it is done again by the next run and, in training, never counts as skill."""
    (Path(folder) / "invalid.json").write_text(json.dumps({"invalid": True, "why": why, "found": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                                          indent=1) + "\n", encoding="utf-8")


__all__ = ["GATE", "MIN_PAIRS", "MIN_REPEATS", "PAUSE", "ARMS", "pause_file", "done", "run_probe", "judge", "mark_invalid"]
