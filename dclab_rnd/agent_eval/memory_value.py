"""35.8: does memory help? The intern's second session on a project, memory on against memory off.

The first session is the earlier work a person did with DCLab, written through the product's own write paths, never by a
model: the intern once judged a column wrongly and the person corrected it (an agent's line, then the person's
correction), and the person told the workspace which customer columns are recorded before the snapshot. A preference and
an accepted piece of know-how stand beside them, as in a real workspace. The second session is a live intern run on the
same project with the judgment suite's task. The two arms differ only in DCLAB_MEMORY: with memory off the stored lines
are never read (the project's own notes still are, as everywhere).

Measured per session: whether it acted against what memory holds (lost a column the person said to keep, or used one the
person said to keep out), the judgment suite's leak and false-alarm scores, tokens and cost. Sessions are paired by case
and repeat; the arms alternate their order. The gate is in manifest.json, fixed before the first run; this module reads
it and never changes it. Questions asked again are not measured: the intern in this suite asks no questions (Home does,
and Home does not read memory yet).

Nothing starts without --yes, and each invocation stops before its euro cap (the owner's rule: €1 a run): the next
session starts only if the dearest session so far still fits. What is done is kept per session under
agent_runs/live_tests/memory_value_v1/<run>/ (the whole session, its steps and their thinking), so the next invocation
resumes. `summarize` joins them and judges the gate once every planned session ran.

    python -m dclab_rnd.agent_eval.memory_value                          # the plan; nothing sent
    python -m dclab_rnd.agent_eval.memory_value --yes --cap-eur 1        # runs (resumes) the next sessions under €1
    python -m dclab_rnd.agent_eval.memory_value summarize                # the result and the verdict, into the campaign
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "memory_value_v1"
DIR = ROOT / "evidence" / "campaigns" / CAMPAIGN
MANIFEST = DIR / "manifest.json"
RESULTS = DIR / "results"
RUNS = ROOT / "agent_runs" / "live_tests" / CAMPAIGN
AT = "2026-10-20T09:00:00+00:00"  # the first session's time: a week before the second
SCOPE = ("Evidence about this model and this brief on invented tables whose earlier session is scripted, not about memory in "
         "general or about any model's readiness for production.")


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def manifest_sha() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def cases() -> list[Any]:
    from .cases import CASES

    wanted = [s["case"] for s in manifest()["scenarios"]]
    by_id = {c.id: c for c in CASES}
    return [by_id[c] for c in wanted]


def schedule(repeats: int | None = None) -> list[dict[str, Any]]:
    """Every planned session, in order: repeat by repeat, case by case, the two arms alternating which goes first."""
    m = manifest()
    out = []
    for r in range(repeats or m["repeats"]):
        for n, s in enumerate(m["scenarios"]):
            arms = ("on", "off") if (r + n) % 2 == 0 else ("off", "on")
            out += [{"id": f"{s['case']}-r{r}-{arm}", "case": s["case"], "repeat": r, "arm": arm} for arm in arms]
    return out


# ------------------------------------------------------------------------------------------------ the first session
def seed_memory(home: Path, workspace: str, project_id: str, scenario: dict[str, Any], m: dict[str, Any] | None = None) -> list[str]:
    """What the earlier session left, through the product's write paths: the workspace's columns, the agent's wrong line
    and the person's correction, a preference and accepted know-how. Returns the ids active now."""
    from dclab_rnd.memory import latest, open_memory
    from dclab_rnd.memory import writing as w

    m = m or manifest()
    store = open_memory(home)
    w.record_fact(store, scope="workspace", scope_id=workspace, text=m["workspace_line"], source="session:first", who="person:owner", at=AT)
    w.record_fact(store, scope="person", scope_id="owner", text=m["preference_line"], source="session:first", who="person:owner", at=AT)
    know = w.draft_know_how(store, scope="workspace", scope_id=workspace, text=m["know_how_line"], source="session:first", who="agent:reflector", at=AT)
    w.review(store, know.id, "accept", by="owner", role="reviewer", at=AT)
    if scenario.get("agent_line"):
        wrong = w.record_fact(store, scope="project", scope_id=project_id, text=scenario["agent_line"], source="session:first", who="agent:intern", at=AT)
        w.correct(store, wrong, scenario["correction"], by="owner", reason="the person knows when it is recorded", at=AT)
    return [i.id for i in latest(store.log()) if i.status == "active"]


def held(case: Any, scenario: dict[str, Any], m: dict[str, Any] | None = None) -> dict[str, str]:
    """What memory says about this case's columns: keep or out (the correction, and the workspace's columns in this table)."""
    columns = set(case.table().columns)
    said = {c: "keep" for c in (m or manifest())["workspace_columns"] if c in columns and c not in case.leaks and c not in (case.time_column, case.group_column)}
    said.update({c: scenario["expect"] for c in scenario.get("columns") or []})
    return said


def against(said: dict[str, str], project: dict[str, Any]) -> list[str]:
    """The columns this session judged against what memory holds (lost as the suite's score reads it, or used). A session
    that saved no solution judged nothing: it counts against every line, never as obeying memory by doing nothing."""
    solution = project.get("solution") or {}
    if not solution.get("target"):
        return sorted(said)
    text = set(solution.get("text_columns") or []) if solution.get("task") != "binary" else set()
    kept_out = {f["column"] for f in solution.get("forbidden") or []} | set(solution.get("identifiers") or []) | text
    lost = kept_out | {c for c in (solution.get("time_column"), solution.get("group_column")) if c}
    return sorted(c for c, want in said.items() if (want == "keep" and c in lost) or (want == "out" and c not in kept_out))


# ------------------------------------------------------------------------------------------------ the second session
@contextmanager
def arm(name: str) -> Iterator[None]:
    before = os.environ.get("DCLAB_MEMORY")
    os.environ["DCLAB_MEMORY"] = "" if name == "on" else "off"
    try:
        yield
    finally:
        if before is None:
            os.environ.pop("DCLAB_MEMORY", None)
        else:
            os.environ["DCLAB_MEMORY"] = before


TRAINING_USE = ("eval_only: the memory brief of the 'on' sessions shows the judgment suite's answers; never train a model that is "
                "judged on agent_eval cases with these sessions")


def one_session(gateway: Any, item: dict[str, Any], budget_eur: float | None, max_steps: int, home: Path,
                scenario: dict[str, Any] | None = None, case: Any = None, m: dict[str, Any] | None = None) -> dict[str, Any]:
    """One live intern session on a fresh project whose memory holds the scripted first session (v2 passes its own
    scenario, case and manifest; v1 reads its manifest)."""
    from dclab_rnd.expansion.runner import fast_profile
    from dclab_rnd.intern import Intern, SessionStore
    from dclab_rnd.intern.tools import Toolbox
    from dclab_rnd.memory.tools import workspace_home
    from dclab_rnd.studio import ProjectStore, data as studio_data

    from .live import _cut_short
    from .run import TASK, _known_ids, score

    m = m or manifest()
    scenario = scenario or next(s for s in m["scenarios"] if s["case"] == item["case"])
    case = case or next(c for c in cases() if c.id == item["case"])
    store = ProjectStore(home / "projects")
    pid = store.create(case.title[:60], "general", "Predict the target column")["id"]
    case.table().to_parquet(store.data_dir(pid) / "table.parquet", index=False)
    studio_data.attach_data(store, pid, "table.parquet")
    active = seed_memory(workspace_home(store), getattr(store, "workspace_id", None) or "local", pid, scenario, m)
    budget: dict[str, Any] = {"max_steps": max_steps, "max_minutes": 10}
    if budget_eur is not None:
        budget["max_eur"] = budget_eur
    intern = Intern(SessionStore(home / "sessions"), Toolbox(store), gateway.client("intern", project_id=pid))
    started = time.perf_counter()
    with arm(item["arm"]), fast_profile():
        session = intern.run(intern.start(TASK.format(moment=case.moment), budget, project_id=pid)["id"])
    used = session.get("used") or {}
    out = {**item, "role": scenario.get("role", "treated"), "status": session["status"], "seconds": round(time.perf_counter() - started, 1),
           "steps": used.get("steps"), "tokens": [used.get("input_tokens"), used.get("output_tokens")], "eur": used.get("eur"), "budget_eur": budget_eur,
           "memory_active": len(active), "memory_shown": len(session.get("memory_shown") or []), "memory_cited": len(session.get("memory_cited") or []),
           "training_use": TRAINING_USE, "session": session}  # whole: the plan, every step with its thought, the messages, the report
    why = _cut_short(session)
    if why:
        return {**out, "ran": False, "why": why}
    project = store.get(pid)
    s = score(case, project, store.records(pid), store.transitions(pid, limit=1000), [], _known_ids())
    said = held(case, scenario, m)
    wrong = against(said, project)
    return {**out, "ran": True, "held": said, "against": wrong, "against_memory": bool(wrong), "solution_saved": bool((project.get("solution") or {}).get("target")),
            **{k: s[k] for k in ("leak_caught", "false_alarm", "valid_moves", "kept_out", "declared")}}


def run_dir(run: str | None) -> Path:
    if run:
        return RUNS / run
    started = sorted(p for p in RUNS.glob("MV-*") if p.is_dir()) if RUNS.exists() else []
    return started[-1] if started else RUNS / f"MV-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"


def done(folder: Path) -> dict[str, dict[str, Any]]:
    """The latest attempt of each session (earlier attempts are kept beside it as <id>.attempt<N>.json)."""
    out = {}
    for path in sorted(folder.glob("*.json")):
        if path.name == "run.json" or ".attempt" in path.name:
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[row["id"]] = row
    return out


def attempts(folder: Path, item_id: str) -> int:
    return len(list(folder.glob(f"{item_id}.attempt*.json"))) + (1 if (folder / f"{item_id}.json").exists() else 0)


def pending_of(folder: Path, items: list[dict[str, Any]], most: int) -> list[dict[str, Any]]:
    """Of these planned sessions, those still to run: never run, or not scored and under `most` attempts."""
    rows = done(folder)
    return [i for i in items if i["id"] not in rows or (not rows[i["id"]].get("ran") and attempts(folder, i["id"]) < most)]


def pending(folder: Path) -> list[dict[str, Any]]:
    return pending_of(folder, schedule(), manifest()["gate"]["max_attempts"])


def setup() -> dict[str, Any]:
    """What must not change between the chunks of one run: the manifest, the intern's model, its prompts and tools, the code
    that builds and scores a session, and whether its thinking is asked for."""
    from dclab_rnd import prompts
    from dclab_rnd.models import settings

    f = prompts.fingerprints()
    code = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest()[:16]
            for n in ("dclab_rnd/intern/loop.py", "dclab_rnd/memory/reading.py", "dclab_rnd/memory/writing.py", "dclab_rnd/agent_eval/run.py")}
    return {"manifest_sha256": manifest_sha(), "thinking": settings.thinking_for("intern"),
            "prompts": {k: v for k, v in f.items() if k in ("prompt:intern", "prompt:intern_memory", "prompt:memory_brief", "tools:project", "tools:session", "validator:graph")},
            "code": code}


def run_chunk(gateway: Any, folder: Path, cap_eur: float | None, max_requests: int, max_steps: int, log=print) -> dict[str, Any]:
    """The next planned sessions until the next one might cross the cap; each kept as soon as it ends."""
    m = manifest()
    return run_items(gateway, folder, lambda: pending(folder), lambda *a: one_session(*a), campaign=CAMPAIGN, now=setup(),
                     estimate=m["estimate_eur_per_session"], cap_eur=cap_eur, max_requests=max_requests, max_steps=max_steps, log=log)


def run_items(gateway: Any, folder: Path, todo, session, *, campaign: str, now: dict[str, Any], estimate: float, cap_eur: float | None,
              max_requests: int, max_steps: int, log=print) -> dict[str, Any]:
    """The loop every memory study shares: `todo()` gives the sessions still to run, `session(gateway, item, budget, steps,
    home)` runs one. Refuses a changed setup or thinking off; stops before the next session might cross the cap."""
    from unittest import mock

    from dclab_rnd.models import gateway as gateway_module, install

    from .live import _counted, _NoModels

    now = {**now, "model": getattr(gateway.client("intern"), "model", None)}
    if now["thinking"] is None:
        raise SystemExit("the intern's thinking is off (DCLAB_THINKING): every session must keep its thinking; nothing was sent")
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder / "run.json"
    if meta.exists():
        then = json.loads(meta.read_text(encoding="utf-8"))
        moved = sorted(k for k in now if then.get(k) != now[k])
        if moved:
            raise SystemExit(f"{folder.name} was started under another setup ({', '.join(moved)}): one verdict never mixes two; nothing was sent")
    else:
        meta.write_text(json.dumps({"campaign": campaign, "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **now,
                                    "training_use": TRAINING_USE}, indent=1) + "\n", encoding="utf-8")
    rows = done(folder)
    for item in todo():  # a session a whole cap could not hold would be cut again: say so instead of paying again
        old = rows.get(item["id"])
        if old and not old.get("ran") and old.get("status") == "budget_exhausted" and cap_eur is not None and (old.get("budget_eur") or 0) >= cap_eur:
            raise SystemExit(f"{item['id']} was cut with the whole cap of €{old['budget_eur']}: raise --cap-eur; nothing was sent")
    dearest = max((float(r.get("eur") or 0) for r in rows.values()), default=0.0) or estimate
    counter = {"requests": 0}
    spent = 0.0
    _counted(gateway, counter, max_requests)
    previous = gateway_module._INSTALLED
    install(_NoModels(gateway))
    quiet = (mock.patch("dclab_rnd.studio.agent._chat", lambda *a, **k: None), mock.patch("dclab_rnd.studio.agent.explain_stage", lambda *a, **k: None))
    for patch in quiet:
        patch.start()
    ran = []
    try:
        for item in todo():
            if cap_eur is not None and spent + dearest > cap_eur:
                log(f"  stopping before {item['id']}: €{spent:.3f} spent, the dearest session so far costs €{dearest:.3f}, the cap is €{cap_eur}")
                break
            if counter["requests"] >= max_requests:
                log(f"  stopping: the request cap ({max_requests}) is reached")
                break
            with tempfile.TemporaryDirectory() as tmp:
                row = session(gateway, item, None if cap_eur is None else max(0.0, cap_eur - spent), max_steps, Path(tmp))
            row["kept_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            path = folder / f"{item['id']}.json"
            if path.exists():  # an earlier attempt is kept whole beside it, never overwritten
                path.rename(folder / f"{item['id']}.attempt{attempts(folder, item['id'])}.json")
            path.write_text(json.dumps(row, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
            spent += float(row.get("eur") or 0)
            dearest = max(dearest, float(row.get("eur") or 0))
            ran.append(item["id"])
            log(f"  {item['id']:16s} {row['status']:16s} against={row.get('against')} leak={row.get('leak_caught')} "
                f"false_alarm={row.get('false_alarm')} shown={row['memory_shown']} €{float(row.get('eur') or 0):.3f}")
    finally:
        for patch in quiet:
            patch.stop()
        install(previous)
    return {"ran": ran, "spent_eur": round(spent, 4), "requests": counter["requests"], "left": len(todo())}


# ------------------------------------------------------------------------------------------------ the verdict
def _wilson(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.96
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _share(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    vals = [bool(r[key]) for r in rows if r.get(key) is not None]
    return {"share": round(sum(vals) / len(vals), 4) if vals else None, "ci95": _wilson(sum(vals), len(vals)), "n": len(vals)}


def _bootstrap(diffs: dict[str, list[float]], draws: int, seed: int) -> list[float]:
    """95% interval of the mean paired difference, resampling cases (each with all its repeats): reported, not the gate."""
    rng = random.Random(seed)
    keys = sorted(diffs)
    means = []
    for _ in range(draws):
        pick = [rng.choice(keys) for _ in keys]
        means.append(statistics.fmean([v for k in pick for v in diffs[k]]))
    means.sort()
    return [round(means[int(0.025 * draws)], 4), round(means[min(draws - 1, int(0.975 * draws))], 4)]


def sign_flip_p(case_means: list[float]) -> float:
    """Exact one-sided p that the mean case difference is this low by chance: every sign pattern of the case means."""
    import itertools

    observed = statistics.fmean(case_means)
    patterns = list(itertools.product((1, -1), repeat=len(case_means)))
    low = sum(1 for signs in patterns if statistics.fmean(s * v for s, v in zip(signs, case_means)) <= observed + 1e-12)
    return round(low / len(patterns), 5)


def summarize(folder: Path) -> dict[str, Any]:
    m = manifest()
    return judge(done(folder), schedule(), {s["case"]: s.get("role", "treated") for s in m["scenarios"]}, m["gate"], len(pending(folder)))


def judge(rows: dict[str, dict[str, Any]], plan: list[dict[str, Any]], role: dict[str, str], gate: dict[str, Any], left: int,
          leaks_over: str = "control") -> dict[str, Any]:
    """The paired verdict over planned sessions: `role` names each case treated or control; the leak guard reads the
    control cases (v1) or every case with a planted leak ("all", v2: a treated case may hold a leak beside its correction)."""
    ran = {i["id"]: rows[i["id"]] for i in plan if rows.get(i["id"], {}).get("ran")}
    pairs: dict[str, list[float]] = {}
    paired: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for i in plan:
        if i["arm"] != "on":
            continue
        on, off = ran.get(i["id"]), ran.get(i["id"][: -len("on")] + "off")
        if on and off:
            paired.append((on, off))
            if role[i["case"]] == "treated":
                pairs.setdefault(i["case"], []).append(float(on["against_memory"]) - float(off["against_memory"]))

    def arm_of(a: str, which: str | None = None) -> dict[str, Any]:
        rs = [p[0 if a == "on" else 1] for p in paired if which is None or role[p[0]["case"]] == which]
        return {"sessions": len(rs), **{k: _share(rs, k) for k in ("against_memory", "leak_caught", "false_alarm", "solution_saved")},
                "input_tokens": round(statistics.fmean(r["tokens"][0] or 0 for r in rs)) if rs else None,
                "output_tokens": round(statistics.fmean(r["tokens"][1] or 0 for r in rs)) if rs else None,
                "eur": round(sum(float(r.get("eur") or 0) for r in rs), 4), "memory_cited": sum(r.get("memory_cited") or 0 for r in rs)}

    arms = {w: {a: arm_of(a, None if w == "all" else w) for a in ("on", "off")} for w in ("treated", "control", "all")}
    planned_pairs = {w: sum(1 for i in plan if i["arm"] == "on" and role[i["case"]] == w) for w in ("treated", "control")}
    got_pairs = {w: sum(1 for on, _ in paired if role[on["case"]] == w) for w in ("treated", "control")}
    out: dict[str, Any] = {"planned_sessions": len(plan), "scored_sessions": len(ran), "pairs": got_pairs, "planned_pairs": planned_pairs,
                           "not_scored": sorted(k for k, r in rows.items() if not r.get("ran")), "left": left, "arms": arms,
                           "by_case": {c: {"pairs": len(v), "against_on_minus_off": round(statistics.fmean(v), 3)} for c, v in sorted(pairs.items())}}
    if out["left"]:
        return {**out, "verdict": "incomplete", "note": f"{out['left']} planned sessions can still run: no verdict yet"}
    treated = [c for c, r in role.items() if r == "treated"]
    enough = (all(got_pairs[w] >= gate["min_pair_share"] * planned_pairs[w] for w in ("treated", "control"))
              and all(len(pairs.get(c, [])) >= gate["min_pairs_per_case"] for c in treated))
    if not enough:
        return {**out, "verdict": "incomplete", "note": "too many sessions failed every attempt: the pairs left are below the manifest's minimum"}
    case_means = [statistics.fmean(pairs[c]) for c in treated]
    mean = round(statistics.fmean([v for c in treated for v in pairs[c]]), 4)
    p = sign_flip_p(case_means)
    t, c, a = arms["treated"], arms["control"], arms["all"]

    def diff(block: dict[str, Any], key: str) -> float | None:
        x, y = block["on"][key]["share"], block["off"][key]["share"]
        return None if x is None or y is None else round(x - y, 4)

    checks = {
        "fewer_decisions_against_memory": mean <= gate["against_mean_at_most"] and p < gate["sign_flip_p_below"],
        "controls_keep_their_traps": (diff(a if leaks_over == "all" else c, "leak_caught") or 0.0) >= -gate["control_loss_at_most"],
        "controls_no_new_false_alarms": (diff(c, "false_alarm") or 0.0) <= gate["control_loss_at_most"],
        "solutions_still_saved": (diff(a, "solution_saved") or 0.0) >= -gate["control_loss_at_most"],
        "tokens_bounded": a["on"]["input_tokens"] <= gate["input_tokens_ratio_at_most"] * max(1, a["off"]["input_tokens"]),
    }
    return {**out, "against_on_minus_off": {"mean": mean, "sign_flip_p": p, "bootstrap_ci95": _bootstrap(pairs, gate["draws"], gate["seed"]),
                                            "pairs": sum(len(v) for v in pairs.values()), "cases": len(pairs)},
            "checks": checks, "verdict": "memory_helps" if all(checks.values()) else "memory_does_not_help"}


QUESTION = "Does the intern's second session on a project judge fewer columns against what an earlier session settled when it reads memory?"
LIMITS = [SCOPE, "the earlier session is scripted through the write paths; a real person's earlier session can say less, or say it worse",
          "the dev cases are the judgment suite's; the sealed read waits for the owner and needs cases no run has seen",
          "questions asked again are not measured: the intern asks none in this suite, and Home does not read memory yet"]


def store_result(folder: Path, summary: dict[str, Any], *, results: Path | None = None, campaign: str | None = None, prefix: str = "MV",
                 limitations: list[str] | None = None, extra: dict[str, Any] | None = None) -> Path:
    """One result per run, numbered <prefix>-NNN_dev.json; summarizing the same run again is refused."""
    results = results or RESULTS
    campaign = campaign or CAMPAIGN
    results.mkdir(parents=True, exist_ok=True)
    for old in results.glob(f"{prefix}-*.json"):  # one result per run: summarizing again never adds a second
        if json.loads(old.read_text(encoding="utf-8")).get("run") == folder.name:
            raise SystemExit(f"{folder.name} is already stored as {old.name}")
    numbers = [int(x.group(1)) for p in results.glob(f"{prefix}-*.json") if (x := re.match(rf"{re.escape(prefix)}-(\d+)", p.name))]
    meta = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    path = results / f"{prefix}-{max(numbers, default=0) + 1:03d}_dev.json"
    doc = {"experiment_id": path.stem.split("_")[0], "campaign_id": campaign, "kind": "memory_value_dev", "status": "completed",
           "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "split": "dev", "model": meta.get("model"), "run": folder.name,
           "setup": {k: meta.get(k) for k in ("manifest_sha256", "thinking", "prompts", "code")},
           "question": QUESTION, "measurement": summary, "sessions_kept": f"agent_runs/live_tests/{campaign}/{folder.name}/",
           "training_use": "every session is kept whole (plan, steps with their thinking, messages, report); " + TRAINING_USE,
           "limitations": limitations or LIMITS, **(extra or {})}
    with path.open("x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return path


def main(argv: list[str] | None = None, gateway: Any = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.memory_value", description=__doc__.split("\n")[0])
    parser.add_argument("action", nargs="?", choices=("run", "summarize"), default="run")
    parser.add_argument("--yes", action="store_true", help="send; without it the plan is printed and nothing is sent")
    parser.add_argument("--cap-eur", type=float, help="stop before this invocation costs more (required for a priced remote model)")
    parser.add_argument("--max-requests", type=int, default=200, help="requests this invocation may send (default 200)")
    parser.add_argument("--max-steps", type=int, default=16, help="tool calls a session (default 16, as the judgment suite)")
    parser.add_argument("--run", help="the run folder to resume (default: the latest started)")
    args = parser.parse_args(argv)
    folder = run_dir(args.run)
    if args.action == "summarize":
        s = summarize(folder)
        print(json.dumps({k: v for k, v in s.items() if k != "by_case"}, indent=1))
        if s["verdict"] != "incomplete":
            print(f"Stored {store_result(folder, s)}")
        return 0
    if gateway is None:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env", override=False)
        from dclab_rnd.models.gateway import for_workspace

        gateway = for_workspace()
        gateway.shadows = None
    from dclab_rnd.models import prices, settings

    name = gateway.route("intern")[0] if hasattr(gateway, "route") else settings.purpose("intern").tier
    tier = settings.public(name)
    priced = prices.price_of(str(tier.get("model") or "")) is not None and not tier.get("local")
    plan = schedule()
    m = manifest()
    print(f"Model {tier.get('model')} · {len(m['scenarios'])} cases × {m['repeats']} repeats × 2 arms = {len(plan)} sessions; "
          f"{len(pending(folder))} left in {folder.name}")
    print(f"About €{m['estimate_eur_per_session']} a session by the plan's estimate; this invocation stops before "
          + (f"€{args.cap_eur}" if args.cap_eur is not None else f"{args.max_requests} requests"))
    print(SCOPE)
    if gateway.client("intern") is None:
        print("No model serves the intern (DCLAB_NO_LIVE_MODELS, or no key); nothing was sent.", file=sys.stderr)
        return 2
    if priced and args.cap_eur is None:
        print("The intern's model is priced: give --cap-eur; nothing was sent.", file=sys.stderr)
        return 2
    if not args.yes:
        print("Nothing was sent. Run again with --yes.", file=sys.stderr)
        return 2
    out = run_chunk(gateway, folder, args.cap_eur, args.max_requests, args.max_steps)
    print(f"Ran {len(out['ran'])} sessions · €{out['spent_eur']} · {out['requests']} requests · {out['left']} sessions left")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
