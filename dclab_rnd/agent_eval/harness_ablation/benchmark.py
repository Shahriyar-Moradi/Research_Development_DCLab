"""The harness ablation's benchmark: every judged run by arm, case by case, where the lead agent's code stepped in, the
same cases on two models side by side, and what the earlier Claude-era exams showed.

``python -m dclab_rnd.agent_eval.harness_ablation.benchmark`` rebuilds ``BENCHMARK.json`` and ``BENCHMARK.md`` beside the
campaign's results and ``docs/evaluations/harness-vs-model/benchmark.html`` (one self-contained page) from the result
files and the kept conversations. Nothing is run and nothing is spent. A sealed exam is read only as its stored summary
(its result file), never a conversation.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping, Optional

from . import ALL_ARMS, ARMS, CAMPAIGN, HELD_OUT, RESULTS, ROOT, compare
from . import first_divergence as F

PAGE = ROOT / "docs" / "evaluations" / "harness-vs-model" / "benchmark.html"
EXAM_RESULTS = ROOT / "evidence" / "campaigns" / "conversation_exam_v1" / "results"
DIMENSIONS = ("understanding", "solution_design", "planning", "evidence_discipline")
NAMES = {"blueprint_rules": "No-data blueprint: rules only", "blueprint_pipeline": "No-data blueprint: today's pipeline of roles",
         "blueprint_architect": "No-data blueprint: architect, checks, critic",
         "model_only_short": "Model only, held to 450 words", "model_only": "Model only", "open": "Minimal agent", "plain": "Minimal agent + DCLab's 6 kinds of work", "case": "Current DCLab (lead agent)"}
WORDS = {"understanding": "Understands the problem", "solution_design": "Designs a solution", "planning": "Makes a plan",
         "evidence_discipline": "Sticks to evidence", "accepted": "Accepted"}
REASON_WORDS = {
    "query_cannot_express": "The table tool cannot ask the question", "query_cap": "Limit on table questions per turn",
    "other_refusal": "Another refusal (a build without a yes, a bad argument)", "build_needs_yes": "A build asked for without the person's yes",
    "build_pick_not_yes": "A build asked for on a pick (a pick is not a yes)", "build_needs_start": "A build asked for before a project was started",
    "outcome_not_named": "An outcome column the person never named", "reply_cut": "A reply cut at its length limit", "route_overruled": "A second small model overrules the kind of work",
    "options_dropped": "An option removed by a check", "options_trimmed": "An option kept, one sentence with a number removed", "budget_exhausted": "The turn's step budget ran out",
    "start_refused_not_ready": "Work refused: the case is \"not ready\"", "propose_refused_not_ready": "Options refused: the case is \"not ready\"",
    "advice_refused": "Advice refused: the words ask for data work", "needs_table": "A table is needed first", "bad_arguments": "The model's call was malformed",
    "closed_list": "A kind of work outside DCLab's list", "guard_refused": "No new options after work started", "route_set_by_code": "Code set the kind of work",
    "no_data_redirect": "Sent to the no-data blueprint", "chip_bound": "Bound to the person's own pick", "route_change_refused": "A change of work refused",
    "work_done_no_route": "Work done, no kind of work recorded", "picked_wrong_option": "The simulated person picked a wrong option (no refusal recorded)",
    "model_error": "The model chose the wrong kind of work (no refusal recorded)", "arm_error": "The run broke", "no_gate": "Nothing stepped in",
    "no_route": "No kind of work was set"}
# A run played before the runner had its --label switch, with the code state it measured (HAB-006 started at 12:14 UTC on
# 2026-11-15, after the three fixes of that day went in: the same state as the runs labelled so)
LABEL_OF = {"HAB-006": "after-three-fixes"}


def same_judge(model: Any) -> Any:
    """A judge model's name without its provider variant: the free and the paid serving of one model are one judge."""
    return str(model)[:-len(":free")] if isinstance(model, str) and model.endswith(":free") else model


def think_first_of(doc: Mapping[str, Any]) -> bool:
    """Whether a run played the lead agent with its think-first switch on (stage 4), as its manifest recorded it: such a run
    is never read as one with a run of the switch off, whatever their labels."""
    value = str((((doc.get("manifest") or {}).get("flags") or {}).get("DCLAB_CASE_THINK_FIRST")) or "").strip().lower()
    return value in ("1", "on", "true", "yes")


def label_of(doc: Mapping[str, Any]) -> Optional[str]:
    return doc.get("label") or (doc.get("manifest") or {}).get("label") or LABEL_OF.get(str(doc.get("id")))


SEALED_EXAMS = ("CEX-013_sealed_v1", "CEX-017_sealed_v2", "CEX-032_sealed_v3")  # the Claude-era sealed exams: plain against the lead agent


def _mean(values: list[Any]) -> Optional[float]:
    kept = [float(v) for v in values if v is not None]
    return round(sum(kept) / len(kept), 2) if kept else None


def load(path: Path) -> dict[str, Any]:
    """A result file; each of its conversations remembers the folder its run was kept in (runs can be joined)."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    for conv in doc.get("conversations") or []:
        conv["_kept"] = doc.get("kept")
    return doc


def join(docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Several runs of the same cases on one model under one judge, read as one (a run that stopped early and the run that
    finished it): each case and arm keeps its latest run's conversation."""
    if len(docs) == 1:
        return docs[0]
    docs = sorted(docs, key=lambda d: str(d.get("id")))
    rows: dict[str, dict[str, Any]] = {}
    convs: dict[tuple[Any, Any], Any] = {}
    for doc in docs:
        answered = {(c.get("arm"), c.get("id")) for c in doc.get("conversations") or [] if c.get("ended") != "error"}
        for arm, arm_rows in (doc.get("scores") or {}).items():
            for r in arm_rows:
                if (arm, r.get("case")) in answered or r.get("case") not in rows.get(arm, {}):  # a later run's refused conversation never replaces an answered one
                    rows.setdefault(arm, {})[r.get("case")] = r
        for c in doc.get("conversations") or []:
            if c.get("ended") != "error" or (c.get("arm"), c.get("id")) not in convs:
                convs[(c.get("arm"), c.get("id"))] = c
    return {**docs[-1], "id": " + ".join(str(d.get("id")) for d in docs), "scores": {arm: [by[c] for c in sorted(by)] for arm, by in rows.items()},
            "conversations": list(convs.values()), "paired": None, "spent_eur": round(sum(float(d.get("spent_eur") or 0) for d in docs), 4),
            "incomplete": None}


def read_run(doc: dict[str, Any]) -> dict[str, Any]:
    """One judged run: its arms' rows by case, their means, the interventions of code in the lead agent's conversations."""
    # a conversation the provider refused (the key's limit, no credit, the run's cap) was never answered: it is no arm's failure
    refused = {(c.get("arm"), c.get("id")) for c in doc.get("conversations") or []
               if c.get("ended") == "error" and any(w in str(c.get("error")) for w in ("refused the key", "no credit", "ModelPaused", "BudgetExceeded"))}
    scores = {arm: [r for r in doc["scores"][arm] if (arm, r.get("case")) not in refused] for arm in ALL_ARMS if arm in (doc.get("scores") or {})}
    interventions: dict[str, int] = {}
    per_case: dict[str, dict[str, int]] = {}
    counted: dict[str, dict[str, int]] = {}
    for conv in doc.get("conversations") or []:
        if conv.get("arm") != "case":
            continue
        folder = ROOT / str(conv.get("_kept") or doc.get("kept") or "") / "case" / f"{conv['id']}-r{conv.get('repeat', 1)}"
        found = F.read(folder, person={"id": conv["id"], "route": None}, conv=conv)["all"] if folder.exists() else (conv.get("first_divergence") or {}).get("all") or []
        kept = F.counters(folder, conv) if folder.exists() else (conv.get("first_divergence") or {}).get("counters")
        if kept:  # the gate's zero checks: counted from the kept conversation, or as the run stored them
            s = conv.get("safety") or {}
            counted[conv["id"]] = {"blocked_runs": len(kept["blocked_runs"]), "fallback_lines": kept["fallback_lines"], "empty_answers": kept["empty_answers"],
                                   "cut_replies": kept["cut_replies"], "built_without_yes": int(s.get("built_without_yes") or 0),
                                   "numbers_nobody_gave": len(s.get("uncited") or [])}
        for item in found:
            interventions[item["reason"]] = interventions.get(item["reason"], 0) + 1
            per_case.setdefault(conv["id"], {})[item["reason"]] = per_case.setdefault(conv["id"], {}).get(item["reason"], 0) + 1
    arms = {}
    for arm, rows in scores.items():
        judged = [r for r in rows if r.get("judged") or r.get("failed")]
        chars = {c["id"]: sum(len(t.get("text") or "") for t in c.get("transcript") or [] if t.get("role") == "assistant")
                 for c in doc.get("conversations") or [] if c.get("arm") == arm}
        arms[arm] = {"cases": len(rows), "answered": sum(1 for r in rows if (r.get("assistant_turns") or 0) > 0), "judged": sum(bool(r.get("judged")) for r in rows),
                     **{d: _mean([r.get(d) for r in judged]) for d in DIMENSIONS},
                     "accepted": sum(bool(r.get("accepted")) for r in judged), "forbidden_done": sum(int(r.get("forbidden_done") or 0) for r in rows),
                     "numbers_nobody_gave": sum(int(r.get("safety_failures") or 0) for r in rows),
                     "seconds": _mean([r.get("seconds") for r in rows]), "assistant_turns": _mean([r.get("assistant_turns") for r in rows]),
                     "spent_eur": round(sum(float(r.get("spent_eur") or 0) for r in rows), 4),
                     "empty_or_short_answers": sorted(cid for cid, n in chars.items() if n < 1000 and arm != "case"),
                     "by_case": {r["case"]: {**{d: r.get(d) for d in DIMENSIONS}, "accepted": r.get("accepted"), "seconds": r.get("seconds"),
                                             "forbidden_done": r.get("forbidden_done")} for r in rows}}
    return {"id": doc["id"], "at": doc.get("at"), "set": doc.get("set") or "dev", "think_first": think_first_of(doc), "model": (doc.get("models") or {}).get("arms"),
            "judge": same_judge((doc.get("models") or {}).get("judge")), "judge_as_run": (doc.get("models") or {}).get("judge"),
            "label": label_of(doc) if " + " not in str(doc.get("id")) else doc.get("label") or (doc.get("manifest") or {}).get("label") or next(
                (LABEL_OF[p] for p in str(doc["id"]).split(" + ") if p in LABEL_OF), None),
            "commit": ((doc.get("manifest") or {}).get("code") or {}).get("commit"), "judge_spent_eur": doc.get("spent_eur"),
            "arms": arms, "paired": doc.get("paired") or compare(scores), "interventions": dict(sorted(interventions.items(), key=lambda kv: -kv[1])),
            "interventions_by_case": per_case, "counted": counted, "incomplete": doc.get("incomplete")}


def across_models(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """Run ``b`` against run ``a`` on the same cases, arm by arm: the paired difference by case (positive: ``b`` is higher)."""
    from dclab_core import bench as core

    out: dict[str, Any] = {}
    for arm in ALL_ARMS:
        if arm not in a["arms"] or arm not in b["arms"]:
            continue
        out[arm] = {}
        for name in (*DIMENSIONS, "accepted"):
            def by_case(run: Mapping[str, Any]) -> dict[str, float]:
                return {cid: float(row[name]) for cid, row in run["arms"][arm]["by_case"].items() if row.get(name) is not None}

            out[arm][name] = core.paired(by_case(a), by_case(b))
    return out


LABEL_ORDER = (None, "after-three-fixes", "stage-1", "stage-2", "stage-3", "stage-4")  # the states of the lead agent, in the order they were made


def _order(label: Any) -> int:
    return LABEL_ORDER.index(label) if label in LABEL_ORDER else len(LABEL_ORDER)


def before_after(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The lead agent before and after a change: two runs of one model under one judge whose labels differ (the unlabelled
    run is the earlier state). The other arms do not change with the lead agent's code, so only its arm is compared."""
    from dclab_core import bench as core

    out = []
    for a in runs:
        for b in runs:
            if a is b or a["model"] != b["model"] or a["judge"] != b["judge"] or a["set"] != b["set"] or not b.get("label"):
                continue
            # each state against the one before it that was measured (no label: the pilot; then the labels in their order)
            earlier = [r for r in runs if r["model"] == b["model"] and r["judge"] == b["judge"] and r["set"] == b["set"] and "case" in r["arms"]
                       and _order(r.get("label")) < _order(b.get("label"))]
            if not earlier or a is not max(earlier, key=lambda r: _order(r.get("label"))):
                continue
            if "case" not in a["arms"] or "case" not in b["arms"]:
                continue
            before, after = a["arms"]["case"], b["arms"]["case"]
            paired = {}
            for name in (*DIMENSIONS, "accepted"):
                def by_case(arm: Mapping[str, Any]) -> dict[str, float]:
                    return {cid: float(row[name]) for cid, row in arm["by_case"].items() if row.get(name) is not None}

                paired[name] = core.paired(by_case(before), by_case(after))
            counted = {key: (before.get(key), after.get(key)) for key in ("seconds", "assistant_turns", "numbers_nobody_gave", "spent_eur", "accepted", "cases")}
            reasons = sorted(set(a["interventions"]) | set(b["interventions"]), key=lambda r: -(a["interventions"].get(r, 0) + b["interventions"].get(r, 0)))
            out.append({"model": a["model"], "judge": a["judge"], "set": b["set"], "before": a["id"], "after": b["id"], "label": b["label"],
                        "means": {d: (before.get(d), after.get(d)) for d in DIMENSIONS}, "paired": paired, "counted": counted,
                        "interventions": {r: (a["interventions"].get(r, 0), b["interventions"].get(r, 0)) for r in reasons}})
    return out


def judges_agree(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Two judges on the same conversations: how often they agree on "accepted", and how far apart their scores are."""
    out = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if a["id"] != b["id"] or a["judge"] == b["judge"]:
                continue
            pairs = [(ra, b["arms"][arm]["by_case"][cid]) for arm, s in a["arms"].items() if arm in b["arms"]
                     for cid, ra in s["by_case"].items() if cid in b["arms"][arm]["by_case"]]
            both = [(x, y) for x, y in pairs if x.get("accepted") is not None and y.get("accepted") is not None]
            gaps = [abs(float(x[d]) - float(y[d])) for x, y in pairs for d in DIMENSIONS if x.get(d) is not None and y.get(d) is not None]
            out.append({"run": a["id"], "judge_a": a["judge"], "judge_b": b["judge"], "conversations": len(both),
                        "accepted_agree": sum(1 for x, y in both if bool(x["accepted"]) == bool(y["accepted"])),
                        "mean_score_gap": round(sum(gaps) / len(gaps), 2) if gaps else None,
                        "within_one_point": round(sum(1 for g in gaps if g <= 1) / len(gaps), 2) if gaps else None})
    return out


def claude_era() -> dict[str, Any]:
    """What the earlier exams on Claude showed: the kept dev runs read again (HAB-000), and each sealed exam's stored summary
    for the model alone and the lead agent. Different people and a different judge from the pilot: context, not a pairing."""
    out: dict[str, Any] = {"recorded_dev": None, "sealed": []}
    hab0 = RESULTS / "HAB-000_recorded_dev.json"
    if hab0.exists():
        doc = json.loads(hab0.read_text(encoding="utf-8"))
        out["recorded_dev"] = {"runs": len(doc.get("runs") or []), "lost": doc.get("lost"), "won": doc.get("won"), "first_reason_counts": doc.get("first_reason_counts")}
    for name in SEALED_EXAMS:
        path = EXAM_RESULTS / f"{name}.json"
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        summary = doc.get("summary") or {}
        row = {"run": doc.get("id"), "split": doc.get("split"), "model": (doc.get("models") or {}).get("arms"), "judge": (doc.get("models") or {}).get("judge")}
        for arm in ("plain", "case"):
            s = summary.get(arm) or {}
            row[arm] = {k: s.get(k) for k in ("people", "understanding", "options_fit", "route_reached", "safety_failures", "seconds_per_turn")}
        out["sealed"].append(row)
    return out


MODEL_REPORTS = ROOT / "evidence" / "bench" / "models"
CROSS = {  # suite -> (its name here, the arms shown, metric -> (plain words, higher is better))
    "harness_ablation_v1": ("These eight cases (the Claude run stopped early: 2 to 3 cases, not judged)", ("model_only", "open", "plain", "case"),
                            {"seconds": ("Seconds per conversation", False), "assistant_turns": ("Turns", False), "safety_failures": ("Numbers nobody gave", False),
                             "spent_eur": ("Euros per conversation", False)}),
    "conversation_exam_v1": ("The conversation exam, practice people (Claude judged by Opus, Qwen by Qwen)", ("plain", "case"),
                             {"understanding": ("Understanding (0 to 1)", True), "options_fit": ("Option fit (1 to 5)", True), "route_right": ("Right kind of work", True),
                              "trap_pass": ("Traps handled", True), "safety_failures": ("Numbers nobody gave", False), "asked": ("Questions asked", False),
                              "seconds_per_turn": ("Seconds per turn", False), "spent_eur": ("Euros per conversation", False)}),
    "home_v1": ("Home conversations, 25 written people on both models", ("home",),
                {"right": ("Right kind of help chosen", True), "right_offered": ("Right kind of help offered", True), "turns_before_options": ("Turns before options", False)}),
}
THIN = '<span class="chip warn">thin</span>'
CROSS_ARMS = {**NAMES, "home": "Home (rules first, with the model)"}


def cross_model() -> dict[str, Any]:
    """Claude beside Qwen, from the newest cross-model report (``python -m dclab_rnd.bench models``): the same suite, split and
    arm, each side's mean and how many cases it stands on. Only practice splits; a side with fewer than five cases is marked."""
    reports = sorted(MODEL_REPORTS.glob("QWEN_VS_CLAUDE_*.json"))
    if not reports:
        return {"report": None, "suites": []}
    doc = json.loads(reports[-1].read_text(encoding="utf-8"))
    suites = []
    for suite, (title, arms, metrics) in CROSS.items():
        rows = []
        for r in doc.get("rows") or []:
            if r.get("suite") != suite or r.get("split") != "dev" or r.get("arm") not in arms or r.get("metric") not in metrics:
                continue
            a, b = r.get("baseline") or {}, r.get("target") or {}
            words, higher = metrics[r["metric"]]
            better = None
            if a.get("mean") is not None and b.get("mean") is not None and a["mean"] != b["mean"]:
                better = "claude" if (a["mean"] > b["mean"]) == higher else "qwen"
            rows.append({"arm": r["arm"], "metric": r["metric"], "words": words, "higher_is_better": higher,
                         "claude": a.get("mean"), "claude_n": a.get("n"), "claude_model": a.get("model"),
                         "qwen": b.get("mean"), "qwen_n": b.get("n"), "qwen_model": b.get("model"),
                         "better": better, "thin": min(int(a.get("n") or 0), int(b.get("n") or 0)) < 5, "judges_differ": bool(r.get("judges_differ"))})
        rows.sort(key=lambda x: (arms.index(x["arm"]), list(metrics).index(x["metric"])))
        if rows:
            suites.append({"suite": suite, "title": title, "rows": rows})
    return {"report": str(reports[-1].relative_to(ROOT)), "target": doc.get("target"), "baseline": doc.get("baseline"), "suites": suites}


def build() -> dict[str, Any]:
    def grouped(paths: list[Path]) -> list[dict[str, Any]]:
        """Result files as runs: the files of one model under one judge are one run (a stopped run and the run that finished it)."""
        groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
        for path in paths:
            doc = load(path)
            models = doc.get("models") or {}
            label = label_of(doc)  # the state of the code a run measures
            groups.setdefault((models.get("arms"), same_judge(models.get("judge")), label, doc.get("set") or "dev", think_first_of(doc)), []).append(doc)
        return [read_run(join(docs)) for docs in groups.values()]

    # a run's first judgment, and any other judge's judgment of the same conversations, each as its own row
    runs = grouped([*sorted(RESULTS.glob("HAB-*_dev.json")), *sorted(RESULTS.glob("HAB-*_dev_judge-*.json"))])
    runs = [r for r in runs if any(a["judged"] for a in r["arms"].values())]
    judged = {part for r in runs for part in str(r["id"]).split(" + ")}
    # a run that was played and not judged yet (the judge had no credit, or the run stopped): what code can count without a judge
    waiting = [r for r in grouped([p for p in sorted(RESULTS.glob("HAB-*_dev_unjudged.json")) if p.name.split("_")[0] not in judged])
               if any(a["answered"] for a in r["arms"].values())]
    pairs = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if a["model"] != b["model"] and a["judge"] == b["judge"] and a.get("label") == b.get("label") and a["set"] == b["set"]:  # two models: one judge, one state of the code, one set of people
                pairs.append({"a": a["id"], "a_model": a["model"], "b": b["id"], "b_model": b["model"], "judge": a["judge"], "arms": across_models(a, b)})
    derived = {"all_models": all_models(runs), "gate": gated(runs), "across_models": pairs, "before_after": before_after(runs), "judges": judges_agree(runs)}
    shown = [means_only(r) if r.get("set") in HELD_OUT else r for r in runs]  # after everything that reads a run by person was computed
    waiting = [means_only(r) if r.get("set") in HELD_OUT else r for r in waiting]
    return {"what": "the harness ablation's benchmark: built from the result files, nothing run", "runs": shown, "not_judged_yet": waiting,
            **derived, "cross_model": cross_model(), "claude_era": claude_era(),
            "limits": ["eight written cases, each once: differences inside the intervals are noise",
                       "one judge model scored every run; it is from the same family as one of the two arm models",
                       "the cases and their answer key were written by the runner's author",
                       "an empty or cut answer of a baseline arm is scored as given (the runner's limits of five model calls a turn and its token limit)",
                       "the Claude-era exams used other people, a simulated person and another judge: they are context, not a pairing"]}


def gated(runs: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The gate (``gate.py``) on every judged run that holds the lead agent: its rows against the same model alone under the
    same judge (from the same run, or from another run of that model: the model alone does not change with the lead agent's code)."""
    from . import gate as G

    def rows(run: Mapping[str, Any], arm: str) -> list[dict[str, Any]]:
        counted = run.get("counted") or {}
        return [{"case": case, "judged": r.get("planning") is not None, **r, **(counted.get(case) or {} if arm == "case" else {})}
                for case, r in ((run["arms"].get(arm) or {}).get("by_case") or {}).items()]

    out = []
    for run in runs:
        if "case" not in run["arms"]:
            continue
        alone = next((r for r in [run, *runs] if r["model"] == run["model"] and r["judge"] == run["judge"] and r["set"] == run["set"]
                      and "model_only" in r["arms"]), None)
        verdict = G.verdict(str(run["model"]), run["judge"], rows(run, "case"), rows(alone, "model_only") if alone else [], gate=G.gate_of(run["set"]))
        out.append({"run": run["id"], "label": run.get("label"), "set": run["set"], "think_first": run.get("think_first"),
                    "alone_from": alone["id"] if alone else None, "pinned": G.pinned(G.gate_of(run["set"])) is None, **verdict})
    return out


GATE_WORDS = {"hab_gate_v1": "fixed before the repair, judged on the people it was then tuned on",
              "hab_gate_v2": "fixed before stage 4, judged on the held-out people only"}


def means_only(run: dict[str, Any]) -> dict[str, Any]:
    """A held-out set's run as the benchmark keeps it: its means and counts, nothing by person (the review of stage 4: the
    page hid the by-person tables and the JSON beside it still held them)."""
    arms = {arm: {**{k: v for k, v in s.items() if k != "by_case"}, "empty_or_short_answers": len(s.get("empty_or_short_answers") or [])}
            for arm, s in run["arms"].items()}
    return {**{k: v for k, v in run.items() if k not in ("interventions_by_case", "counted")}, "arms": arms, "held_out": True}


COUNTED = (("accepted", "Accepted"), ("forbidden_done", "Must-not-do done"), ("numbers_nobody_gave", "Numbers nobody gave"),
           ("assistant_turns", "Turns"), ("seconds", "Seconds"), ("spent_eur", "€ per conversation"), ("short", "Empty or cut answers"))


def all_models(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Every judged run side by side: one row per model and judge, every measure of every arm, what the harness changes
    inside each model, and where the lead agent's code stepped in. Rows under different judges are not a pairing."""
    rows = []
    for run in runs:
        arms = {}
        for arm, s in run["arms"].items():
            arms[arm] = {**{d: s[d] for d in DIMENSIONS}, "cases": s["cases"], "judged": s["judged"], "accepted": s["accepted"],
                         "forbidden_done": s["forbidden_done"], "numbers_nobody_gave": s["numbers_nobody_gave"],
                         "assistant_turns": s["assistant_turns"], "seconds": s["seconds"],
                         "spent_eur": (s["spent_eur"] / s["answered"]) if s.get("answered") and s.get("spent_eur") is not None else None,
                         "short": len(s["empty_or_short_answers"])}
        pair = (run.get("paired") or {}).get("case_minus_open") or {}
        rows.append({"id": run["id"], "model": run["model"], "judge": run["judge"], "label": run.get("label"), "set": run.get("set"),
                     "think_first": run.get("think_first"), "arms": arms,
                     "harness": {d: pair[d] for d in (*DIMENSIONS, "accepted") if d in pair},
                     "interventions": dict(run["interventions"]), "interventions_total": sum(run["interventions"].values())})
    reasons = sorted({r for row in rows for r in row["interventions"]}, key=lambda r: -sum(row["interventions"].get(r, 0) for row in rows))
    return {"rows": rows, "reasons": reasons}


def _row_name(row: Mapping[str, Any]) -> str:
    return (f"{row['model']}" + (f" ({row['label']})" if row.get("label") else "") + f" · judge {row['judge']}"
            + (f" · people {row['set']}" if row.get("set") not in (None, "dev") else "") + (" · think first" if row.get("think_first") else ""))


def _counted(arm: Mapping[str, Any], key: str) -> str:
    if key == "accepted":
        return f"{arm['accepted']} of {arm['cases']}"
    return _num(arm.get(key), 1 if key == "assistant_turns" else (0 if key == "seconds" else 3 if key == "spent_eur" else 2))


def _num(value: Any, digits: int = 2) -> str:
    return "–" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def _interval(p: Mapping[str, Any]) -> str:
    low, high = (p.get("interval") or [None, None])
    return f"{_num(p.get('difference'))} [{_num(low)}, {_num(high)}]"


def markdown(data: Mapping[str, Any]) -> str:
    lines = ["# Harness ablation: the benchmark", "",
             "Rebuilt by `python -m dclab_rnd.agent_eval.harness_ablation.benchmark` from the result files; do not edit by hand.",
             "Scores are the judge's, 0 to 4. \"Numbers nobody gave\" are counted by code. A difference is `mean [95% interval]` by case.", ""]
    for gate_id in sorted({g["gate"] for g in data.get("gate") or []}, reverse=True):  # each gate under its own headings (v2 holds checks v1 lacks)
        of = [g for g in data["gate"] if g["gate"] == gate_id]
        lines += [f"## The gate (`{gate_id}`, {GATE_WORDS.get(gate_id, 'fixed before its runs')})", "",
                  "Every check must hold. \"not read\" means the run kept nothing to count it from, or was judged by another judge than the gate names: it is not a pass.", "",
                  "| run | model | judge | passed | " + " | ".join(c["check"] for c in of[0]["checks"]) + " |",
                  "|---|---|---|---|" + "---|" * len(of[0]["checks"])]
        for g in of:
            cells = [("ok " if c["ok"] else "NOT READ " if not c["read"] else "FAIL ") + _num(c["value"]) for c in g["checks"]]
            name = f"{g['run']}" + (f" ({g['label']})" if g.get("label") else "") + (" · think first" if g.get("think_first") else "")
            lines.append(f"| {name} | {g['model']} | {g['judge']} | {'PASS' if g['passed'] else 'fail'} | " + " | ".join(cells) + " |")
        lines.append("")
    am = data.get("all_models") or {"rows": [], "reasons": []}
    if am["rows"]:
        lines += ["## All models side by side", "",
                  "One row per model and judge. Compare arms inside a row; rows under different judges are not a pairing. "
                  "\"after-three-fixes\" marks runs on the lead agent after the fixes of 2026-11-15.", ""]
        for arm in [a for a in ALL_ARMS if any(a in row["arms"] for row in am["rows"])]:
            lines += [f"### {NAMES[arm]}", "", "| model | " + " | ".join(WORDS[d] for d in DIMENSIONS) + " | " + " | ".join(w for _, w in COUNTED) + " |",
                      "|---|" + "---:|" * (len(DIMENSIONS) + len(COUNTED))]
            lines += [f"| {_row_name(row)} | " + " | ".join(_num(row["arms"][arm][d]) for d in DIMENSIONS) + " | "
                      + " | ".join(_counted(row["arms"][arm], k) for k, _ in COUNTED) + " |" for row in am["rows"] if arm in row["arms"]]
            lines.append("")
        lines += ["### What the harness changes inside each model (current DCLab minus the minimal agent, by case, 95% interval)", "",
                  "| model | " + " | ".join(WORDS[d] for d in (*DIMENSIONS, "accepted")) + " |", "|---|" + "---|" * (len(DIMENSIONS) + 1)]
        lines += [f"| {_row_name(row)} | " + " | ".join(_interval(row["harness"][d]) if d in row["harness"] else "–" for d in (*DIMENSIONS, "accepted")) + " |"
                  for row in am["rows"] if row["harness"]]
        lines += ["", "### Where the lead agent's code stepped in, by model (all 8 conversations)", "",
                  "| what | " + " | ".join(_row_name(row) for row in am["rows"]) + " |", "|---|" + "---:|" * len(am["rows"])]
        lines += [f"| {REASON_WORDS.get(r, r)} | " + " | ".join(str(row["interventions"].get(r, 0)) for row in am["rows"]) + " |" for r in am["reasons"]]
        lines += ["| **all** | " + " | ".join(f"**{row['interventions_total']}**" for row in am["rows"]) + " |", ""]
    for run in data["runs"]:
        lines += [f"## {run['id']}: arms on `{run['model']}`, judged by `{run['judge']}`", "",
                  "| arm | " + " | ".join(WORDS[d] for d in DIMENSIONS) + " | accepted | must-not-do done | numbers nobody gave | turns | seconds | € |",
                  "|---|" + "---:|" * (len(DIMENSIONS) + 6)]
        for arm, s in run["arms"].items():
            lines.append(f"| {NAMES[arm]} | " + " | ".join(_num(s[d]) for d in DIMENSIONS) + f" | {s['accepted']} of {s['cases']} | {s['forbidden_done']} | "
                         f"{s['numbers_nobody_gave']} | {_num(s['assistant_turns'], 1)} | {_num(s['seconds'], 0)} | {_num(s['spent_eur'])} |")
        short = {NAMES[a]: s["empty_or_short_answers"] for a, s in run["arms"].items() if s["empty_or_short_answers"] and run.get("set") not in HELD_OUT}
        if short:
            lines += ["", "Empty or very short answers (scored as given): " + "; ".join(f"{k}: {', '.join(v)}" for k, v in short.items()) + "."]
        lines += ["", "**The harness inside this model** (current DCLab minus the minimal agent):", ""]
        pair = (run.get("paired") or {}).get("case_minus_open") or {}
        lines += ["| " + " | ".join(WORDS[d] for d in (*DIMENSIONS, "accepted")) + " |", "|" + "---|" * (len(DIMENSIONS) + 1),
                  "| " + " | ".join(_interval(pair[d]) if d in pair else "–" for d in (*DIMENSIONS, "accepted")) + " |", "",
                  "**Where the lead agent's code stepped in** (all 8 conversations):", "", "| what | times |", "|---|---:|"]
        lines += [f"| {REASON_WORDS.get(reason, reason)} | {n} |" for reason, n in run["interventions"].items()] or ["| nothing recorded | 0 |"]
        lines.append("")
    for run in data.get("not_judged_yet") or []:
        lines += [f"## {run['id']}: arms on `{run['model']}`, played and not judged yet" + (f" (stopped: {run['incomplete']})" if run.get("incomplete") else ""), "",
                  "No judge has scored these conversations. What code can count:", "",
                  "| arm | conversations | turns | seconds | numbers nobody gave | € |", "|---|---:|---:|---:|---:|---:|"]
        lines += [f"| {NAMES[arm]} | {s['answered']} | {_num(s['assistant_turns'], 1)} | {_num(s['seconds'], 0)} | {s['numbers_nobody_gave']} | {_num(s['spent_eur'])} |" for arm, s in run["arms"].items()]
        lines += ["", "Where the lead agent's code stepped in: " + (", ".join(f"{REASON_WORDS.get(r, r)} ({n})" for r, n in run["interventions"].items()) or "nothing recorded") + ".", ""]
    for pair in data["across_models"]:
        lines += [f"## The same cases on two models: `{pair['b_model']}` minus `{pair['a_model']}`", "",
                  "| arm | " + " | ".join(WORDS[d] for d in (*DIMENSIONS, "accepted")) + " |", "|---|" + "---|" * (len(DIMENSIONS) + 1)]
        lines += [f"| {NAMES[arm]} | " + " | ".join(_interval(p[d]) for d in (*DIMENSIONS, "accepted")) + " |" for arm, p in pair["arms"].items()]
        lines.append("")
    for ba in data.get("before_after") or []:
        c = ba["counted"]
        lines += [f"## Current DCLab before and after: {ba['label']} (`{ba['model']}`, judge `{ba['judge']}`)", "",
                  f"Before: {ba['before']}. After: {ba['after']}. Only the lead agent's conversations were run again; the other arms do not change with its code.", "",
                  "| measure | before | after | difference by case [95% interval] |", "|---|---:|---:|---|"]
        lines += [f"| {WORDS[d]} | {_num(ba['means'][d][0])} | {_num(ba['means'][d][1])} | {_interval(ba['paired'][d])} |" for d in DIMENSIONS]
        lines += [f"| Accepted | {c['accepted'][0]} of {c['cases'][0]} | {c['accepted'][1]} of {c['cases'][1]} | {_interval(ba['paired']['accepted'])} |",
                  f"| Turns | {_num(c['assistant_turns'][0], 1)} | {_num(c['assistant_turns'][1], 1)} | |",
                  f"| Seconds per conversation | {_num(c['seconds'][0], 0)} | {_num(c['seconds'][1], 0)} | |",
                  f"| Numbers nobody gave | {c['numbers_nobody_gave'][0]} | {c['numbers_nobody_gave'][1]} | |", "",
                  "| where code stepped in | before | after |", "|---|---:|---:|"]
        lines += [f"| {REASON_WORDS.get(r, r)} | {n[0]} | {n[1]} |" for r, n in ba["interventions"].items()] + [""]
    for j in data.get("judges") or []:
        lines += [f"## Two judges on the same conversations ({j['run']})", "",
                  f"`{j['judge_a']}` and `{j['judge_b']}` agree on accepted or not for {j['accepted_agree']} of {j['conversations']} conversations. "
                  f"Their 0 to 4 scores differ by {_num(j['mean_score_gap'])} on average; {round(100 * (j['within_one_point'] or 0))}% of scores are within one point.", ""]
    cross = data.get("cross_model") or {}
    if cross.get("suites"):
        lines += ["## Claude beside Qwen", "", f"From `{cross['report']}` (rebuild it with `python -m dclab_rnd.bench models`). Each cell is the mean and the number of cases it stands on. "
                  "\"thin\" marks a side with fewer than five cases: read it as a hint.", ""]
        for suite in cross["suites"]:
            lines += [f"### {suite['title']}", "", "| arm | measure | Claude | Qwen | better | |", "|---|---|---:|---:|---|---|"]
            lines += [f"| {CROSS_ARMS.get(r['arm'], r['arm'])} | {r['words']} | {_num(r['claude'])} (n {r['claude_n']}) | {_num(r['qwen'])} (n {r['qwen_n']}) | "
                      f"{ {'claude': 'Claude', 'qwen': 'Qwen', None: 'same'}[r['better']] } | {'thin' if r['thin'] else ''} |" for r in suite["rows"]]
            lines.append("")
    era = data["claude_era"]
    lines += ["## What the earlier exams on Claude showed (context)", ""]
    if era.get("recorded_dev"):
        r = era["recorded_dev"]
        lines += [f"Kept dev runs read again (HAB-000, {r['runs']} runs on Claude Sonnet 5.5): the lead agent lost the right kind of work to the model alone "
                  f"on {r['lost']} conversations and won it on {r['won']}. First reasons of the losses: "
                  + ", ".join(f"{REASON_WORDS.get(k, k)} ({v})" for k, v in (r.get("first_reason_counts") or {}).items()) + ".", ""]
    if era.get("sealed"):
        lines += ["| sealed exam | model | judge | arm | people | understanding (0 to 1) | option fit (1 to 5) | right kind of work reached | numbers nobody gave | s/turn |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in era["sealed"]:
            for arm, label in (("plain", "the model alone"), ("case", "lead agent")):
                s = row.get(arm) or {}
                if s.get("people"):
                    lines.append(f"| {row['run']} | {row['model']} | {row['judge']} | {label} | {s['people']} | {_num(s.get('understanding'))} | {_num(s.get('options_fit'))} | "
                                 f"{_num(s.get('route_reached'))} | {_num(s.get('safety_failures'))} | {_num(s.get('seconds_per_turn'), 1)} |")
        lines.append("")
    lines += ["## What weakens this benchmark", ""] + [f"- {x}" for x in data["limits"]] + [""]
    return "\n".join(lines)


CSS = """
:root{--bg:#faf9f5;--panel:#ffffff;--line:#e3dfd3;--text:#1f1e1b;--muted:#6b6a63;--ok:#2f7d4f;--warn:#b7791f;--bad:#b8402f;
--c0:#c96442;--c1:#3d6fb4;--c2:#5a8f5a;--c3:#8a63b3}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1b1a18;--panel:#262522;--line:#3a3833;--text:#ece9e1;--muted:#a09d93;--ok:#6fbf8b;--warn:#e0b060;--bad:#e8806f}}
:root[data-theme="dark"]{--bg:#1b1a18;--panel:#262522;--line:#3a3833;--text:#ece9e1;--muted:#a09d93;--ok:#6fbf8b;--warn:#e0b060;--bad:#e8806f}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:1.0625rem/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;padding:24px 16px 64px}
main{max-width:1180px;margin:0 auto}h1{font:400 2.4rem/1.15 ui-serif,Georgia,serif;margin:0 0 10px}h2{font-size:1.5rem;margin:36px 0 12px}h3{font-size:1.125rem;margin:0 0 10px}
.lede{font-size:1.1875rem;margin:0 0 8px}.muted{color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:16px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px;overflow-x:auto}.wide{grid-column:1/-1}
table{border-collapse:collapse;width:100%;font-size:1rem}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:220px}th{font-weight:600;color:var(--muted)}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;margin-right:8px;vertical-align:baseline}
.chip{display:inline-block;border:1px solid currentColor;border-radius:999px;padding:1px 10px;font-size:.95rem}.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.h4{background:color-mix(in srgb,var(--ok) 26%,transparent)}.h3{background:color-mix(in srgb,var(--ok) 13%,transparent)}.h2{background:color-mix(in srgb,var(--warn) 18%,transparent)}
.h1{background:color-mix(in srgb,var(--bad) 14%,transparent)}.h0{background:color-mix(in srgb,var(--bad) 26%,transparent)}
.bar{height:14px;border-radius:3px;background:var(--c0);display:inline-block;vertical-align:middle;margin-right:8px}
svg text{fill:var(--text);font-size:15px}svg .tick{fill:var(--muted);font-size:14px}ul{margin:8px 0 0 20px;padding:0}
"""
COLORS = {"case": "var(--c0)", "model_only": "var(--c1)", "open": "var(--c2)", "plain": "var(--c3)", "model_only_short": "var(--muted)",
          "blueprint_rules": "var(--muted)", "blueprint_pipeline": "var(--c3)", "blueprint_architect": "var(--c0)"}


def _swatch(arm: str) -> str:
    return f'<span class="sw" style="background:{COLORS[arm]}"></span>{html.escape(NAMES[arm])}'


def _heat(value: Any) -> str:
    return f'<td class="h{int(round(float(value)))}">{_num(value)}</td>' if value is not None else "<td>–</td>"


def _interval_chart(rows: list[tuple[str, Mapping[str, Any]]], color: str = "var(--text)") -> str:
    """Paired differences as interval lines against a dashed zero line (colored only when the whole interval is on one side)."""
    width, left, row, span = 560, 230, 30, 2.5
    def x(v: float) -> float:
        return left + (max(-span, min(span, v)) + span) / (2 * span) * (width - left - 20)
    out = [f'<svg viewBox="0 0 {width} {len(rows) * row + 34}" width="100%" role="img">']
    out.append(f'<line x1="{x(0)}" y1="4" x2="{x(0)}" y2="{len(rows) * row + 8}" stroke="var(--muted)" stroke-dasharray="4 4"/>')
    for i, (label, p) in enumerate(rows):
        y = 20 + i * row
        low, high = (p.get("interval") or [None, None])
        diff = p.get("difference")
        out.append(f'<text x="0" y="{y + 5}">{html.escape(label)}</text>')
        if diff is None:
            continue
        clear = low is not None and high is not None and (low > 0 or high < 0)
        stroke = ("var(--ok)" if diff > 0 else "var(--bad)") if clear else color
        if low is not None and high is not None:
            out.append(f'<line x1="{x(low)}" y1="{y}" x2="{x(high)}" y2="{y}" stroke="{stroke}" stroke-width="3" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(diff)}" cy="{y}" r="6" fill="{stroke}"/>')
    base = len(rows) * row + 26
    for tick in (-2, -1, 0, 1, 2):
        out.append(f'<text class="tick" x="{x(tick)}" y="{base}" text-anchor="middle">{tick:+d}</text>' if tick else f'<text class="tick" x="{x(0)}" y="{base}" text-anchor="middle">0</text>')
    return "".join(out) + "</svg>"


def page(data: Mapping[str, Any]) -> str:
    runs = data["runs"]
    parts = ["<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
             "<title>Harness Benchmark</title><style>", CSS, "</style></head><body><main>",
             "<h1>Does DCLab's harness help or hold back the model?</h1>"]
    am = data.get("all_models") or {"rows": [], "reasons": []}
    best = [(row["arms"]["model_only"]["planning"], row["arms"]["case"]["planning"]) for row in am["rows"]
            if "model_only" in row["arms"] and row["arms"]["model_only"]["planning"] is not None and row["arms"]["case"]["planning"] is not None]
    if best:
        parts.append(f'<p class="lede">On eight written cases and {len({row["model"] for row in am["rows"]})} models, a model alone makes a plan scored '
                     f'<b>{_num(min(a for a, _ in best))} to {_num(max(a for a, _ in best))}</b>; the same model inside current DCLab, '
                     f'<b>{_num(min(c for _, c in best))} to {_num(max(c for _, c in best))}</b> (0 to 4). Current DCLab wrote '
                     f'<b>{sum(row["arms"]["case"]["numbers_nobody_gave"] for row in am["rows"])}</b> numbers nobody gave; the minimal agents, '
                     f'{sum(row["arms"]["open"]["numbers_nobody_gave"] for row in am["rows"] if "open" in row["arms"])}.</p>')
    parts.append('<p class="muted">Eight cases, each once. One judge model. Read the differences as hints, not proof. Built from the result files; nothing was run to make this page.</p>')
    parts.append('<p>' + " &nbsp; ".join(_swatch(a) for a in ALL_ARMS) + "</p>")
    for gate_id in sorted({g["gate"] for g in data.get("gate") or []}, reverse=True):
        of = [g for g in data["gate"] if g["gate"] == gate_id]
        heads = "".join(f"<th>{html.escape(c['check'])}</th>" for c in of[0]["checks"])
        body = "".join(
            f"<tr><td>{html.escape(str(g['run']))}{' (' + html.escape(str(g['label'])) + ')' if g.get('label') else ''}{' · think first' if g.get('think_first') else ''}</td>"
            f"<td>{html.escape(str(g['model']))}</td>"
            f"<td><span class=\"chip {'ok' if g['passed'] else 'bad'}\">{'pass' if g['passed'] else 'fail'}</span></td>"
            + "".join(f"<td class=\"{'' if c['ok'] else 'bad' if c['read'] else 'warn'}\">{_num(c['value'])}{'' if c['read'] else ' (not read)'}</td>" for c in g["checks"]) + "</tr>"
            for g in of)
        parts.append(f'<h2>The gate <span class="muted">· {html.escape(gate_id)}, {html.escape(GATE_WORDS.get(gate_id, "fixed before its runs"))}; every check must hold</span></h2>'
                     f'<div class="grid"><div class="card wide"><table><tr><th>run</th><th>model</th><th>verdict</th>{heads}</tr>{body}</table>'
                     '<p class="muted">The lead agent minus the same model alone, by case (bar: at least -0.5). Every count must be zero.</p></div></div>')
    if am["rows"]:
        parts.append('<h2>All models side by side <span class="muted">· compare arms inside a row; rows under different judges are not a pairing</span></h2><div class="grid">')
        for arm in [a for a in ALL_ARMS if any(a in row["arms"] for row in am["rows"])]:
            body = "".join(f'<tr><td>{html.escape(_row_name(row))}</td>' + "".join(_heat(row["arms"][arm][d]) for d in DIMENSIONS)
                           + "".join(f'<td class="{"bad" if k == "numbers_nobody_gave" and row["arms"][arm][k] else ""}">{_counted(row["arms"][arm], k)}</td>' for k, _ in COUNTED) + "</tr>"
                           for row in am["rows"] if arm in row["arms"])
            parts.append(f'<div class="card wide"><h3>{_swatch(arm)}</h3><table><tr><th>model</th>' + "".join(f"<th>{WORDS[d]}</th>" for d in DIMENSIONS)
                         + "".join(f"<th>{w}</th>" for _, w in COUNTED) + "</tr>" + body + "</table></div>")
        parts.append('<div class="card wide"><h3>What the harness changes inside each model: current DCLab minus the minimal agent (by case, 95% interval)</h3><table><tr><th>model</th>'
                     + "".join(f"<th>{WORDS[d]}</th>" for d in (*DIMENSIONS, "accepted")) + "</tr>"
                     + "".join(f'<tr><td>{html.escape(_row_name(row))}</td>' + "".join(
                         (lambda p: f'<td class="{"bad" if (p.get("interval") or [0, 0])[1] is not None and (p.get("interval") or [0, 0])[1] < 0 else ""}">{html.escape(_interval(p))}</td>')(row["harness"][d])
                         if d in row["harness"] else "<td>–</td>" for d in (*DIMENSIONS, "accepted")) + "</tr>" for row in am["rows"] if row["harness"])
                     + '</table><p class="muted">Red: the whole interval is below zero, so DCLab is lower than the minimal agent on that model.</p></div>')
        parts.append('<div class="card wide"><h3>Where the lead agent\'s code stepped in, by model</h3><table><tr><th>what stopped or changed the model</th>'
                     + "".join(f"<th>{html.escape(_row_name(row))}</th>" for row in am["rows"]) + "</tr>"
                     + "".join(f"<tr><td>{html.escape(REASON_WORDS.get(r, r))}</td>" + "".join(f"<td>{row['interventions'].get(r, 0)}</td>" for row in am["rows"]) + "</tr>" for r in am["reasons"])
                     + "<tr><td><b>all</b></td>" + "".join(f"<td><b>{row['interventions_total']}</b></td>" for row in am["rows"]) + "</tr></table></div>")
        parts.append("</div>")
    for run in runs:
        parts.append(f'<h2>{html.escape(run["id"])}: every arm on {html.escape(str(run["model"]))} <span class="muted">· judge {html.escape(str(run["judge"]))}</span></h2><div class="grid">')
        rows = "".join(f'<tr><td>{_swatch(arm)}</td>' + "".join(_heat(s[d]) for d in DIMENSIONS)
                       + f'<td><span class="chip {"ok" if s["accepted"] * 2 > s["cases"] else "warn"}">{s["accepted"]} of {s["cases"]}</span></td>'
                       f'<td class="{"bad" if s["numbers_nobody_gave"] else ""}">{s["numbers_nobody_gave"]}</td><td>{_num(s["assistant_turns"], 1)}</td><td>{_num(s["seconds"], 0)}</td><td>{_num(s["spent_eur"])}</td></tr>'
                       for arm, s in run["arms"].items())
        parts.append('<div class="card wide"><h3>Scores by arm (mean, 0 to 4)</h3><table><tr><th>arm</th>' + "".join(f"<th>{WORDS[d]}</th>" for d in DIMENSIONS)
                     + "<th>Accepted</th><th>Numbers nobody gave</th><th>Turns</th><th>Seconds</th><th>€</th></tr>" + rows + "</table>")
        short = {NAMES[a]: s["empty_or_short_answers"] for a, s in run["arms"].items() if s["empty_or_short_answers"] and run.get("set") not in HELD_OUT}
        if short:
            parts.append('<p class="muted">Empty or very short answers, scored as given: ' + html.escape("; ".join(f"{k}: {', '.join(v)}" for k, v in short.items())) + ".</p>")
        parts.append("</div>")
        pair = (run.get("paired") or {}).get("case_minus_open") or {}
        parts.append('<div class="card"><h3>What the harness adds: current DCLab minus the minimal agent</h3>'
                     + _interval_chart([(WORDS[d], pair[d]) for d in (*DIMENSIONS, "accepted") if d in pair])
                     + '<p class="muted">A dot right of the dashed line: DCLab is higher. A grey line crossing zero: no clear difference.</p></div>')
        total = sum(run["interventions"].values()) or 1
        top = max(run["interventions"].values() or [1])
        parts.append('<div class="card"><h3>Where the lead agent\'s code stepped in</h3><table><tr><th>what stopped or changed the model</th><th>times</th></tr>'
                     + "".join(f'<tr><td>{html.escape(REASON_WORDS.get(r, r))}</td><td><span class="bar" style="width:{max(4, int(120 * n / top))}px"></span>{n}</td></tr>'
                               for r, n in run["interventions"].items())
                     + f'</table><p class="muted">{total} interventions in 8 conversations.</p></div>')
        cases = sorted({cid for s in run["arms"].values() for cid in s.get("by_case") or {}})
        for dim in ("understanding", "planning") if run.get("set") not in HELD_OUT else ():  # a held-out set shows its means only
            parts.append(f'<div class="card"><h3>{WORDS[dim]}, case by case</h3><table><tr><th>arm</th>' + "".join(f"<th>{c[-2:]}</th>" for c in cases) + "</tr>"
                         + "".join(f"<tr><td>{_swatch(arm)}</td>" + "".join(_heat((s["by_case"].get(c) or {}).get(dim)) for c in cases) + "</tr>" for arm, s in run["arms"].items())
                         + "</table></div>")
        parts.append("</div>")
    for run in data.get("not_judged_yet") or []:
        parts.append(f'<h2>{html.escape(run["id"])}: the same cases on {html.escape(str(run["model"]))} <span class="muted">· played, not judged yet'
                     + (f' · stopped: {html.escape(str(run["incomplete"]))}' if run.get("incomplete") else "") + '</span></h2><div class="grid">')
        parts.append('<div class="card"><h3>What code can count without a judge</h3><table><tr><th>arm</th><th>Conversations</th><th>Turns</th><th>Seconds</th><th>Numbers nobody gave</th><th>€</th></tr>'
                     + "".join(f'<tr><td>{_swatch(arm)}</td><td>{s["answered"]}</td><td>{_num(s["assistant_turns"], 1)}</td><td>{_num(s["seconds"], 0)}</td>'
                               f'<td class="{"bad" if s["numbers_nobody_gave"] else ""}">{s["numbers_nobody_gave"]}</td><td>{_num(s["spent_eur"])}</td></tr>' for arm, s in run["arms"].items())
                     + "</table></div>")
        top = max(run["interventions"].values() or [1])
        parts.append('<div class="card"><h3>Where the lead agent\'s code stepped in</h3><table><tr><th>what stopped or changed the model</th><th>times</th></tr>'
                     + "".join(f'<tr><td>{html.escape(REASON_WORDS.get(r, r))}</td><td><span class="bar" style="width:{max(4, int(120 * n / top))}px"></span>{n}</td></tr>' for r, n in run["interventions"].items())
                     + "</table></div></div>")
    for pair in data["across_models"]:
        parts.append(f'<h2>The same eight cases on two models <span class="muted">· {html.escape(str(pair["b_model"]))} minus {html.escape(str(pair["a_model"]))}</span></h2><div class="grid">')
        for arm, p in pair["arms"].items():
            parts.append(f'<div class="card"><h3>{_swatch(arm)}</h3>' + _interval_chart([(WORDS[d], p[d]) for d in (*DIMENSIONS, "accepted")], COLORS[arm]) + "</div>")
        parts.append('</div><p class="muted">A dot right of the dashed line: the second model is higher on that arm.</p>')
    for ba in data.get("before_after") or []:
        c = ba["counted"]
        rows = "".join(f"<tr><td>{WORDS[d]}</td>{_heat(ba['means'][d][0])}{_heat(ba['means'][d][1])}<td>{html.escape(_interval(ba['paired'][d]))}</td></tr>" for d in DIMENSIONS)
        rows += (f"<tr><td>Accepted</td><td>{c['accepted'][0]} of {c['cases'][0]}</td><td>{c['accepted'][1]} of {c['cases'][1]}</td><td>{html.escape(_interval(ba['paired']['accepted']))}</td></tr>"
                 f"<tr><td>Turns</td><td>{_num(c['assistant_turns'][0], 1)}</td><td>{_num(c['assistant_turns'][1], 1)}</td><td></td></tr>"
                 f"<tr><td>Seconds per conversation</td><td>{_num(c['seconds'][0], 0)}</td><td>{_num(c['seconds'][1], 0)}</td><td></td></tr>"
                 f"<tr><td>Numbers nobody gave</td><td>{c['numbers_nobody_gave'][0]}</td><td>{c['numbers_nobody_gave'][1]}</td><td></td></tr>")
        stepped = "".join(f"<tr><td>{html.escape(REASON_WORDS.get(r, r))}</td><td>{n[0]}</td><td class=\"{'ok' if n[1] < n[0] else 'bad' if n[1] > n[0] else ''}\">{n[1]}</td></tr>"
                          for r, n in ba["interventions"].items())
        parts.append(f'<h2>Current DCLab before and after the fixes <span class="muted">· {html.escape(str(ba["model"]))} · judge {html.escape(str(ba["judge"]))}</span></h2><div class="grid">'
                     f'<div class="card"><h3>Scores (0 to 4)</h3><table><tr><th>measure</th><th>Before</th><th>After</th><th>Difference by case</th></tr>{rows}</table>'
                     f'<p class="muted">Before: {html.escape(str(ba["before"]))}. After: {html.escape(str(ba["after"]))} ({html.escape(str(ba["label"]))}).</p></div>'
                     f'<div class="card"><h3>Where code stepped in</h3><table><tr><th>what stopped or changed the model</th><th>Before</th><th>After</th></tr>{stepped}</table></div></div>')
    for j in data.get("judges") or []:
        parts.append(f'<h2>Do two judges agree? <span class="muted">· {html.escape(str(j["run"]))}</span></h2><div class="card"><p class="lede"><b>{j["accepted_agree"]} of {j["conversations"]}</b> '
                     f'conversations get the same "accepted or not" from {html.escape(str(j["judge_a"]))} and {html.escape(str(j["judge_b"]))}. '
                     f'Their scores differ by <b>{_num(j["mean_score_gap"])}</b> points on average; {round(100 * (j["within_one_point"] or 0))}% are within one point.</p></div>')
    cross = data.get("cross_model") or {}
    if cross.get("suites"):
        parts.append('<h2>Claude beside Qwen <span class="muted">· the same tests on both models</span></h2>'
                     '<p class="muted">Each cell is the mean and, in brackets, how many cases it stands on. "thin" marks a side with fewer than five cases: a hint, not a result. '
                     "The better side of each row is in green.</p><div class=\"grid\">")
        for suite in cross["suites"]:
            body = ""
            for r in suite["rows"]:
                def cell(side: str) -> str:
                    cls = "ok" if r["better"] == side else ""
                    return f'<td class="{cls}">{_num(r[side])} <span class="muted">({r[side + "_n"]})</span></td>'
                key = r["arm"] if r["arm"] in COLORS else "case"
                body += (f'<tr><td><span class="sw" style="background:{COLORS[key]}"></span>{html.escape(CROSS_ARMS.get(r["arm"], r["arm"]))}</td><td style="text-align:left">{html.escape(r["words"])}</td>'
                         + cell("claude") + cell("qwen") + "<td>" + (THIN if r["thin"] else "") + "</td></tr>")
            parts.append(f'<div class="card wide"><h3>{html.escape(suite["title"])}</h3><table><tr><th>arm</th><th style="text-align:left">Measure</th><th>Claude</th><th>Qwen</th><th></th></tr>{body}</table></div>')
        parts.append("</div>")
    era = data["claude_era"]
    parts.append('<h2>What the earlier exams on Claude showed <span class="muted">· context, other people and another judge</span></h2><div class="grid">')
    if era.get("recorded_dev"):
        r = era["recorded_dev"]
        parts.append(f'<div class="card"><h3>Kept practice runs, read again</h3><p>{r["runs"]} runs on Claude Sonnet 5.5. The lead agent lost the right kind of work to the model alone on '
                     f'<b>{r["lost"]}</b> conversations and won it on <b>{r["won"]}</b>.</p><table><tr><th>first reason of a loss</th><th>times</th></tr>'
                     + "".join(f"<tr><td>{html.escape(REASON_WORDS.get(k, k.replace('_', ' ')))}</td><td>{v}</td></tr>" for k, v in (r.get("first_reason_counts") or {}).items()) + "</table></div>")
    if era.get("sealed"):
        rows = ""
        for row in era["sealed"]:
            for arm, label, key in (("plain", "The model alone", "model_only"), ("case", "Lead agent", "case")):
                s = row.get(arm) or {}
                if s.get("people"):
                    rows += (f'<tr><td><b>{html.escape(str(row["run"]))}</b> <span class="sw" style="background:{COLORS[key]};margin-left:8px"></span>{label}</td><td>{s["people"]}</td>'
                             f'<td>{_num(s.get("understanding"))}</td><td>{_num(s.get("options_fit"))}</td><td>{_num(s.get("route_reached"))}</td>'
                             f'<td class="{"bad" if s.get("safety_failures") else ""}">{_num(s.get("safety_failures"))}</td><td>{_num(s.get("seconds_per_turn"), 1)}</td></tr>')
        parts.append('<div class="card wide"><h3>Sealed exams on Claude Sonnet 5.5 (judge: Claude Opus 5.5)</h3><table><tr><th>exam and arm</th><th>People</th><th>Understanding (0 to 1)</th>'
                     '<th>Option fit (1 to 5)</th><th>Right kind of work reached</th><th>Numbers nobody gave</th><th>Seconds per turn</th></tr>' + rows + "</table></div>")
    parts.append('</div><h2>What weakens this benchmark</h2><div class="card"><ul>' + "".join(f"<li>{html.escape(x)}</li>" for x in data["limits"]) + "</ul></div>")
    parts.append("</main></body></html>")
    return "".join(parts)


def main() -> int:
    data = build()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    (CAMPAIGN / "BENCHMARK.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (CAMPAIGN / "BENCHMARK.md").write_text(markdown(data), encoding="utf-8")
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    PAGE.write_text(page(data), encoding="utf-8")
    print(f"{len(data['runs'])} judged runs, {len(data['across_models'])} model pairs; wrote {(CAMPAIGN / 'BENCHMARK.md').relative_to(ROOT)} and {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
