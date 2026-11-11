"""The conversation exam's benchmark: every run's numbers by arm in one table, the judge against the people who labelled,
and how much of the exams' model work was kept with its thinking summary.

``python -m dclab_rnd.agent_eval.conversation_exam.benchmark`` rebuilds ``BENCHMARK.md`` and ``BENCHMARK.json`` beside the
campaign's results from the result files and the kept exchanges; nothing is run and nothing is spent. A sealed run's
exchanges are counted from its vault by shape only (calls, thinking kept): never a word of them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from . import CAMPAIGN

ARMS = ("home", "plain", "safe", "case", "case_chip")  # safe: Claude alone with DCLab's safety layer (its rows were missing, 2026-11-01)
NAMES = {"home": "Home today", "plain": "Claude alone", "safe": "Claude alone + DCLab safety", "case": "lead agent",
         "case_chip": "lead agent with the person's chip"}
EXAM_PURPOSES = ("case_agent", "case_specialist", "exam_plain", "exam_person", "exam_judge", "home_agent", "home_intent",
                 "home_understand", "home_query")


def _name(arm: str, model: str = "") -> str:
    """The arm's name; "Claude alone" is "the model alone" when another model answered (OpenRouter, 2026-11-11)."""
    name = NAMES[arm]
    return name if not model or model.startswith("claude") or model == "unknown" else name.replace("Claude", "the model")


def _trap_mean(traps: Mapping[str, Any]) -> Optional[float]:
    vals = [float(v) for v in (traps or {}).values() if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def runs(results: Path) -> list[dict[str, Any]]:
    """One row per run and arm, from each run's own result file (the label files are read by ``labels_rows``)."""
    rows = []
    docs = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(results.glob("CEX-*.json"))]
    # a joined result (a run finished by later runs) holds those runs' conversations: each is marked, so no person is read
    # twice (the review of 2026-11-01)
    inside = {r: doc.get("id") for _, doc in docs for r in ((doc.get("joined") or {}).get("runs") or []) if r != doc.get("id")}
    for path, doc in docs:
        summary = doc.get("summary")
        if not isinstance(summary, dict) or not summary:
            continue
        if not any((s or {}).get("people") for s in summary.values()):
            continue  # a run cut before any person was scored (its conversations are kept on disk and in the store)
        rid = doc.get("id") or path.stem.split("_")[0]
        label = (f"{rid} (joined: {', '.join(r for r in doc['joined']['runs'] if r != rid)})" if doc.get("joined")
                 else f"{rid} (in {inside[rid]})" if rid in inside else rid)
        from dclab_rnd.bench.models import judge_of, model_of

        judge = judge_of(doc)
        for arm in ARMS:
            s = summary.get(arm)
            if not s:
                continue
            judged, listed = s.get("judged"), s.get("people")
            cut = judged is not None and listed is not None and judged < listed  # CEX-025: five of ten conversations failed on an empty account
            rows.append({"run": label, "split": doc.get("split"), "at": str(doc.get("at") or "")[:10],
                         "arm": arm, "model": model_of(doc, arm)[0], "judge": judge, "people": f"{judged} of {listed}" if cut else listed, "understanding": s.get("understanding"), "options_fit": s.get("options_fit"),
                         "route_reached": s.get("route_reached", s.get("route_accuracy")), "already_answered": s.get("already_answered"),
                         "safety_failures": s.get("safety_failures"), "traps": _trap_mean(s.get("traps") or {}), "silent": s.get("silent"),
                         "seconds_per_turn": s.get("seconds_per_turn"), "spent_eur": s.get("spent_eur"),
                         "verdict": (f"incomplete: stopped ({doc['incomplete']})" if doc.get("incomplete") else "incomplete: the others never ran" if cut else None) or ((doc.get("verdict") or {}).get("status") if arm == "case" else None)})
    return rows


def labels_rows(results: Path) -> list[dict[str, Any]]:
    """The judge against each labeller, from the files written after the labels."""
    out = []
    for path in sorted(results.glob("CEX-*_owner_labels.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        out.append({"run": doc.get("id"), "pair": "judge and the owner", "agreement": doc.get("judge_owner_agreement"),
                    "n": (doc.get("owner_labels") or {}).get("n"), "verdict": (doc.get("verdict") or {}).get("status")})
    for path in sorted(results.glob("CEX-*_second_labeller.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        a = doc.get("agreement") or {}
        out += [{"run": doc.get("id"), "pair": "judge and a second Opus labeller", "agreement": a.get("judge_second"), "n": 10, "verdict": None},
                {"run": doc.get("id"), "pair": "the owner and a second Opus labeller", "agreement": a.get("owner_second"), "n": 10, "verdict": None}]
    return out


def _lines(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                yield json.loads(line)
            except ValueError:
                continue


def capture(workspace: Path) -> dict[str, Any]:
    """The exams' model calls and how many kept a thinking summary: the workspace log (practice runs, sealed runs by shape)
    and each sealed vault (counted by shape here, never read for its words)."""
    def count(rows: Iterable[Mapping[str, Any]], whole_only: bool) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for d in rows:
            p = str(d.get("purpose") or "")
            if p not in EXAM_PURPOSES or (whole_only and not d.get("whole")):
                continue
            th = d.get("thinking") or d.get("thinking_redacted") or []
            c = out.setdefault(p, {"calls": 0, "whole": 0, "thinking": 0})
            c["calls"] += 1
            c["whole"] += int(bool(d.get("whole")))
            c["thinking"] += int(any(isinstance(b, dict) and str(b.get("text") or "").strip() for b in th))
        return out

    practice = count(_lines(sorted((workspace / "model_exchanges").rglob("*.jsonl"))), whole_only=False)
    vaults = {v.name: count(_lines(sorted((v / "model_exchanges").rglob("*.jsonl"))), whole_only=False)
              for v in sorted((workspace / "exam_vault").glob("*")) if (v / "model_exchanges").is_dir()}
    return {"workspace": practice, "vaults": vaults,
            "rule": "thinking on (adaptive) with display 'summarized' for every purpose; the summary kept whole; never asked for in the reply"}


def _f(v: Any, digits: int = 2) -> str:
    if v is None:
        return "–"
    return f"{v:.{digits}f}" if isinstance(v, float) else str(v)


def render(doc: Mapping[str, Any]) -> str:
    out = ["# Conversation exam: the benchmark", "",
           "Rebuilt by `python -m dclab_rnd.agent_eval.conversation_exam.benchmark` from the result files; do not edit by hand.",
           "Understanding is the judge's (0 to 1); option fit 1 to 5; route reached is the share of people whose right route was",
           "offered or taken; traps the mean of the planted traps' rates; seconds per turn the assistant's reply time.", "",
           "The model is the one that answered in the arm; the judge's model is beside it (the Claude-era exams were judged by Opus,",
           "the exams on OpenRouter by the same model that answered: a gap between them mixes the student and the judge).", "",
           "| run | split | student | model | judge | people | understanding | option fit | route reached | already answered | safety failures | traps | s/turn | € | verdict |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in doc["runs"]:
        out.append(f"| {r['run']} | {r['split']} | {_name(r['arm'], r.get('model') or '')} | {r.get('model') or '–'} | {r.get('judge') or '–'} | {_f(r['people'])} | {_f(r['understanding'])} | {_f(r['options_fit'])} | "
                   f"{_f(r['route_reached'])} | {_f(r['already_answered'])} | {_f(r['safety_failures'])} | {_f(r['traps'])} | "
                   f"{_f(r['seconds_per_turn'], 1)} | {_f(r['spent_eur'])} | {r['verdict'] or ''} |")
    out += ["", "## The judge against the people who labelled", "", "| run | pair | agreement | papers | verdict |", "|---|---|---|---|---|"]
    for r in doc["labels"]:
        out.append(f"| {r['run']} | {r['pair']} | {_f(r['agreement'])} | {_f(r['n'])} | {r['verdict'] or ''} |")
    out += ["", "## Kept with the thinking summary", "", doc["capture"]["rule"] + ".", "",
            "| where | purpose | calls | kept whole | with thinking |", "|---|---|---|---|---|"]
    for where, counts in [("workspace log (every use of the purpose)", doc["capture"]["workspace"])] + [(f"vault {k}", v) for k, v in doc["capture"]["vaults"].items()]:
        for p, c in sorted(counts.items()):
            out.append(f"| {where} | {p} | {c['calls']} | {c['whole']} | {c['thinking']} |")
    out += ["", "In the workspace log a sealed run's calls are kept by shape only (whole only in its vault), and before 2026-10-31",
            "a practice call that saw table cells kept a redacted thinking without its words; a call can also return no thinking",
            "when the model chose not to think (adaptive).", ""]
    return "\n".join(out)


def build(results: Optional[Path] = None, workspace: Optional[Path] = None) -> dict[str, Any]:
    from dclab_rnd.settings import Settings

    results = Path(results or CAMPAIGN / "results")
    workspace = Path(workspace or Settings().agent_home)
    rows, labelled = runs(results), labels_rows(results)
    final = {r["run"]: r["verdict"] for r in labelled if r["pair"] == "judge and the owner" and r["verdict"]}
    for r in rows:  # the verdict after the owner's labels replaces the run's own "pending"
        if r["verdict"] and r["run"] in final:
            r["verdict"] = final[r["run"]]
    return {"runs": rows, "labels": labelled, "capture": capture(workspace)}


def main(argv: Optional[list[str]] = None) -> int:
    doc = build()
    (CAMPAIGN / "BENCHMARK.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (CAMPAIGN / "BENCHMARK.md").write_text(render(doc), encoding="utf-8")
    print(f"{len(doc['runs'])} rows, {len(doc['labels'])} label rows; wrote {CAMPAIGN / 'BENCHMARK.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
