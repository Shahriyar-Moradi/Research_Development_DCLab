"""The acceptance suite of conversations (package 20.10): a written person per family, from the first sentence to the
holdout number.

Each journey (``journeys_dev.json``, ``journeys_sealed.json``; the sealed file's SHA-256 pinned in ``sealed.sha256``)
is a person: the problem in their words, their answers keyed by the field a question asks, and the replica of a problem
kind (``dclab_core.replicas`` through ``dclab_rnd.kinds``) as the files they bring. :func:`play` runs it as the product
does, with no model: a draft per file the person brings (the cases apart from their table, as on the owner's morning),
the coordinator (21.2) walking the draft to a built project with the person answering, the tables the kind reads
attached by name, and every stage run by the engine. It is scored on four things, each against the replica's planted
truth:

- **card**: the problem card's family is the person's (one of those the journey accepts), with the labels where given;
- **workflow**: the draft's solution workflow validates and no move of the project's graph was refused;
- **pipeline**: every step behind the project's numbers has its run, none stale (20.8);
- **holdout**: the kind's acceptance decisions (``kinds.<kind>.accept``: the leak kept out, the right split, the
  holdout number above its planted floor and below its planted ceiling) all hold on the finished run.

A family the engine does not run yet is expected to stop at the build with its reason said (``"built": false``), and
is scored on the card and that refusal. The gate (``GATE``, fixed before any run) is judged on the sealed journeys,
run once; nothing is tuned on them. Since 16.15 B the engine runs segmentation, uplift and ranking: the six sealed
journeys of those families still expect the refusal, as they were sealed (JRN-002 stands); a later play of the sealed
set would score them wrong for that reason alone. A second sealed set (``journeys_sealed_v2.json``, ``GATE_V2``: the same
bar on the replicas' seed 2) was written by an agent that read only the replicas' columns and the format, and frozen
before any rule changed for v1's failures; it ran once (JRN-004: 24 of 30, a fail). A third (``journeys_sealed_v3.json``,
``GATE_V3``, seed 3), written the same way, was frozen before the second set's misses were fixed (``--split sealed_v3``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "journeys_v1" / "results"
SEALED = {"sealed": "sealed.sha256", "sealed_v2": "sealed_v2.sha256", "sealed_v3": "sealed_v3.sha256", "sealed_v4": "sealed_v4.sha256",
          "sealed_v5": "sealed_v5.sha256", "sealed_v6": "sealed_v6.sha256", "sealed_v7": "sealed_v7.sha256"}
GATE = {
    "version": "journeys_gate_v1",
    "split": "sealed",
    "journey": "right on the card, the workflow, the pipeline and the holdout (or, for a family the engine does not run, "
               "on the card and a build refused with its reason)",
    "suite": "at least 85% of the sealed journeys right, the lower end of the 95% Wilson interval at least 0.6, every family "
             "with a journey, and no move the graph's validator refused in any journey",
    "seed": 1,  # the replicas' seed for the sealed run: not the tables the dev failures were fixed on (seed 0)
}
# 2026-10-15: a second sealed set, written by an agent that read only the replicas' columns and the journeys' format (never
# the rules), frozen and committed before any rule was changed for v1's failures; v1's sealed set is dev material from then
GATE_V2 = {**GATE, "version": "journeys_gate_v2", "split": "sealed_v2",
           "seed": 2,  # neither the dev tables (0) nor v1's sealed ones (1)
           "author": "an agent that never read the rules, only the replicas' columns and the format"}
# 2026-10-15: the second set failed once (JRN-004, 24 of 30) and became dev material; a third, by another agent under the
# same restrictions, frozen and committed before any rule changed for the second set's misses
GATE_V3 = {**GATE_V2, "version": "journeys_gate_v3", "split": "sealed_v3", "seed": 3}
# 2026-10-15: the third failed once (JRN-005, 25 of 30); a fourth, written the same way, frozen before the conversation reader
# (a model) was measured: it judges the reader against the rules on the same people, both arms run once
GATE_V4 = {**GATE_V2, "version": "journeys_gate_v4", "split": "sealed_v4", "seed": 4,
           # fixed before the run (2026-10-15): the reader serves the product only when its arm passes the suite bar on its own
           # and is right on at least as many of these people as the rules arm, with every reader request answered
           "reader": "the reader arm passes the suite bar alone, is right on at least as many people as the rules arm on the same "
                     "people, and no reader request failed (a run with a failed request is not stored)"}
# 2026-10-18: a fifth, written the same way (an agent that read only the replicas' tables and the format), frozen before the
# team (28.7) was measured: it judges the team against the rules and the single agent on the same people, the three arms once
GATE_V5 = {**GATE_V2, "version": "journeys_gate_v5", "split": "sealed_v5", "seed": 5,
           "team": "the team passes TEAM_GATE on these people: at least the rules on every part, a card right more often than the single "
                   "agent's, no request failed, no move refused, no holdout read twice, within the cap a journey"}
# 28.7, again: the fifth set judged the team once (it failed); a sixth, written by a fresh agent from the replicas' tables only and
# pinned before any run, judges the team as it stands after 28.4 and 28.6 (with any pattern that passed), under the same gate
GATE_V6 = {**GATE_V5, "version": "journeys_gate_v6", "split": "sealed_v6", "seed": 6}
# 2026-10-28: every set before this one is spent (v1 to v4 on 20.10, v5 and v6 on the team). A seventh, written by a fresh
# agent that read only a pack of the replicas' tables and the format (never the rules or an earlier set), two people a kind,
# pinned before any run: it judges the product's walk as it stands (the rules, the loop) under the suite bar of v2, on its
# own seed. Its one run spends it for every arm; the team, when asked for in that run (--team-arms), under TEAM_GATE too.
GATE_V7 = {**GATE_V2, "version": "journeys_gate_v7", "split": "sealed_v7", "seed": 7,
           "author": "a fresh agent that read only the replicas' tables and the format, never the rules or an earlier set",
           "arms": "the rules arm (the loop) is judged by the suite bar; any other arm runs in the same one run, never after it"}
GATES = {"sealed": GATE, "sealed_v2": GATE_V2, "sealed_v3": GATE_V3, "sealed_v4": GATE_V4, "sealed_v5": GATE_V5, "sealed_v6": GATE_V6,
         "sealed_v7": GATE_V7}
# 28.7: the team (the walk's graph, the conversation reader and the understanding critic with a model) beside the rules and
# the single agent (the loop with the same reader), fixed before any run; on a sealed set "beats" is strict
TEAM_GATE = {"version": "team_gate_v1",
             "rule": "the team is right on at least as many journeys as the rules on each of the card, the workflow, the pipeline and the "
                     "holdout; its card is right at least as often as the single agent's on dev, and more often on a sealed set; no "
                     "request failed, no move refused, no holdout read twice; the team arm costs at most cap_per_journey_eur a journey",
             "cap_per_journey_eur": 0.06}
# 28.6: the team's patterns, each against its baseline in the same run, on the same people and seed, fixed before any run. The
# material: the dev people and the fifth sealed set, spent in 28.7 and read as dev material (its failures were read while the
# patterns were designed): dev evidence, never a sealed verdict
PATTERN_MATERIAL = (("dev", 0), ("sealed_v5", 5))
SECOND_OPINION_GATE = {"version": "second_opinion_gate_v1", "target": "card", "baseline": "single", "cost_ratio": 1.5, "extra_questions": 1.0,
                       "rule": "against the single agent (the loop with the conversation reader) on the same 45 people: more cards gained where "
                               "the judge asked and the person chose than cards lost anywhere (a gain the judge did not make is the reader's "
                               "noise, never the pattern's), no fewer right on the workflow, the pipeline or the holdout, at most one more "
                               "question a person on average, nobody asked a field twice, no request failed, no journey broke, no move "
                               "refused, the arm's cost at most cost_ratio times the single agent's (a priced baseline)"}
DEBATE_GATE = {"version": "debate_gate_v1", "target": "holdout", "baseline": "designer", "cost_ratio": 1.5, "extra_questions": None,
               "rule": "against the single designer (the solution planner alone, its answer shared by both arms) on the same 45 people, "
                       "the conversation by the rules: more holdouts gained where the debate changed the plan than holdouts lost anywhere, "
                       "no fewer right on the card, the workflow or the pipeline, no solution refused, nobody asked a field twice who "
                       "was not in the baseline (the debate never touches the conversation, which is the rules' in both arms: revised "
                       "before the rerun, the void first run showing six people asked again by the rules alike in both), no request "
                       "failed, no journey broke, no move refused, the arm's cost at most cost_ratio times the single designer arm's "
                       "(a priced baseline)"}
PATTERN_GATES = {"second_opinion": SECOND_OPINION_GATE, "debate": DEBATE_GATE}
PARTS = ("card", "workflow", "pipeline", "holdout")
REFUSED = "family not built yet"  # dclab_core.coordinator.FAMILY_NOT_RUN: the build's rule for a family the engine does not run


def _attributable(name: str, row: dict[str, Any]) -> bool:
    """The pattern acted on this person: the judge asked and the person chose (the second opinion), or the debate changed
    the single designer's plan (the debate)."""
    if name == "second_opinion":
        made = row.get("second_opinion") or {}
        return int(made.get("chose") or 0) > 0  # the person answered and the answer chose (the review: asked is not chose)
    return bool((row.get("debate") or {}).get("changed"))


def _broke(row: dict[str, Any]) -> bool:
    return any(str(w).startswith("broke:") for w in row.get("why") or [])


def pattern_verdict(name: str, baseline: list[dict[str, Any]], arm: list[dict[str, Any]], spent: dict[str, float], failures: int) -> dict[str, Any]:
    """A pattern's arm against its baseline under the pattern's gate (``PATTERN_GATES``): the parts, the gains the pattern
    made against every loss, the questions, anyone asked a field twice, broken journeys, the refusals, the cost; ``passed``
    only when every condition holds (the review of 28.6: a gain the pattern did not make, a broken baseline journey, a
    subset or an unpriced run must never pass)."""
    gate = PATTERN_GATES[name]
    if [r["id"] for r in baseline] != [r["id"] for r in arm]:
        raise ValueError("the pattern and its baseline must be played on the same people, in the same order")
    parts = {side: {p: sum(1 for r in rows if r["parts"].get(p) is True) for p in PARTS} for side, rows in (("baseline", baseline), ("arm", arm))}
    target = gate["target"]
    gained = sorted(r["id"] for r, b in zip(arm, baseline) if r["parts"].get(target) is True and b["parts"].get(target) is not True)
    made = sorted(r["id"] for r, b in zip(arm, baseline) if r["id"] in gained and _attributable(name, r))
    lost = sorted(r["id"] for r, b in zip(arm, baseline) if r["parts"].get(target) is not True and b["parts"].get(target) is True)
    no_worse = all(parts["arm"][p] >= parts["baseline"][p] for p in PARTS if p != target)
    questions = {side: round(sum(int(r.get("questions") or 0) for r in rows) / max(1, len(rows)), 3) for side, rows in (("baseline", baseline), ("arm", arm))}
    twice = {side: sorted(r["id"] for r in rows if len(f := [str(a).split("→")[-1] for a in r.get("asked") or []]) != len(set(f)))
             for side, rows in (("baseline", baseline), ("arm", arm))}
    broke = sorted({r["id"] for r in (*baseline, *arm) if _broke(r)})
    refused = sum(int(r.get("refused_moves") or 0) + int(r.get("refused_plans") or 0) for r in arm)
    priced = spent.get("baseline", 0.0) > 0
    cost_ok = priced and spent.get("arm", 0.0) <= gate["cost_ratio"] * spent["baseline"] + 1e-9
    extra = gate.get("extra_questions")
    questions_ok = extra is None or questions["arm"] - questions["baseline"] <= extra
    twice_ok = not twice["arm"] if name == "second_opinion" else set(twice["arm"]) <= set(twice["baseline"])  # the debate: the conversation is the baseline's
    passed = bool(len(made) > len(lost) and no_worse and questions_ok and twice_ok and not broke and refused == 0 and failures == 0 and cost_ok)
    return {"gate": gate, "parts": parts, "questions": questions, "asked_twice": twice, "broke": broke, "refused": refused, "failures": failures,
            "spent_eur": spent, "priced": priced, "cost_ok": cost_ok, "gained": gained, "gained_by_the_pattern": made, "lost": lost, "passed": passed}


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(split: str) -> list[dict[str, Any]]:
    path = HERE / f"journeys_{split}.json"
    if split in SEALED and fingerprint(path) != (HERE / SEALED[split]).read_text(encoding="utf-8").strip():
        raise ValueError("the sealed journeys changed since they were sealed: a sealed set is never edited")
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(right: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = right / n
    centre, half = p + z * z / (2 * n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((centre - half) / (1 + z * z / n), 3), round((centre + half) / (1 + z * z / n), 3))


class _Person:
    """The written person: answers a question by the field it asks (or, 26.5, by what its words ask when its field is not one
    they hold), approves every gate, and keeps what was asked."""

    UNSURE = "I don't know."  # S8 B3: what a person who does not know says (dclab_core.conversation reads it as not said)

    def __init__(self, answers: dict[str, str], dont_know: bool = False):
        from dclab_rnd.studio import team

        self.asked: list[str] = []
        self.person = team.Person(answer=self._answer, approve=lambda gate, summary: True, who="written person")
        self.answers = answers
        # S8 B3: a person written with answers left out on purpose says "I don't know" to a question they cannot answer,
        # as people do, instead of falling silent (the older journeys keep their silence: their results stay comparable)
        self.dont_know = bool(dont_know)

    SAME = {"outcome": "target", "target_check": "target", "labels_check": "labels"}  # one question asked under two names
    FIELDS = ("target", "prediction_moment", "action", "entity", "labels", "budget", "costs", "data_plan", "family")  # the questions a person holds answers for
    # 26.5: what a question asks, read from its words, when its field is not one the person holds (a model asks in its own
    # words, under no field or another one); the first field whose words match and that the person can answer
    MEANING = (
        # the card's contradiction about the labels says "when nothing is labelled": read before the moment's "when" (live, 2026-10-19)
        ("labels", r"\blearn from the known cases\b|\bsay nothing is labelled\b"),
        ("prediction_moment", r"\b(when\b|moment|at what point|known (at|then|when)|before .{0,40}(made|scored|known)|time of (the )?prediction)"),
        ("budget", r"\b(how many|review (each|per|a) |capacity|per day|a day\b|each (day|morning|week)|budget)"),
        ("labels", r"\b(label|labelled|labeled|confirmed (cases|frauds?)|cases (you|we|they) (found|confirmed)|known cases|ground truth)"),
        ("action", r"\b(action|do with|happens (when|after|with|if)|cost|wrong|mistake|misrout|miss(ed|es)?\b|hurts more|consequence)"),
        ("entity", r"\b(who\b|which (customer|user|account|biker|rider|store|entity|person)|per (customer|user|account|store))"),
        # the card's family question names a forecast among its options (live, HRN-022: read as the outcome, answered twice)
        ("family", r"\bwhat should dclab give for each row\b|\ba yes or no, a class\b"),
        ("target", r"\b(predict|outcome|target|forecast|estimate|which column|what (exactly )?(do you|should we) (want|need) to know)"),
        ("data_plan", r"\b(data|table|file|upload|export|columns?|source)\b"),
        ("family", r"\b(kind of problem|type of problem|classif|regress)"),
    )

    # the agent asking leave to build: the written person approves every gate (live, 2026-10-17: "Shall I build it now?" was
    # answered with the label's words three times, and the walk stopped)
    BUILD = r"\b(shall i|should i|do you want me to|want me to|may i|can i|ready to|okay to|ok to)\b.{0,40}\b(build|start|run|go ahead|proceed)"
    GO = "Yes, go ahead and build it."

    def reply(self, says: str, last: str) -> Optional[str]:
        """A person's reply in the chat to a walk that stopped: leave to build when the agent last asked for it, else the
        answer to what the stop and the agent's last words ask."""
        import re

        said = self._find("chat", says)  # the blocker the walk stopped on comes first: leave to build never leaves it open
        if said is not None:
            return said
        self.asked.pop()
        if re.search(self.BUILD, str(last or ""), re.I):
            self.asked.append("chat→build")
            return self.GO
        return self._unsure(self._find("chat", f"{says} {last}"))

    def _answer(self, field: str, question: str) -> Optional[str]:
        return self._unsure(self._find(field, question))

    def _unsure(self, said: Optional[str]) -> Optional[str]:
        if said is None and self.dont_know:
            self.asked[-1] = f"{self.asked[-1]}→unsure"
            return self.UNSURE
        return said

    def _find(self, field: str, question: str) -> Optional[str]:
        self.asked.append(field)
        said = self.answers.get(field) or self.answers.get(self.SAME.get(field, field))
        if said:
            return said
        if self.dont_know and self.SAME.get(field, field) in self.FIELDS:
            return None  # S8 B3: asked a field they cannot answer, a person says so; another field's words are no answer
        import re

        for name, pattern in self.MEANING:
            if self.answers.get(name) and re.search(pattern, str(question or ""), re.I):
                self.asked[-1] = f"{field}→{name}"  # what the question meant, kept beside what it was tagged
                return self.answers[name]
        target = str(self.answers.get("target") or "")  # live pass 3: a yes-or-no confirmation of the outcome column, when nothing else matched
        if target and re.search(rf"(^|[.?!]\s*)is\s+(the\s+)?`?{re.escape(target)}`?\s+(the\s+)?(outcome|target|label|column|time|number|count|flag)\b",
                                str(question or ""), re.I):
            self.asked[-1] = f"{field}→confirm_target"
            return f"Yes, {target} is the outcome we predict, as the table records it."
        return None


def _walk(coordinator: Any, draft_id: str, person: Any, walk: str, home: Any, critic: Any = None) -> dict[str, Any]:
    """The coordinator's loop, or (28.3) its state machine as the orchestration layer's graph, driven by the same person;
    ``critic`` (28.7): the understanding critic's model, its rules otherwise."""
    if walk != "graph":
        return coordinator.run(draft_id, person)
    from dclab_rnd.orchestra import checkpoints
    from dclab_rnd.orchestra.walk import Walk, drive

    with checkpoints.open_checkpointer(Path(home), database_url="") as saver:
        out = drive(Walk(coordinator, saver, home=Path(home), critic_advisor=critic), draft_id, person)
    return out


def _last_agent_text(draft: dict[str, Any]) -> str:
    """What the agent last said in the chat (where a model asks in its own words)."""
    for m in reversed(draft.get("messages") or []):
        if m.get("role") == "agent" and m.get("kind") in (None, "text", "question"):
            return str(m.get("content") or m.get("text") or "")
    return ""


def _upload(drafts: Any, agent: Any, draft_id: str, name: str, source: Path) -> None:
    from dclab_rnd.draft import pipeline

    (drafts.data_dir(draft_id) / name).write_bytes(source.read_bytes())
    asset = pipeline.new_asset(drafts, draft_id, "upload", name, name)
    pipeline.run(drafts, draft_id, asset["id"], agent=agent)


def _files(replica: Any, folder: Path) -> tuple[str, Optional[str]]:
    """The person's files on their disk: the table, and the label file when the replica has one apart."""
    if replica.files is not None:
        return replica.files(folder)
    replica.table.to_csv(folder / "table.csv", index=False)
    if replica.labels is not None:
        replica.labels[0].to_csv(folder / "labels.csv", index=False)
        return "table.csv", "labels.csv"
    return "table.csv", None


def arms() -> dict[str, dict[str, Any]]:
    """22.9: the routings a journey is played under: the rules alone, each decision classifier that passed its sealed
    gate routed to act at its own point (measured on its own), and all of them together. A measurement's setting, not a
    workspace's routing: a workspace routes a model only through ``ledger.routing.switch``."""
    from dclab_rnd.ledger import routing

    passed = {info["point"]: name for name, info in sorted(routing.passed().items())}
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def doc(points: dict[str, str], level: int = 3) -> dict[str, Any]:
        return {"decisions": {p: {"answerer": n, "level": level, "bar": 0.9, "since": at} for p, n in points.items()}}

    out = {"rules": doc({})}
    for point, name in passed.items():  # each on its own: acting (level 3), and acting with a check (level 2: a difference waits)
        out[point] = doc({point: name})
        out[f"{point}@check"] = doc({point: name}, 2)
    if len(passed) > 1:
        out["all"] = doc(passed)
        out["all@check"] = doc(passed, 2)
    return out


@contextlib.contextmanager
def _kept(home: Path, keep: Optional[Path]):
    """The play's folder (drafts, projects, the person's files) copied into ``keep`` when the play ends, however it ends."""
    try:
        yield
    finally:
        if keep is not None and home.is_dir():
            shutil.copytree(home, Path(keep) / "workspace", dirs_exist_ok=True)


def play(journey: dict[str, Any], seed: int = 0, routing_doc: Optional[dict[str, Any]] = None, live: Optional[dict[str, Any]] = None,
         keep: Optional[Path] = None, walk: str = "loop") -> dict[str, Any]:
    """One journey played from its first sentence; its scores and what happened at each part. ``routing_doc``: the
    decision models routed while it plays (22.9); the rules alone by default. ``keep``: a folder the play's whole record is
    copied into (its drafts with every turn, its project, its events and its row), a live run's records kept for good."""
    from dclab_rnd.ledger import use

    from dclab_core import signals

    from dclab_rnd import experience

    with tempfile.TemporaryDirectory() as tmp, use.scope(routing_doc or {"decisions": {}}), use.recording() as seen:
        log, before = experience.FileLog(Path(tmp), "journey"), experience.current_log()  # 21.5: the play's own record, graded
        experience.install(log)
        try:
            row = _play(journey, seed, live, keep, walk)
        finally:
            experience.install(before)
        events = log.read(limit=100_000)
        if keep is not None:
            Path(keep).mkdir(parents=True, exist_ok=True)
            (Path(keep) / "events.jsonl").write_text("".join(e.model_dump_json() + "\n" for e in events), encoding="utf-8")
    row["flags"] = sorted({f.detector for f in signals.detect(events)})
    asked_models = [e.payload for e in events if e.kind == "model.request" and e.payload.get("model")]  # 21.5: who answered, at what cost
    row["models"] = {str(p.get("purpose")): str(p.get("model")) for p in asked_models if p.get("purpose")}
    row["spent_eur"] = round(sum(float(p.get("cost") or 0.0) for p in asked_models), 6)
    row["coordinator"] = [e.payload.get("tool") for e in events if e.kind == "agent.step" and e.payload.get("agent") == "coordinator"]
    row["decisions"] = [{"point": c.point, "subject": c.subject, "answerer": c.answerer, "level": c.level, "model": c.model_value,
                         "rule": c.rule_value, "used": c.used, "changed": c.changed, "refused": c.refused, "review": c.review}
                        for c in seen if c.level]
    row["by_model"] = sum(1 for d in row["decisions"] if d["changed"])  # what the model changed, not an agreement
    row["agreed"] = sum(1 for d in row["decisions"] if d["used"] not in ("rule", "none") and not d["changed"])
    row["refused_by_validator"] = sum(1 for d in row["decisions"] if d["refused"])
    if keep is not None:
        (Path(keep) / "row.json").write_text(json.dumps(row, indent=1, default=str) + "\n", encoding="utf-8")
    return row


def _play(journey: dict[str, Any], seed: int = 0, live: Optional[dict[str, Any]] = None, keep: Optional[Path] = None,
          walk: str = "loop") -> dict[str, Any]:
    """One journey played from its first sentence; its scores and what happened at each part."""
    from dclab_rnd import kinds
    from dclab_rnd.draft import workflow as wf
    from dclab_rnd.draft.card import card_of
    from dclab_rnd.draft.chat import HomeAgent
    from dclab_rnd.draft.coordinator import Coordinator
    from dclab_rnd.draft.store import DraftStore
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds.acceptance import Run
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import ProjectStore, data as sd, engine, pipeline

    started = time.perf_counter()
    kind = kinds.get(journey["kind"]) if journey.get("kind") else None  # 21.5: a recorded session has no kind's replica
    replica = kind.case(seed) if kind is not None else _from_card(journey["data_card"], seed)
    row: dict[str, Any] = {"id": journey["id"], "kind": journey["kind"], "family": journey["family"], "parts": {}, "why": []}
    with tempfile.TemporaryDirectory() as tmp, _kept(Path(tmp), keep):
        home, mine = Path(tmp), Path(tmp) / "person"
        mine.mkdir()
        drafts, projects = DraftStore(home / "drafts"), ProjectStore(home / "projects")
        agent = HomeAgent(drafts, (live or {}).get("client"), reader=(live or {}).get("reader"),  # 21.5: the served model in a live run; the rules otherwise
                          second_opinion=bool((live or {}).get("second_opinion")))  # 28.6: the second opinion on the family
        table, labels = _files(replica, mine)
        first = drafts.create(journey["words"])["id"]
        agent.start(first)
        start_at = first
        if labels is not None:  # the cases apart from their table, each in its own draft, as the owner brought them
            _upload(drafts, agent, first, labels, mine / labels)
            other = drafts.create(journey["words"])["id"]
            agent.start(other)
            _upload(drafts, agent, other, table, mine / table)
        else:
            _upload(drafts, agent, first, table, mine / table)
        person = _Person(journey["answers"], dont_know=bool(journey.get("dont_know")))
        advisor = (live or {}).get("advisor")
        if journey.get("advisor"):  # 21.5: a role's scripted hand-off (the morning's data investigator naming an absent column)
            from dclab_core.agents import ScriptedAdvisor

            advisor = ScriptedAdvisor(journey["advisor"])
        import contextlib

        only_reader = live is not None and set(live) <= {"reader", "critic", "second_opinion", "designer", "challenger", "judge"}  # 28.6: the patterns' arms under the same scope  # 20.10 v4: the reader (28.7: and the critic) alone measured; every other role keeps the rules
        with private_scope(True) if live is None or only_reader else contextlib.nullcontext():  # 21.5: in a live run the roles ask their models
            coordinator = Coordinator(drafts, projects, agent, advisor)
            out = _walk(coordinator, start_at, person.person, walk, home, (live or {}).get("critic"))
            for _ in range(3):  # 26.5: a walk that stops on a question no tagged one carries (a model asks in its own words, or the card's
                # blocker names what is missing) is answered as a person in the chat would answer it, and the walk goes on
                says = str(out.get("says") or "")
                if out.get("project_id") or out.get("next") != "stop" or "?" not in says or says.startswith(("Waiting for the person", "Not built")):
                    break
                said = person.reply(says, _last_agent_text(drafts.get(start_at)))
                if not said:
                    break
                agent.reply(start_at, said)
                out = _walk(Coordinator(drafts, projects, agent, advisor), start_at, person.person, walk, home, (live or {}).get("critic"))
        row["handoffs"] = [{k: h.get(k) for k in ("role", "allowed", "refused_columns", "source")}
                           for h in (drafts.get(start_at).get("coordinator") or {}).get("handoffs") or []]
        if (live or {}).get("second_opinion"):  # 28.6: what the judge asked, and whether the person chose
            asked_again = [q for q in drafts.get(start_at).get("questions") or [] if q.get("source") == "second_opinion"]
            row["second_opinion"] = {"asked": len(asked_again), "declined": sum(bool(q.get("declined")) for q in asked_again),
                                     "chose": sum(bool(q.get("answered")) and not q.get("declined") for q in asked_again),
                                     "family": (drafts.get(start_at).get("understanding") or {}).get("family")}
        if walk == "graph":  # 28.4: what the team's readers did: the analyst's fills, the critic's rounds, each send-back with its reason
            own = drafts.get(start_at).get("coordinator") or {}
            history = (own.get("critic") or {}).get("history") or []
            why = {c.get("field"): c.get("why") or c.get("question") for h in history for c in h.get("concerns") or []}
            row["team"] = {"analyst": own.get("analyst"), "critic_rounds": len(history),
                           "sent_back": [{"field": h["sent_back"], "why": why.get(h["sent_back"]), "source": h.get("source")} for h in history if h.get("sent_back")],
                           "concerns_standing": [c.get("field") for c in (own.get("critic") or {}).get("awaiting") or []],
                           "overridden": bool((own.get("critic") or {}).get("overridden"))}
        understood = card_of(drafts.get(start_at))
        card = understood.family.value
        accepted = journey["family"] if isinstance(journey.get("family"), list) else [journey.get("family")]
        labels_ok = "labels" not in journey or understood.labels.value == journey["labels"]
        row.update(asked=person.asked, questions=len(person.asked), steps=[s["action"] for s in out["steps"]], state=out["state"],
                   said=out.get("says"), got={"family": card})
        row["parts"]["card"] = (card in accepted or journey.get("family") is None) and labels_ok  # a session's case may state no family
        if not row["parts"]["card"]:
            row["why"].append(f"card: {card!r}, the person's {accepted}" + ("" if labels_ok else "; the labels differ"))
        built = out.get("project_id")
        expect_built = journey.get("built", True)
        if not expect_built:  # right only when the build itself refused the family, never on another stop (the review's R2)
            said_no = built is None and out["next"] == "stop" and any(b.get("code") == REFUSED for b in out.get("blockers") or [])
            row["parts"].update(workflow=said_no, pipeline=said_no, holdout=said_no)
            if not said_no:
                row["why"].append("built: the engine does not run this family, and the build was not refused with a reason")
            return _end(row, started)
        if built is None:
            row["parts"].update(workflow=False, pipeline=False, holdout=False)
            row["why"].append(f"not built: stopped at {out['state']}: {out.get('says')}")
            return _end(row, started)
        draft = drafts.get(start_at)
        _, problems = wf.validate(draft.get("workflow") or {}, (draft.get("pack") or {}).get("key"))
        problems = list(problems) + _solution_workflow(understood, projects.get(built))
        for name, frame in (replica.tables or {}).items():  # the related tables the kind reads, attached by name on the project
            frame.to_csv(projects.data_dir(built) / f"{name}.csv", index=False)
            sd.attach_table(projects, built, f"{name}.csv", name)
        project = projects.get(built)
        refusal = None
        designed: dict[str, Any] = {}
        designers = (live or {}).get("designer") is not None
        if not project.get("solution"):  # the person saves the Solution page: the proposal, and what they state on it
            design = (lambda p, f, t, k, m: _design(p, f, t, k, m, journey.get("words") or "", live)) if designers else None  # 28.6
            refusal = _save(projects, project, journey.get("solution") or {}, design=design, record=designed)
        if designers:
            row["debate" if "changed" in designed else "design"] = designed
            row["refused_plans"] = int(bool(refusal))
        if refusal:
            row["parts"].update(workflow=False, pipeline=False, holdout=False)
            row["why"].append(f"solution refused: {refusal}")
            return _end(row, started)
        try:
            with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
                engine.run_all(projects, built)
            ran = None
        except Exception as error:  # noqa: BLE001 - a stage that fails is the journey's failure, said
            import traceback

            ran = f"{type(error).__name__}: {str(error)[:200]} at {traceback.extract_tb(error.__traceback__)[-1].name}"
        refused_moves = refused(projects, built)
        row["refused_moves"] = len(refused_moves)
        row["parts"]["workflow"] = not problems and not refused_moves
        if problems or refused_moves:
            row["why"].append(f"workflow: {problems[:2]}; refused moves {[t.get('move') for t in refused_moves[:3]]}")
        if ran is not None:
            row["parts"].update(pipeline=False, holdout=False)
            row["why"].append(f"engine: {ran}")
            return _end(row, started)
        with runner.fast_profile():  # the profile a run used is one of its inputs: read under the same one
            steps = pipeline.of(projects, built)
        stale = [f"{s.id} ({s.why})" for s in steps.steps if s.status == "stale"]
        ran = kind if kind is not None else kinds.resolve(projects.get(built)["solution"], projects.get(built).get("pack"))
        own = ran.own_machine() is not None  # a kind with its own machine has no analysis of tabular training rows to cite
        missing = [f"{s.id} ({s.status})" for s in steps.steps if s.status not in ("done", "stale")  # every step run and recorded
                   and not (own and s.id == "analysis" and s.status == "not_recorded")]
        row["parts"]["pipeline"] = not stale and not missing
        if stale or missing:
            row["why"].append(f"pipeline: stale {stale}, no run {missing}")
        project = projects.get(built)
        records = {stage: projects.read_stage(built, stage) for stage in engine.STAGE_KEYS}
        done = Run(replica, project["solution"], records, project.get("labels") or {})
        ran_as = kinds.resolve(project["solution"], project.get("pack")).key
        if kind is None:  # a session's case: its card's traps are the truth (every leak suspect kept out, a holdout read)
            checks = _card_checks(journey["data_card"], done)
        else:
            checks = kind.accept_engine(done) if kind.own_machine() is not None else kind.accept(done)
        failed = [c["decision"] for c in checks if not c["ok"]]
        if kind is not None and ran_as != journey["kind"]:  # the problem ran as the kind of the person's problem (stated or proposed)
            failed.append(f"ran as {ran_as}, not {journey['kind']}")
        row["parts"]["holdout"] = bool(checks) and not failed
        row["holdout"] = (done.evidence("final").get("holdout_metrics") or {})
        row["got"].update(target=project["solution"].get("target"), kind=ran_as, kind_stated=bool((journey.get("solution") or {}).get("kind")))
        if failed or not checks:
            row["why"].append(f"holdout: {failed or 'no acceptance decision'}")
    return _end(row, started)


def _from_card(card: dict[str, Any], seed: int) -> Any:
    """A recorded session's table, invented from its data card (19.11): the same columns, roles, bands and traps, no row
    of the person's."""
    from types import SimpleNamespace

    from dclab_core.datacard import DataCard, synthesize

    model = DataCard.read(card)
    frames = synthesize(model, seed=seed)
    main = next((t.name for t in model.tables if t.role == "main"), model.tables[0].name)
    labels = next((t.name for t in model.tables if t.role == "labels"), None)  # the file of known cases, in its own draft as it came
    return SimpleNamespace(table=frames[main], files=None, labels=(frames[labels], None) if labels else None,
                           tables={k: v for k, v in frames.items() if k not in (main, labels)}, facts={})


def _card_checks(card: dict[str, Any], run: Any) -> list[dict[str, Any]]:
    """The holdout of a session's case: a holdout number recorded, and every column the card marks a leak suspect kept
    out of the model."""
    from dclab_rnd.kinds.acceptance import check

    suspects = sorted({c.get("name") for t in card.get("tables") or [] for c in t.get("columns") or [] if c.get("purpose") == "leak_suspect"})
    roles = run.evidence("data").get("column_roles") or {}
    kept = [c for c in suspects if c in roles and roles[c] == "feature"]
    return [check("a holdout number is recorded", bool(run.evidence("final").get("holdout_metrics")), str(run.evidence("final").get("holdout_metrics"))),
            check("every leak suspect of the card is kept out", not kept, f"modelled: {kept}" if kept else f"kept out: {suspects}")]


def refused(projects: Any, project_id: str) -> list[dict[str, Any]]:
    """The project's moves the graph's validator refused: its verdict is the entry's status (``allowed`` lists the moves
    open at that state)."""
    return [t for t in projects.transitions(project_id, limit=10_000) if t.get("status") == "blocked"]


def _solution_workflow(card: Any, project: dict[str, Any]) -> list[str]:
    """20.6's solution workflow from the problem card and the table's data card, validated on its content."""
    from dclab_core import solution_workflow as sw
    from dclab_core.datacard import DataCard

    try:
        data_card = (project.get("data") or {}).get("card")
        flow = sw.solution_workflow(card, DataCard.read(data_card) if data_card else None)
    except Exception as error:  # noqa: BLE001 - no workflow for this card is the journey's failure, said
        return [f"no solution workflow: {str(error)[:200]}"]
    return [f"solution workflow: {p}" for p in sw.validate(flow)]


def _design(project: dict[str, Any], frame: Any, target: str, task: str, moment: str, words: str, live: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """28.6: the designers given in ``live`` over the rules' plan: the solution planner alone (``designer``), or the debate
    (``designer``, ``challenger``, ``judge``). Returns what their plan changes from the rules' plan, as solution fields
    (kept-out columns, identifiers, a group split), and what they did; the code checks every change as the planner's."""
    import json as _json

    from dclab_core.analyse import analyse
    from dclab_core.understand import understand
    from dclab_rnd.core_adapter import solution_fields_from_plan
    from dclab_rnd.studio.solution import provisional_training_rows
    from dclab_rnd.studio.team import _fact

    brief = understand(words or "", {"table": [str(c) for c in frame.columns]})
    brief = _fact(brief, "target_column", target, target)
    if moment:
        brief = _fact(brief, "prediction_moment", moment, moment)
    times = [c for c in ((project.get("data") or {}).get("profile") or {}).get("time_candidates") or [] if c in frame.columns]
    present = frame.index[frame[target].notna()]
    train = provisional_training_rows(present, frame.loc[present, times[0]] if times else None)
    findings = analyse({"table": frame}, target=target, task=task, brief=brief, train_index=train, moment=moment)
    def applied(rule_plan: Any, plan: Any) -> tuple[dict[str, Any], Optional[str]]:
        """What a plan changes from the rules' plan, as the solution's fields; and what could not be applied."""
        rule, final = solution_fields_from_plan(rule_plan, target, moment), solution_fields_from_plan(plan, target, moment)
        fields: dict[str, Any] = {}
        ruled_out = {f["column"] for f in rule["forbidden"]}
        added = [f for f in final["forbidden"] if f["column"] not in ruled_out and f["column"] != target and f["column"] in frame.columns]
        if added:
            fields["forbidden"] = added
        if (set(final["identifiers"]) - set(rule["identifiers"])) & set(map(str, frame.columns)):
            fields["identifiers"] = sorted((set(final["identifiers"]) - set(rule["identifiers"])) & set(map(str, frame.columns)))
        if final.get("group_column") and final["group_column"] != rule.get("group_column") and final["group_column"] in frame.columns:
            if not times:  # a table with no time: a group split, applied
                fields["group_column"] = final["group_column"]
            elif not final.get("time_column"):  # a group split that drops the time order is the person's call (30.11): said, not applied
                return fields, "a group split without the time column would drop the time order"
            else:  # 28.6 (the debate's live runs): applied only where the engine can test it, by its own rule
                from dclab_rnd.expansion.datasets import time_group_feasible

                ok, unseen, later = time_group_feasible(frame[times[0]], frame[final["group_column"]])
                if not ok:
                    return fields, (f"{final['group_column']} recurs over time: {unseen} of the last {later} rows are of one unseen before, "
                                    "too few to test a grouped forward split")
                fields["group_column"] = final["group_column"]
        return fields, None

    if live.get("challenger") is not None or live.get("judge") is not None:
        from dclab_core.debate import debate

        out = debate(live.get("designer"), live.get("challenger"), live.get("judge"), brief, findings, columns=[str(c) for c in frame.columns])
        fields, skipped = applied(out["rule_plan"], out["plan"])
        single, _ = applied(out["rule_plan"], out["single"]["plan"])
        rec = out["debate"]
        done = {"changed": sorted(k for k in set(fields) | set(single) if _json.dumps(fields.get(k), sort_keys=True) != _json.dumps(single.get(k), sort_keys=True)),
                "agreed": rec.agreed, "a": [c.key for c in rec.a], "b": [c.key for c in rec.b], "kept": [c.key for c in rec.kept],
                "rulings": [{"key": r.key, "keep": r.keep, "by": r.by} for r in rec.rulings], "sources": rec.sources}  # changed: what reaches the solution
    else:
        from dclab_core.plan import plan_with

        out = plan_with(live.get("designer"), brief, findings, columns=[str(c) for c in frame.columns])
        fields, skipped = applied(out["rule_plan"], out["plan"])
        done = {"kept": [c["key"] for c in out.get("kept") or []], "refused": len(out.get("refused") or []), "source": out.get("source")}
    if skipped:
        done["not_applied"] = skipped
    done["applied"] = sorted(fields)
    return fields, done


def _save(projects: Any, project: dict[str, Any], statements: dict[str, Any], design: Optional[Callable[..., tuple[dict[str, Any], dict[str, Any]]]] = None,
          record: Optional[dict[str, Any]] = None) -> Optional[str]:
    """The person saves the Solution page through the same checks as the route: the graph's validator, then the
    columns; the refusal's words, or None when saved. 28.6: ``design`` (the designers' changes to the proposal, the
    person's statements still over them), what it did kept in ``record``."""
    from dclab_rnd.kinds.acceptance import adopt
    from dclab_rnd.studio import graph
    from dclab_rnd.studio.solution import Solution
    from dclab_rnd.studio import data as sd
    from dclab_rnd.studio.solution import propose

    suggestion = project.get("suggestion") or {}
    target = statements.get("target", suggestion.get("target"))
    if project.get("labels") and suggestion:  # a label file's join: its suggestion is the starting point
        proposal = {**suggestion}
    elif target:
        frame = sd.load_table(sd.data_path(projects, project))
        from dclab_rnd.kinds.propose import NO_OUTCOME
        from dclab_rnd.studio.solution import card_context

        proposal = propose(frame, project["data"]["profile"], target, statements.get("task"), project.get("pack"), None, **card_context(project))
        if proposal.get("kind") in NO_OUTCOME and proposal.get("target") is None and "target" not in statements:
            target = None  # the kind reads no outcome: the one named describes it (its own field), and the save leaves the target empty
    elif project.get("solution") and not statements:  # S9: a kind that reads no outcome, its solution built from the card: kept
        proposal = {k: v for k, v in project["solution"].items() if v is not None}
    else:
        proposal = {}
    stated_kind = statements.get("kind")
    if stated_kind:  # a kind the person states runs as its own base task (segmentation: multiclass)
        from dclab_rnd import kinds

        task = statements.get("task") or kinds.get(stated_kind).base_task
    else:
        task = statements.get("task") or proposal.get("task") or "binary"
    moment = (suggestion.get("prediction_moment") or ((project.get("draft") or {}).get("understanding") or {}).get("prediction_moment")
              or str(proposal.get("prediction_moment") or ""))  # S9: a kept solution's own moment
    if design is not None and target and proposal:
        frame = sd.load_table(sd.data_path(projects, project))
        if (project.get("labels") or {}).get("target") == target:  # the outcome from a file of known cases, joined as the engine joins it
            from dclab_rnd.studio import labels as label_file

            frame = label_file.apply(projects, project, frame)
        fields, done = design(project, frame, target, task, moment)
        already = {f["column"] for f in proposal.get("forbidden") or []}  # the review of 28.6: a column kept out twice refused the solution
        proposal = {**proposal, "forbidden": [*(proposal.get("forbidden") or []), *(f for f in fields.get("forbidden", []) if f["column"] not in already)],
                    "identifiers": sorted({*(proposal.get("identifiers") or []), *fields.get("identifiers", [])}),
                    **({"group_column": fields["group_column"]} if fields.get("group_column") else {})}
        if record is not None:
            record.update(done)
    verdict = graph.check(project, "set_solution", "human", target=target)
    graph.log(projects, project["id"], verdict, project, outcome="done: solution saved" if verdict.allowed else None)
    if not verdict.allowed:
        return verdict.message
    try:
        adopt(projects, project, proposal, target, task, moment, {k: v for k, v in statements.items() if k not in ("target", "task")})
        saved = projects.get(project["id"])
        label_target = (saved.get("labels") or {}).get("target")  # as the route: the label file's target joins the table
        Solution(**saved["solution"]).check_columns(list(saved["data"]["columns"]) + ([label_target] if label_target else []))
    except ValueError as error:
        return str(error).split("\n")[0][:300]
    return None


def _end(row: dict[str, Any], started: float) -> dict[str, Any]:
    row["right"] = all(row["parts"].get(p) for p in PARTS)
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def _play_one(args: tuple[Any, ...]) -> dict[str, Any]:
    journey, seed, doc = args[:3]
    walk = args[3] if len(args) > 3 else "loop"
    try:
        return play(journey, seed, doc, walk=walk)
    except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason
        return {"id": journey["id"], "kind": journey["kind"], "family": journey["family"], "parts": {p: False for p in PARTS},
                "right": False, "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]}


def measure(split: str, seed: int = 0, workers: int = 1, only: Optional[list[str]] = None,
            routing_doc: Optional[dict[str, Any]] = None, walk: str = "loop") -> list[dict[str, Any]]:
    """Every journey of ``split`` played; ``walk``: the coordinator's loop, or (28.3) its graph."""
    journeys = [j for j in load(split) if not only or j["id"] in only or j["kind"] in only]
    if workers <= 1:
        return [_play_one((j, seed, routing_doc, walk)) for j in journeys]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_play_one, [(j, seed, routing_doc, walk) for j in journeys]))


def compare(by_arm: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """22.9's reading: each arm against the rules, part by part and journey by journey (what a model changed, for the
    better or the worse), with the validator's refusals; the gate: every arm right at least as often as the rules on
    every part, and no graph move refused."""
    base = {r["id"]: r for r in by_arm.get("rules") or []}
    out: dict[str, Any] = {}
    for arm, rows in by_arm.items():
        parts = {p: sum(bool(r["parts"].get(p)) for r in rows) for p in PARTS}
        changed = [{"id": r["id"], "rules": base[r["id"]]["right"], "arm": r["right"]} for r in rows
                   if r["id"] in base and r["right"] != base[r["id"]]["right"]]
        out[arm] = {"journeys": len(rows), "right": sum(r["right"] for r in rows), "parts": parts, "changed": changed,
                    "decisions_by_model": sum(r.get("by_model") or 0 for r in rows),  # changed from the rule's answer
                    "agreed_with_the_rule": sum(r.get("agreed") or 0 for r in rows),
                    "held_for_review": sum(sum(1 for d in r.get("decisions") or [] if d.get("review")) for r in rows),
                    "refused_by_validator": sum(r.get("refused_by_validator") or 0 for r in rows),
                    "refused_moves": sum(r.get("refused_moves") or 0 for r in rows)}
    rules = out.get("rules") or {}
    for a in out.values():  # each arm on its own: a model that may act must pass alone
        a["ok"] = bool(rules) and a["refused_moves"] == 0 and all(a["parts"][p] >= rules["parts"][p] for p in PARTS)
    ok = all(a["ok"] for a in out.values())
    worse = [f"{arm} on {p}" for arm, a in out.items() for p in PARTS if rules and a["parts"][p] < rules["parts"][p]]
    return {"arms": out, "verdict": "pass: every arm right at least as often as the rules on each part, and no move refused" if ok else
            "fail: " + ("; ".join(worse) or "a move was refused, or no rules arm")}


def families(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every family listed (one with no journey too, its interval [0, 1]), with its journeys right and the interval."""
    from dclab_core.card import FAMILIES

    out: dict[str, list[bool]] = {f: [] for f in FAMILIES}
    for r in rows:
        for family in (r["family"] if isinstance(r["family"], list) else [r["family"]])[:1]:
            out.setdefault(family, []).append(bool(r["right"]))
    return [{"family": f, "journeys": len(v), "right": sum(v), "interval": list(wilson(sum(v), len(v)))} for f, v in sorted(out.items())]


def verdict(rows: list[dict[str, Any]]) -> str:
    right, n = sum(r["right"] for r in rows), len(rows)
    low, _ = wilson(right, n)
    refused = sum(r.get("refused_moves") or 0 for r in rows)
    missing = [f["family"] for f in families(rows) if not f["journeys"]]
    ok = n > 0 and right / n >= 0.85 and low >= 0.6 and refused == 0 and not missing
    return (f"{'pass' if ok else 'fail'}: {right} of {n} journeys right (95% interval {low} to {wilson(right, n)[1]}), "
            f"{refused} moves refused by the validator" + (f", no journey for {', '.join(missing)}" if missing else ""))


def reader_passed() -> bool:
    """Whether the conversation reader passed its sealed measurement (``GATE_V4``'s reader rule), from the stored run."""
    for path in sorted(RESULTS.glob("JRN-*_sealed_v4.json")) if RESULTS.exists() else []:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("arm") == "reader":
            return bool((doc.get("reader_against_rules") or {}).get("passed"))
    return False


def sealed_runs(split: str = "sealed") -> list[Path]:
    return sorted(RESULTS.glob(f"JRN-*_{split}.json")) if RESULTS.exists() else []


def store(rows: list[dict[str, Any]], split: str, result: str, arm: str = "rules", seed: int = 0, spent_eur: float = 0.0,
          extra: Optional[dict[str, Any]] = None) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("JRN-*.json")] or [0])
    ident = f"JRN-{n:03d}"
    doc = {"id": ident, "campaign_id": "journeys_v1", "kind": "acceptance_journeys", "split": split, "arm": arm,
           "journeys": fingerprint(HERE / f"journeys_{split}.json"), "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gate": GATES.get(split, GATE), "seed": seed, "spent_eur": spent_eur, "rows": rows, "families": families(rows),
           "summary": {"journeys": len(rows), "right": sum(r["right"] for r in rows),
                       "by_part": {p: sum(bool(r["parts"].get(p)) for r in rows) for p in PARTS}},
           "verdict": result, "status": "completed", "experiment_id": ident, **(extra or {}),
           "rule": f"Scores on {len(rows)} written journeys over seeded replicas say how DCLab carries these people from their "
                   "first sentence to a holdout number; they are not a claim about any other problem or about production readiness."}
    path = RESULTS / f"{ident}_{split}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def store_compare(by_arm: dict[str, list[dict[str, Any]]], result: dict[str, Any], split: str, seed: int, spent_eur: float = 0.0,
                  routing: Optional[dict[str, Any]] = None, extra: Optional[dict[str, Any]] = None) -> Path:
    """22.9's measurement, beside the journeys' runs: one result with every arm. 26.5: ``routing`` the arms' own (a live
    arm's), ``spent_eur`` what the live arms cost, ``extra`` what the run adds (its models, its cap)."""
    out = ROOT / "evidence" / "campaigns" / "decision_workflow_v1" / "results"
    out.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in out.glob("DWF-*.json")] or [0])
    ident = f"DWF-{n:03d}"
    doc = {"id": ident, "campaign_id": "decision_workflow_v1", "kind": "decision_models_in_the_workflow", "split": split, "seed": seed,
           "journeys": fingerprint(HERE / f"journeys_{split}.json"), "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gate": {"version": "decision_workflow_gate_v1", "rule": "with decision models routed to act, the journeys are right at least as "
                    "often as with the rules on the card, the workflow, the pipeline and the holdout, no graph move is refused, and each "
                    "model is measured on its own"},
           "arms": {arm: [{k: r.get(k) for k in ("id", "kind", "right", "parts", "why", "decisions", "by_model", "agreed", "refused_by_validator",
                                                 "refused_moves", "holdout")} for r in rows] for arm, rows in by_arm.items()},
           "summary": result["arms"], "verdict": result["verdict"], "spent_eur": spent_eur, "status": "completed", "experiment_id": ident,
           "routing": arms() if routing is None else routing,  # 21.5: which answerer at which level each arm ran, read by the decision switch
           **(extra or {}),
           "rule": "Arms on the dev journeys over seeded replicas: what each routed decision model changes in these journeys, not a "
                   "claim about any other problem; routing a model in a workspace still needs its audited switch."}
    path = out / f"{ident}_{split}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def workflow_measured(answerer: str, point: str, level: int) -> Optional[dict[str, Any]]:
    """21.5: the latest measurement (22.9) in which ``answerer`` acted at ``point`` at ``level`` on its own, or None."""
    out = ROOT / "evidence" / "campaigns" / "decision_workflow_v1" / "results"
    for path in sorted(out.glob("DWF-*.json"), reverse=True) if out.exists() else []:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("journeys") != fingerprint(HERE / f"journeys_{doc.get('split', 'dev')}.json"):
            continue  # measured on journeys that have changed since: not this suite's measurement
        for arm, routing in (doc.get("routing") or {}).items():
            entries = (routing or {}).get("decisions") or {}
            if list(entries) == [point] and entries[point].get("answerer") == answerer and int(entries[point].get("level") or 0) == level:
                return {"run": doc["id"], "arm": arm, **(doc.get("summary") or {}).get(arm, {})}
    return None
