"""The routing collector (P8, 2026-10-29): one example per routing decision Home makes, the 7th model's data (``m7_routing``).

It runs after the shapers, as the other set builders do, and joins what the store already keeps (so a deleted draft's
decisions stay: the store never forgets what it kept):

- every routing decision of the ledger (:data:`POINTS`, and any ``request.*`` point added later), from the workspaces'
  ledgers and databases;
- the person's words, from the draft as the store kept it (the problem and their messages up to the decision's moment),
  only where the decision's and the draft's consent is ``train`` and the draft did not turn training off;
- every answerer's reading: the rules', the model's with its API thinking summary (the thinking blocks of its kept exchange:
  by the request id the answer names, else the same draft's exchange of that purpose just before the decision), our own
  model's answer in shadow (a classifier's), and the person's;
- the route taken, what the person did with the result (``track.outcome``), and the label, by class: ``real`` (the person's
  confirm or change on the card, a result kept or changed, a measurement), ``written`` (a written request's truth, an eval's
  gold, a scripted person's click) and ``silver`` (Claude's reading of a real request no one confirmed), kept apart, the
  written and silver ones weighted lower;
- Home's written requests (P6, ``intent_requests_v1.json``; the checker's flagged ones never) and Home's eval runs, from
  the routing keeper's entries; a sealed run as its shape only, in the sealed split.

One request's words are in one split everywhere (the split group is the request's hash): a written request replayed live,
or an eval's request kept as a draft, lands where its written row is. A request of a Home exam (its words, or the words a
model read for it) is sealed everywhere, run or not: the sealed partition is written once, so a request first sealed and
later "dev" would sit in two splits. The model's thinking summary is kept only on rows whose words are kept (it reads them).
Exchanges are joined only from workspaces whose consent is train, and by the same words only.

    python -m dclab_rnd.store.sets.routing          # build the sets now and print the report
    python -m dclab_rnd.store.sets.routing --counts # what they hold, per point and label class
    python -m dclab_rnd.store.sets.routing --export # chat JSONL and a table (GLiNER2's format: python -m dclab_rnd.lake answers-export)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from dclab_core.datasets import split_of
from dclab_core.routing_example import WEIGHTS, RoutingExample, RoutingReading, label_class, language
from dclab_core.store import Provenance

from ..layout import Layout
from ..store import Store
from . import write_set

MODEL_TYPE = "m7_routing"
POINTS = ("request.intent", "request.exit", "request.suggestion", "request.tool", "workflow.next_question", "problem.pack")  # Home's routing decisions on a draft
READING_PURPOSES = ("home_intent", "home_agent")  # the model purposes whose exchanges read a request (plus any answerer named)
BEFORE, AFTER = 600.0, 30.0  # an exchange joined by time: at most this many seconds before (after) its decision
OUTCOME_BY = "person:outcome"  # ledger.capture.OUTCOME_BY
RANK = {"card": 0, "outcome": 1, "measured": 2, "blind": 3, "benchmark": 4, "teacher": 5}
TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")  # draft.intent.TRACKS
ABOUT = ("Every routing decision Home made on a request, with the person's words (train consent only), every answerer's reading "
         "(the model's with its API thinking summary, the rules', our own model's in shadow) and the label by class: real, written "
         "(a writer's or an eval's truth, a script's click) and silver (Claude's reading of an unconfirmed real request), apart.")


def routing_point(point: Any) -> bool:
    return point in POINTS or str(point or "").startswith("request.")


def norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def text_key(text: Any) -> str:
    return hashlib.sha256(norm(text).encode()).hexdigest()[:16]


def _at(value: Any) -> Optional[float]:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def _gz(base: Path, needles: Iterable[str] = ()) -> Iterator[dict[str, Any]]:
    """A kind's shaped records as plain dicts, only the lines holding one of ``needles`` (no contract parsing: a join needs none)."""
    needles = tuple(needles)
    for path in sorted(base.rglob("*.jsonl.gz")) if base.exists() else []:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                if needles and not any(n in line for n in needles):
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


# ------------------------------------------------------------------------------------------------------------- reading
def decisions(layout: Layout) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Every routing decision once per (workspace, id), and every label of one by the record it labels."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    labels: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    needles = [f'"{p}"' for p in POINTS] + ['"request.']
    for r in _gz(layout.records_of("decisions"), needles):
        b = r.get("body") or {}
        if not routing_point(b.get("point")):
            continue
        if r.get("type") == "decision" and b.get("id"):
            found.setdefault((str((r.get("provenance") or {}).get("workspace")), str(b["id"])), r)
        elif r.get("type") == "label" and b.get("record") and b.get("id"):
            labels[str(b["record"])].setdefault(str(b["id"]), b)
    return found, {k: list(v.values()) for k, v in labels.items()}


def drafts(layout: Layout, wanted: set[tuple[str, str]]) -> dict[tuple[str, str], dict[str, Any]]:
    """The latest kept snapshot of each wanted draft (its words, consent and provenance; a deleted draft's last one)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _gz(layout.records_of("conversations"), ('draft.json"', '"db/draft')):
        prov = r.get("provenance") or {}
        source = str(prov.get("source") or "")
        if r.get("type") != "snapshot" or not (re.match(r"drafts/[^/]+/draft\.json$", source) or source.startswith("db/draft")):
            continue
        b = r.get("body") or {}
        ident = str(b.get("id") or (source.split("/")[1] if source.startswith("drafts/") else ""))
        key = (str(prov.get("workspace")), ident)
        if key not in wanted:
            continue
        if key in out and str(out[key]["prov"].get("at") or "") > str(prov.get("at") or ""):
            continue
        said = [(str(m.get("at") or ""), str(m.get("text") or "")) for m in b.get("messages") or []
                if isinstance(m, dict) and m.get("role") == "user" and m.get("text")]
        consent = ((b.get("settings") or {}).get("consent") or {}) if isinstance(b.get("settings"), dict) else {}
        out[key] = {"problem": str(b.get("problem") or ""), "said": said, "prov": prov,
                    "train_off": consent.get("train") is False, "data": bool(b.get("assets"))}
    return out


def _thinking(blocks: Any) -> str:
    """The API thinking summary as text: each returned block's text, in order (never its signature)."""
    if isinstance(blocks, str):
        return blocks.strip()
    parts = [str(b.get("text") or b.get("thinking") or "") for b in blocks or [] if isinstance(b, dict)]
    return "".join(parts).strip()


def _user(messages: Any) -> str:
    if not isinstance(messages, list):  # an exchange kept as its shape only (a count, never the words)
        return ""
    users = [m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    last = users[-1] if users else ""
    if isinstance(last, list):  # content blocks
        last = " ".join(str(c.get("text") or "") for c in last if isinstance(c, dict))
    return str(last or "")


def exchanges(layout: Layout, purposes: set[str]) -> list[dict[str, Any]]:
    """The kept exchanges of these purposes: who asked, when, the request it read, the reply and the thinking summary."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _gz(layout.records_of("model_requests"), ('"model_exchanges',)):
        prov, b = r.get("provenance") or {}, r.get("body") or {}
        if not str(prov.get("source") or "").startswith("model_exchanges") or b.get("purpose") not in purposes:
            continue
        if prov.get("consent") != "train":  # an exchange whose workspace does not allow training (its thinking reads the words)
            continue
        key = (str(prov.get("workspace")), str(b.get("model_request") or r.get("id")))
        out.setdefault(key, {"workspace": key[0], "request": b.get("model_request"), "purpose": b.get("purpose"), "draft": b.get("draft"),
                             "at": _at(b.get("at")), "model": b.get("model"), "user": _user(b.get("messages")), "reply": b.get("reply"),
                             "thinking": _thinking(b.get("thinking")), "whole": bool(b.get("whole")), "prov": prov})
    return list(out.values())


def outcomes(layout: Layout) -> dict[tuple[str, str], str]:
    """What the person did with each draft's result (36.2), the latest verdict per (workspace, draft)."""
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for r in _gz(layout.records_of("sessions"), ('"track.outcome"',)):
        b = r.get("body") or {}
        if b.get("kind") != "track.outcome":
            continue
        draft = (b.get("ids") or {}).get("conversation") or b.get("conversation")
        verdict = (b.get("payload") or {}).get("verdict")
        key = (str((r.get("provenance") or {}).get("workspace")), str(draft))
        if draft and verdict and str(b.get("at") or "") >= out.get(key, ("", ""))[0]:
            out[key] = (str(b.get("at") or ""), str(verdict))
    return {k: v[1] for k, v in out.items()}


def _kept(layout: Layout, prefix: str) -> dict[str, tuple[Any, Any]]:
    """The routing keeper's latest entry of each source under ``prefix``: its parsed JSON and its entry."""
    store = Store(layout)
    latest: dict[str, Any] = {}
    for e in store.entries("tests"):
        if e.workspace == "rnd" and e.source.startswith(prefix) and (e.source not in latest or e.version > latest[e.source].version):
            latest[e.source] = e
    out = {}
    for source, e in sorted(latest.items()):
        try:
            out[source] = (json.loads(store.read(e)), e)
        except (ValueError, OSError):
            continue
    return out


def _entry_prov(e: Any, **extra: Any) -> Provenance:
    return Provenance(entry=e.id, sha256=e.sha256, source=e.source, workspace=e.workspace, consent=e.consent, at=e.captured_at, **extra)


def exams(home: Optional[Path] = None, results: Optional[Path] = None) -> dict[str, tuple[str, bool]]:
    """Every Home exam's requests by their words' hash: (the exam, whether it was run). Only hashes are read into memory."""
    from ..keepers.routing import HOME, RESULTS

    home, results = Path(home or HOME), Path(results or RESULTS)
    out: dict[str, tuple[str, bool]] = {}
    for path in sorted(home.glob("*sealed*.json")):
        name = path.stem.replace("requests_", "")
        spent = any(results.glob(f"HOME-*_{name}.json")) if results.is_dir() else False
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for row in doc if isinstance(doc, list) else doc.get("requests") or []:
            if isinstance(row, dict) and row.get("text"):
                out[text_key(row["text"])] = (f"home:{name}", spent)
    return out


# ---------------------------------------------------------------------------------------------------------- the rows
class Joiner:
    """The store's records, read once, and the joins of one build."""

    def __init__(self, layout: Layout, scripted: frozenset[str], home: Optional[Path] = None, results: Optional[Path] = None):
        from ..keepers.routing import HOME, RESULTS

        self.layout, self.scripted = layout, scripted
        self.home, self.results = Path(home or HOME), Path(results or RESULTS)  # the exams' files (their hashes and labels only)
        self.decisions, self.labels = decisions(layout)
        wanted = {(ws, str((r.get("body") or {}).get("draft") or ((r.get("body") or {}).get("subject") or {}).get("key")))
                  for (ws, _), r in self.decisions.items()}
        self.drafts = drafts(layout, wanted)
        named = {str(a.get("answerer")) for r in self.decisions.values() for a in (r.get("body") or {}).get("answers") or []
                 if isinstance(a, dict) and a.get("by") == "model_role"}
        self.exchanges = exchanges(layout, set(READING_PURPOSES) | named)
        self.by_request = {(x["workspace"], str(x["request"])): x for x in self.exchanges if x.get("request")}
        # by draft and purpose, whatever workspace kept it: a live test's exchanges may sit in the home workspace while its
        # ledger is the test's own (models/gateway._exchange_home); a draft's id is unique, its own workspace preferred
        self.by_draft: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for x in self.exchanges:
            if x.get("draft"):
                self.by_draft[(str(x["draft"]), str(x["purpose"]))].append(x)
        self.outcomes = outcomes(layout)
        self.exams = exams(self.home, self.results)
        self.truth: dict[str, dict[str, Any]] = {}   # a written or eval request's truth by its words' hash
        self.flagged: set[str] = set()
        # 2026-10-29: Claude's routing labels (dclab_rnd.store.routing_labels), by example id: silver, apart, weighted lower
        self.claude: dict[str, dict[str, Any]] = {}
        claude_labels = Path(layout.root) / "labels" / "m7_routing_claude.jsonl"
        for line in claude_labels.read_text(encoding="utf-8").splitlines() if claude_labels.exists() else []:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("example") and row.get("value") is not None:
                self.claude[str(row["example"])] = row
        self.out: Counter = Counter()

    def exchange(self, ws: str, draft: str, purpose: str, at: Optional[float], request: Optional[str]) -> Optional[dict[str, Any]]:
        if request:
            exact = self.by_request.get((ws, request)) or next((x for (w, r), x in self.by_request.items() if r == request), None)
            if exact is not None:
                return exact
        if at is None:
            return None
        near = [x for x in self.by_draft.get((draft, purpose), []) if x["at"] is not None and -BEFORE <= x["at"] - at <= AFTER]
        return min(near, key=lambda x: (x["workspace"] != ws, abs(x["at"] - at))) if near else None

    def by_text(self, text: str, purposes: Iterable[str], at: Optional[float], within: float) -> Optional[dict[str, Any]]:
        """A draft-less exchange (a checker's, an eval's plain arm) that read these words, the nearest in time."""
        n = norm(text)
        if not n:
            return None
        found = [x for x in self.exchanges if x["purpose"] in purposes and norm(x["user"]) == n  # the same words, never a part of others
                 and (at is None or x["at"] is None or abs(x["at"] - at) <= within)]
        return min(found, key=lambda x: abs((x["at"] or 0) - (at or 0))) if found else None


def _model_reading(x: Optional[dict[str, Any]], answerer: str, value: Any, *, used: bool, confidence: Any = None,
                   why_none: str) -> RoutingReading:
    conf = confidence if isinstance(confidence, (int, float)) and 0 <= confidence <= 1 else None
    if x is not None and x.get("thinking"):
        return RoutingReading(by="model", answerer=answerer, value=value, used=used, confidence=conf, model=x.get("model"),
                              request=x.get("request"), thinking=x["thinking"][:20000], thinking_source="recorded")
    why = ("its exchange kept no thinking summary" if x is not None else why_none)
    return RoutingReading(by="model", answerer=answerer, value=value, used=used, confidence=conf, model=(x or {}).get("model"),
                          request=(x or {}).get("request"), no_thinking=why)


def _reply_track(reply: Any) -> Any:
    """The first track a home_intent reply names (None: it said the words do not tell, or it was not JSON)."""
    text = str(reply or "")
    m = re.search(r"\{.*\}", text, re.S)
    try:
        doc = json.loads(m.group(0)) if m else {}
    except ValueError:
        return None
    tracks = [t for t in doc.get("tracks") or [doc.get("track")] if t in TRACKS] if isinstance(doc, dict) else []
    return tracks[0] if tracks else None


def _label(j: Joiner, rid: str, ws: str) -> tuple[Any, Optional[str], Optional[str]]:
    """The best standing label of a decision: (value, class, by), superseded and retracted ones left out."""
    own = j.labels.get(rid, [])
    gone = {lab.get("supersedes") for lab in own if lab.get("supersedes")}
    best: Optional[tuple[int, Any, str, str]] = None
    for lab in own:
        if lab.get("id") in gone or lab.get("value") is None:
            continue
        source, by = str(lab.get("source")), str(lab.get("by") or "")
        cls = label_class(source, scripted=ws in j.scripted)
        if cls is None:
            continue
        kind = ("outcome" if by == OUTCOME_BY else "card") if source == "user_decision_exposed" else \
            {"model": "teacher"}.get(source, source)
        rank = RANK.get(kind, 9)
        if best is None or rank < best[0]:
            who = (f"scripted:{kind}" if ws in j.scripted else f"person:{kind}") if kind in ("card", "outcome") else f"{source}:{by or '-'}"
            best = (rank, lab["value"], cls, who)
    return (best[1], best[2], best[3]) if best else (None, None, None)


def _ledger_row(j: Joiner, ws: str, r: dict[str, Any]) -> Optional[RoutingExample]:
    b, prov = r.get("body") or {}, r.get("provenance") or {}
    if prov.get("consent") != "train":
        j.out["no train consent (the decision)"] += 1
        return None
    draft_id = str(b.get("draft") or (b.get("subject") or {}).get("key") or "")
    at = _at(b.get("at"))
    draft = j.drafts.get((ws, draft_id))
    more = []
    text, no_text, problem = None, None, ""
    if draft is None:
        no_text = "the draft is not in the store"
    elif draft["prov"].get("consent") != "train" or draft["train_off"]:
        no_text = "no consent to train on the person's words"
    else:
        problem = draft["problem"]
        said = [problem] if problem else []
        for when, words in draft["said"]:
            moment = _at(when)
            # what was said by then: a message's time is whole seconds, so one in the decision's own second is not counted
            if (at is None or moment is None or moment + 1 <= at) and norm(words) not in {norm(s) for s in said}:
                said.append(words)
        text = "\n".join(s for s in said if s.strip()) or None
        no_text = None if text else "the draft holds no words of the person"
        more.append(Provenance(**{k: v for k, v in draft["prov"].items() if k in Provenance.model_fields}))
    if text is None and no_text is None:
        no_text = "the draft holds no words of the person"
    key = text_key(problem) if problem else None
    point = str(b.get("point"))
    readings: list[RoutingReading] = []
    seen: list[dict[str, Any]] = []
    model_seen = False
    for a in b.get("answers") or []:
        if not isinstance(a, dict):
            continue
        by, answerer, value = a.get("by"), str(a.get("answerer") or "-")[:120], a.get("value")
        used = b.get("used") == by
        if by == "rule":
            readings.append(RoutingReading(by="rule", answerer=answerer, value=value, used=used))
        elif by == "model_role":
            model_seen = True
            request = a.get("source") if str(a.get("source") or "").startswith("mr") else None
            x = j.exchange(ws, draft_id, answerer, at, request)
            if x is not None:
                seen.append(x)
                more.append(Provenance(**{k: v for k, v in x["prov"].items() if k in Provenance.model_fields}))
            readings.append(_model_reading(x, answerer, value, used=used, confidence=a.get("confidence"),
                                           why_none="no exchange of this purpose for this draft is in the store"))
        elif by == "classifier":
            readings.append(RoutingReading(by="own_model", answerer=answerer, value=value, confidence=a.get("confidence"), shadow=True))
        elif by == "person":
            readings.append(RoutingReading(by="person", answerer=answerer, value=value, used=used))
    if point == "request.intent" and not model_seen:  # the model read it beside the rules (a tie, an unclear reading): kept, not used
        x = j.exchange(ws, draft_id, "home_intent", at, None)
        if x is not None:
            seen.append(x)
            more.append(Provenance(**{k: v for k, v in x["prov"].items() if k in Provenance.model_fields}))
            readings.append(_model_reading(x, "home_intent", _reply_track(x.get("reply")), used=False, why_none=""))
    if text is None:  # the thinking summary reads (and may quote) the person's words: never kept where the words may not be
        readings = [r_.model_copy(update={"thinking": None, "thinking_source": None,
                                          "no_thinking": "the person's words may not be trained on, and the thinking reads them"})
                    if r_.thinking else r_ for r_ in readings]
    value, cls, label_by = _label(j, str(b.get("id")), ws)
    truth = j.truth.get(key) if key else None
    origin = "scripted" if ws in j.scripted else (truth["origin"] if truth else "real")
    if value is None and truth is not None and point == "request.intent" and key not in j.flagged:
        value, cls, label_by = truth["label"], "written", truth["by"]
    claude = j.claude.get(f"{MODEL_TYPE}:{ws}:{b.get('id')}")
    if value is None and claude is not None and str(claude["value"]) in ([str(o) for o in b.get("options") or []] or [str(claude["value"])]):
        # the owner (2026-10-29): with no person's label, Claude's blind reading of the words is the label: silver, apart, half weight
        value, cls, label_by = claude["value"], "silver", f"model:{claude.get('model') or 'claude'}:routing_labeler"
    if value is None and origin == "real":  # the owner's rule: silver only on real requests, apart, weighted lower
        model = next((x for x in readings if x.by == "model" and x.value is not None and x.value in (b.get("options") or [x.value])), None)
        if model is not None:
            value, cls, label_by = model.value, "silver", f"model:{model.answerer}"
    exam = j.exams.get(key) if key else None
    for x in seen:  # the words the model read, where the draft gives none: an exam's request is an exam's wherever it is read
        exam = exam or j.exams.get(text_key(x.get("user")))
    group = f"request:{key}" if key else f"draft:{ws}:{draft_id}"
    split = "sealed" if exam else split_of(group)  # an exam's request is sealed everywhere, run or not (one split per request)
    try:
        return RoutingExample(id=f"{MODEL_TYPE}:{ws}:{b.get('id')}", split=split, provenance=Provenance(**{k: v for k, v in prov.items() if k in Provenance.model_fields}),
                              more=more, weight=WEIGHTS.get(cls or "", 1.0) if cls else 1.0, point=point, origin=origin, split_group=group,
                              text=text, no_text=no_text, language=language(text), state=dict(b.get("state") or {}),
                              options=[str(o) for o in b.get("options") or []], readings=readings, value_used=b.get("value_used"),
                              used_by=b.get("used"), outcome=j.outcomes.get((ws, draft_id)) if point.startswith("request.") else None,
                              label=value, label_source=cls, label_by=label_by, exam=exam[0] if exam else None)
    except Exception as exc:  # noqa: BLE001 - a record the contract refuses (a value in its state) never enters a set
        j.out[f"refused by its contract: {type(exc).__name__}"] += 1
        return None


def _written_rows(j: Joiner) -> list[RoutingExample]:
    """P6's written requests (the kept file only; the checker's flagged ones never) and the dev set's, with the owner's blind
    answers as real labels where given."""
    rows: list[RoutingExample] = []
    kept = _kept(j.layout, "home_requests/")
    blind: dict[str, str] = {}
    for source, (doc, _) in kept.items():
        if source.endswith("_blind_answers.json"):
            blind.update({str(a.get("id")): a["track"] for a in doc if isinstance(a, dict) and a.get("track") in TRACKS})
        elif source.endswith("_flagged.json"):
            j.flagged |= {text_key(r.get("text")) for r in doc if isinstance(r, dict) and r.get("text")}
    for source, (doc, entry) in kept.items():
        name = source.split("/", 1)[1].rsplit(".", 1)[0]
        if source.endswith("requests_dev.json"):
            for case in doc if isinstance(doc, list) else []:
                tracks = [t for t in ((case.get("gold") or {}).get("tracks") or []) if t in TRACKS]
                if case.get("text") and tracks:
                    j.truth.setdefault(text_key(case["text"]), {"origin": "eval", "label": tracks[0], "by": "eval:requests_dev", "case": case.get("id")})
            continue
        if not re.fullmatch(r"intent_requests_v\d+", name) or not isinstance(doc, list):
            continue  # the manifest, the flagged file, the blind sample: read above or not rows
        for r in doc:
            if not isinstance(r, dict) or not r.get("text") or r.get("track") not in TRACKS:
                continue
            key = text_key(r["text"])
            if key in j.flagged:
                j.out["a written request the checker flagged"] += 1
                continue
            by = f"written:{name}:{r.get('writer') or '-'}"
            j.truth.setdefault(key, {"origin": "written", "label": r["track"], "by": by})
            label, cls, label_by = r["track"], "written", by
            if str(r.get("id")) in blind:
                label, cls, label_by = blind[str(r["id"])], "real", "person:blind_labelling"
            plain = (r.get("plain_reading") or [None])[0] if isinstance(r.get("plain_reading"), list) else None
            readings = []
            if r.get("plain_reading") is not None:
                x = j.by_text(r["text"], ("home_agent",), None, 0)
                readings.append(_model_reading(x, "home_eval_plain", plain if plain in TRACKS else None, used=False,
                                               why_none="the checker's exchange is not in the store yet"))
            exam = j.exams.get(key)
            group = f"request:{key}"
            try:
                rows.append(RoutingExample(id=f"{MODEL_TYPE}:written:{r.get('id') or key}", split="sealed" if exam else split_of(group),
                                           provenance=_entry_prov(entry), weight=WEIGHTS[cls], point="request.intent", origin="written",
                                           split_group=group, text=str(r["text"]), language=r.get("lang") if r.get("lang") in ("en", "fa", "mixed") else language(r["text"]),
                                           state={"has_data": False, "words": len(str(r["text"]).split())}, options=list(TRACKS), readings=readings,
                                           label=label, label_source=cls, label_by=label_by, exam=exam[0] if exam else None))
            except Exception as exc:  # noqa: BLE001
                j.out[f"refused by its contract: {type(exc).__name__}"] += 1
    return rows


ARMS = {"rules": ("rule", "rule:home_eval"), "model": ("model", "home_agent"), "plain": ("model", "home_eval_plain")}


def _eval_rows(j: Joiner) -> list[RoutingExample]:
    """Every Home eval run's rows: each arm's route as a reading, the request's gold as a written label. A sealed run's rows hold
    no words (its shape only) and go to the sealed split."""
    rows: list[RoutingExample] = []
    dev = {str(c.get("id")): c for c in _kept(j.layout, "home_requests/").get("home_requests/requests_dev.json", ([], None))[0] or []
           if isinstance(c, dict)}
    for source, (doc, entry) in _kept(j.layout, "home_eval/").items():
        if not isinstance(doc, dict) or not doc.get("rows"):
            continue
        split_name, run, run_at = str(doc.get("split") or ""), str(doc.get("id") or source), _at(doc.get("at"))
        sealed = "sealed" in split_name or bool(doc.get("shape_only"))
        gold: dict[str, list[str]] = {}
        if sealed:  # the exam's truth only (its labels, never its words), from the repository's file
            path = j.home / f"requests_{split_name}.json"
            try:
                gold = {str(c.get("id")): (c.get("gold") or {}).get("tracks") or [] for c in json.loads(path.read_text(encoding="utf-8"))}
            except (OSError, ValueError):
                gold = {}
        for row in doc["rows"]:
            case = str(row.get("case"))
            text = None if sealed else (dev.get(case) or {}).get("text")
            tracks = gold.get(case) if sealed else (dev.get(case) or {}).get("gold", {}).get("tracks")
            label = next((t for t in tracks or [] if t in TRACKS), None)
            readings = []
            for arm, observed in (row.get("observed") or {}).items():
                if arm not in ARMS or not isinstance(observed, dict):
                    continue
                by, answerer = ARMS[arm]
                value = observed.get("track") if observed.get("track") in TRACKS else None
                if by == "rule":
                    readings.append(RoutingReading(by="rule", answerer=answerer, value=value))
                else:
                    readings.append(_model_reading(None, answerer, value, used=False,
                                                   why_none="an exam's call keeps its shape only" if sealed else
                                                   "an eval run's calls are kept in its live test workspace; its Home arm's reading, with "
                                                   "its thinking, is that workspace's own request.intent row"))
            key = text_key(text) if text else None
            group = f"request:{key}" if key else f"exam:{split_name}:{case}"
            try:
                rows.append(RoutingExample(id=f"{MODEL_TYPE}:eval:{run}:{case}", split="sealed" if sealed else split_of(group),
                                           provenance=_entry_prov(entry), weight=WEIGHTS["written"] if label else 1.0, point="request.intent",
                                           origin="eval", split_group=group, text=text,
                                           no_text=None if text else ("an exam's request: its shape only" if sealed else "the case is no longer in the dev set"),
                                           language=row.get("lang") if row.get("lang") in ("en", "fa", "mixed") else None,
                                           state={"has_data": bool((dev.get(case) or {}).get("data"))} if not sealed else {},
                                           options=list(TRACKS), readings=readings, label=label, label_source="written" if label else None,
                                           label_by=f"eval:requests_{split_name}" if label else None, exam=f"home:{split_name}" if sealed else None))
            except Exception as exc:  # noqa: BLE001
                j.out[f"refused by its contract: {type(exc).__name__}"] += 1
    return rows


def collect(layout: Optional[Layout] = None, scripted: Optional[frozenset[str]] = None, home: Optional[Path] = None,
            results: Optional[Path] = None) -> tuple[list[RoutingExample], Counter]:
    """Every routing example the store's records give today, and what was kept out (with why)."""
    from ..keepers.workspaces import scripted_names

    layout = layout or Layout()
    j = Joiner(layout, scripted_names() if scripted is None else scripted, home, results)
    rows = _written_rows(j)  # first: the written truths the ledger rows' words are matched against
    rows += _eval_rows(j)
    for (ws, _), r in sorted(j.decisions.items()):
        ex = _ledger_row(j, ws, r)
        if ex is not None:
            rows.append(ex)
    return rows, j.out


def build(layout: Optional[Layout] = None, scripted: Optional[frozenset[str]] = None, home: Optional[Path] = None,
          results: Optional[Path] = None) -> dict[str, Any]:
    """The m7_routing sets, one per routing point, rebuilt from the records (the sealed partition written once)."""
    layout = layout or Layout()
    rows, out = collect(layout, scripted, home, results)
    by_point: dict[str, list[RoutingExample]] = defaultdict(list)
    for ex in rows:
        by_point[ex.point].append(ex)
    report: dict[str, Any] = {"sets": {}, "out_of_sets": dict(out), "rows": len(rows)}
    for point, examples in sorted(by_point.items()):
        sources = {(e.provenance.entry, e.provenance.sha256): {"entry": e.provenance.entry, "sha256": e.provenance.sha256} for e in examples}
        m = write_set(layout, MODEL_TYPE, point, examples, sources=sources.values(), about=ABOUT,
                      decontaminated=("the store's own sealed partition", "Home's exams (requests_sealed*): sealed until run, dev after"))
        report["sets"][f"{MODEL_TYPE}/{point}"] = m.counts
    return report


def counts(layout: Optional[Layout] = None) -> dict[str, Any]:
    """What the latest m7_routing sets hold, per routing point: rows by label class and origin, with words, with a model's thinking
    summary, with our own model's shadow answer (train and dev read; the sealed split counted from its manifest, never read)."""
    layout = layout or Layout()
    base = layout.sets(MODEL_TYPE)
    out: dict[str, Any] = {}
    for manifest in sorted(base.rglob("MANIFEST.json")) if base.exists() else []:
        folder = manifest.parent
        point = folder.parent.name
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        row = out.setdefault(point, {"rows": 0, "sealed_not_read": int((doc.get("counts") or {}).get("sealed") or 0), "label": Counter(),
                                     "origin": Counter(), "text": 0, "thinking": 0, "model_readings": 0, "own_model": 0, "fa": 0, "en": 0, "mixed": 0})
        for split in ("train", "dev"):
            path = folder / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
                ex = json.loads(line)
                row["rows"] += 1
                row["label"][ex.get("label_source") or "none"] += 1
                row["origin"][ex.get("origin") or "-"] += 1
                row["text"] += bool(ex.get("text"))
                models = [r for r in ex.get("readings") or [] if r.get("by") == "model"]
                row["model_readings"] += len(models)
                row["thinking"] += sum(bool(r.get("thinking")) for r in models)
                row["own_model"] += any(r.get("by") == "own_model" for r in ex.get("readings") or [])
                if ex.get("language") in ("fa", "en", "mixed"):
                    row[ex["language"]] += 1
    return {p: {**v, "label": dict(v["label"]), "origin": dict(v["origin"])} for p, v in out.items()}


def _rows_of(layout: Layout) -> Iterator[tuple[str, dict[str, Any]]]:
    base = layout.sets(MODEL_TYPE)
    for point_dir in sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []:
        versions = sorted((p for p in point_dir.iterdir() if p.is_dir() and re.fullmatch(r"v\d+", p.name)), key=lambda p: int(p.name[1:]))
        for split in ("train", "dev", "sealed") if versions else ():
            path = versions[-1] / f"{split}.jsonl"
            for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else []:
                yield split, json.loads(line)


def chat_row(ex: dict[str, Any]) -> dict[str, Any]:
    """One labelled row as a chat example: the task and its options, the person's words, the label as the answer. The answer's
    reasoning is the API thinking summary of the model reading that gave this same answer (recorded), else none, said why."""
    from dclab_core.routing_example import TRACK_MEANINGS

    options = [str(o) for o in ex.get("options") or []]
    if ex.get("point") == "request.intent":
        task = "Which kind of help does this request need? " + "; ".join(f"{k}: {TRACK_MEANINGS[k]}" for k in options if k in TRACK_MEANINGS)
    else:
        task = f"Home's next routing step ({ex.get('point')}): choose one of {', '.join(options)}."
    said = "; ".join(f"{k}: {v}" for k, v in sorted((ex.get("state") or {}).items()) if v is not None)
    user = str(ex.get("text") or "") + (f"\n[{said}]" if said and ex.get("point") != "request.intent" else "")
    reading = next((r for r in ex.get("readings") or [] if r.get("by") == "model" and r.get("value") == ex.get("label") and r.get("thinking")), None)
    answer = {"role": "assistant", "content": str(ex["label"])}
    answer.update({"reasoning": reading["thinking"], "reasoning_source": "recorded"} if reading else
                  {"no_reasoning": "no model reading with this answer kept its API thinking summary"})
    return {"id": ex["id"], "point": ex.get("point"), "label_source": ex.get("label_source"), "label_by": ex.get("label_by"),
            "weight": ex.get("weight"), "origin": ex.get("origin"), "language": ex.get("language"), "split_group": ex.get("split_group"),
            "messages": [{"role": "system", "content": task + " Answer with the option only."}, {"role": "user", "content": user}, answer]}


TABLE = ("id", "split", "point", "origin", "language", "text", "label", "label_source", "label_by", "weight", "value_used", "used_by",
         "outcome", "rule", "model", "model_thinking", "own_model", "split_group")


def export(layout: Optional[Layout] = None, out: Optional[Path] = None, version: str = "v1") -> dict[str, Any]:
    """The m7_routing sets as chat JSONL (``chat/<point>/<class>/<split>.jsonl``, each label class apart) and as one table
    (``table.csv``, every row with its words, its readings by answerer and its label class). An exam's request is never written
    here; a sealed row only in a sealed file. Rebuilt whole."""
    import csv
    import shutil

    from ..layout import ROOT

    layout = layout or Layout()
    root = Path(out or ROOT / "training_records" / "datasets" / "routing") / version
    shutil.rmtree(root, ignore_errors=True)
    files: dict[Path, list[str]] = defaultdict(list)
    table: list[dict[str, Any]] = []
    skipped: Counter = Counter()
    for split, ex in _rows_of(layout):
        if ex.get("exam"):
            skipped["an exam's request"] += 1
            continue
        if not ex.get("text"):
            skipped["no words (no consent, or shape only)"] += 1
            continue
        readings = ex.get("readings") or []
        first = lambda by: next((r for r in readings if r.get("by") == by), {})  # noqa: E731
        model = first("model")
        table.append({"id": ex["id"], "split": split, "point": ex.get("point"), "origin": ex.get("origin"), "language": ex.get("language"),
                      "text": ex["text"], "label": ex.get("label"), "label_source": ex.get("label_source"), "label_by": ex.get("label_by"),
                      "weight": ex.get("weight"), "value_used": ex.get("value_used"), "used_by": ex.get("used_by"), "outcome": ex.get("outcome"),
                      "rule": first("rule").get("value"), "model": model.get("value"), "model_thinking": model.get("thinking"),
                      "own_model": first("own_model").get("value"), "split_group": ex.get("split_group")})
        if ex.get("label") is not None and ex.get("label_source"):
            files[root / "chat" / str(ex.get("point")) / str(ex["label_source"]) / f"{split}.jsonl"].append(json.dumps(chat_row(ex), ensure_ascii=False))
    for path, lines in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "table.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=TABLE)
        w.writeheader()
        w.writerows(table)
    manifest = {"model_type": MODEL_TYPE, "version": version, "rows": len(table), "skipped": dict(skipped),
                "chat": {str(p.relative_to(root)): len(v) for p, v in sorted(files.items())},
                "rule": "train may train, dev validates, sealed is evaluation only. Label classes stay apart: real (a person's), written (a "
                        "writer's, an eval's or a script's truth, weight 0.5), silver (Claude's reading of an unconfirmed real request, "
                        "weight 0.5). A reasoning is an API thinking summary (recorded), never written afterwards. An exam's request is "
                        "never here."}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store.sets.routing", description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="the store's root (DCLAB_STORE, else training_records/store)")
    parser.add_argument("--counts", action="store_true", help="only print what the sets hold now")
    parser.add_argument("--export", action="store_true", help="write the chat JSONL and the table (training_records/datasets/routing/v1)")
    args = parser.parse_args(argv)
    layout = Layout(args.root)
    if args.export:
        doc = export(layout)
        print(json.dumps({k: doc[k] for k in ("rows", "skipped", "chat")}, indent=1))
        return 0
    print(json.dumps(counts(layout) if args.counts else build(layout), indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["POINTS", "MODEL_TYPE", "routing_point", "collect", "build", "counts", "export", "chat_row", "text_key", "exams"]
