"""The frozen exams of the store (package 36.4): drawn once per model type, pinned, never trained on, sat by every version.

A model trained again can forget what it knew. Each model type gets one exam (:class:`dclab_core.exam.Exam`), drawn once
from its sets' **sealed** rows, so by the split rule none of its rows is in a train or dev split:

- one section per set of the type (its folder: a decision point, a stage, a kind of trajectory), the latest version of
  each; sets with fewer than ``MIN_SECTION`` items are pooled into ``small_sets``; at most ``MAX_ITEMS`` per section,
  chosen by a salted hash of the row's key (never by its answer);
- for the language models (``GENERAL_FOR``), general skills, from the sealed rows of the *other* model types (a skill the
  type's own sets may not hold): ``tool_calling`` (a trajectory's tool step), ``code`` (a step that writes code),
  ``persian`` (a turn in Persian) and ``refusals`` (an answer that refuses, a move the validator refused). A section
  with no sealed row today is kept empty with its reason: the gate names it unprotected.

An item's expected answer is the row's label, else its rule's answer, else what the agent recorded doing (each named in
``expect_source``). For the no-forgetting gate what matters is the change between versions on the same items.

**Pinned.** ``exam_pins.json`` (in this repository) holds each exam's hash; an exam whose file does not match its pin is
refused, and an exam is never drawn twice. **Never trained on.** :func:`verify` refuses an exam when any of its rows (by
key, or by content: a copy kept again) is in any train or dev split of any set; :func:`guard` is the same check that
``store.sets.write_set`` applies to every set it writes, beside ``llm.Contamination`` (the sealed people's words), so an
exam row is held out of every training set. **Sat by every version.** :func:`sit` scores a version's answers and keeps
its :class:`dclab_core.exam.ExamResult` with the version (``results/<version>.json``); :func:`accept` applies the
no-forgetting gate against the version accepted before it and records the verdict in ``versions.json``; ``models.tuned``
serves a tuned model only once it is accepted (:func:`standing`).

    python -m dclab_rnd.store.exams draw [--type m4_workflow_llm] [--yes] | status | verify
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

from dclab_core import exam as core
from dclab_core.store import MODEL_TYPES

from .layout import Layout

EXAM_VERSION = "exam_v1"
MAX_ITEMS = 200
MIN_SECTION = 10
SALT = "dclab-exam-v1"
GENERAL_FOR = ("m4_workflow_llm", "m5_solver_llm")
PINS = Path(__file__).resolve().parent / "exam_pins.json"
CODE_TOOLS = ("write", "edit", "notebookedit", "multiedit", "create_file", "write_file", "run_python", "python")
_PERSIAN = re.compile(r"[؀-ۿ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)
REFUSAL = re.compile(r"\b(i can(?:no|')t|i won't|i will not|i refuse|refused|not allowed|cannot do that)\b|نمی‌توانم|نمیتوانم", re.I)
_NOT_CONTENT = ("id", "split", "provenance", "more", "weight", "model_type")
RULE = (f"{EXAM_VERSION}: every set's sealed rows (the latest version of each folder), one section per folder (below {MIN_SECTION} items "
        f"pooled as small_sets), at most {MAX_ITEMS} per section by sha256('{SALT}' + key); general sections (tool_calling, code, "
        "persian, refusals) from the other model types' sealed rows for the language models; expected: the label, else the rule's "
        "answer, else what the agent recorded; each row once.")


class ExamError(ValueError):
    """An exam that may not be used: changed after its pin, contaminated, or its row gone."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------ a row's identity
def row_key(ex: Mapping[str, Any]) -> str:
    """The row's key: its raw entry and its place there (``#n``), the same in every model type's set."""
    ident = str(ex.get("id") or "")
    if "#" not in ident:  # a trajectory, a pair, a team walk: the whole id (without its model type) is the row, never its entry alone
        mt = str(ex.get("model_type") or "")
        return ident[len(mt) + 1:] if mt and ident.startswith(mt + ":") else ident
    place = ident.rsplit("#", 1)[1]
    entry = str((ex.get("provenance") or {}).get("entry") or ident)
    return f"{entry}#{place}"


def content_hash(ex: Mapping[str, Any]) -> str:
    body = {k: v for k, v in ex.items() if k not in _NOT_CONTENT}
    return _sha(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str))


def group_of(ex: Mapping[str, Any]) -> str:
    p = ex.get("provenance") or {}
    if ex.get("split_group"):
        return str(ex["split_group"])
    if p.get("project"):
        return f"project:{p['project']}"
    if p.get("draft"):
        return f"draft:{p['draft']}"
    return f"entry:{p.get('entry') or ex.get('id')}"


def _order(key: str) -> str:
    return _sha(SALT + key)


# ------------------------------------------------------------------------------------------------ the sets
def sealed_sets(layout: Layout, model_type: str) -> list[tuple[str, str, Path]]:
    """(folder, version, sealed file) of the latest version of every set of the type."""
    base = layout.sets(model_type)
    latest: dict[str, tuple[int, str, Path]] = {}
    for path in sorted(base.rglob("sealed.jsonl")) if base.is_dir() else []:
        version = path.parent.name
        folder = path.parent.parent.relative_to(base).as_posix()
        n = int(version[1:]) if re.fullmatch(r"v\d+", version) else 0
        if folder not in latest or n > latest[folder][0]:
            latest[folder] = (n, version, path)
    return [(f, v, p) for f, (_, v, p) in sorted(latest.items())]


def _rows(path: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield line, json.loads(line)


def _texts(messages: Iterable[Mapping[str, Any]], role: str) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages if m.get("role") == role)


def _persian(text: str) -> bool:
    letters = _LETTER.findall(text or "")
    return len(letters) >= 10 and len(_PERSIAN.findall(text)) / len(letters) >= 0.5


def _code_step(step: Mapping[str, Any]) -> bool:
    tool = str(step.get("tool") or "").lower()
    return any(t == tool or tool.endswith("." + t) for t in CODE_TOOLS)


# ------------------------------------------------------------------------------------------------ items
def _rule_answer(ex: Mapping[str, Any]) -> Any:
    return next((a.get("value") for a in ex.get("answers") or [] if str(a.get("by", "")).startswith("rule") and a.get("value") is not None), None)


def item_of(ex: dict[str, Any], set_path: str, line: str, skill: Optional[str] = None) -> Optional[core.ExamItem]:
    """One exam item from a sealed row (None: the row has no answer to expect). ``skill``: a general section's skill."""
    key, shape = row_key(ex), str(ex.get("schema_version", "")).split(".")[1] if "." in str(ex.get("schema_version", "")) else ""
    common = {"id": key, "example": str(ex["id"]), "set": set_path, "sha256": _sha(line), "content": content_hash(ex), "group": group_of(ex)}
    pick = int(_order(key)[:8], 16)
    if shape == "decision_example":
        if ex.get("label") is not None:
            return core.ExamItem(**common, check="label", expect=ex["label"], expect_source=f"label:{ex.get('label_source')}")
        rule = _rule_answer(ex)
        return core.ExamItem(**common, check="label", expect=rule, expect_source="rule") if rule is not None else None
    if shape == "episode":
        steps = ex.get("steps") or []
        if skill == "refusals":
            refused = [s for s in steps if "refus" in str(s.get("outcome") or "").lower()]
            return core.ExamItem(**common, check="label", expect=refused[0]["outcome"], expect_source="recorded", step=refused[0]["n"]) if refused else None
        step = steps[pick % len(steps)] if steps else None
        return core.ExamItem(**common, check="label", expect=step["action"], expect_source="recorded", step=step["n"]) if step else None
    if shape == "solver_trajectory":
        steps = [s for s in ex.get("steps") or [] if s.get("tool")]
        if skill == "code":
            steps = [s for s in steps if _code_step(s)]
            return core.ExamItem(**common, check="code", expect=steps[0]["tool"], expect_source="recorded", step=steps[0]["n"]) if steps else None
        step = steps[pick % len(steps)] if steps else None
        return core.ExamItem(**common, check="tool_name", expect=step["tool"], expect_source="recorded", step=step["n"]) if step else None
    if shape == "chat_example":
        msgs = ex.get("messages") or []
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        if last is None:
            return None
        m = msgs[last]
        if skill == "persian":
            return core.ExamItem(**common, check="persian_reply", expect=True, expect_source="recorded", step=last)
        if skill == "refusals":
            return core.ExamItem(**common, check="refusal", expect=True, expect_source="recorded", step=last)
        if m.get("tool_calls"):
            name = (m["tool_calls"][0].get("function") or {}).get("name") or m["tool_calls"][0].get("name")
            return core.ExamItem(**common, check="tool_name", expect=name, expect_source="recorded", step=last) if name else None
        text = str(m.get("content") or "")[:4000]
        return core.ExamItem(**common, check="text_f1", expect=text, expect_source="recorded", step=last) if text.strip() else None
    if shape == "preference_pair":
        swap = pick % 2 == 1
        return core.ExamItem(**common, check="choice", expect="B" if swap else "A", expect_source=f"label:{ex.get('decided_by')}", swap=swap)
    if shape == "judged_example":
        if ex.get("preferred"):
            swap = pick % 2 == 1
            first = "candidate" if not swap else "other"
            return core.ExamItem(**common, check="choice", expect="A" if ex["preferred"] == first else "B",
                                 expect_source=f"label:{ex.get('evidence')}", swap=swap)
        if ex.get("score") is not None:
            return core.ExamItem(**common, check="score_close", expect=float(ex["score"]), expect_source=f"label:{ex.get('evidence')}")
        return None
    if shape == "retrieval_example":
        used = sorted(set(ex.get("cited") or ex.get("used") or []))
        return core.ExamItem(**common, check="set_f1", expect=used, expect_source="recorded") if used else None
    if shape == "experiment_record":
        return core.ExamItem(**common, check="label", expect=bool(ex.get("selected")), expect_source="measured")
    return None


def skill_of(ex: Mapping[str, Any]) -> Optional[str]:
    """The general skill a sealed row of another model type tests, or None."""
    shape = str(ex.get("schema_version", "")).split(".")[1] if "." in str(ex.get("schema_version", "")) else ""
    if shape == "chat_example":
        msgs = ex.get("messages") or []
        if _persian(_texts(msgs, "user")):
            return "persian"
        answers = [m for m in msgs if m.get("role") == "assistant"]
        if answers and REFUSAL.search(str(answers[-1].get("content") or "")):
            return "refusals"
        return "tool_calling" if answers and answers[-1].get("tool_calls") else None
    if shape == "solver_trajectory":
        steps = [s for s in ex.get("steps") or [] if s.get("tool")]
        if any(_code_step(s) for s in steps):
            return "code"
        return "tool_calling" if steps else None
    if shape == "episode":
        return "refusals" if any("refus" in str(s.get("outcome") or "").lower() for s in ex.get("steps") or []) else None
    return None


GENERAL_NOTES = {"tool_calling": "a tool step of another model type's sealed trajectory or chat turn",
                 "code": "a step that writes code, of another model type's sealed trajectory",
                 "persian": "a sealed chat whose person writes in Persian",
                 "refusals": "a sealed answer that refuses, or a move the validator refused"}


def _cap(items: list[core.ExamItem]) -> list[core.ExamItem]:
    return sorted(sorted(items, key=lambda i: _order(i.id))[:MAX_ITEMS], key=lambda i: i.id)


def _trained(layout: Layout) -> tuple[set[str], set[str]]:
    """The keys and contents of every train and dev row of every set: a row there is never an exam's."""
    keys: set[str] = set()
    contents: set[str] = set()
    for mt in sorted(MODEL_TYPES):
        base = layout.sets(mt)
        for split in ("train", "dev"):
            for path in sorted(base.rglob(f"{split}.jsonl")) if base.is_dir() else []:
                for _, ex in _rows(path):
                    keys.add(row_key(ex))
                    contents.add(content_hash(ex))
    return keys, contents


def draft(layout: Layout, model_type: str, version: str = EXAM_VERSION) -> core.Exam:
    """The exam the store's sealed rows give today (not written: :func:`draw` writes it once). A sealed row whose key or
    content is also in a train or dev split is skipped, and one item is kept per content (copies are not independent)."""
    trained_keys, trained_contents = _trained(layout)
    seen: set[str] = set()
    seen_content: set[str] = set()

    def fresh(item: Optional[core.ExamItem]) -> bool:
        if item is None or item.id in seen or item.content in seen_content or item.id in trained_keys or item.content in trained_contents:
            return False
        seen.add(item.id)
        seen_content.add(item.content)
        return True

    own: dict[str, list[core.ExamItem]] = {}
    for folder, ver, path in sealed_sets(layout, model_type):
        for line, ex in _rows(path):
            item = item_of(ex, f"{model_type}/{folder}/{ver}", line)
            if fresh(item):
                own.setdefault(folder, []).append(item)
    sections: list[core.ExamSection] = []
    small: list[core.ExamItem] = []
    for folder, items in sorted(own.items()):
        if len(items) < MIN_SECTION:
            small += items
        else:
            sections.append(core.ExamSection(name=folder, items=_cap(items), note=f"sealed rows of {model_type}/{folder}"))
    if small:
        sections.append(core.ExamSection(name="small_sets", items=_cap(small), note=f"the sets with fewer than {MIN_SECTION} items, pooled"))
    if model_type in GENERAL_FOR:
        general: dict[str, list[core.ExamItem]] = {s: [] for s in core.GENERAL}
        for other in sorted(MODEL_TYPES):
            if other == model_type:
                continue
            for folder, ver, path in sealed_sets(layout, other):
                for line, ex in _rows(path):
                    skill = skill_of(ex)
                    if skill is None:
                        continue
                    item = item_of(ex, f"{other}/{folder}/{ver}", line, skill)
                    if fresh(item):
                        general[skill].append(item)
        for skill in core.GENERAL:
            items = _cap(general[skill])
            note = GENERAL_NOTES[skill] if items else f"empty: no sealed row yet ({GENERAL_NOTES[skill]}); a later exam version may add it"
            sections.append(core.ExamSection(name=f"general:{skill}", general=True, items=items, note=note))
    if not sections:
        raise ExamError(f"{model_type} has no sealed row with an answer to expect: no exam can be drawn")
    return core.build(model_type, version, RULE, sections, _now())


# ------------------------------------------------------------------------------------------------ pins and files
def _pins(path: Optional[Path] = None) -> dict[str, dict[str, str]]:
    path = path or PINS
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def exam_file(layout: Layout, model_type: str, version: str = EXAM_VERSION) -> Path:
    return layout.exams(model_type, version, "exam.json")


def load(layout: Layout, model_type: str, version: str = EXAM_VERSION, pins: Optional[Path] = None) -> Optional[core.Exam]:
    """The exam, refused when it was changed (its hash) or does not match its pin; None when none was drawn."""
    path = exam_file(layout, model_type, version)
    pin = _pins(pins).get(model_type, {}).get(version)
    if not path.is_file():
        if pin:
            raise ExamError(f"{model_type} {version} is pinned ({pin[:12]}) but its file is gone: restore it from the store's history")
        return None
    try:
        exam = core.Exam.read(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ExamError(f"{model_type} {version}: {e}") from None
    if pin and exam.hash != pin:
        raise ExamError(f"{model_type} {version}: its hash {exam.hash[:12]} is not its pin {pin[:12]}: the exam was changed")
    return exam


def all_exams(layout: Layout, pins: Optional[Path] = None) -> list[core.Exam]:
    out = []
    base = layout.exams_root()
    for model_type in sorted(MODEL_TYPES):
        folder = base / model_type
        for vdir in sorted(folder.iterdir()) if folder.is_dir() else []:
            exam = load(layout, model_type, vdir.name, pins) if (vdir / "exam.json").is_file() else None
            if exam is not None:
                out.append(exam)
    return out


# ------------------------------------------------------------------------------------------------ never trained on
class ExamGuard:
    """Every frozen exam's rows by key and by content: a train or dev example matching one is held out."""

    def __init__(self, exams: Iterable[core.Exam]):
        self.keys: dict[str, str] = {}
        self.contents: dict[str, str] = {}
        for e in exams:
            for i in e.items():
                self.keys[i.id] = self.contents[i.content] = f"{e.model_type} {e.version}"

    def __bool__(self) -> bool:
        return bool(self.keys)

    def why(self, example: Any) -> Optional[str]:
        ex = example if isinstance(example, Mapping) else json.loads(example.model_dump_json())
        hit = self.keys.get(row_key(ex)) or self.contents.get(content_hash(ex))
        return f"a frozen exam's row ({hit})" if hit else None


_GUARDS: dict[tuple[str, tuple[tuple[str, float], ...]], ExamGuard] = {}


def guard(layout: Layout, pins: Optional[Path] = None) -> ExamGuard:
    """The guard of every exam in the store (cached while no exam file changes)."""
    base = layout.exams_root()
    files = tuple(sorted((p.as_posix(), p.stat().st_mtime) for p in base.rglob("exam.json"))) if base.is_dir() else ()
    key = (str(layout.root.resolve()), files)
    if key not in _GUARDS:
        _GUARDS.clear()
        _GUARDS[key] = ExamGuard(all_exams(layout, pins))
    return _GUARDS[key]


def verify(layout: Layout, exam: core.Exam) -> dict[str, Any]:
    """Refuse the exam when any of its rows is in a train or dev split of any set, or no longer in its sealed file."""
    g = ExamGuard([exam])
    found: list[str] = []
    for model_type in sorted(MODEL_TYPES):
        base = layout.sets(model_type)
        for split in ("train", "dev"):
            for path in sorted(base.rglob(f"{split}.jsonl")) if base.is_dir() else []:
                for _, ex in _rows(path):
                    if g.why(ex):
                        found.append(f"{layout.relative(path)}: {ex.get('id')}")
    if found:
        raise ExamError(f"{exam.model_type} {exam.version} is in a training set ({len(found)} rows; first: {found[0]})")
    by_set: dict[str, list[core.ExamItem]] = {}
    for i in exam.items():
        by_set.setdefault(i.set, []).append(i)
    gone = []
    for set_path, items in sorted(by_set.items()):
        mt, rest = set_path.split("/", 1)
        path = layout.sets(mt, *rest.split("/"), "sealed.jsonl")
        lines = {_sha(line) for line, _ in _rows(path)} if path.is_file() else set()
        gone += [i.id for i in items if i.sha256 not in lines]
    if gone:
        raise ExamError(f"{exam.model_type} {exam.version}: {len(gone)} rows are no longer in their sealed files (first: {gone[0]})")
    return {"exam": exam.version, "model_type": exam.model_type, "hash": exam.hash, "items": len(exam.items()), "clean": True}


def draw(layout: Layout, model_type: str, version: str = EXAM_VERSION, pins: Optional[Path] = None) -> core.Exam:
    """Draw the type's exam once, check it is in no training set, write it and pin its hash. An exam already drawn is
    returned as it is (checked against its pin), never drawn again."""
    pins = pins or PINS
    existing = load(layout, model_type, version, pins)
    if existing is not None:
        return existing
    exam = draft(layout, model_type, version)
    verify(layout, exam)
    path = exam_file(layout, model_type, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(exam.model_dump_json(indent=1) + "\n", encoding="utf-8")
    doc = _pins(pins)
    doc.setdefault(model_type, {})[version] = exam.hash
    pins.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return exam


# ------------------------------------------------------------------------------------------------ sitting it
def _row_of(layout: Layout, item: core.ExamItem) -> dict[str, Any]:
    mt, rest = item.set.split("/", 1)
    for line, ex in _rows(layout.sets(mt, *rest.split("/"), "sealed.jsonl")):
        if _sha(line) == item.sha256:
            return ex
    raise ExamError(f"{item.id}: its row is no longer in {item.set}")


def question(ex: Mapping[str, Any], item: core.ExamItem) -> dict[str, Any]:
    """What the examinee is shown for one item (built from its sealed row; the expected answer is never in it)."""
    q: dict[str, Any] = {"item": item.id, "check": item.check}
    shape = str(ex.get("schema_version", "")).split(".")[1]
    if shape == "decision_example":
        q.update(point=ex.get("point"), state=ex.get("state"), options=ex.get("options"))
    elif shape == "episode":
        steps = ex.get("steps") or []
        q.update(process=ex.get("process"), before=[s for s in steps if s["n"] < item.step],
                 state=next(s["state"] for s in steps if s["n"] == item.step), options=next(s["options"] for s in steps if s["n"] == item.step))
        if item.check == "label" and str(item.expect).lower().startswith("refus"):
            q["action"] = next(s["action"] for s in steps if s["n"] == item.step)  # is this move allowed?
    elif shape == "solver_trajectory":
        q.update(problem_kind=ex.get("problem_kind"), problem=ex.get("problem"), tools=ex.get("tools"),
                 steps=[s for s in ex.get("steps") or [] if s["n"] < item.step])
    elif shape == "chat_example":
        q.update(messages=(ex.get("messages") or [])[: item.step], tools=ex.get("tools"))
    elif shape == "preference_pair":
        a, b = (ex["rejected"], ex["chosen"]) if item.swap else (ex["chosen"], ex["rejected"])
        q.update(prompt=ex.get("prompt"), A=a, B=b)
    elif shape == "judged_example":
        if item.check == "choice":
            a, b = (ex.get("other"), ex.get("candidate")) if item.swap else (ex.get("candidate"), ex.get("other"))
            q.update(judged=ex.get("judged"), A=a, B=b)
        else:
            q.update(judged=ex.get("judged"), candidate=ex.get("candidate"))
    elif shape == "retrieval_example":
        q.update(query=ex.get("query"), offered=ex.get("offered"))
    elif shape == "experiment_record":
        q.update(task=ex.get("task"), table=ex.get("table"), model=ex.get("model"), params=ex.get("params"), metrics=ex.get("metrics"))
    return q


def questions(layout: Layout, exam: core.Exam) -> list[dict[str, Any]]:
    return [question(_row_of(layout, i), i) for i in exam.items()]


def _norm(value: Any) -> str:
    text = str(value).strip().lower()
    return {"true": "yes", "false": "no"}.get(text, text)


def _f1(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return float(a == b)
    common = sum(min(a.count(t), b.count(t)) for t in set(a))
    if not common:
        return 0.0
    p, r = common / len(a), common / len(b)
    return 2 * p * r / (p + r)


def score_answer(item: core.ExamItem, answer: Any) -> Optional[float]:
    """One answer's score in [0, 1] by the item's check (None: no answer)."""
    if answer is None:
        return None
    a = answer if isinstance(answer, Mapping) else {"answer": answer}
    said = a.get("answer", a.get("tool", a.get("text")))
    if item.check == "label":
        return float(_norm(said) == _norm(item.expect))
    if item.check == "tool_name":
        return float(_norm(a.get("tool", said)) == _norm(item.expect))
    if item.check == "code":
        code = str(a.get("code") or "")
        try:
            ast.parse(code)
        except SyntaxError:
            return 0.0
        return float(bool(code.strip()) and _code_step({"tool": a.get("tool", item.expect)}))
    if item.check == "text_f1":
        return round(_f1(re.findall(r"\w+", str(said or "").lower()), re.findall(r"\w+", str(item.expect).lower())), 4)
    if item.check == "choice":
        return float(str(said).strip().upper() == item.expect)
    if item.check == "score_close":
        try:
            return max(0.0, 1.0 - abs(float(said) - float(item.expect)))
        except (TypeError, ValueError):
            return 0.0
    if item.check == "set_f1":
        got = sorted(set(said if isinstance(said, list) else []))
        return round(_f1(got, list(item.expect)), 4)
    if item.check == "persian_reply":
        return float(_persian(str(said or "")))
    if item.check == "refusal":
        return float(bool(REFUSAL.search(str(said or ""))))
    return None


def result_file(layout: Layout, exam: core.Exam, model_version: str) -> Path:
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_version.lower()).strip("-.")[:60] or "model"
    return layout.exams(exam.model_type, exam.version, "results", f"{slug}-{_sha(model_version)[:8]}.json")  # the real name is in the file


def sit(layout: Layout, exam: core.Exam, answers: Mapping[str, Any] | Callable[[dict[str, Any]], Any], model_version: str,
        model: Optional[str] = None) -> core.ExamResult:
    """Score one model version on the exam and keep the result with it. ``answers``: item id -> answer, or a function of
    a question. A version sits an exam once: its result is never rewritten."""
    path = result_file(layout, exam, model_version)
    if path.exists():
        raise ExamError(f"{model_version} already sat {exam.model_type} {exam.version}: its result is kept as it is")
    load(layout, exam.model_type, exam.version)  # the pinned exam, unchanged
    scores: dict[str, Optional[float]] = {}
    for item in exam.items():
        if callable(answers):
            given = answers(question(_row_of(layout, item), item))
        else:
            given = answers.get(item.id)
        scores[item.id] = score_answer(item, given)
    result = core.score(exam, scores, model_version, _now(), model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=1) + "\n", encoding="utf-8")
    return result


def result(layout: Layout, exam: core.Exam, model_version: str) -> Optional[core.ExamResult]:
    path = result_file(layout, exam, model_version)
    return core.ExamResult.read(path.read_text(encoding="utf-8")) if path.is_file() else None


# ------------------------------------------------------------------------------------------------ the gate
def versions_file(layout: Layout, model_type: str, version: str = EXAM_VERSION) -> Path:
    return layout.exams(model_type, version, "versions.json")


def versions(layout: Layout, model_type: str, version: str = EXAM_VERSION) -> list[dict[str, Any]]:
    path = versions_file(layout, model_type, version)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def current(layout: Layout, model_type: str, version: str = EXAM_VERSION) -> Optional[str]:
    """The model version accepted last and not retired since."""
    live = [v for v in versions(layout, model_type, version) if v["status"] in ("accepted", "retired")]
    out: list[str] = []
    for v in live:
        if v["status"] == "accepted":
            out.append(v["model_version"])
        elif v["model_version"] in out:
            out.remove(v["model_version"])
    return out[-1] if out else None


def gate(layout: Layout, model_type: str, model_version: str, previous: Optional[str] = None, version: str = EXAM_VERSION) -> dict[str, Any]:
    """The no-forgetting verdict of ``model_version`` against ``previous`` (default: the version accepted last)."""
    exam = load(layout, model_type, version)
    if exam is None:
        raise ExamError(f"{model_type} has no exam yet: python -m dclab_rnd.store.exams draw --type {model_type}")
    new = result(layout, exam, model_version)
    if new is None:
        raise ExamError(f"{model_version} has not sat {model_type} {version}")
    previous = previous if previous is not None else current(layout, model_type, version)
    before = result(layout, exam, previous) if previous else None
    if previous and before is None:
        raise ExamError(f"{previous} has no result on {model_type} {version}: it cannot be compared")
    return core.no_forgetting(exam, before, new)


def _record(layout: Layout, model_type: str, version: str, entry: dict[str, Any]) -> None:
    path = versions_file(layout, model_type, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([*versions(layout, model_type, version), entry], indent=1) + "\n", encoding="utf-8")


def accept(layout: Layout, model_type: str, model_version: str, reason: str, by: str = "owner", version: str = EXAM_VERSION) -> dict[str, Any]:
    """Accept a version as the type's current one only when the no-forgetting gate passes against the version accepted
    last and against the first one accepted and still standing (small losses never add up version after version); a
    refusal is recorded too."""
    verdict = gate(layout, model_type, model_version, version=version)
    first = _first(layout, model_type, version)
    if verdict["pass"] and first and first not in (verdict["previous"], model_version):
        against_first = gate(layout, model_type, model_version, previous=first, version=version)
        if not against_first["pass"]:
            verdict = {**against_first, "note": f"kept against {verdict['previous']}, but forgot against the first accepted, {first}"}
    exam = load(layout, model_type, version)
    res = result(layout, exam, model_version) if exam else None
    _record(layout, model_type, version, {"model_version": model_version, "model": res.model if res else None, "at": _now(), "by": by,
                                          "reason": reason, "status": "accepted" if verdict["pass"] else "refused",
                                          "previous": verdict["previous"], "gate": verdict["gate"], "forgotten": verdict["forgotten"],
                                          "exam_hash": verdict["exam_hash"]})
    if not verdict["pass"]:
        raise ExamError(f"{model_version} forgets: {', '.join(verdict['forgotten']) or verdict.get('refused')}")
    return verdict


def _first(layout: Layout, model_type: str, version: str) -> Optional[str]:
    """The first version accepted on this exam and not retired since."""
    accepted: list[str] = []
    for v in versions(layout, model_type, version):
        if v["status"] == "accepted" and v["model_version"] not in accepted:
            accepted.append(v["model_version"])
        elif v["status"] == "retired" and v["model_version"] in accepted:
            accepted.remove(v["model_version"])
    return accepted[0] if accepted else None


def retire(layout: Layout, model_type: str, model_version: str, reason: str, by: str = "owner", version: str = EXAM_VERSION) -> Optional[str]:
    """Retire a version; the version accepted before it is current again. Returns the current version after."""
    if model_version not in {v["model_version"] for v in versions(layout, model_type, version) if v["status"] == "accepted"}:
        raise ExamError(f"{model_version} was never accepted on {model_type} {version}")
    _record(layout, model_type, version, {"model_version": model_version, "at": _now(), "by": by, "reason": reason, "status": "retired"})
    return current(layout, model_type, version)


def standing(layout: Layout, model: str) -> Optional[dict[str, Any]]:
    """The last exam decision about the model served as ``model`` (any type, any exam version), or None if it sat none."""
    out = None
    base = layout.exams_root()
    for path in sorted(base.glob("*/*/versions.json")) if base.is_dir() else []:  # an exam's own record, never a result file
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ExamError(f"{path.name} of {path.parent.parent.name} {path.parent.name} is not a list of decisions")
        names = {r["model_version"]: r.get("model") for r in rows if r.get("model")}
        for r in rows:
            if (r.get("model") or names.get(r["model_version"])) == model and (out is None or r["at"] >= out["at"]):
                out = {**r, "model_type": path.parent.parent.name, "exam": path.parent.name}
    return out


# ------------------------------------------------------------------------------------------------ command line
def status(layout: Layout) -> dict[str, Any]:
    out = {}
    for e in all_exams(layout):
        out[e.model_type] = {"version": e.version, "hash": e.hash, "items": len(e.items()), "current": current(layout, e.model_type, e.version),
                             "sections": {s.name: len(s.items) for s in e.sections}}
    return out


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.store.exams", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("draw", "status", "verify"))
    parser.add_argument("--type", dest="model_type", default=None, help="one model type (default: every type)")
    parser.add_argument("--yes", action="store_true", help="draw: write the exams and their pins (without it: what would be drawn)")
    args = parser.parse_args(argv)
    layout = Layout()
    types = [args.model_type] if args.model_type else sorted(MODEL_TYPES)
    try:
        if args.command == "status":
            print(json.dumps(status(layout), indent=1))
        elif args.command == "verify":
            print(json.dumps([verify(layout, e) for e in all_exams(layout) if e.model_type in types], indent=1))
        else:
            out = {}
            for mt in types:
                try:
                    exam = draw(layout, mt) if args.yes else (load(layout, mt) or draft(layout, mt))
                except ExamError as e:
                    out[mt] = {"refused": str(e)}
                    continue
                out[mt] = {"hash": exam.hash, "items": len(exam.items()), "written": args.yes,
                           "sections": {s.name: len(s.items) for s in exam.sections}}
            print(json.dumps(out, indent=1))
    except ExamError as e:
        print(f"refused: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
