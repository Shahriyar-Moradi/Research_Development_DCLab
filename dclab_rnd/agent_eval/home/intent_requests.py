"""About 900 written first requests (600 by writers A and B in English, Persian and mixed; 300 by writer C in English only) to Home, each with its kind of help fixed by code (package P6 of the 2026-10-29 plan).

Home misroutes first requests ("I have data analysis problems" became a prediction), and there are too few real labelled
requests to train a routing model. This writes them, the way ``people_writer`` writes people:

- **The truth is never the writer's.** :func:`slots` draws every request's kind of help (a track of ``draft.intent``),
  its need (``segments`` among the analysis needs), its language (60% English, 25% Persian, 15% mixed), its style
  (plain, vague, typos, short, long, two kinds) and a business, from a fixed seed. The writer only writes the words.
- **Three writers** (prompts ``home_request_writer``, ``home_request_writer_b`` and, added after the first measurement,
  ``home_request_writer_c`` for direct requests that use the usual words), so a classifier measured across writers
  cannot pass by learning one writer's style.
- **A check**: the plain single-call reading of the Home eval (prompt ``home_eval_plain``, purpose ``home_agent``) reads
  every request; one whose reading disagrees with its assigned kind is kept apart (``intent_requests_v1_flagged.json``),
  never trained on. The owner labels 60 of them blind (:func:`blind_sample`, ``docs/guides/BLIND_LABELLING.md``).
- Every label here is tagged ``written``: kept apart from real labels, weighted lower, never enough to promote a model.
- Every request and reply is kept whole in ``agent_runs/live_tests/2026-10-29-intent-requests/``; the gateway keeps the
  model's thinking as the API thinking summary (never reasoning asked for in the reply).

``python -m dclab_rnd.agent_eval.home.intent_requests plan`` prints what it would send; ``write --writer a --yes
--cap-eur 1`` writes (resumable: a batch already answered is not asked again), ``check --yes --cap-eur 1`` checks,
``assemble`` rebuilds the files, ``blind`` draws the owner's sample, ``blind-score`` reads the owner's answers.
A run may spend at most EUR 1, and all runs together EUR 4 (the package's cap; the owner gave $10 more on 2026-10-29).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "intent_requests_v1.json"
FLAGGED = HERE / "intent_requests_v1_flagged.json"
MANIFEST = HERE / "intent_requests_v1_manifest.json"
BLIND = HERE / "intent_requests_v1_blind.json"
BLIND_ANSWERS = HERE / "intent_requests_v1_blind_answers.json"
HOME = ROOT / "agent_runs" / "live_tests" / "2026-10-29-intent-requests"
PROMPTS = {"a": "home_request_writer", "b": "home_request_writer_b", "c": "home_request_writer_c"}
WRITERS = tuple(PROMPTS)
WRITER_PURPOSE, CHECK_PURPOSE = "journey_writer", "home_agent"  # the check is the Home eval's plain arm, as it runs there
RUN_CAP_EUR, TOTAL_CAP_EUR = 1.0, 4.0
PER_TRACK, PER_BATCH, SEED = 100, 10, 20261029
# writer C (added after the first measurement, 2026-10-29): people who say what they want with the usual words and name
# their tables, columns and periods; A and B were told to avoid the usual word in half the messages, and the classifier
# trained on them missed plain direct requests ("predict which bookings get cancelled", "average order value per channel")
PER_TRACK_C = 50
LANGUAGES_C = (("en", 100),)  # English only (the owner, 2026-10-29: "for now you can skip persian")
LABEL_SOURCE = "written"

TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")  # draft.intent.TRACKS (a test pins them)
KIND_MEANS = {
    "prediction": "a model that predicts something not yet known (who will leave, how much will sell, which case is risky)",
    "analysis": "numbers about what already happened in their data: totals, counts per group, comparisons, trends, the groups "
                "(segments) their customers fall into",
    "preparation": "cleaning, combining or reshaping their data into a new table",
    "reporting": "a report or dashboard that is kept up to date (refreshed every day, week or month)",
    "business": "a business question or decision: why a number changed, what to do, which option, whether it is worth it",
    "advice": "an explanation or a recommendation, with no data work done on their files",
}
NEEDS: dict[str, dict[str, str]] = {
    "prediction": {
        "will_leave": "know in advance which customers, members or users will leave, cancel or stop buying",
        "how_much_next": "know how much or how many of something there will be in a coming period (demand, sales, calls, stock)",
        "risky_cases": "catch risky, fraudulent or failing cases before they happen or cost money",
        "which_category": "have each new item (ticket, email, claim, product) put in the right category automatically",
        "how_long": "know how long something will take or when it will happen (a delivery, a repair, a payment)",
        "rank_leads": "know which leads, accounts or customers are most likely to respond, buy or upgrade, to call them first",
    },
    "analysis": {
        "segments": "find or describe the groups (segments) their customers, members or users fall into, from data they already have",
        "numbers_by_group": "totals, averages or counts of what already happened, per group (region, product, month, branch)",
        "compare": "compare two groups or two past periods in their data",
        "trend": "see how something changed over the past months or years",
        "top_items": "find the best or worst products, branches, staff or channels in their past data",
        "explore": "understand what is in their data in general, what it says",
    },
    "preparation": {
        "clean": "fix messy values: spelling variants, wrong formats, stray spaces, broken dates",
        "combine": "join or merge several files, sheets or exports into one table",
        "dedupe": "remove duplicate rows or customers entered twice",
        "reshape": "change the table's shape: one row per customer, columns to rows, a pivot",
        "fill_missing": "deal with empty or missing cells",
        "convert": "convert units, currencies or date formats, or split one column into several",
    },
    "reporting": {
        "weekly_dashboard": "a dashboard that updates itself every week or every day",
        "monthly_report": "a report that goes to their manager or board every month",
        "kpi_tracking": "keep an eye on a few key numbers as they change",
        "team_view": "a screen a team opens every morning to see where things stand",
        "replace_manual": "stop rebuilding the same Excel report by hand every week",
    },
    "business": {
        "why_changed": "find out why a number went down or up",
        "should_we": "decide whether to do something (open a branch, raise prices, stop a product)",
        "which_option": "choose between two or more options",
        "worth_it": "know whether a campaign, discount or hire was worth it",
        "where_focus": "decide where to put their effort or budget next",
    },
    "advice": {
        "explain_term": "understand what a term or a number they saw means",
        "which_method": "know which method, tool or approach fits their situation",
        "how_much_data": "know how much data they need, or whether their data is enough",
        "is_it_ok": "know whether something they plan to do with data is right or safe",
        "how_to_start": "know where to start with data or AI in their business",
    },
}
SEGMENTS_SHARE = 30  # of the 100 analysis requests, about segments (the owner's sentence, 2026-10-29)
SECOND = {  # a second, smaller kind a two-kind request also asks for (its main goal stays the slot's kind)
    "prediction": ("preparation", "reporting", "analysis"),
    "analysis": ("preparation", "reporting", "advice"),
    "preparation": ("analysis", "advice"),
    "reporting": ("preparation", "analysis"),
    "business": ("analysis", "advice", "reporting"),
    "advice": ("analysis", "preparation", "prediction"),
}
LANGUAGES = (("en", 60), ("fa", 25), ("mixed", 15))
STYLES = (("plain", 20), ("vague", 15), ("typos", 15), ("short", 20), ("long", 15), ("two_kinds", 15))
STYLE_MEANS = {
    "plain": "an ordinary, clear request of 1 to 3 sentences",
    "vague": "few words and unclear: the person cannot name what they need, but what they say still leans toward this kind "
             "(like \"I have data analysis problems\")",
    "typos": "English: written by a non-native speaker, with typos and broken grammar (like \"I want analyze my customer and "
             "segment of the my customer\"). Persian or mixed: informal spoken Persian with spelling slips and no punctuation",
    "short": "fewer than 10 words",
    "long": "4 to 8 sentences: the business and what they tried first, the ask at the end",
    "two_kinds": "1 to 4 sentences asking for the main kind and, as a smaller part, the second kind",
}
LANGUAGE_MEANS = {
    "en": "English",
    "fa": "Persian in Persian script, everyday words (formal or colloquial); English only for a file or column name",
    "mixed": "Persian in Persian script with English words or phrases mixed in, as data people in Iran often write "
             "(dashboard, churn, Excel, segment, report, KPI, model)",
    "finglish": "Persian written in Latin letters (Finglish), with some English words mixed in",
}


def _businesses() -> tuple[str, ...]:
    from dclab_rnd.people_writer import BUSINESSES

    return BUSINESSES


def _deck(spec: tuple[tuple[str, int], ...], n: int, rng: random.Random) -> list[str]:
    """``n`` values in the shares of ``spec`` (scaled when n is not 100), shuffled."""
    out: list[str] = []
    total = sum(k for _, k in spec)
    for value, k in spec:
        out += [value] * round(k * n / total)
    while len(out) < n:
        out.append(spec[0][0])
    out = out[:n]
    rng.shuffle(out)
    return out


def slots(per_track: int = PER_TRACK) -> list[dict[str, Any]]:
    """Every request's truth, drawn by code from a fixed seed: kind, need, language, style, second kind, business, writer
    and batch. The writers never see another slot's truth and never choose one."""
    out = []
    businesses = _businesses()
    for track in TRACKS:
        rng = random.Random(f"{SEED}:{track}")
        langs, styles = _deck(LANGUAGES, per_track, rng), _deck(STYLES, per_track, rng)
        needs = list(NEEDS[track])
        if track == "analysis":
            k = round(SEGMENTS_SHARE * per_track / 100)
            rest = [n for n in needs if n != "segments"]
            drawn = ["segments"] * k + [rest[i % len(rest)] for i in range(per_track - k)]
        else:
            drawn = [needs[i % len(needs)] for i in range(per_track)]
        rng.shuffle(drawn)
        rows = []
        for i in range(per_track):
            lang = langs[i]
            row = {"slot": f"{track[:4]}-{i:03d}", "kind": track, "need": drawn[i], "language": lang, "style": styles[i],
                   "business": rng.choice(businesses)}
            if lang == "mixed" and rng.random() < 0.2:
                row["mixed_as"] = "finglish"
            if styles[i] == "two_kinds":
                row["second_kind"] = rng.choice(SECOND[track])
            rows.append(row)
        # writers by turns over the rows sorted by language and style: each writer gets half of every language and style
        for j, row in enumerate(sorted(rows, key=lambda r: (r["language"], r["style"], r["slot"]))):
            row["writer"] = "ab"[j % 2]
        for writer in "ab":
            mine = [r for r in rows if r["writer"] == writer]
            rng.shuffle(mine)
            for j, row in enumerate(mine):
                row["batch"] = f"{track}-{writer}-{j // PER_BATCH}"
        out += rows
    return out


def slots_c(per_track: int = PER_TRACK_C) -> list[dict[str, Any]]:
    """Writer C's own slots: English only, the same shares of style and need, drawn from their own seed."""
    out = []
    businesses = _businesses()
    for track in TRACKS:
        rng = random.Random(f"{SEED}:c:{track}")
        langs, styles = _deck(LANGUAGES_C, per_track, rng), _deck(STYLES, per_track, rng)
        needs = list(NEEDS[track])
        if track == "analysis":
            k = round(SEGMENTS_SHARE * per_track / 100)
            rest = [n for n in needs if n != "segments"]
            drawn = ["segments"] * k + [rest[i % len(rest)] for i in range(per_track - k)]
        else:
            drawn = [needs[i % len(needs)] for i in range(per_track)]
        rng.shuffle(drawn)
        for i in range(per_track):
            row = {"slot": f"{track[:4]}-c{i:03d}", "kind": track, "need": drawn[i], "language": langs[i], "style": styles[i],
                   "business": rng.choice(businesses), "writer": "c", "batch": f"{track}-c-{i // PER_BATCH}"}
            if langs[i] == "mixed" and rng.random() < 0.2:
                row["mixed_as"] = "finglish"
            if styles[i] == "two_kinds":
                row["second_kind"] = rng.choice(SECOND[track])
            out.append(row)
    return out


def all_slots() -> list[dict[str, Any]]:
    return slots() + slots_c()


def batches(writer: str, per_track: int = PER_TRACK) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for row in (slots_c() if writer == "c" else slots(per_track)):
        if row["writer"] == writer:
            found.setdefault(row["batch"], []).append(row)
    return dict(sorted(found.items(), key=lambda kv: (int(kv[0].rsplit("-", 1)[1]), TRACKS.index(kv[0].split("-")[0]))))


def _slot_for_writer(row: dict[str, Any]) -> dict[str, Any]:
    lang = row.get("mixed_as") or row["language"]
    out = {"slot": row["slot"], "kind": row["kind"], "kind_means": KIND_MEANS[row["kind"]], "need": row["need"],
           "need_means": NEEDS[row["kind"]][row["need"]], "language": "mixed" if lang == "finglish" else lang,
           "language_means": LANGUAGE_MEANS[lang], "style": row["style"], "style_means": STYLE_MEANS[row["style"]],
           "business": row["business"]}
    if row.get("second_kind"):
        out["second_kind"] = row["second_kind"]
        out["second_kind_means"] = KIND_MEANS[row["second_kind"]]
    return out


def request(writer: str, batch: list[dict[str, Any]]) -> list[dict[str, str]]:
    from dclab_rnd import prompts

    ask = {"slots": [_slot_for_writer(r) for r in batch]}
    return [{"role": "system", "content": prompts.text(PROMPTS[writer])}, {"role": "user", "content": json.dumps(ask, ensure_ascii=False)}]


def prompt_hash(writer: str) -> str:
    from dclab_rnd import prompts

    return hashlib.sha256(prompts.text(PROMPTS[writer]).encode()).hexdigest()[:12]


def parse(content: str, batch: list[dict[str, Any]]) -> dict[str, str]:
    """Each slot's message from a reply: only slots of this batch, a text of 2 to 1500 characters."""
    from dclab_rnd.core_advisor import _first_object

    doc = _first_object(content or "") or {}
    wanted = {r["slot"] for r in batch}
    out = {}
    for item in doc.get("requests") or []:
        if not isinstance(item, dict):
            continue
        slot, text = item.get("slot"), item.get("text")
        if slot in wanted and slot not in out and isinstance(text, str) and 2 <= len(text.strip()) <= 1500:
            out[slot] = text.strip()
    return out


# ------------------------------------------------------------------------------------------------ spending

def _spent_in(folder: Path) -> float:
    return sum(float(json.loads(p.read_text(encoding="utf-8")).get("spent_eur") or 0.0) for p in folder.glob("*.json")) if folder.is_dir() else 0.0


def spent_total(home: Optional[Path] = None) -> float:
    """Every euro spent, the superseded batches included (writer C's first three batches, written with Persian before the
    owner said to skip it; kept whole, never assembled)."""
    home = Path(home or HOME)
    return round(sum(_spent_in(home / d) for d in (*(f"writer_{w}" for w in WRITERS), "writer_c_superseded", "checks")), 6)


def _run_cap(cap_eur: float, home: Path) -> float:
    """What this run may spend: its cap (at most EUR 1), and never past EUR 4 over every run."""
    if cap_eur > RUN_CAP_EUR:
        raise SystemExit(f"a run may spend at most EUR {RUN_CAP_EUR}: --cap-eur {cap_eur} refused")
    return max(0.0, min(cap_eur, TOTAL_CAP_EUR - spent_total(home)))


def _client(purpose: str, limit: float) -> Any:
    from dclab_rnd.models.gateway import for_workspace
    from dclab_rnd.models.prices import price_of

    from dclab_rnd.models.gateway import mark_invented_tables
    _gateway = mark_invented_tables(for_workspace(), "home intent requests")  # every table here is invented by code
    client = _gateway.client(purpose)
    model = str(getattr(client, "model", "") or "")
    if client is None or not model.startswith("claude") or price_of(model) is None:
        raise SystemExit(f"{purpose} must be served by a priced Claude model here (it is {model or 'no model'}): nothing sent")
    client.run_limit_eur = max(0.0, limit)
    return client


def _budget_refused(exc: Exception) -> bool:
    return str(exc).startswith("BudgetExceeded") or type(exc).__name__ == "BudgetExceeded"


# ------------------------------------------------------------------------------------------------ writing

def write(writer: str, cap_eur: float, home: Optional[Path] = None, client_for: Any = None) -> dict[str, Any]:
    """Every batch of ``writer`` not answered yet, sent and kept whole in ``home/writer_<w>/<batch>.json``."""
    home = Path(home or HOME)
    folder = home / f"writer_{writer}"
    folder.mkdir(parents=True, exist_ok=True)
    cap = _run_cap(cap_eur, home)
    spent = 0.0
    phash = prompt_hash(writer)
    for name, batch in batches(writer).items():
        path = folder / f"{name}.json"
        if path.is_file():
            continue
        if spent >= cap:
            return {"stopped": f"this run's cap of EUR {round(cap, 4)} is reached", "spent_eur": round(spent, 4)}
        client = (client_for or _client)(WRITER_PURPOSE, cap - spent)
        messages = request(writer, batch)
        try:
            reply = client.complete(messages, None, max_tokens=6000)
        except RuntimeError as exc:
            if _budget_refused(exc):
                return {"stopped": f"this run's cap of EUR {round(cap, 4)} is reached", "spent_eur": round(spent, 4)}
            raise
        cost = float(getattr(client, "spent_eur", 0.0) or 0.0)
        spent += cost
        got = {"content": reply.get("content") or "", "finish_reason": reply.get("finish_reason"), "usage": reply.get("usage"),
               "model": getattr(client, "model", None)}
        path.write_text(json.dumps({"batch": name, "writer": writer, "prompt": PROMPTS[writer], "prompt_hash": phash, "slots": batch,
                                    "request": messages, "reply": got, "spent_eur": round(cost, 6)}, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    return {"spent_eur": round(spent, 4), "batches": len(list(folder.glob("*.json")))}


def written(home: Optional[Path] = None) -> list[dict[str, Any]]:
    """Every request written so far, with its truth (from the slot, never from the reply)."""
    home = Path(home or HOME)
    out = []
    for writer in WRITERS:
        for path in sorted((home / f"writer_{writer}").glob("*.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            texts = parse(doc["reply"]["content"], doc["slots"])
            for row in doc["slots"]:
                if row["slot"] not in texts:
                    continue
                tracks = [row["kind"]] + ([row["second_kind"]] if row.get("second_kind") else [])
                out.append({"id": f"IR1-{row['slot']}", "text": texts[row["slot"]], "track": row["kind"], "tracks": tracks,
                            "need": row["need"], "lang": row["language"], "mixed_as": row.get("mixed_as"), "style": row["style"],
                            "business": row["business"], "writer": writer, "written_by": f"{doc['prompt']}@{doc['prompt_hash']}",
                            "batch": row["batch"], "model": doc["reply"].get("model"), "label_source": LABEL_SOURCE})
    return out


# ------------------------------------------------------------------------------------------------ checking

def agrees(row: dict[str, Any], reading: dict[str, Any]) -> tuple[bool, str]:
    """A plain reading agrees when its main track is the request's kind (a two-kind request: when it names the kind among
    its tracks)."""
    tracks = reading.get("tracks") or []
    if not tracks:
        return False, "the plain reading named no known track"
    if row["style"] == "two_kinds":
        return (row["track"] in tracks, "" if row["track"] in tracks else f"read as {tracks}, not {row['track']}")
    return (tracks[0] == row["track"], "" if tracks[0] == row["track"] else f"read as {tracks[0]}, not {row['track']}")


def check(cap_eur: float, home: Optional[Path] = None, client_for: Any = None, workers: int = 4) -> dict[str, Any]:
    """Every written request not checked yet, read once by the plain arm; the reading and its whole reply kept in
    ``home/checks/<id>.json``."""
    from . import read_plain

    home = Path(home or HOME)
    folder = home / "checks"
    folder.mkdir(parents=True, exist_ok=True)
    cap = _run_cap(cap_eur, home)
    todo = [r for r in written(home) if not (folder / f"{r['id']}.json").is_file()]
    lock, state = threading.Lock(), {"spent": 0.0, "done": 0, "stopped": None}

    def one(row: dict[str, Any]) -> None:
        with lock:
            if state["spent"] >= cap or state["stopped"]:
                state["stopped"] = state["stopped"] or f"this run's cap of EUR {round(cap, 4)} is reached"
                return
            limit = cap - state["spent"]
        client = (client_for or _client)(CHECK_PURPOSE, limit)
        keep = home / "checks_replies" / f"{row['id']}.json"
        try:
            reading = read_plain({"id": row["id"], "text": row["text"], "data": []}, client, keep=keep)
        except RuntimeError as exc:
            if _budget_refused(exc):
                with lock:
                    state["stopped"] = f"this run's cap of EUR {round(cap, 4)} is reached"
                return
            raise
        cost = float(getattr(client, "spent_eur", 0.0) or 0.0)
        ok, why = agrees(row, reading)
        (folder / f"{row['id']}.json").write_text(json.dumps({"id": row["id"], "reading": {"track": reading.get("track"), "tracks": reading.get("tracks"),
                                                                                          "exit": reading.get("exit")},
                                                              "agrees": ok, "why": why, "model": getattr(client, "model", None),
                                                              "prompt": "home_eval_plain", "spent_eur": round(cost, 6)},
                                                             ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        with lock:
            state["spent"] += cost
            state["done"] += 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))
    out = {"checked": state["done"], "left": len(todo) - state["done"], "spent_eur": round(state["spent"], 4)}
    if state["stopped"]:
        out["stopped"] = state["stopped"]
    return out


# ------------------------------------------------------------------------------------------------ the files

def assemble(home: Optional[Path] = None) -> dict[str, Any]:
    """The checked requests split into the training file (the reading agrees) and the kept-apart file (it does not); a
    request not checked yet is in neither. The manifest counts them."""
    home = Path(home or HOME)
    kept, flagged, unchecked = [], [], []
    for row in written(home):
        path = home / "checks" / f"{row['id']}.json"
        if not path.is_file():
            unchecked.append(row)
            continue
        c = json.loads(path.read_text(encoding="utf-8"))
        row = {**row, "plain_reading": c["reading"]["tracks"]}
        (kept if c["agrees"] else flagged).append(row if c["agrees"] else {**row, "check": c["why"]})
    OUT.write_text(json.dumps(kept, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    FLAGGED.write_text(json.dumps(flagged, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    manifest = build_manifest(kept, flagged, unchecked, home)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return {"kept": len(kept), "flagged": len(flagged), "unchecked": len(unchecked), "spent_eur": manifest["spent_eur"]}


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(r.get(key)) for r in rows).items()))


def build_manifest(kept: list[dict[str, Any]], flagged: list[dict[str, Any]], unchecked: list[dict[str, Any]], home: Path) -> dict[str, Any]:
    every = kept + flagged + unchecked
    models: Counter = Counter()
    for d in ("checks",):
        for p in (home / d).glob("*.json"):
            models[f"check:{json.loads(p.read_text(encoding='utf-8')).get('model')}"] += 1
    for r in every:
        models[f"writer:{r.get('model')}"] += 1
    planned = all_slots()
    return {
        "set": "intent_requests_v1", "package": "P6 (plan 2026-10-29)", "label_source": LABEL_SOURCE,
        "truth": "each request's kind, need, language and style were drawn by code (intent_requests.slots, seed "
                 f"{SEED}) before it was written; the writer only wrote the words",
        "planned": len(planned), "written": len(every), "kept": len(kept), "flagged": len(flagged), "unchecked": len(unchecked),
        "check": "the plain single-call reading (prompt home_eval_plain, purpose home_agent): a request whose main track "
                 "differs from its assigned kind (a two-kind request: whose tracks miss it) is kept apart, never trained on",
        "writers": {w: {"prompt": PROMPTS[w], "prompt_hash": prompt_hash(w)} for w in WRITERS},
        "models": dict(models),
        "kept_by": {k: _counts(kept, k) for k in ("track", "lang", "style", "writer")},
        "flagged_by": {k: _counts(flagged, k) for k in ("track", "lang", "style", "writer")},
        "written_by": {k: _counts(every, k) for k in ("track", "lang", "style", "writer")},
        "track_by_writer": {t: _counts([r for r in every if r["track"] == t], "writer") for t in TRACKS},
        "segments": {"written": sum(r["need"] == "segments" for r in every), "kept": sum(r["need"] == "segments" for r in kept)},
        "spent_eur": spent_total(home),
        "caps": {"per_run_eur": RUN_CAP_EUR, "total_eur": TOTAL_CAP_EUR},
        "kept_whole_in": str(home.relative_to(ROOT)) if str(home).startswith(str(ROOT)) else str(home),
        "limits": ["Written by a model from code-assigned slots: these are not real people's requests, and a kind the writer "
                   "found hard to express may be under-represented after the check.",
                   "The check is one model's reading: agreement with it is not correctness; the owner's blind sample "
                   "(docs/guides/BLIND_LABELLING.md) measures the labels' noise.",
                   "Labels are 'written': kept apart from real labels, weighted lower, never enough to promote a model."],
    }


# ------------------------------------------------------------------------------------------------ the owner's blind sample

def blind_sample(n_per_track: int = 10, seed: int = 7) -> dict[str, Any]:
    """60 English requests (10 per assigned kind, from the kept and the flagged alike), shown with their words only, in a drawn
    order; the owner's answers go to ``BLIND_ANSWERS`` (written once, never overwritten)."""
    rows = json.loads(OUT.read_text(encoding="utf-8")) + json.loads(FLAGGED.read_text(encoding="utf-8"))
    rows = [r for r in rows if r.get("lang", "en") == "en"]  # English only for now (the owner, 2026-10-29: "skip persian")
    rng = random.Random(seed)
    drawn = []
    for track in TRACKS:
        mine = sorted((r for r in rows if r["track"] == track), key=lambda r: r["id"])
        drawn += rng.sample(mine, min(n_per_track, len(mine)))
    rng.shuffle(drawn)
    sample = [{"n": i + 1, "id": r["id"], "text": r["text"]} for i, r in enumerate(drawn)]
    BLIND.write_text(json.dumps(sample, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if not BLIND_ANSWERS.is_file():
        BLIND_ANSWERS.write_text(json.dumps([{"n": s["n"], "id": s["id"], "track": None, "second_track": None, "segments": None, "note": ""}
                                             for s in sample], ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    shown = [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in (BLIND, BLIND_ANSWERS)]
    return {"sample": len(sample), "file": shown[0], "answers": shown[1]}


def blind_score() -> dict[str, Any]:
    """The owner's labels against the written labels and the plain reading (labels source ``owner``; nothing changed)."""
    from dclab_core.decision_models import wilson

    rows = {r["id"]: r for r in json.loads(OUT.read_text(encoding="utf-8")) + json.loads(FLAGGED.read_text(encoding="utf-8"))}
    answers = [a for a in json.loads(BLIND_ANSWERS.read_text(encoding="utf-8")) if a.get("track") in TRACKS]
    if not answers:
        return {"labelled": 0}
    written_right = sum(rows[a["id"]]["track"] == a["track"] for a in answers)
    plain_right = sum((rows[a["id"]].get("plain_reading") or [None])[0] == a["track"] for a in answers)
    flagged = [a for a in answers if "check" in rows[a["id"]]]
    return {"labelled": len(answers), "written_label_agrees": written_right, "written_ci": wilson(written_right, len(answers)),
            "plain_reading_agrees": plain_right, "flagged_in_sample": len(flagged),
            "flagged_where_owner_agrees_with_written": sum(rows[a["id"]]["track"] == a["track"] for a in flagged),
            "by_track": {t: {"n": sum(rows[a["id"]]["track"] == t for a in answers),
                             "agrees": sum(rows[a["id"]]["track"] == t == a["track"] for a in answers)} for t in TRACKS}}


# ------------------------------------------------------------------------------------------------ command line

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.home.intent_requests", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "write", "check", "assemble", "blind", "blind-score", "status"))
    ap.add_argument("--writer", choices=WRITERS, default="a")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap-eur", type=float)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    if args.command == "plan":
        planned = all_slots()
        print(f"Plan: {len(planned)} requests in {sum(len(batches(w)) for w in WRITERS)} writer requests through {WRITER_PURPOSE} "
              f"(Claude only), then {len(planned)} plain readings through {CHECK_PURPOSE}; each run at most EUR {RUN_CAP_EUR}, "
              f"all runs EUR {TOTAL_CAP_EUR}; spent so far EUR {spent_total()}; kept whole in {HOME}.")
        print({k: _counts(planned, k) for k in ("kind", "language", "style", "writer")})
        print(json.dumps(request("a", next(iter(batches("a").values())))[1], ensure_ascii=False)[:1500])
        return 0
    if args.command == "status":
        print({"written": len(written()), "checked": len(list((HOME / "checks").glob("*.json"))), "spent_eur": spent_total()})
        return 0
    if args.command == "assemble":
        print(assemble())
        return 0
    if args.command == "blind":
        print(blind_sample())
        return 0
    if args.command == "blind-score":
        print(json.dumps(blind_score(), indent=1))
        return 0
    if not args.yes or args.cap_eur is None:
        print("Add --yes --cap-eur N (N at most 1) to send the requests.")
        return 0
    from dclab_rnd import real_shadow

    real_shadow.load_env()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    result = write(args.writer, args.cap_eur) if args.command == "write" else check(args.cap_eur, workers=args.workers)
    print(json.dumps({**result, "spent_total_eur": spent_total(), **assemble()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
