"""Claude helps label, blind (the owner's request, 2026-10-15): a model is shown exactly what a blind reviewer sees of
each drawn decision (``dclab_core.ledger.blind_view``: the question, the subject, the state and the options; never
DCLab's answer) and answers it through the gateway's ``decision_labeler`` purpose.

Its labels are source ``model`` with ``by="model:<name>"``, the draw's weight beside them, and are kept apart from a
person's everywhere: ``effective_label`` never takes one as the truth (agreement with a model is not correctness, and a
model may itself be one of the answerers measured), the summary counts them on their own, and they are never exported
as the main product's labels. A reviewer reads them as a second opinion: where the model and the rule disagree is where
a person's label is worth most.

    python -m dclab_rnd.ledger.model_labels                       # the plan: how many drawn decisions, the cap
    python -m dclab_rnd.ledger.model_labels --yes --cap-eur 0.5   # label them; never more than the cap
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

SEED = "blind-v1"


def _value(answer: Any, view: dict[str, Any]) -> Any:
    """The option as a label's value: yes or no as a flag for a yes-or-no point; None when it is not an option."""
    if not isinstance(answer, str) or answer not in (view.get("options") or []):
        return None
    if view.get("answer") == "noul":
        return answer == "yes"
    return answer


def to_label(store: Any, ask: Callable[[dict[str, Any]], Optional[dict[str, Any]]], model: str, *, point: Optional[str] = None,
             limit: Optional[int] = None, every: bool = False, workers: int = 1) -> dict[str, int]:
    """Ask ``ask`` (the blind view -> {"answer", "confidence", "reason"}) about every drawn decision this model has not
    labelled yet, and write its labels. Returns counts: asked, labelled, unsure (no answer), refused (not an option).
    ``every`` (29.18, the owner's judge of every past decision): each decision, drawn or not; a drawn one keeps its draw's
    weight, the rest weigh 1 (a census, not a sample). ``workers``: the model's requests at once; labels are written in
    order, one at a time."""
    from concurrent.futures import ThreadPoolExecutor

    from dclab_core import ledger as contract

    from . import label

    by = f"model:{model}"[:120]
    done = {x.record for x in store.labels() if x.source == "model" and x.by == by}
    out: dict[str, Any] = {"asked": 0, "labelled": 0, "unsure": 0, "refused": 0, "stopped": 0}
    pending = [r for r in store.records(point=point) if r.id not in done and (every or contract.blind_draw(r, SEED))]
    if limit is not None:
        pending = pending[:limit]
    step = max(1, int(workers))
    for start in range(0, len(pending), step):
        chunk = pending[start:start + step]
        views = [contract.blind_view(r) for r in chunk]
        out["asked"] += len(chunk)
        with ThreadPoolExecutor(max_workers=len(chunk)) as pool:
            futures = [pool.submit(ask, v) for v in views]
        replies, stop = [], None
        for f in futures:
            try:
                replies.append(f.result() or {})
            except RuntimeError as error:  # the cap reached or the provider refusing: stop, keep what was written, say why
                replies.append(None)
                stop = stop or str(error)
        for record, view, reply in zip(chunk, views, replies):
            if reply is None:
                continue
            if reply.get("answer") is None:
                out["unsure"] += 1
                continue
            value = _value(reply.get("answer"), view)
            if value is None:
                out["refused"] += 1
                continue
            reason = str(reply.get("reason") or "")[:180]
            drawn = contract.blind_draw(record, SEED)
            written = label(record, value, "model", by, weight=round(1 / contract.blind_probability(record), 6) if drawn else 1.0, ledger=store,
                            evidence={"kind": "review", "ref": f"{reply.get('confidence') or 'unstated'}: {reason}"[:200]})
            out["labelled"] += written is not None
        if stop:
            out["stopped"] = 1
            out["why_stopped"] = stop[:200]
            break
    return out


def asker(gateway: Any, cap_eur: float, keep: Optional[Any] = None) -> tuple[Callable[[dict[str, Any]], Optional[dict[str, Any]]], Any]:
    """The model's answer to a blind view, through the gateway (capped for the whole run); and the client, for its spend.
    ``keep`` (29.18, the owner: every judgment with all its details): a folder whose ``judgments.jsonl`` receives one line
    per request: the decision's id (never sent), exactly what was sent, the model's whole reply, what was read from it,
    the tokens, the seconds and the cost; the ledger's label keeps the answer and a reason cut to its contract's length."""
    import threading
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    from dclab_core import prompts

    client = gateway.client("decision_labeler")
    if client is None:
        raise RuntimeError("no model serves decision_labeler: set the provider's key")
    client.run_limit_eur = cap_eur
    log = Path(keep) / "judgments.jsonl" if keep is not None else None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    model = getattr(client, "model", None)
    fingerprint = prompts.fingerprints().get("decision_labeler")

    def ask(view: dict[str, Any]) -> Optional[dict[str, Any]]:
        evidence = {k: view.get(k) for k in ("point", "question", "subject", "state", "options")}
        clock = time.monotonic()
        reply = client.complete([{"role": "system", "content": prompts.text("decision_labeler")},
                                 {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, default=str)}], None, max_tokens=1200)
        text = (reply.get("content") or "").strip()
        start, end = text.find("{"), text.rfind("}")
        try:
            parsed = json.loads(text[start:end + 1]) if start >= 0 else None
        except ValueError:
            parsed = None
        if log is not None:
            usage = reply.get("usage") or {}
            cost = None
            try:
                from ..models.prices import price_of

                price = price_of(str(model or ""))
                if price is not None:
                    cost = round((int(usage.get("input_tokens") or 0) * price.input + int(usage.get("output_tokens") or 0) * price.output) / 1e6, 6)
            except Exception:  # noqa: BLE001 - a price that cannot be read leaves the cost unstated, never the line
                cost = None
            line = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "record": view.get("record"), "point": view.get("point"),
                    "model": model, "purpose": "decision_labeler", "prompt": fingerprint, "sent": evidence, "reply": text,
                    "parsed": parsed, "readable": isinstance(parsed, dict), "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"), "seconds": round(time.monotonic() - clock, 3), "cost_eur": cost}
            with lock, log.open("a", encoding="utf-8") as out:
                out.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        return parsed

    return ask, client


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    from dotenv import load_dotenv

    from ..settings import ROOT

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.ledger.model_labels", description=__doc__.split("\n")[0])
    parser.add_argument("--yes", action="store_true", help="send the requests")
    parser.add_argument("--cap-eur", type=float, default=0.5)
    parser.add_argument("--point", default=None, help="one decision point only")
    parser.add_argument("--limit", type=int, default=None, help="at most this many decisions")
    parser.add_argument("--every", action="store_true", help="29.18: every decision, not only the blind draw (a census of the past)")
    parser.add_argument("--ledger", default=None, help="29.18: a ledger folder to judge (a live test's, a twin lane's), instead of the workspace's")
    parser.add_argument("--workers", type=int, default=1, help="the model's requests at once")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    from dclab_core import ledger as contract

    from ..agentic.pool import Pool
    from ..models.gateway import for_workspace
    from ..settings import Settings

    if args.ledger:  # one ledger folder on its own: its records judged where they are kept
        from pathlib import Path

        from . import FileLedger

        folder = Path(args.ledger)
        store = FileLedger(folder, folder.parent.name) if (folder / "decisions").is_dir() else None
    else:
        store = getattr(Pool(Settings.load()).default, "ledger", None)  # the workspace's ledger, on files or PostgreSQL, as the server opens it
    if store is None:
        print("No decision ledger in this workspace yet (start make dev once: it creates the tables).")
        return 2
    drawn = [r for r in store.records(point=args.point) if args.every or contract.blind_draw(r, SEED)]
    if not args.yes:
        print(f"Plan: {len(drawn)} {'' if args.every else 'drawn '}decisions{' at ' + args.point if args.point else ''}; the model sees each blind view only, "
              f"about half a cent each, capped at {args.cap_eur} EUR. Add --yes to send them.")
        return 0
    gateway = for_workspace()
    from pathlib import Path as _Path

    keep = (_Path(args.ledger).parent if args.ledger else _Path(Settings.load().agent_home)) / "judgments"
    ask, client = asker(gateway, args.cap_eur, keep=keep)
    counts = to_label(store, ask, getattr(client, "model", "model"), point=args.point, limit=args.limit, every=args.every, workers=args.workers)
    print(f"{counts}; spent EUR {round(float(getattr(client, 'spent_eur', 0.0) or 0.0), 4)} (cap {args.cap_eur})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
