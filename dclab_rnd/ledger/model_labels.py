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
             limit: Optional[int] = None) -> dict[str, int]:
    """Ask ``ask`` (the blind view -> {"answer", "confidence", "reason"}) about every drawn decision this model has not
    labelled yet, and write its labels. Returns counts: asked, labelled, unsure (no answer), refused (not an option)."""
    from dclab_core import ledger as contract

    from . import label

    by = f"model:{model}"[:120]
    done = {x.record for x in store.labels() if x.source == "model" and x.by == by}
    out = {"asked": 0, "labelled": 0, "unsure": 0, "refused": 0}
    for record in store.records(point=point):
        if limit is not None and out["asked"] >= limit:
            break
        if record.id in done or not contract.blind_draw(record, SEED):
            continue
        view = contract.blind_view(record)
        out["asked"] += 1
        reply = ask(view) or {}
        if reply.get("answer") is None:
            out["unsure"] += 1
            continue
        value = _value(reply.get("answer"), view)
        if value is None:
            out["refused"] += 1
            continue
        reason = str(reply.get("reason") or "")[:180]
        written = label(record, value, "model", by, weight=round(1 / contract.blind_probability(record), 6), ledger=store,
                        evidence={"kind": "review", "ref": f"{reply.get('confidence') or 'unstated'}: {reason}"[:200]})
        out["labelled"] += written is not None
    return out


def asker(gateway: Any, cap_eur: float) -> tuple[Callable[[dict[str, Any]], Optional[dict[str, Any]]], Any]:
    """The model's answer to a blind view, through the gateway (capped for the whole run); and the client, for its spend."""
    from dclab_core import prompts

    client = gateway.client("decision_labeler")
    if client is None:
        raise RuntimeError("no model serves decision_labeler: set the provider's key")
    client.run_limit_eur = cap_eur

    def ask(view: dict[str, Any]) -> Optional[dict[str, Any]]:
        evidence = {k: view.get(k) for k in ("point", "question", "subject", "state", "options")}
        reply = client.complete([{"role": "system", "content": prompts.text("decision_labeler")},
                                 {"role": "user", "content": json.dumps(evidence, ensure_ascii=False, default=str)}], None, max_tokens=1200)
        text = (reply.get("content") or "").strip()
        start, end = text.find("{"), text.rfind("}")
        try:
            return json.loads(text[start:end + 1]) if start >= 0 else None
        except ValueError:
            return None

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
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env", override=False)
    from dclab_core import ledger as contract

    from ..agentic.pool import Pool
    from ..models.gateway import for_workspace
    from ..settings import Settings

    store = getattr(Pool(Settings.load()).default, "ledger", None)  # the workspace's ledger, on files or PostgreSQL, as the server opens it
    if store is None:
        print("No decision ledger in this workspace yet (start make dev once: it creates the tables).")
        return 2
    drawn = [r for r in store.records(point=args.point) if contract.blind_draw(r, SEED)]
    if not args.yes:
        print(f"Plan: {len(drawn)} drawn decisions{' at ' + args.point if args.point else ''}; the model sees each blind view only, "
              f"about half a cent each, capped at {args.cap_eur} EUR. Add --yes to send them.")
        return 0
    gateway = for_workspace()
    ask, client = asker(gateway, args.cap_eur)
    counts = to_label(store, ask, getattr(client, "model", "model"), point=args.point, limit=args.limit)
    print(f"{counts}; spent EUR {round(float(getattr(client, 'spent_eur', 0.0) or 0.0), 4)} (cap {args.cap_eur})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
