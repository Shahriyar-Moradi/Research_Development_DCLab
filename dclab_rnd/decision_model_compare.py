"""A language model as a decision answerer, compared with the rules and the classical classifiers (package 22.5).

``python -m dclab_rnd.decision_model_compare --ollama qwen2.5-coder:1.5b`` asks a local model (Ollama, free) the
decisions of the generated sets' dev partition (``--per-point`` of each point, drawn with a seed), through the gateway's
``decision_answer`` purpose (the question, the options and the state's bands and name tokens: never a value), and
compares it with the recorded rule and the fitted classifiers on the same decisions (``dclab_core.answerers.compare``):
coverage, agreement with the rule and accuracy against the benchmark labels. ``--tuned`` asks the tuned tier
(``DCLAB_TIER_TUNED_*``: the owner's fine-tuned model). A local endpoint is asked for nothing; a remote tuned one only
with ``--cap-eur`` (the most the run may spend); the purpose must be served by the tier asked for, or the run stops
(never a fallback to a paid tier). The sealed partition is never read here; a sealed verdict for a tuned model needs a
new generated set (every sealed set of v1 was judged in 22.3) and a gate runner that takes any answerer: both open. ``--write`` stores the result in ``evidence/campaigns/decision_llm_v1``.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SETS = ROOT / "research/llm-fine-tuning/experiments/decisions/generated_v1/decisions"
RESULTS = ROOT / "evidence/campaigns/decision_llm_v1/results"
OLLAMA = "http://localhost:11434/v1"


def dev_rows(per_point: int, seed: int = 0, sets: Path = SETS) -> list[dict[str, Any]]:
    """``per_point`` dev decisions of each point, drawn with ``seed`` (train rows and the sealed file are not read)."""
    rng = random.Random(seed)
    out = []
    for folder in sorted(p for p in Path(sets).iterdir() if p.is_dir()):
        with open(folder / "rows.jsonl", encoding="utf-8") as handle:
            dev = [r for r in (json.loads(line) for line in handle if line.strip()) if r.get("split") == "dev" and (r.get("label") or {}).get("value") is not None]
        out += rng.sample(dev, min(per_point, len(dev)))
    return out


def as_records(rows: list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    """Ledger records and benchmark labels from generated rows (the rule's recorded answer kept)."""
    from dclab_core import ledger as contract

    records, labels = [], []
    for r in rows:
        rec = contract.DecisionRecord(id=r["record"], point=r["point"], subject=contract.Subject(**r["subject"]), state=r["state"],
                                      options=[str(o) for o in r.get("options") or []],
                                      answers=[contract.Answer(**a) for a in r.get("answers") or []], value_used=r.get("value_used"),
                                      used=r.get("used") or "rule", level=0, at="2026-10-15T00:00:00+00:00")
        records.append(rec)
        labels.append(contract.DecisionLabel(id="bl" + r["record"][2:], record=rec.id, point=rec.point, value=r["label"]["value"],
                                             source="benchmark", by="benchgen", at="2026-10-15T00:00:00+00:00"))
    return records, labels


def local_gateway(model: str | None, tuned: bool, cap_eur: float | None = None) -> Any:
    """A gateway whose ``decision_answer`` purpose is served by a local model (or the tuned tier: local, or remote with a
    cap), and the tier's description; refused when the purpose would be served by anything else."""
    from dclab_rnd.models import settings
    from dclab_rnd.models.gateway import Gateway

    if model:
        os.environ.update({"DCLAB_LLM_PROVIDER": "", "OPENAI_API_KEY": "ollama", "OPENAI_BASE_URL": OLLAMA, "DCLAB_LLM_BASE_URL": OLLAMA, "OPENAI_MODEL": model})
        os.environ.update({f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.ROUTABLE for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")})
        os.environ.pop("DCLAB_INTERN_MODEL", None)  # it would name another model on the same endpoint
    tier = settings.TUNED if tuned else settings.purpose("decision_answer").tier
    info = settings.tier(tier)
    if tuned and not (info.get("base_url") and info.get("model")):
        raise SystemExit("the tuned tier is not set (DCLAB_TIER_TUNED_BASE_URL and DCLAB_TIER_TUNED_MODEL): nothing to compare")
    local = settings.is_local(info.get("base_url") or "")
    if not local and not (tuned and cap_eur is not None):
        raise SystemExit(f"the {tier} tier is not local ({info.get('base_url')}): give --cap-eur for a remote tuned model, or ask a local one")
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    gateway = Gateway()
    if tuned:  # the purpose served by the tuned tier for this run only, as an approved routing would serve it
        from types import SimpleNamespace

        current = settings.public(settings.TUNED)
        doc = {"purposes": {"decision_answer": {"serve": settings.TUNED, "serve_model": current["model"], "serve_endpoint": current["endpoint"]}}}
        gateway.routing = SimpleNamespace(get=lambda: doc)
    served = gateway.route("decision_answer")[0]
    if served != tier:  # never a fallback to another tier (a paid one)
        raise SystemExit(f"decision_answer would be served by the {served} tier, not {tier}: refused")
    return gateway, {**info, "local": local}


def compare(rows: list[dict[str, Any]], answerer: Any, classifiers: list[Any]) -> dict[str, Any]:
    from dclab_core.answerers import RecordedRule, compare as core_compare

    records, labels = as_records(rows)
    return core_compare(records, labels, [RecordedRule(), *classifiers, answerer])


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decision_model_compare")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--ollama", metavar="MODEL", help="a local model served by Ollama, for example qwen2.5-coder:1.5b")
    which.add_argument("--tuned", action="store_true", help="the tuned tier (DCLAB_TIER_TUNED_*): the owner's fine-tuned model")
    parser.add_argument("--cap-eur", type=float, default=None, help="with a remote tuned model: the most this run may spend")
    parser.add_argument("--per-point", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if not SETS.exists():
        print("no generated decision sets: run make decisions-gen first")
        return 2
    from dclab_core import prompts
    from dclab_core.answerers import ModelAnswerer
    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.ledger import shadow

    gateway, info = local_gateway(args.ollama, args.tuned, args.cap_eur)
    model_name = args.ollama or info.get("model") or "tuned"
    rows = dev_rows(args.per_point, args.seed)
    answerer = ModelAnswerer(GatewayAdvisor(gateway, max_tokens=300, cap_eur=args.cap_eur, models={"decision_answer": model_name} if args.ollama else None),
                             {r["point"] for r in rows}, name=f"llm:{model_name}",
                             version=prompts.fingerprints()["decision_answer"])
    result = compare(rows, answerer, list(shadow.answerers()))
    doc = {"model": model_name, "local": info["local"], "cap_eur": args.cap_eur, "split": "dev", "per_point": args.per_point, "seed": args.seed, "decisions": len(rows),
           "model_errors": answerer.errors,
           "prompt": prompts.fingerprints()["decision_answer"], "comparison": result}
    print(json.dumps(doc, indent=1, default=str))
    if args.write:
        from datetime import datetime, timezone

        RESULTS.mkdir(parents=True, exist_ok=True)
        n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in RESULTS.glob("DLM-*.json")] or [0])
        ident = f"DLM-{n:03d}"
        doc = {"id": ident, "experiment_id": ident, "campaign_id": "decision_llm_v1", "kind": "decision_llm_compare", "status": "completed",
               "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc,
               "rule": "A language model asked the generated sets' dev decisions through the decision_answer purpose (bands and name tokens, "
                       "never values), beside the recorded rule and the classical classifiers, against the benchmark labels. Dev only: "
                       "nothing here is a sealed verdict, and nothing is used."}
        path = RESULTS / f"{ident}_{model_name.replace(':', '-').replace('/', '-')}.json"
        path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
