#!/usr/bin/env python3
"""Build the v3 SFT corpus: self-contained, evidence-grounded, critic-gated.

Why v3 exists
-------------
v1 (``build_sft_dataset.py``) asked questions like "What did EXP-023 find?" without
telling the model what the dataset predicts, what the stage does, or what was
compared. Training on that teaches recall of IDs, not reasoning.

Every v3 example that is about an experiment carries three context blocks pulled
from real repository data (never invented), in the user turn:

1. **Dataset card** - what the data is, what it predicts, size, decision-time contract.
2. **Stage card**   - what this workflow stage does methodologically, and its question.
3. **Setup**        - what this run actually compared, with the numbers.

This is RAFT-style data (retrieval-augmented fine-tuning): the model learns to
reason from evidence placed in front of it, which is exactly how the DCLab agent
will be prompted at run time with records from ``dclab_rnd.evidence_index``.

Quality gates
-------------
* Critic challenges that a deterministic recomputation disproves are dropped
  (``dclab_rnd.critic_gate``); they are never used as answers.
* Provenance noise (hashes, absolute paths, machine info) is never emitted, and a
  final scan rejects any example containing an absolute path or a SHA-256 string.
* Exact duplicates (same normalized user turn) are removed.
* The validation split is **grouped by dataset**: whole datasets are held out, so
  validation measures transfer to an unseen dataset instead of memorization.

Task families
-------------
explain_experiment, critique_claim, apply_selection_rule, leakage_judgment,
grounded_qa (RAFT with distractors), rule_reasoning, workflow_steps.

Usage::

    python research/llm-fine-tuning/sft/build_sft_dataset_v3.py                 # writes sft/out_v3/
    python research/llm-fine-tuning/sft/build_sft_dataset_v3.py --check         # exit 1 if committed output is stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]  # research/llm-fine-tuning/sft -> repo root
SFT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dclab_rnd.critic_gate import gate_result, recompute  # noqa: E402
from dclab_rnd.evidence_index import (  # noqa: E402
    STAGE_CARDS,
    build_records,
    dataset_cards,
    experiment_context,
    experiment_files,
)

OUT_DIR = SFT_DIR / "out_v3"
SEED = 20260918
VAL_DATASETS = ("heart_disease", "spambase", "ecommerce_clothing_reviews")
GENERAL_VAL_FRACTION = 0.12

SYSTEM_PROMPT = (
    "You are DCLab's senior data-science and ML-engineering assistant. Reason only from the "
    "evidence given in the conversation. Structure answers as: Evidence (what was measured), "
    "Interpretation (what it means and how uncertain it is), Decision (what to do now), "
    "Risks (what is missing or could be wrong), and Next test (the smallest falsifiable "
    "experiment). Never claim production readiness from benchmark evidence alone."
)

_ABS_PATH = re.compile(r"(/Users/|/home/|C:\\\\|/tmp/)")
_SHA = re.compile(r"\b[0-9a-f]{40,64}\b")


# --------------------------------------------------------------------------- helpers


def _context_block(ctx: dict[str, str]) -> str:
    return (
        f"### Dataset\n{ctx['dataset_card']}\n\n"
        f"### Stage\n{ctx['stage_card']}\n\n"
        f"### What this run compared\n{ctx['setup_summary']}"
    )


def _answer(evidence: str, interpretation: str, decision: str, risks: str, next_test: str) -> str:
    return (
        f"**Evidence.** {evidence.strip()}\n\n"
        f"**Interpretation.** {interpretation.strip()}\n\n"
        f"**Decision.** {decision.strip()}\n\n"
        f"**Risks.** {risks.strip()}\n\n"
        f"**Next test.** {next_test.strip()}"
    )


def _example(task: str, user: str, assistant: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"task": task, "user": user.strip(), "assistant": assistant.strip(), "metadata": {"task": task, **meta}}


def _claims_text(result: dict[str, Any], kinds: tuple[str, ...] | None = None) -> str:
    claims = result.get("claims") or []
    return " ".join(c.get("statement", "") for c in claims if kinds is None or c.get("kind") in kinds).strip()


def _limitations(result: dict[str, Any]) -> list[str]:
    return sorted({lim for c in result.get("claims") or [] for lim in c.get("limitations", [])})


def _next_test(result: dict[str, Any]) -> str:
    review = ((result.get("llm_review") or {}).get("review")) or {}
    nxt = review.get("next_experiment") or {}
    if nxt.get("title") or nxt.get("success_gate"):
        title = nxt.get("title", "Confirmatory test")
        gate = nxt.get("success_gate", "")
        return f"{title}. Success gate: {gate}".strip()
    defaults = {
        "data_understanding": "Write a column-level decision-time availability table and re-profile on a time-ordered or group-ordered slice.",
        "leakage_audit": "Get a signed data-dictionary entry for each candidate column showing when it is created, then rerun the safe-vs-unsafe ablation on identical folds.",
        "feature_engineering": "Repeat the recipe comparison with repeated CV on identical folds and accept the richer recipe only if the paired fold-wise gain stays positive.",
        "model_selection": "Re-run the top two families on a different seed or a time/group slice; keep the simpler one if the gap is inside fold noise.",
        "optimization_reliability": "Confirm on fresh data or a later time slice, check calibration and the business threshold, and define the rollback rule before any promotion.",
    }
    return defaults.get(result.get("kind", ""), "Design the smallest experiment that could falsify the current decision.")


# --------------------------------------------------------------------------- task builders


def explain_experiment(result: dict[str, Any], ctx: dict[str, str], gate: dict[str, Any]) -> dict[str, Any]:
    review = ((result.get("llm_review") or {}).get("review")) or {}
    kept = [c for c in gate["challenges"] if c["gate"] != "contradicted"]
    risks = _limitations(result)
    risks += [c["issue"] for c in kept if c.get("severity") == "high"][:2]
    check = gate.get("rule_check")
    interpretation = review.get("interpretation") if review and not gate["contradicted"] else ""
    if not interpretation:
        interpretation = (
            "The holdout score is one draw on rows used exactly once; its interval, not the point value, is the "
            "honest summary, and it is still benchmark evidence rather than production validation."
            if result.get("kind") == "optimization_reliability" else
            "These are training-CV development numbers for one dataset; they support a provisional choice "
            "for the next stage, not a general or production claim."
        )
        if check:
            interpretation += (
                f" Re-applying the stated rule ({check['rule']}) gives '{check['recomputed']}', which "
                f"{'matches' if check['consistent'] else 'does NOT match'} the recorded choice '{check['recorded']}'."
            )
    evidence = _claims_text(result, ("fact", "risk")) or ctx["setup_summary"]
    recommendation = _claims_text(result, ("recommendation", "decision"))
    decision = review.get("decision") if review and not gate["contradicted"] else ""
    if not decision:
        if recommendation:
            decision = recommendation
            if check and check["consistent"]:
                decision += " The selection rule was re-applied to the recorded numbers and confirms this choice; treat it as provisional until a confirmatory test passes."
        else:
            decision = "Carry these facts forward as context for the next stage; no model choice is made at this stage."
    if recommendation and evidence == ctx["setup_summary"]:
        evidence = ctx["setup_summary"]
    user = (
        f"{_context_block(ctx)}\n\n"
        "Explain what this run shows, how much to trust it, and what to do next."
    )
    answer = _answer(
        evidence=evidence,
        interpretation=interpretation,
        decision=decision,
        risks=" ".join(risks) if risks else "No additional risks were recorded beyond the general limits of training-CV evidence.",
        next_test=_next_test(result),
    )
    return _example("explain_experiment", user, answer, _meta(result))


def critique_claims(result: dict[str, Any], ctx: dict[str, str], gate: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    claims = {c.get("claim_id"): c for c in result.get("claims") or []}
    kept_by_claim: dict[str, list[dict[str, Any]]] = {}
    for challenge in gate["challenges"]:
        if challenge["gate"] == "contradicted":
            continue
        kept_by_claim.setdefault(challenge.get("claim_id"), []).append(challenge)
    for claim_id, challenges in sorted(kept_by_claim.items(), key=lambda kv: str(kv[0])):
        claim = claims.get(claim_id)
        if not claim:
            continue
        user = (
            f"{_context_block(ctx)}\n\n### Claim under review\n{claim['statement']}\n\n"
            "Act as a skeptical reviewer. Which parts of this claim are not supported by the evidence above, "
            "and how severe is each gap?"
        )
        lines = [f"- **{c.get('severity', 'unrated')}**: {c['issue']}" for c in challenges]
        answer = "Gaps between the claim and the evidence:\n" + "\n".join(lines)
        out.append(_example("critique_claim", user, answer, {**_meta(result), "claim_id": claim_id}))
    return out


def apply_selection_rule(result: dict[str, Any], ctx: dict[str, str], gate: dict[str, Any]) -> dict[str, Any] | None:
    check = gate.get("rule_check")
    if not check:
        return None
    user = (
        f"{_context_block(ctx)}\n\n"
        f"Apply the declared rule exactly ({check['rule']}). Which option does it select, and why? "
        "Note: a recipe may be *named* 'selected' (a feature-selection subset); that name is not the rule's choice."
    )
    detail = ""
    if "eligible" in check:
        detail = f" Options within the tolerance of the best mean: {', '.join(check['eligible'])}; the rule then takes the one with the fewest features."
    if "scores" in check:
        detail = " Penalized scores: " + ", ".join(f"{k} {v:.4f}" for k, v in sorted(check["scores"].items(), key=lambda kv: -kv[1])) + "."
    answer = (
        f"The rule selects **{check['recomputed']}**.{detail} "
        f"This {'matches' if check['consistent'] else 'does not match'} the recorded choice ({check['recorded']}). "
        "The choice is provisional: it is a training-CV ranking on one dataset, so the next stage should confirm it "
        "on identical folds rather than treat it as a universal winner."
    )
    return _example("apply_selection_rule", user, answer, _meta(result))


def leakage_judgments(result: dict[str, Any], ctx: dict[str, str]) -> list[dict[str, Any]]:
    if result.get("kind") != "leakage_audit":
        return []
    ev = result.get("evidence") or {}
    declared = ev.get("declared_leakage_features") or []
    candidates = [c for c in ev.get("heuristic_review_candidates") or [] if c.get("feature") not in declared]
    lift = (ev.get("safe_vs_unsafe_training_cv") or {}).get("apparent_auc_lift", ev.get("apparent_lift"))
    metric_name = {"roc_auc": "ROC-AUC", "average_precision": "average precision", "macro_f1": "macro-F1",
                   "mae": "MAE"}.get(result.get("primary_metric", "roc_auc"), result.get("primary_metric", "score"))
    lower_better = result.get("primary_metric") in ("mae", "rmse", "log_loss", "brier")
    out = []
    for column in declared:
        user = (
            f"{_context_block(ctx)}\n\nShould `{column}` be used as an input to a model that will be deployed "
            "at the declared prediction moment?"
        )
        answer = _answer(
            evidence=(f"Including the blocked column(s) made the validation {metric_name} look better by {abs(lift):.4g}"
                      + (" (lower is better for this metric)." if lower_better else ".")
                      if isinstance(lift, (int, float)) and lift > 0 else
                      f"Including the blocked column(s) did not improve validation {metric_name} (change {lift:+.4g}); it is excluded on decision-time grounds alone."
                      if isinstance(lift, (int, float)) else "The column is declared unavailable at decision time."),
            interpretation=(ev.get("policy_rationale") or "The column is not available when the prediction is made.")
            + " A large score gain from such a column is the size of the illusion, not evidence of value.",
            decision=f"Exclude `{column}` from every deployable model and from feature engineering built on top of it.",
            risks="Columns derived from it (ratios, interactions, aggregates) carry the same leak and must be removed too; confirm with the data owner when the value is written.",
            next_test="Get a data-dictionary entry with the creation timestamp of the column and check a production-like request trace to confirm it is absent at scoring time.",
        )
        out.append(_example("leakage_judgment", user, answer, {**_meta(result), "column": column, "label": "exclude"}))
    for cand in candidates[:3]:
        column = cand.get("feature")
        reasons = ", ".join(cand.get("reasons", [])) or "heuristic flag"
        user = (
            f"{_context_block(ctx)}\n\nThe automated detector flagged `{column}` ({reasons}). Should we drop it?"
        )
        answer = _answer(
            evidence=f"`{column}` was flagged by heuristics ({reasons})"
            + (f", single-feature AUC {cand['univariate_auc']:.3f}" if isinstance(cand.get("univariate_auc"), (int, float)) else "")
            + ".",
            interpretation="Heuristics only propose a review. Strong predictive power alone is not proof of leakage; only the time at which the value exists relative to the decision can confirm it.",
            decision=f"Do not auto-delete `{column}`. Keep it out of the deployable candidate until a human confirms its decision-time availability, and record the decision.",
            risks="Deleting legitimate strong features loses real signal; keeping a true leak inflates every downstream result.",
            next_test=f"Run a safe-vs-unsafe ablation for `{column}` on identical folds and obtain its creation timestamp from the source system.",
        )
        out.append(_example("leakage_judgment", user, answer, {**_meta(result), "column": column, "label": "review"}))
    if not declared and not candidates:
        user = f"{_context_block(ctx)}\n\nAre there columns we must block before modeling?"
        answer = _answer(
            evidence="No column was declared post-outcome for this dataset and no heuristic candidate was raised.",
            interpretation="Absence of flags is weak evidence: heuristics miss leaks whose signal is moderate, and the decision-time contract above still has assumptions.",
            decision="Proceed with all columns provisionally, but keep the decision-time contract and its assumptions attached to the model card.",
            risks="Undeclared timing assumptions (when each field is written) remain the main risk.",
            next_test="Ask the data owner for creation timestamps of the five most important features and re-check them against the prediction moment.",
        )
        out.append(_example("leakage_judgment", user, answer, {**_meta(result), "label": "none_found"}))
    return out


def rule_reasoning(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rules:
        category = r["category"].replace("_", " ")
        user = (
            f"A teammate wants to skip this practice to move faster: \"{r['statement']}\" "
            "What could go wrong, how would we notice, and what is the cheapest check?"
        )
        answer = (
            f"**Why it matters.** {r['why']}\n\n"
            f"**How you notice it was skipped.** {r.get('failure_signal', '')}\n\n"
            f"**Cheapest check.** {r.get('next_test', '')}\n\n"
            f"**Confidence.** {r.get('confidence', '')} (rule {r['rule_id']}, {category})."
        )
        out.append(_example("rule_reasoning", user, answer, {"rule_id": r["rule_id"], "category": r["category"], "dataset": None}))
    return out


def workflow_steps(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for b in blocks:
        user = f"List, in order, the steps of the '{b['name']}' block of a reliable model-building workflow, and say why the order matters."
        steps = "\n".join(f"{i}. {step}" for i, step in enumerate(b["flow"], 1))
        answer = (
            f"{steps}\n\nThe order matters because each step fixes information the next step is allowed to use; "
            "doing a later step first lets decisions leak information they should not have (for example, choosing "
            f"features before the split is locked). This is workflow block {b['id']}."
        )
        out.append(_example("workflow_steps", user, answer, {"workflow_id": b["id"], "dataset": None}))
    return out


def _leak_answer(target: dict[str, Any]) -> str:
    meta = target["metadata"]
    cols = ", ".join(f"`{c}`" for c in meta.get("columns", []))
    lift = meta.get("apparent_lift")
    rationale = (meta.get("rationale") or "it is not available at the prediction moment").rstrip(". ")
    text = f"No. According to [{target['record_id']}], {cols} must be excluded from deployable models because {rationale}."
    if isinstance(lift, (int, float)):
        text += f" Including it made validation look {lift:+.4f} better, which is the size of the illusion, not real skill."
    text += " Any feature derived from it must be dropped too."
    return text


def grounded_qa(records: list[dict[str, Any]], results: list[dict[str, Any]], cards: dict[str, dict[str, Any]],
                rng: random.Random) -> list[dict[str, Any]]:
    """RAFT examples: the relevant record plus distractors; the answer uses and cites only the right one.

    Experiment questions use *hard negatives*: the same stage on other datasets, so the model must
    match the dataset rather than the topic.
    """
    general = [r for r in records if r["type"] in ("rule", "finding", "workflow")]
    out = []
    for target in [r for r in records if r["type"] in ("leakage_precedent", "finding")]:
        distractors = [r for r in general + [x for x in records if x["type"] == "leakage_precedent"] if r["record_id"] != target["record_id"]]
        rng.shuffle(distractors)
        docs = [target] + distractors[:2]
        rng.shuffle(docs)
        if target["type"] == "leakage_precedent":
            cols = ", ".join(target["metadata"].get("columns", []))
            question = f"Can we use {cols} as features for the {target['metadata'].get('dataset')} model?"
            answer = _leak_answer(target)
        else:
            question = f"What does our evidence say about this: {target['title'].lower()}?"
            answer = f"According to [{target['record_id']}]: {target['text']}"
        context = "\n\n".join(f"[{d['record_id']}] {d['text']}" for d in docs)
        user = f"Retrieved evidence:\n\n{context}\n\nQuestion: {question}\nAnswer only from the relevant record and cite its ID."
        out.append(_example("grounded_qa", user, answer + "\n\nThe other retrieved records are not needed for this question.",
                            {"record_id": target["record_id"], "dataset": target["metadata"].get("dataset")}))

    by_id = {r["record_id"]: r for r in records}
    questions = {
        "data_understanding": "What did profiling find about the {name} data before modeling?",
        "leakage_audit": "Which columns are unsafe for the {name} model, and how much did they inflate validation?",
        "feature_engineering": "Which feature recipe was chosen for {name}, and on what evidence?",
        "model_selection": "Which algorithm family should we carry forward for {name}?",
        "optimization_reliability": "How did the final {name} model do on the locked holdout, and how certain is that number?",
    }
    for result in results:
        target = by_id.get(result.get("experiment_id"))
        kind = result.get("kind")
        if not target or kind not in questions:
            continue
        negatives = [r for r in records if r["type"] == "experiment" and r["metadata"].get("stage") == kind
                     and r["metadata"].get("dataset") != result.get("dataset")]
        rng.shuffle(negatives)
        docs = [target] + negatives[:2]
        rng.shuffle(docs)
        name = cards.get(result.get("dataset"), {}).get("name", result.get("dataset"))
        question = questions[kind].format(name=name)
        context = "\n\n".join(f"[{d['record_id']}] {d['text']}" for d in docs)
        user = f"Retrieved evidence:\n\n{context}\n\nQuestion: {question}\nAnswer only from the record about this dataset and cite its ID."
        claims = _claims_text(result) or "The record lists no claims."
        limits = _limitations(result)
        answer = f"From [{target['record_id']}] ({name}): {claims}"
        if limits:
            answer += " Caveat: " + " ".join(limits)
        answer += f"\n\nThe other records describe different datasets ({', '.join(sorted(d['metadata'].get('dataset') for d in docs if d is not target))}) and do not answer this question."
        out.append(_example("grounded_qa", user, answer, {**_meta(result), "record_id": target["record_id"], "hard_negatives": True}))
    return out


def _meta(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": result.get("experiment_id"),
        "campaign": result.get("campaign_id"),
        "dataset": result.get("dataset"),
        "stage": result.get("kind"),
        "task_type": result.get("task_type", "binary"),
    }


# --------------------------------------------------------------------------- assembly


def _clean(example: dict[str, Any]) -> bool:
    blob = example["user"] + "\n" + example["assistant"]
    return not _ABS_PATH.search(blob) and not _SHA.search(blob) and 40 <= len(example["assistant"]) <= 6000


def _split(example: dict[str, Any]) -> str:
    dataset = example["metadata"].get("dataset")
    if dataset:
        return "val" if dataset in VAL_DATASETS else "train"
    digest = int(hashlib.sha256(example["user"].encode("utf-8")).hexdigest()[:8], 16)
    return "val" if (digest % 1000) / 1000 < GENERAL_VAL_FRACTION else "train"


def to_chat(ex: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["user"]},
            {"role": "assistant", "content": ex["assistant"]},
        ],
        "metadata": ex["metadata"],
    }


def to_alpaca(ex: dict[str, Any]) -> dict[str, Any]:
    return {"instruction": SYSTEM_PROMPT, "input": ex["user"], "output": ex["assistant"], "metadata": ex["metadata"]}


def to_sharegpt(ex: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": SYSTEM_PROMPT,
        "conversations": [{"from": "human", "value": ex["user"]}, {"from": "gpt", "value": ex["assistant"]}],
        "metadata": ex["metadata"],
    }


FORMATS = {"chat": to_chat, "alpaca": to_alpaca, "sharegpt": to_sharegpt}


def build(root: Path = ROOT) -> tuple[dict[str, str], dict[str, Any]]:
    rng = random.Random(SEED)
    cards = dataset_cards(root)
    records = build_records(root)
    examples: list[dict[str, Any]] = []
    gate_stats = Counter()
    results = [json.loads(path.read_text(encoding="utf-8")) for path in experiment_files(root)]
    for result in results:
        ctx = experiment_context(result, cards)
        gate = gate_result(result)
        gate_stats["contradicted_challenges_dropped"] += gate["contradicted"]
        gate_stats["challenges_kept"] += gate["kept"]
        examples.append(explain_experiment(result, ctx, gate))
        examples += critique_claims(result, ctx, gate)
        selection = apply_selection_rule(result, ctx, gate)
        if selection:
            examples.append(selection)
        examples += leakage_judgments(result, ctx)
    rules = [json.loads(line) for line in (root / "evidence/knowledge" / "model_building_rules.jsonl").read_text().splitlines() if line.strip()]
    blocks = json.loads((root / "evidence/knowledge" / "workflow_blocks.json").read_text())
    examples += rule_reasoning(rules)
    examples += workflow_steps(blocks)
    examples += grounded_qa(records, results, cards, rng)

    seen, unique, rejected = set(), [], 0
    for ex in examples:
        key = re.sub(r"\s+", " ", ex["user"].lower())
        if key in seen:
            continue
        seen.add(key)
        if not _clean(ex):
            rejected += 1
            continue
        unique.append(ex)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for ex in unique:
        splits[_split(ex)].append(ex)

    files: dict[str, str] = {}
    for fmt, convert in FORMATS.items():
        for split, items in splits.items():
            files[f"{split}.{fmt}.jsonl"] = "".join(json.dumps(convert(ex), ensure_ascii=False) + "\n" for ex in items)

    manifest = {
        "schema_version": 3,
        "seed": SEED,
        "system_prompt": SYSTEM_PROMPT,
        "total": len(unique),
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "validation_policy": f"datasets held out entirely: {', '.join(VAL_DATASETS)}; dataset-free examples use a {GENERAL_VAL_FRACTION:.0%} hash split",
        "by_task": dict(sorted(Counter(ex["task"] for ex in unique).items())),
        "by_split_and_task": {s: dict(sorted(Counter(ex["task"] for ex in items).items())) for s, items in splits.items()},
        "by_campaign": dict(sorted(Counter(str(ex["metadata"].get("campaign")) for ex in unique).items())),
        "quality_gates": {
            "duplicates_removed": len(examples) - len(seen),
            "rejected_by_path_or_hash_or_length": rejected,
            **dict(gate_stats),
        },
        "formats": {
            "chat": "OpenAI / TRL messages format",
            "alpaca": "instruction / input / output",
            "sharegpt": "system + conversations[from, value]",
        },
        "stage_cards": {k: v["title"] for k, v in STAGE_CARDS.items()},
    }
    files["MANIFEST.json"] = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    return files, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--check", action="store_true", help="exit 1 if the files in --out-dir are stale")
    args = parser.parse_args(argv)
    files, manifest = build(ROOT)
    if args.check:
        stale = [name for name, text in files.items() if not (args.out_dir / name).exists() or (args.out_dir / name).read_text(encoding="utf-8") != text]
        if stale:
            print("Stale SFT v3 files: " + ", ".join(stale) + ". Run: python research/llm-fine-tuning/sft/build_sft_dataset_v3.py")
            return 1
        print("SFT v3 corpus is current.")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (args.out_dir / name).write_text(text, encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("total", "train", "val", "by_task", "quality_gates")}, indent=2))
    print(f"Written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
