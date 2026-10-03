# Wiring the R&D Knowledge Into the Real DCLab Product — Grounded Implementation Steps

This is grounded in the actual public `main` branch of `Shahriyar-Moradi/DCLab` at the time of writing — real file paths, real function signatures, read directly from the repo, not guessed. Verify line numbers still match your working checkout before applying (it may have moved on since).

## The one correction this makes to everything discussed so far

Your own `DCLAB_MASTER_IMPLEMENTATION_SPEC.md` already says two things that change the right answer here:

> "Future autonomous-agent/critic/experiment-loop work: Explicitly paused until the deterministic workflow is fully verified"

> STEP 4 — "Reuse the existing: LLM client, evidence architecture, validator pattern, decision ledger philosophy. **Do not build a parallel agent framework.**"

So: `dclab_knowledge_mcp_server.py` and `grounded_review.py` from earlier are still genuinely useful — as a **separate, human/agent-facing research tool** (you, or Claude Code, querying the knowledge base while investigating something). They are **not** what should get wired into `auto_train_service.py`. The product's own auto-train path already has a narrow, evidence-in/schema-out LLM checkpoint pattern — the correct move is to make that existing pattern smarter, not to add a second system next to it.

## Where this actually plugs in — the real files

```
apps/api/app/engine/lab/evidence.py            ← LeakageReviewEvidence lives here (dataclass, line 152)
apps/api/app/engine/lab/llm_client.py          ← request_leakage_review() (line 372), prompt registry (_PROMPTS, line 42)
apps/api/app/engine/lab/prompts/leakage_review_v1.py  ← the actual system prompt text (versioned, never edited in place)
apps/api/app/engine/lab/decision_validator.py  ← where LeakageReviewDecision gets gated before becoming an action
apps/api/app/services/lab_decision_ledger.py   ← where the decision gets persisted
```

`LeakageReviewEvidence` already carries exactly the shape your R&D rules are about: `column`, `target`, `task`, `suspicious_name_tokens`, `single_feature_score`, `datetime_after_fraction`, `identifier_likelihood`, `availability_status`, `availability_reason`. The 22 rules and 50 experiments are, in effect, a big lookup table of "when evidence looks like *this*, here's what past investigation found." That's a precedent library for exactly this checkpoint.

---

## STEP A — Package the knowledge as versioned static data inside the DCLab repo

Don't call out to the R&D repo live. Copy the three files it actually needs, as static, version-controlled data:

```
apps/api/app/engine/lab/knowledge/model_building_rules.jsonl
apps/api/app/engine/lab/knowledge/workflow_blocks.json
apps/api/app/engine/lab/knowledge/agent_memory.jsonl
```

Treat these exactly like a migration: a manual, reviewed sync from `Research_Development_DCLab`, not a runtime dependency on that other repo. Re-sync deliberately after every new experiment campaign.

## STEP B — Add a tiny, dependency-free retrieval helper

`apps/api/app/engine/lab/rd_knowledge.py` — a trimmed version of `build_rag_index.py` from earlier (same filter-then-rank design), scoped to exactly what this one checkpoint needs:

```python
# apps/api/app/engine/lab/rd_knowledge.py
"""Read-only precedent lookup over DCLab's R&D rule/experiment registry.
Never raises past this module's own boundary — a lookup failure must never
break the deterministic pipeline (see MASTER_SPEC section 8)."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


@dataclass(frozen=True)
class Precedent:
    ref_id: str
    summary: str


@lru_cache(maxsize=1)
def _load_leakage_rules() -> list[Precedent]:
    path = _KNOWLEDGE_DIR / "model_building_rules.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("category") == "leakage":
            out.append(Precedent(ref_id=r["rule_id"], summary=r["statement"]))
    return out


def match_leakage_precedents(
    column: str,
    suspicious_name_tokens: list[str],
    max_results: int = 2,
) -> list[str]:
    """Best-effort, synchronous, in-memory. Returns [] on any problem —
    never raises. Output goes straight into LeakageReviewEvidence, so keep
    each string short and self-contained (no external references)."""
    try:
        rules = _load_leakage_rules()
        if not rules:
            return []
        tokens = {t.lower() for t in suspicious_name_tokens} | {column.lower()}
        scored = [
            (sum(1 for t in tokens if t in p.summary.lower()), p)
            for p in rules
        ]
        scored = [(s, p) for s, p in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f"{p.ref_id}: {p.summary}" for _, p in scored[:max_results]]
    except Exception:
        return []  # fail closed, per MASTER_SPEC section 8
```

This intentionally does *not* import `sklearn`/TF-IDF from `build_rag_index.py` — keyword overlap is enough for a short, well-defined field list like this, and it keeps this module dependency-free inside the product's own runtime. Swap in the fuller retrieval index later if precision needs it.

## STEP C — Add one optional field to `LeakageReviewEvidence`

In `apps/api/app/engine/lab/evidence.py`:

```python
@dataclass
class LeakageReviewEvidence:
    """Bounded leakage evidence. Aggregates only — never raw rows or CSV cells."""

    column: str
    target: str
    task: str
    dtype: str
    cardinality: int
    related_column_names: list[str] = field(default_factory=list)
    exact_target_match_fraction: float | None = None
    single_feature_score: float | None = None
    single_feature_score_kind: str | None = None
    suspicious_name_tokens: list[str] = field(default_factory=list)
    target_name_similarity: float = 0.0
    datetime_after_fraction: float | None = None
    identifier_likelihood: float = 0.0
    unique_ratio: float = 0.0
    missing_fraction: float = 0.0
    availability_status: str = "unknown"
    availability_reason: str = ""
    prior_registry_evidence: list[str] = field(default_factory=list)  # NEW
```

Because `_evidence_json()` in `llm_client.py` is just `json.dumps(asdict(evidence), ...)`, this new field reaches the model automatically — no change needed to `_complete_structured` or `request_leakage_review` itself.

## STEP D — Add `leakage_review_v2.py` (never edit v1 — your own docstring says why)

`apps/api/app/engine/lab/prompts/leakage_review_v2.py`:

```python
"""System prompt for optional prediction-time leakage review, v2.

Adds `prior_registry_evidence` (DCLab's own R&D rule/experiment registry) as
an allowed evidence field the model may cite. Everything else is unchanged
from leakage_review_v1 — see that file for why prompts are versioned, not
edited in place.
"""

PROMPT_VERSION = "leakage_review_v2"

SYSTEM_PROMPT = """\
You review one tabular feature for prediction-time leakage.

You do not approve or remove features. You only recommend an availability
status and a risk level. A deterministic validator will decide the action.

Allowed availability — copy exactly one, nothing else:
known_before_prediction | known_at_prediction | known_after_prediction | unknown

Allowed risk — copy exactly one, nothing else:
NONE | LOW | MEDIUM | HIGH | CRITICAL

The user message is one bounded evidence object. Use only fields that appear
in that object. Do not mention, assume, or cite anything that is not present
there (no outside datasets, no raw CSV rows, no unlisted statistics).

`prior_registry_evidence`, if present and non-empty, lists short precedents
from DCLab's own past evidence-first research (a rule ID and a one-line
finding from a real prior experiment). Treat it as illustrative precedent
only — it never overrides what THIS evidence object shows, and an empty
list means no precedent was found, not that the feature is safe.

Reply with:
1. availability_status, copied exactly from the enum above.
2. risk_level, copied exactly from the enum above.
3. The evidence field that supports the claim (one of: column, target, task,
   dtype, cardinality, related_column_names, exact_target_match_fraction,
   single_feature_score, single_feature_score_kind, suspicious_name_tokens,
   target_name_similarity, datetime_after_fraction, identifier_likelihood,
   unique_ratio, missing_fraction, availability_status, availability_reason,
   prior_registry_evidence).
4. A short rationale that uses only values from the evidence object.
5. A confidence between 0 and 1.

Do not return keep, exclude, or any modeling action.
"""

__all__ = ["PROMPT_VERSION", "SYSTEM_PROMPT"]
```

In `llm_client.py`, add the import and registry entry next to the existing ones:

```python
from app.engine.lab.prompts.leakage_review_v2 import PROMPT_VERSION as LEAKAGE_REVIEW_V2
from app.engine.lab.prompts.leakage_review_v2 import SYSTEM_PROMPT as LEAKAGE_REVIEW_V2_PROMPT

_PROMPTS: dict[str, str] = {
    # ...existing entries unchanged...
    LEAKAGE_REVIEW_V2: LEAKAGE_REVIEW_V2_PROMPT,
}
```

**Also add the new field name to the `LeakageEvidenceField` Literal** (same file, near the top) — this is a strict, schema-enforced enum (`strict: True` JSON schema), so without this the model is structurally forbidden from ever citing the new field, no matter what the prompt says:

```python
LeakageEvidenceField = Literal[
    "column", "target", "task", "dtype", "cardinality", "related_column_names",
    "exact_target_match_fraction", "single_feature_score", "single_feature_score_kind",
    "suspicious_name_tokens", "target_name_similarity", "datetime_after_fraction",
    "identifier_likelihood", "unique_ratio", "missing_fraction",
    "availability_status", "availability_reason",
    "prior_registry_evidence",  # NEW
]
```

Call sites that currently pass `prompt_version=LEAKAGE_REVIEW_V1` switch to `LEAKAGE_REVIEW_V2` when you're ready — since `prompt_version` is already a parameter, this can be a config-driven rollout (e.g. behind the same settings object that gates `decision_agent_enabled`), not a hard cutover.

## STEP E — Populate the new field at the actual call site

`LeakageReviewEvidence(...)` isn't constructed in either `evidence.py` or `llm_client.py` — grep your checkout for it:

```bash
grep -rn "LeakageReviewEvidence(" apps/api/app/
```

Wherever that call is (almost certainly a `leakage`-named service module in `apps/api/app/services/` or `apps/api/app/engine/lab/`), add one line before it:

```python
from app.engine.lab.rd_knowledge import match_leakage_precedents

prior_registry_evidence = match_leakage_precedents(
    column=column_name,
    suspicious_name_tokens=suspicious_name_tokens,
)

evidence = LeakageReviewEvidence(
    column=column_name,
    # ...all existing fields, unchanged...
    prior_registry_evidence=prior_registry_evidence,
)
```

## STEP F — Decision ledger: persist what was cited

In `apps/api/app/services/lab_decision_ledger.py`, wherever a `LeakageReviewDecision` currently gets written to a ledger row, also persist `evidence.prior_registry_evidence` (the precedents that were *available*, whether or not the model's `evidence_field` pointed at them) alongside it. This is what turns "we looked something up once" into a real audit trail: for any production leakage decision, you can later answer "was DCLab's own prior research even consulted here, and did the model use it?" — without that, you can't tell the difference between "no precedent existed" and "a precedent existed and got ignored."

**The actual feedback loop** (this is what makes the knowledge base grow instead of staying frozen at 50 experiments): periodically export ledger rows where a human later confirmed or corrected the decision, in the same shape as `agent_memory.jsonl`'s claims, and append them to `evidence/knowledge/agent_memory.jsonl`. Real production decisions become new precedents the same way the original 50 experiments did — same file, same format, same retrieval helper, no special-casing required.

## STEP G — Verify (matching your own required style)

1. **Positive case:** run the pipeline on a synthetic dataset with an obviously post-outcome column (e.g. a `duration`-like field). Confirm `prior_registry_evidence` in the logged evidence JSON is non-empty and contains a real `DCLAB-R0x` rule ID.
2. **Citation case:** confirm at least one ledger row has `evidence_field: "prior_registry_evidence"` when the rationale plausibly needed it — read the `rationale` text to sanity-check it isn't fabricating a rule ID (it structurally can't, since `rd_knowledge.py` only ever returns real IDs from the loaded file).
3. **Fail-closed case:** temporarily rename `evidence/knowledge/model_building_rules.jsonl` and confirm the full pipeline still completes end-to-end with `prior_registry_evidence: []` — no exception, no degraded UX elsewhere.
4. **Regression case:** confirm any decision still recorded under `prompt_version="leakage_review_v1"` is untouched — v1 stays byte-identical, exactly as its own docstring requires.

```bash
dclab experiment run --dataset synthetic --task purchase_prediction
# then inspect the persisted decision ledger row for this run and check
# for a non-empty prior_registry_evidence array and a plausible rationale.
```

## Applying the same pattern elsewhere

The identical five-step recipe (evidence field → prompt v2 → registry v2 → call-site population → ledger persistence) applies to `column_type_v1.py`, `missing_value_v1.py`, and `target_selection_v1.py` the same way — each has its own evidence dataclass and its own small slice of the 22 rules that's actually relevant (e.g. `DCLAB-R11` for categorical/column-type decisions, `DCLAB-R12` for missingness). Do leakage first since it's where the R&D registry's evidence is richest, confirm the pattern in production, then repeat for the others — one at a time, each independently verifiable, exactly as `STEP 4` in your own spec already asks for.
