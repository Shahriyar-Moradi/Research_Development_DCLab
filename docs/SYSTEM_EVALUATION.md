# How DCLab R&D earns trust

The question is whether the system makes useful, evidence-supported decisions on
new work. Passing software tests is necessary, but it does not establish that an
agent's next experiment is a good choice or that a model will work in production.

## Run the audit now (DCLab Bench, 2026-11-03)

```bash
make bench-read   # every layer's state and why, from the recorded results (no run)
make bench        # run the free suites of every layer, record the run in evidence/bench/runs/, then read
```

DCLab Bench reads every result in `evidence/campaigns/` into one shape (`dclab_core.bench.BenchRow`), each campaign in its
layer (software, agents, lead agent, LLM roles, team, decision models, retrieval and memory, data-science quality, system
tasks, external), and says each layer red, amber, thin, green or empty with its reasons. There is no total score. Sealed
results are shown and never rerun. The Admin page's Benchmark tab shows the same. See `docs/guides/BENCHMARK.md`.

The commands this page used to name (`make rd-eval`, `dclab_rnd.system_eval`, `make notebook-eval`, `make
agent-live-eval`, `make agent-trace-eval`) no longer exist; the paragraphs below describe the earlier pilots as history.

## Live specialist and real-notebook pilots (history)

Two additive, frozen development suites tested actual behavior rather than
counting archived experiments as agent competence (their commands were removed):

The notebook suite reads the existing HyperAck classification and churn demo
notebooks without executing or editing them. `evaluation/notebook_cases_v1.json`
specifies code-cell hashes, expected signals and rationales. The runner checks
the existing notebook copilot's detection and its retrieved proof rule. A changed
cell cannot silently keep an old label. The suite is small and researcher-authored;
it is a *development pilot*, not a blinded accuracy estimate. Full report JSON is
written under `output/evaluation/` with a unique filename and cannot be overwritten.

The live suite uses the configured OpenAI model and actual NOOA agent classes on
five fixed cases in `evaluation/live_cases_v1.json`. It records the typed response,
provider failures, calls, available token usage and conservative hard checks.
It does not train a model or execute model-generated code. A provider failure is
recorded as a failed case, not replaced by a synthetic answer. The automated
checks only screen for necessary conditions; an expert must still judge whether
the decision is scientifically correct. The same cases must not later be used as
an unseen test set after prompt or model changes informed by these results.

`make agent-trace-eval` audits the **actual saved** planner, reviewer, designer,
critic and curator outputs in historical trajectories. It checks typed schema
compatibility, prior-success citations, explicit comparison references,
policy-blocked feature ancestors, first baselines, failure acknowledgment and
evidence-backed lessons. These are necessary contract checks, not an estimate
of decision accuracy. Historical traces may use an older schema; a mismatch is
reported rather than retroactively rewritten. A decision that passes every
contract check can still be scientifically poor.

For a deterministic execution check, rerun a fingerprinted trial with the
current worker into a **new** output directory and compare it with
`python -m dclab_rnd.replay_compare`. If the worker or catalog hash has
changed, label the result a cross-version reproduction, never an exact-version
replay—even when every prediction is identical.

For these notebooks, the owner still needs to specify the intended prediction
moment and target definition. In particular, a column named `cancellation_date`
is a *candidate* availability problem, not proven leakage from its name alone.
Proof snippets must cite a rule relevant to the exact claim; a citation that
merely contains the word “leakage” is insufficient. The safe next step is to
correct these errors and then evaluate a source-held-out set of notebooks with
blinded labels and a human user study.
It independently checks:

1. Every saved out-of-fold prediction has a valid probability and target; each
   repeat covers every selected row once.
2. Recomputed fold and pooled-repeat ROC-AUC, average precision, log loss and
   Brier score agree with reported values. Overall reported metrics agree with
   the arithmetic mean of the saved folds, which is how this worker defines them.
3. The duplicate-row group remains in one fold per repeat, and the recorded
   split hash can be reconstructed from saved row IDs.
4. Dataset hashes match both the recipe and the currently available source files.
   A changed source means an old trial cannot be replayed from the current cache.
5. A completed trial has its result, recipe and predictions; the plan is identical
   in all three records; policy-blocked fields do not enter the model.
6. Proposals, critiques and lessons cite prior successful trial IDs. Failed trials
   cannot support a lesson. Completed cycles have a synthesis.
7. Research remains marked as unapproved for production and unready for SLM
   training.

The report distinguishes `repeat_summary_uses_fold_mean` from a numerical
metric mismatch. A fold average can be a legitimate *fold summary*; it is not the
pooled score obtained from concatenating all held-out predictions in a repeat.
This distinction especially matters for ROC-AUC and average precision.

## Four separate evaluation questions

| Question | Primary measure | How to establish it |
|---|---|---|
| Does the software obey its rules? | Critical invariant violation count | Unit, integration and adversarial tests; archived-evidence audit |
| Is the scientific evidence internally valid? | Fraction of trials with fully reconstructed metrics, splits, policy and provenance | Independent audit above; a reviewer checks prediction-time meaning and raw-source lineage |
| Does the agent choose good actions? | Valid task completion on previously unseen cases | Freeze a human-reviewed case set; execute complete trajectories; compare with a fixed baseline |
| Does it help a data scientist? | Correctly completed user tasks, time, and trust | Observe users solving real notebook tasks with and without assistance |

The overall release decision is an **AND gate** over these questions. Do not
average a safety failure into a high score for writing quality or speed.
Historical experiment rows are often correlated; 579 registry records are not 579
independent tests of agent decision quality.

## A frozen agent benchmark

Construct 100 or more cases from datasets and notebooks excluded by source from
training, prompt development and prior agent research. Include clean cases and
deliberate traps: a post-outcome feature, a scaler fitted before splitting, the
same customer on both sides of a split, a temporal prediction with a random split,
an apparently high score after repeated holdout peeking, missing fields at scoring
time, a failed training call, and a challenger that improves AUC but worsens log
loss. Give each case a prediction contract, initial notebook state, allowed
actions, observation stream, and an expert-reviewed set of acceptable decisions.
Several actions may be acceptable; grade the action and its consequences, not
the wording of the explanation.

For each case, record: completion; invalid or unsafe action; whether the proposed
experiment can run; factual support of every metric and evidence ID; whether it
stops when evidence is insufficient; number of model calls; tokens; wall time;
and human correction time. Run the same cases against a fixed prompted base model,
the current agent, and any fine-tuned SLM. Use identical action capabilities and
budgets. Every error stays in the result, including provider and executor failures.
If an LLM helps review explanations, compare its decisions against blinded human
labels; it is not the sole judge of its own system.

Grade each specialist on the decision it owns:

| Specialist | Measurable question |
|---|---|
| Research planner | Did it define a feasible prediction contract, evaluation design and budget, or ask when facts were missing? |
| Data and leakage reviewer | Did it detect known leakage traps without falsely excluding safe columns? Report recall and false-alarm rate by trap type. |
| Experiment designer | Was the proposal executable, distinct from prior trials and capable of answering the stated question? |
| Results critic | Did every number and claimed difference match saved evidence? Did it catch negative results and uncertainty? |
| Knowledge curator | Is each reusable rule scoped to its evidence and accompanied by a relevant counterexample or open test? |

Also test the entire loop when the provider rejects a call, an experiment fails,
the dataset hash changes, the same case is resumed, or a suspicious instruction
appears inside dataset text. These cases measure recovery and authority boundaries.

Predeclare success criteria before seeing the hidden cases. An initial proposed
gate is 100% pass on hard safety tests, zero unsupported evidence IDs, at least
90% valid case completion, a 95% Wilson lower bound above 80%, and no material
regression against the baseline on paired cases. Require a human review of every
severe failure and of a random sample of apparent successes. These are research
targets, not proven guarantees. Publish per-dataset and per-failure-type results;
an aggregate score can conceal an entire weak category.

Use a genuinely fresh confirmation set after the agent and the benchmark are
frozen. A case that influenced prompts, training, model selection or benchmark
thresholds belongs to development, even if it appears in a separate file.
For predictive-model claims, confirmation should also reflect later time, unseen
entities and unavailable inputs where those are real operating conditions.

## What a score proves

If 0 failures appear in 100 independent cases, the one-sided 95% upper bound on
the failure probability is still about 3%. With 300 such cases it is about 1%.
These bounds only describe the sampled case distribution; they say nothing about
unrepresented workflows or future data changes. Report both numerator and
denominator, the case definition, and a confidence interval. Repeated folds over
the same people are correlated and must not be counted as independent cases.

No current archive supplies a blinded agent decision benchmark, verified user
task study, or fresh production confirmation set. The scorecard reports those
dimensions as `null`, and `overall_release_ready` remains false. A failed API
request counts against cycle completion. Resolving provider access and testing
recovery are parts of the system evaluation.

NIST's [AI RMF Measure function](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/)
calls for repeatable testing, validation and monitoring. Google's
[Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
explain why independent pipeline checks and training/serving consistency matter.
Official OpenAI documentation likewise separates [agent trace analysis from
repeatable dataset evaluations](https://developers.openai.com/api/docs/guides/agent-evals)
and warns that automated scores need calibration against
[human review](https://developers.openai.com/api/docs/guides/evaluation-best-practices).
The repo-owned manifests and graders stay portable: the OpenAI-hosted Evals
platform has a [published shutdown schedule](https://developers.openai.com/api/docs/deprecations),
so this workflow does not depend on that dashboard.
