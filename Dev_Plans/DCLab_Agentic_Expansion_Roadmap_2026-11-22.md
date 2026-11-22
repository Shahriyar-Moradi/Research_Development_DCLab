# Where DCLab can give its agents more responsibility

**Product improvement roadmap · 22 November 2026**

The most valuable first decision is small. When a team request stalls, the agent should decide what would unblock it. Sometimes the answer already exists in project memory. Sometimes the right move is to ask Sales, inspect a table or tell Business that the available evidence cannot answer the question. The workspace can measure the value of that choice from the resulting task and report.

Use the [implementation and live test plan](Team_Workspace_Implementation_and_Live_Test_Plan_2026-11-22.md) to close the current defects and evaluate the first choice. This roadmap describes where to go beyond that release. The horizons below express dependency order. They are not calendar estimates.

## Horizon 1 — A workspace that follows through

| Opportunity | What the agent chooses | Small first experiment and success measure |
|---|---|---|
| Business intake across teams | Turn a vague request into one decision, find the person who can resolve an ambiguity and choose whether to ask now or proceed with an explicit assumption. | Give Business “we’re losing customers” with different definitions in different projects. Measure correct definitions, avoidable questions and wrong-project citations. |
| Task coordination | Split a question into independent requests, decide when a partial answer is useful and continue when the missing reply arrives. | Request Sales capacity and Marketing eligibility with one team offline. Measure time to first useful answer, eventual resolution and duplicate handoffs. Compare with sequential fixed routing. |
| Active memory and an advisor | Select the current authorized facts, identify conflicts and explain which decision needs revision. | Correct capacity from 50 to 30 after approval. The advisor identifies the affected shortlist, owner and source revision. Measure stale advice, repeated questions and unnecessary recomputation. |
| Better reporting for each audience | Choose which evidence and uncertainty the requester needs. Business gets the decision; ML Engineering can inspect the run and input revision. | Give two roles the same completed case. Score factual consistency, decision comprehension and restricted-data exposure. Keep the canonical report common to both views. |

Existing seams include [team routing](../dclab_core/team_work.py), [agent context](../dclab_rnd/agents/context.py), [project memory](../dclab_rnd/studio/memory.py) and [team reports](../dclab_rnd/teamspace/tools.py). Prioritize routing first, memory conflict resolution second and audience-specific explanations third. They can be evaluated separately.

## Horizon 2 — Agents that investigate and run useful analysis

| Area | Example decision and action | Boundary and measured payoff |
|---|---|---|
| Data intake and linking | Data Science sees order events and courier offers with different grains. It checks candidate keys, asks about repeated offers and requests the smallest missing table. | Code verifies keys, temporal joins and cardinality. Evaluate on invented one-to-many traps. Measure correct diagnosis and time until an analyzable dataset exists. [Problem–data check](../dclab_core/meet.py) |
| Features and leakage | For dispatch ETA, inspect restaurant prep history available at order time. Propose one past-only feature; reject a final fare or actual delivery timestamp. Decide which feature deserves an ablation. | Fit and assess on training folds. Report out-of-fold improvement, feature cost and planted leaks admitted. Compare with the same model given the existing feature set. [Feature language](../dclab_core/features.py) · [leakage advisor](../dclab_rnd/studio/leakage_advisor.py) |
| Experiment selection | ML Engineering chooses a baseline, examines validation errors and selects one follow-up experiment, or stops when the result cannot affect the business decision. | The studio controls splits, budget, comparison and sealed evaluation. Measure decision-quality improvement per run and per unit of spend against the fixed sequence. [Experiment operators](../dclab_rnd/studio/experiment_operators.py) · [lineage](../dclab_rnd/studio/lineage.py) |
| Pricing and commercial scenarios | Business asks about a 5% increase. The agent checks price variation, exposure, cohort support and whether Business wants a forecast or an intervention estimate. It can ask Marketing about historical discount policy. | Scenario code recomputes affected features on the same cohort. Report units, revenue, uncertainty and assumptions. Score on simulation truth and on cases where the correct answer is insufficient causal evidence. [Price scenarios](../dclab_rnd/studio/price_scenarios.py) |
| Notebook assistance | When a cell changes a result, the companion chooses whether to inspect a transformation, request the missing prediction moment or propose a reversible edit with a comparison. | Run the edit on a scratch copy and show its effect. Measure correct diagnoses, successful fixes and unnecessary changes on unseen notebooks. [Notebook companion](../dclab_rnd/notebook_assist.py) |

Start with a single feature investigation because a baseline, an allowed feature language and an ablation give a measurable result. Expand to experiment selection only after the agent regularly avoids redundant runs and respects the leakage boundary.

## Horizon 3 — Agents that learn from operations

| Area | Example decision and action | Dependency and acceptance check |
|---|---|---|
| Delivery operations | Data Science joins ETA and HyperAck evidence. A kitchen delay revises arrival; a new courier offer triggers a fresh acceptance task. A missing snapshot produces a partial answer and an owner. | Requires versioned scoring adapters and event-time snapshots. Replay unseen orders; score ETA error, probability calibration over many offers, correct partial answers and unnecessary rescores. Keep individual predicted and observed outcomes distinct. |
| Monitoring and recovery | An agent sees a shift in latency or error and chooses among checking source freshness, examining a cohort, investigating calibration or escalating to a person. | Monitoring code calculates drift; lineage identifies affected work. On planted incidents measure correct cause, time to recovery, needless retraining and unaffected tasks reopened. [Monitoring](../dclab_core/monitoring.py) · [pipeline replay](../dclab_rnd/studio/pipeline.py) |
| Decision follow-up | After a retention campaign, Business asks whether the chosen policy worked. The agent requests outcomes, finds comparison evidence and identifies what remains unknown. | Requires a recorded policy, eligibility, dates and outcome definition. Distinguish response among contacted accounts from incremental effect attributable to outreach. Measure evidenced conclusions and unsupported lift claims. |
| Organizational lessons | A team agent proposes a scoped lesson from a completed case and retrieves it when a relevant new problem arrives. The reviewer can correct or retire it. | Requires source decisions, review and permission checks at retrieval. In a second project measure useful reuse, incorrect transfer and whether an obsolete lesson stops influencing answers. [Lesson proposal and review](../dclab_rnd/lessons.py) |

## Improvements that make the whole product easier to use

- **Show the outstanding question where the user works.** In the original conversation, show what is known, the unanswered question, its owner and the action that resumes it. Test whether a new member can identify the blocker without opening the activity log.
- **Use one capability directory for routing and display.** Include availability, required inputs, expected report and supported recovery actions. A feature flag or missing adapter should change both what the UI offers and what the agent can select. Test the enabled, unavailable and degraded cases.
- **Make provenance inspectable in one click.** Link each material number to its dataset snapshot, run, model version or dated human input. Mark invented inputs and obsolete revisions. Measure how often a reviewer can reproduce a number from the report.
- **Show why the agent stopped.** Distinguish a deliberate stop, a missing fact, an exhausted budget and an execution error. Track how often a team can resume without re-entering its context.
- **Measure work resolved.** Track correct decisions, unsupported conclusions, handoffs per resolved issue, time waiting on a human, avoidable model calls and cost. Break results down by task family so one strong workflow cannot hide another team's failures.

## Investment and release decisions

| Priority | Scope | Relative effort | Evidence that justifies the next investment |
|---|---|---|---|
| First | Task unblocking, independent replies and memory revisions in one churn project | Small once the implementation foundations land | Better correct resolution than rules on paired cases, no stale approvals or privacy failures |
| Next | One real-data feature investigation and one model-backed delivery question | Medium; requires dataset and model contracts | Reproducible numeric evidence, useful abstention on unsupported inputs, acceptable latency and cost |
| Then | Commercial scenarios and monitoring triage | Medium to large; depends on dated price exposure, outcomes or telemetry | Correct distinction between prediction and intervention; reliable diagnosis without unnecessary retraining |
| Later | Cross-project lessons and intervention follow-up | Large; depends on longitudinal outcomes and review | Measured benefit on new eligible projects and demonstrable correction, revocation and access isolation |

A deterministic action remains useful wherever the answer is fixed by contract: enforcing permissions, recording a receipt, computing a metric, applying an approved change or validating a sealed split. Give the model room where several actions are plausible and where the outcome reveals which choice helped. Existing live evidence makes testing that distinction important. The bounded Lead experiment found lower planning scores on 11 of 16 people with the fixed loop enabled. [Experiment results](../evidence/campaigns/lead_loop_live_v1/RESULT.md)

For each proposed capability, nominate an owning team, freeze cases and a baseline, set a cost envelope, run in shadow, then measure isolated end-to-end outcomes. Release the smallest choice that earns its cost. If the model ties the rules, preserve the comparison and move investment to the next opportunity.
