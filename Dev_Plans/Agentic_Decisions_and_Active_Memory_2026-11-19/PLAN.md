# Give DCLab agents more room to decide

Planning review, 19 November 2026. This is a proposed sequence of experiments. The review compared the plan library with selected implementation paths and saved evaluations. No new evaluation was run.

## Main gaps

1. **Lead Agent performance varies on unfamiliar conversations.** All three recorded held-out runs in the [harness benchmark](../../evidence/campaigns/harness_ablation_v1/BENCHMARK.md) missed at least one requirement. One met the planning and understanding bars but recorded a forbidden event. Understanding, planning and actual actions need separate scores.
2. **The value of each additional decision delegated to the model is uncertain.** [Evidence-driven replanning](../../dclab_rnd/tasks/catalog.py) exists. The [end-to-end comparison](../../evidence/campaigns/e2e_v1/RESULT_gate.md) passed its specified margins on eight invented cases from two problem kinds. It did not isolate the benefit of letting the agent choose its next investigative action.
3. **Home can miss a fact the user already stated.** Some [prediction-moment extraction](../../dclab_rnd/draft/plan.py) depends on phrasing. A missed fact can lead to an unnecessary question or a poorly specified feature audit.
4. **Problem-specific continuity needs an explicit contract.** [Home](../../dclab_rnd/draft/memory.py) summarizes conversations and hands stated facts to a project. [Shared memory](../../dclab_core/memory.py) has person, project and session scopes but no distinct problem scope. Continuity across two problems in a shared project needs a dedicated test.
5. **The scientific comparison covers only part of the final outcome.** Its reference model scores each arm's chosen columns. The performance of each arm's own exported pipeline on future observations is a separate question.
6. **Data and exam readiness have unresolved conditions.** The [provider decision](../../dclab_rnd/store/owner_decisions.json) holds affected training data pending rights review. The [exam loader](../../dclab_rnd/store/exams.py) checks an exam against an external pin only when one exists. A release decision should distinguish a verified exam from one without a registered pin.
7. **Historical specifications can give the wrong starting point.** The [November 12 guide](../../docs/guides/TASK_PLANNING_NEXT_STEPS.md) describes opt-in rollout, while the [current defaults](../../dclab_rnd/orchestra/settings.py) enable task preview, execution and proposal admission. An individual workspace can override those defaults.

## Improvements

1. Extend the existing decision loop so the lead can ask a clarifying question, inspect metadata, run an allowed training-only experiment, consult an advisor, revise its plan or stop. Record the available options, selected action, short justification, evidence and observed result.
2. Measure which choices warrant autonomy. Use the same model, tools, budget and cases across comparison arms. Compare critical mistakes, task quality, avoidable questions, cost and time.
3. Let Home propose structured interpretations with a supporting passage from the user's words. Ask for clarification when competing interpretations would materially change the work.
4. Supply role-specific context inside the loop. Let authorized users maintain project and problem base facts. Before a decision, retrieve the current user's permitted facts, the active problem and relevant evidence. After a material event, record the decision, source and consequence, then refresh a versioned problem summary. A correction supersedes stale guidance. Advisor suggestions remain distinguishable from confirmed decisions.
5. Evaluate both the chosen columns and the actual exported pipeline on fresh future data, with intervals and report accuracy. Keep scientifically comparable measures separate.
6. Expose counts and reasons for training exclusions. Require a verified, registered exam for a release claim. Preserve held material under its existing policy.
7. Give each active plan a status, implementation owner, latest evidence and next open decision in a short plan index.

## Delivery order

| Stage | First deliverable | Acceptance evidence |
|---|---|---|
| 1. Agentic decisions first | Extend the existing Lead workflow for one delivery-time investigation. Give the agent the result of each action and a fresh choice of next step. Keep tool permission, spend, splits, approvals and holdout access enforced by code. | On development scenarios, show a justified question when prediction time is unknown, a data check when it is known, and a changed next action after new evidence. Confirm that refusing an action or stopping causes no unauthorized effect. |
| 2. Active memory and advice within that loop | Home, Lead and Advisor share authorized project and problem context. Save important events and a resumable summary of confirmed facts, unresolved questions, evidence and proposed next action. Show what changed and let an authorized person correct the base facts. | Resume in a fresh session without asking an answered question. Change the prediction moment and identify the affected feature choice. Keep a sibling problem's facts separate. A second user's private preferences must never surface. Verify that corrections update the summary and stale advice is no longer served. |
| 3. Compare and decide | Compare current behavior, expanded autonomy with existing memory, and expanded autonomy with active problem memory. Freeze the evaluation questions, thresholds and attribution rules before measurement. | Report paired results and uncertainty on independently authored cases. Attribute gains or regressions to the added decision or memory. Expand only where the measured benefit and safety justify it. |

Start with a reversible development case. A historical test budget does not authorize a new live campaign. Apply the current repository rules for focused checks, model-call capture and archive verification when implementing or evaluating this plan.
