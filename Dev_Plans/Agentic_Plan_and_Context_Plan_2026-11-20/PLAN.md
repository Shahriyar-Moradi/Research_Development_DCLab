# Agentic Plan and Context Plan

**20 November 2026 · Implementation handoff · Proposed**

Start with one real decision the Lead can make in the existing Studio. Then give that decision a reliable memory of the signed-in person, the project and the particular problem. The first delivery is a Home conversation about predicting delivery time. The user should be able to close the conversation, return through Lead, correct when the prediction is made, and receive advice that explains which earlier choice now needs review.

[Implementation sequence and checks](IMPLEMENTATION_PLAN.md) · [Coding prompt](PROMPT.md) · [November 19 repository-wide improvements](../Five_Improvements_and_Five_Gaps_2026-11-19/PLAN.md)

## What exists in this checkout

| Starting point | What the implementation can reuse |
|---|---|
| [Lead turn context](../../dclab_rnd/agents/context.py), [case agent](../../dclab_rnd/agents/case_agent.py), [Home case entry](../../dclab_rnd/draft/case_home.py) | The Lead already sees sourced facts, assumptions, unknowns and table structure. Its bounded recent conversation is useful working context. |
| [Task catalog](../../dclab_rnd/tasks/catalog.py), [team settings](../../dclab_rnd/orchestra/settings.py), [solution walk](../../dclab_rnd/orchestra/solution.py) | Evidence can trigger replanning. Task preview, execution and proposal admission default to enabled. The overall team switch is separate and workspace settings can override the defaults. Keep these distinctions during rollout. |
| [Home memory](../../dclab_rnd/draft/memory.py), [project memory](../../dclab_rnd/studio/memory.py), [shared memory contract](../../dclab_core/memory.py) | Home summarizes and hands facts to a project. Shared items already have sources, review status and corrections. Existing scopes cover person, workspace, project and session, among others. Problem-level identity and retrieval need an explicit contract. |
| [Recall tool](../../dclab_rnd/memory/tools.py), [memory routes](../../dclab_rnd/agentic/routers/memory.py), [Advisor](../../dclab_rnd/core_advisor.py) | Extend existing readers, review routes and advisory roles. The recall chain currently falls back to “owner” without a person. A signed-in worker needs its actual initiating principal. |
| [Problem identity design](../DCLab_Priority_and_Agents_Plans_English_2026-11-15/02_Model_Problem_Project_Agents_v2/plans/PLAN_3A_problem_identity_and_knowledge.md), [project identity design](../DCLab_Priority_and_Agents_Plans_English_2026-11-15/02_Model_Problem_Project_Agents_v2/plans/PLAN_4A_project_identity_relationships_and_organizational_memory.md) | Use these designs to resolve identity across a draft, a project and a reopened problem. Confirm implemented fields against storage before extending them. |

## Decisions the AI can make

At each turn the host supplies the goal, current scope, outstanding blockers, relevant evidence and a list of executable choices. The AI chooses its next action and gives one short reason tied to evidence. The host validates the choice against current permissions, data boundaries, budget and workflow state, performs the allowed action, then returns the observation. The next choice can change. One question or an explicit stop may be a successful outcome.

| Choice | First useful example | Host obligation |
|---|---|---|
| Ask a question | Prediction time is unknown and determines which features are available. | Ask the person once and persist the open question across resume. |
| Inspect | Time is known but the availability of a dispatch feature is uncertain. | Expose permitted schema and training-side diagnostics. Cite the observation. |
| Consult | Two feature policies are plausible. | Advisor returns evidence, uncertainty and a suggested next check. Lead owns the next proposal. |
| Compare | The usefulness of two allowed feature sets is unclear. | Admit an existing training/validation-only operation, when available and funded. Enforce fold-local preprocessing and the experiment cap. |
| Revise or stop | Evidence rules out a feature, the budget expires or an approval is needed. | Record the change, blocker or handoff. Revalidate the next action. Keep final approval, split policy, metric computation and release decisions under their existing controls. |

## What the agent remembers

| Information | Who can see it | Example and update rule |
|---|---|---|
| Personal preference | The same authenticated person within the permitted workspace | Preferred explanation detail. Opt-in sharing across people requires a separate explicit project edit. |
| Project base | Authorized members of that project | Common calendar or business glossary. An authorized editor can add, correct and retire an entry with a source. |
| Problem base | Authorized members working on that problem within its project | Target, prediction time, exclusions and acceptance criteria. Different problems in one project can have different values. |
| Material event or decision | Readers of the underlying source, within its scope | A confirmed requirement, rejected feature with reason, result that changed the plan, approval or unresolved blocker. Store a reference to the authoritative record and its revision. |
| Resumable summary | Readers authorized for all included facts | Confirmed facts, recent decisions with reasons, unresolved questions, evidence, pending approvals and proposed next step. Rebuild after a correction, handoff or material change. |

A memory writer proposes uncertain interpretations as drafts. A confirmed business requirement comes from an authorized person. A measured result comes from its run record. Advice remains a suggestion until accepted through the workflow. A user correction can supersede a requirement but cannot rewrite a measured result. Keep immutable historical decisions and mark their validity for the current problem separately.

**Visibility on handoff.** Home emphasizes missing business facts, Lead emphasizes executable actions and constraints, and Advisor receives the evidence and question it needs for the consultation. The three views share the same current facts and revisions. A shared problem summary contains only shareable facts. Apply personal preferences as a viewer-specific overlay after authorization. A draft can hold pre-project facts under its existing access rules. Bind it to a problem on build once, retaining the source references.

**Memory budget.** Preserve required constraints, pending approvals and corrections before optional history. Omit lower-value episodes with a visible count. If required constraints exceed the available context, pause the decision or load a smaller verified slice before a write. Treat recalled words as data. Rank only after limiting candidates to authorized scope. Recheck permissions and source revisions before acting on an earlier brief.

## Acceptance story

1. In Home, Dana says this delivery ETA is predicted after dispatch. The host records the stated constraint on the active problem and cites its origin. Lead can inspect whether a candidate column exists by that time.
2. Dana opens a fresh session tomorrow. Lead resumes the outstanding investigation. Advisor can explain how the recorded constraint motivated it. Neither needs to ask Dana for the prediction time again.
3. Dana corrects the prediction time to order creation. The previous statement remains in history. The new version is active, the old recommendation becomes stale, and Lead rechecks whether dispatch information would leak future knowledge. A pending action based on the old version cannot execute unchecked.
4. Dana opens a sibling problem in the same project. Its separate prediction time still applies. A colleague with project access can read shared project facts and cannot see Dana’s private notes. A person outside the project receives no problem facts, including in summary, retrieval or audit-facing product responses.

An invented example can establish the mechanics. Concluding that greater autonomy improves outcomes requires the paired evaluation and release gates below. [Implementation sequence](IMPLEMENTATION_PLAN.md)
