# Implementation plan for Agentic Plan and Context Plan

**Delivery order:** A1 and A2 add choice to the existing agent. A3 through A6 supply context and memory inside that choice. A7 measures both changes. Each package ends at a runnable boundary. A small coding model can complete one package per session using the [coding prompt](PROMPT.md).

## A1 · Give Lead a bounded choice in one real turn

1. Trace Home's case entry through `dclab_rnd/draft/case_home.py`, `dclab_rnd/agents/case_agent.py`, its harness and registry, and the dispatch of the selected tool. Determine which decision can already ask, inspect and pause. Capture that behavior with a scripted gateway and the current workspace settings.
2. At this decision, construct an allowed menu from registered tools, workflow state, approvals and remaining budget. Reuse existing typed calls and validations. If a semantic choice has no admitted implementation, omit it. First deliver ask, a read-only inspection and stop.
3. Let the model choose a menu entry. Bind the arguments to the authenticated case in host code, validate them at dispatch, and return a typed observation (succeeded, blocked, waiting for person, stopped or failed). Reuse existing step and decision records for the choice, available options, evidence references, brief reason and result. Emit at most one committed effect per decision.
4. Add a workspace switch for this new choice boundary, initially off. With it off, normal Home and Lead behavior uses the existing path. An invalid reply gets a bounded fallback to the safe existing action or a visible blocker. Keep existing task-planning switches intact.

**Done when:** a scripted Lead asks when the prediction moment is missing, inspects when it is confirmed, and stops without performing a tool when asked to stop. A forbidden or stale choice has no effect and has a traceable outcome. Exercise the normal Home entry, not just the menu function. Relevant neighbors: `tests/test_case_agent.py`, `tests/test_case_home.py`.

## A2 · Change the next action after an observation

Extend the same boundary to consultation, admissible training-side experiments and revision. Trace `replan_from_evidence` in `dclab_rnd/tasks/catalog.py`, the executable task adapter, and the graph guard. Extend the existing path once. Advisor can recommend a comparison, state uncertainty, cite evidence or return “insufficient evidence.” An experiment request names a registered operation and training split. The host owns the split, resource limits and call. A recommendation carries no approval.

Charge all children, including consultation, to the same run budget across resume. Persist the decision and observation before offering another choice. Enforce the existing revision/step caps and stop repeated identical choices with unchanged evidence. On retry use the existing operation ID or an equivalent durable idempotency key. After a crash resolve the recorded job outcome before considering replay. Report waiting approval, unavailable model, denied tool or exhausted budget as explicit outcomes.

**Done when:** an invented post-dispatch feature causes Lead to revise its plan at order creation. An alternative observation leads to a different next action. A repeated job delivery creates one experiment. A budget stop, cancellation and missing approval create no further work. Use scripted cases around `tests/test_task_execution_plan.py` and `tests/test_agent_harness.py`. Capture the current-behavior and new-choice arms before adding new memory.

## A3 · Resolve identity and authorized scope

Trace the principal from HTTP, Home draft, project, worker, resume and existing memory API to the read/write calls. Reuse durable problem identity if available. Otherwise introduce an opaque problem ID and an explicit workspace/project binding through the storage interface. Bind a draft on build idempotently. Create a second problem through a supported project operation and expose its selection. Resolve the active problem from validated server state. If old data maps to exactly one problem, create or resolve that binding lazily. With several candidates ask for selection before a problem-specific read or write.

Pass a host-resolved scope containing workspace, principal, project, problem, optional draft/session and current role. Verify parentage and current membership before retrieval and before writes. Preserve local single-user behavior only in explicit no-auth mode. A background job carries its initiating subject and resource IDs, revalidates on execution, and fails closed if access was revoked. Normalize the full parent chain into scope keys where current item identity only carries one scope ID. Version stored contracts compatibly. Preserve existing item IDs and correction links on legacy reads. Exercise file and PostgreSQL stores if touched.

**Done when:** two people, two workspaces, two projects, two sibling problems, and two sessions have the expected positive and negative reads. Include reused display names, guessed resource IDs, an unbound draft, a missing worker identity, revoked access and legacy reads. Relevant tests include `tests/test_core_memory.py` and `tests/test_memory_store.py`.

## A4 · Maintain project and problem bases and record material changes

Use existing memory write, review, correction and retirement operations in `dclab_rnd/memory/` and `dclab_rnd/agentic/routers/memory.py`. Expose authorized add, view, correct and retire operations on both bases through the ordinary Home/project UI. Show source, scope, active version and editor. Use the existing project/draft guard for writes. Keep personal preferences private by default.

| Input to the writer | Store or project | Retry or conflict behavior |
|---|---|---|
| Explicit user base edit | Sourced active fact with subject, field, author, scope, prior revision and effective revision | Compare expected revision. On conflict reload and show both edits to the editor. |
| Committed decision or consequential outcome | Short episode or decision projection referencing the original event, decision or run | Unique source type, source ID, source revision and destination scope. Replay converges on the same entry. |
| AI interpretation or recommendation | Draft assumption or proposal with references and review state | A retry cannot activate a draft. Only the normal confirmation or acceptance event promotes it. |
| Correction, retirement, deletion | New revision superseding an entry, or purge according to existing retention rules | Invalidate dependent summaries, cache entries and pending decisions. Reconcile failed invalidations from source revisions. |

Capture when an omission would cause a repeated question, invalid action or lost decision rationale. Skip greetings, raw transcripts and unchanged restatements. Use a transactional write with the source where feasible or a durable source cursor and idempotent replay. Never treat the presence of an episode as proof that its proposed action succeeded. Save approved brief reasons and source references, keep raw rows, credentials, provider exchanges and sealed evaluation material out of product memory. Follow current workspace consent/export and deletion controls for all derived views.

**Done when:** one event delivered twice yields one entry, two racing edits cannot overwrite each other silently, an Advisor suggestion stays provisional, and a user can correct a project base separately from a problem base. Start with `tests/test_memory_writing.py`, `tests/test_home_memory.py` and the affected API path.

## A5 · Build an active, versioned summary in the loop

Store a compact projection keyed by project/problem and visibility class. Suggested contents are source cursor, schema version, active fact references and revisions, last consequential decisions and outcomes, open questions, pending approval, next proposed action, and omitted-item count. Persist references plus the minimum readable text. Use the existing bounded memory brief for retrieval and the case context builder for presentation. A deterministic summary built from active records is sufficient for the first release.

Trigger refresh on confirmed base edit, material outcome, correction, handoff and resume with an outdated cursor. Coalesce repeated events. Rebuild from source if missing or stale. Publish with compare-and-swap on the cursor or equivalent concurrency protection. Reject late builds from an older source state. On read, verify scope, references and current revisions. If a fresh projection is unavailable, use authorized current facts directly or pause an affected action. Delete derived content when a source is purged, including search vectors and caches.

**Done when:** a fresh session resumes an open decision, correction arrives during summary creation, and the later reader sees the correction. A stale worker cannot reverse it. Deleting the source removes it from a rebuilt brief. Relevant checks include `tests/test_memory_reading.py`, `tests/test_memory_upkeep.py` and `tests/test_memory_retrieval.py`.

## A6 · Wire Home, Lead and Advisor to one version of the facts

Home shows the current problem and unresolved decision, promotes permitted draft facts on build, and lets the user edit base memory. Lead reads the current problem summary before each material decision. Advisor receives the current question, evidence and permitted source references. Return a recommendation with its dependencies, uncertainty, short reason and suggested check. Display which source changed when a previous recommendation becomes stale. On approval or tool dispatch, revalidate those dependencies. Include a resume link to the correct problem and a clear selection when ambiguous. Use existing authored frontend sources and API generation conventions.

**Done when:** run all four steps of the [acceptance story](PLAN.md#acceptance-story) through the product entry points on invented data, including a browser or route/UI integration check for editing a problem base. Also test a delayed Advisor response after a correction. Toggle the new behavior off and reopen an older project. Existing paths must still work. Tests touching core logic, UI and the Advisor each need only the affected test group.

## A7 · Measure the benefit before expanding autonomy

Freeze the scenario IDs, assertions, model configuration, cost/time caps, tool permissions and decision rules before scoring. Start with development cases covering missing facts, evidence that changes a choice, next-day resume, correction, sibling isolation, revoked access, budget exhaustion and no model. Build independent variations in wording and case order. Use synthetic inputs. Keep sealed material separate and respect the repository's one-use policy.

| Arm | Agent choice | Context |
|---|---|---|
| A | Current shipped decision path at the frozen revision and settings | Current brief and memory |
| B | New bounded choice and observation loop | Same current brief and memory as A |
| C | Same loop as B | Authorized problem base, active material memory and revision-checked summary |

Pair runs by case, model version, initial data, allowed tools, seed where supported and maximum budget. B minus A estimates the value of added choice. C minus B estimates the additional value of scoped memory. Publish per-case outcomes and paired intervals grouped by independent case, counts of correct next actions, repeated answered questions, stale advice acted on, task completions, context size, time and spend. Keep scripted correctness separate from a live model's judgment. Record refusals, timeouts and missing results in the denominator.

**Proposed release rule:** require zero unauthorized reads/writes, extra holdout reads, approval bypasses and actions taken on superseded facts in all designated scenarios. Require every scripted acceptance scenario to pass. For each incremental live arm, require the lower bound of a paired 95% interval on its primary improvement to exceed zero: correct investigative action for B versus A, avoided repeated answered questions on resume for C versus B. Require no loss of critical task completions on the frozen cases and spend within the declared cap. If evidence is inconclusive, keep the switch in pilot and report the uncertainty. Freeze the number of cases and repetitions before live measurement. Roll back only the affected new switch if a safety condition fails. Keep the recorded history readable.

Use the repository gateway and capture/archive rules for any separately authorized live evaluation. A scripted delivery can be complete with live promotion pending. Record the exact limits of the evidence in the handoff.
