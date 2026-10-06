# The agentic foundation: design, phases, and prompts

This is the design for DCLab's AI and agent layer, written as phases with small plans and a prompt for each. It
extends Part 2 of `PRODUCT_BUILD_PLAYBOOK.md` (packages 11.1 to 11.4) and uses its shared preamble.
Read the playbook's "Shared preamble" first and paste it before any prompt below.

**خلاصه فارسی.** این سند طراحی لایه‌ی هوش مصنوعی و ایجنت DCLab است: اصل‌ها، معماری، شش فاز (A1 تا A6) و برای هر
بسته یک پرامپت آماده. هسته‌ی ایده: **کد تصمیم می‌گیرد، مدل پیشنهاد می‌دهد.** هیچ مدلی تقسیم داده یا معیار را
محاسبه نمی‌کند؛ هر حرکت ایجنت از اعتبارسنج گراف می‌گذرد؛ همه‌ی درخواست‌های مدل از یک دروازه عبور می‌کنند؛ و هر ایجنت
با مجموعه‌ی آزمونی سنجیده می‌شود که تله‌های واقعی (نشت) دارد.

## Status

| Package | Status | Notes |
|---|---|---|
| A1.1 | Done | Every model request goes through `dclab_rnd/models` (the intern too). NOOA's research clients cannot be swapped and report each request to the usage log. Tests cannot reach a live model (`DCLAB_NO_LIVE_MODELS=1` in the make targets). |
| A1.2 | Done | Costs from a dated price table (`models/prices.py` or `DCLAB_PRICES_FILE`; none is guessed, local is free). A request whose upper-bound cost could pass a run, project or workspace cap is refused before it is sent, under a lock. Open: no price is configured yet, so remote requests are counted in tokens until prices are added. |
| A1.3 | Done | Each purpose names its tier and the most cell values it may contain; `tests/test_model_data_limits.py` runs every caller on a table of findable values (text, numbers, dates; all five stages) and fails on a leak (shown to fail on a deliberate one). Every caller that checks a model's output reports a verdict; a model failing a purpose twice in a month is listed on Admin with its tier. Run on a local model (qwen2.5-coder 1.5B in Ollama) for the cheap tier: both cheap purposes ran on it, logged as local at €0; its answers failed both checks and the code fell back. |
| A2.1 | Done | `dclab_rnd/agents/`: `Tool` (schema, handler, read or write effect, the workflow move a project write makes, legacy aliases), `Registry` (scoped: "project" for the intern and `/mcp`, "draft" for the Home agent), `Policy` and `run()`. The 20 intern tools and the 8 Home tools are registered once in `default_registry()`; the MCP server reads the list at each request. In `run()` a call is checked against the tool's JSON schema (`jsonschema`, now in base and CI requirements; a null counts as not given) and a bad one comes back as an error the model can read; the budget is checked before every call and nothing runs after a terminal tool. The intern and MCP still call through `Toolbox.call`, which checks only unknown and missing arguments so the handlers' clamping and coercion stay as they were; the agents move onto `run()` in A2.4. |
| A2.2 | Done | A write tool cannot be registered without the move it makes, and a write in a scope without a guard never runs. Project writes pass `studio.graph.check` as the agent before the handler; a refusal is logged as a transition and returned with the reason, the failed checks and the rules. The graph gained checks for the four project writes it did not cover (`create_project`, `attach_data`, `set_settings`, `propose_solution`): for example the agent may not replace the data under a solution the owner signed. Home writes have their own checks (pack list and the user's choice, `workflow.validate`, one preparation at a time); a refused workflow now keeps the current one instead of falling back to the template. `tests/test_agents_validator.py` refuses every project write tool and fails when a new one has no refusal case; `create_project` is refused before a project exists, so its refusal is returned but has no log to go to. Its allowed move is logged with the name only (the goal is the user's words). |
| A2.3 | Done | `dclab_rnd/agents/traces.py`: a row per step (run, step, model reply, graph state seen, tool, arguments, verdict, result summary, tokens, seconds) in `agent_step` (migration 0005) or `agent_steps.jsonl`; `run()` writes them, and so does the intern's own loop until A2.4. The state is taken before the move. A result is summarised (360 characters) after the keys that carry table cells (column examples, previews) are removed, and a test runs the data tools and finds none of the table's values in the trace; arguments over 8,000 characters are cut and such a run is refused by replay rather than misreported. `replay()` sends the stored replies through `run()` again with the tools answering from the trace (no model, no project change) and reports whether the moves are the same; it continues across a run's stops (a question, then a follow-up). The Intern page shows each step's trace line (verdict, graph state, model reply, tokens), matched by the session's own step number. Open: the tokens of a reply that makes no step (a final answer) are in the session's usage but not in the trace. Intern traces become replayable when the intern runs on `run()` (A2.4); the standard plan has no model replies to replay. |
| A2.4 | Done | The Home agent and the intern run on `agents.run()`; their own loops are gone. The runtime gained what they need: `stop_when` (a simulation ends the Home turn), `max_replies` (Home: 5 model requests per message), per-scope contexts, a result size per policy, a terminal tool is never refused for the step budget (even after earlier calls in the same reply used it up), and only a call that ran ends the run or the turn (a refused `finish` or question goes back to the model). Model calls of both agents are now checked against the full schemas, so a value the handlers used to clamp (the intern's `set_settings max_rows=500000`) is now an error the model reads. Home keeps the scripted fallback, the turn lock and the streamed bubble (an adapter around its streaming call); its model calls are now checked against the schemas (the schemas no longer require what the handlers default: `ask_user` needs only the question, `set_pack` only the key, `request_data` nothing), and its steps are traced under the draft's id (state: which of the five fields are known), returned by `GET /api/drafts/{id}` and replayable. The intern's `write_plan` and `finish` are registered tools (scope "session"); `finish` is now a step of the session and of its trace, and an intern trace replays; the report is stored in the trace as the model wrote it, as the session already stores it. Trace files share one lock per file in a process (the server's and a campaign's). The campaign agent (LangGraph and NOOA, typed answers, no tool calls) stays a graph: its model requests already go through the gateway adapter (A1), its one write `run_experiment` is a registered tool (scope "campaign", `agentic/campaign_tools.py`) behind the campaign guard (scope, citations, paired reference, duplicates: the same refusals as before), and each phase and experiment is a trace row. `make check-all` now runs the campaign tests in `.venv-agent`; there the server test is skipped (no pandas). |
| A3.1 | Done | `draft/plan.py`: the draft keeps a plan for the outcome, the prediction moment, the action and cost, and the data: a status (stated, inferred, unknown), where it came from (the sentence, an answer, the model, the data) and the user's words. The questions come from it: only an unknown field is asked, outcome, then moment, then action and cost, then data; an outcome known only in words is asked again for its column once a table arrives, and a question past the limit ends in the summary. At the start a model, when configured, proposes a reading of the sentence (one request through the gateway); a quote that is not whole words of the user's text, or a value the quote does not support, is dropped. Without a model, keyword rules read the outcome (only with who or what and a window: "which subscribers cancel within 30 days") and the moment (only when set apart by a comma or a verb such as "scored", so "cancel before renewal" stays part of the outcome). The draft guard refuses a model's `record` of a planned field that is neither an answer to a question the user has since answered nor backed by such a quote, and an `ask_user` the plan does not ask next. Editing the sentence reads it again and keeps the answers. The "DCLab understood" sheet shows each line's source and the user's words. Example: "Predict which subscribers cancel within 30 days, scored on the first day of each month" takes two questions instead of four. Open: the costs line is still part of the action question; inferred fields are shown for checking but not asked; the planner request is not a trace step. |
| A3.2 | Done | `studio/leakage_review.py`: the reviewer reads the prediction moment, the column names with their kind, missing rate and unique count (never a value) and the column audit's flags, and returns items (column, "after the moment" or "available", a reason, evidence records). Code drops an item with no reason, a record not in the index, a column not in the table, or (from a model) a reason that does not quote the moment; a column the audit flagged stays flagged whatever a model says (its "available" is shown as a disagreement). Only the audit's flags are ticked; the rest is shown unticked and never applied. Without a model the audit answers, plus one rule: a clause of the moment that excludes something ("Exclude … bounce/exit rates", "… are post-outcome and forbidden", "… are not available") names the columns whose words it contains. A model reads it through the gateway purpose `leakage_review` (its data-limit test finds no cell value). The wizard's solution step shows the suggestions; the intern has `review_leakage` (21 project tools, also on `/mcp`). Measured on the studied samples without a model: it finds 14 of the R&D's 17 forbidden columns (12 by the audit alone), including `duration` on bank_marketing; the 3 it misses are online_shoppers' "completed-session counts" (Administrative, Informational, ProductRelated), which name no column. Of the 23 columns proposed beyond the R&D's list, 22 are the audit's own flags (mostly strong single-column separators such as breast_cancer's measurements, ticked as before) and 1 comes from the moment rule (hyperack's `first_customer_fare`, shown only). Open: the model path is tested with a scripted model only; the project's solution page does not show the review yet. |
| A3.3 | Done | `dclab_rnd/cited.py` is the one check for model-written prose from records, shared with the Evidence library: a sentence is kept only when it cites a source that was given, every number in it is in a source it cites, and it does not claim production readiness, a guarantee, a proof or a cause. Numbers are read as written (0.30 is 0.3, `.99` is 0.99), keep their unit (61, 61% and 61k differ) and their sign when the sentence writes one; a count word is read as its digits with the unit that follows (ten folds is 10; ten percent is not 10; twenty-one is 21; one only where it is a quantity), other number words (half, twice, hundreds, million) must be words of a cited source, and a name with digits (C1, R22) is hidden only when a cited source writes it; ids such as WF-03 are not numbers; a rounded, converted or computed number removes the sentence. Overclaims are matched after Unicode and whitespace are normalised ("production‑ready", "production  ready", "deployment-ready", "fit for production", "ready to ship", "can go to production"), and a denial counts only right before the phrase (four words at most, none of them doubt, deny, wrong, fail, hard…) ("is not yet production-ready"; not "not only", and not a "no" elsewhere in the sentence). A sentence ends at a stop, a stop glued to a capital or a line break, but not at the dot of a code name (pd.DataFrame, sklearn.pipeline.Pipeline), so a bullet is a sentence and an uncited one cannot ride on the next one's citation. The Evidence library's check is now per cited record (it pooled the numbers of all retrieved records before). `studio/explain.py`: when a model is configured, each stage run asks for at most 120 words about the stage record, citing its claim ids (`PRJ-…-final-C1`), its deterministic notes (`PRJ-…-final-N2`) and the rule records the notes cite (rules first, six at most); the kept sentences are stored as `record.explanation` beside the unchanged notes, with the label "written by a model from the stage record" and the number of sentences removed; nothing is stored when nothing passes or there is no model. Approving a stage with a choice that overrides the rule's removes the explanation (it described the rule's pick). It goes through the existing `stage_notes` purpose (its description now names the notes and rule records; the data-limit test covers it), inside the stage move. The Notebook stage cards, the Reliability page (production gate) and the brief (Risks pane) show it, labelled, with its citations. The model-written critique note and the project answer lose any sentence that says production-ready, guaranteed, proven or caused (before, only the prompt asked). Open: a number is checked against the sources the sentence cites, not against what it describes (a sentence citing two sources can put a number from one beside the subject of the other); the explanation is one more request per stage (the gateway counts and caps it), made even when the critique request just failed; its cites to claims and notes are tags, not links; the critique and project answer are filtered for overclaims but not citation-checked; the claims and notes still describe the rule's pick after an override. |
| A3.4 | Done | `dclab_rnd/copilot/fixes.py`: beside each finding of the notebook review a model can write a fix in at most two sentences, from the finding and the records it cites. The findings are exactly what the detectors give (nothing in a finding, its severity, its message or its proof depends on a model; the fix is a separate `fix` key added to a copy). One request (eight findings at most; a longer review takes more) asks for the findings' fixes as JSON; for each, code keeps a sentence only if it passes the shared check (`cited.py`: it cites a record of that finding's own proof, every number is in a cited record, no production-ready claim) and drops the whole fix unless it cites one of the finding's rules and, when the finding has a pitfall, precedent or finding record, that record too. The model sees each finding's title, message and suggestion and the text of its records, through the new gateway purpose `notebook_review`. A finding quotes the code in backticks (a column, a path, an example line): only a plain name (`duration`, `imblearn.pipeline.Pipeline`) is kept and anything else becomes `<code>` before the request, so a path or a selector string from the notebook never leaves; the two detectors that quote the code itself are redacted whole (`absolute_data_path` is replaced by a fixed text, every name in `suspicious_column_name` by `<column>`); the notebook's code and data never do (the data-limit test plants an absolute path, a selector string and a value in the code and finds none in the prompt). Where: `python -m dclab_rnd.copilot review NB --fixes` (Markdown, HTML and the annotated notebook show the fix, labelled; without a model it says so and prints the usual review); `POST /api/projects/{id}/review/fixes` on the product (a person asks for it, since it is a model request; one at a time per project; `GET .../review` now says `fixes_available`; fixes come back keyed by finding, `detector:cell:line`) and a button on the Notebook page's companion, which shows each fix under the unchanged finding and ignores a reply for a review that is no longer the one on screen. On the leaky bank-marketing notebook, with a scripted model, all 10 findings are identical to before and each gets a fix that cites its rule and its pitfall or precedent record (DCLAB-R02 and PIT-003 for the oversampling finding). Open: only the project's own exported notebook is reviewed in the product (no upload of a user's notebook, so the demo mode has no fixes); `notebook_assist` (the VS Code companion's engine) is untouched; the model path is tested with a scripted model only; a fix's quality beyond the checks (is the advice right) is not measured, and a fix written for one finding would pass on another that shares its rule and has no pitfall record (citations are what is checked). |
| A4.1 | Done | `dclab_rnd/agent_eval/`: 20 seeded cases (`cases.py`), each a 1,500-row table with one planted trap or a clean control and its own prediction moment (given to every policy): post-outcome columns (named, and one with a neutral name), noisy copies of the target, an identifier assigned in target order, a unit conversion and a sum identity of a regression target, a score missing exactly when the outcome happened, an account whose rows share the label, a drifting time column split at random, repeated patients, 300 repeated rows, and controls (constant columns, informative missingness known at signup, a benign `post_` name, a strong honest driver, two clean tables). Names avoid the R&D's precedent names, so a case measures the policy and not a lucky name. Each policy works on a fresh project through the same tools and validator as the intern; code scores what it left behind: leak kept out (forbidden, an identifier, or a text column on a task where text is not modelled: on a binary task text is a feature), the time or group column declared; a clean column (the case's own and the honest base inputs, in every case) kept out or made the time or group column is a false alarm; the repeated-rows note is an engine check, the same for every policy, kept outside the policy scores; unsafe moves refused; the share of valid moves; every cited record exists. Three scripted policies: the intern's standard plan, the audit's own defaults (what the wizard pre-fills) as a reference, and a deliberately bad policy that ignores the audit and tries two moves out of order. Measured (AEV-001): standard plan 1 of 13 leaks caught (95% 1–33%), 1 false alarm in 20 cases; audit defaults 10 of 13 (50–92%), 3 false alarms in 20 (5–36%); bad policy 0 of 13, 0 false alarms, 20 of 20 unsafe moves refused. The suite runs offline whoever starts it (it sets DCLAB_NO_LIVE_MODELS for its run), and the test checks every case's fingerprint against the stored run. So on a table that is not one of the studied samples the standard plan barely uses the audit (it forbids only columns named like a known precedent), and the audit alone over-flags (a benign post_ name, a strong honest driver together with an honest base input, and rounded income taken for an identifier). The suite runs in the test gate (about a minute) with the scorecard pinned in `tests/test_agent_eval.py`; `make agent-eval` stores a new AEV result; the Benchmark page's leaderboard shows the three scored rows and the planned suites stay planned. Open: what the standard plan should forbid on unseen tables is an owner decision (between today's 1/13 and the audit's 10/13 with 3 false alarms in 20); the identifier heuristic flags rounded continuous values; misses no policy catches: a neutral-named post-outcome measurement, a score missing only for the outcome, repeated patients (no group column is proposed for them); the repeated-rows note counts repeats within the training rows only and does not remove copies from the holdout; the PostgreSQL pass runs the suite on file stores. |
| A4.2 | Done | `dclab_rnd/agent_eval/live.py`, `make agent-eval-live`: the intern, driven by the configured model through the workspace gateway (its usage log and monthly cap), on the A4.1 cases (all, or `--cases`), each run `--repeats` times (5 by default, at most 10). It first prints the model, the runs, upper bounds on the intern's requests (runs × (tool calls + 2)), on input tokens (each request carries the policy, the tool schemas and up to all earlier tool results) and, when priced, on euros, plus the caps; nothing is sent without `--yes`, without a model serving the intern, for a priced remote intern model without `--cap-eur`, or with a euro cap on an unpriced model (it could not stop anything). Only the intern talks to the model during the suite: the tools that would ask one of their own (the leakage review, a project answer, the stages' notes and explanations) take their deterministic path, so the session's caps and accounting cover every request. `--max-requests` (default: the printed bound) counts them at the transport, so only what the gateway accepted. `--cap-eur` gives each run what is left of the suite's euros as its session cap. A run cut short by a cap or an error is recorded as not run, never scored, and the suite stops there. Each score is the mean over complete repeats of the suite-level rate with a 95% t-interval (a partial repeat stays out of the interval; per-case counts include every scored run), plus requests, tokens and euros. Results go to `evidence/campaigns/agent_eval_v1/results/AEV-NNN_judgment_v1_live_<model>.json` with a run id and the model's name, never overwritten; the Benchmark page reads only the scripted results. First live run (AEV-002, local qwen2.5-coder:1.5b through Ollama, 3 cases × 2 repeats, free): the model answered each case in one reply without calling a tool, so it kept 0 of 2 leaks out per repeat and made no move; 6 requests, about 16k input tokens. Open: no larger local or remote model has run it; the token bound is generous; a live run does not set the Benchmark leaderboard yet. |
| A4.3 | Done | `dclab_rnd/prompts/`: the ten prompts DCLab sends to a model are versioned files (`intern`, `home_agent`, `home_plan`, `synthetic_schema`, `parse_pattern`, `stage_critique`, `stage_explain`, `leakage_review`, `notebook_fixes`, `evidence_answer`), read with `prompts.text(name)`; a number a prompt states (120 words, two sentences, the evidence page's "not covered" reply) is a placeholder filled from the module's constant, and a placeholder left unfilled is an error. The move changed no prompt: each module's text was compared byte for byte before and after. `prompts.fingerprints()` hashes every prompt file, each tool scope's schemas (project, draft, session, campaign; the campaign's pydantic-made schema by its name, description and fields, so a library upgrade does not move it) and the workflow validator's rules (`studio/graph.py`). Only a complete scripted run (every policy on every case, no errors) stores them (AEV-003), and `tests/test_prompt_gate.py` fails the gate when the latest scripted run is not complete or when a prompt, a tool schema or the validator changed since it, naming what changed and asking for `make agent-eval`. A live run stores the hashes too and, after its result is safely written, compares itself with the last completed live result of the same model on the same cases (same fingerprints), printing the change per score and per case and saying when the prompts, schemas or rules changed in between; a smoke run on other cases is never a baseline. A test keeps code from writing the prompt files: people edit them. Open: the campaign's NOOA prompts, the research critics (`llm_review`, `guide_review`) and the SFT corpus prompt stay in their modules; the gate asks for a rerun, not for the scores to hold (a drop is for the person to judge); the scripted suite sends no prompt to a model, so for a prompt change the rerun records it rather than tests it (the live suite tests it). |
| A5.1 | Partial | `dclab_rnd/retrieval/`: 60 questions written by hand (`questions_v1.json`), each with every record id that answers it, in six groups: rules (15), paraphrases (10; a test checks that none shares a word, after the index's tokenizer, with the title or text of a record that answers it), measured pitfalls (8), leakage precedents with their dataset cards (6), experiments (13; trivially found by every method, since titles carry dataset and stage), workflow blocks and findings (8). Three retrievers: keyword search (BM25), a vector search over the index's own words (latent semantic analysis: TF-IDF reduced by SVD, local, deterministic, no model) and a hybrid by reciprocal rank fusion. Recall at 5 (RET-001, `make retrieval-eval`): keyword 0.700 overall (rules 0.800, paraphrases 0.000, pitfalls 0.688, precedents 0.834, experiments 1.000, workflow 0.812); vectors 0.706 (rules 0.733, pitfalls 0.812, precedents 0.889); hybrid 0.689 (rules 0.733, pitfalls 0.688, precedents 0.889). The hybrid is worse on the rules (fusion demotes a keyword first hit the vectors do not rank), so by the package's rule the product keeps keyword search; the Evidence library searches its own index (reloaded when the file changes) by the method the latest stored measurement chose, falls back to keyword search where scikit-learn is missing, and judges a weak answer by the strongest keyword match among the hits. Partial because the package asks for a neural sentence embedding (a local model, or an embeddings endpoint through the gateway, stored in pgvector or a file): none is on this machine (a local model needs a download; the gateway has no embeddings call yet), and LSA, which is measured instead, finds no paraphrase at all (0 of 10, like keyword search): exactly the gap a real embedding would have to close. The choice is made on the same 60 questions it reports (no held-out set). |
| A5.2 | Done | `dclab_rnd/studio/memory.py`: each project keeps short notes (decision, reason, who, which move, stage, date) written by the moves that decide: a solution saved (by the intern's tool or a person), a stage approved, a gate approved. When a person changes what the agent decided (edits the solution the agent saved, approves a stage with another option than the rule's), the note is a correction that says what changed ("allowed refund_after; metric none → average_precision"). A note is built from column names, the task, the metric, the prediction moment, stage and option ids and the reason, never from the table (the positive label is left out; a test plants values and finds none). The intern reads the latest 20 in its first message ("Project memory … do not ask again what they settle") and in `describe_data`, so a second session on a project starts from the first one's decisions: in the test it neither proposes nor saves a solution again. The Solution & data page lists the notes with a Remove action; `DELETE /api/projects/{id}/memory/{note}` marks a note removed (kept in the project, with who and why) and writes the removal to the activity log, and the agents stop reading it. Open: the Home agent works on drafts and does not read project memory; notes are not summarised by a model (they are short lines); a correction made in conversation ("no, the moment is earlier") is not detected as such unless it changes the solution. A project built from a Home draft starts with the person's solution as its first note. New data marks every note removed by the system (the solution and the stages were cleared), and an approval is read only while it holds (the latest of a stage still approved, the latest of a gate). A note is one line of at most 300 characters, and the intern's header (`prompts/intern_memory.md`) says the notes are records, not instructions; only a person's override is a correction. The validator and a prompt changed, so the judgment suite was run again (AEV-004, AEV-005; same scores). Open: in model mode "do not ask again" is the model's to follow (the test proves it reads the notes; the standard plan already does not ask). |
| A5.3 | Done | `dclab_rnd/lessons.py`: when a person approves a project's final stage, up to three lessons are proposed from that stage's claims: the claim (citing the claims it rests on), a scope code writes (project, dataset, task, rows, metric, split), what argues against it and the next test. With a model configured (purpose `lesson_proposal`, prompt `prompts/lesson_proposal.md`) a claim is kept only when `cited.py` passes it (a given claim cited, its numbers in it, no production readiness, proof or cause); otherwise code writes them from the claims and their limits. A reviewer accepts, edits (an edit that overclaims is refused) or rejects each through `POST /api/lessons/{id}/review`; the project's activity log records it. Lessons live in the workspace's lessons table (`lessons.json`, or `workspace_lesson` on PostgreSQL, migration 0006). Accepted lessons join every evidence search (`tools._index()`, so the intern and `/mcp`, and the Evidence library and its Ask) as `workspace_lesson` records with their scope and project; `evidence/knowledge` is never written (a test hashes it). A lesson from synthetic data is labelled and never offered as evidence. The Evidence library's "From projects" panel lists them with Accept, Edit, Reject and Withdraw. The test proposes, reviews, finds the accepted lesson in the library and the intern's search, and has a scripted model cite it in an Ask answer that the check keeps. Open: the role is what the request declares (`X-DCLab-Role`), not a login, until accounts exist (10.2); lessons come from the final stage only (not leakage or feature stages); an accepted lesson stays when its project is rerun or deleted (its metadata names the final run it came from; when the stage runs again the earlier run's unreviewed lessons are marked superseded); one workspace per process (the server installs its lessons as it installs the gateway); no model judges whether a lesson is worth keeping, a person does. A lesson's text holds only the numbers of its claim, counter-argument, next test and scope, so an answer citing it cannot borrow a date. |

## 1. What exists today

Three agents, built separately, plus the tools they share.

| Piece | File | What it does | Gap |
|---|---|---|---|
| Home agent | `draft/chat.py` (492 lines) | Up to 4 questions, outcome and workflow, simulation, streamed replies; a scripted fallback | Own loop, own tool list; no trace of its steps |
| Intern | `intern/loop.py` (247), `intern/tools.py` (294, 20 tools) | Builds a project through the validator, with a step and minute budget; standard plan with no model | A third loop; budget in steps and minutes only |
| Campaign agent | `agentic/agents.py`, `worker.py` (NOOA, LangGraph) | Research campaigns that fill `evidence/campaigns/` | Separate runtime and model client; web UI trigger lives in `/classic` |
| Notebook copilot | `copilot/`, `notebook_assist.py` | Reviews a notebook against the rules | Deterministic; no model |
| Model client | `intern/llm.py` | OpenAI-compatible, streaming, safe failure reasons, pause after a hard refusal | Created in several places; no budget, no usage log, no tiers |
| Validator | `studio/graph.py` (364) | WF-01…10, every move checked and logged | The one thing that must stay in front of every state change |
| Evidence | `evidence_index.py`, `tools.py`, `critic_gate.py` | Search over 137 records; critique checked against numbers | Retrieval is keyword (BM25) only |
| Evaluation | `notebook_eval.py`, `system_eval.py`, `trace_eval.py`, `replay_compare.py`, `agentic/live_eval.py` | Pilot scorecards, trace grading | Not one suite; not run per agent change |
| SFT | `studio/sft.py`, `research/llm-fine-tuning` | 330-example corpus; project-derived examples | No trajectory data from real runs yet |

## 2. Principles (they do not change between phases)

1. **Code decides, a model advises.** A model proposes JSON (a parse pattern, a workflow, a synthetic schema, a plan step). Deterministic code validates and applies it. A model never computes a split, a metric or a selection.
2. **Every state change goes through the workflow validator.** No agent writes a record, a solution, an approval or a split except through `studio.graph.check`. A refused move is logged and returned to the agent as a reason it can read.
3. **Aggregates, not rows.** An agent may read column summaries and descriptive findings. Cell values stay on the machine. The single exception (sample lines for an unreadable file) is disclosed and can be switched off.
4. **No model, no problem.** Every agent has a deterministic path that finishes the job. A model improves the conversation; it is never required for correctness.
5. **Every request is counted.** One gateway, one usage log, one budget. No module creates its own client.
6. **Every step is replayable.** What the agent saw, which tool it called with which arguments, what the validator said and what came back, in order, per run.
7. **Evidence cited or absent.** A claim cites a record that exists, with numbers that appear in it, or it is dropped.
8. **Unsafe is measured.** Planted traps (a post-outcome column, an identifier that predicts the target, a random split on timestamped rows) are the test of every agent, scripted in CI and live on request.

## 3. The architecture

```text
   user / MCP client / API
          │
   ┌──────▼───────┐     ┌────────────────┐     ┌───────────────┐
   │ Agent runtime │────▶│  Tool registry │────▶│ Workflow       │
   │ (one loop)    │     │  (schemas,     │     │ validator      │──▶ stores (PostgreSQL)
   │ budget, trace │     │   effects)     │     │ studio.graph   │
   └──────┬───────┘     └───────┬────────┘     └───────────────┘
          │                     │ read-only tools: evidence, profile, records
   ┌──────▼───────┐     ┌───────▼────────┐
   │ Model gateway │     │ Evidence index │
   │ tiers, budget,│     │ (+ later:      │
   │ usage log,    │     │  embeddings)   │
   │ pause, retry  │     └────────────────┘
   └──────┬───────┘
          │ OpenAI-compatible endpoints: strong · standard · cheap (a local model can serve cheap)
```

- **Agent runtime** (`dclab_rnd/agents/`): one loop. A *Tool* has a name, a JSON schema, a handler and a flag saying whether it changes state. A *Policy* is a system prompt plus a tool list plus a budget. The Home agent, the intern and the campaign agent become three policies on this loop.
- **Model gateway** (`dclab_rnd/models/gateway.py`): `complete()` and `stream()` take a *purpose* and a *tier*. It owns retries, timeouts, the safe failure reasons and the pause, counts tokens, prices them from one table, checks the project and workspace budget, and writes a usage row.
- **Trace store**: each runtime step is a row (state summary, tool, arguments, verdict, result summary, tokens, time). The trace feeds the Compute page, the audit trail and the training data.

## 4. Phases

| Phase | Goal | Packages | Depends on |
|---|---|---|---|
| A1 | One gateway | A1.1 to A1.3 | Part 2: 9.2 |
| A2 | One runtime | A2.1 to A2.4 | A1 |
| A3 | Make the agents better at the real job | A3.1 to A3.4 | A2 |
| A4 | Prove it: evaluation | A4.1 to A4.3 | A2 |
| A5 | Retrieval and memory | A5.1 to A5.3 | A2 |
| A6 | Learn from runs: the policy model | A6.1 to A6.4 | A3, A4 |

Order: A1, A2, then A3 and A4 together (every A3 change must pass A4), then A5, then A6.

---

## Phase A1: One gateway

### A1.1 The gateway and its usage log

- **Delivers:** `dclab_rnd/models/gateway.py`, a `model_request` table (migration), tiers in settings, and every current caller moved onto it.
- **Done when:** a search finds no `openai.OpenAI(` or `ChatClient(` outside the gateway; the usage table fills during `make product-e2e` with a model configured; `make rd-check` passes with scripted clients.

```text
Package A1.1. Build the model gateway. Today ChatClient (dclab_rnd/intern/llm.py) is created in the Home agent,
the intern, the evidence answers and the synthetic-data path, and studio/agent.py and the campaign worker call
OpenAI directly. Write dclab_rnd/models/gateway.py with complete(messages, tools, purpose, tier, project_id) and
stream(...). Settings map each tier (strong, standard, cheap) to an OpenAI-compatible endpoint and model; an
unset tier falls back to the standard one. Keep what llm.py already does (streaming, the safe failure reason,
the pause after a hard refusal) and add: per-purpose timeouts, retry with backoff on 429 and 5xx, and a
model_request row for every call (purpose, tier, model, input and output tokens, latency, outcome, project;
the prompt text only when DCLAB_LOG_PROMPTS=1). Add the table through an Alembic migration. Move every caller
onto the gateway. Tests use scripted clients; none may touch a live endpoint.
```

### A1.2 Budgets and prices

- **Delivers:** one price table, a per-project and per-workspace monthly cap, a refusal with a clear message when a request would pass it, and real spend on the Compute and Admin pages.
- **Done when:** a scripted run over its cap is refused before the request is sent; the pages show euros from the usage table, and "not metered" is gone only where it is true.

```text
Package A1.2. Add money to the gateway. Keep prices per model in one file (dclab_rnd/models/prices.py, euros per
million tokens, with a date) and compute a request's cost from its token counts. Add caps: per project per run,
per project per month, per workspace per month; the wizard's euro budget becomes a real cap. Check before
sending; refuse with "this project has used its budget" and say how much is left. Show spend on the Compute and
Admin pages from the usage table and remove "not metered" only where it is now true (local models cost zero:
say so). Never guess a price: an unknown model is "price not configured" and is counted in tokens only.
```

### A1.3 Tiers by purpose

- **Delivers:** a table of which purpose uses which tier, a local small model for cheap purposes, and a check that a purpose never sees more than its allowed data.
- **Done when:** with a local model on the cheap tier, parse patterns and evidence answers run on it and the usage log says so.

```text
Package A1.3. Assign tiers by purpose in one table: parse_pattern and evidence_answer on cheap, home_agent and
synthetic_schema on standard, intern and campaign on strong. Make each tier configurable from the environment
(for example DCLAB_TIER_CHEAP_BASE_URL=http://127.0.0.1:11434/v1, a local model). For each purpose write down in
the table what data it may be shown (problem text, answers, column summaries, findings, evidence text, raw lines)
and test that the gateway's callers never exceed it. A model that fails a purpose's output check twice is
reported on the Admin page, with the purpose and tier.
```

---

## Phase A2: One runtime

### A2.1 The runtime and the tool registry

- **Delivers:** `dclab_rnd/agents/` with `Tool`, `Registry`, `Policy` and `run()`; the 20 intern tools and the Home agent's tools registered once; the MCP server reads the same registry.
- **Done when:** adding a tool is one registration; the intern, the Home agent and `/mcp` all list it; `tests/test_mcp.py` and `tests/test_intern.py` pass unchanged.

```text
Package A2.1. Extract one agent runtime. Write dclab_rnd/agents/: Tool (name, description, JSON schema, handler,
effect: "read" or "write"), Registry, Policy (system prompt, tool names, budget) and run(policy, state, gateway).
run() asks the gateway, validates the arguments against the tool's schema (a bad call is returned to the model
as an error it can read, not raised), executes the tool, and loops until a final answer, the step budget or the
time budget. Register the 20 intern tools (dclab_rnd/intern/tools.py) and the Home agent's tools once. Make
dclab_rnd/mcp_server.py list the registry. Do not change any tool's behaviour or name; keep the legacy names.
```

### A2.2 The validator in front of every write

- **Delivers:** every `write` tool goes through `studio.graph.check` inside the runtime; a refusal becomes the tool result with the rule it broke.
- **Done when:** a test calls every write tool with a move the validator refuses and gets the reason and a logged transition; no write tool bypasses the check.

```text
Package A2.2. Make the validator impossible to skip. In the runtime, every tool with effect "write" declares the
workflow move it makes (run_stage, set_solution, approve_gate, approve_stage, capture). Before the handler runs,
the runtime calls studio.graph.check for that move with the actor "agent"; a refusal is logged by graph.log and
returned to the model as the tool result with the verdict's reason and the rules it cites. Add a test that
enumerates the registry and fails if a write tool has no declared move. The Home agent's writes (record,
set_pack, propose_workflow, simulate_data) change a draft, not a project: give them their own checks (workflow
validation, pack list, one simulation at a time).
```

### A2.3 Traces

- **Delivers:** an `agent_step` table; every runtime step stored with what the agent saw (a summary), the tool and arguments, the verdict, the result summary, tokens and time; a trace view on the Intern page.
- **Done when:** any finished run can be replayed from its trace with a scripted model and gives the same moves.

```text
Package A2.3. Store a trace for every run. Add an agent_step table (run id, step number, state summary, tool,
arguments, verdict, result summary, tokens, seconds) through an Alembic migration. The runtime writes a row per
step. Add replay(run_id): feed the stored assistant messages back through the runtime with a scripted gateway
and check that the same moves result. Show the trace on the Intern page with the existing step list. Never store
cell values: arguments and results are the summaries the tools already return.
```

### A2.4 The three agents as policies

- **Delivers:** the Home agent, the intern and the campaign agent run on the runtime with their behaviour unchanged.
- **Done when:** `make product-e2e` and the agent tests pass; `draft/chat.py`'s own loop and `intern/loop.py`'s loop are gone; the campaign agent runs one campaign on the same runtime.

```text
Package A2.4. Rebuild the three agents as policies on the runtime, one at a time, with the existing tests as the
safety net. Home agent: keep the scripted fallback, the one-turn-per-draft lock and the streamed replies. Intern:
keep the standard plan when no model is configured. Campaign agent (dclab_rnd/agentic/worker.py, NOOA and
LangGraph): put it behind the same runtime and gateway, or if NOOA cannot be wrapped, keep it as an adapter that
reports its requests to the gateway. Delete a loop only when its tests pass on the runtime.
```

---

## Phase A3: Make the agents better at the real job

### A3.1 A planner for the Home agent

- **Delivers:** the agent keeps a short plan (what is known, what is unknown, what to ask next, why) as state; the questions come from the plan, not from a fixed order.
- **Done when:** with a scripted model, a problem sentence that already states the outcome and the moment is not asked those again; the question count falls without losing a field.

```text
Package A3.1. Today the Home agent asks the same four things in order. Give it a plan in the draft: for each of
outcome, prediction moment, action and cost, and data, the status (stated, inferred, unknown) and where it came
from. After the problem sentence and after each answer, update the plan from the text (a model proposes, code
checks that the quoted words are in the user's text). Ask only unknown fields, in the order that matters most for
leakage (the prediction moment before the cost). Show the plan as the existing "DCLab understood" sheet. The
script path does the same with keyword rules.
```

### A3.2 A leakage reviewer

- **Delivers:** a reviewer step in the wizard and for the intern: it reads the column audit, the prediction moment and the column names, and proposes which columns are unavailable at that moment, each with a reason and a record; code decides what is applied.
- **Done when:** on the studied samples the reviewer's proposals include the R&D's own forbidden columns (for example `duration` on bank_marketing); a column it proposes without a reason is dropped.

```text
Package A3.2. Add a leakage reviewer. Input: the prediction moment text, the column names, kinds and summaries,
and the column audit's flagged list. Output (JSON, validated): per column, "available" or "after the moment", a
one-sentence reason that quotes the moment, and the evidence record IDs that support it (they must exist in the
index). Code drops any item with no reason or a missing record and never un-forbids a column the audit
flagged; the user ticks what to forbid. Without a model, the existing audit and name rules answer. Measure it on
the 16 studied samples: how many of the R&D's forbidden columns it finds, and how many it adds that the R&D did
not (those are shown to the user, never applied).
```

### A3.3 Explanations with citations

- **Delivers:** the notebook, reliability and brief pages can show a short model-written explanation of each result, with every sentence cited and every number checked against the stage record.
- **Done when:** a sentence with a number that is not in the cited record is removed; the page labels the text "written by a model from these records".

```text
Package A3.3. Reuse the Evidence library's answer check (dclab_rnd/agentic/pages/evidence.py: keep only sentences
that cite a retrieved record and whose numbers appear in it) for stage explanations. Input: one stage record
(claims, evidence, notes). Output: at most 120 words, each sentence citing a claim id or a record id. Drop the
rest. Show it on the Notebook and Reliability pages and in the brief, labelled as written by a model from the
stage record, with the deterministic notes unchanged beside it. Never let the explanation say "production-ready".
```

### A3.4 A reviewer for the user's own notebooks

- **Delivers:** the copilot reviews an uploaded notebook with a model for the explanations and code for the findings.
- **Done when:** the leaky demo notebook gives the same findings as today, with a model-written fix per finding that cites the rule.

```text
Package A3.4. The notebook copilot (dclab_rnd/copilot, notebook_assist.py) finds problems deterministically.
Keep every finding exactly as it is. For each finding, ask the model for a fix in two sentences that cites the
rule and the matching pitfall record, and show it under the finding. A finding never depends on the model. Test
with the leaky bank-marketing example notebook and a scripted client.
```

---

## Phase A4: Prove it

### A4.1 The judgment suite

- **Delivers:** `dclab_rnd/agent_eval/` with at least 20 cases, each a small table with a planted trap and the expected behaviour; the scripted run is part of `make rd-check`.
- **Done when:** the standard plan scores on every case; a deliberately broken policy (one that ignores the audit) fails the cases it should.

```text
Package A4.1. Build the judgment suite the Benchmark page describes. Each case: a generated table (seeded), the
trap, the prediction moment, and the expected outcome. Traps: a post-outcome column; a copy of the target with
noise; an identifier that predicts the target; a time column with a random split; group leakage; a duplicated
row set across the split; a column that is a unit conversion of the target; a constant column; a high-missing
column with informative missingness (not a leak). For each case score: leak caught (yes/no), false alarm (yes/no),
move refused by the validator when it should be, and citations that exist. Run the standard plan and a scripted
"bad" policy on all cases in make rd-check, and expect the bad one to fail the right ones.
```

### A4.2 Live runs with a budget

- **Delivers:** `make agent-eval-live`, which needs an explicit flag and prints requests, tokens and cost before it starts and refuses to exceed a stated cap; scores with intervals over repeated runs.
- **Done when:** a run on a local model prints per-case results and a table of scores with their spread; no run starts without the flag.

```text
Package A4.2. Add a live evaluation of a configured model on the A4.1 cases. It must show the estimated number
of requests and the cap and ask for --yes, run each case several times (default 5) because a model's answer
varies, and report each score as a mean with a 95% interval, plus tokens and cost from the gateway. Store the
result under evidence/campaigns with a run id and the model name so it can be compared later. Say in the output
that scores on these cases are evidence about this model on these traps, not about production readiness.
```

### A4.3 A regression gate for prompt and tool changes

- **Delivers:** a change to a policy prompt, a tool schema or a validator rule must pass the scripted suite; the live results of the last release are the baseline to compare.
- **Done when:** a test fails when a prompt file changes without its recorded hash being updated by the suite run.

```text
Package A4.3. Treat prompts as code with tests. Keep each policy's system prompt and tool schemas in versioned
files. Record a hash of each next to the latest scripted-suite result; make rd-check fails when a prompt or
schema changed and the suite has not been re-run for that hash. When a live baseline exists, print the change
in score per case after a run. No prompt text is edited by a model.
```

---

## Phase A5: Retrieval and memory

### A5.1 Better evidence search

- **Delivers:** hybrid retrieval (the current keyword search plus embeddings), measured on a set of real questions with known answers.
- **Done when:** recall at 5 on the question set is reported for keyword only and for hybrid; hybrid is used only if it is not worse on any question group.

```text
Package A5.1. The evidence index uses BM25 only. Write 60 questions with the record IDs that answer them (from
the rules, pitfalls and experiments; include paraphrases that share no words with the record). Measure recall at
5 for BM25. Add an embedding retriever (a local sentence-embedding model or an OpenAI-compatible embeddings
endpoint through the gateway) stored in PostgreSQL (pgvector) or a file when pgvector is absent, and a hybrid
that merges both lists. Report recall at 5 for each and use hybrid only if it is not worse on any group of
questions. The index stays generated: rebuild it with make knowledge.
```

### A5.2 Project memory

- **Delivers:** what a project has learned (decisions, why, user corrections) kept as short notes the agents read, each tied to the move that produced it.
- **Done when:** a second intern session on the same project reads the first session's decisions instead of asking again.

```text
Package A5.2. Give each project a memory: short notes (decision, reason, who, which move, date), written when a
stage is approved, a solution is saved, a gate is approved or the user corrects the agent. Agents read the notes
as part of their state (the latest 20, summarised). A note never holds a cell value. Show the notes on the
Solution & data page. A note can be removed by a person; the audit trail keeps the removal.
```

### A5.3 Workspace lessons

- **Delivers:** lessons from finished projects, reviewed by a person, that can join the evidence index with their scope and the project they came from.
- **Done when:** a lesson accepted by a reviewer appears in the evidence index with its scope and is cited by an agent.

```text
Package A5.3. Build the "From projects" flow the Evidence library says is not built. After a project's final
stage, the agent proposes up to three lessons, each with its claim, its scope (this dataset, this task, this size),
what argues against it and the next test, citing the stage records. A reviewer accepts, edits or rejects each
(role: reviewer). Accepted lessons are stored in the workspace's lessons table and searched together with the
repository's records, labelled as workspace lessons; they never change evidence/knowledge. A lesson from synthetic
data is labelled synthetic and is not offered as evidence.
```

---

## Phase A6: Learn from runs

### A6.1 Trajectory export

- **Delivers:** finished runs exported as training records: state, move, validator verdict, evidence, outcome; only from runs the owner opted in.
- **Done when:** `python -m dclab_rnd.studio.sft --trajectories` writes records for opted-in projects and none for others; no cell value appears in any record.

```text
Package A6.1. Turn the traces (A2.3) and the transition logs into trajectory examples: (state summary, allowed
tools, chosen move, validator verdict, cited records, what happened next). Export only projects whose owner opted
in (a setting on the project; off by default; says so on the Policy model page). Remove anything that could hold
a value: tool arguments are kept only as the schema's non-free-text fields. Write to a new folder, never into
out_v3. Add a test that scans an export of a project built on the Telco CSV for any cell value.
```

### A6.2 The data recipe and the held-out split

- **Delivers:** a corpus v4 recipe that mixes declarative examples (v3) with trajectories, held out by whole projects, with the number of held-out projects shown against the threshold.
- **Done when:** the Policy model page shows "N of 50 trajectories" from the real count; stage 2 is marked ready only at the threshold.

```text
Package A6.2. Write the recipe for corpus v4: the 330 declarative examples of v3 plus trajectory examples,
split by whole project (about one in five held out), with the same quality gates as v3 (no benchmark
contamination, no near-duplicate prompts across the split). Do not generate anything until at least 50
trajectories from at least 10 different projects exist; until then print how many exist. Keep generated files out
of version control as the repository does for out_v3.
```

### A6.3 Train and evaluate a small policy model

- **Delivers:** a LoRA run on a small model with the existing scripts, evaluated on the A4.1 suite against the standard plan and the untuned base model.
- **Done when:** the Benchmark page shows the three side by side with intervals; the tuned model is promoted only if it is not worse on any trap class.

```text
Package A6.3. Only when A6.2's threshold is met and I approve the compute and cost. Run research/llm-fine-tuning's
train_lora.py on the v4 corpus, then evaluate the tuned model, the base model and the standard plan on the A4.1
cases with A4.2's protocol. Report per trap class with intervals. Do not call the tuned model better unless it
is not worse on every class and better on the total. Register it as a tier in the gateway behind a switch, off by
default.
```

### A6.4 Serve it as a tier, with a safety net

- **Delivers:** the tuned model as an optional cheap or standard tier, with a shadow mode that runs it beside the current model and logs disagreements without acting on them.
- **Done when:** shadow mode logs disagreements for a week of scripted traffic and the validator still sees only the production model's moves.

```text
Package A6.4. Add shadow mode to the gateway: a purpose can name a shadow model that receives the same request;
its answer is stored and compared (same tool chosen? same validated output?) but never used. Show agreement
rates on the Admin page. Switching a purpose to the tuned model requires a reviewer's approval, recorded in the
audit trail, and can be undone in one setting.
```

---

## 5. How a package is checked

Every package in this document is finished only when all of these hold:

1. `make rd-check` passes on files, and `make test-pg` passes on PostgreSQL.
2. `make product-e2e` passes (a model path is also run once with a scripted client).
3. A unit test fails without the change (it was written first or shown to fail).
4. No model request left the machine in a test (the gateway refuses a live endpoint under test).
5. The Admin page's statement about what a model may see is still true; if the package changes it, the page and `.env.example` change in the same commit.

## 6. What this plan does not do

- It does not make a model's judgment trustworthy by itself. Scores here are evidence about a model on these traps; the validator and the user's approvals are what protect a project.
- It does not train anything without enough real trajectories (A6.2) and your approval of the cost (A6.3).
- It does not send raw rows to a model. Sample lines for an unreadable file are the one disclosed exception, and `DCLAB_MODEL_READS_SAMPLE_LINES=0` turns them off.
