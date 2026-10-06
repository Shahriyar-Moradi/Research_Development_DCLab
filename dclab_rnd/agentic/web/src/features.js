/* Feature registry for the blueprint layer.
   status: built (exists in the repository today), partial (some of it exists), todo (designed, not built),
   research (needs an experiment before it can be designed properly).
   phase: Now, P1 proof core, P2 reach, P3 domain packs, P4 own model, R research. */
window.FEATURES = [
  // ---- shell
  { id: 'shell.nav', view: 'shell', name: 'One navigation for every product surface', status: 'partial', phase: 'P1', where: 'dclab_rnd/agentic/static/index.html', note: 'Today: Projects, Intern, Research map, Knowledge, Agent campaigns. The demo adds Compute, Benchmark, Policy model, Packs, Integrations, Admin.' },
  { id: 'shell.roles', view: 'shell', name: 'Role views: developer, business, researcher, admin', status: 'todo', phase: 'P2', where: '', note: 'Same project, different depth. Business hides code and tool calls.' },
  { id: 'shell.palette', view: 'shell', name: 'Command palette (Ctrl/⌘ K) over pages, actions and evidence', status: 'todo', phase: 'P2', where: '', note: 'Searches the real evidence records in this demo.' },
  { id: 'shell.tours', view: 'shell', name: 'Guided tours for new users', status: 'todo', phase: 'P2', where: '', note: 'Onboarding for developers and for business readers.' },

  // ---- home
  { id: 'home.composer', view: 'home', name: 'Start from one sentence', status: 'built', phase: 'Now', where: 'draft/api.py · draft/chat.py · views/home.html', note: 'One sentence starts a draft on Home; the agent answers on the live event stream.' },
  { id: 'home.inbox', view: 'home', name: '"Needs you" inbox: questions, approvals, sign-offs', status: 'built', phase: 'Now', where: 'server.py /api/workspace', note: 'Open questions from drafts and projects without a solution.' },
  { id: 'home.projects', view: 'home', name: 'Projects table with 10-step progress and honest score', status: 'partial', phase: 'P1', where: 'server.py /api/workspace · views/home.html', note: 'Real projects with their ten-step state; they open in the classic notebook until their pages are wired.' },
  { id: 'home.kpis', view: 'home', name: 'Workspace health: leaks blocked, holdout discipline, spend', status: 'built', phase: 'Now', where: 'server.py /api/workspace', note: 'Counted from projects and the transition log. Model spend is on the Compute page.' },
  { id: 'home.activity', view: 'home', name: 'Live activity (intern sessions, jobs, approvals)', status: 'built', phase: 'Now', where: 'pages/ops.py /api/ops/jobs · views/home.html', note: 'One feed of running and queued stage runs, intern sessions, data pipelines and research runs; refreshes while something runs.' },
  { id: 'home.evidence', view: 'home', name: 'Evidence updates (new records, rules strengthened)', status: 'built', phase: 'Now', where: 'pages/ops.py /api/ops/evidence-recent', note: 'Newest records from the evidence index, dated by the campaign result they cite; undated records follow in file order.' },
  { id: 'home.quickstarts', view: 'home', name: 'Example tasks per domain pack', status: 'built', phase: 'Now', where: 'intern/sessions.py EXAMPLE_TASKS', note: '' },

  // ---- new project
  { id: 'home.bring', view: 'home', name: 'Bring the data on Home: upload, samples, connectors, synthetic', status: 'built', phase: 'Now', where: 'draft/api.py · draft/pipeline.py', note: 'Upload, studied samples, Kaggle, Hugging Face, databases, S3/GCS and synthetic data, all through the same pipeline.' },
  { id: 'home.chat', view: 'home', name: 'Home conversation: a few questions, data analysis as it gets ready', status: 'built', phase: 'Now', where: 'draft/chat.py · intern/llm.py stream()', note: 'With a model, replies stream into the thread as they are written; without one, a deterministic question script.' },
  { id: 'home.solution', view: 'home', name: 'Solution workflow for this problem, updated from the chat', status: 'built', phase: 'Now', where: 'draft/workflow.py', note: 'Every step maps to a WF block; gates added by code; a model proposal is validated first.' },
  { id: 'home.pack', view: 'home', name: 'Optional domain pack choice on Home', status: 'built', phase: 'Now', where: 'draft/pack.py', note: '' },
  { id: 'new.data', view: 'new', name: 'Data step: structure, clean and describe before the target is used', status: 'built', phase: 'Now', where: 'draft/pipeline.py', note: '' },
  { id: 'new.synthetic', view: 'new', name: 'Synthetic data from a description, labelled synthetic everywhere', status: 'built', phase: 'Now', where: 'draft/synthetic.py', note: 'With a model, the table is designed from the description (validated by code). Without one, the user picks a built-in template, and the chat says the rows do not follow the description. Seeded generation.' },
  { id: 'new.goal', view: 'new', name: 'Goal in plain words, parsed into unit, target, decision', status: 'built', phase: 'Now', where: 'draft/chat.py · views/new.html', note: '"DCLab understood" is filled from the Home conversation.' },
  { id: 'new.pack', view: 'new', name: 'Domain pack picker with maturity labels', status: 'built', phase: 'Now', where: 'draft/pack.py', note: 'Detected from the problem and the data; the user can change it. Packs beyond tabular still run the tabular engine.' },
  { id: 'new.upload', view: 'new', name: 'Upload CSV / Parquet / Excel', status: 'built', phase: 'Now', where: 'draft/structure.py · draft/clean.py', note: 'Any file: tables, JSON, JSON lines, logs and text are turned into a table, then cleaned with a logged, lossless step list.' },
  { id: 'new.samples', view: 'new', name: 'Sample library: 16 studied datasets with their solutions', status: 'built', phase: 'Now', where: 'studio/data.py sample_catalog', note: '' },
  { id: 'new.hub', view: 'new', name: 'Import from Kaggle and Hugging Face', status: 'partial', phase: 'Now', where: 'connectors/kaggle.py · connectors/hf.py', note: 'Search and import are wired with licence, version and SHA-256 recorded; not yet tested against the live Kaggle and Hugging Face services.' },
  { id: 'new.warehouse', view: 'new', name: 'Read-only warehouse connectors', status: 'built', phase: 'Now', where: 'connectors/db.py · connectors/cloud.py', note: 'Read-only SELECT on server-defined connections (DCLAB_DB_<NAME>); S3 and GCS objects. Postgres and MySQL need their drivers installed.' },
  { id: 'new.profile', view: 'new', name: 'Data profile: kinds, missing, unique, candidates', status: 'built', phase: 'Now', where: 'studio/data.py profile_table · tool describe_data', note: '' },
  { id: 'new.lineage', view: 'new', name: 'Lineage capture: source, license, snapshot hash, time range', status: 'partial', phase: 'P1', where: 'dclab_rnd/provenance.py', note: 'Provenance exists for campaigns, not yet for user uploads.' },
  { id: 'new.solution', view: 'new', name: 'Solution draft with forbidden columns and proof', status: 'built', phase: 'Now', where: 'studio/solution.py propose · tools._audit_frame', note: '' },
  { id: 'new.split', view: 'new', name: 'Split strategy: time, group or stratified', status: 'built', phase: 'Now', where: 'studio/engine.py split_for', note: 'The solution decides: a time column means the latest period is the holdout, a group column keeps groups on one side, otherwise stratified. The wizard shows it and cannot contradict it.' },
  { id: 'new.budget', view: 'new', name: 'Compute and budget choice', status: 'partial', phase: 'P2', where: 'draft/api.py settings · intern budget', note: 'Rows, quick mode and folds reach the stages and the exported notebook; the euro budget is the project\'s monthly cap on priced model requests. The call and minute budgets apply to intern sessions. Sandboxes and GPU jobs are not switched on yet.' },
  { id: 'new.handoff', view: 'new', name: 'Start in the notebook or hand to the intern', status: 'built', phase: 'Now', where: 'draft/api.py build · views/new.html', note: 'Creates the project with data, solution and settings; opens the classic notebook until the new project pages are wired.' },

  // ---- project workflow graph
  { id: 'proj.graph', view: 'project', name: 'Workflow graph WF-01…WF-10 with allowed moves', status: 'built', phase: 'P1', where: 'dclab_rnd/studio/graph.py · Graph tab in views/project.js', note: 'The LLM proposes a move; the graph decides if the move exists. Revisit edges are listed; free reopening of earlier steps is next.' },
  { id: 'proj.validator', view: 'project', name: 'Transition validator (edge, prerequisites, evidence IDs, approvals)', status: 'built', phase: 'Now', where: 'studio/graph.py check · views/project.html', note: 'Checks the real project\'s moves through /graph/check.' },
  { id: 'proj.log', view: 'project', name: 'Transition log: every move with who, why and evidence', status: 'built', phase: 'P1', where: 'transitions.jsonl per project · store.transition()', note: 'Actor, move, state string, verdict, failed checks, rules, evidence and outcome.' },
  { id: 'proj.replay', view: 'project', name: 'Replay a project step by step', status: 'built', phase: 'Now', where: '', note: 'Replays the real transition log step by step.' },
  { id: 'proj.stages', view: 'project', name: 'Stage runs with prerequisites ("run X first")', status: 'built', phase: 'Now', where: 'studio/engine.py execute', note: '' },
  { id: 'proj.approvals', view: 'project', name: 'Approve a stage or choose another option', status: 'built', phase: 'Now', where: 'studio/engine.py approve', note: '' },
  { id: 'proj.revisit', view: 'project', name: 'Go back to an earlier step with a reason', status: 'todo', phase: 'P1', where: '', note: 'Today a new solution clears results; targeted revisits are new.' },
  { id: 'proj.team', view: 'project', name: 'Project members and reviewers', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'proj.projects', view: 'project', name: 'Switch between projects (HyperAck waits on the owner)', status: 'built', phase: 'Now', where: 'studio/store.py ProjectStore.list', note: 'The three most recent projects, then the samples.' },

  // ---- solution & data
  { id: 'solution.fields', view: 'solution', name: 'Solution fields: target, task, moment, forbidden, IDs, time, group, text, metric', status: 'built', phase: 'Now', where: 'studio/solution.py Solution', note: '' },
  { id: 'solution.timeline', view: 'solution', name: 'Prediction-moment timeline: what is known when', status: 'built', phase: 'Now', where: '', note: 'Forbidden columns placed after the prediction moment; open ones in amber.' },
  { id: 'solution.costs', view: 'solution', name: 'Decision costs choose the metric and threshold', status: 'partial', phase: 'Now', where: 'studio/engine.py stage_final threshold_analysis', note: 'Calculator runs on the real holdout ROC-AUC; costs are examples until the owner enters them.' },
  { id: 'solution.signoff', view: 'solution', name: 'Owner sign-off with versions', status: 'built', phase: 'Now', where: 'server.py /approvals · studio/graph.py', note: 'Sign a version (approvals gate); versions come from set_solution moves.' },
  { id: 'solution.worksheet', view: 'solution', name: 'Owner worksheet: questions only the owner can answer', status: 'built', phase: 'Now', where: '', note: 'Built from the solution, the Home conversation and open items.' },
  { id: 'solution.lineage', view: 'solution', name: 'Source and lineage gate (WF-02)', status: 'partial', phase: 'P1', where: 'dclab_rnd/provenance.py · dataset cards', note: '' },
  { id: 'solution.split', view: 'solution', name: 'Split design with independence unit and sealed holdout (WF-03)', status: 'built', phase: 'Now', where: 'studio/engine.py prepare', note: 'The visual is new; the logic exists.' },
  { id: 'solution.checks', view: 'solution', name: 'Cross-split duplicate and entity checks', status: 'partial', phase: 'P1', where: 'group split · expansion runner', note: '' },

  // ---- notebook
  { id: 'nb.stagecells', view: 'notebook', name: 'Stage cells with records (today\'s notebook)', status: 'built', phase: 'Now', where: 'views/project.js · studio/engine.py', note: '' },
  { id: 'nb.cells', view: 'notebook', name: 'Live code cells on a sandboxed kernel', status: 'todo', phase: 'P2', where: '', note: 'The step from "agentic notebook" to "replaces Colab".' },
  { id: 'nb.companion', view: 'notebook', name: 'Proof beside every cell (the companion)', status: 'built', phase: 'Now', where: 'server.py /review · dclab_rnd/copilot', note: 'Real copilot findings on the exported notebook, read-only route /review.' },
  { id: 'nb.fix', view: 'notebook', name: 'One-click fix with a diff', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'nb.precedent', view: 'notebook', name: '"Someone hit this before": precedents and pitfalls', status: 'built', phase: 'Now', where: 'copilot proof · LEAK-*/PIT-* records', note: '' },
  { id: 'nb.agentcells', view: 'notebook', name: 'Agent writes cells; you accept or reject', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'nb.ask', view: 'notebook', name: 'Ask the project (answers cite records)', status: 'built', phase: 'Now', where: 'studio/agent.py answer · tool ask_project', note: '' },
  { id: 'nb.export', view: 'notebook', name: 'Export .ipynb, report and training examples', status: 'built', phase: 'Now', where: 'studio/export.py · studio/sft.py', note: '' },
  { id: 'nb.openin', view: 'notebook', name: 'Open in VS Code, Jupyter or Colab', status: 'partial', phase: 'P2', where: '.ipynb export', note: '' },
  { id: 'nb.vars', view: 'notebook', name: 'Variables and data inspector', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'nb.review', view: 'notebook', name: 'Review any imported notebook', status: 'built', phase: 'Now', where: 'python -m dclab_rnd.copilot review', note: 'CLI today; the in-app import is new.' },

  // ---- audit
  { id: 'audit.matrix', view: 'audit', name: 'Availability matrix: every column against the prediction moment', status: 'built', phase: 'Now', where: 'solution forbidden list · _audit_frame', note: 'Every profiled column with its role and verdict.' },
  { id: 'audit.heuristics', view: 'audit', name: 'Suspicion heuristics (name, univariate signal, proxy, uniqueness, canary)', status: 'built', phase: 'Now', where: 'tools._audit_frame · expansion suspicious_features', note: '' },
  { id: 'audit.ablation', view: 'audit', name: 'Safe-vs-unsafe severity ablation on identical folds', status: 'built', phase: 'Now', where: 'studio/engine.py stage_leakage', note: '' },
  { id: 'audit.verdict', view: 'audit', name: 'Verdicts with reasons, and owner questions for unclear columns', status: 'built', phase: 'Now', where: 'solution forbidden reasons', note: '' },
  { id: 'audit.critic', view: 'audit', name: 'Critic review checked against the numbers', status: 'built', phase: 'Now', where: 'dclab_rnd/critic_gate.py', note: '' },
  { id: 'audit.eda', view: 'audit', name: 'Train-only EDA charts', status: 'partial', phase: 'Now', where: 'studio/engine.py stage_data', note: 'Outcome balance, top-value shares and train-vs-holdout PSI; per-category target rates are not recorded.' },
  { id: 'audit.ladder', view: 'audit', name: 'Feature ladder with a predeclared tolerance', status: 'built', phase: 'Now', where: 'studio/engine.py stage_features', note: '' },
  { id: 'audit.lineage', view: 'audit', name: 'Feature lineage cards (availability, units, null behaviour)', status: 'partial', phase: 'Now', where: '', note: 'Top model features traced to source columns; units and null behaviour not recorded.' },

  // ---- models
  { id: 'models.screen', view: 'models', name: 'Algorithm screen on identical folds with spread', status: 'built', phase: 'Now', where: 'studio/engine.py stage_models', note: '' },
  { id: 'models.rule', view: 'models', name: 'Selection rule: mean minus 0.25×std, then runtime', status: 'built', phase: 'Now', where: 'expansion runner rank_rows', note: '' },
  { id: 'models.folds', view: 'models', name: 'Fold-by-fold paired view', status: 'built', phase: 'Now', where: 'fold scores are stored in records', note: 'Stored fold scores of every family.' },
  { id: 'models.tuning', view: 'models', name: 'Conservative tuning: keep only a predeclared gain', status: 'built', phase: 'Now', where: 'studio/engine.py stage_final tuning_decision', note: '' },
  { id: 'models.ops', view: 'models', name: 'Runtime, latency and size per candidate', status: 'partial', phase: 'Now', where: 'fit time recorded', note: 'Fit time measured; scoring latency and size not measured.' },
  { id: 'models.override', view: 'models', name: 'Override the pick with a logged reason', status: 'built', phase: 'Now', where: 'studio/engine.py approve(choice)', note: 'Choose another family through the graph; the later stages are cleared.' },
  { id: 'models.foundation', view: 'models', name: 'Tabular foundation models (TabPFN) in the screen', status: 'research', phase: 'P3', where: 'research/tabular-foundation-models', note: 'Needs a leakage-safe rerun on HyperAck first.' },

  // ---- reliability
  { id: 'rel.holdout', view: 'reliability', name: 'Holdout used once, with a bootstrap interval', status: 'built', phase: 'Now', where: 'studio/engine.py stage_final · bootstrap_ci', note: '' },
  { id: 'rel.seal', view: 'reliability', name: 'Holdout seal: who opened it, when, under which checklist', status: 'built', phase: 'Now', where: '', note: 'Date and who opened it, from the transition log.' },
  { id: 'rel.calibration', view: 'reliability', name: 'Calibration (Brier, ECE, reliability bins)', status: 'built', phase: 'Now', where: 'stage_final calibration_bins', note: '' },
  { id: 'rel.threshold', view: 'reliability', name: 'Operating points (threshold table)', status: 'built', phase: 'Now', where: 'stage_final threshold_analysis', note: '' },
  { id: 'rel.slices', view: 'reliability', name: 'Slice performance with intervals', status: 'todo', phase: 'P1', where: '', note: 'Rule DCLAB-R19.' },
  { id: 'rel.shift', view: 'reliability', name: 'Time and source shift tests', status: 'partial', phase: 'Now', where: 'PIT-005 protocol', note: 'Train-vs-holdout PSI; a time-ordered shift test needs a time column.' },
  { id: 'rel.missing', view: 'reliability', name: 'Missing-input simulation', status: 'todo', phase: 'P2', where: '', note: 'Rule DCLAB-R12.' },
  { id: 'rel.gate', view: 'reliability', name: 'Production gate checklist (R22)', status: 'partial', phase: 'Now', where: '', note: 'Six checks read from the records; fresh data, monitoring and rollback are not recorded yet.' },

  // ---- brief
  { id: 'brief.summary', view: 'brief', name: 'Plain-language decision brief', status: 'built', phase: 'Now', where: 'studio/export.py report', note: 'Written from the stage records.' },
  { id: 'brief.value', view: 'brief', name: 'Value at the chosen operating point, with uncertainty', status: 'partial', phase: 'Now', where: '', note: 'Expected positives at a call volume from the real ROC-AUC; costs are examples.' },
  { id: 'brief.explain', view: 'brief', name: 'Stage explanations written by a model, cited and number-checked', status: 'built', phase: 'Now', where: 'studio/explain.py · cited.py', note: 'Only when a model is configured; sentences that fail the check are removed.' },
  { id: 'brief.risks', view: 'brief', name: 'Risks, assumptions and what is not proven', status: 'built', phase: 'Now', where: 'report sections · critic reviews', note: 'Stage notes with warnings.' },
  { id: 'brief.fa', view: 'brief', name: 'Brief in English and Persian', status: 'built', phase: 'Now', where: '', note: 'The same brief in Persian, with Persian digits.' },
  { id: 'brief.signoff', view: 'brief', name: 'Business sign-off and next decision', status: 'built', phase: 'Now', where: '', note: 'Signs the solution gate with a reason.' },
  { id: 'brief.capture', view: 'brief', name: 'Knowledge capture: lessons to evidence, examples to SFT (WF-10)', status: 'partial', phase: 'P1', where: 'studio/sft.py', note: 'SFT examples from projects exist; project lessons into the evidence index do not.' },
  { id: 'brief.share', view: 'brief', name: 'Share link and exports', status: 'partial', phase: 'P2', where: 'report export', note: '' },

  // ---- intern
  { id: 'intern.session', view: 'intern', name: 'Task → plan → tool calls → report', status: 'built', phase: 'Now', where: 'dclab_rnd/intern/loop.py', note: '' },
  { id: 'intern.standard', view: 'intern', name: 'Standard plan when no model is configured', status: 'built', phase: 'Now', where: 'intern/loop.py _standard_plan', note: '' },
  { id: 'intern.budget', view: 'intern', name: 'Budget: tool calls and minutes (enforced)', status: 'built', phase: 'Now', where: 'intern/sessions.py DEFAULT_BUDGET', note: 'Money and token caps are partial.' },
  { id: 'intern.model', view: 'intern', name: 'Model choice: any OpenAI-compatible endpoint', status: 'partial', phase: 'P2', where: 'intern/llm.py settings (.env)', note: 'Works via .env; the in-app picker is new.' },
  { id: 'intern.ask', view: 'intern', name: 'Blocking questions to the owner', status: 'todo', phase: 'P1', where: '', note: 'The "ask user" move of the graph.' },
  { id: 'intern.guard', view: 'intern', name: 'Guardrails: holdout once, forbidden columns, no deletes', status: 'partial', phase: 'P1', where: 'tools enforce holdout-once and the solution', note: '' },
  { id: 'intern.graphpos', view: 'intern', name: 'Live position on the workflow graph', status: 'built', phase: 'Now', where: '', note: 'The session project\'s graph states.' },
  { id: 'intern.sandbox', view: 'intern', name: 'Sandboxed code execution', status: 'todo', phase: 'P2', where: '', note: 'ML Intern\'s biggest advantage today.' },
  { id: 'intern.jobs', view: 'intern', name: 'GPU jobs (Hugging Face Jobs)', status: 'partial', phase: 'P2', where: 'scripts/chat_ui.py --ml-intern', note: 'Reachable through Chat UI\'s ML Intern mode, not from DCLab\'s own intern.' },
  { id: 'intern.ladder', view: 'intern', name: 'Cost ladder accounting (typed → NOOA → LLM)', status: 'todo', phase: 'P4', where: '', note: '' },

  // ---- compute
  { id: 'compute.local', view: 'compute', name: 'Local runs on this machine', status: 'built', phase: 'Now', where: 'pages/ops.py · views/compute.html', note: 'Every stage run, intern session, data pipeline and research run, with logs, metrics, artifacts and the reproduce fields.' },
  { id: 'compute.sandbox', view: 'compute', name: 'Sandboxes per project', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'compute.gpu', view: 'compute', name: 'GPU jobs with logs, metrics and artifacts', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'compute.repro', view: 'compute', name: 'Reproducibility record (seed, data hash, environment)', status: 'partial', phase: 'P2', where: 'pages/ops.py job detail · stage records', note: 'Seed, data sha256, sampling, folds, holdout and prediction moment per stage; the code version is not recorded yet.' },
  { id: 'compute.spend', view: 'compute', name: 'Spend caps and usage', status: 'built', phase: 'Now', where: 'models/gateway.py · models/prices.py · pages/ops.py', note: 'Every model request is counted and priced from a dated table (never guessed; local models are free). A request that would pass a run, project or workspace cap is refused before it is sent. A model without a configured price is counted in tokens only.' },

  // ---- evidence
  { id: 'ev.search', view: 'evidence', name: 'Evidence search (filter, then rank)', status: 'built', phase: 'Now', where: 'dclab_rnd/evidence_index.py · pages/evidence.py GET /api/evidence', note: 'The page reads the live index from the server; the bundled snapshot is only the first paint.' },
  { id: 'ev.record', view: 'evidence', name: 'Record viewer with citations', status: 'built', phase: 'Now', where: 'drawer.js openRecord', note: '' },
  { id: 'ev.rules', view: 'evidence', name: 'The 22 rules with their evidence', status: 'built', phase: 'Now', where: 'evidence/knowledge/model_building_rules.jsonl', note: '' },
  { id: 'ev.workflow', view: 'evidence', name: 'The 10 workflow blocks', status: 'built', phase: 'Now', where: 'evidence/knowledge/workflow_blocks.json', note: '' },
  { id: 'ev.scope', view: 'evidence', name: 'Scoped claims with counter-evidence and the next test', status: 'partial', phase: 'P2', where: 'rule records carry "next test"', note: 'Rule DCLAB-R21.' },
  { id: 'ev.ask', view: 'evidence', name: 'Ask the evidence (answers cite records)', status: 'partial', phase: 'P1', where: 'pages/evidence.py POST /api/evidence/ask', note: 'Search, then a model answer kept only where citations and numbers check out, or the closest records. The page asks four fixed questions; the route takes any.' },
  { id: 'ev.guide', view: 'evidence', name: 'Field guide and outside-reader narrative', status: 'built', phase: 'Now', where: 'evidence/knowledge/MODEL_BUILDING_FIELD_GUIDE.html', note: '' },
  { id: 'ev.kaggle', view: 'evidence', name: 'Kaggle solution corpus ("50 challenges solved")', status: 'todo', phase: 'P3', where: '', note: '' },
  { id: 'ev.contribute', view: 'evidence', name: 'Project lessons flow into the evidence', status: 'todo', phase: 'P1', where: '', note: '' },

  // ---- lab
  { id: 'lab.registry', view: 'lab', name: 'Experiment registry (569 of 579 validated)', status: 'built', phase: 'Now', where: 'dclab_rnd/registry.py', note: '' },
  { id: 'lab.campaigns', view: 'lab', name: 'Campaigns and their results', status: 'built', phase: 'Now', where: 'evidence/campaigns/', note: '' },
  { id: 'lab.designer', view: 'lab', name: 'Campaign designer (hypothesis × datasets × protocol)', status: 'partial', phase: 'P2', where: 'dclab_rnd/agentic/worker.py', note: 'The LLM campaign loop exists; the designer form is new.' },
  { id: 'lab.critic', view: 'lab', name: 'Critic gate: LLM critique checked against numbers', status: 'built', phase: 'Now', where: 'dclab_rnd/critic_gate.py', note: '' },
  { id: 'lab.pitfalls', view: 'lab', name: 'Pitfall experiments with measured costs', status: 'built', phase: 'Now', where: 'dclab_rnd/pitfalls.py', note: '' },
  { id: 'lab.map', view: 'lab', name: 'Research map (22 tracks)', status: 'built', phase: 'Now', where: 'dclab_rnd/research_map.py', note: '' },

  // ---- benchmark
  { id: 'bench.pilot', view: 'benchmark', name: 'Notebook pilot scorecard', status: 'built', phase: 'Now', where: 'dclab_rnd/notebook_eval.py', note: '' },
  { id: 'bench.suites', view: 'benchmark', name: 'Judgment suites with planted traps', status: 'todo', phase: 'P1', where: '', note: 'Proves that DCLab judges correctly on work it has not seen.' },
  { id: 'bench.board', view: 'benchmark', name: 'Policy leaderboard (standard plan, LLMs, own model)', status: 'todo', phase: 'P1', where: '', note: '' },
  { id: 'bench.metrics', view: 'benchmark', name: 'Metrics: valid moves, leaks caught, citations, unsafe actions, tokens', status: 'todo', phase: 'P1', where: '', note: '' },
  { id: 'bench.case', view: 'benchmark', name: 'Case replay with trajectory and verdict', status: 'todo', phase: 'P1', where: '', note: '' },
  { id: 'bench.human', view: 'benchmark', name: 'Human agreement study', status: 'todo', phase: 'P4', where: '', note: '' },

  // ---- policy model
  { id: 'policy.sft', view: 'policy', name: 'SFT corpus v3 (330 examples, 7 tasks)', status: 'built', phase: 'Now', where: 'pages/learn.py · sft/out_v3 manifest', note: '' },
  { id: 'policy.projectsft', view: 'policy', name: 'Training examples from finished projects', status: 'built', phase: 'Now', where: 'studio/sft.py', note: '' },
  { id: 'policy.traj', view: 'policy', name: 'Graph trajectories (state → move → evidence)', status: 'partial', phase: 'P4', where: 'pages/learn.py · transitions.jsonl', note: 'Every move is logged and counted; the exporter that turns logs into training records is not built.' },
  { id: 'policy.review', view: 'policy', name: 'Expert review queue for decisions', status: 'todo', phase: 'P4', where: '', note: '' },
  { id: 'policy.train', view: 'policy', name: 'Training runs (LoRA / QLoRA)', status: 'partial', phase: 'P4', where: 'sft/train_lora.py', note: 'Script ready; no run yet.' },
  { id: 'policy.eval', view: 'policy', name: 'Evaluation against the benchmark', status: 'partial', phase: 'P4', where: 'sft/eval_sft.py', note: '' },
  { id: 'policy.deploy', view: 'policy', name: 'Serve as the intern\'s model', status: 'partial', phase: 'P4', where: 'intern/llm.py (OpenAI-compatible)', note: 'Any OpenAI-compatible server works; routing and A/B are new.' },
  { id: 'policy.ladder', view: 'policy', name: 'Cost ladder: typed classifier → NOOA → LLM', status: 'partial', phase: 'P4', where: 'dclab_rnd/agentic/agents.py (NOOA)', note: 'The typed classifier tier is not in the repository yet.' },
  { id: 'policy.macro', view: 'policy', name: 'Macro-actions ("cake" tokens)', status: 'research', phase: 'R', where: 'research/workflow-actions', note: '' },

  // ---- packs
  { id: 'packs.tabular', view: 'packs', name: 'Tabular: binary, multiclass, regression', status: 'built', phase: 'Now', where: 'studio/engine.py capabilities', note: '' },
  { id: 'packs.imbalanced', view: 'packs', name: 'Imbalanced and fraud (PR-AUC, time split)', status: 'built', phase: 'Now', where: 'EXP-051…055', note: '' },
  { id: 'packs.text', view: 'packs', name: 'Text + tabular', status: 'partial', phase: 'P3', where: 'TF-IDF text columns (binary only)', note: '' },
  { id: 'packs.timeseries', view: 'packs', name: 'Time-series forecasting (horizons, backtests)', status: 'partial', phase: 'P3', where: 'time split + regression (EXP-061…065)', note: '' },
  { id: 'packs.cv', view: 'packs', name: 'Computer vision (detection, classification)', status: 'todo', phase: 'P3', where: 'research/computer-vision', note: '' },
  { id: 'packs.driving', view: 'packs', name: 'Driving perception with an ODD solution', status: 'research', phase: 'P3', where: 'research/driving-maps', note: '' },
  { id: 'packs.maps', view: 'packs', name: 'Maps and road flow (graphs)', status: 'research', phase: 'R', where: 'research/driving-maps · temporal-gnn', note: '' },
  { id: 'packs.scene', view: 'packs', name: 'Scene graphs (video + VLM)', status: 'research', phase: 'R', where: 'research/vision-scene-graphs', note: '' },
  { id: 'packs.llm', view: 'packs', name: 'LLM / SLM fine-tuning', status: 'partial', phase: 'P4', where: 'research/llm-fine-tuning', note: '' },
  { id: 'packs.software', view: 'packs', name: 'Software bug-fixing (first non-ML pack)', status: 'research', phase: 'R', where: 'research/cross-industry-workflows', note: '' },
  { id: 'packs.builder', view: 'packs', name: 'Pack builder (steps, rules, tools, metrics, validators)', status: 'todo', phase: 'P3', where: '', note: '' },

  // ---- integrations
  { id: 'int.mcp', view: 'integrations', name: 'MCP endpoint /mcp with 21 tools', status: 'built', phase: 'Now', where: 'dclab_rnd/mcp_server.py', note: '' },
  { id: 'int.chatui', view: 'integrations', name: 'Hugging Face Chat UI + ML Intern mode', status: 'built', phase: 'Now', where: 'scripts/chat_ui.py · make chat-ui', note: '' },
  { id: 'int.api', view: 'integrations', name: 'REST API', status: 'built', phase: 'Now', where: 'dclab_rnd/agentic/server.py', note: '' },
  { id: 'int.cli', view: 'integrations', name: 'CLI (python -m dclab_rnd, copilot review)', status: 'built', phase: 'Now', where: 'dclab_rnd/cli.py', note: '' },
  { id: 'int.vscode', view: 'integrations', name: 'VS Code companion extension', status: 'partial', phase: 'P2', where: 'dclab_rnd/notebook_assist.py', note: 'The engine exists; the extension does not.' },
  { id: 'int.connectors', view: 'integrations', name: 'Data connectors (Kaggle, HF Hub, S3, warehouses)', status: 'built', phase: 'Now', where: 'dclab_rnd/connectors/ (Kaggle, Hugging Face, databases, S3/GCS)', note: 'Credentials stay on the server; every import goes through the draft pipeline.' },
  { id: 'int.github', view: 'integrations', name: 'GitHub: open a PR with notebook and report', status: 'todo', phase: 'P2', where: '', note: '' },

  // ---- admin
  { id: 'admin.models', view: 'admin', name: 'Model endpoints and routing per tier', status: 'partial', phase: 'P2', where: '.env OPENAI_BASE_URL / model', note: '' },
  { id: 'admin.keys', view: 'admin', name: 'Keys stay on the server, never in the browser', status: 'built', phase: 'Now', where: 'server reads .env', note: '' },
  { id: 'admin.policies', view: 'admin', name: 'Governance policies as switches', status: 'built', phase: 'Now', where: 'pages/platform.py · PATCH /api/projects/{id}', note: 'Locked invariants are checked live; the two gate switches are set per project and every change is audited.' },
  { id: 'admin.roles', view: 'admin', name: 'Roles and permissions', status: 'todo', phase: 'P2', where: '', note: '' },
  { id: 'admin.budgets', view: 'admin', name: 'Budgets and caps', status: 'built', phase: 'Now', where: 'models/gateway.py caps · pages/platform.py', note: 'Intern sessions: tool calls and minutes, plus an optional euro cap; projects: the wizard budget as a monthly cap; the workspace: DCLAB_WORKSPACE_MONTHLY_EUR.' },
  { id: 'admin.audit', view: 'admin', name: 'Audit log', status: 'built', phase: 'Now', where: 'pages/platform.py /api/platform/audit', note: 'Validated moves, gate approvals and gate switch changes across every project, newest first.' },
  { id: 'admin.data', view: 'admin', name: 'Data retention and privacy', status: 'partial', phase: 'P2', where: 'pages/platform.py limits.privacy', note: 'What a model sees is stated from the code; no personal-data scan and no timed retention yet.' },
];
