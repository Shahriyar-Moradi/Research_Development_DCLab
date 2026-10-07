/* GENERATED from the server's OpenAPI schema by `make web` (dclab_rnd/agentic/web/client.py, package 13.1).
   Do not edit: change the route, then rebuild. One function per route under /api: DC.client.<name>(path params…, {query, body, headers}). */

(function () {
  'use strict';

  /** @typedef {Object} Answer
   *  @property {*=} answer
   *  @property {*=} proof
   */
  /** @typedef {Object} AnyObject
   */
  /** @typedef {Object} Config
   *  @property {(string|null)=} csrf
   *  @property {(boolean|null)=} api_key_configured
   *  @property {(string|null)=} default_model
   *  @property {(Array<*>|null)=} projects
   *  @property {*=} datasets
   */
  /** @typedef {Object} Doc
   */
  /** @typedef {Object} Draft
   *  @property {*=} id
   *  @property {*=} problem
   *  @property {*=} status
   *  @property {*=} assets
   *  @property {*=} solution
   */
  /** @typedef {Object} DraftCreate
   *  @property {*=} problem
   *  @property {*=} pack
   */
  /** @typedef {Object} Fixes
   *  @property {*=} fixes
   *  @property {*=} findings
   */
  /** @typedef {Object} GateApproval
   *  @property {*=} gate
   *  @property {*=} reason
   */
  /** @typedef {Object} Graph
   *  @property {*=} nodes
   *  @property {*=} current
   *  @property {*=} state
   *  @property {*=} moves
   *  @property {*=} transitions
   */
  /** @typedef {Object} HTTPValidationError
   *  @property {Array<ValidationError>=} detail
   */
  /** @typedef {Object} InternMessage
   *  @property {*=} text
   */
  /** @typedef {Object} InternStart
   *  @property {*=} task
   *  @property {*=} project_id
   *  @property {*=} budget
   */
  /** @typedef {Object} InternStatus
   *  @property {*=} mode
   *  @property {*=} model
   *  @property {*=} tools
   */
  /** @typedef {Object} Lesson
   *  @property {*=} id
   *  @property {*=} status
   *  @property {*=} claim
   *  @property {*=} scope
   */
  /** @typedef {Object} LessonReview
   *  @property {*=} action
   *  @property {*=} reason
   *  @property {*=} claim
   *  @property {*=} against
   *  @property {*=} next_test
   */
  /** @typedef {Object} Lessons
   *  @property {(Array<*>|null)=} lessons
   *  @property {*=} label
   */
  /** @typedef {Object} ModelsSummary
   *  @property {*=} tiers
   *  @property {*=} purposes
   *  @property {*=} usage
   *  @property {*=} agreement
   */
  /** @typedef {Object} MoveCheck
   *  @property {*=} move
   *  @property {*=} actor
   *  @property {*=} stage
   *  @property {*=} choice
   *  @property {*=} gate
   *  @property {*=} reuse_reason
   */
  /** @typedef {Object} Page
   */
  /** @typedef {Object} Project
   *  @property {(string|null)=} id
   *  @property {*=} name
   *  @property {*=} goal
   *  @property {*=} industry
   *  @property {*=} data
   *  @property {*=} solution
   *  @property {*=} settings
   *  @property {*=} stages
   *  @property {*=} records
   *  @property {*=} activity
   *  @property {*=} transitions
   *  @property {*=} graph
   *  @property {*=} memory
   */
  /** @typedef {Object} ProjectCreate
   *  @property {*=} name
   *  @property {*=} industry
   *  @property {*=} goal
   */
  /** @typedef {Object} ProjectPatch
   *  @property {*=} name
   *  @property {*=} goal
   *  @property {*=} industry
   *  @property {*=} settings
   *  @property {*=} policy
   */
  /** @typedef {Object} Proposal
   *  @property {*=} target
   *  @property {*=} task
   *  @property {*=} forbidden
   *  @property {*=} identifiers
   */
  /** @typedef {Object} ProposalRequest
   *  @property {*=} target
   *  @property {*=} task
   */
  /** @typedef {Object} Question
   *  @property {*=} question
   */
  /** @typedef {Object} Review
   *  @property {*=} summary
   *  @property {*=} cells
   *  @property {*=} findings
   *  @property {*=} fixes_available
   */
  /** @typedef {Object} RoutingChange
   *  @property {*=} purpose
   *  @property {*=} setting
   *  @property {*=} tier
   *  @property {*=} reason
   */
  /** @typedef {Object} Run
   *  @property {(string|null)=} id
   *  @property {*=} status
   *  @property {(Array<*>|null)=} events
   */
  /** @typedef {Object} RunRequest
   *  @property {string=} project
   *  @property {string=} goal
   *  @property {Array<string>=} datasets
   *  @property {string=} model
   *  @property {number=} max_experiments
   *  @property {number=} max_rows
   *  @property {number=} repeats
   *  @property {number=} max_minutes
   */
  /** @typedef {Object} SampleChoice
   *  @property {*=} key
   */
  /** @typedef {Object} Session
   *  @property {*=} id
   *  @property {*=} status
   *  @property {*=} task
   *  @property {*=} project_id
   *  @property {*=} steps
   *  @property {*=} trace
   */
  /** @typedef {Object} SolutionBody
   *  @property {*=} target
   *  @property {*=} task
   *  @property {*=} positive_label
   *  @property {*=} prediction_moment
   *  @property {*=} forbidden
   *  @property {*=} identifiers
   *  @property {*=} time_column
   *  @property {*=} group_column
   *  @property {*=} text_columns
   *  @property {*=} metric
   *  @property {*=} notes
   */
  /** @typedef {Object} StageApproval
   *  @property {*=} choice
   */
  /** @typedef {Object} Status
   *  @property {(string|null)=} status
   */
  /** @typedef {Object} ValidationError
   *  @property {Array<(string|number)>} loc
   *  @property {string} msg
   *  @property {string} type
   *  @property {*=} input
   *  @property {Object=} ctx
   */
  /** @typedef {Object} Verdict
   *  @property {*=} move
   *  @property {*=} status
   *  @property {*=} message
   */
  /** @typedef {Object} Workspace
   *  @property {*=} projects
   *  @property {*=} needs
   *  @property {*=} stats
   */

  const ROUTES = {
    me: ["GET", "/auth/me", [], []],
    signIn: ["POST", "/auth/password", [], []],
    signOut: ["POST", "/auth/signout", [], []],
    tokens: ["GET", "/auth/tokens", [], []],
    newToken: ["POST", "/auth/tokens", [], []],
    revokeToken: ["DELETE", "/auth/tokens/{token_id}", ["token_id"], []],
    switchWorkspace: ["POST", "/auth/workspace", [], []],
    configuration: ["GET", "/config", [], []],
    connectorsStatus: ["GET", "/connectors", [], []],
    kaggleSearch: ["POST", "/connectors/kaggle/search", [], []],
    listDrafts: ["GET", "/drafts", [], []],
    createDraft: ["POST", "/drafts", [], []],
    deleteDraft: ["DELETE", "/drafts/{draft_id}", ["draft_id"], []],
    readDraft: ["GET", "/drafts/{draft_id}", ["draft_id"], []],
    editDraft: ["PATCH", "/drafts/{draft_id}", ["draft_id"], []],
    build: ["POST", "/drafts/{draft_id}/build", ["draft_id"], []],
    upload: ["PUT", "/drafts/{draft_id}/data", ["draft_id"], ["filename"]],
    dataCloud: ["POST", "/drafts/{draft_id}/data/cloud", ["draft_id"], []],
    dataDatabase: ["POST", "/drafts/{draft_id}/data/database", ["draft_id"], []],
    dataHf: ["POST", "/drafts/{draft_id}/data/hf", ["draft_id"], []],
    dataKaggle: ["POST", "/drafts/{draft_id}/data/kaggle", ["draft_id"], []],
    sample: ["POST", "/drafts/{draft_id}/data/sample", ["draft_id"], []],
    syntheticData: ["POST", "/drafts/{draft_id}/data/synthetic", ["draft_id"], []],
    events: ["GET", "/drafts/{draft_id}/events", ["draft_id"], ["after", "wait"]],
    message: ["POST", "/drafts/{draft_id}/messages", ["draft_id"], []],
    choosePack: ["POST", "/drafts/{draft_id}/pack", ["draft_id"], []],
    draftSettings: ["PUT", "/drafts/{draft_id}/settings", ["draft_id"], []],
    draftSolution: ["PUT", "/drafts/{draft_id}/solution", ["draft_id"], []],
    draftProposal: ["POST", "/drafts/{draft_id}/solution/proposal", ["draft_id"], []],
    evidenceLibrary: ["GET", "/evidence", [], []],
    evidenceAsk: ["POST", "/evidence/ask", [], []],
    evidenceRecord: ["GET", "/evidence/{record_id}", ["record_id"], []],
    internStatus: ["GET", "/intern", [], []],
    internList: ["GET", "/intern/sessions", [], []],
    internStart: ["POST", "/intern/sessions", [], ["wait"]],
    internDelete: ["DELETE", "/intern/sessions/{session_id}", ["session_id"], []],
    internGet: ["GET", "/intern/sessions/{session_id}", ["session_id"], []],
    internMessage: ["POST", "/intern/sessions/{session_id}/message", ["session_id"], ["wait"]],
    listJobs: ["GET", "/jobs", [], ["kind", "active", "limit"]],
    readJob: ["GET", "/jobs/{job_id}", ["job_id"], []],
    retryJob: ["POST", "/jobs/{job_id}/retry", ["job_id"], []],
    stopJob: ["POST", "/jobs/{job_id}/stop", ["job_id"], []],
    knowledge: ["GET", "/knowledge", [], []],
    labBenchmarkRoute: ["GET", "/lab/benchmark", [], []],
    labSummaryRoute: ["GET", "/lab/summary", [], []],
    learnPacks: ["GET", "/learn/packs", [], []],
    learnPolicy: ["GET", "/learn/policy", [], []],
    listLessons: ["GET", "/lessons", [], ["project_id", "status"]],
    reviewLesson: ["POST", "/lessons/{lesson_id}/review", ["lesson_id"], []],
    modelOverview: ["GET", "/models", [], []],
    changeRouting: ["POST", "/models/routing", [], []],
    opsEvidenceRecent: ["GET", "/ops/evidence-recent", [], ["limit"]],
    opsJobs: ["GET", "/ops/jobs", [], ["limit"]],
    opsJob: ["GET", "/ops/jobs/{job_id}", ["job_id"], []],
    listPacks: ["GET", "/packs", [], []],
    platformAudit: ["GET", "/platform/audit", [], ["limit", "offset", "kind", "status", "actor", "project"]],
    platformIntegrations: ["GET", "/platform/integrations", [], []],
    platformLimits: ["GET", "/platform/limits", [], []],
    platformPolicies: ["GET", "/platform/policies", [], []],
    platformQuotas: ["GET", "/platform/quotas", [], []],
    listProjects: ["GET", "/projects", [], []],
    createProject: ["POST", "/projects", [], []],
    deleteProject: ["DELETE", "/projects/{project_id}", ["project_id"], []],
    getProject: ["GET", "/projects/{project_id}", ["project_id"], []],
    patchProject: ["PATCH", "/projects/{project_id}", ["project_id"], []],
    approveGate: ["POST", "/projects/{project_id}/approvals", ["project_id"], []],
    askAgent: ["POST", "/projects/{project_id}/ask", ["project_id"], []],
    uploadData: ["PUT", "/projects/{project_id}/data", ["project_id"], ["filename"]],
    useSample: ["POST", "/projects/{project_id}/data/sample", ["project_id"], []],
    exportNotebook: ["GET", "/projects/{project_id}/export/notebook", ["project_id"], []],
    exportReport: ["GET", "/projects/{project_id}/export/report", ["project_id"], []],
    exportSft: ["GET", "/projects/{project_id}/export/sft", ["project_id"], []],
    projectGraph: ["GET", "/projects/{project_id}/graph", ["project_id"], []],
    checkMove: ["POST", "/projects/{project_id}/graph/check", ["project_id"], []],
    projectLessons: ["POST", "/projects/{project_id}/lessons", ["project_id"], []],
    removeMemoryNote: ["DELETE", "/projects/{project_id}/memory/{note_id}", ["project_id", "note_id"], []],
    reviewProjectNotebook: ["GET", "/projects/{project_id}/review", ["project_id"], []],
    reviewProjectFixes: ["POST", "/projects/{project_id}/review/fixes", ["project_id"], []],
    runAll: ["POST", "/projects/{project_id}/run", ["project_id"], ["start", "wait"]],
    saveSolution: ["PUT", "/projects/{project_id}/solution", ["project_id"], []],
    solutionProposal: ["POST", "/projects/{project_id}/solution/proposal", ["project_id"], []],
    approveStage: ["POST", "/projects/{project_id}/stages/{stage}/approve", ["project_id", "stage"], []],
    runStage: ["POST", "/projects/{project_id}/stages/{stage}/run", ["project_id", "stage"], ["wait", "reuse_reason"]],
    research: ["GET", "/research", [], []],
    researchFile: ["GET", "/research/file", [], ["path"]],
    runs: ["GET", "/runs", [], []],
    start: ["POST", "/runs", [], []],
    detail: ["GET", "/runs/{run_id}", ["run_id"], []],
    export: ["GET", "/runs/{run_id}/export", ["run_id"], []],
    pause: ["POST", "/runs/{run_id}/pause", ["run_id"], []],
    resume: ["POST", "/runs/{run_id}/resume", ["run_id"], []],
    artifact: ["GET", "/runs/{run_id}/trials/{trial_id}/{filename}", ["run_id", "trial_id", "filename"], []],
    samples: ["GET", "/samples", [], []],
    studioStatus: ["GET", "/studio", [], []],
    syntheticTemplates: ["GET", "/synthetic/templates", [], []],
    workspace: ["GET", "/workspace", [], []],
    members: ["GET", "/workspace/members", [], []],
    addMember: ["POST", "/workspace/members", [], []],
    removeMember: ["DELETE", "/workspace/members/{user_id}", ["user_id"], []],
    changeMember: ["PATCH", "/workspace/members/{user_id}", ["user_id"], []],
  };

  function build(name, args) {
    const [, template, params, query] = ROUTES[name];
    let i = 0;
    const path = template.replace(/\{(\w+)\}/g, (_, key) => {
      const value = args[i++];
      if (value === undefined || value === null || value === '') throw new Error(`DC.client.${name} needs ${key}`);
      return encodeURIComponent(value);
    });
    const opts = args[params.length] || {};
    const pairs = [];
    Object.entries(opts.query || {}).forEach(([key, value]) => {
      if (!query.includes(key)) throw new Error(`DC.client.${name} has no query parameter ${key}`);
      if (value !== undefined && value !== null && value !== '') pairs.push(encodeURIComponent(key) + '=' + encodeURIComponent(value));
    });
    return { path: pairs.length ? path + '?' + pairs.join('&') : path, opts };
  }
  function call(name, args) {
    const { path, opts } = build(name, args);
    const { query, ...rest } = opts;
    return window.DC.api(path, Object.assign({}, rest, { method: ROUTES[name][0] }));
  }

  const client = { path: {}, href: {} };
  /** GET /api/auth/me: Me
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.me = (...args) => call('me', args);
  client.path.me = (...args) => build('me', args).path;
  client.href.me = (...args) => '/api' + build('me', args).path;
  /** POST /api/auth/password: Sign In
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.signIn = (...args) => call('signIn', args);
  client.path.signIn = (...args) => build('signIn', args).path;
  /** POST /api/auth/signout: Sign Out
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.signOut = (...args) => call('signOut', args);
  client.path.signOut = (...args) => build('signOut', args).path;
  /** GET /api/auth/tokens: Tokens
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.tokens = (...args) => call('tokens', args);
  client.path.tokens = (...args) => build('tokens', args).path;
  client.href.tokens = (...args) => '/api' + build('tokens', args).path;
  /** POST /api/auth/tokens: New Token
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.newToken = (...args) => call('newToken', args);
  client.path.newToken = (...args) => build('newToken', args).path;
  /** DELETE /api/auth/tokens/{token_id}: Revoke Token
   *  @param {string} token_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<null>} */
  client.revokeToken = (...args) => call('revokeToken', args);
  client.path.revokeToken = (...args) => build('revokeToken', args).path;
  /** POST /api/auth/workspace: Switch Workspace
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.switchWorkspace = (...args) => call('switchWorkspace', args);
  client.path.switchWorkspace = (...args) => build('switchWorkspace', args).path;
  /** GET /api/config: Configuration
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Config>} */
  client.configuration = (...args) => call('configuration', args);
  client.path.configuration = (...args) => build('configuration', args).path;
  client.href.configuration = (...args) => '/api' + build('configuration', args).path;
  /** GET /api/connectors: Connectors Status
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.connectorsStatus = (...args) => call('connectorsStatus', args);
  client.path.connectorsStatus = (...args) => build('connectorsStatus', args).path;
  client.href.connectorsStatus = (...args) => '/api' + build('connectorsStatus', args).path;
  /** POST /api/connectors/kaggle/search: Kaggle Search
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.kaggleSearch = (...args) => call('kaggleSearch', args);
  client.path.kaggleSearch = (...args) => build('kaggleSearch', args).path;
  /** GET /api/drafts: List Drafts
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.listDrafts = (...args) => call('listDrafts', args);
  client.path.listDrafts = (...args) => build('listDrafts', args).path;
  client.href.listDrafts = (...args) => '/api' + build('listDrafts', args).path;
  /** POST /api/drafts: Create Draft
   *  @param {{body: DraftCreate, headers?: Object}} [opts]
   *  @returns {Promise<Draft>} */
  client.createDraft = (...args) => call('createDraft', args);
  client.path.createDraft = (...args) => build('createDraft', args).path;
  /** DELETE /api/drafts/{draft_id}: Delete Draft
   *  @param {string} draft_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<null>} */
  client.deleteDraft = (...args) => call('deleteDraft', args);
  client.path.deleteDraft = (...args) => build('deleteDraft', args).path;
  /** GET /api/drafts/{draft_id}: Read Draft
   *  @param {string} draft_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Draft>} */
  client.readDraft = (...args) => call('readDraft', args);
  client.path.readDraft = (...args) => build('readDraft', args).path;
  client.href.readDraft = (...args) => '/api' + build('readDraft', args).path;
  /** PATCH /api/drafts/{draft_id}: Edit Draft
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Draft>} */
  client.editDraft = (...args) => call('editDraft', args);
  client.path.editDraft = (...args) => build('editDraft', args).path;
  /** POST /api/drafts/{draft_id}/build: Build
   *  @param {string} draft_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.build = (...args) => call('build', args);
  client.path.build = (...args) => build('build', args).path;
  /** PUT /api/drafts/{draft_id}/data: Upload
   *  @param {string} draft_id
   *  @param {{query?: {filename?: *}, body: Blob, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.upload = (...args) => call('upload', args);
  client.path.upload = (...args) => build('upload', args).path;
  /** POST /api/drafts/{draft_id}/data/cloud: Data Cloud
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.dataCloud = (...args) => call('dataCloud', args);
  client.path.dataCloud = (...args) => build('dataCloud', args).path;
  /** POST /api/drafts/{draft_id}/data/database: Data Database
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.dataDatabase = (...args) => call('dataDatabase', args);
  client.path.dataDatabase = (...args) => build('dataDatabase', args).path;
  /** POST /api/drafts/{draft_id}/data/hf: Data Hf
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.dataHf = (...args) => call('dataHf', args);
  client.path.dataHf = (...args) => build('dataHf', args).path;
  /** POST /api/drafts/{draft_id}/data/kaggle: Data Kaggle
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.dataKaggle = (...args) => call('dataKaggle', args);
  client.path.dataKaggle = (...args) => build('dataKaggle', args).path;
  /** POST /api/drafts/{draft_id}/data/sample: Sample
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.sample = (...args) => call('sample', args);
  client.path.sample = (...args) => build('sample', args).path;
  /** POST /api/drafts/{draft_id}/data/synthetic: Synthetic Data
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.syntheticData = (...args) => call('syntheticData', args);
  client.path.syntheticData = (...args) => build('syntheticData', args).path;
  /** GET /api/drafts/{draft_id}/events: Events
   *  @param {string} draft_id
   *  @param {{query?: {after?: *, wait?: *}, headers?: Object}} [opts]
   *  @returns {Promise<*>} */
  client.events = (...args) => call('events', args);
  client.path.events = (...args) => build('events', args).path;
  client.href.events = (...args) => '/api' + build('events', args).path;
  /** POST /api/drafts/{draft_id}/messages: Message
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.message = (...args) => call('message', args);
  client.path.message = (...args) => build('message', args).path;
  /** POST /api/drafts/{draft_id}/pack: Choose Pack
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Draft>} */
  client.choosePack = (...args) => call('choosePack', args);
  client.path.choosePack = (...args) => build('choosePack', args).path;
  /** PUT /api/drafts/{draft_id}/settings: Draft Settings
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Draft>} */
  client.draftSettings = (...args) => call('draftSettings', args);
  client.path.draftSettings = (...args) => build('draftSettings', args).path;
  /** PUT /api/drafts/{draft_id}/solution: Draft Solution
   *  @param {string} draft_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Draft>} */
  client.draftSolution = (...args) => call('draftSolution', args);
  client.path.draftSolution = (...args) => build('draftSolution', args).path;
  /** POST /api/drafts/{draft_id}/solution/proposal: Draft Proposal
   *  @param {string} draft_id
   *  @param {{body?: (AnyObject|null), headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.draftProposal = (...args) => call('draftProposal', args);
  client.path.draftProposal = (...args) => build('draftProposal', args).path;
  /** GET /api/evidence: Evidence Library
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.evidenceLibrary = (...args) => call('evidenceLibrary', args);
  client.path.evidenceLibrary = (...args) => build('evidenceLibrary', args).path;
  client.href.evidenceLibrary = (...args) => '/api' + build('evidenceLibrary', args).path;
  /** POST /api/evidence/ask: Evidence Ask
   *  @param {{body: Question, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.evidenceAsk = (...args) => call('evidenceAsk', args);
  client.path.evidenceAsk = (...args) => build('evidenceAsk', args).path;
  /** GET /api/evidence/{record_id}: Evidence Record
   *  @param {string} record_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.evidenceRecord = (...args) => call('evidenceRecord', args);
  client.path.evidenceRecord = (...args) => build('evidenceRecord', args).path;
  client.href.evidenceRecord = (...args) => '/api' + build('evidenceRecord', args).path;
  /** GET /api/intern: Intern Status
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<InternStatus>} */
  client.internStatus = (...args) => call('internStatus', args);
  client.path.internStatus = (...args) => build('internStatus', args).path;
  client.href.internStatus = (...args) => '/api' + build('internStatus', args).path;
  /** GET /api/intern/sessions: Intern List
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Session>>} */
  client.internList = (...args) => call('internList', args);
  client.path.internList = (...args) => build('internList', args).path;
  client.href.internList = (...args) => '/api' + build('internList', args).path;
  /** POST /api/intern/sessions: Intern Start
   *  @param {{query?: {wait?: *}, body: InternStart, headers?: Object}} [opts]
   *  @returns {Promise<Session>} */
  client.internStart = (...args) => call('internStart', args);
  client.path.internStart = (...args) => build('internStart', args).path;
  /** DELETE /api/intern/sessions/{session_id}: Intern Delete
   *  @param {string} session_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<null>} */
  client.internDelete = (...args) => call('internDelete', args);
  client.path.internDelete = (...args) => build('internDelete', args).path;
  /** GET /api/intern/sessions/{session_id}: Intern Get
   *  @param {string} session_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Session>} */
  client.internGet = (...args) => call('internGet', args);
  client.path.internGet = (...args) => build('internGet', args).path;
  client.href.internGet = (...args) => '/api' + build('internGet', args).path;
  /** POST /api/intern/sessions/{session_id}/message: Intern Message
   *  @param {string} session_id
   *  @param {{query?: {wait?: *}, body: InternMessage, headers?: Object}} [opts]
   *  @returns {Promise<Session>} */
  client.internMessage = (...args) => call('internMessage', args);
  client.path.internMessage = (...args) => build('internMessage', args).path;
  /** GET /api/jobs: List Jobs
   *  @param {{query?: {kind?: *, active?: *, limit?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.listJobs = (...args) => call('listJobs', args);
  client.path.listJobs = (...args) => build('listJobs', args).path;
  client.href.listJobs = (...args) => '/api' + build('listJobs', args).path;
  /** GET /api/jobs/{job_id}: Read Job
   *  @param {string} job_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.readJob = (...args) => call('readJob', args);
  client.path.readJob = (...args) => build('readJob', args).path;
  client.href.readJob = (...args) => '/api' + build('readJob', args).path;
  /** POST /api/jobs/{job_id}/retry: Retry Job
   *  @param {string} job_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.retryJob = (...args) => call('retryJob', args);
  client.path.retryJob = (...args) => build('retryJob', args).path;
  /** POST /api/jobs/{job_id}/stop: Stop Job
   *  @param {string} job_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.stopJob = (...args) => call('stopJob', args);
  client.path.stopJob = (...args) => build('stopJob', args).path;
  /** GET /api/knowledge: Knowledge
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.knowledge = (...args) => call('knowledge', args);
  client.path.knowledge = (...args) => build('knowledge', args).path;
  client.href.knowledge = (...args) => '/api' + build('knowledge', args).path;
  /** GET /api/lab/benchmark: Lab Benchmark Route
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.labBenchmarkRoute = (...args) => call('labBenchmarkRoute', args);
  client.path.labBenchmarkRoute = (...args) => build('labBenchmarkRoute', args).path;
  client.href.labBenchmarkRoute = (...args) => '/api' + build('labBenchmarkRoute', args).path;
  /** GET /api/lab/summary: Lab Summary Route
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.labSummaryRoute = (...args) => call('labSummaryRoute', args);
  client.path.labSummaryRoute = (...args) => build('labSummaryRoute', args).path;
  client.href.labSummaryRoute = (...args) => '/api' + build('labSummaryRoute', args).path;
  /** GET /api/learn/packs: Learn Packs
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.learnPacks = (...args) => call('learnPacks', args);
  client.path.learnPacks = (...args) => build('learnPacks', args).path;
  client.href.learnPacks = (...args) => '/api' + build('learnPacks', args).path;
  /** GET /api/learn/policy: Learn Policy
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.learnPolicy = (...args) => call('learnPolicy', args);
  client.path.learnPolicy = (...args) => build('learnPolicy', args).path;
  client.href.learnPolicy = (...args) => '/api' + build('learnPolicy', args).path;
  /** GET /api/lessons: List Lessons
   *  @param {{query?: {project_id?: *, status?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Lessons>} */
  client.listLessons = (...args) => call('listLessons', args);
  client.path.listLessons = (...args) => build('listLessons', args).path;
  client.href.listLessons = (...args) => '/api' + build('listLessons', args).path;
  /** POST /api/lessons/{lesson_id}/review: Review Lesson
   *  @param {string} lesson_id
   *  @param {{body: LessonReview, headers?: Object}} [opts]
   *  @returns {Promise<Lesson>} */
  client.reviewLesson = (...args) => call('reviewLesson', args);
  client.path.reviewLesson = (...args) => build('reviewLesson', args).path;
  /** GET /api/models: Model Overview
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<ModelsSummary>} */
  client.modelOverview = (...args) => call('modelOverview', args);
  client.path.modelOverview = (...args) => build('modelOverview', args).path;
  client.href.modelOverview = (...args) => '/api' + build('modelOverview', args).path;
  /** POST /api/models/routing: Change Routing
   *  @param {{body: RoutingChange, headers?: Object}} [opts]
   *  @returns {Promise<ModelsSummary>} */
  client.changeRouting = (...args) => call('changeRouting', args);
  client.path.changeRouting = (...args) => build('changeRouting', args).path;
  /** GET /api/ops/evidence-recent: Ops Evidence Recent
   *  @param {{query?: {limit?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.opsEvidenceRecent = (...args) => call('opsEvidenceRecent', args);
  client.path.opsEvidenceRecent = (...args) => build('opsEvidenceRecent', args).path;
  client.href.opsEvidenceRecent = (...args) => '/api' + build('opsEvidenceRecent', args).path;
  /** GET /api/ops/jobs: Ops Jobs
   *  @param {{query?: {limit?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.opsJobs = (...args) => call('opsJobs', args);
  client.path.opsJobs = (...args) => build('opsJobs', args).path;
  client.href.opsJobs = (...args) => '/api' + build('opsJobs', args).path;
  /** GET /api/ops/jobs/{job_id}: Ops Job
   *  @param {string} job_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.opsJob = (...args) => call('opsJob', args);
  client.path.opsJob = (...args) => build('opsJob', args).path;
  client.href.opsJob = (...args) => '/api' + build('opsJob', args).path;
  /** GET /api/packs: List Packs
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.listPacks = (...args) => call('listPacks', args);
  client.path.listPacks = (...args) => build('listPacks', args).path;
  client.href.listPacks = (...args) => '/api' + build('listPacks', args).path;
  /** GET /api/platform/audit: Platform Audit
   *  @param {{query?: {limit?: *, offset?: *, kind?: *, status?: *, actor?: *, project?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.platformAudit = (...args) => call('platformAudit', args);
  client.path.platformAudit = (...args) => build('platformAudit', args).path;
  client.href.platformAudit = (...args) => '/api' + build('platformAudit', args).path;
  /** GET /api/platform/integrations: Platform Integrations
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.platformIntegrations = (...args) => call('platformIntegrations', args);
  client.path.platformIntegrations = (...args) => build('platformIntegrations', args).path;
  client.href.platformIntegrations = (...args) => '/api' + build('platformIntegrations', args).path;
  /** GET /api/platform/limits: Platform Limits
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.platformLimits = (...args) => call('platformLimits', args);
  client.path.platformLimits = (...args) => build('platformLimits', args).path;
  client.href.platformLimits = (...args) => '/api' + build('platformLimits', args).path;
  /** GET /api/platform/policies: Platform Policies
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.platformPolicies = (...args) => call('platformPolicies', args);
  client.path.platformPolicies = (...args) => build('platformPolicies', args).path;
  client.href.platformPolicies = (...args) => '/api' + build('platformPolicies', args).path;
  /** GET /api/platform/quotas: Platform Quotas
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.platformQuotas = (...args) => call('platformQuotas', args);
  client.path.platformQuotas = (...args) => build('platformQuotas', args).path;
  client.href.platformQuotas = (...args) => '/api' + build('platformQuotas', args).path;
  /** GET /api/projects: List Projects
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Project>>} */
  client.listProjects = (...args) => call('listProjects', args);
  client.path.listProjects = (...args) => build('listProjects', args).path;
  client.href.listProjects = (...args) => '/api' + build('listProjects', args).path;
  /** POST /api/projects: Create Project
   *  @param {{body: ProjectCreate, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.createProject = (...args) => call('createProject', args);
  client.path.createProject = (...args) => build('createProject', args).path;
  /** DELETE /api/projects/{project_id}: Delete Project
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<null>} */
  client.deleteProject = (...args) => call('deleteProject', args);
  client.path.deleteProject = (...args) => build('deleteProject', args).path;
  /** GET /api/projects/{project_id}: Get Project
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.getProject = (...args) => call('getProject', args);
  client.path.getProject = (...args) => build('getProject', args).path;
  client.href.getProject = (...args) => '/api' + build('getProject', args).path;
  /** PATCH /api/projects/{project_id}: Patch Project
   *  @param {string} project_id
   *  @param {{body: ProjectPatch, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.patchProject = (...args) => call('patchProject', args);
  client.path.patchProject = (...args) => build('patchProject', args).path;
  /** POST /api/projects/{project_id}/approvals: Approve Gate
   *  @param {string} project_id
   *  @param {{body: GateApproval, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.approveGate = (...args) => call('approveGate', args);
  client.path.approveGate = (...args) => build('approveGate', args).path;
  /** POST /api/projects/{project_id}/ask: Ask Agent
   *  @param {string} project_id
   *  @param {{body: Question, headers?: Object}} [opts]
   *  @returns {Promise<Answer>} */
  client.askAgent = (...args) => call('askAgent', args);
  client.path.askAgent = (...args) => build('askAgent', args).path;
  /** PUT /api/projects/{project_id}/data: Upload Data
   *  @param {string} project_id
   *  @param {{query?: {filename?: *}, body: Blob, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.uploadData = (...args) => call('uploadData', args);
  client.path.uploadData = (...args) => build('uploadData', args).path;
  /** POST /api/projects/{project_id}/data/sample: Use Sample
   *  @param {string} project_id
   *  @param {{body: SampleChoice, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.useSample = (...args) => call('useSample', args);
  client.path.useSample = (...args) => build('useSample', args).path;
  /** GET /api/projects/{project_id}/export/notebook: Export Notebook
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<*>} */
  client.exportNotebook = (...args) => call('exportNotebook', args);
  client.path.exportNotebook = (...args) => build('exportNotebook', args).path;
  client.href.exportNotebook = (...args) => '/api' + build('exportNotebook', args).path;
  /** GET /api/projects/{project_id}/export/report: Export Report
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<*>} */
  client.exportReport = (...args) => call('exportReport', args);
  client.path.exportReport = (...args) => build('exportReport', args).path;
  client.href.exportReport = (...args) => '/api' + build('exportReport', args).path;
  /** GET /api/projects/{project_id}/export/sft: Export Sft
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<*>} */
  client.exportSft = (...args) => call('exportSft', args);
  client.path.exportSft = (...args) => build('exportSft', args).path;
  client.href.exportSft = (...args) => '/api' + build('exportSft', args).path;
  /** GET /api/projects/{project_id}/graph: Project Graph
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Graph>} */
  client.projectGraph = (...args) => call('projectGraph', args);
  client.path.projectGraph = (...args) => build('projectGraph', args).path;
  client.href.projectGraph = (...args) => '/api' + build('projectGraph', args).path;
  /** POST /api/projects/{project_id}/graph/check: Check Move
   *  @param {string} project_id
   *  @param {{body: MoveCheck, headers?: Object}} [opts]
   *  @returns {Promise<Verdict>} */
  client.checkMove = (...args) => call('checkMove', args);
  client.path.checkMove = (...args) => build('checkMove', args).path;
  /** POST /api/projects/{project_id}/lessons: Project Lessons
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Lessons>} */
  client.projectLessons = (...args) => call('projectLessons', args);
  client.path.projectLessons = (...args) => build('projectLessons', args).path;
  /** DELETE /api/projects/{project_id}/memory/{note_id}: Remove Memory Note
   *  @param {string} project_id
   *  @param {string} note_id
   *  @param {{body?: *, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.removeMemoryNote = (...args) => call('removeMemoryNote', args);
  client.path.removeMemoryNote = (...args) => build('removeMemoryNote', args).path;
  /** GET /api/projects/{project_id}/review: Review Project Notebook
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Review>} */
  client.reviewProjectNotebook = (...args) => call('reviewProjectNotebook', args);
  client.path.reviewProjectNotebook = (...args) => build('reviewProjectNotebook', args).path;
  client.href.reviewProjectNotebook = (...args) => '/api' + build('reviewProjectNotebook', args).path;
  /** POST /api/projects/{project_id}/review/fixes: Review Project Fixes
   *  @param {string} project_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Fixes>} */
  client.reviewProjectFixes = (...args) => call('reviewProjectFixes', args);
  client.path.reviewProjectFixes = (...args) => build('reviewProjectFixes', args).path;
  /** POST /api/projects/{project_id}/run: Run All
   *  @param {string} project_id
   *  @param {{query?: {start?: *, wait?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.runAll = (...args) => call('runAll', args);
  client.path.runAll = (...args) => build('runAll', args).path;
  /** PUT /api/projects/{project_id}/solution: Save Solution
   *  @param {string} project_id
   *  @param {{body: SolutionBody, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.saveSolution = (...args) => call('saveSolution', args);
  client.path.saveSolution = (...args) => build('saveSolution', args).path;
  /** POST /api/projects/{project_id}/solution/proposal: Solution Proposal
   *  @param {string} project_id
   *  @param {{body: ProposalRequest, headers?: Object}} [opts]
   *  @returns {Promise<Proposal>} */
  client.solutionProposal = (...args) => call('solutionProposal', args);
  client.path.solutionProposal = (...args) => build('solutionProposal', args).path;
  /** POST /api/projects/{project_id}/stages/{stage}/approve: Approve Stage
   *  @param {string} project_id
   *  @param {string} stage
   *  @param {{body?: (StageApproval|null), headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.approveStage = (...args) => call('approveStage', args);
  client.path.approveStage = (...args) => build('approveStage', args).path;
  /** POST /api/projects/{project_id}/stages/{stage}/run: Run Stage
   *  @param {string} project_id
   *  @param {string} stage
   *  @param {{query?: {wait?: *, reuse_reason?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Project>} */
  client.runStage = (...args) => call('runStage', args);
  client.path.runStage = (...args) => build('runStage', args).path;
  /** GET /api/research: Research
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.research = (...args) => call('research', args);
  client.path.research = (...args) => build('research', args).path;
  client.href.research = (...args) => '/api' + build('research', args).path;
  /** GET /api/research/file: Research File
   *  @param {{query?: {path?: *}, headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.researchFile = (...args) => call('researchFile', args);
  client.path.researchFile = (...args) => build('researchFile', args).path;
  client.href.researchFile = (...args) => '/api' + build('researchFile', args).path;
  /** GET /api/runs: Runs
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Run>>} */
  client.runs = (...args) => call('runs', args);
  client.path.runs = (...args) => build('runs', args).path;
  client.href.runs = (...args) => '/api' + build('runs', args).path;
  /** POST /api/runs: Start
   *  @param {{body: RunRequest, headers?: Object}} [opts]
   *  @returns {Promise<Run>} */
  client.start = (...args) => call('start', args);
  client.path.start = (...args) => build('start', args).path;
  /** GET /api/runs/{run_id}: Detail
   *  @param {string} run_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Run>} */
  client.detail = (...args) => call('detail', args);
  client.path.detail = (...args) => build('detail', args).path;
  client.href.detail = (...args) => '/api' + build('detail', args).path;
  /** GET /api/runs/{run_id}/export: Export
   *  @param {string} run_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<*>} */
  client.export = (...args) => call('export', args);
  client.path.export = (...args) => build('export', args).path;
  client.href.export = (...args) => '/api' + build('export', args).path;
  /** POST /api/runs/{run_id}/pause: Pause
   *  @param {string} run_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Status>} */
  client.pause = (...args) => call('pause', args);
  client.path.pause = (...args) => build('pause', args).path;
  /** POST /api/runs/{run_id}/resume: Resume
   *  @param {string} run_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Status>} */
  client.resume = (...args) => call('resume', args);
  client.path.resume = (...args) => build('resume', args).path;
  /** GET /api/runs/{run_id}/trials/{trial_id}/{filename}: Artifact
   *  @param {string} run_id
   *  @param {string} trial_id
   *  @param {string} filename
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<*>} */
  client.artifact = (...args) => call('artifact', args);
  client.path.artifact = (...args) => build('artifact', args).path;
  client.href.artifact = (...args) => '/api' + build('artifact', args).path;
  /** GET /api/samples: Samples
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.samples = (...args) => call('samples', args);
  client.path.samples = (...args) => build('samples', args).path;
  client.href.samples = (...args) => '/api' + build('samples', args).path;
  /** GET /api/studio: Studio Status
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.studioStatus = (...args) => call('studioStatus', args);
  client.path.studioStatus = (...args) => build('studioStatus', args).path;
  client.href.studioStatus = (...args) => '/api' + build('studioStatus', args).path;
  /** GET /api/synthetic/templates: Synthetic Templates
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Page>} */
  client.syntheticTemplates = (...args) => call('syntheticTemplates', args);
  client.path.syntheticTemplates = (...args) => build('syntheticTemplates', args).path;
  client.href.syntheticTemplates = (...args) => '/api' + build('syntheticTemplates', args).path;
  /** GET /api/workspace: Workspace
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Workspace>} */
  client.workspace = (...args) => call('workspace', args);
  client.path.workspace = (...args) => build('workspace', args).path;
  client.href.workspace = (...args) => '/api' + build('workspace', args).path;
  /** GET /api/workspace/members: Members
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<Array<Doc>>} */
  client.members = (...args) => call('members', args);
  client.path.members = (...args) => build('members', args).path;
  client.href.members = (...args) => '/api' + build('members', args).path;
  /** POST /api/workspace/members: Add Member
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.addMember = (...args) => call('addMember', args);
  client.path.addMember = (...args) => build('addMember', args).path;
  /** DELETE /api/workspace/members/{user_id}: Remove Member
   *  @param {string} user_id
   *  @param {{headers?: Object}} [opts]
   *  @returns {Promise<null>} */
  client.removeMember = (...args) => call('removeMember', args);
  client.path.removeMember = (...args) => build('removeMember', args).path;
  /** PATCH /api/workspace/members/{user_id}: Change Member
   *  @param {string} user_id
   *  @param {{body: AnyObject, headers?: Object}} [opts]
   *  @returns {Promise<Doc>} */
  client.changeMember = (...args) => call('changeMember', args);
  client.path.changeMember = (...args) => build('changeMember', args).path;

  window.DC.client = client;
})();
