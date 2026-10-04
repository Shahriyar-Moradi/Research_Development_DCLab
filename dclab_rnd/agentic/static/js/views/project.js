/* One project: the agentic notebook. Cells are stages, not code.
   Left: the workflow rail. Middle: one cell per step. Right: the agent (notes with proof, questions, activity). */
import {$, esc, state, api, notice} from '../core.js';
import {recordChip} from '../drawer.js';
import {refreshProjects} from './projects.js';

const STAGES = ['data', 'leakage', 'features', 'models', 'final'];
const STEP_TITLES = {upload: 'Data', contract: 'Prediction contract', data: 'Understand the data', leakage: 'Audit leakage',
  features: 'Climb the feature ladder', models: 'Screen algorithms', final: 'Tune and confirm on the holdout', report: 'Report and export'};
const STATUS_LABEL = {pending: 'not run', queued: 'queued…', running: 'running…', completed: 'done · awaiting your review', approved: 'approved', failed: 'failed'};
const METRIC_LABEL = {roc_auc: 'ROC-AUC', average_precision: 'average precision', macro_f1: 'macro-F1', mae: 'MAE'};
const TASK_LABEL = {binary: 'binary classification', multiclass: 'multiclass classification', regression: 'regression'};
let current = null, samples = null, timer = null, questions = [], agentTab = 'notes', proposalBusy = false;

const num = (v, d = 4) => (typeof v === 'number' && Number.isFinite(v)) ? v.toFixed(d) : '—';
const pct = v => (typeof v === 'number') ? (v * 100).toFixed(1) + '%' : '—';
const ms = (row, metric) => row?.metrics?.[metric] ? `${num(row.metrics[metric].mean)} ± ${num(row.metrics[metric].std)}` : '—';
const label = m => METRIC_LABEL[m] || m;
const proofChips = ids => (ids || []).map(recordChip).join('');
const stageStatus = (p, s) => p.stages?.[s]?.status || 'pending';
const stageDone = (p, s) => ['completed', 'approved'].includes(stageStatus(p, s));
const prevDone = (p, s) => { const i = STAGES.indexOf(s); return i === 0 || stageDone(p, STAGES[i - 1]); };
const busy = p => !!p.running || STAGES.some(s => ['running', 'queued'].includes(stageStatus(p, s)));
const sev = s => s === 'high' ? 'high' : s === 'warning' ? 'warning' : 'info';
const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;

async function call(path, options) { try { return await api(path, options); } catch (error) { notice(error.message); throw error; } }
async function reload(full = false) {
  current = await api('/projects/' + state.project);
  render(full);
  return current;
}
async function act(path, options = {method: 'POST'}) {
  const before = busy(current);
  try { current = await api(path, options); notice(''); } catch (error) { notice(error.message); return; }
  render(!before && !busy(current));
  schedule(true);
}
function schedule(force = false) {
  clearTimeout(timer);
  if (current && (busy(current) || force)) timer = setTimeout(async () => { try { await reload(false); } catch {} schedule(); }, force ? 1200 : 2500);
}
export function leaveProject() { clearTimeout(timer); current = null; }

/* ---------------------------------------------------------------- render */
export async function renderProject(id) {
  state.project = id;
  questions = [];
  $('project-view').innerHTML = '<p class="muted small">Loading project…</p>';
  try { await reload(true); } catch (error) { $('project-view').innerHTML = `<div class="card empty-card"><h2>Project not found.</h2><p>${esc(error.message)}</p><a class="secondary" href="#projects">Back to projects</a></div>`; return; }
  schedule();
}

function render(full) {
  const p = current;
  $('page-label').textContent = 'Projects / ' + p.name;
  if (full || !$('ws-cells')) {
    $('project-view').innerHTML = `
      <div class="ws-head"><div><div class="eyebrow">${esc(p.industry.toUpperCase())} · ${p.contract ? esc(TASK_LABEL[p.contract.task]) : 'NEW PROJECT'}</div>
        <h1 class="ws-title">${esc(p.name)}</h1><p class="ws-goal">${esc(p.goal || 'No goal written yet. Edit it below.')}</p></div>
        <div class="actions ws-actions"><button class="secondary" id="run-all" ${(!p.contract || busy(p)) ? 'disabled' : ''}>▶ Run every stage</button>
          <a class="secondary ${p.contract ? '' : 'disabled'}" href="/api/projects/${esc(p.id)}/export/notebook" ${p.contract ? '' : 'tabindex="-1"'}>Export notebook ↗</a>
          <a class="secondary" href="/api/projects/${esc(p.id)}/export/report">Export report ↗</a>
          <a class="secondary" href="#intern/new/${esc(p.id)}">Hand to the intern ↗</a>
          <button class="secondary danger" id="delete-project">Delete</button></div></div>
      <div class="ws-layout"><aside class="card ws-rail" id="ws-rail"></aside><div class="ws-cells" id="ws-cells"></div><aside class="card ws-agent" id="ws-agent"></aside></div>`;
    $('run-all').addEventListener('click', () => act(`/projects/${p.id}/run`));
    $('delete-project').addEventListener('click', async () => {
      if (!confirm(`Delete "${p.name}" and its results? This cannot be undone.`)) return;
      await call(`/projects/${p.id}`, {method: 'DELETE'}); await refreshProjects(); location.hash = '#projects';
    });
    $('ws-cells').innerHTML = `<section class="cell" id="cell-upload"></section><section class="cell" id="cell-contract"></section>`
      + STAGES.map(s => `<section class="cell" id="cell-${s}"></section>`).join('') + `<section class="cell" id="cell-report"></section>`;
    renderUpload(); renderContract();
  }
  renderRail(); STAGES.forEach(renderStage); renderReport(); renderAgent();
  if ($('run-all')) $('run-all').disabled = !p.contract || busy(p);
  refreshProjects();
}

/* ---------------------------------------------------------------- rail */
function renderRail() {
  const p = current;
  const steps = [['upload', p.data ? 'approved' : 'pending', p.data ? `${Number(p.data.rows).toLocaleString()} rows` : 'upload a table'],
    ['contract', p.contract ? 'approved' : p.data ? 'pending' : 'locked', p.contract ? `target ${p.contract.target}` : 'target and prediction moment'],
    ...STAGES.map(s => [s, p.contract ? stageStatus(p, s) : 'locked', p.stages?.[s]?.elapsed_seconds ? `${p.stages[s].elapsed_seconds.toFixed(1)} s` : (STATUS_LABEL[stageStatus(p, s)] || '')]),
    ['report', stageDone(p, 'final') ? 'approved' : 'locked', stageDone(p, 'final') ? 'notebook and report' : 'after the final stage']];
  $('ws-rail').innerHTML = `<div class="rail-title">Workflow</div>` + steps.map(([key, status, sub], i) =>
    `<a class="rail-step ${esc(status)}" href="#cell-${key}" data-jump="${key}"><span class="rail-no">${i}</span><span><b>${esc(STEP_TITLES[key])}</b><small>${esc(sub)}</small></span><i class="rail-dot"></i></a>`).join('')
    + `<div class="rail-foot"><small>Holdout used ${p.holdout_uses || 0}×</small><small>${p.settings?.quick ? 'quick mode · 3,000 rows' : `up to ${Number(p.settings?.max_rows || 20000).toLocaleString()} rows`}</small></div>`;
  $('ws-rail').querySelectorAll('[data-jump]').forEach(a => a.addEventListener('click', e => { e.preventDefault(); $('cell-' + a.dataset.jump).scrollIntoView({behavior: 'smooth', block: 'start'}); }));
}

function cellHead(no, key, status, extra = '') {
  return `<div class="cell-head"><span class="cell-no">${no}</span><h2>${esc(STEP_TITLES[key])}</h2>${status ? `<span class="stage-chip ${esc(status)}">${esc(STATUS_LABEL[status] || status)}</span>` : ''}<div class="cell-tools">${extra}</div></div>`;
}

/* ---------------------------------------------------------------- 0 · data */
function renderUpload() {
  const p = current, el = $('cell-upload');
  const d = p.data;
  const profile = d?.profile;
  el.innerHTML = cellHead(0, 'upload', d ? 'approved' : 'pending', d ? '<button class="secondary small-btn" id="replace-data">Replace data</button>' : '') + (d ? `
    <p class="cell-lead"><code>${esc(d.filename)}</code> · ${Number(d.rows).toLocaleString()} rows × ${d.columns.length} columns · ${pct(profile.duplicate_row_rate)} duplicate rows</p>
    <div class="table-wrap cols-wrap"><table class="cols"><thead><tr><th>COLUMN</th><th>KIND</th><th>MISSING</th><th>UNIQUE</th><th>EXAMPLES</th><th>ROLE</th></tr></thead><tbody>
      ${profile.columns.map(c => `<tr><td>${esc(c.name)}</td><td>${esc(c.kind)}</td><td>${pct(c.missing_rate)}</td><td>${c.unique}</td><td class="muted examples" title="${esc(c.preview.join(', '))}">${esc(c.preview.slice(0, 3).join(', '))}</td><td>${roleChip(c.name)}</td></tr>`).join('')}
    </tbody></table></div>
    <div id="upload-zone" class="hidden"></div>` : `
    <p class="cell-lead">Bring the table you want to model: one row per thing you predict, the outcome in one column. CSV, TSV or Parquet.</p>
    <div id="upload-zone"></div>`);
  renderUploadZone();
  $('replace-data')?.addEventListener('click', () => { $('upload-zone').classList.toggle('hidden'); });
}
function roleChip(name) {
  const c = current.contract;
  if (!c) return '<span class="muted small">—</span>';
  const role = name === c.target ? 'target' : c.forbidden.some(f => f.column === name) ? 'forbidden' : c.identifiers.includes(name) ? 'identifier'
    : name === c.time_column ? 'time' : name === c.group_column ? 'group' : c.text_columns.includes(name) ? 'text' : 'input';
  return `<span class="role ${role}">${role}</span>`;
}
async function renderUploadZone() {
  const zone = $('upload-zone');
  if (!zone) return;
  if (!samples) { try { samples = await api('/samples'); } catch { samples = []; } }
  zone.innerHTML = `<div class="upload-grid">
    <label class="dropzone" id="dropzone"><input type="file" id="file-input" accept=".csv,.tsv,.parquet,.txt" class="sr-only"><b>Choose a file</b><span>or drop it here · up to 200 MB · stays on this computer</span></label>
    <div class="samples"><div class="field-label">OR START FROM A DATASET THE R&D ALREADY STUDIED</div><div class="sample-list">
      ${samples.filter(s => s.available).map(s => `<button type="button" class="sample" data-sample="${esc(s.key)}"><b>${esc(s.name)}</b><small>${esc(s.goal)}</small></button>`).join('')}</div></div></div>`;
  const upload = async file => {
    if (!file) return;
    notice(`Uploading ${file.name}…`);
    try {
      current = await api(`/projects/${current.id}/data?filename=${encodeURIComponent(file.name)}`, {method: 'PUT', body: file, headers: {'Content-Type': 'application/octet-stream'}});
      notice(''); render(true);
    } catch (error) { notice(error.message); }
  };
  $('file-input').addEventListener('change', e => upload(e.target.files[0]));
  const dz = $('dropzone');
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('over'));
  dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('over'); upload(e.dataTransfer.files[0]); });
  zone.querySelectorAll('[data-sample]').forEach(b => b.addEventListener('click', async () => {
    notice('Loading the sample…');
    try { current = await api(`/projects/${current.id}/data/sample`, {method: 'POST', body: JSON.stringify({key: b.dataset.sample})}); notice(''); render(true); }
    catch (error) { notice(error.message); }
  }));
}

/* ---------------------------------------------------------------- 1 · contract */
function renderContract() {
  const p = current, el = $('cell-contract');
  if (!p.data) { el.innerHTML = cellHead(1, 'contract', 'locked') + '<p class="cell-lead muted">Upload data first. The contract says what is predicted, when, and what cannot be known at that moment.</p>'; return; }
  const c = p.contract, prop = p.proposal, sug = p.suggestion || {};
  const columns = p.data.columns;
  const candidates = p.data.profile.target_candidates;
  const target = c?.target || prop?.target || sug.target || candidates[0];
  const task = c?.task || prop?.task || 'binary';
  const forbidden = c ? c.forbidden : (prop ? prop.forbidden.map(f => ({column: f.column, reason: f.reason})) : (sug.forbidden || []));
  const ids = c ? c.identifiers : (prop?.identifiers || sug.identifiers || []);
  const time = c ? c.time_column : (prop?.time_candidates?.[0] || sug.time_column || '');
  const group = c?.group_column || '';
  const text = c ? c.text_columns : (prop?.text_columns || []);
  const moment = c?.prediction_moment || sug.prediction_moment || '';
  const metric = c?.metric || prop?.metric || '';
  const opt = (list, chosen, allowEmpty) => (allowEmpty ? '<option value="">— none —</option>' : '') + list.map(x => `<option value="${esc(x)}" ${x === chosen ? 'selected' : ''}>${esc(x)}</option>`).join('');
  const notTarget = columns.filter(x => x !== target);
  el.innerHTML = cellHead(1, 'contract', c ? 'approved' : 'pending', c ? '<span class="muted small">Saving a change clears the stage results.</span>' : '')
    + `<p class="cell-lead">Leakage is defined by <em>when</em> you predict, not by a column's name. Write that moment down first; the agent proposes what to exclude and you decide.</p>
    <div class="contract">
      <div class="form-grid">
        <label>Target column<select id="c-target">${opt([...candidates, ...columns.filter(x => !candidates.includes(x))], target)}</select><small>${esc(prop?.detected?.note || 'The outcome you want to predict.')}</small></label>
        <label>Task<select id="c-task">${opt(['binary', 'multiclass', 'regression'], task)}</select><small>${prop && prop.detected ? `detected: ${esc(prop.detected.task)}` : 'detected from the target'}</small></label>
        <label id="c-positive-wrap" class="${task === 'binary' ? '' : 'hidden'}">Positive class<select id="c-positive">${opt(prop?.detected?.classes || (c?.positive_label ? [c.positive_label] : []), c?.positive_label || prop?.positive_label)}</select><small>the class the score ranks first</small></label>
        <label>Metric<select id="c-metric"><option value="">default for the task</option>${(prop?.metric_options || []).map(([m, d]) => `<option value="${esc(m)}" ${m === metric ? 'selected' : ''}>${esc(d)}</option>`).join('')}</select><small>${prop ? `agent suggests ${esc(label(prop.metric))}` : ''}</small></label>
      </div>
      <label class="wide">Prediction moment<textarea id="c-moment" rows="2" placeholder="e.g. When the order is created, before dispatch; final fares are only known after delivery.">${esc(moment)}</textarea><small>${esc(prop?.prediction_moment_hint || sug.prediction_moment ? (prop?.prediction_moment_hint || '') : 'What is known at the moment of prediction, and what is not.')}</small></label>
      <div class="field-label">FORBIDDEN AT PREDICTION TIME <span>${prop ? `${plural(prop.forbidden.length, 'proposal')} from the audit` : 'choose the target to get proposals'}</span></div>
      <div class="forbid-list" id="c-forbidden">${(prop?.forbidden || []).map(f => `<label class="forbid ${forbidden.some(x => x.column === f.column) ? 'on' : ''}"><input type="checkbox" value="${esc(f.column)}" data-reason="${esc(f.reason)}" ${forbidden.some(x => x.column === f.column) ? 'checked' : ''}><b>${esc(f.column)}</b><span>${esc(f.reason)}</span><i>${proofChips(f.proof)}</i></label>`).join('')}
        ${forbidden.filter(f => !(prop?.forbidden || []).some(x => x.column === f.column)).map(f => `<label class="forbid on"><input type="checkbox" value="${esc(f.column)}" data-reason="${esc(f.reason || '')}" checked><b>${esc(f.column)}</b><span>${esc(f.reason || 'declared by you')}</span></label>`).join('')}
        <div class="forbid-add"><select id="c-forbid-add"><option value="">add another column…</option>${opt(notTarget, '')}</select></div></div>
      <div class="form-grid">
        <label>Identifiers<select id="c-ids" multiple size="4">${opt(notTarget, '')}</select><small>keys, never features · ${plural(ids.length, 'selected')}</small></label>
        <label>Time column<select id="c-time">${opt(notTarget, time, true)}</select><small>orders the split: the holdout is the latest 20%</small></label>
        <label>Group column<select id="c-group">${opt(notTarget, group, true)}</select><small>rows sharing it never split across train and holdout</small></label>
        <label>Text columns<select id="c-text" multiple size="4">${opt(notTarget, '')}</select><small>free text, binary targets only · ${plural(text.length, 'selected')}</small></label>
      </div>
      <div class="contract-foot"><button class="primary" id="c-save">${c ? 'Save contract' : 'Save contract and unlock the stages'}</button><button class="secondary" id="c-propose">${prop ? 'Ask the agent again' : 'Ask the agent to propose'}</button><span class="muted small" id="c-status"></span></div>
    </div>`;
  [...$('c-ids').options].forEach(o => { o.selected = ids.includes(o.value); });
  [...$('c-text').options].forEach(o => { o.selected = text.includes(o.value); });
  $('c-task').addEventListener('change', () => $('c-positive-wrap').classList.toggle('hidden', $('c-task').value !== 'binary'));
  $('c-target').addEventListener('change', () => propose());
  $('c-propose').addEventListener('click', () => propose(true));
  $('c-forbid-add').addEventListener('change', e => {
    const name = e.target.value; if (!name) return;
    $('c-forbidden').insertAdjacentHTML('afterbegin', `<label class="forbid on"><input type="checkbox" value="${esc(name)}" data-reason="declared by you" checked><b>${esc(name)}</b><span>declared by you</span></label>`);
    e.target.value = '';
  });
  $('c-save').addEventListener('click', saveContract);
  if (!c && !prop && target) propose();
}
async function propose(force) {
  if (proposalBusy) return;
  proposalBusy = true;
  const target = $('c-target').value, task = $('c-task').value;
  $('c-status').textContent = 'The agent is auditing the columns…';
  try {
    current.proposal = await api(`/projects/${current.id}/contract/proposal`, {method: 'POST', body: JSON.stringify({target, task: force ? task : undefined})});
    if (!current.contract) renderContract();
    else $('c-status').textContent = `${plural(current.proposal.forbidden.length, 'column')} proposed as forbidden; tick them above if they are unknown at the prediction moment.`;
  } catch (error) { $('c-status').textContent = error.message; } finally { proposalBusy = false; }
}
async function saveContract() {
  const body = {
    target: $('c-target').value, task: $('c-task').value, positive_label: $('c-task').value === 'binary' ? ($('c-positive').value || null) : null,
    prediction_moment: $('c-moment').value.trim(),
    forbidden: [...$('c-forbidden').querySelectorAll('input:checked')].map(i => ({column: i.value, reason: i.dataset.reason || ''})),
    identifiers: [...$('c-ids').selectedOptions].map(o => o.value), time_column: $('c-time').value || null, group_column: $('c-group').value || null,
    text_columns: [...$('c-text').selectedOptions].map(o => o.value), metric: $('c-metric').value || null, notes: '',
  };
  if (body.prediction_moment.length < 12) { notice('Describe the prediction moment in at least a sentence.'); $('c-moment').focus(); return; }
  $('c-save').disabled = true;
  try { current = await api(`/projects/${current.id}/contract`, {method: 'PUT', body: JSON.stringify(body)}); notice(''); render(true); $('cell-data').scrollIntoView({behavior: 'smooth'}); }
  catch (error) { notice(error.message); } finally { const b = $('c-save'); if (b) b.disabled = false; }
}

/* ---------------------------------------------------------------- 2–6 · stages */
function renderStage(key) {
  const p = current, el = $('cell-' + key), no = STAGES.indexOf(key) + 2;
  const status = p.contract ? stageStatus(p, key) : 'locked';
  const meta = p.stage_meta.find(m => m.key === key);
  const record = p.records?.[key];
  const canRun = p.contract && !busy(p) && prevDone(p, key);
  const tools = status === 'locked' ? '' : ['running', 'queued'].includes(status) ? '<span class="running-dot"></span><span class="muted small">working…</span>'
    : `<button class="secondary small-btn" data-run="${key}" ${canRun ? '' : 'disabled'}>${record ? 'Run again' : 'Run this stage'}</button>`;
  let body = `<p class="cell-lead"><em>${esc(meta.question)}</em> <span class="muted">· ${esc(meta.workflow)}</span></p>`;
  if (status === 'locked') body += '<p class="muted small">Unlocks when the contract is saved.</p>';
  else if (status === 'pending') body += `<p class="muted small">${prevDone(p, key) ? 'Ready. Deterministic code runs it; the agent explains and cites.' : 'Waiting for the previous stage.'}</p>`;
  else if (status === 'failed') body += `<p class="map-empty">${esc(p.stages[key].error)}</p>`;
  if (record) body += stageBody(key, record) + decisionBlock(key, record) + notesBlock(record);
  el.innerHTML = cellHead(no, key, status, tools) + body;
  el.querySelector('[data-run]')?.addEventListener('click', () => act(`/projects/${p.id}/stages/${key}/run`));
  el.querySelectorAll('[data-approve]').forEach(b => b.addEventListener('click', async () => {
    const choice = el.querySelector('input[name="choice-' + key + '"]:checked')?.value;
    await act(`/projects/${p.id}/stages/${key}/approve`, {method: 'POST', body: JSON.stringify(choice ? {choice} : {})});
    if (b.dataset.approve === 'next' && STAGES[STAGES.indexOf(key) + 1]) await act(`/projects/${p.id}/stages/${STAGES[STAGES.indexOf(key) + 1]}/run`);
  }));
}
const tiles = items => `<div class="tiles">${items.map(([k, v, s]) => `<div class="tile"><small>${esc(k)}</small><strong>${esc(v)}</strong>${s ? `<span>${esc(s)}</span>` : ''}</div>`).join('')}</div>`;
const table = (head, rows) => `<div class="table-wrap"><table><thead><tr>${head.map(h => `<th>${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
function stageBody(key, r) {
  const ev = r.evidence, m = r.primary_metric;
  let html = `<p class="summary">${esc(r.setup_summary)}</p>`;
  if (key === 'data') {
    const ts = ev.target_summary || {};
    html += tiles([['training rows', Number(ev.train_rows).toLocaleString(), ev.sampling?.rule || ''], ['holdout rows', Number(ev.holdout_rows).toLocaleString(), 'sealed until the final stage'],
      ['usable inputs', ev.usable_feature_count, `of ${ev.raw_feature_count} columns`],
      r.task === 'binary' ? ['positive rate', pct(ts.positive_rate_train), `${ts.positives_train} positives`] : r.task === 'multiclass' ? ['classes', ts.classes, `smallest ${ts.smallest_class_count} rows`] : ['target mean', num(ts.mean_train, 3), `std ${num(ts.std_train, 3)}`]]);
    html += `<p class="small muted">Folds: ${esc(ev.cv_protocol)}. Holdout: ${esc(ev.holdout_split)}.</p>`;
    const flagged = ev.feature_profiles.filter(f => f.production_risks.length);
    html += `<details class="more"><summary>${plural(ev.feature_profiles.length, 'column')} profiled · ${plural(flagged.length, 'flag')}</summary>` + table(['COLUMN', 'ROLE', 'MISSING', 'UNIQUE', 'PSI', 'FLAGS'],
      ev.feature_profiles.map(f => `<tr class="${f.production_risks.length ? 'warn' : ''}"><td>${esc(f.feature)}</td><td>${esc(f.role)}</td><td>${pct(f.missing_rate)}</td><td>${f.unique_count}</td><td>${num(f.train_vs_holdout_psi, 3)}</td><td>${esc(f.production_risks.join(', ').replaceAll('_', ' '))}</td></tr>`)) + '</details>';
  } else if (key === 'leakage') {
    const cmp = ev.safe_vs_unsafe_training_cv;
    html += tiles([['apparent lift', (ev.apparent_lift >= 0 ? '+' : '') + num(ev.apparent_lift), `${label(m)} with the forbidden columns`], ['safe CV', ms(cmp.safe, m), `${cmp.model} · ${cmp.recipe}`],
      ['unsafe CV', cmp.unsafe ? ms(cmp.unsafe, m) : '—', ev.declared_leakage_features.length ? ev.declared_leakage_features.join(', ') : 'nothing forbidden'],
      ['detector canary', ev.detector_canary_passed ? 'passed' : 'missed', 'self-test of the heuristics']]);
    if (cmp.per_column) html += `<p class="small">Per column: ${Object.entries(cmp.per_column).map(([c, v]) => `<code>${esc(c)}</code> ${v.apparent_lift >= 0 ? '+' : ''}${num(v.apparent_lift)}`).join(' · ')}</p>`;
    if (ev.random_vs_time_cv) html += `<p class="small">Random KFold would report ${num(ev.random_vs_time_cv.random_kfold.metrics[m].mean)} vs ${num(ev.random_vs_time_cv.time_ordered.metric_mean)} time-ordered: gap ${num(ev.random_vs_time_cv.optimism_gap)}.</p>`;
    const cands = ev.heuristic_review_candidates.filter(f => !f.reasons.includes('declared_post_outcome_or_contested'));
    html += cands.length ? table(['COLUMN TO REVIEW', 'WHY', 'SIGNAL'], cands.map(f => `<tr class="warn"><td>${esc(f.feature)}</td><td>${esc(f.reasons.join(', ').replaceAll('_', ' '))}</td><td>${num(f.univariate_auc ?? f.normalized_mutual_information ?? f.binned_r2, 3)}</td></tr>`))
      : '<p class="small muted">No non-declared column tripped a heuristic.</p>';
    html += `<details class="more"><summary>Strongest single columns</summary>${table(['COLUMN', 'SIGNAL'], ev.top_univariate_signals.map(s => `<tr><td>${esc(s.feature)}</td><td>${num(s.signal, 3)}</td></tr>`))}</details>`;
  } else if (key === 'features') {
    html += table(['RECIPE', 'WHAT IT ADDS', 'FEATURES', label(m).toUpperCase() + ' (CV)', 'TIME'], ev.stage_results.map(x => `<tr class="${x.recipe === r.decision.chosen ? 'chosen' : ''}"><td>${esc(x.recipe)}${x.recipe === ev.best_recipe ? ' <small>best mean</small>' : ''}</td><td>${esc(x.description)}</td><td>${Math.round(x.feature_count_mean)}</td><td>${ms(x, m)}</td><td>${x.elapsed_seconds.toFixed(1)} s</td></tr>`));
    html += `<p class="small muted">${esc(ev.selection_rule)}</p>`;
  } else if (key === 'models') {
    html += table(['MODEL', label(m).toUpperCase() + ' (CV)', 'ADJUSTED', 'FEATURES', 'TIME'], ev.ranking.map(x => `<tr class="${x.model === r.decision.chosen ? 'chosen' : ''}"><td>${esc(x.model)}</td><td>${num(x.mean)} ± ${num(x.std)}</td><td>${num(x.adjusted_score)}</td><td>${Math.round(x.feature_count_mean)}</td><td>${x.elapsed_seconds.toFixed(1)} s</td></tr>`));
    html += `<p class="small muted">${esc(ev.selection_rule)}</p>`;
  } else if (key === 'final') {
    const h = ev.holdout_metrics, ci = ev.holdout_primary_metric_ci;
    html += tiles([['holdout ' + label(m), num(h[m]), `95% ${num(ci.low)}–${num(ci.high)} · ${ci.method}`], ['CV mean', num(ev.cv_selected_metric_mean), `gap ${(ev.cv_to_holdout_gap >= 0 ? '+' : '') + num(ev.cv_to_holdout_gap)}`],
      ['tuning', ev.tuning_decision.accepted ? 'accepted' : 'rejected', `${ev.tuning_decision.gain >= 0 ? '+' : ''}${num(ev.tuning_decision.gain)} vs ≥ ${num(ev.tuning_decision.required_gain)}`],
      ['holdout used', `${ev.holdout_uses_in_this_project}×`, ev.holdout_uses_in_this_project > 1 ? 'no longer a clean test' : 'the honest number']]);
    html += table(['CONFIGURATION', label(m).toUpperCase() + ' (CV)', 'PARAMETERS'], ev.configuration_results.map(x => `<tr class="${x.config_id === ev.selected_optimization ? 'chosen' : ''}"><td>${esc(x.config_id)}</td><td>${ms(x, m)}</td><td class="muted">${esc(Object.entries(x.model_params).map(([k, v]) => `${k.replace('model__', '')}=${v}`).join(', ') || 'defaults')}</td></tr>`));
    const extra = Object.entries(h).filter(([k, v]) => typeof v === 'number' && k !== m && !['threshold', 'precision_target', 'fit_seconds'].includes(k)).slice(0, 8);
    html += `<p class="small">Other holdout metrics: ${extra.map(([k, v]) => `${esc(k.replaceAll('_', ' '))} ${num(v, 3)}`).join(' · ')}.</p>`;
    html += `<p class="small">Trivial baselines: ${Object.entries(ev.baselines).map(([k, b]) => `${esc(k.replaceAll('_', ' '))} (${Object.entries(b).filter(([, v]) => typeof v === 'number').map(([kk, v]) => `${esc(kk.replaceAll('_', ' '))} ${num(v, 3)}`).join(', ')})`).join(' · ')}.</p>`;
    if (ev.threshold_analysis) html += `<details class="more"><summary>Threshold table (holdout)</summary>${table(['THRESHOLD', 'ALERTS', 'TRUE POSITIVES', 'PRECISION', 'RECALL'], ev.threshold_analysis.map(t => `<tr><td>${t.threshold}</td><td>${t.alerts}</td><td>${t.true_positives}</td><td>${num(t.precision, 3)}</td><td>${num(t.recall, 3)}</td></tr>`))}</details>`;
    html += `<details class="more"><summary>Top features of the final model</summary><div class="chips">${ev.top_final_features.slice(0, 15).map(f => `<span class="badge">${esc(f.feature)} ${num(f.normalized_importance, 3)}</span>`).join('')}</div></details>`;
  }
  html += `<details class="more claims"><summary>${plural(r.claims.length, 'claim')} with their limits</summary><ul>${r.claims.map(c => `<li><b>${esc(c.kind)}</b> ${esc(c.statement)}${c.limitations.length ? `<small>limits: ${esc(c.limitations.join(' '))}</small>` : ''}</li>`).join('')}</ul></details>`;
  return html;
}
function decisionBlock(key, r) {
  const p = current, d = r.decision, status = stageStatus(p, key);
  if (!d) return '';
  const next = STAGES[STAGES.indexOf(key) + 1];
  const options = d.options?.length ? `<div class="options">${d.options.map(o => `<label class="option ${o.id === d.chosen ? 'on' : ''}"><input type="radio" name="choice-${key}" value="${esc(o.id)}" ${o.id === d.chosen ? 'checked' : ''} ${status === 'approved' ? 'disabled' : ''}><b>${esc(o.label)}</b><span>${num(o.mean)} ± ${num(o.std)}${o.id === d.selected ? ' · rule choice' : ''}</span></label>`).join('')}</div>` : '';
  return `<div class="decision"><div class="decision-head"><b>${d.kind === 'recipe' ? 'Recipe to carry forward' : d.kind === 'model' ? 'Model to carry forward' : 'Final configuration'}</b>
    <span class="muted small">${status === 'approved' ? `approved${d.overridden ? ' · you overrode the rule' : ''}` : 'rule choice shown; pick another if you disagree'}</span></div>${options}
    ${status !== 'approved' ? `<div class="actions"><button class="primary small-btn" data-approve="only">Approve</button>${next ? `<button class="secondary small-btn" data-approve="next" ${busy(p) ? 'disabled' : ''}>Approve and run the next stage</button>` : ''}</div>` : ''}</div>`;
}
function notesBlock(r) {
  let notes = (r.notes || []).filter(n => n.severity !== 'info' || n.source === 'llm').slice(0, 5);
  if (!notes.length) notes = (r.notes || []).slice(0, 2);
  const rest = (r.notes || []).length - notes.length;
  return `<div class="notes"><div class="notes-head"><b>Agent notes</b><small>${plural((r.notes || []).length, 'note')} · every note cites its proof</small></div>
    ${notes.map(noteHtml).join('')}${rest > 0 ? `<p class="small muted">${rest} more in the agent panel →</p>` : ''}</div>`;
}
const noteHtml = n => `<div class="note ${sev(n.severity)} ${n.source === 'llm' ? 'llm' : ''}"><b>${esc(n.title)}</b><p>${esc(n.text)}</p>${n.action ? `<em>→ ${esc(n.action)}</em>` : ''}<div class="proof">${n.source === 'llm' ? '<span class="badge">LLM · advisory</span>' : ''}${proofChips(n.proof)}</div></div>`;

/* ---------------------------------------------------------------- 7 · report */
function renderReport() {
  const p = current, el = $('cell-report');
  if (!stageDone(p, 'final')) { el.innerHTML = cellHead(7, 'report', 'locked') + '<p class="cell-lead muted">After the final stage: a runnable notebook with the approved workflow, and a report with every decision and its proof.</p>'; return; }
  const f = p.records.final, ev = f.evidence, m = f.primary_metric, ci = ev.holdout_primary_metric_ci;
  const important = STAGES.flatMap(s => (p.records[s]?.notes || []).filter(n => n.severity !== 'info').map(n => ({...n, stage: s})));
  el.innerHTML = cellHead(7, 'report', 'approved') + `
    <div class="headline"><div><small>HOLDOUT ${esc(label(m).toUpperCase())}</small><strong>${num(ev.holdout_metrics[m])}</strong><span>95% ${num(ci.low)}–${num(ci.high)} · ${esc(ev.model)} on <code>${esc(ev.feature_recipe)}</code> · holdout used ${ev.holdout_uses_in_this_project}×</span></div>
      <div class="actions"><a class="primary" href="/api/projects/${esc(p.id)}/export/notebook">Download notebook (.ipynb) ↗</a><a class="secondary" href="/api/projects/${esc(p.id)}/export/report">Download report (.md) ↗</a><a class="secondary" href="/api/projects/${esc(p.id)}/export/sft" title="RAFT-style chat examples built from this project's records">Training examples (.jsonl) ↗</a></div></div>
    <p class="cell-lead">The notebook reproduces this workflow in plain scikit-learn: the contract, the split, the <code>${esc(ev.feature_recipe)}</code> recipe, <code>${esc(ev.model)}</code>${Object.keys(ev.selected_model_params).length ? ' with the accepted parameters' : ' with default parameters'}, the training-only folds and the once-only holdout.</p>
    ${important.length ? `<div class="notes"><div class="notes-head"><b>What to remember</b><small>${plural(important.length, 'warning')} across the stages</small></div>${important.map(n => `<div class="note ${sev(n.severity)}"><b>${esc(STEP_TITLES[n.stage])} · ${esc(n.title)}</b><p>${esc(n.text)}</p><div class="proof">${proofChips(n.proof)}</div></div>`).join('')}</div>` : ''}
    <p class="small muted">Research evidence is not production approval (DCLAB-R22): lineage, monitoring and a fresh confirmation come next.</p>`;
}

/* ---------------------------------------------------------------- agent panel */
function renderAgent() {
  const p = current, el = $('ws-agent');
  const notes = STAGES.flatMap(s => (p.records?.[s]?.notes || []).map(n => ({...n, stage: s})));
  const order = {high: 0, warning: 1, info: 2};
  notes.sort((a, b) => (order[a.severity] ?? 2) - (order[b.severity] ?? 2));
  const counts = {high: notes.filter(n => n.severity === 'high').length, warning: notes.filter(n => n.severity === 'warning').length};
  el.innerHTML = `<div class="agent-head"><span class="live-dot"></span><b>Agent</b><small>${p.contract ? (busy(p) ? `running ${esc(p.running || '')}…` : 'evidence first, cited') : 'waiting for the contract'}</small></div>
    <div class="agent-tabs">${[['notes', `Notes${notes.length ? ` · ${notes.length}` : ''}`], ['ask', 'Ask'], ['activity', 'Activity']].map(([k, l]) => `<button type="button" class="${agentTab === k ? 'active' : ''}" data-tab="${k}">${l}</button>`).join('')}</div>
    <div class="agent-body">${agentTab === 'notes' ? (notes.length ? `<p class="small muted">${counts.high} high · ${counts.warning} warning · ${notes.length - counts.high - counts.warning} info</p>` + notes.map(n => `<div class="note ${sev(n.severity)}"><small>${esc(STEP_TITLES[n.stage])}</small><b>${esc(n.title)}</b><p>${esc(n.text)}</p><div class="proof">${proofChips(n.proof)}</div></div>`).join('')
        : '<p class="small muted">Notes appear here as stages complete. Each one names the DCLab rule it applies and the measured precedent behind it.</p>')
      : agentTab === 'ask' ? `<div class="ask"><textarea id="ask-input" rows="2" placeholder="Ask about this project or the evidence: is X a leak? which model? why this recipe?"></textarea><button class="primary small-btn" id="ask-send">Ask</button></div>
        ${questions.map(q => `<div class="qa"><b>${esc(q.question)}</b><p>${esc(q.answer)}</p>${q.llm_answer ? `<p class="llm">${esc(q.llm_answer)} <span class="badge">LLM · advisory</span></p>` : ''}<div class="proof">${proofChips(q.proof)}</div></div>`).join('') || '<p class="small muted">Answers come from this project\'s results and the R&D evidence index; proofs are clickable.</p>'}`
      : `<ol class="timeline">${(p.activity || []).slice().reverse().map(a => `<li>${esc(a.kind.replaceAll('_', ' '))}${a.payload?.stage ? ` · ${esc(a.payload.stage)}` : ''}${a.payload?.error ? `<small>${esc(a.payload.error)}</small>` : ''}<small>${new Date(a.at).toLocaleString([], {hour: '2-digit', minute: '2-digit', day: '2-digit', month: 'short'})}</small></li>`).join('') || '<li>Nothing yet</li>'}</ol>`}</div>`;
  el.querySelectorAll('[data-tab]').forEach(b => b.addEventListener('click', () => { agentTab = b.dataset.tab; renderAgent(); }));
  $('ask-send')?.addEventListener('click', async () => {
    const question = $('ask-input').value.trim(); if (!question) return;
    $('ask-send').disabled = true;
    try { questions.unshift(await api(`/projects/${p.id}/ask`, {method: 'POST', body: JSON.stringify({question})})); renderAgent(); }
    catch (error) { notice(error.message); $('ask-send').disabled = false; }
  });
}
