/* The intern: describe a task, watch the plan and the tool calls, read the report, open the project it built. */
import {$, esc, state, api, notice} from '../core.js';
import {recordChip} from '../drawer.js';
import {refreshProjects} from './projects.js';

let status = null, sessions = [], current = null, timer = null;
const STATUS = {queued: 'queued', running: 'working…', completed: 'done', failed: 'failed', budget_exhausted: 'budget used up', stopped: 'stopped'};
const live = s => s && ['queued', 'running'].includes(s.status);

export function leaveIntern() { clearTimeout(timer); current = null; }

async function load() {
  if (!status) status = await api('/intern');
  sessions = await api('/intern/sessions');
}

export async function renderIntern(arg, extra) {
  $('page-label').textContent = 'Intern';
  try { await load(); } catch (error) { notice(error.message); return; }
  if (arg && arg !== 'new') {
    try { current = await api('/intern/sessions/' + arg); } catch (error) { notice(error.message); current = null; }
  } else current = null;
  const prefillProject = arg === 'new' ? extra : null;
  const el = $('intern-view');
  el.innerHTML = `
    <div class="page-heading"><div class="eyebrow">THE DCLAB INTERN</div><h1>Describe the task.<br>The intern runs the notebook.</h1>
      <p>A chat mode with tools and a budget, like Hugging Face's ML Intern, but the compute is the DCLab notebook on this machine:<br>every tool it can call is one the R&D already verified, and the model never owns a split, a metric or a selection rule.</p></div>
    <div class="intern-status card"><span class="live-dot"></span><b>${status.mode === 'llm' ? `Model: ${esc(status.model)} via ${esc(status.endpoint)}` : 'No model configured · standard plan'}</b><span>${esc(status.note)}</span></div>
    ${status.mcp_url ? `<div class="card chatui"><div class="eyebrow">PREFER HUGGINGCHAT'S INTERFACE?</div>
      <p>Hugging Face's <a href="https://github.com/huggingface/chat-ui" target="_blank" rel="noopener">Chat UI</a> runs locally with the DCLab notebook attached as an MCP server, so its ML Intern mode (or any tool-capable model) can create projects, write contracts, run the stages and export notebooks here.
      Run <code>${esc(status.chat_ui.command)}</code> (or <code>${esc(status.chat_ui.intern_command)}</code> for ML Intern mode) in a second terminal, then open <a href="${esc(status.chat_ui.intern_url)}" target="_blank" rel="noopener">${esc(status.chat_ui.intern_url)}</a>.
      MCP endpoint: <code>${esc(status.mcp_url)}</code> · tools: ${status.tools.length}.</p></div>` : ''}
    <div class="intern-layout">
      <aside class="card intern-list"><div class="section-label">SESSIONS <a href="#intern" aria-label="New session">+</a></div><div id="intern-sessions"></div></aside>
      <div id="intern-main"></div>
    </div>`;
  renderList();
  if (current) renderSession(); else renderComposer(prefillProject);
  schedule();
}

function renderList() {
  $('intern-sessions').innerHTML = sessions.length ? sessions.slice(0, 12).map(s => `<a class="run-nav ${current?.id === s.id ? 'active' : ''}" href="#intern/${esc(s.id)}"><strong>${esc(s.task)}</strong><small>${esc(STATUS[s.status] || s.status)} · ${s.steps} steps${s.project_id ? ' · project' : ''}</small></a>`).join('')
    : '<p class="muted small">No sessions yet.</p>';
}

function renderComposer(projectId) {
  const main = $('intern-main');
  const budget = status.default_budget;
  main.innerHTML = `<div class="card composer intern-composer">
    <div class="card-heading"><h2>${projectId ? 'Hand a project to the intern' : 'What should the intern do?'}</h2></div>
    ${projectId ? `<p class="small muted">It will continue project <code>${esc(projectId)}</code>: read its data and contract, run what is missing, and report.</p>` : ''}
    <label class="sr-only" for="intern-task">Task</label><textarea id="intern-task" rows="4" placeholder="e.g. Build a leakage-safe churn model on the Telco sample and tell me the honest score.">${projectId ? 'Continue this project: run every missing stage, then report the honest score, the leakage findings and the decisions with their proof.' : ''}</textarea>
    <div class="field-label">EXAMPLES <span>click to use</span></div>
    <div class="chips examples">${status.examples.map(t => `<button type="button" class="sample example" data-example="${esc(t)}"><small>${esc(t)}</small></button>`).join('')}</div>
    <div class="settings-grid intern-budget">
      <label>Tool-call budget<select id="intern-steps"><option value="12">12 calls · a quick pass</option><option value="24" ${budget.max_steps === 24 ? 'selected' : ''}>24 calls</option><option value="48">48 calls · room to iterate</option></select></label>
      <label>Time budget<select id="intern-minutes"><option value="5">5 minutes</option><option value="20" selected>20 minutes</option><option value="60">60 minutes</option></select></label>
    </div>
    <p class="small muted">The budget is enforced on every tool call. New projects start in quick mode (3,000 rows); ask for "full" rows in the task if you want all of them.</p>
    <div class="composer-footer"><p><span class="lock">▣</span> Tools only. No code execution, no network, no key in the browser.</p><button id="intern-start" class="primary">Start <span>↗</span></button></div>
  </div>`;
  main.querySelectorAll('[data-example]').forEach(b => b.addEventListener('click', () => { $('intern-task').value = b.dataset.example; }));
  $('intern-start').addEventListener('click', async () => {
    const task = $('intern-task').value.trim();
    if (task.length < 8) { notice('Describe the task in at least a sentence.'); return; }
    $('intern-start').disabled = true;
    try {
      const s = await api('/intern/sessions', {method: 'POST', body: JSON.stringify({task, project_id: projectId || null, budget: {max_steps: Number($('intern-steps').value), max_minutes: Number($('intern-minutes').value)}})});
      notice(''); location.hash = '#intern/' + s.id;
    } catch (error) { notice(error.message); $('intern-start').disabled = false; }
  });
}

function linkify(text) {
  return esc(text).replace(/#project\/([0-9a-f]{12})/g, (m, id) => `<a href="#project/${id}">open project ${id} ↗</a>`)
    .replace(/\b((?:DCLAB-R\d+|WF-\d+|EXP-\d+|PIT-\d+|LEAK-[\w-]+|FINDING-[\w-]+|DATASET-[\w-]+))\b/g, (m, id) => recordChip(id));
}

function renderSession() {
  const s = current, main = $('intern-main');
  const used = s.used || {};
  main.innerHTML = `<div class="card intern-session">
    <div class="intern-head"><div><div class="eyebrow">${esc(s.mode === 'llm' ? `MODEL · ${s.model || ''}` : 'STANDARD PLAN')}</div><h2>${esc(s.task)}</h2></div>
      <div class="actions"><span class="stage-chip ${esc(s.status === 'completed' ? 'approved' : live(s) ? 'running' : s.status === 'failed' ? 'failed' : 'completed')}">${esc(STATUS[s.status] || s.status)}</span>
        ${s.project_id ? `<a class="secondary" href="#project/${esc(s.project_id)}">Open project ↗</a>` : ''}<button class="secondary danger" id="intern-delete">Delete</button></div></div>
    <div class="budget-strip"><span>${used.steps || 0} / ${s.budget.max_steps} tool calls</span><span>${(used.minutes || 0).toFixed(1)} / ${s.budget.max_minutes} min</span>${s.mode === 'llm' ? `<span>${Number(used.input_tokens || 0).toLocaleString()} in · ${Number(used.output_tokens || 0).toLocaleString()} out tokens</span>` : ''}${s.error ? `<span class="err">${esc(s.error)}</span>` : ''}</div>
    ${s.plan ? `<div class="plan"><b>Plan</b><pre>${esc(s.plan)}</pre></div>` : ''}
    <div class="steps">${(s.steps || []).map(st => `<details class="step ${st.ok ? '' : 'err'}"><summary><span class="step-no">${st.n}</span><code>${esc(st.tool)}</code><span class="step-summary">${esc(st.summary)}</span><small>${st.elapsed_seconds ? st.elapsed_seconds.toFixed(1) + ' s' : ''}</small></summary><pre>${esc(JSON.stringify(st.arguments, null, 1))}</pre></details>`).join('') || (live(s) ? '<p class="muted small"><span class="running-dot"></span>Starting…</p>' : '')}
      ${live(s) ? '<p class="muted small"><span class="running-dot"></span>working…</p>' : ''}</div>
    ${s.final ? `<div class="report"><b>Report</b><div class="report-text">${linkify(s.final).replaceAll('\n', '<br>')}</div></div>` : ''}
    ${!live(s) ? `<div class="ask followup"><textarea id="intern-followup" rows="2" placeholder="${s.mode === 'llm' ? 'Ask a follow-up or give the next instruction…' : 'Ask about the project; the deterministic agent answers from its results and the evidence…'}"></textarea><button class="primary small-btn" id="intern-send">Send</button></div>` : ''}
  </div>`;
  $('intern-delete').addEventListener('click', async () => {
    if (!confirm('Delete this session? The project it built stays.')) return;
    try { await api('/intern/sessions/' + s.id, {method: 'DELETE'}); location.hash = '#intern'; } catch (error) { notice(error.message); }
  });
  $('intern-send')?.addEventListener('click', async () => {
    const text = $('intern-followup').value.trim(); if (!text) return;
    $('intern-send').disabled = true;
    try { current = await api(`/intern/sessions/${s.id}/message`, {method: 'POST', body: JSON.stringify({text})}); renderSession(); schedule(); }
    catch (error) { notice(error.message); $('intern-send').disabled = false; }
  });
}

function schedule() {
  clearTimeout(timer);
  if (!current || !live(current)) return;
  timer = setTimeout(async () => {
    try { current = await api('/intern/sessions/' + current.id); sessions = await api('/intern/sessions'); renderList(); renderSession(); if (current.project_id) refreshProjects(); } catch {}
    schedule();
  }, 2000);
}
