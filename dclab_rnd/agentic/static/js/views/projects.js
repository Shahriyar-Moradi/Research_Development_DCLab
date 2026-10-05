/* Projects home: start a project, or open one. */
import {$, esc, state, api, notice} from '../core.js';

const STAGE_ORDER = ['data', 'leakage', 'features', 'models', 'final'];
const INDUSTRIES = ['general', 'fintech and banking', 'insurance', 'retail and e-commerce', 'telecom', 'logistics and delivery', 'health', 'manufacturing', 'energy', 'marketing', 'human resources', 'public sector'];

export function progress(p) {
  const done = STAGE_ORDER.filter(s => ['completed', 'approved'].includes(p.stages?.[s]?.status)).length;
  const steps = (p.data ? 1 : 0) + (p.solution ? 1 : 0) + done;
  return {steps, total: 7, done, label: !p.data ? 'needs data' : !p.solution ? 'needs a solution' : p.running ? `running: ${p.running}` : done === 5 ? 'finished' : `${done} of 5 stages`};
}

export function renderProjectList(list) {
  const el = $('project-list');
  if (!el) return;
  el.innerHTML = list.length ? list.slice(0, 8).map(p => `<a class="run-nav ${state.project === p.id ? 'active' : ''}" href="#project/${esc(p.id)}"><strong>${esc(p.name)}</strong><small>${esc(progress(p).label)}</small></a>`).join('')
    : '<p class="muted small">No projects yet.</p>';
}

export async function refreshProjects() {
  try {
    state.projects = await api('/projects');
    renderProjectList(state.projects);
  } catch (error) { notice(error.message); }
}

export async function renderProjects() {
  await refreshProjects();
  const cards = $('project-cards');
  const list = state.projects || [];
  cards.innerHTML = list.length ? list.map(p => {
    const pr = progress(p);
    return `<a class="card project-card" href="#project/${esc(p.id)}">
      <div class="project-card-head"><h3>${esc(p.name)}</h3><span class="stage-chip ${pr.done === 5 ? 'approved' : p.running ? 'running' : 'pending'}">${esc(pr.label)}</span></div>
      <p>${esc(p.goal || 'No goal written yet.')}</p>
      <div class="project-card-meta"><span>${esc(p.industry)}</span>${p.data ? `<span>${esc(p.data.filename)} · ${Number(p.data.rows).toLocaleString()} rows</span>` : ''}${p.solution ? `<span>target <code>${esc(p.solution.target)}</code> · ${esc(p.solution.task)}</span>` : ''}</div>
      <div class="progress-steps">${Array.from({length: 7}, (_, i) => `<i class="${i < pr.steps ? 'on' : ''}"></i>`).join('')}</div>
    </a>`;
  }).join('') : '<div class="card empty-card"><h2>No projects yet.</h2><p>Create one on the left: a name, the industry and the question you want the model to answer.</p></div>';
}

export function initProjects() {
  $('industry').innerHTML = INDUSTRIES.map(i => `<option value="${esc(i)}">${esc(i)}</option>`).join('');
  $('create-project').addEventListener('click', async () => {
    const name = $('project-name').value.trim();
    if (!name) { notice('Give the project a name.'); return; }
    $('create-project').disabled = true;
    try {
      const p = await api('/projects', {method: 'POST', body: JSON.stringify({name, industry: $('industry').value, goal: $('project-goal').value})});
      $('project-name').value = ''; $('project-goal').value = '';
      location.hash = '#project/' + p.id;
    } catch (error) { notice(error.message); } finally { $('create-project').disabled = false; }
  });
}
