/* DCLab notebook entry point: hash router + boot.
   Routes: #projects · #project/<id> · #map · #map/<idea> · #knowledge · #lab · #run/<id> */
import {$, state, api, notice} from './core.js';
import {initDrawer} from './drawer.js';
import {initProjects, renderProjects, refreshProjects} from './views/projects.js';
import {renderProject, leaveProject} from './views/project.js';
import {initMap, renderMap} from './views/map.js';
import {initComposer, refreshRuns, showComposer, showRun} from './views/runs.js';
import {renderKnowledge} from './views/knowledge.js';

const VIEWS = {projects: 'Projects', project: 'Projects', map: 'Research map', knowledge: 'Knowledge', research: 'Agent campaigns'};
const SECTION = {projects: 'projects-view', project: 'project-view', map: 'map-view', knowledge: 'knowledge-view', research: 'research-view'};

function show(name) {
  state.view = name;
  Object.entries(SECTION).forEach(([v, id]) => $(id).classList.toggle('hidden', v !== name));
  const navKey = name === 'project' ? 'projects' : name;
  document.querySelectorAll('.nav').forEach(a => a.classList.toggle('active', a.dataset.view === navKey));
  $('page-label').textContent = VIEWS[name];
  window.scrollTo(0, 0);
}

async function route() {
  const [head, arg] = location.hash.replace(/^#/, '').split('/').map(decodeURIComponent);
  if (state.view === 'project' && head !== 'project') { leaveProject(); state.project = null; }
  if (head === 'run' && arg) { show('research'); await showRun(arg); return; }
  if (head === 'project' && arg) { show('project'); await renderProject(arg); return; }
  if (head === 'lab') { show('research'); if (!state.selected) showComposer(); return; }
  const name = VIEWS[head] && head !== 'project' ? head : 'projects';
  show(name);
  if (name === 'projects') await renderProjects();
  if (name === 'map') await renderMap(arg);
  if (name === 'knowledge') await renderKnowledge();
}

async function tick() {
  if (!state.config) return;
  await refreshRuns();
}

(async () => {
  initDrawer();
  initMap();
  initProjects();
  window.addEventListener('hashchange', route);
  route();  // projects and the map need no agent runtime, so they render before the config loads
  try {
    state.config = await api('/config');
    initComposer();
    $('connection').textContent = state.config.api_key_configured ? '● LLM connected' : '○ no API key · deterministic agent';
    $('connection').classList.toggle('off', !state.config.api_key_configured);
    await Promise.all([refreshRuns(), refreshProjects()]);
    setInterval(tick, 4000);
  } catch (error) { notice(error.message); $('connection').textContent = 'Connection unavailable'; }
})();
