/* Research Studio entry point: hash router + boot.
   Routes: #map · #map/<idea> · #research · #run/<id> · #knowledge */
import {$, state, api, notice} from './core.js';
import {initMap, renderMap} from './views/map.js';
import {initComposer, refreshRuns, showComposer, showRun} from './views/runs.js';
import {renderKnowledge} from './views/knowledge.js';

const VIEWS = {map: 'Research map', research: 'Agent research', knowledge: 'Knowledge'};

function show(name) {
  state.view = name;
  Object.keys(VIEWS).forEach(v => $(v + '-view').classList.toggle('hidden', v !== name));
  document.querySelectorAll('.nav').forEach(a => a.classList.toggle('active', a.dataset.view === name));
  $('page-label').textContent = VIEWS[name];
  window.scrollTo(0, 0);
}

async function route() {
  const [head, arg] = location.hash.replace(/^#/, '').split('/').map(decodeURIComponent);
  if (head === 'run' && arg) { show('research'); await showRun(arg); return; }
  const name = VIEWS[head] ? head : 'map';
  show(name);
  if (name === 'map') await renderMap(arg);
  if (name === 'research' && !state.selected) showComposer();
  if (name === 'knowledge') await renderKnowledge();
}

async function tick() {
  if (!state.config) return;
  await refreshRuns();
}

(async () => {
  initMap();
  window.addEventListener('hashchange', route);
  route();  // the map needs no agent runtime, so it renders before the config loads
  try {
    state.config = await api('/config');
    initComposer();
    $('connection').textContent = state.config.api_key_configured ? '● OpenAI connected · key configured' : '○ API key needed for agent runs';
    $('connection').classList.toggle('off', !state.config.api_key_configured);
    await refreshRuns();
    const active = state.runs.find(r => ['running', 'queued'].includes(r.status));
    if (active && !location.hash) location.hash = '#run/' + active.id;
    setInterval(tick, 4000);
  } catch (error) { notice(error.message); $('connection').textContent = 'Connection unavailable'; }
})();
