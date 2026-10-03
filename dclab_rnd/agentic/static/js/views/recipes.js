/* Recipes & traces: every run's replayable recipes and exportable agent trajectory. */
import {$, esc, state, labels} from '../core.js';

export function renderRecipes(){
  const runs = state.runs;
  $('recipe-list').innerHTML=runs.length?runs.map(r=>`<article class="card recipe-row"><div><h3>${esc(r.config.datasets.map(d=>labels[d]).join(' + '))}</h3><p>${esc(r.config.goal.slice(0,130))}</p><p>${esc(r.status)} · ${new Date(r.created).toLocaleDateString()} · source-grouped curation required</p></div><div class="actions"><a class="secondary" href="#run/${esc(r.id)}">View recipes</a><a class="secondary" href="/api/runs/${esc(r.id)}/export">Export trajectory ↗</a></div></article>`).join(''):'<div class="card empty-card"><h2>A workflow worth repeating.</h2><p>Each completed experiment saves its configuration, code fingerprints and evaluation procedure.</p></div>';
}
