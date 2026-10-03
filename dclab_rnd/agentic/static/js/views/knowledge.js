/* Knowledge: the field guide plus provisional, evidence-linked lessons from finished runs. */
import {$, esc, labels, api, notice} from '../core.js';

export async function renderKnowledge() {
  try {
    const lessons=await api('/knowledge');
    const guide=`<article class="card lesson guide-card"><span class="label">DCLAB MODEL-BUILDING FIELD GUIDE</span><h3>From raw data to reliable evidence.</h3><p>An outsider-readable synthesis of EDA, leakage, feature engineering, algorithms, optimization, reliability, agent roles, 22 evidence-linked rules, and 10 reusable workflow blocks.</p><a class="primary" href="/guide" target="_blank" rel="noopener">Open complete guide ↗</a></article>`;
    $('knowledge-list').innerHTML=guide+(lessons.length?lessons.map(l=>`<article class="card lesson"><span class="label">PROVISIONAL FINDING · ${esc(l.datasets.map(d=>labels[d]).join(' / '))}</span><h3>${esc(l.claim)}</h3><p>${esc(l.scope)}</p><p><b>Counterevidence:</b> ${esc(l.counterevidence)}<br><b>Next test:</b> ${esc(l.follow_up)}</p><a class="secondary" href="#run/${esc(l.run_id)}">Inspect ${l.evidence_ids.length} cited experiment${l.evidence_ids.length===1?'':'s'} ↗</a></article>`).join(''):'<div class="card empty-card"><h2>Evidence grows into knowledge.</h2><p>Finish a research run to see its critiqued, cited lessons here.</p></div>');
  }catch(e){notice(e.message);}
}
