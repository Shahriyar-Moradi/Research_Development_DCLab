/* Research map: every research idea as a tree (by theme) and a graph (by relation), and one page per idea
   with the same seven sections everywhere: idea · champion · experiments · notebooks · evaluation · reports · related.
   Data comes from /api/research, the same records that generate research/<track>/INDEX.md. */
import {$, esc, api, notice} from '../core.js';
import {fileChip as file} from '../drawer.js';

let map = null, filter = '';
const SECTIONS = [
  ['idea', 'Idea', 'The question this track answers'],
  ['champion', 'Champion', 'The best measured result so far'],
  ['experiments', 'Experiments', 'Code and results in experiments/'],
  ['notebooks', 'Notebooks', 'Exploration and walkthroughs'],
  ['evaluation', 'Evaluation', 'Benchmarks, plots and test cases'],
  ['reports', 'Reports & research', 'Written findings and notes'],
  ['related', 'Related', 'Connected ideas and shared evidence'],
];
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const stage = t => /^active/i.test(t.status) ? 'active' : /conclu/i.test(t.status) ? 'concluded' : /paus/i.test(t.status) ? 'paused' : 'planned';
const stageLabel = t => t.status || 'idea · not started';
const short = path => path.split('/').filter(Boolean).pop();
const empty = text => `<p class="map-empty">${esc(text)}</p>`;

async function load() {
  if (!map) map = await api('/research');
  return map;
}

/* ---------- tree: themes → ideas ---------- */
function renderTree(current) {
  const q = filter.toLowerCase();
  const match = t => !q || [t.name, t.title, t.idea, t.theme].join(' ').toLowerCase().includes(q);
  $('map-tree').innerHTML = map.themes.map(theme => {
    const tracks = theme.tracks.map(n => map.tracks[n]).filter(match);
    if (!tracks.length) return '';
    return `<details class="tree-theme" open><summary><span>${esc(theme.name)}</span><small>${tracks.length}</small></summary><ul>${tracks.map(t => `
      <li><a class="tree-track ${t.name === current ? 'active' : ''}" href="#map/${esc(t.name)}">
        <span class="stage-dot ${stage(t)}" title="${esc(stageLabel(t))}"></span>
        <span class="tree-name">${esc(t.title.replace(/\s*\(.*\)$/, ''))}</span>
        <span class="tree-meta">${t.counts.champions ? `<b title="champions">★ ${t.counts.champions}</b>` : ''}${t.counts.experiments ? `<i title="experiments">⚗ ${t.counts.experiments}</i>` : ''}${t.counts.notebooks ? `<i title="notebooks">▤ ${t.counts.notebooks}</i>` : ''}</span>
      </a></li>`).join('')}</ul></details>`;
  }).join('') || empty('No idea matches this search.');
}

/* ---------- graph: one column per theme, lines between related ideas ---------- */
function renderGraph(current) {
  const colW = 196, rowH = 54, top = 54, nodeW = 172, nodeH = 36;
  const pos = {};
  map.themes.forEach((theme, i) => theme.tracks.forEach((n, j) => { pos[n] = {x: 12 + i * colW, y: top + j * rowH}; }));
  const rows = Math.max(...map.themes.map(t => t.tracks.length));
  const width = 12 + map.themes.length * colW, height = top + rows * rowH + 8;
  const near = new Set(current ? [current, ...map.edges.filter(e => e.includes(current)).flat()] : []);
  const edges = map.edges.map(([a, b]) => {
    const p = pos[a], q = pos[b];
    if (!p || !q) return '';
    const [l, r] = p.x <= q.x ? [p, q] : [q, p];
    const same = l.x === r.x;
    const x1 = same ? l.x + nodeW : l.x + nodeW, y1 = l.y + nodeH / 2, x2 = same ? r.x + nodeW : r.x, y2 = r.y + nodeH / 2;
    const bend = same ? `C ${x1 + 22} ${y1}, ${x2 + 22} ${y2}` : `C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}`;
    const cls = current ? (a === current || b === current ? 'edge on' : 'edge off') : 'edge';
    return `<path class="${cls}" d="M ${x1} ${y1} ${bend}, ${x2} ${y2}"></path>`;
  }).join('');
  const heads = map.themes.map((theme, i) => {  // long theme names wrap onto two lines so columns never overlap
    const words = theme.name.toUpperCase().split(' '), cut = words.join(' ').length > 22 ? Math.ceil(words.length / 2) : words.length;
    const x = 12 + i * colW + 4;
    return `<text class="graph-theme" x="${x}" y="${cut < words.length ? 18 : 26}">${esc(words.slice(0, cut).join(' '))}${cut < words.length ? `<tspan x="${x}" dy="13">${esc(words.slice(cut).join(' '))}</tspan>` : ''}</text>`;
  }).join('');
  const nodes = Object.entries(pos).map(([n, p]) => {
    const t = map.tracks[n];
    const cls = ['node', stage(t), n === current ? 'selected' : '', current && !near.has(n) ? 'off' : ''].join(' ');
    const label = t.title.replace(/\s*\(.*\)$/, '');
    return `<a href="#map/${esc(n)}" class="${cls}"><title>${esc(t.title)}</title><rect x="${p.x}" y="${p.y}" width="${nodeW}" height="${nodeH}" rx="9"></rect>
      <circle cx="${p.x + 14}" cy="${p.y + nodeH / 2}" r="4"></circle><text x="${p.x + 25}" y="${p.y + nodeH / 2 + 4}">${esc(label.length > 23 ? label.slice(0, 22) + '…' : label)}</text></a>`;
  }).join('');
  $('map-graph').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Research ideas grouped by theme, with lines between related ideas">${heads}${edges}${nodes}</svg>`;
}

/* ---------- overview when no idea is selected ---------- */
function renderOverview() {
  const tracks = Object.values(map.tracks).sort((a, b) => b.counts.files - a.counts.files);
  $('map-detail').innerHTML = `
    <div class="card map-section"><div class="section-head"><h2>How to read this map</h2></div>
      <p>Every research idea lives in <code>research/&lt;idea&gt;/</code> and has the same shape, so you always know where to look:</p>
      <ol class="shape">${SECTIONS.slice(0, 6).map(([, title, hint], i) => `<li><span>${i + 1}</span><b>${esc(title)}</b><small>${esc(hint)}</small></li>`).join('')}</ol>
      <p class="muted small">Pick an idea in the tree or the graph. A filled dot means the idea is active, an outline means planned.</p></div>
    <div class="card map-section"><div class="section-head"><h2>All ideas</h2><small>sorted by how much material each has</small></div>
      <div class="table-wrap"><table class="idea-table"><thead><tr><th>IDEA</th><th>THEME</th><th>★</th><th>EXP.</th><th>NOTEBOOKS</th><th>REPORTS</th></tr></thead><tbody>
      ${tracks.map(t => `<tr><td><a href="#map/${esc(t.name)}"><span class="stage-dot ${stage(t)}"></span>${esc(t.title)}</a><small>${esc(stageLabel(t))}</small></td><td>${esc(t.theme)}</td><td>${t.counts.champions || '—'}</td><td>${t.counts.experiments || '—'}</td><td>${t.counts.notebooks || '—'}</td><td>${t.counts.reports || '—'}</td></tr>`).join('')}
      </tbody></table></div></div>`;
}

/* ---------- one idea, seven sections ---------- */
function groups(list, noun) {
  return list.map(g => {
    // Same file name in several sub-folders (one report set per sub-project): show the sub-folder too.
    const names = g.files.map(short), twice = new Set(names.filter((n, i) => names.indexOf(n) !== i));
    const label = f => (twice.has(short(f)) ? f.split('/').slice(-2).join('/') : short(f)).replace(/\.ipynb$/, '');
    const shown = g.files.slice(0, 12), rest = g.files.slice(12);
    return `<div class="file-group"><div class="file-group-head">${file(g.path, g.label)}<small>${plural(g.files.length, noun)}${g.folders > 1 ? ` · ${g.folders} folders` : ''}</small></div>
      <div class="chips">${shown.map(f => file(f, label(f))).join('')}</div>
      ${rest.length ? `<details class="more"><summary>${rest.length} more</summary><div class="chips">${rest.map(f => file(f, label(f))).join('')}</div></details>` : ''}</div>`;
  }).join('');
}
function section(key, n, body, count) {
  const [, title, hint] = SECTIONS.find(s => s[0] === key);
  return `<section class="card map-section" id="sec-${key}"><div class="section-head"><span class="section-no">${n}</span><h2>${esc(title)}</h2>${count !== undefined ? `<span class="count">${count}</span>` : ''}<small>${esc(hint)}</small></div>${body}</section>`;
}
function renderTrack(t) {
  const c = t.counts;
  const champion = t.champion.rows.length ? `<p class="muted small">${esc(t.champion.definition)}</p><div class="table-wrap"><table><thead><tr><th>DATASET</th><th>CHAMPION</th><th>SCORE</th><th>INTERVAL / NOTE</th><th>SOURCE</th></tr></thead><tbody>
      ${t.champion.rows.map(r => `<tr class="${r.dataset.includes('⚠') ? 'warn' : ''}"><td>${esc(r.dataset)}</td><td>${esc(r.model)}</td><td><b>${esc(r.value)}</b><small>${esc(r.metric)}</small></td><td>${esc(r.interval) || '—'}</td><td>${file(r.source)}</td></tr>`).join('')}</tbody></table></div>`
    : empty('No champion yet. The first measured result recorded in this track becomes the baseline to beat.');
  const experiments = (t.experiments.length ? `<div class="exp-grid">${t.experiments.map(e => `<article class="exp-card"><h3>${file(e.path, e.name)}</h3>
      <div class="badges"><span>${plural(e.results, 'result file')}</span><span>${plural(e.notebooks, 'notebook')}</span><span>${plural(e.scripts, 'script')}</span></div>
      <p class="small">Start with ${file(e.entry)}</p>${e.subprojects.length ? `<details class="more"><summary>${plural(e.subprojects.length, 'sub-project')}</summary><div class="chips">${e.subprojects.map(s => file(s.path, s.name)).join('')}</div></details>` : ''}</article>`).join('')}</div>`
    : empty('No experiments in this folder yet.'))
    + (t.campaigns.length ? `<h3 class="sub">Shared evidence campaigns</h3><ul class="rows">${t.campaigns.map(cp => `<li>${file(cp.path, cp.name)}<span>${plural(cp.results, 'result file')}</span>${file(cp.report)}</li>`).join('')}</ul>` : '');
  const evaluation = (t.evaluation.length || t.shared_evaluation.length)
    ? `<ul class="rows">${t.evaluation.map(e => `<li>${file(e.path, e.name)}<span>${plural(e.files.length, 'file')}</span></li>`).join('')}${t.shared_evaluation.map(r => `<li>${file(r)}<span>shared evaluation evidence</span></li>`).join('')}</ul>`
    : empty('No evaluation artifacts yet.');
  const related = (t.related.length ? `<div class="chips">${t.related.map(r => map.tracks[r] ? `<a class="idea-chip" href="#map/${esc(r)}"><span class="stage-dot ${stage(map.tracks[r])}"></span>${esc(map.tracks[r].title.replace(/\s*\(.*\)$/, ''))}</a>` : '').join('')}</div>` : empty('No related ideas recorded.'));

  $('map-detail').innerHTML = `
    <div class="card track-head"><div class="eyebrow">${esc(t.theme.toUpperCase())}</div>
      <div class="track-title"><h2>${esc(t.title)}</h2><span class="stage ${stage(t)}">${esc(stageLabel(t))}</span></div>
      <p class="muted small"><code>${esc(t.path)}/</code> · ${plural(c.files, 'file')}</p>
      <div class="chips">${file(t.readme, 'README.md · the idea')}${file(t.index, 'INDEX.md · the generated map')}${file(t.path + '/', 'Browse folder')}</div>
      <nav class="section-nav" aria-label="Sections">${SECTIONS.map(([key, title], i) => {
        const n = {champion: c.champions, experiments: c.experiments + c.campaigns, notebooks: c.notebooks, evaluation: c.evaluation, reports: c.reports, related: t.related.length}[key];
        return `<button type="button" data-jump="${key}" class="${n === 0 ? 'none' : ''}"><span>${i + 1}</span>${esc(title)}${n !== undefined ? `<b>${n}</b>` : ''}</button>`;
      }).join('')}</nav></div>
    ${section('idea', 1, `<p class="idea">${esc(t.idea || 'See README.md.')}</p>`)}
    ${section('champion', 2, champion, c.champions)}
    ${section('experiments', 3, experiments, c.experiments)}
    ${section('notebooks', 4, t.notebooks.length ? groups(t.notebooks, 'notebook') : empty('No notebooks yet.'), c.notebooks)}
    ${section('evaluation', 5, evaluation, c.evaluation)}
    ${section('reports', 6, t.reports.length ? groups(t.reports, 'document') : empty('Research notes live in README.md until there is enough material for reports/.'), c.reports)}
    ${section('related', 7, related, t.related.length)}`;
  $('map-detail').querySelectorAll('[data-jump]').forEach(b => b.addEventListener('click', () => $('sec-' + b.dataset.jump).scrollIntoView({behavior: 'smooth', block: 'start'})));
}

export function initMap() {
  $('map-search').addEventListener('input', e => { filter = e.target.value; if (map) renderTree(location.hash.split('/')[1]); });
}

export async function renderMap(trackName) {
  try {
    await load();
    const t = map.tracks[trackName];
    const s = map.totals;
    $('map-totals').innerHTML = [[s.tracks, 'research ideas'], [s.experiments, 'experiment folders'], [s.notebooks, 'notebooks'], [s.champions, 'champion results'], [s.reports, 'reports & notes'], [s.campaigns, 'shared evidence campaigns']]
      .map(([n, label]) => `<div><strong>${n}</strong><span>${esc(label)}</span></div>`).join('');
    renderTree(t ? trackName : null);
    renderGraph(t ? trackName : null);
    if (t) renderTrack(t); else renderOverview();
    $('page-label').textContent = t ? 'Research map / ' + t.title.replace(/\s*\(.*\)$/, '') : 'Research map';
  } catch (error) { notice(error.message); }
}
