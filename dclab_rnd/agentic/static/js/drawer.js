/* The proof drawer: shows an evidence record (rule, precedent, experiment…) or a repository file in place.
   Any element with data-record="DCLAB-R04" or data-preview="research/…/README.md" opens it. */
import {$, esc, api} from './core.js';

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const TYPE_LABEL = {rule: 'Rule', workflow: 'Workflow block', dataset: 'Dataset card', experiment: 'Measured experiment',
  leakage_precedent: 'Leakage precedent', pitfall: 'Measured pitfall', finding: 'Cross-campaign finding'};

function show(title, meta) {
  $('preview').classList.remove('hidden');
  $('preview-title').textContent = title;
  $('preview-meta').textContent = meta || '';
  $('preview-body').innerHTML = '<p class="muted small">Loading…</p>';
}
function close() { $('preview').classList.add('hidden'); }

export const fileChip = (path, label) => `<button type="button" class="file-chip ${/\.[a-z0-9]+$/i.test(path) ? '' : 'folder'}" data-preview="${esc(path)}" title="${esc(path)}">${esc(label ?? path.split('/').filter(Boolean).pop())}</button>`;
export const recordChip = (id) => `<button type="button" class="proof-chip" data-record="${esc(id)}" title="Show proof: ${esc(id)}">${esc(id)}</button>`;

export async function openFile(path) {
  path = path.replace(/\/$/, '');
  show(path, 'repository file');
  try {
    const f = await api('/research/file?path=' + encodeURIComponent(path));
    $('preview-meta').textContent = f.kind === 'folder' ? `folder · ${plural(f.files, 'tracked file')}` : f.kind + (f.truncated ? ' · first 200 KB' : '');
    $('preview-body').innerHTML = f.kind === 'folder'
      ? `<div class="chips column">${f.entries.map(e => fileChip(e, e.slice(f.path.length + 1))).join('')}</div>`
      : `<pre>${esc(f.text)}</pre>`;
  } catch (error) {
    $('preview-meta').textContent = '';
    $('preview-body').innerHTML = `<p class="map-empty">${esc(error.message)}. Open it from the repository instead.</p>`;
  }
}

export async function openRecord(id) {
  show(id, 'evidence record');
  try {
    const r = await api('/evidence/' + encodeURIComponent(id));
    const meta = Object.entries(r.metadata || {}).filter(([, v]) => v !== null && v !== '' && !(Array.isArray(v) && !v.length));
    const files = (r.citations || []).filter(Boolean);
    $('preview-meta').textContent = TYPE_LABEL[r.type] || r.type;
    $('preview-body').innerHTML = `
      <div class="record"><span class="record-type ${esc(r.type)}">${esc(TYPE_LABEL[r.type] || r.type)}</span><h3>${esc(r.title)}</h3>
      <p class="record-text">${esc(r.text)}</p>
      ${meta.length ? `<table class="record-meta"><tbody>${meta.map(([k, v]) => `<tr><th>${esc(k.replaceAll('_', ' '))}</th><td>${esc(Array.isArray(v) ? v.join(', ') : typeof v === 'number' ? v.toFixed(4).replace(/\.?0+$/, '') : v)}</td></tr>`).join('')}</tbody></table>` : ''}
      ${files.length ? `<h4>Where this comes from</h4><div class="chips column">${files.map(f => /^(research|evidence\/campaigns|docs)\//.test(f) ? fileChip(f, f) : `<code class="path">${esc(f)}</code>`).join('')}</div>` : ''}
      </div>`;
  } catch (error) {
    $('preview-meta').textContent = '';
    $('preview-body').innerHTML = `<p class="map-empty">${esc(error.message)}</p>`;
  }
}

export function initDrawer() {
  document.addEventListener('click', e => {
    const file = e.target.closest('[data-preview]');
    if (file) { e.preventDefault(); openFile(file.dataset.preview); return; }
    const record = e.target.closest('[data-record]');
    if (record) { e.preventDefault(); openRecord(record.dataset.record); }
  });
  $('preview-close').addEventListener('click', close);
  $('preview').addEventListener('click', e => { if (e.target === $('preview')) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
}
