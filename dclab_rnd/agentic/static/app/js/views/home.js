DC.view('home', {
  init(el) {
    const { $, $$, esc, chip, charts } = DC;
    // fill inline tokens like ${chip:ID} and ${spark}
    el.innerHTML = el.innerHTML
      .replace(/\$\{chip:([A-Z0-9_-]+)\}/g, (m, id) => chip(id))
      .replace('${spark}', charts.spark([2.1, 3.4, 4.0, 6.2, 7.9, 9.4, 12.6, 14.1, 16.0, 18.4], { width: 110, height: 22 }));
    const tbody = $('#home-projects tbody', el);
    const segClass = { d: 'd', c: 'c', b: 'b', '-': '' };
    function rows(filter) {
      tbody.innerHTML = DEMO.projects.filter(p => filter === 'all' || p.filter === filter).map(p => `
        <tr data-open="${esc(p.open || '')}" data-name="${esc(p.name)}">
          <td><div class="cell-main">${esc(p.name)}${p.sample ? ' <span class="pill outline">sample</span>' : ''}</div><div class="cell-sub">${esc(p.dataset)} · ${esc(p.pack)}</div></td>
          <td><span class="wf-mini" aria-label="${esc(p.at)}">${p.states.split('').map(s => `<i class="${segClass[s] || ''}"></i>`).join('')}</span><div class="cell-sub">${esc(p.at)}</div></td>
          <td><div class="score">${esc(p.score)}${p.ci ? ` <span class="ci">[${esc(p.ci)}]</span>` : ''}</div><div class="cell-sub">${esc(p.note)} ${p.ev ? chip(p.ev) : ''}</div></td>
          <td class="num">${p.leaks}</td>
          <td><span class="pill ${esc(p.cls)}">${esc(p.status)}</span><div class="cell-sub row" data-style="margin-top:4px"><span class="avatar sm ${p.owner === 'AV' ? 'b' : ''}">${esc(p.owner)}</span>${esc(p.updated)}</div></td>
        </tr>`).join('');
    }
    rows('all');
    $('#proj-filter', el).addEventListener('segchange', e => rows(e.detail));
    tbody.addEventListener('click', e => {
      if (e.target.closest('[data-record]')) return;
      const tr = e.target.closest('tr'); if (!tr) return;
      const open = tr.dataset.open;
      if (open === 'project') { DC.state.project = 'bank'; location.hash = 'project'; }
      else if (open === 'hyperack' || open === 'telco') { DC.state.project = open; location.hash = 'project'; }
      else DC.toast(`In the demo, ${tr.dataset.name} opens as a summary only. Its numbers come from the cited record.`);
    });
    el.addEventListener('click', e => {
      const ex = e.target.closest('[data-example]');
      if (ex) { $('#home-task', el).value = ex.dataset.example; $('#home-task', el).focus(); }
      const link = e.target.closest('[data-project]');
      if (link) DC.state.project = link.dataset.project;
    });
    $('#home-start', el).addEventListener('click', () => {
      const mode = $('#home-mode button[aria-pressed="true"]', el).dataset.v;
      if (mode === 'intern') { DC.toast('Handed to the intern. It drafts the contract and asks before anything irreversible.'); location.hash = 'intern'; }
      else { DC.toast('Contract draft ready. Check the goal, then the data.'); location.hash = 'new'; }
    });
  },
});
