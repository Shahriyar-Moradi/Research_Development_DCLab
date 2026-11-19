DC.view('decision', {
  async enter(el) {
    const { $, esc } = DC;
    const box = $('#dc-box', el);
    const prices = $('#dc-prices', el);
    prices.hidden = true;
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { p = null; }
    DC.markSample(!p);
    if (!p) {
      $('#dc-eyebrow', el).textContent = 'Sample · decision';
      DC.states.empty(box, 'No project is open, so there is no decision to show. Start one on Home and say what is done with each prediction, what a case is worth and what acting costs.', { href: '#home', action: 'Go to Home' });
      return;
    }
    $('#dc-eyebrow', el).textContent = `${p.name} · decision`;
    prices.hidden = false;
    const form = $('#dc-price-form', el);
    form.reset();
    const priceResult = form.querySelector('[data-result]');
    priceResult.replaceChildren();
    let activeRequest;
    const current = token => activeRequest === token && form.dataset.project === p.id;
    form.dataset.project = p.id;
    async function comparePrices(mode) {
      const token = activeRequest = {};
      const fields = new FormData(form);
      const val = key => Number(fields.get(key));
      const query = {target_revenue: val('target'), independent: fields.has('independent')};
      DC.states.loading(priceResult, { text: 'Comparing prices…' });
      try {
        let result;
        if (mode === 'assume') {
          result = await DC.client.priceAssumptions(p.id, {body: {price: val('price'), audience: val('audience'),
            target_revenue: query.target_revenue, outcome: fields.get('outcome'), independent: query.independent,
            probabilities: ['base','five','ten'].map(k => fields.get(k) === '' ? null : val(k)/100)}});
        } else {
          if (mode === 'prepare') await DC.client.preparePriceStudy(p.id, {body: {base_price: val('price'),
            horizon_days: val('horizon'), outcome: fields.get('outcome'), design: fields.get('design'), design_basis: fields.get('basis')}});
          result = await DC.client.getPriceStudy(p.id, {query});
        }
        if (!current(token)) return;
        const rows = result.rows.map(r => `<tr><td>${num(r.price)}</td><td>${num(r.break_even_units)}</td><td>${num(r.expected_events)}</td><td>${num(r.expected_revenue)}</td><td>${pct(r.downside_probability)}</td></tr>`).join('');
        const effects = (result.effects || []).map((e,i) => e ? `+${i ? 10 : 5}%: ${num(e.change_pp)} pp (95% interval ${e.interval95_pp.map(x => num(x)).join(' to ')})` : 'Insufficient support').join('; ');
        priceResult.innerHTML = `<p>${esc(result.source)} · ${num(result.rows[0].audience,0)} people · ${esc(result.outcome)} ${esc(result.currency || '')}</p>
          <div class="table-wrap"><table class="data"><thead><tr><th>Price</th><th>Break-even units</th><th>Expected events</th><th>Revenue</th><th>Risk below target</th></tr></thead><tbody>${rows}</tbody></table></div>
          <p>${esc([result.reason, result.effect_source, result.design_basis, effects].filter(Boolean).join(' · '))}</p>`;
      } catch (e) { if (current(token)) DC.states.failed(priceResult, e, {lead: 'Price comparison unavailable: '}); }
    }
    form.onsubmit = e => { e.preventDefault(); comparePrices('assume'); };
    form.querySelector('[data-prepare]').onclick = () => comparePrices('prepare');
    form.querySelector('[data-score]').onclick = () => comparePrices('score');
    form.querySelector('[data-upload]').onclick = async () => {
      const file = form.elements.upload.files[0];
      if (!file) { priceResult.textContent = 'Choose a table file first.'; return; }
      const token = activeRequest = {};
      const name = form.elements.table.value;
      try {
        await DC.client.uploadTable(p.id, name, {query:{filename:file.name},body:file});
        if (current(token)) priceResult.textContent = 'Attached ' + name + '.';
      } catch (e) { if (current(token)) DC.states.failed(priceResult, e, {lead:'Could not attach table: '}); }
    };
    const LABELS = {
      value_per_case: 'Value of a case', cost_per_action: 'Cost of acting on a row', cost_false_positive: 'Cost of a false alarm',
      cost_false_negative: 'Cost of a miss', capacity_share: 'Capacity (share of rows)', base_rate: 'Share of rows that are cases',
      value_scale: 'Every row’s value times',
    };
    const num = (v, d = 4) => (v == null || Number.isNaN(+v)) ? '—' : (+v).toLocaleString(undefined, { maximumFractionDigits: d });
    const pct = v => v == null ? '—' : `${(+v * 100).toFixed(2)}%`;

    async function load() {
      DC.states.loading(box, { text: 'Reading the decision…' });
      let data;
      try { data = await DC.client.getDecision(p.id); }
      catch (e) { DC.states.failed(box, e, { lead: 'Could not read the decision: ', retry: load }); return; }
      render(data);
    }

    function render(data) {
      if (data.status === 'none') {
        const h = data.heard;
        if (h) {
          const said = ['value_per_case', 'cost_per_action', 'cost_false_positive', 'cost_false_negative'].filter(k => h[k]).map(k => `${LABELS[k]}: ${num(h[k])}`).join(' · ');
          box.innerHTML = `<div class="panel"><div class="panel-body stack tight"><div class="eyebrow">Heard, not added</div><p>Home heard these amounts${h.what ? ` for “${esc(h.what)}”` : ''}: ${esc(said)}. They came from an answer that can also hold counts and capacities, so they were not added on their own. If they are right, use them: the final stage then chooses on how many rows to act.</p><p class="small muted">${esc(h.notes || '')}</p><div><button class="btn primary" id="dc-use" type="button">Use these amounts</button></div><p class="small" id="dc-use-note" hidden></p></div></div>`;
          wireUse(h);
          return;
        }
        DC.states.empty(box, 'This project states no decision yet. In the solution, say what is done to a row (call, review, send an offer), what a case is worth and what acting costs; the final stage then chooses on how many rows to act.', { href: '#solution', action: 'Open the solution' });
        return;
      }
      if (data.status === 'pending' || data.status === 'missing') {
        DC.states.empty(box, 'The decision is chosen by the final stage, on the selected model’s out-of-fold scores. Run it to see the choice here.', { href: '#project', action: 'Open the workflow' });
        return;
      }
      const c = data.chosen || {};
      if (data.status === 'refused') {
        box.innerHTML = `<div class="panel"><div class="panel-body"><div class="eyebrow">No decision was drawn</div><p>${esc(c.warning || 'The decision could not be made from this data.')}</p></div></div>`;
        return;
      }
      let html = `<div class="panel"><div class="panel-body stack tight"><div class="eyebrow">The decision</div><p>${esc(data.sentence || '')}</p></div></div>`;
      if (c.decision === 'operating_point') html += operatingPoint(c, data);
      else if (c.decision === 'cost_matrix') html += costMatrix(c);
      else if (c.decision === 'price' && c.curve?.length) {
        const points = c.curve.map(r => `<tr><td>${num(r.multiple)}×</td><td>${pct(r.acceptance)}</td><td>${num(r.expected_margin_per_offer)}</td></tr>`).join('');
        html += `<details class="panel"><summary>Recorded price response · ${esc(c.side)}</summary><table class="data"><thead><tr><th>Relative price</th><th>Acceptance</th><th>Margin per offer</th></tr></thead><tbody>${points}</tbody></table><p>${esc(c.assumption || '')}</p></details>`;
      }
      else html += `<div class="panel"><div class="panel-body small muted">A review budget or a price is shown in the brief; a what-if is asked of a decision stated with its values and costs.</div></div>`;
      box.innerHTML = html;
      wire(data);
    }

    function operatingPoint(c, data) {
      const iv = c.expected_value_per_row_interval || [];
      const h = c.holdout || {};
      const pi = h.prediction_interval || [];
      let s = `<div class="kpi-strip panel flat">
        <div class="kpi"><span class="label">Act on</span><span class="value">${pct(c.share)}</span><span class="delta">${num(c.actions_training, 0)} training rows${c.capacity_binds ? ' · the capacity binds' : ''}${c.actions_per_day != null ? ` · ${num(c.actions_per_day, 1)} a day` : ''}</span></div>
        <div class="kpi"><span class="label">Worth per row</span><span class="value">${num(c.expected_value_per_row)}</span><span class="delta">90% interval ${num(iv[0])} to ${num(iv[1])}</span></div>
        <div class="kpi"><span class="label">Holdout, read once</span><span class="value">${num(h.expected_value_per_row)}</span><span class="delta">${h.actions != null ? `${num(h.actions, 0)} acted on, ${num(h.cases_found, 0)} cases · ` : ''}expected ${num(pi[0])} to ${num(pi[1])}</span></div>
      </div>`;
      const curve = (c.curve || []).filter(q => q.share != null && q.expected_value_per_row != null);
      if (curve.length > 1) {
        const chart = DC.charts.line([{ pts: curve.map(q => [q.share, q.expected_value_per_row]), dots: true }],
          { xMin: 0, xMax: 1, xFmt: v => `${Math.round(v * 100)}%`, xLabel: 'share of rows acted on', yLabel: 'value per row',
            markers: [{ x: c.share, label: 'chosen' }], aria: 'expected value per row by the share acted on' });
        s += `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>Value by how many rows are acted on</h2><span class="sub">On the out-of-fold scores of the training rows, ranked by ${esc(c.ranked_by || 'score')}; the chosen share is marked.</span></div></div><div class="panel-body" id="dc-curve">${chart}</div></div>`;
      }
      const sens = (c.sensitivity || {}).rows || [];
      if (sens.length) {
        const mults = (c.sensitivity || {}).multipliers || [];
        const head = mults.map(m => `<th class="num">×${num(m, 2)}</th>`).join('');
        const rows = sens.map(r => `<tr><td>${esc(LABELS[r.term] || r.term)}</td>${(r.at || []).map(a => `<td class="num">${a.share == null ? '—' : pct(a.share)}</td>`).join('')}<td>${r.moves_at == null ? 'never in this range' : `at ×${num(r.moves_at, 2)}`}</td></tr>`).join('');
        s += `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>What moves the decision</h2><span class="sub">The share acted on when each stated number is multiplied; nothing is refitted. “Moves” means the share changes by more than a fifth.</span></div></div>
          <div class="table-wrap"><table class="data compact"><thead><tr><th>Number</th>${head}<th>Moves</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
      }
      if ((data.what_if || {}).available) {
        const spec = c.spec || {};
        const fields = data.what_if.terms.map(t => {
          const now = t === 'value_scale' ? 1 : t === 'base_rate' ? '' : (spec[t] != null ? spec[t] : '');
          return `<div class="field"><label for="dc-in-${t}">${esc(LABELS[t] || t)}</label><input type="number" step="any" min="0" id="dc-in-${t}" data-term="${t}" data-now="${esc(String(now))}" value="${esc(String(now))}" placeholder="${t === 'base_rate' ? 'as in the data' : ''}"></div>`;
        }).join('');
        s += `<div class="panel"><div class="panel-head"><div class="panel-title"><h2>What if</h2><span class="sub">Change one or more numbers and ask: the decision is chosen again on the same rows. The model, the stages and the holdout are untouched.</span></div></div>
          <div class="panel-body stack"><div class="grid g3">${fields}</div><div><button class="btn primary" id="dc-ask" type="button">Ask what if</button></div><div id="dc-answer"></div></div></div>`;
      } else if ((data.what_if || {}).why_not) {
        s += `<div class="panel"><div class="panel-body small muted">${esc(data.what_if.why_not)}</div></div>`;
      }
      return s;
    }

    function costMatrix(c) {
      const h = c.holdout || {};
      const counts = Object.entries(c.chosen_counts || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num">${num(v, 0)}</td></tr>`).join('');
      return `<div class="kpi-strip panel flat">
          <div class="kpi"><span class="label">Cost per row, lowest expected cost</span><span class="value">${num(h.cost_per_row)}</span><span class="delta">on ${num(h.rows, 0)} holdout rows</span></div>
          <div class="kpi"><span class="label">Cost per row, likeliest class</span><span class="value">${num(h.cost_per_row_likeliest_class)}</span><span class="delta">the usual choice, for comparison</span></div>
        </div>
        <div class="panel"><div class="panel-head"><div class="panel-title"><h2>Rows per class chosen</h2></div></div><div class="table-wrap"><table class="data compact"><thead><tr><th>Class</th><th class="num">Holdout rows</th></tr></thead><tbody>${counts}</tbody></table></div></div>`;
    }

    function wireUse(h) {
      const button = $('#dc-use', box), note = $('#dc-use-note', box);
      let armed = false;
      button.addEventListener('click', async () => {
        if (!armed) {  // a second click saves: the person sees what saving does first
          armed = true;
          note.hidden = false;
          note.textContent = 'Saving changes the solution, so the stages already run are cleared and run again; the final stage then reads the held-back cases again, with your reason. Click again to save.';
          button.textContent = 'Save the decision';
          return;
        }
        button.disabled = true;
        let project;
        try { project = await DC.client.getProject(p.id); }
        catch (e) { DC.states.failed(note, e, { lead: 'Could not read the solution: ' }); button.disabled = false; return; }
        const decision = { action: 'act', what: h.what || 'act', notes: h.notes || '' };
        ['value_per_case', 'cost_per_action', 'cost_false_positive', 'cost_false_negative'].forEach(k => { if (h[k]) decision[k] = h[k]; });
        try { await DC.client.saveSolution(p.id, { body: { ...(project.solution || {}), decision } }); }
        catch (e) { DC.states.failed(note, e, { lead: 'Could not save the decision: ' }); button.disabled = false; return; }
        note.textContent = 'Saved. Run the stages again on the Workflow page to choose on how many rows to act.';
      });
    }

    function wire(data) {
      const ask = $('#dc-ask', box);
      if (!ask) return;
      ask.addEventListener('click', async () => {
        const changes = {};
        box.querySelectorAll('input[data-term]').forEach(inp => {
          if (inp.value === '' || inp.value === inp.dataset.now) return;
          const v = Number(inp.value);
          if (!Number.isNaN(v)) changes[inp.dataset.term] = v;
        });
        const out = $('#dc-answer', box);
        if (!Object.keys(changes).length) { out.innerHTML = '<p class="small muted">Change a number first.</p>'; return; }
        DC.states.loading(out, { text: 'Choosing again…' });
        let r;
        try { r = await DC.client.decisionWhatIf(p.id, { body: { changes } }); }
        catch (e) { DC.states.failed(out, e, { lead: 'Could not answer the what-if: ' }); return; }
        const keep = r.keep_the_old_threshold || {};
        out.innerHTML = `<div class="table-wrap"><table class="data compact"><thead><tr><th></th><th class="num">Share acted on</th><th class="num">Value per row</th></tr></thead><tbody>
            <tr><td>As chosen</td><td class="num">${pct(r.before.share)}</td><td class="num">${num(r.before.value_per_row)}</td></tr>
            <tr><td>Chosen again under the change</td><td class="num">${pct(r.after.share)}</td><td class="num">${num(r.after.value_per_row)}</td></tr>
            <tr><td>The old decision kept${keep.cut_to_capacity ? ' (cut to the new capacity)' : ''}</td><td class="num">${pct(keep.share)}</td><td class="num">${num(keep.value_per_row)}</td></tr>
          </tbody></table></div>
          <p class="small">Not changing would cost <b>${num(r.regret_if_unchanged)}</b> per row. ${esc(r.note || '')}</p>`;
      });
    }

    await load();
  },
});
