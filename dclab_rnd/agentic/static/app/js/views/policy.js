DC.view('policy', {
  init(el) {
    const { $, esc, chip, charts, codeBlock, toast } = DC;
    const S = window.DEMO_SFT || {};
    const by = (S.manifest && S.manifest.by_task) || {};
    const names = { critique_claim: 'critique a claim', grounded_qa: 'grounded Q&A', explain_experiment: 'explain an experiment', apply_selection_rule: 'apply a selection rule', leakage_judgment: 'leakage judgment', rule_reasoning: 'rule reasoning', workflow_steps: 'workflow steps' };
    $('#sft-chart', el).innerHTML = charts.barsH(Object.entries(by).sort((a, b) => b[1] - a[1]).map(([k, v]) => ({ label: names[k] || k, value: v, valueText: String(v), cls: k === 'leakage_judgment' || k === 'workflow_steps' ? 'proof' : '' })),
      { width: 360, labelW: 140, valW: 30, min: 0, max: 80, rowH: 24, ticks: [0, 40, 80], tickFmt: v => v, aria: 'Examples per task' });
    let full = false;
    function ex() {
      const msgs = (S.example && S.example.messages) || [];
      $('#sft-example', el).innerHTML = msgs.map(m => {
        const t = full ? m.content : m.content.slice(0, m.role === 'assistant' ? 700 : 360) + (m.content.length > (m.role === 'assistant' ? 700 : 360) ? '…' : '');
        return `<div class="inset small"><div class="eyebrow ${m.role === 'assistant' ? 'accent' : ''}" data-style="margin-bottom:4px">${esc(m.role)}</div><div data-style="white-space:pre-wrap;overflow-wrap:anywhere">${DC.linkIds(t)}</div></div>`;
      }).join('') || '<div class="empty">Example not available.</div>';
      $('#sft-more', el).textContent = full ? 'Show less' : 'Show the full example';
      DC.hydrate(el);
    }
    $('#sft-more', el).addEventListener('click', () => { full = !full; ex(); });
    ex();
    $('#traj-json', el).replaceWith(Object.assign(document.createElement('div'), { innerHTML: codeBlock(`{
  "project": "term-deposit-calls", "move": 15, "solution": "v2",
  "state": {"node": "WF-07", "done": ["WF-01", "WF-02", "WF-03", "WF-04", "WF-05", "WF-06"],
            "holdout": "sealed", "evidence": ["EXP-009"]},
  "proposal": {"by": "intern", "move": "open_holdout",
               "reason": "compare extra_trees with lightgbm"},
  "validator": {"valid": false, "edge": "WF-07 -> WF-09 does not exist",
                "rules": ["DCLAB-R17"], "precedent": "PIT-006"},
  "next": {"move": "proceed", "node": "WF-08", "tool": "tune", "result": "kept C02"}
}`) }).firstChild);
    const RQ = [
      ['HyperAck · WF-05', 'Intern kept deliverey_category_id as an input after the auditor flagged it.', 'Strong signal, plausibly known at order time. Was keeping it with a flag right?'],
      ['Card fraud · WF-08', 'Intern rejected tuning: gain −0.0048 against a required +0.0050.', 'Was the margin right for PR-AUC on 0.17% positives?'],
      ['Telco churn · WF-07', 'Intern proposed reopening the solution after the critic\'s note.', 'Should it have asked the owner first?'],
    ];
    let left = RQ.length;
    $('#review-queue', el).innerHTML = RQ.map(([where, what, q], i) => `<div class="list-item" data-rq="${i}"><div class="li-main"><span class="li-sub">${esc(where)}</span><span class="li-title">${esc(what)}</span><span class="small muted">${esc(q)}</span></div><div class="li-side"><div class="decide"><button type="button" data-rv="right">Right</button><button type="button" data-rv="edit">Edit</button><button type="button" data-rv="wrong">Wrong</button></div></div></div>`).join('');
    $('#review-queue', el).addEventListener('click', e => {
      const b = e.target.closest('[data-rv]'); if (!b) return;
      const item = b.closest('[data-rq]'); if (item.dataset.done) return;
      item.dataset.done = '1'; item.style.opacity = '.55'; b.setAttribute('aria-pressed', 'true');
      left--; $('#rq-count', el).textContent = left ? `${left} to review` : 'all reviewed'; $('#rq-count', el).className = 'pill ' + (left ? 'warn' : 'ok');
      toast(b.dataset.rv === 'edit' ? 'Opened for editing. Your corrected move becomes the training target.' : 'Label saved. It becomes training data and benchmark ground truth.');
    });
  },
});
