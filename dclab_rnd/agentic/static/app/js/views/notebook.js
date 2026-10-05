DC.view('notebook', {
  init(el) {
    const { $, $$, esc, chip, chips, icon, codeBlock, toast } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const SAMPLE_R = window.DEMO_REVIEW || { cells: [], findings: [], summary: {} };
    let R = SAMPLE_R;
    let real = null;  // {p, review} when a real project is open
    const FIX = {
      3: `# duration is known only after the call ends: forbidden by solution v2
features = ["age", "balance", "campaign", "pdays", "previous", "job", "month"]
# job's subscription rate is learned inside the training folds (cell 5)`,
      4: `# keep the natural class balance (11.7% positive);
# the model weights the rare class instead of copying rows`,
      5: `from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df["target"], test_size=0.2, stratify=df["target"], random_state=42)
prep = ColumnTransformer([("job", TargetEncoder(random_state=42), ["job"])],
                         remainder=StandardScaler())`,
      6: `from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
cv = StratifiedKFold(3, shuffle=True, random_state=42)
def forest(depth):
    return Pipeline([("prep", prep), ("rf", RandomForestClassifier(
        n_estimators=200, max_depth=depth, class_weight="balanced", random_state=42))])
scores = {d: cross_val_score(forest(d), X_train, y_train, cv=cv, scoring="roc_auc").mean()
          for d in [4, 8, 16, None]}
best_depth = max(scores, key=scores.get)
p = forest(best_depth).fit(X_train, y_train).predict_proba(X_test)[:, 1]  # test set used once
print(best_depth, *(round(f(y_test, p), 4) for f in (roc_auc_score, average_precision_score, brier_score_loss)))`,
    };
    const FIXES_FINDING = { known_leakage_column: 3, aggregate_before_split: 3, resample_before_split: 4, preprocess_before_split: 5, missing_random_state: null, unstratified_split: 5, holdout_reuse: 6, accuracy_only: 6 };
    const OUT = { 2: 'X: 45,211 rows × 16 columns · y: 11.7% positive', 6: 'None 0.9689398209029996' };
    let fixed = new Set();
    function isFixed(f) {
      if (real) return false;
      if (f.detector === 'single_model_family') return false;
      const c = f.detector === 'missing_random_state' ? f.cell : FIXES_FINDING[f.detector];
      return c != null && fixed.has(c);
    }
    const SEVS = ['high', 'medium', 'low', 'info'];
    const known = ids => (ids || []).filter((x, i, a) => DC.REC[x] && a.indexOf(x) === i);
    const codeNo = i => R.cells.slice(0, i + 1).filter(x => x.type === 'code').length;
    /* Markdown cells of the exported notebook: headings, paragraphs, bullet lists, **bold**, *italic*, `code`, record IDs. */
    function md(src) {
      const inline = t => esc(t).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>').replace(/(^|[^*\w])\*([^*\s][^*]*?)\*/g, '$1<i>$2</i>')
        .replace(/\b(DCLAB-R\d{2}|WF-\d{2}|EXP-\d{3}|PIT-\d{3}|LEAK-[a-z_]+|FINDING-[a-z-]+|DATASET-[a-z_]+)\b/g, m => (DC.REC[m] ? chip(m) : m));
      return '<div class="stack tight">' + String(src || '').trim().split(/\n{2,}/).map(block => {
        const h = block.match(/^#{1,4}\s+(.*)$/);
        if (h && !block.includes('\n')) return `<h3>${inline(h[1])}</h3>`;
        const lines = block.split('\n');
        if (lines.every(l => /^\s*- /.test(l))) return `<ul>${lines.map(l => `<li>${inline(l.replace(/^\s*- /, ''))}</li>`).join('')}</ul>`;
        return `<p>${lines.map(inline).join('<br>')}</p>`;
      }).join('') + '</div>';
    }
    const cellLabel = (c, i) => c.type === 'markdown'
      ? ((String(c.source).match(/^#{1,4}\s+(.*)$/m) || [])[1] || 'Notes').replace(/[`*]/g, '').slice(0, 30)
      : (String(c.source).split('\n')[0].replace(/^#\s*/, '').slice(0, 26) || 'cell ' + i);

    function render() {
      const open = R.findings.filter(f => !isFixed(f));
      const sev = s => open.filter(f => f.severity === s).length;
      $('#nb-count', el).textContent = open.length;
      $('#nb-count', el).className = 'pill ' + (sev('high') ? 'bad' : open.length ? 'warn' : 'ok');
      $('#sev-counts', el).innerHTML = SEVS.map(s => sev(s) ? `<span class="sev ${s}">${sev(s)} ${s}</span>` : '').join('');
      const banner = $('#nb-banner', el);
      if (real) renderRealBanner(banner, sev);
      else if (fixed.size < 4) { banner.className = 'callout bad'; $('#nb-banner-text', el).innerHTML = `<b>${sev('high')} findings would change the reported score.</b> The last cell prints accuracy 0.9689, but the duration leak, the target-mean feature and copies of the same rows in train and test inflate it. Fix the high findings first.`; }
      else { banner.className = 'callout ok'; $('#nb-banner-text', el).innerHTML = `<b>The honest score: ROC-AUC 0.7577, average precision 0.3454</b> on 9,043 untouched test rows, after the fixes. The notebook first reported accuracy 0.9689. The R&D's own campaign landed in the same range ${chip('EXP-007')} ${chip('EXP-010')}. One finding is still open: only one model family was tried.`; }
      $('#nb-cells', el).innerHTML = R.cells.map((c, i) => {
        if (c.type === 'markdown') return `<div class="cell" id="nbc-${i}"><div class="cell-gutter"></div><div class="cell-md">${real ? md(c.source) : '<h3>Bank marketing: will the client subscribe?</h3><p>A typical first-pass notebook from a teammate. It runs, and the score looks great.</p>'}</div></div>`;
        const fs = R.findings.filter(f => f.cell === i);
        const openF = fs.filter(f => !isFixed(f));
        const worst = SEVS.find(s => openF.some(f => f.severity === s));
        const src = fixed.has(i) ? FIX[i] : c.source.replace(/\n+$/, '');
        const hl = fixed.has(i) ? [] : [...new Set(openF.map(f => f.line))];
        const okLines = fixed.has(i) ? src.split('\n').map((_, k) => k + 1) : [];
        let out = real ? '' : OUT[i] || '';
        if (!real && i === 6 && fixed.has(6)) out = '8 0.7577 0.3454 0.1866';
        return `<div class="cell" id="nbc-${i}">
          <div class="cell-gutter">[${codeNo(i)}]${worst ? `<span class="cell-flag ${worst === 'info' ? 'low' : worst}" title="${openF.length} open findings">${openF.length}</span>` : `<span class="cell-flag ok">✓</span>`}</div>
          <div class="cell-body">
            ${codeBlock(src, { hl, ok: okLines })}
            ${out ? `<div class="cell-out">${esc(out)}${i === 6 && !fixed.has(6) ? '  <span class="pill bad">inflated</span>' : i === 6 ? '  <span class="pill ok">honest · test used once</span>' : ''}</div>` : ''}
            ${openF.length ? `<div class="stack tight">${openF.map(f => `<div class="cell-tools"><span class="sev ${f.severity}">${f.severity}</span><span>${esc(f.title)}</span><span class="faint">line ${f.line}</span></div>`).join('')}${!real && FIX[i] && !fixed.has(i) ? `<div><button type="button" class="btn sm proof" data-fix="${i}" data-f="nb.fix">Apply fix to this cell</button></div>` : ''}</div>` : ''}
            ${fixed.has(i) ? `<div class="cell-tools"><span class="pill ok">fixed by the companion</span><button type="button" class="link-btn small" data-undo="${i}">Undo</button></div>` : ''}
          </div></div>`;
      }).join('');
      $('#findings', el).innerHTML = real ? realFindings() : R.findings.map((f, i) => {
        const done = isFixed(f);
        return `<div class="finding ${done ? 'fixed' : f.severity}">
          <div class="spread"><span class="row"><span class="sev ${f.severity}">${f.severity}</span><span class="tag">cell ${f.cell} · line ${f.line}</span></span>${done ? '<span class="pill ok">fixed</span>' : ''}</div>
          <div class="f-title">${esc(f.title)}</div>
          ${done ? '' : `<div class="f-msg">${esc(f.message)}</div><div class="f-fix"><b>Fix.</b> ${esc(f.suggestion)}</div>`}
          <div class="row">${chips(f.proof)}</div>
          ${done ? '' : `<div class="row">${FIX[f.cell] && f.detector !== 'single_model_family' ? `<button type="button" class="btn sm proof" data-fix="${f.detector === 'missing_random_state' ? f.cell : (FIXES_FINDING[f.detector] ?? f.cell)}">Apply fix</button>` : f.detector === 'single_model_family' ? '<a class="btn sm" href="#models">Open the model screen</a>' : ''}<button type="button" class="btn sm ghost" data-jump="${f.cell}">Show cell</button><button type="button" class="btn sm ghost" data-dismiss="${i}">Dismiss…</button></div>`}
        </div>`;
      }).join('') + (!real && fixed.has(6) ? `<div class="finding medium"><div class="spread"><span class="row"><span class="sev medium">medium</span><span class="tag">cell 6 · output</span></span><span class="pill info">new after the run</span></div>
          <div class="f-title">Probabilities are distorted by class_weight="balanced"</div>
          <div class="f-msg">Brier score 0.1866 on the test rows. The R&amp;D's calibrated model scored 0.0870. The cost calculator needs honest probabilities.</div>
          <div class="f-fix"><b>Fix.</b> Calibrate on the training folds (CalibratedClassifierCV) or drop the class weights and pick the cut-off from the costs.</div>
          <div class="row">${chips(['DCLAB-R18', 'EXP-010'])}</div></div>` : '');
      $('#nb-outline', el).innerHTML = R.cells.map((c, i) => { const fs = R.findings.filter(f => f.cell === i && !isFixed(f)); const label = real ? cellLabel(c, i) : c.type === 'markdown' ? 'Title' : (c.source.split('\n')[0].replace(/^#\s*/, '').slice(0, 26) || 'cell ' + i); return `<button type="button" class="row" data-jump="${i}" data-style="justify-content:space-between;width:100%;text-align:left"><span class="mono xs">${esc(label)}</span>${fs.length ? `<span class="cell-flag ${fs.some(f => f.severity === 'high') ? 'high' : !real || fs.some(f => f.severity === 'medium') ? 'medium' : 'low'}" data-style="width:18px;height:18px;font-size:10px">${fs.length}</span>` : ''}</button>`; }).join('');
      $('#nb-vars', el).hidden = !!real; $('#nb-vars-empty', el).hidden = !real;
      if (real) $('#nb-vars-empty', el).textContent = 'No variables: this notebook was generated from the project, not run here. Download it and run it to see them. The numbers on this page come from the stage engine.';
      else $('#nb-vars', el).innerHTML = fixed.has(5)
        ? '<dt>df</dt><dd>45,211 × 17</dd><dt>X_train</dt><dd>36,168 × 7</dd><dt>X_test</dt><dd>9,043 × 7 · untouched</dd><dt>prep</dt><dd>ColumnTransformer</dd>'
        : '<dt>df</dt><dd>45,211 × 18</dd><dt>df_bal</dt><dd>79,844 × 18 <span class="pill bad">copies</span></dd><dt>X_all</dt><dd>79,844 × 9</dd><dt>features</dt><dd>9 incl. duration</dd>';
      DC.hydrate(el);
    }
    function applyFix(i) { fixed.add(Number(i)); render(); toast(`Fix applied to cell ${i}. The original stays in the cell history.`); }
    el.addEventListener('click', e => {
      const fr = e.target.closest('[data-fix-real]'); if (fr) { toast('Applying fixes to the exported notebook is not built yet. Download the .ipynb and follow the fix above; this review updates when the project changes.', { ok: false, ms: 5200 }); return; }
      const f = e.target.closest('[data-fix]'); if (f) { applyFix(f.dataset.fix); return; }
      const u = e.target.closest('[data-undo]'); if (u) { fixed.delete(Number(u.dataset.undo)); render(); return; }
      const j = e.target.closest('[data-jump]'); if (j) { const c = $('#nbc-' + j.dataset.jump, el); if (c) { DC.reveal(c); c.scrollIntoView({ behavior: 'smooth', block: 'center' }); c.classList.add('focus'); setTimeout(() => c.classList.remove('focus'), 1400); } return; }
      const d = e.target.closest('[data-dismiss]');
      if (d) {
        DC.modal.open({ eyebrow: '<span class="eyebrow">Dismiss a finding</span>', title: 'Why is this not a problem here?', confirm: 'Dismiss with reason',
          html: '<p>A dismissal is logged with your reason and becomes a training example for the companion, right or wrong.</p><div class="field"><label for="dismiss-why">Reason</label><textarea id="dismiss-why">The test set here is a quick sanity check; the real evaluation uses the project holdout.</textarea></div>',
          onConfirm: () => toast('Dismissed and logged. The reviewer can reopen it.') });
        return;
      }
      const rq = e.target.closest('[data-rq]');
      if (rq && real) { askReal(rq.dataset.rq); return; }
      const q = e.target.closest('[data-q]');
      if (q) {
        const A = {
          why: `Three things inflate it. duration is the call length, known only after the call ${chip('LEAK-bank_marketing')}. job_rate is the target mean computed on all rows, so each row sees its own label ${chip('PIT-004')}. The resampled copies land on both sides of the split, so the model is graded on rows it memorized ${chip('PIT-003')}. Accuracy also hides the rare class ${chip('DCLAB-R18')}.`,
          smote: `Yes, but only inside each training fold, through an imblearn Pipeline. Evaluate on untouched, naturally imbalanced rows with average precision ${chip('PIT-003')} ${chip('DCLAB-R03')}.`,
          campaign: `Not decided yet. If the count includes the call being predicted, it leaks a little. The solution keeps it with a warning and runs a with/without check on identical folds. Only the owner can settle it ${chip('DCLAB-R05')}.`,
        };
        $('#nb-answer', el).innerHTML = `<div class="inset">${A[q.dataset.q]}</div>`;
      }
    });
    $('#nb-fixall', el).addEventListener('click', () => { [3, 4, 5, 6].forEach(i => fixed.add(i)); render(); toast('Applied 4 cell fixes. One finding is left for the model screen.'); });
    $('#nb-reset', el).addEventListener('click', () => { fixed = new Set(); render(); $('#agent-out', el).innerHTML = ''; });
    $('#nb-runall', el).addEventListener('click', () => toast(real
      ? 'Running cells here is not built yet. The stage engine already ran this workflow; download the .ipynb to run it in your own Jupyter.'
      : fixed.size ? 'In the product: runs in the sandbox kernel. The outputs shown are from a real run of this code.' : 'In the product: runs in the sandbox kernel. The output shown is a real run of this notebook.'));
    $('#agent-go', el).addEventListener('click', () => {
      $('#agent-out', el).innerHTML = `<div class="stack tight reveal">${codeBlock(`from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(forest(best_depth), method="isotonic", cv=cv)
calibrated.fit(X_train, y_train)  # calibration is learned on training folds only
print("Brier before", brier_score_loss(y_test, p))
print("Brier after ", brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1]))`, { ok: [3] })}
        <div class="row small"><span class="pill proof">cites</span>${chips(['DCLAB-R18', 'DCLAB-R03'])}<span class="muted">This reads the test set a second time; the companion marks it as a report-only check, not a selection step.</span></div>
        <div class="row"><button type="button" class="btn sm primary" data-toast="Cell accepted and added below cell 6.">Accept</button><button type="button" class="btn sm" data-toast="Rejected. The reason is kept as feedback for the agent.">Reject</button></div></div>`;
      DC.hydrate(el);
    });

    const PIPE = [
      ['md', '<h3>Term-deposit calls · solution v2</h3><p>Predict subscription immediately before a marketing call. <code>duration</code> is forbidden. Primary metric: net value at 2,000 calls.</p>', []],
      ['code', `from dclab import load_snapshot, load_solution
solution = load_solution("term-deposit-calls", version=2)
X, y = load_snapshot("9c1e4b07", solution)   # raises if a forbidden column is requested
split = solution.split(X, y, holdout=0.2, stratify=True, seed=42)  # holdout sealed`, ['DATASET-bank_marketing', 'DCLAB-R01']],
      ['code', `recipe = solution.feature_recipe("ratios")   # smallest set within 0.002 of the best
pipe = recipe.pipeline(model="extra_trees", params="C02")`, ['EXP-008', 'DCLAB-R07']],
      ['code', `cv = split.folds(k=3)                        # identical folds for every comparison
screen = solution.screen(pipe.families, cv, rule="mean - 0.25*std, then runtime")
screen.selected                                 # 'extra_trees'`, ['EXP-009', 'DCLAB-R13']],
      ['code', `tuning = solution.tune(pipe, cv, candidates=3, margin=0.005)
tuning.kept                                     # 'C02', +0.0235 over the paired baseline`, ['EXP-010', 'DCLAB-R16']],
      ['code', `result = split.open_holdout(pipe, approved_by="Shahriyar")   # once; logged
result.roc_auc, result.interval                  # 0.7718, (0.7126, 0.8290)`, ['EXP-010', 'DCLAB-R17']],
    ];
    function samplePipe() {
      $('#pipe-cells', el).innerHTML = PIPE.map(([t, src, ev], i) => t === 'md'
        ? `<div class="cell"><div class="cell-gutter"></div><div class="cell-md">${src}</div></div>`
        : `<div class="cell"><div class="cell-gutter">[${i}]<span class="cell-flag ok">✓</span></div><div class="cell-body">${codeBlock(src)}<div class="cell-tools">${chips(ev)}</div></div></div>`).join('');
      $('#stage-cards', el).innerHTML = [['data', 'WF-04', 'EXP-006'], ['leakage', 'WF-05', 'EXP-007'], ['features', 'WF-06', 'EXP-008'], ['models', 'WF-07', 'EXP-009'], ['final', 'WF-08/09', 'EXP-010']]
        .map(([n, wf, ev]) => `<div class="inset stack tight"><span class="eyebrow">${wf}</span><b>${n}</b><span class="small muted">runs deterministic code, writes a record, notes cite evidence</span>${chip(ev)}</div>`).join('');
    }
    render();
    samplePipe();
    DC.hydrate(el);

    /* ---------- sample mode keeps the original markup; real mode rewrites the same panels ---------- */
    const KEEP = ['nb-eyebrow', 'nb-stats', 'nb-file', 'nb-kernel', 'nb-comp-sub', 'nb-ask-chips', 'nb-outline-intro', 'nb-prec-sub', 'nb-prec-list', 'nb-clean-title', 'nb-clean-sub', 'nb-clean-pill', 'nb-clean-checks', 'nb-pipe-label', 'nb-pipe-kernel', 'nb-stage-sub', 'nb-stage-pill', 'nb-stage-foot'];
    const SAMPLE = {};
    KEEP.forEach(id => { const n = $('#' + id, el); SAMPLE[id] = { html: n.innerHTML, cls: n.className }; });
    const set = (id, html, cls) => { const n = $('#' + id, el); n.innerHTML = html; if (cls != null) n.className = cls; };
    const precCount = n => { const b = $('.ptab[data-ptab="precedents"] .n', el); if (b) b.textContent = String(n); };
    const realOnly = on => { ['nb-fixall', 'nb-reset', 'nb-agent', 'nb-export-sample'].forEach(id => { $('#' + id, el).hidden = on; }); $('#nb-export-real', el).hidden = !on; };

    function showSample() {
      real = null; R = SAMPLE_R; fixed = new Set();
      KEEP.forEach(id => set(id, SAMPLE[id].html, SAMPLE[id].cls));
      realOnly(false); precCount(4);
      $('#nb-answer', el).innerHTML = ''; $('#agent-out', el).innerHTML = '';
      render(); samplePipe(); DC.hydrate(el);
    }

    /* ---------- real data ---------- */
    const STAGES = ['data', 'leakage', 'features', 'models', 'final'];
    const ML = { roc_auc: 'ROC-AUC', average_precision: 'average precision', macro_f1: 'macro-F1', mae: 'MAE', rmse: 'RMSE' };
    const ml = m => ML[m] || String(m || 'score').replace(/_/g, ' ');
    const f4 = v => DC.fmt(v, 4);
    const fileName = p => `dclab-${String(p.name || 'project').slice(0, 40).replace(/ /g, '_')}.ipynb`;
    function proj(p) {
      const Rr = p.records || {}, fin = Rr.final && Rr.final.evidence && Rr.final.evidence.holdout_metrics ? Rr.final : null;
      const metric = (fin || Rr.models || Rr.features || {}).primary_metric || (p.solution || {}).metric || 'roc_auc';
      const ev = fin ? fin.evidence : {};
      return { Rr, fin, ev, metric, ci: ev.holdout_primary_metric_ci, point: fin ? ev.holdout_metrics[metric] : null };
    }
    function renderRealBanner(banner, sev) {
      const P = proj(real.p), n = R.findings.length;
      DC.setNavCount('notebook', n || '');
      const holdout = P.fin ? `It reproduces the project's workflow, whose holdout ${esc(ml(P.metric))} was ${f4(P.point)}${P.ci ? ` (95% ${f4(P.ci.low)}–${f4(P.ci.high)})` : ''}, scored once.` : 'The holdout has not been scored yet, so the last cell has no result to reproduce.';
      if (sev('high')) { banner.className = 'callout bad'; $('#nb-banner-text', el).innerHTML = `<b>${sev('high')} high finding${sev('high') === 1 ? '' : 's'} would change the reported score.</b> Fix them before trusting any number from this notebook. ${holdout}`; }
      else if (n) { banner.className = 'callout ' + (sev('medium') ? 'warn' : 'info'); $('#nb-banner-text', el).innerHTML = `<b>No finding would change the score.</b> The copilot left ${n} lower-severity note${n === 1 ? '' : 's'} on the exported notebook. ${holdout}`; }
      else { banner.className = 'callout ok'; $('#nb-banner-text', el).innerHTML = `<b>The copilot found nothing to flag.</b> ${holdout}`; }
    }
    function realFindings() {
      if (!R.findings.length) return '<div class="empty">No findings. The copilot reads the exported notebook statically; it never runs the code.</div>';
      const ranking = (((real.p.records || {}).models || {}).evidence || {}).ranking || [];
      return R.findings.map(f => `<div class="finding ${f.severity}">
          <div class="spread"><span class="row"><span class="sev ${f.severity}">${f.severity}</span><span class="tag">cell ${f.cell} · line ${f.line}</span></span></div>
          <div class="f-title">${esc(f.title)}</div>
          <div class="f-msg">${esc(f.message)}</div><div class="f-fix"><b>Fix.</b> ${esc(f.suggestion)}</div>
          ${f.detector === 'single_model_family' && ranking.length > 1 ? `<div class="f-msg">In this project the models stage did screen ${ranking.length} families on identical folds (${ranking.slice(0, 5).map(r => esc(r.model)).join(', ')}); the exported notebook reproduces only the chosen one.</div>` : ''}
          <div class="row">${chips(known(f.proof))}</div>
          <div class="row">${f.detector === 'single_model_family' ? '<a class="btn sm" href="#models">Open the model screen</a>' : `<button type="button" class="btn sm proof" data-fix-real>Apply fix</button>`}<button type="button" class="btn sm ghost" data-jump="${f.cell}">Show cell</button></div>
        </div>`).join('');
    }
    async function askReal(question) {
      const box = $('#nb-answer', el);
      box.innerHTML = '<div class="inset muted">Reading the project records…</div>';
      try {
        const a = await DC.api(`/projects/${encodeURIComponent(real.p.id)}/ask`, { method: 'POST', body: { question } });
        box.innerHTML = `<div class="inset stack tight"><span>${DC.linkIds(a.answer)}</span>${a.llm_answer ? `<span><span class="pill outline">LLM · advisory</span> ${DC.linkIds(a.llm_answer)}</span>` : ''}${known(a.proof).length ? `<span class="row">${chips(known(a.proof))}</span>` : ''}</div>`;
      } catch (err) { box.innerHTML = `<div class="inset">${esc(err.message)}</div>`; }
    }
    /* Which workflow block each exported code cell reproduces, read from the markdown heading above it. */
    const BLOCK = [[/^## 1\./, 'WF-02'], [/^## 2\./, 'WF-03'], [/^## 3\./, 'WF-06'], [/^## 4\./, 'WF-07'], [/^## 5\./, 'WF-08'], [/^## 6\./, 'WF-09']];
    function showReal(p, review) {
      real = { p, review }; R = review; fixed = new Set();
      const P = proj(p), S = review.summary || {}, by = S.by_severity || {}, file = fileName(p);
      set('nb-eyebrow', `${esc(p.name)} · notebook`);
      realOnly(true);
      const ex = $('#nb-export-real', el); ex.href = `/api/projects/${encodeURIComponent(p.id)}/export/notebook`;
      const cv = P.fin ? P.ev.cv_selected_metric_mean : (P.Rr.models ? P.Rr.models.evidence.selected_metric_mean : P.Rr.features ? P.Rr.features.evidence.selected_metric_mean : null);
      set('nb-stats', `<div class="stat"><span class="v ${P.fin ? 'ok' : ''}">${P.fin ? f4(P.point) : '—'}${P.fin && P.ci ? ` <small>${f4(P.ci.low)}–${f4(P.ci.high)}</small>` : ''}</span><span class="l">${P.fin ? `honest holdout ${esc(ml(P.metric))} the last cell reproduces · ${DC.int(P.ev.holdout_rows)} rows, scored once` : 'the holdout has not been scored yet · the last cell has nothing to reproduce'}</span></div>
        <div class="stat"><span class="v">${cv != null ? f4(cv) : '—'}</span><span class="l">${cv != null ? `training-CV ${esc(ml(P.metric))} of the chosen ${P.fin ? 'configuration' : 'setup so far'} · the notebook's cross-validation cell` : 'no cross-validation has run yet'}</span></div>
        <div class="stat"><span class="v ${by.high ? 'bad' : ''}">${S.findings || 0} <small>finding${S.findings === 1 ? '' : 's'}</small></span><span class="l">real copilot output for ${esc(file)} · ${by.high || 0} high</span></div>
        <div class="stat"><span class="v proof">${S.code_cells || 0} <small>code cells</small></span><span class="l">generated from the project's workflow · not run here; download it to run</span></div>`);
      set('nb-file', esc(file));
      set('nb-kernel', `<span class="dot"></span>generated by DCLab · not executed here`);
      set('nb-comp-sub', `Real copilot output for ${esc(file)}`);
      set('nb-ask-chips', '<button type="button" class="chip" data-rq="Which model was chosen and why?">Why this model?</button><button type="button" class="chip" data-rq="Is the holdout score honest, and can I trust it?">Is the score honest?</button><button type="button" class="chip" data-rq="Which columns are forbidden or could leak?">What was left out?</button>');
      $('#nb-answer', el).innerHTML = '';
      set('nb-outline-intro', `Every cell of ${esc(file)} with its open findings. The notebook is generated, not run here, so there are no variables in memory. Click a cell to jump to it.`);
      // precedents: the records behind the real findings
      const ids = known([].concat(...R.findings.map(f => f.proof || [])));
      precCount(ids.length);
      set('nb-prec-sub', 'The rules and measured precedents behind this notebook\'s findings');
      set('nb-prec-list', ids.length ? ids.map(id => { const r = DC.REC[id], behind = R.findings.filter(f => (f.proof || []).includes(id)).map(f => f.title); return `<div class="list-item"><div class="li-main"><span class="li-title">${esc(r.title)}</span><span class="li-sub">${esc(DC.TYPE_LABEL[r.type] || r.type)} · behind: ${esc(behind.join('; '))}</span></div>${chip(id)}</div>`; }).join('')
        : '<div class="empty">The copilot found nothing to flag, so no precedent applies to this notebook.</div>');
      // clean version: the exported notebook, each code cell with the workflow block it reproduces
      const serious = (by.high || 0) + (by.medium || 0);
      set('nb-clean-title', `What makes ${esc(file)} clean`);
      set('nb-clean-sub', `Built by DCLab from the project's solution${(p.graph && (p.graph.gates || []).some(g => g.gate === 'solution' && g.approved)) ? ' (signed)' : ' (not signed yet)'}`);
      set('nb-clean-pill', serious ? `${serious} finding${serious === 1 ? '' : 's'}` : 'clean', 'pill ' + (by.high ? 'bad' : serious ? 'warn' : 'ok'));
      const forb = ((p.solution || {}).forbidden || []).length, idn = ((p.solution || {}).identifiers || []).length;
      set('nb-clean-checks', [
        `The solution is applied first: ${forb} forbidden and ${idn} identifier column${idn === 1 ? '' : 's'} are dropped before anything else.`,
        'Every transform lives in one Pipeline, fitted inside each fold.',
        'The holdout is locked before the target is looked at, and scored in one cell, once, at the end.',
        'Each step states its rule; the record IDs below link to the evidence.',
      ].map(t => `<div class="row top"><span class="cell-flag ok">✓</span><span>${esc(t)}</span></div>`).join(''));
      set('nb-pipe-label', `${esc(file)} · every cell links to its workflow block`);
      set('nb-pipe-kernel', `<span class="dot ok"></span>seed 42${p.data && p.data.sha256 ? ` · snapshot ${esc(p.data.sha256.slice(0, 4))}…${esc(p.data.sha256.slice(-4))}` : ''}`);
      const nodes = ((p.graph || {}).nodes || []).reduce((o, n) => (o[n.id] = n, o), {});
      let block = null, n = 0;
      $('#pipe-cells', el).innerHTML = R.cells.map(c => {
        if (c.type === 'markdown') { const hit = BLOCK.find(([re]) => re.test(c.source)); block = hit ? hit[1] : null; return ''; }
        n += 1;
        const node = block && nodes[block];
        const ev = node ? known(node.rules) : [];
        return `<div class="cell"><div class="cell-gutter">[${n}]<span class="cell-flag ok">✓</span></div><div class="cell-body">${codeBlock(c.source.replace(/\n+$/, ''))}${node ? `<div class="cell-tools"><span class="tag">${esc(node.id)} · ${esc(node.name)}</span>${chips(ev)}</div>` : ''}</div></div>`;
      }).join('');
      // stage cells: the five real stage records
      const meta = (p.stage_meta || []).reduce((o, s) => (o[s.key] = s, o), {});
      const doneN = STAGES.filter(s => P.Rr[s]).length;
      set('nb-stage-sub', 'The five stage cells of this project. Each ran deterministic code and wrote a record with notes that cite evidence.');
      set('nb-stage-pill', `${doneN} of 5 run`, 'pill ' + (doneN === 5 ? 'ok' : 'warn'));
      $('#stage-cards', el).innerHTML = STAGES.map(s => {
        const r = P.Rr[s], st = ((p.stages || {})[s] || {}).status || 'pending', m = meta[s] || {};
        const proof = r ? known([].concat(...(r.notes || []).map(x => x.proof || []))).slice(0, 2) : [];
        return `<div class="inset stack tight"><span class="eyebrow">${esc(m.workflow || '')}</span><b>${esc(s)}</b><span class="small muted">${r ? `${esc(st)} · ${(r.elapsed_seconds || 0).toFixed(1)} s · ${(r.notes || []).length} notes · ${(r.claims || []).length} claims` : esc(st === 'pending' ? 'not run yet' : st)}</span>${proof.map(i => chip(i)).join(' ')}</div>`;
      }).join('');
      set('nb-stage-foot', 'The Overview shows the notebook exported from these records, with the copilot beside each cell. Applying fixes and running cells here are not built yet.');
      render();
    }
    function showError(p, msg) {
      real = { p, review: { cells: [], findings: [], summary: {} } }; R = real.review;
      set('nb-eyebrow', `${esc(p.name)} · notebook`);
      realOnly(true);
      $('#nb-export-real', el).hidden = !p.solution;
      $('#nb-export-real', el).href = `/api/projects/${encodeURIComponent(p.id)}/export/notebook`;
      render();
      const b = $('#nb-banner', el); b.className = 'callout info';
      $('#nb-banner-text', el).innerHTML = esc(msg);
      $('#nb-cells', el).innerHTML = '';
    }
    this.s = { tok: 0, cache: {}, showReal, showSample, showError };
  },
  async enter(el) {
    const s = this.s, tok = ++s.tok;
    let p = null;
    try { p = await DC.currentProject.get(); } catch (e) { p = null; }
    if (tok !== s.tok) return;
    DC.markSample(!p);
    if (!p) { s.showSample(); return; }
    if (!p.solution) { s.showError(p, 'This project has no saved solution yet, so there is no notebook to export or review. Save the solution first.'); return; }
    const key = p.id + ':' + p.updated + ':' + Object.keys(p.records || {}).join(',');
    let review = s.cache[key];
    if (!review) {
      try { review = await DC.api(`/projects/${encodeURIComponent(p.id)}/review`); s.cache = { [key]: review }; }
      catch (e) { if (tok === s.tok) s.showError(p, 'The copilot could not review this notebook: ' + e.message); return; }
    }
    if (tok !== s.tok) return;
    try { s.showReal(p, review); } catch (e) { console.error('notebook', e); }
  },
});
