DC.view('notebook', {
  init(el) {
    const { $, $$, esc, chip, chips, icon, codeBlock, toast } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const R = window.DEMO_REVIEW || { cells: [], findings: [], summary: {} };
    const FIX = {
      3: `# duration is known only after the call ends: forbidden by contract v2
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
      if (f.detector === 'single_model_family') return false;
      const c = f.detector === 'missing_random_state' ? f.cell : FIXES_FINDING[f.detector];
      return c != null && fixed.has(c);
    }
    function render() {
      const open = R.findings.filter(f => !isFixed(f));
      const sev = s => open.filter(f => f.severity === s).length;
      $('#nb-count', el).textContent = open.length;
      $('#nb-count', el).className = 'pill ' + (sev('high') ? 'bad' : open.length ? 'warn' : 'ok');
      $('#sev-counts', el).innerHTML = ['high', 'medium', 'low', 'info'].map(s => sev(s) ? `<span class="sev ${s}">${sev(s)} ${s}</span>` : '').join('');
      const banner = $('#nb-banner', el);
      if (fixed.size < 4) { banner.className = 'callout bad'; $('#nb-banner-text', el).innerHTML = `<b>${sev('high')} findings would change the reported score.</b> The last cell prints accuracy 0.9689, but the duration leak, the target-mean feature and copies of the same rows in train and test inflate it. Fix the high findings first.`; }
      else { banner.className = 'callout ok'; $('#nb-banner-text', el).innerHTML = `<b>The honest score: ROC-AUC 0.7577, average precision 0.3454</b> on 9,043 untouched test rows, after the fixes. The notebook first reported accuracy 0.9689. The R&D's own campaign landed in the same range ${chip('EXP-007')} ${chip('EXP-010')}. One finding is still open: only one model family was tried.`; }
      $('#nb-cells', el).innerHTML = R.cells.map((c, i) => {
        if (c.type === 'markdown') return `<div class="cell" id="nbc-${i}"><div class="cell-gutter"></div><div class="cell-md"><h3>Bank marketing: will the client subscribe?</h3><p>A typical first-pass notebook from a teammate. It runs, and the score looks great.</p></div></div>`;
        const fs = R.findings.filter(f => f.cell === i);
        const openF = fs.filter(f => !isFixed(f));
        const worst = ['high', 'medium', 'low', 'info'].find(s => openF.some(f => f.severity === s));
        const src = fixed.has(i) ? FIX[i] : c.source;
        const hl = fixed.has(i) ? [] : [...new Set(openF.map(f => f.line))];
        const okLines = fixed.has(i) ? src.split('\n').map((_, k) => k + 1) : [];
        const n = R.cells.slice(0, i + 1).filter(x => x.type === 'code').length;
        let out = OUT[i] || '';
        if (i === 6 && fixed.has(6)) out = '8 0.7577 0.3454 0.1866';
        return `<div class="cell" id="nbc-${i}">
          <div class="cell-gutter">[${n}]${worst ? `<span class="cell-flag ${worst === 'info' ? 'low' : worst}" title="${openF.length} open findings">${openF.length}</span>` : `<span class="cell-flag ok">✓</span>`}</div>
          <div class="cell-body">
            ${codeBlock(src, { hl, ok: okLines })}
            ${out ? `<div class="cell-out">${esc(out)}${i === 6 && !fixed.has(6) ? '  <span class="pill bad">inflated</span>' : i === 6 ? '  <span class="pill ok">honest · test used once</span>' : ''}</div>` : ''}
            ${openF.length ? `<div class="stack tight">${openF.map(f => `<div class="cell-tools"><span class="sev ${f.severity}">${f.severity}</span><span>${esc(f.title)}</span><span class="faint">line ${f.line}</span></div>`).join('')}${FIX[i] && !fixed.has(i) ? `<div><button type="button" class="btn sm proof" data-fix="${i}" data-f="nb.fix">Apply fix to this cell</button></div>` : ''}</div>` : ''}
            ${fixed.has(i) ? `<div class="cell-tools"><span class="pill ok">fixed by the companion</span><button type="button" class="link-btn small" data-undo="${i}">Undo</button></div>` : ''}
          </div></div>`;
      }).join('');
      $('#findings', el).innerHTML = R.findings.map((f, i) => {
        const done = isFixed(f);
        return `<div class="finding ${done ? 'fixed' : f.severity}">
          <div class="spread"><span class="row"><span class="sev ${f.severity}">${f.severity}</span><span class="tag">cell ${f.cell} · line ${f.line}</span></span>${done ? '<span class="pill ok">fixed</span>' : ''}</div>
          <div class="f-title">${esc(f.title)}</div>
          ${done ? '' : `<div class="f-msg">${esc(f.message)}</div><div class="f-fix"><b>Fix.</b> ${esc(f.suggestion)}</div>`}
          <div class="row">${chips(f.proof)}</div>
          ${done ? '' : `<div class="row">${FIX[f.cell] && f.detector !== 'single_model_family' ? `<button type="button" class="btn sm proof" data-fix="${f.detector === 'missing_random_state' ? f.cell : (FIXES_FINDING[f.detector] ?? f.cell)}">Apply fix</button>` : f.detector === 'single_model_family' ? '<a class="btn sm" href="#models">Open the model screen</a>' : ''}<button type="button" class="btn sm ghost" data-jump="${f.cell}">Show cell</button><button type="button" class="btn sm ghost" data-dismiss="${i}">Dismiss…</button></div>`}
        </div>`;
      }).join('') + (fixed.has(6) ? `<div class="finding medium"><div class="spread"><span class="row"><span class="sev medium">medium</span><span class="tag">cell 6 · output</span></span><span class="pill info">new after the run</span></div>
          <div class="f-title">Probabilities are distorted by class_weight="balanced"</div>
          <div class="f-msg">Brier score 0.1866 on the test rows. The R&amp;D's calibrated model scored 0.0870. The cost calculator needs honest probabilities.</div>
          <div class="f-fix"><b>Fix.</b> Calibrate on the training folds (CalibratedClassifierCV) or drop the class weights and pick the cut-off from the costs.</div>
          <div class="row">${chips(['DCLAB-R18', 'EXP-010'])}</div></div>` : '');
      $('#nb-outline', el).innerHTML = R.cells.map((c, i) => { const fs = R.findings.filter(f => f.cell === i && !isFixed(f)); const label = c.type === 'markdown' ? 'Title' : (c.source.split('\n')[0].replace(/^#\s*/, '').slice(0, 26) || 'cell ' + i); return `<button type="button" class="row" data-jump="${i}" data-style="justify-content:space-between;width:100%;text-align:left"><span class="mono xs">${esc(label)}</span>${fs.length ? `<span class="cell-flag ${fs.some(f => f.severity === 'high') ? 'high' : 'medium'}" data-style="width:18px;height:18px;font-size:10px">${fs.length}</span>` : ''}</button>`; }).join('');
      $('#nb-vars', el).innerHTML = fixed.has(5)
        ? '<dt>df</dt><dd>45,211 × 17</dd><dt>X_train</dt><dd>36,168 × 7</dd><dt>X_test</dt><dd>9,043 × 7 · untouched</dd><dt>prep</dt><dd>ColumnTransformer</dd>'
        : '<dt>df</dt><dd>45,211 × 18</dd><dt>df_bal</dt><dd>79,844 × 18 <span class="pill bad">copies</span></dd><dt>X_all</dt><dd>79,844 × 9</dd><dt>features</dt><dd>9 incl. duration</dd>';
      DC.hydrate(el);
    }
    function applyFix(i) { fixed.add(Number(i)); render(); toast(`Fix applied to cell ${i}. The original stays in the cell history.`); }
    el.addEventListener('click', e => {
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
      const q = e.target.closest('[data-q]');
      if (q) {
        const A = {
          why: `Three things inflate it. duration is the call length, known only after the call ${chip('LEAK-bank_marketing')}. job_rate is the target mean computed on all rows, so each row sees its own label ${chip('PIT-004')}. The resampled copies land on both sides of the split, so the model is graded on rows it memorized ${chip('PIT-003')}. Accuracy also hides the rare class ${chip('DCLAB-R18')}.`,
          smote: `Yes, but only inside each training fold, through an imblearn Pipeline. Evaluate on untouched, naturally imbalanced rows with average precision ${chip('PIT-003')} ${chip('DCLAB-R03')}.`,
          campaign: `Not decided yet. If the count includes the call being predicted, it leaks a little. The contract keeps it with a warning and runs a with/without check on identical folds. Only the owner can settle it ${chip('DCLAB-R05')}.`,
        };
        $('#nb-answer', el).innerHTML = `<div class="inset">${A[q.dataset.q]}</div>`;
      }
    });
    $('#nb-fixall', el).addEventListener('click', () => { [3, 4, 5, 6].forEach(i => fixed.add(i)); render(); toast('Applied 4 cell fixes. One finding is left for the model screen.'); });
    $('#nb-reset', el).addEventListener('click', () => { fixed = new Set(); render(); $('#agent-out', el).innerHTML = ''; });
    $('#nb-runall', el).addEventListener('click', () => toast(fixed.size ? 'In the product: runs in the sandbox kernel. The outputs shown are from a real run of this code.' : 'In the product: runs in the sandbox kernel. The output shown is a real run of this notebook.'));
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
    render();

    const PIPE = [
      ['md', '<h3>Term-deposit calls · contract v2</h3><p>Predict subscription immediately before a marketing call. <code>duration</code> is forbidden. Primary metric: net value at 2,000 calls.</p>', []],
      ['code', `from dclab import load_snapshot, load_contract
contract = load_contract("term-deposit-calls", version=2)
X, y = load_snapshot("9c1e4b07", contract)   # raises if a forbidden column is requested
split = contract.split(X, y, holdout=0.2, stratify=True, seed=42)  # holdout sealed`, ['DATASET-bank_marketing', 'DCLAB-R01']],
      ['code', `recipe = contract.feature_recipe("ratios")   # smallest set within 0.002 of the best
pipe = recipe.pipeline(model="extra_trees", params="C02")`, ['EXP-008', 'DCLAB-R07']],
      ['code', `cv = split.folds(k=3)                        # identical folds for every comparison
screen = contract.screen(pipe.families, cv, rule="mean - 0.25*std, then runtime")
screen.selected                                 # 'extra_trees'`, ['EXP-009', 'DCLAB-R13']],
      ['code', `tuning = contract.tune(pipe, cv, candidates=3, margin=0.005)
tuning.kept                                     # 'C02', +0.0235 over the paired baseline`, ['EXP-010', 'DCLAB-R16']],
      ['code', `result = split.open_holdout(pipe, approved_by="Shahriyar")   # once; logged
result.roc_auc, result.interval                  # 0.7718, (0.7126, 0.8290)`, ['EXP-010', 'DCLAB-R17']],
    ];
    $('#pipe-cells', el).innerHTML = PIPE.map(([t, src, ev], i) => t === 'md'
      ? `<div class="cell"><div class="cell-gutter"></div><div class="cell-md">${src}</div></div>`
      : `<div class="cell"><div class="cell-gutter">[${i}]<span class="cell-flag ok">✓</span></div><div class="cell-body">${codeBlock(src)}<div class="cell-tools">${chips(ev)}</div></div></div>`).join('');
    $('#stage-cards', el).innerHTML = [['data', 'WF-04', 'EXP-006'], ['leakage', 'WF-05', 'EXP-007'], ['features', 'WF-06', 'EXP-008'], ['models', 'WF-07', 'EXP-009'], ['final', 'WF-08/09', 'EXP-010']]
      .map(([n, wf, ev]) => `<div class="inset stack tight"><span class="eyebrow">${wf}</span><b>${n}</b><span class="small muted">runs deterministic code, writes a record, notes cite evidence</span>${chip(ev)}</div>`).join('');
    DC.hydrate(el);
  },
});
