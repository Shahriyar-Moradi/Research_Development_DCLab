DC.view('admin', {
  init(el) {
    const { $, esc, chip, toast } = DC;
    const POL = [
      ['The holdout opens once, after every choice is locked', 'DCLAB-R17', true, true],
      ['Forbidden columns are blocked at the tool level', 'DCLAB-R04', true, true],
      ['LLM critique is advisory; numbers come from code', 'DCLAB-R20', true, true],
      ['A research result is never labelled production-ready', 'DCLAB-R22', true, true],
      ['The solution must be signed before WF-04', 'DCLAB-R01', true, false],
      ['Tuning needs a margin declared before the run', 'DCLAB-R16', true, false],
      ['Ask the owner when a column\'s timing is unclear', 'DCLAB-R05', true, false],
      ['Allow quick mode (row sampling) for exploration', '', true, false],
      ['Let the intern run code in the sandbox', '', true, false],
      ['Let the intern start GPU jobs under the cap without asking', '', false, false],
    ];
    $('#policy-list', el).innerHTML = POL.map(([t, r, on, locked], i) => `<div class="list-item"><label class="switch" data-style="flex:1"><input type="checkbox" data-pol="${i}" ${on ? 'checked' : ''} ${locked ? 'disabled' : ''}><span class="track"></span><span>${esc(t)}</span></label>${r ? chip(r) : ''}${locked ? '<span class="pill">locked</span>' : ''}</div>`).join('');
    $('#policy-list', el).addEventListener('change', e => { const i = e.target.dataset.pol; if (i != null) toast(`Policy ${e.target.checked ? 'enabled' : 'disabled'} and written to the audit log.`); });
    const ROLES = ['Owner', 'ML engineer', 'Reviewer', 'Business viewer', 'Intern'];
    const ACT = [
      ['Create a project', ['y', 'y', '-', '-', 'y']], ['Sign a solution', ['y', '-', '-', '-', '-']], ['Approve a stage', ['y', 'y', 'y', '-', 'a']],
      ['Open the holdout', ['y', 'a', '-', '-', 'a']], ['Override the rule\'s pick (with a reason)', ['y', 'y', '-', '-', '-']], ['Approve a brief', ['y', '-', '-', 'y', '-']],
      ['Start a GPU job over the cap', ['y', 'a', '-', '-', 'a']], ['Change policies', ['y', '-', '-', '-', '-']], ['See code and tool calls', ['y', 'y', 'y', '-', 'y']],
    ];
    const cell = v => v === 'y' ? '<span class="yes">✓</span>' : v === 'a' ? '<span class="ask">asks</span>' : '<span class="no">—</span>';
    $('#role-matrix', el).innerHTML = `<thead><tr><th>Action</th>${ROLES.map(r => `<th>${r}</th>`).join('')}</tr></thead><tbody>${ACT.map(([a, v]) => `<tr><td>${esc(a)}</td>${v.map(cell).join('')}</tr>`).join('')}</tbody>`;
    DC.hydrate(el);
  },
});
