DC.view('integrations', {
  init(el) {
    const { $, esc, chip, icon } = DC;
    el.innerHTML = el.innerHTML.replace(/\$\{chip:([A-Za-z0-9_-]+)\}/g, (m, id) => chip(id)).replace(/\$\{icon:([a-z]+)\}/g, (m, n) => icon(n));
    const T = [
      ['search_evidence', 'Filter-then-rank search over rules, workflows, datasets, experiments, precedents, pitfalls and findings', 'evidence'],
      ['get_record', 'Return one full evidence record by ID', 'evidence'], ['get_rules', 'List the model-building rules, optionally one category', 'evidence'],
      ['plan_next_stage', 'The next stage, its method, its rules and precedents for this dataset', 'evidence'], ['review_code', 'Static methodology review of one cell or script', 'evidence'],
      ['list_samples', 'Datasets the R&D studied, with their solutions', 'project'], ['create_project', 'Create a notebook project', 'project'],
      ['use_sample', 'Load a studied dataset with its profile and solution suggestion', 'project'], ['describe_data', 'Rows, columns, candidates for target, time, ID and text', 'project'],
      ['propose_solution', 'Audit columns for a target and propose a solution with proof', 'project'], ['set_solution', 'Save the solution; clears earlier stage results', 'project'],
      ['set_settings', 'How many rows the stages use', 'project'], ['run_stage', 'Run one stage; stages run in order', 'project'],
      ['run_all', 'Run the remaining stages; the final one uses the holdout once', 'project'], ['approve_stage', 'Approve a stage, optionally choosing another option', 'project'],
      ['get_results', 'Records of the finished stages', 'project'], ['ask_project', 'Ask the project; answers cite records', 'project'],
      ['export_notebook', 'Write the runnable notebook and the report', 'project'],
    ];
    $('#tool-table tbody', el).innerHTML = T.map(([n, d, g]) => `<tr><td class="mono small">${n}</td><td class="small">${esc(d)}</td><td><span class="pill ${g === 'evidence' ? 'proof' : 'accent'}">${g}</span></td></tr>`).join('');
    DC.hydrate(el);
  },
});
