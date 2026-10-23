"use strict";
const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const { spawn } = require("child_process");

const reports = new Map();
const timers = new Map();
const revisions = new Map();
const changed = new vscode.EventEmitter();

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
}

function findRepo(notebook) {
  const configured = vscode.workspace.getConfiguration("dclabNotebook").get("repoPath", "").trim();
  if (configured) return configured;
  const folder = vscode.workspace.getWorkspaceFolder(notebook.uri);
  let candidate = folder ? folder.uri.fsPath : path.dirname(notebook.uri.fsPath);
  for (let i = 0; i < 10; i++) {
    if (fs.existsSync(path.join(candidate, "knowledge", "model_building_rules.jsonl")) &&
        fs.existsSync(path.join(candidate, "dclab_rnd", "notebook_assist.py"))) return candidate;
    const parent = path.dirname(candidate);
    if (parent === candidate) break;
    candidate = parent;
  }
  return null;
}

function runReview(notebook, showErrors = false) {
  const repo = findRepo(notebook);
  if (!repo || !vscode.workspace.isTrusted) {
    if (showErrors) vscode.window.showWarningMessage(repo ? "Trust this workspace before running the local DCLab reviewer." :
      "Set dclabNotebook.repoPath to your DCLab R&D repository.");
    return;
  }
  const config = vscode.workspace.getConfiguration("dclabNotebook");
  const python = config.get("pythonPath", "").trim() || path.join(repo, ".venv", "bin", "python");
  const uri = notebook.uri.toString();
  const revision = (revisions.get(uri) || 0) + 1;
  revisions.set(uri, revision);
  const input = JSON.stringify({notebook_path: notebook.uri.fsPath,
    cells: notebook.getCells().map(cell => ({cell_type: cell.kind === vscode.NotebookCellKind.Code ? "code" : "markdown",
      source: cell.document.getText()}))});
  const env = {...process.env};
  delete env.OPENAI_API_KEY; // The offline reviewer has no use for credentials.
  const child = spawn(python, ["-B", "-m", "dclab_rnd.notebook_assist", "--stdin"],
    {cwd: repo, env, stdio: ["pipe", "pipe", "pipe"], shell: false});
  let stdout = "";
  child.stdout.on("data", chunk => {
    stdout += chunk.toString();
    if (stdout.length > 8_000_000) child.kill();
  });
  child.stderr.on("data", () => {}); // Never show raw notebook/process text in UI errors.
  child.on("error", error => {
    if (showErrors) vscode.window.showErrorMessage(`DCLab reviewer could not start: ${error.code || error.name}`);
  });
  child.on("close", code => {
    if (revisions.get(uri) !== revision) return; // Ignore stale reviews during typing.
    if (code !== 0) {
      if (showErrors) vscode.window.showErrorMessage(`DCLab reviewer failed (exit ${code}). Check the configured Python path.`);
      return;
    }
    try {
      const report = JSON.parse(stdout);
      if (!Array.isArray(report.code_cells)) throw new Error("invalid report");
      reports.set(uri, report);
      changed.fire();
      if (showErrors) vscode.window.showInformationMessage(`DCLab reviewed ${report.code_cells.length} code cells; ${report.finding_count} suggestions.`);
    } catch (_) {
      if (showErrors) vscode.window.showErrorMessage("DCLab reviewer returned an invalid report.");
    }
  });
  child.stdin.on("error", () => {}); // A failed child may close before reading.
  child.stdin.end(input);
}

function schedule(notebook) {
  if (!vscode.workspace.getConfiguration("dclabNotebook").get("autoReview", true)) return;
  const key = notebook.uri.toString();
  clearTimeout(timers.get(key));
  timers.set(key, setTimeout(() => { timers.delete(key); runReview(notebook); }, 700));
}

function htmlFor(cell, finding) {
  const proof = finding.proof || {records: []};
  const items = (proof.records || []).map(record => `<article><div class="tag">${escapeHtml(record.type)} · ${escapeHtml(record.id)}</div>
    <p>${escapeHtml(record.statement)}</p>${record.why ? `<small>Why: ${escapeHtml(record.why)}</small>` : ""}
    ${record.source_path ? `<small>Source: ${escapeHtml(record.source_path)}</small>` : ""}
    ${record.source_references?.length ? `<small>Rule references (not necessarily exact precedents): ${escapeHtml(record.source_references.join("; "))}</small>` : ""}
    ${record.limitations?.length ? `<small>Limitations: ${escapeHtml(record.limitations.join("; "))}</small>` : ""}</article>`).join("");
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
    <style>body{font:14px/1.6 system-ui;padding:28px;max-width:850px;color:var(--vscode-foreground)}h1{line-height:1.2}h2{margin-top:30px}
    article{padding:14px 18px;margin:12px 0;border:1px solid var(--vscode-panel-border);border-radius:9px}.tag{font-size:11px;letter-spacing:.07em;text-transform:uppercase;opacity:.7}
    .lead{font-size:16px}.note{padding:12px 16px;border-left:3px solid var(--vscode-textLink-foreground)}small{display:block;margin-top:7px;opacity:.8}</style></head><body>
    <div class="tag">DCLab notebook companion · cell ${cell + 1} · ${escapeHtml(finding.severity)} · ${escapeHtml(finding.confidence)}</div>
    <h1>${escapeHtml(finding.title)}</h1><p class="lead">${escapeHtml(finding.explanation)}</p>
    <h2>Suggested next check</h2><p>${escapeHtml(finding.next_step)}</p>
    <h2>Show proof</h2>${items || `<p>No directly matching DCLab rule or experiment is cited for this suggestion.</p>`}
    <p class="note">${proof.exact_analogous_experiment ? "A same-dataset experiment claim is recorded above. Verify its scope before applying it here." :
      "No exact analogous experiment is asserted. A methodological rule is not proof that this notebook has a bug."}</p>
    <small>Read-only static review. Execution order, prediction timing and production availability still need human verification.</small></body></html>`;
}

function showProof(uri, cellIndex, kind) {
  const report = reports.get(uri.toString());
  const cell = report?.code_cells.find(item => item.notebook_cell_index === cellIndex);
  const finding = cell?.findings.find(item => item.kind === kind);
  if (!finding) return vscode.window.showInformationMessage("DCLab finding changed; review the notebook again.");
  const panel = vscode.window.createWebviewPanel("dclabProof", finding.title, vscode.ViewColumn.Beside,
    {enableScripts: false, retainContextWhenHidden: false});
  panel.webview.html = htmlFor(cellIndex, finding);
}

function activate(context) {
  const provider = {
    onDidChangeCellStatusBarItems: changed.event,
    provideCellStatusBarItems(cell) {
      if (cell.kind !== vscode.NotebookCellKind.Code) return [];
      const report = reports.get(cell.notebook.uri.toString());
      const reviewed = report?.code_cells.find(item => item.notebook_cell_index === cell.index);
      if (!reviewed) return [];
      if (!reviewed.findings.length) {
        const partial = reviewed.review_status === "unsupported_syntax" || reviewed.review_status === "partial_static_review";
        const item = new vscode.NotebookCellStatusBarItem(partial ? "$(question) DCLab · review incomplete" :
          "$(check) DCLab · no rule matched", vscode.NotebookCellStatusBarAlignment.Left);
        item.tooltip = partial ? "This cell has unsupported syntax or ambiguous call order; DCLab could not establish a complete static review." :
          "Static checks found no matching rule. This is not a safety certification.";
        return [item];
      }
      return reviewed.findings.map(finding => {
        const item = new vscode.NotebookCellStatusBarItem(`$(lightbulb) DCLab · ${finding.title}`,
          vscode.NotebookCellStatusBarAlignment.Left);
        item.tooltip = `${finding.explanation}\nClick to show evidence and limitations.`;
        item.command = {title: "Show DCLab evidence", command: "dclabNotebook.showProof",
          arguments: [cell.notebook.uri, cell.index, finding.kind]};
        return item;
      });
    }
  };
  context.subscriptions.push(changed,
    vscode.notebooks.registerNotebookCellStatusBarItemProvider("jupyter-notebook", provider),
    vscode.commands.registerCommand("dclabNotebook.reviewActive", () => {
      const editor = vscode.window.activeNotebookEditor;
      if (!editor) return vscode.window.showInformationMessage("Open a VS Code notebook first.");
      runReview(editor.notebook, true);
    }),
    vscode.commands.registerCommand("dclabNotebook.showProof", showProof),
    vscode.workspace.onDidOpenNotebookDocument(schedule),
    vscode.workspace.onDidChangeNotebookDocument(event => schedule(event.notebook)),
    vscode.workspace.onDidCloseNotebookDocument(notebook => {
      const key = notebook.uri.toString();
      clearTimeout(timers.get(key)); timers.delete(key); reports.delete(key); revisions.delete(key);
      changed.fire();
    }));
  vscode.workspace.notebookDocuments.forEach(schedule);
}

function deactivate() { for (const timer of timers.values()) clearTimeout(timer); }
module.exports = {activate, deactivate};
