#!/usr/bin/env node
// npx dclabai [command] [args]: runs the `dclab` command of the Python package dclabai (default command: start).
// It uses uv (uvx) when it is installed, else pipx, else a virtual environment of its own in ~/.cache/dclabai/venv
// (made on the first run with python3 -m venv, then pip install dclabai). Nothing else is downloaded.
"use strict";
const { spawnSync } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const PKG = process.env.DCLABAI_PACKAGE || "dclabai"; // e.g. "dclabai==0.1.0" or a local path while developing
const args = process.argv.slice(2);
const command = args.length ? args : ["start"];

function has(tool) {
  const probe = spawnSync(tool, ["--version"], { stdio: "ignore", shell: process.platform === "win32" });
  return probe.status === 0;
}

function recentPython(tool) {
  const probe = spawnSync(tool, ["-c", "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"],
                          { stdio: "ignore", shell: process.platform === "win32" });
  return probe.status === 0;
}

function run(tool, toolArgs) {
  const done = spawnSync(tool, toolArgs, { stdio: "inherit", shell: process.platform === "win32" });
  if (done.error) {
    console.error(`dclabai: could not run ${tool}: ${done.error.message}`);
    return 1;
  }
  return done.status === null ? 1 : done.status;
}

function ownVenv() {
  const home = process.env.DCLABAI_CACHE || path.join(process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache"), "dclabai");  // never the workspace
  const venv = path.join(home, "venv");
  const bin = process.platform === "win32" ? path.join(venv, "Scripts") : path.join(venv, "bin");
  const dclab = path.join(bin, process.platform === "win32" ? "dclab.exe" : "dclab");
  if (!fs.existsSync(dclab)) {
    const python = ["python3.13", "python3.12", "python3.11", "python3.10", "python3", "python"].find(recentPython);
    if (!python) {
      console.error("dclabai needs Python 3.10 or later (https://www.python.org/downloads/), or uv (https://docs.astral.sh/uv/).");
      return null;
    }
    console.error(`dclabai: first run, installing ${PKG} into ${venv} (a few minutes)...`);
    fs.mkdirSync(home, { recursive: true });
    if (run(python, ["-m", "venv", venv]) !== 0) return null;
    const pip = path.join(bin, process.platform === "win32" ? "pip.exe" : "pip");
    if (run(pip, ["install", "--upgrade", PKG]) !== 0) return null;
  }
  return dclab;
}

let status;
if (has("uvx")) {
  status = run("uvx", ["--from", PKG, "dclab", ...command]);
} else if (has("pipx")) {
  status = run("pipx", ["run", "--spec", PKG, "dclab", ...command]);
} else {
  const dclab = ownVenv();
  status = dclab ? run(dclab, command) : 1;
}
process.exit(status);
