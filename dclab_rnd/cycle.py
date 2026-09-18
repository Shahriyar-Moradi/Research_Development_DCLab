"""Safe orchestration of the repository's existing experiment runners."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def build_cycle_command(
    *,
    suite: str,
    dataset: str | None,
    model: str,
    mode: str,
    optimization: str,
    reports_only: bool = False,
) -> list[str]:
    """Build an argv list without invoking a shell or accepting arbitrary commands."""
    python = sys.executable
    if suite == "hyperack":
        return [
            python,
            "general_pipeline/run_experiments.py",
            "--mode",
            mode,
            "--optimization",
            optimization,
            "--model",
            model,
        ]
    if suite == "external":
        if not dataset:
            raise ValueError("--dataset is required for the external suite")
        return [
            python,
            "general_pipeline/run_multi_dataset.py",
            "--dataset",
            dataset,
            "--optimization",
            optimization,
            "--model",
            model,
        ]
    if suite == "playbook":
        if not dataset:
            raise ValueError("--dataset is required for the playbook suite")
        command = [python, "general_pipeline/playbook/run_playbook.py", "--dataset", dataset]
        if reports_only:
            command.append("--reports-only")
        return command
    raise ValueError(f"unknown suite: {suite}")


def run_cycle(root: Path, command: list[str]) -> int:
    """Run a controlled experiment command in the repository root."""
    completed = subprocess.run(command, cwd=root, check=False)
    return completed.returncode

