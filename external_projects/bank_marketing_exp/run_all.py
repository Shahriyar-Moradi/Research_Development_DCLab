#!/usr/bin/env python3
"""Run full experiment suite + benchmark for bank_marketing."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from general_pipeline.external_project import run_project  # noqa: E402

if __name__ == "__main__":
    run_project("bank_marketing", models=None, include_transformer=False, save_report=True)
