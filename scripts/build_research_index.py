#!/usr/bin/env python3
"""Regenerate research/<track>/INDEX.md and the track graph. Kept for old commands;
the code lives in dclab_rnd/research_map.py (python -m dclab_rnd.research_map)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dclab_rnd.research_map import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
