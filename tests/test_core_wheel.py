"""Package 17.8: dclab_core as a versioned wheel the main product can pin.

The version is one number in three places (pyproject.toml, dclab_core.__version__, the changelog's latest entry).
Building the wheel and using it from a clean Python 3.12 environment takes a minute and may need the network for the
dependencies, so it runs only when asked: DCLAB_CORE_WHEEL_CHECK=1 (or `make core-wheel-check`, which runs the same
script directly).
"""

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "dclab_core"


def _pyproject_version() -> str:
    return re.search(r'^version = "([^"]+)"', (CORE / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)


class VersionTest(unittest.TestCase):
    def test_one_version_in_the_package_the_module_and_the_changelog(self):
        version = _pyproject_version()
        init = re.search(r'^__version__ = "([^"]+)"', (CORE / "__init__.py").read_text(encoding="utf-8"), re.M).group(1)
        changelog = (CORE / "CHANGELOG.md").read_text(encoding="utf-8")
        latest = re.search(r"^## (\S+)", changelog, re.M).group(1)
        self.assertEqual(init, version)
        self.assertEqual(latest, version)
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

    def test_every_release_is_in_the_changelog_newest_first(self):
        releases = re.findall(r"^## (\d+)\.(\d+)\.(\d+)", (CORE / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
        numbers = [tuple(map(int, r)) for r in releases]
        self.assertIn((0, 1, 0), numbers)
        self.assertEqual(numbers, sorted(numbers, reverse=True))

    def test_the_changelog_ships_in_the_wheel(self):
        self.assertIn('"CHANGELOG.md"', (CORE / "pyproject.toml").read_text(encoding="utf-8"))


@unittest.skipUnless(os.environ.get("DCLAB_CORE_WHEEL_CHECK") == "1",
                     "builds and installs the wheel in a clean Python 3.12 (about a minute; may need the network): "
                     "set DCLAB_CORE_WHEEL_CHECK=1 or run make core-wheel-check")
class WheelTest(unittest.TestCase):
    def test_the_wheel_installs_in_a_clean_python_3_12_and_solves_a_replica(self):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / "core_wheel_check.py")], capture_output=True,
                                text=True, timeout=900)
        summary = json.loads(result.stdout.strip().splitlines()[-1])
        if result.returncode == 75:
            self.skipTest(summary.get("reason", "could not run here"))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["python"].startswith("3.12."))
        self.assertEqual(summary["dclab_core"], _pyproject_version())
        self.assertEqual(summary["unexpected_distributions"], [])
        self.assertEqual(summary["split"], "group_disjoint")


if __name__ == "__main__":
    unittest.main()
