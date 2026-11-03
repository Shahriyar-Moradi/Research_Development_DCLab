"""DCLab Bench B8 (2026-11-03): each agent's requests now under fixed rules, and where flagged sessions stand in the bug loop."""

import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.bench import monitors  # noqa: E402

NOW = datetime(2026, 11, 4, 12, tzinfo=timezone.utc)


def row(purpose, outcome="ok", seconds=2.0, eur=0.001, days_ago=1, now=None):
    return {"at": ((now or NOW) - timedelta(days=days_ago)).isoformat(), "purpose": purpose, "outcome": outcome, "seconds": seconds,
            "cost_eur": eur, "model": "claude-sonnet-5-5", "input_tokens": 100, "output_tokens": 20}


class MonitorTests(unittest.TestCase):
    def test_the_rules_mark_red_amber_and_ok(self):
        rows = ([row("home_agent", "error") for _ in range(10)] + [row("home_agent") for _ in range(30)]  # 25% failing
                + [row("campaign", seconds=120.0) for _ in range(5)]  # slow
                + [row("case_agent", eur=10.0) for _ in range(5)]  # EUR 50 in 7 days
                + [row("critic") for _ in range(5)] + [row("critic", days_ago=30)])  # old rows are not read
        got = {a["purpose"]: a for a in monitors.agents(rows, days=7, now=NOW)}
        self.assertEqual(got["home_agent"]["mark"], "red")
        self.assertEqual(got["campaign"]["mark"], "amber")
        self.assertEqual(got["case_agent"]["mark"], "amber")
        self.assertEqual((got["critic"]["mark"], got["critic"]["requests"]), ("ok", 5))

    def test_few_failures_on_few_requests_are_not_red(self):
        got = monitors.agents([row("x", "error"), row("x"), row("x")], days=7, now=NOW)[0]
        self.assertEqual(got["mark"], "ok")  # under 20 requests no failure rate marks red

    def test_the_bug_loop_counts_where_each_case_stands(self):
        home = Path(tempfile.mkdtemp())
        folder = home / "datasets" / "failures" / "benchmark_candidates"
        folder.mkdir(parents=True)
        (folder / "rows.jsonl").write_text("\n".join(json.dumps({"id": f"s{i}"}) for i in range(3)))
        (home / "datasets" / "failures" / "reviews.json").write_text(json.dumps({"s0": {"verdict": "accept"}, "s1": {"verdict": "reject"}}))
        loop = monitors.bug_loop(home)
        self.assertEqual((loop["candidates"], loop["waiting_review"], loop["accepted"], loop["rejected"]), (3, 1, 1, 1))

    def test_the_summary_reads_only_the_window_without_prompt_text(self):
        """Review finding 7: every page load read 50,000 rows with their prompt text."""
        class Usage:
            def window(self, since, limit):
                return [row("home_agent", now=datetime.now(timezone.utc))]

            def recent(self, limit):
                raise AssertionError("the whole log must not be read")

        s = monitors.summary(Path(tempfile.mkdtemp()), days=7, usage=Usage())
        self.assertEqual((s["agents"][0]["purpose"], s["truncated"]), ("home_agent", False))


if __name__ == "__main__":
    unittest.main()
