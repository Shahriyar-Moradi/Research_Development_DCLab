"""The studio's proposal offers the core's entities as groups (2026-10-24).

It offered a group only when the column's name looked like an identifier, so a plain ``account`` or ``borrower`` was never
offered and, predicting well alone, was proposed as a leak: the local solver copied that in four of five homeworks
(LRN-003, LRN-008). Now every challenge variant's group is offered, kept out as an identifier and never called a leak;
a group the names found stays first; a plain category is not made a group; the card reads only the rows the proposal
reads.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
except ImportError as error:
    MISSING = str(error)


def propose(frame, target, task=None):
    from dclab_rnd.studio import data as sd, solution as sc

    return sc.propose(frame, sd.profile_table(frame), target, task)


@unittest.skipIf(MISSING, MISSING)
class ProposalEntityTests(unittest.TestCase):
    def test_every_challenge_group_is_offered_kept_out_and_never_a_leak(self):
        from dclab_core import variants as V

        for seed in V.seeds("groups", "learning", 8):
            v = V.generate("groups", seed)
            p = propose(v.table, v.pack.brief.target, v.pack.brief.task)
            g = v.key.group_column
            offered = p["group_candidates"] + [x["column"] for x in p["groups_to_confirm"]]
            self.assertIn(g, offered, seed)  # a group, or offered to confirm
            self.assertNotIn(g, [f["column"] for f in p["forbidden"]], seed)  # never a leak

    def test_a_group_the_names_found_stays_first(self):
        rng = np.random.default_rng(0)
        n = 600
        frame = pd.DataFrame({"store_id": rng.choice([f"s{i}" for i in range(40)], n), "account": rng.choice([f"a{i}" for i in range(60)], n),
                              "amount": rng.normal(size=n), "y": rng.integers(0, 2, n)})
        p = propose(frame, "y", "binary")
        self.assertEqual(p["group_candidates"][0], "store_id")

    def test_a_plain_category_is_not_made_a_group(self):
        rng = np.random.default_rng(1)
        n = 600
        frame = pd.DataFrame({"colour": rng.choice(["red", "green", "blue", "black"], n), "amount": rng.normal(size=n), "y": rng.integers(0, 2, n)})
        p = propose(frame, "y", "binary")
        self.assertNotIn("colour", p["group_candidates"])
        self.assertNotIn("colour", p["identifiers"])

    def test_properties_named_like_keys_categories_numbers_and_text_dates_stay_features(self):
        """The review's cases (2026-10-24): each was dropped as an identifier, or made the split's group, by the card's whole
        entity rule; none is a key."""
        rng = np.random.default_rng(3)
        n = 5000
        frame = pd.DataFrame({
            "PhoneService": rng.choice(["Yes", "No"], n),                                   # a yes or no named like a key
            "ring-number": rng.choice(["one", "two", "none"], n),                           # three levels named like a key
            "brand": rng.choice([f"brand {chr(65 + i)}" for i in range(21)], n),            # a word category of 21 levels
            "state": rng.choice([f"State {i}" for i in range(50)], n),
            "NumberOfOpenCreditLines": rng.integers(0, 30, n),                             # a count named like a key
            "word_freq_email": np.round(rng.random(n) * 3, 2),                              # a number named like a key
            "created_date": [f"{m}/{d}/2022" for m, d in zip(rng.integers(1, 13, n), rng.integers(1, 29, n))],  # a date as text
            "y": rng.integers(0, 2, n)})
        p = propose(frame, "y", "binary")
        for name in ("PhoneService", "ring-number", "brand", "state", "NumberOfOpenCreditLines", "word_freq_email", "created_date"):
            self.assertNotIn(name, p["identifiers"], name)
            self.assertNotIn(name, p["group_candidates"], name)
        self.assertIsNone(p.get("group_column"))

    def test_a_key_word_inside_a_name_is_not_a_key(self):
        rng = np.random.default_rng(5)
        n = 5000
        frame = pd.DataFrame({"email_domain": rng.choice([f"mail{i}.com" for i in range(15)], n), "phone_brand": rng.choice([f"maker {i}" for i in range(30)], n),
                              "user_email": rng.choice([f"u{i}@x.com" for i in range(900)], n), "y": rng.integers(0, 2, n)})
        p = propose(frame, "y", "binary")
        for name in ("email_domain", "phone_brand"):
            self.assertNotIn(name, p["identifiers"], name)
            self.assertNotIn(name, p["group_candidates"], name)
        self.assertIn("user_email", p["group_candidates"])  # the key's word ends its name

    def test_a_small_tables_recurring_category_is_offered_never_taken(self):
        """The second review: product_name in a 600-row table recurs as an account does; shape cannot tell them apart, so it
        is offered to confirm, kept as a feature, and never the split's group by itself."""
        rng = np.random.default_rng(6)
        n = 600
        frame = pd.DataFrame({"product_name": rng.choice([f"Widget {i}" for i in range(100)], n), "amount": rng.normal(size=n), "y": rng.integers(0, 2, n)})
        p = propose(frame, "y", "binary")
        self.assertNotIn("product_name", p["identifiers"])
        self.assertNotIn("product_name", p["group_candidates"])
        self.assertIn("product_name", [x["column"] for x in p["groups_to_confirm"]])
        self.assertIsNone(p.get("group_column"))

    def test_a_proposal_never_holds_more_identifiers_than_a_solution_keeps(self):
        from dclab_rnd.studio import solution as sc

        rng = np.random.default_rng(4)
        n = 3000
        frame = pd.DataFrame({f"key_{i}": rng.choice([f"K{i}-{j:04d}" for j in range(200)], n) for i in range(60)} | {"y": rng.integers(0, 2, n)})
        p = propose(frame, "y", "binary")
        self.assertLessEqual(len(p["identifiers"]), sc.IDENTIFIERS_MAX)

    def test_the_card_reads_only_the_rows_the_proposal_reads(self):
        from dclab_rnd.studio import solution as sc

        seen = []
        real = sc.core_keys

        def spy(frame, target):
            seen.append(len(frame))
            return real(frame, target)

        rng = np.random.default_rng(2)
        n = 400
        frame = pd.DataFrame({"day": pd.date_range("2025-01-01", periods=n, freq="D").astype(str), "account_id": rng.choice([f"AC-{i:05d}" for i in range(30)], n),
                              "amount": rng.normal(size=n), "y": rng.integers(0, 2, n)})
        with mock.patch.object(sc, "core_keys", spy):
            p = propose(frame, "y", "binary")
        self.assertTrue(seen and seen[0] < n, seen)  # ordered by time: the latest rows, a holdout's, are not read
        self.assertIn("account_id", p["group_candidates"])

    def test_a_card_that_fails_leaves_the_proposal_as_the_rules_make_it(self):
        from dclab_rnd.studio import solution as sc

        with mock.patch("dclab_core.datacard.card_of", side_effect=RuntimeError("boom")):
            self.assertEqual(sc.core_keys(pd.DataFrame({"a": [1, 2], "y": [0, 1]}), "y"), [])


if __name__ == "__main__":
    unittest.main()
