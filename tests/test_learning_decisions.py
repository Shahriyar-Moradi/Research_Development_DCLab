"""Challenge packs as decision data: the keys label each column's decisions, the rows split by the pack's partition, a
sealed variant is never read, the answerers are scored by the column's part in the trap, and GLiNER's kept answers are
read back.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dclab_rnd import experience

    MISSING = "" if experience.contract is not None else "dclab_core is not installed here"
    import pandas  # noqa: F401
except ImportError as error:
    MISSING = str(error)


@unittest.skipIf(MISSING, MISSING)
class DecisionSetTests(unittest.TestCase):
    def test_the_key_labels_every_named_column_and_nothing_else(self):
        from dclab_core.challenge_pack import ChallengeKey
        from dclab_rnd.learning.decision_sets import roles_of, truths_of

        key = ChallengeKey(pack="pk-000000000000", family="groups", seed=0, partition="learning", leaks=["closed_balance"],
                           group_column="borrower", row_key="loan_id", decoys=["prior_closed_loans"], known_before=["amount"],
                           time_column="applied_at", knobs={})
        roles = roles_of(key)
        self.assertEqual(roles, {"amount": "honest", "prior_closed_loans": "decoy", "applied_at": "time", "closed_balance": "leak",
                                 "loan_id": "row_key", "borrower": "group"})
        t = truths_of(roles)
        self.assertEqual(t["borrower"], {"column.group": True, "feature.leakage_suspect": False, "column.is_identifier": True})
        self.assertEqual(t["prior_closed_loans"], {"column.group": False, "feature.leakage_suspect": False, "column.is_identifier": False})
        self.assertTrue(t["closed_balance"]["feature.leakage_suspect"])

    def test_rows_carry_the_keys_labels_the_rules_and_the_proposals_answers(self):
        from dclab_core import variants as V
        from dclab_rnd.learning.decision_sets import rows_of

        seed = V.seeds("groups", "learning", 1)[0]
        key = V.generate("groups", seed).key
        rows = rows_of("groups", seed, "learning")
        self.assertTrue(rows and all(r["split"] == "train" for r in rows))
        group = [r for r in rows if r["point"] == "column.group" and r["subject"]["key"] == key.group_column]
        self.assertEqual(len(group), 1)
        self.assertIs(group[0]["label"]["value"], True)
        self.assertIn(group[0]["proposal"], (True, False))
        self.assertIn(group[0]["value_used"], (True, False))
        self.assertTrue(all(r["role"] in ("group", "row_key", "leak", "decoy", "honest", "time") for r in rows))
        self.assertNotIn(V.generate("groups", seed).pack.brief.target, {r["subject"]["key"] for r in rows})  # the outcome is no column decision

    def test_build_reads_learning_and_check_variants_never_a_sealed_one(self):
        from dclab_core import variants as V
        from dclab_rnd.learning import decision_sets as D

        asked = []
        real = V.seeds

        def seeds(family, partition, count, start=0):
            asked.append(partition)
            return real(family, partition, count, start)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(V, "seeds", seeds):
            manifest = D.build("groups", 1, out=Path(tmp), log=lambda *_: None)
            rows = D.read("column.group", Path(tmp))
        self.assertEqual(sorted(set(asked)), ["check", "learning"])
        self.assertEqual({r["split"] for r in rows}, {"train", "dev"})
        self.assertIn("never read", manifest["rule"])

    def test_scores_say_how_each_kind_of_column_was_answered(self):
        from dclab_rnd.learning.decision_sets import score

        rows = [{"role": "group", "label": {"value": True}}, {"role": "group", "label": {"value": True}},
                {"role": "decoy", "label": {"value": False}}, {"role": "honest", "label": {"value": False}}]
        s = score(rows, [True, False, True, None])
        self.assertEqual((s["n"], s["answered"], s["accuracy"]), (4, 3, 0.25))
        self.assertEqual(s["by_role"]["group"], {"n": 2, "said_yes": 1, "right": 1})
        self.assertEqual(s["by_role"]["decoy"], {"n": 1, "said_yes": 1, "right": 0})

    def test_an_answerer_that_never_ran_is_not_measured_not_scored_wrong(self):
        from dclab_rnd.learning.decision_sets import score

        s = score([{"role": "group", "label": {"value": True}}] * 3, [None, None, None])
        self.assertEqual((s["answered"], s["accuracy"], s["interval"]), (0, None, None))

    def test_a_held_out_skin_is_never_trained_on_and_no_sealed_row_is(self):
        from dclab_rnd.learning.decision_sets import held_out

        generated = [{"split": "train", "id": "g1"}, {"split": "sealed", "id": "g2"}, {"split": "dev", "id": "g3"}]
        rows = [{"split": s, "skin": k, "id": f"{s}-{k}"} for s in ("train", "dev") for k in ("loan", "ticket")]
        fit, test = held_out(generated, rows, "loan")
        self.assertEqual({r["id"] for r in fit}, {"g1", "train-ticket"})
        self.assertEqual({r["id"] for r in test}, {"dev-loan"})
        self.assertFalse(any(r.get("split") == "sealed" for r in fit))

    def test_glINER_answers_kept_beside_the_rows_are_read_back(self):
        from dclab_rnd.learning.decision_sets import _gliner_answers

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "decisions" / "column.group"
            folder.mkdir(parents=True)
            (folder / "gliner.jsonl").write_text(json.dumps({"record": "r1", "label": "yes"}) + "\n" + json.dumps({"record": "r2", "label": "no"}) + "\n")
            self.assertEqual(_gliner_answers("column.group", Path(tmp)), {"r1": True, "r2": False})
            self.assertEqual(_gliner_answers("feature.leakage_suspect", Path(tmp)), {})


if __name__ == "__main__":
    unittest.main()
