"""The kind that is built is the kind that was understood (S9, 2026-10-25; the live first conversation: a forecast the
card read right ran as a regression). Each kind reads the fields its table shows, says what it assumed, and names
what only the person can give; a complete proposal sets the kind, an incomplete one is only suggested; a kind that
reads no outcome is built from the card."""

import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

READ = {"segmentation": ("outcome_column",), "embedded": ("source_column", "text_column"), "forecasting": ("series_column",),
        "time_to_event": ("duration_column", "event_column", "start_column"),
        "ranking": ("user_column", "item_column", "time_column", "event_column", "relevant_event"),
        "uplift": ("treatment_column",), "streams": ("unit_column", "time_column", "sensor_columns")}
PERSON_ONLY = {"uplift": {"randomised"}, "streams": {"incident_time_column"}, "entity_tables": {"target_table", "horizon_days", "cutoffs"}}


class KindFromFamilyTest(unittest.TestCase):
    def setUp(self):
        warnings.filterwarnings("ignore")
        env = mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1", "DCLAB_DECISION_SHADOW": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_each_kind_reads_what_its_table_shows_and_names_what_only_the_person_knows(self):
        from dclab_rnd import kinds
        from dclab_rnd.kinds.propose import propose_fields

        for kind in ("segmentation", "embedded", "forecasting", "time_to_event", "ranking", "uplift", "entity_tables", "streams"):
            for seed in (0, 1, 2):
                case = kinds.get(kind).case(seed)
                want = (case.solution or {}).get("kind_fields") or {}
                got = propose_fields(kind, case.table, case.target)
                for name in READ.get(kind, ()):
                    self.assertEqual(got.fields.get(name), want.get(name), f"{kind} seed {seed}: {name}")
                self.assertEqual(set(got.missing), PERSON_ONLY.get(kind, set()), f"{kind} seed {seed}")
        fc = propose_fields("forecasting", kinds.get("forecasting").case(0).table, "units")
        self.assertIn("horizon", fc.assumed)  # how far ahead is the person's: assumed and said, never silent

    def test_a_complete_proposal_sets_the_kind_and_an_incomplete_one_is_only_suggested(self):
        from dclab_rnd import kinds
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import propose

        case = kinds.get("forecasting").case(0)
        plain = propose(case.table, profile_table(case.table), "units")
        self.assertEqual(plain["kind"], "regression")  # without the card's family: today's answer
        p = propose(case.table, profile_table(case.table), "units", family="forecasting")
        self.assertEqual((p["kind"], p["task"], p["kind_fields"]["series_column"], p["time_candidates"][0]), ("forecasting", "regression", "series", "date"))
        self.assertTrue(p["kind_proposed"]["why"])
        case = kinds.get("uplift").case(0)
        u = propose(case.table, profile_table(case.table), "bought", family="uplift")
        self.assertNotEqual(u["kind"], "uplift")  # whether the offer was random is the person's to say: never applied silently
        self.assertEqual(u["kind_suggested"]["missing"], ["randomised"])
        self.assertNotIn("kind_proposed", propose(case.table, profile_table(case.table), "bought", family="classification"))

    def test_a_kind_that_reads_no_outcome_is_built_from_the_card(self):
        import tempfile

        from dclab_rnd import kinds
        from dclab_rnd.draft.api import _family_solution
        from dclab_rnd.studio import ProjectStore
        from dclab_rnd.studio import data as sd

        store = ProjectStore(Path(tempfile.mkdtemp()))
        for kind, family in (("segmentation", "segmentation"), ("ranking", "ranking"), ("streams", "streams")):
            case = kinds.get(kind).case(0)
            project = store.create(kind, goal="test")
            case.table.to_csv(store.data_dir(project["id"]) / "table.csv", index=False)
            sd.attach_data(store, project["id"], "table.csv")
            got = _family_solution(store, store.get(project["id"]), {}, family)
            if kind == "streams":
                self.assertIsNone(got)  # its incidents come in their own table: nothing built in its name
                continue
            self.assertEqual((got["kind"], got["target"]), (kind, None))


if __name__ == "__main__":
    unittest.main()
