"""The kind that is built is the kind that was understood (S9, 2026-10-25; the live first conversation: a forecast the
card read right ran as a regression). Each kind reads the fields its table shows, says what it assumed, and names
what only the person can give; a complete proposal sets the kind, an incomplete one is only suggested; a kind that
reads no outcome is built from the card. S9.3: Home asks a kind's missing field in the kind's words, once; "I don't
know" leads to the stated assumption or to the plain task, said; a field from another table names that table."""

import os
import sys
import tempfile
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
            if kind in ("streams", "ranking"):
                self.assertIsNone(got)  # its incidents, its items, come in their own table: nothing built in its name before
                if kind == "ranking":  # the items attached by their name: the ranking is built
                    case.tables["items"].to_csv(store.data_dir(project["id"]) / "items.csv", index=False)
                    sd.attach_table(store, project["id"], "items.csv", name="items")
                    got = _family_solution(store, store.get(project["id"]), {}, family)
                    self.assertEqual((got["kind"], got["target"]), (kind, None))
                continue
            self.assertEqual((got["kind"], got["target"]), (kind, None))

    def test_home_asks_a_kinds_missing_field_once_and_says_what_follows_from_i_dont_know(self):
        import tempfile

        import pandas as pd

        from dclab_rnd import kinds
        from dclab_rnd.draft import chat
        from dclab_rnd.draft.store import DraftStore
        from dclab_rnd.kinds.propose import QUESTIONS, read_said, said_of
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import propose

        case = kinds.get("uplift").case(0)
        store = DraftStore(Path(tempfile.mkdtemp()))
        agent = chat.HomeAgent(store, None)
        d = store.create("We sent a discount to some customers. Which customers does the offer actually change?")
        case.table.to_parquet(Path(store.data_dir(d["id"])) / "clean.parquet")
        profile = profile_table(case.table)

        def ready(x):
            x["assets"], x["active_asset"] = [{"id": "a1", "status": "ready", "clean_file": "clean.parquet"}], "a1"
            x["analysis"] = {"profile": profile, "columns": [{"name": c} for c in case.table.columns]}
            x.setdefault("understanding", {}).update(target=case.target, prediction_moment="When the offer is sent, using what is known then",
                                                     action="We send the offer to the customers it changes", family="uplift")
        store.update(d["id"], ready)
        agent.next_turn(d["id"])
        d = store.get(d["id"])
        self.assertEqual([q["field"] for q in d["questions"]], ["kind_randomised"])  # in the kind's words, after the card's own
        agent.take_answer(d["id"], agent.pending(d), "I don't know.")
        d = store.get(d["id"])
        self.assertEqual([q["field"] for q in d["questions"]], ["kind_randomised"])  # never asked twice
        self.assertTrue(d["questions"][0]["declined"])
        summary = [m["text"] for m in d["messages"] if m.get("kind") == "summary"][-1]
        self.assertIn("Assumed: not at random", summary)  # the cautious assumption, said: no claim of what the offer caused
        p = propose(case.table, profile, case.target, family="uplift", said=said_of(d["understanding"]))
        self.assertEqual((p["kind"], p["kind_fields"]["randomised"]), ("uplift", False))
        self.assertIn("randomised", p["kind_proposed"]["assumed"])
        q = QUESTIONS[("uplift", "randomised")]
        for words, value in (("we ran an A/B test with a control group", True), ("It was random, not chosen", True),
                             ("no, the sales team picked them", False), ("It was not done at random", False)):
            self.assertEqual(read_said(q, words, pd.DataFrame(), {}), (True, value), words)
        for words in ("It was not an experiment, the sales team chose", "There was no control group", "nobody was left out",
                      "we did not run an A/B test", "no randomisation, we targeted big spenders"):  # the review of S9: read as random
            self.assertEqual(read_said(q, words, pd.DataFrame(), {}), (True, False), words)
        self.assertEqual(read_said(q, "the offer went out last spring", pd.DataFrame(), {}), (False, None))  # never guessed
        self.assertEqual(read_said(q, "random, but the manager picked some", pd.DataFrame(), {}), (False, None))  # both: unread

    def test_a_column_question_offers_names_never_values_and_a_table_need_is_said(self):
        import pandas as pd

        from dclab_rnd import kinds
        from dclab_rnd.kinds.propose import options_for, propose_fields, question_for, still_needed

        frame = pd.DataFrame({"body": [f"a long message about the delivery number {i} that came late again and again" for i in range(40)],
                              "penname": [f"w{i % 5}" for i in range(40)], "label": [i % 2 for i in range(40)]})
        p = propose_fields("embedded", frame, "label")
        q = question_for(p)
        self.assertEqual(q.key, "kind_source_column")
        options = options_for(q, frame, p)
        self.assertTrue(set(options) <= {*frame.columns, "I don't know"})  # names only: no value of the table is shown or kept
        self.assertNotIn("body", options)  # a field already read is not offered again
        done = propose_fields("embedded", frame, "label", {"source_column": "it is the penname column"})
        self.assertEqual((done.complete, done.fields["source_column"]), (True, "penname"))
        declined = propose_fields("embedded", frame, "label", {"source_column": None})
        self.assertFalse(declined.complete)  # no source said: never guessed; the plain task runs, said
        self.assertIn("plain task", still_needed(declined)[0])
        case = kinds.get("ranking").case(0)
        self.assertIn("items_table", propose_fields("ranking", case.table, None, tables=()).missing)
        self.assertNotIn("items_table", propose_fields("ranking", case.table, None, tables=["items"]).missing)
        self.assertIn("`items`", still_needed(propose_fields("ranking", case.table, None, tables=()))[0])
        self.assertIsNone(question_for(propose_fields("ranking", case.table, None, tables=())))  # a table is attached, never asked

    def test_build_it_on_a_forecast_draft_saves_a_forecast_with_the_kinds_metric(self):
        """The review of S9: the proposal's task became the kind's but its metric stayed the plain task's, and "build it"
        dropped the kind: a forecast the card read was built as a regression, or refused for its metric."""
        import time

        from fastapi.testclient import TestClient

        from dclab_rnd import kinds
        from dclab_rnd.agentic.server import create_app

        case = kinds.get("forecasting").case(0)
        client = TestClient(create_app(Path(tempfile.mkdtemp())))
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
        d = client.post("/api/drafts", json={"problem": "Forecast how many units each series sells per day, a week ahead"}, headers=h).json()
        client.put(f"/api/drafts/{d['id']}/data?filename=sales.csv", content=case.table.to_csv(index=False).encode(), headers=h)
        for _ in range(600):
            if any(a["status"] == "ready" for a in client.get(f"/api/drafts/{d['id']}").json()["assets"]):
                break
            time.sleep(0.1)

        def settle(x):
            x.setdefault("understanding", {}).update(target=case.target, family="forecasting",
                                                     prediction_moment="Each evening, from the sales up to that day")
        client.app.state.drafts.update(d["id"], settle)
        proposal = client.post(f"/api/drafts/{d['id']}/solution/proposal", json={"target": case.target}, headers=h).json()
        self.assertEqual((proposal["kind"], proposal["task"], proposal["metric"]), ("forecasting", "regression", "mase"))
        services = client.app.state.services
        with mock.patch.object(type(services), "start_intern_job", return_value=None):
            out = client.post(f"/api/drafts/{d['id']}/build/run", headers=h)
        self.assertEqual(out.status_code, 201, out.text)
        solution = client.get(f"/api/projects/{out.json()['project_id']}").json()["solution"]
        self.assertEqual((solution["kind"], solution["metric"], solution["kind_fields"]["series_column"], solution["time_column"]),
                         ("forecasting", "mase", "series", "date"))

    def test_the_outcome_never_forms_the_segments_and_each_kind_reads_its_own_target(self):
        """The review of S9: Telco's `Churn` was missed by the outcome words and, named as the target, formed the segments;
        a time-to-event proposal on its event read complete and no save could pass; a single series' weekday was taken as
        the series."""
        import numpy as np
        import pandas as pd

        from dclab_rnd import kinds
        from dclab_rnd.kinds.propose import propose_fields, still_needed
        from dclab_rnd.studio.data import profile_table
        from dclab_rnd.studio.solution import Solution, propose

        rng = np.random.default_rng(0)
        telco = pd.DataFrame({"is_male": rng.choice([0, 1], 300), "tenure": rng.integers(1, 70, 300), "MonthlyCharges": rng.normal(60, 20, 300),
                              "Contract": rng.choice(["month", "year"], 300), "Churn": rng.choice(["Yes", "No"], 300)})
        self.assertEqual(propose_fields("segmentation", telco, None).fields["outcome_column"], "Churn")  # not is_male: one slot only
        p = propose(telco, profile_table(telco), "Churn", family="segmentation")
        self.assertEqual((p["kind"], p["target"], p["kind_fields"]["outcome_column"]), ("segmentation", None, "Churn"))
        saved = Solution(target=None, task=p["task"], kind="segmentation", kind_fields=p["kind_fields"], metric=p["metric"],
                         prediction_moment="When the customers are grouped, from what is known then")
        self.assertEqual(saved.kind_fields["outcome_column"], "Churn")  # the machine leaves its outcome column out of the segments
        case = kinds.get("time_to_event").case(0)
        event = case.solution["kind_fields"]["event_column"]
        on_event = propose_fields("time_to_event", case.table, event)
        self.assertFalse(on_event.complete)  # suggested only: its outcome is the duration, said, and no other count named as it
        self.assertNotIn("duration_column", on_event.fields)
        self.assertIn(f"`{event}` as its event", still_needed(on_event)[0])
        self.assertTrue(propose_fields("time_to_event", case.table, case.target).complete)
        days = pd.date_range("2025-01-01", periods=120, freq="D")
        one = pd.DataFrame({"date": days.astype(str), "weekday": days.day_name(), "units": np.arange(120)})
        self.assertNotIn("series_column", propose_fields("forecasting", one, "units").fields)  # one series: asked, never a weekday
        months = pd.date_range("2020-01-01", periods=36, freq="MS")
        many = pd.DataFrame({"month": np.tile(months.astype(str), 5), "store": np.repeat(list("abcde"), 36), "sales": np.arange(180)})
        got = propose_fields("forecasting", many, "sales").fields
        self.assertEqual((got["series_column"], got["season"]), ("store", 12))  # the step read on distinct dates


if __name__ == "__main__":
    unittest.main()
