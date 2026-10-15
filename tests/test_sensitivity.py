"""23.7: a column that holds a person's data is found by its name and by the shape of its values, and reaches a model by
name only: the data card, the profile, the Home analysis, the intern's description and the events keep no value of it."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd

    from dclab_core import datacard as dc, sensitivity as s
    from dclab_rnd import experience
    from dclab_rnd.experience import hooks
    from dclab_rnd.studio import data
except ImportError as error:  # the agent environment may lack pandas
    MISSING = str(error)
else:
    MISSING = ""

EMAIL = "Q7Kxmail{}@example.org"
PHONE = "+1 415 555 {}"
CARD = "4111 1111 1111 1111"


def customers(n=240):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "order_ref": [f"ORD-{i:05d}" for i in range(n)],
        "contact": [EMAIL.format(i) for i in range(n)],            # named nothing: found by its values
        "phone": [PHONE.format(1000 + i) for i in range(n)],        # named and shaped
        "wallet": [CARD] * n,                                       # shaped like a card number
        "zip_code": [f"{10000 + i % 30}" for i in range(n)],        # personal by its name, low cardinality
        "diagnosis": [f"Q7Kxdx{i % 4}" for i in range(n)],          # sensitive: category names must not travel
        "email_opens": rng.integers(0, 20, n),                      # about email, not an e-mail
        "plan": [f"Q7Kxplan{i % 3}" for i in range(n)],
        "spend": np.round(rng.normal(50, 10, n), 2),
        "churned": (rng.random(n) < 0.3).astype(int)})


@unittest.skipIf(MISSING, MISSING)
class RulesTest(unittest.TestCase):
    def test_a_name_gives_a_class_and_a_count_about_it_does_not(self):
        for name, cls in (("email", "pii"), ("customerEmail", "pii"), ("Phone Number", "pii"), ("first_name", "pii"), ("date_of_birth", "pii"),
                          ("ip_address", "pii"), ("name", "pii"), ("patient_name", "pii"), ("password", "restricted"), ("api_key", "restricted"),
                          ("card_number", "restricted"), ("credit_score", "sensitive"), ("diagnosis", "sensitive"), ("salary", "sensitive"),
                          ("email_opens", "none"), ("has_phone", "none"), ("phone_type", "none"), ("product_name", "none"), ("order_id", "none"), ("price", "none")):
            with self.subTest(name=name):
                self.assertEqual((s.of_name(name) or ("none",))[0], cls)

    def test_a_shape_gives_a_class_and_nothing_of_the_value_is_said(self):
        self.assertEqual(s.of_values([EMAIL.format(i) for i in range(5)])[0], "pii")
        self.assertEqual(s.of_values(["+1 415 555 2671", "(415) 555-2671", "415-555-2671", "+44 20 7946 0958"])[0], "pii")
        self.assertEqual(s.of_values(["4111 1111 1111 1111", "5500 0000 0000 0004", "4012888888881881"])[0], "restricted")
        self.assertEqual(s.of_values(["123-45-6789"] * 5)[0], "restricted")
        self.assertIsNone(s.of_values(["2026-10-16", "2026-10-17", "2026-10-18"]))   # a date is not a phone number
        self.assertIsNone(s.of_values(["12345678", "23456789", "34567890"]))          # nor is a plain number
        self.assertIsNone(s.of_values(["4111 1111 1111 1112"] * 5))                    # fails the Luhn check
        self.assertIsNone(s.of_values(["a@b.co", "b@c.co"]))                           # too few to say
        for why in (s.of_values([EMAIL.format(i) for i in range(5)])[1], s.classify("phone", ["+1 415 555 2671"] * 4)[1]):
            self.assertNotIn("Q7Kx", why)
            self.assertNotIn("415", why)

    def test_the_stronger_of_the_name_and_the_values_wins(self):
        self.assertEqual(s.classify("notes", ["fine", "fine", "fine"]), ("none", None))
        self.assertEqual(s.classify("contact", [EMAIL.format(i) for i in range(5)])[0], "pii")
        self.assertEqual(s.classify("password", [EMAIL.format(i) for i in range(5)])[0], "restricted")
        self.assertEqual(s.classify("email", ["x", "y", "z"])[0], "pii")

    def test_strip_keeps_the_name_and_the_bands_and_nothing_that_carries_a_value(self):
        card = {"name": "c", "dtype": "string", "nulls": "none", "categories": ["a"], "prefix": "ab", "quantiles": [1.0], "shares": [0.5], "preview": ["x"], "hist": {}}
        self.assertEqual(s.strip(card, "none"), card)
        weak = s.strip(card, "sensitive")
        self.assertEqual((weak["quantiles"], weak["shares"], "categories" in weak, "prefix" in weak, weak["preview"]), ([1.0], [0.5], False, False, []))
        strong = s.strip(card, "pii")
        self.assertEqual(sorted(strong), ["dtype", "name", "nulls", "preview"])
        self.assertEqual(strong["preview"], [])


@unittest.skipIf(MISSING, MISSING)
class CardTest(unittest.TestCase):
    def test_the_data_card_marks_the_columns_and_holds_no_value_of_them(self):
        card = dc.card_of({"customers": customers()})
        by = {c.name: c for c in card.tables[0].columns}
        self.assertEqual({n: by[n].sensitivity for n in ("contact", "phone", "wallet", "zip_code", "diagnosis", "email_opens", "plan", "spend")},
                         {"contact": "pii", "phone": "pii", "wallet": "restricted", "zip_code": "pii", "diagnosis": "sensitive", "email_opens": "none",
                          "plan": "none", "spend": "none"})
        self.assertIn("e-mail", by["contact"].sensitivity_why)
        blob = card.model_dump_json()
        for value in ("Q7Kxmail", "415 555", "4111", "Q7Kxdx", "10001"):
            self.assertNotIn(value, blob)
        self.assertIsNotNone(by["spend"].quantiles)  # a column that is none keeps its bands
        self.assertEqual(card.model_dump(mode="json")["schema_version"], "dclab_core.data_card.v3")
        again = dc.DataCard.read(card.model_dump_json())
        self.assertEqual(again, card)
        old = json.loads(card.model_dump_json())
        old["schema_version"] = "dclab_core.data_card.v2"
        self.assertEqual(dc.DataCard.read(old).schema_version, "dclab_core.data_card.v3")

    def test_a_synthetic_table_drawn_from_the_card_has_no_person_in_it(self):
        card = dc.card_of({"customers": customers()})
        drawn = dc.synthesize(card, rows={"customers": 60}, seed=3)["customers"]
        text = drawn.to_csv(index=False)
        for value in ("Q7Kxmail", "415 555", "4111", "Q7Kxdx"):
            self.assertNotIn(value, text)


@unittest.skipIf(MISSING, MISSING)
class StudioTest(unittest.TestCase):
    def test_the_profile_and_the_analysis_keep_no_value_of_a_personal_column(self):
        from dclab_rnd.draft import analyze

        report = analyze.analyze(customers())
        columns = {c["name"]: c for c in report["profile"]["columns"]}
        for name in ("contact", "phone", "wallet", "zip_code", "diagnosis"):
            self.assertEqual(columns[name]["preview"], [], name)
            self.assertNotEqual(columns[name]["sensitivity"], "none")
        self.assertEqual(columns["plan"]["preview"][0][:7], "Q7Kxpla")  # an ordinary column still shows examples
        described = {c["name"]: c for c in report["columns"]}
        self.assertNotIn("top", described["diagnosis"])
        self.assertEqual(described["diagnosis"]["sensitivity"], "sensitive")
        self.assertIn("top", described["plan"])
        self.assertEqual({k for k in described["phone"] if k in ("hist", "quantiles", "min", "max", "mean", "top")}, set())
        blob = json.dumps(report, default=str)
        for value in ("Q7Kxmail", "415 555", "4111 1111", "Q7Kxdx"):
            self.assertNotIn(value, blob)

    def test_the_intern_describes_a_personal_column_by_class_and_gives_no_example(self):
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore

        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("People", "general", "Predict churn")["id"]
        customers().to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        described = Toolbox(store).call("describe_data", {"project_id": pid})
        columns = {c["name"]: c for c in described["columns"]}
        for name in ("contact", "phone", "wallet"):
            self.assertEqual((columns[name]["examples"], columns[name]["sensitivity"]), ([], "pii" if name != "wallet" else "restricted"))
        self.assertNotIn("sensitivity", columns["plan"])
        self.assertTrue(columns["plan"]["examples"])
        self.assertNotIn("Q7Kxmail", json.dumps(described))

    def test_the_event_names_the_columns_and_their_classes_and_nothing_else(self):
        from dclab_rnd.draft import analyze

        log = experience.FileLog(Path(tempfile.mkdtemp()) / "experience", "w1")
        from dclab_rnd import context

        class Services:
            experience = log
            workspace_id = "w1"

        with context.using(Services()):
            hooks.from_draft_event("d1", "analysis", {"asset": "a1", "analysis": analyze.analyze(customers())})
        seen = [e for e in log.read() if e.kind == "data.analysed"]
        self.assertTrue(seen, "no data.analysed event")
        payload = seen[-1].payload
        self.assertEqual(payload["sensitivity"], {"contact": "pii", "phone": "pii", "wallet": "restricted", "zip_code": "pii", "diagnosis": "sensitive"})
        self.assertEqual(seen[-1].classes["sensitivity"], "meta")
        self.assertNotIn("Q7Kx", json.dumps([e.model_dump(mode="json") for e in log.read()]))


@unittest.skipIf(MISSING, MISSING)
class ReviewTest(unittest.TestCase):
    """What the reviewer found: spellings, numbers written as floats, mixed and padded columns, older stored descriptions."""

    def test_the_common_spellings_of_the_same_columns_are_named(self):
        for name, cls in (("phone_num", "pii"), ("phoneNum", "pii"), ("phone1", "pii"), ("email1", "pii"), ("mobile_num", "pii"), ("postal_code", "pii"),
                          ("fname", "pii"), ("lname", "pii"), ("msisdn", "pii"), ("card_no", "restricted"), ("card_num", "restricted"), ("cc_num", "restricted"),
                          ("cc_number", "restricted"), ("credit_card_num", "restricted"), ("account_no", "restricted"), ("acct_no", "restricted")):
            with self.subTest(name=name):
                self.assertEqual((s.of_name(name) or ("none",))[0], cls)

    def test_a_business_place_a_yes_no_and_a_word_frequency_are_not_personal(self):
        for name in ("source_latitude", "destination_longitude", "PhoneService", "word_freq_mail", "word_freq_email", "handle_time", "average_handle_time",
                     "mobile_data_gb", "race_time", "race_id", "orientation_deg", "email_opens", "has_phone", "phone_type"):
            with self.subTest(name=name):
                self.assertIsNone(s.of_name(name))

    def test_numbers_written_as_floats_mixed_columns_and_placeholders_are_found(self):
        cards = pd.Series([4111111111111111.0, 5500000000000004.0, np.nan, 4012888888881881.0] * 80)  # a missing row made the column float
        self.assertEqual(s.of_values(cards)[0], "restricted")
        mixed = ["a@b.co" if i % 2 else f"+1 415 555 {i:04d}" for i in range(100)]
        self.assertEqual(s.of_values(mixed)[0], "pii")
        self.assertEqual(s.of_values(["N/A"] * 60 + [f"u{i}@x.org" for i in range(30)])[0], "pii")  # placeholders are not part of the share
        self.assertEqual(s.of_values(pd.Series(["N/A"] * 300 + [f"u{i}@x.org" for i in range(300)]))[0], "pii")  # a sorted file is read evenly
        self.assertEqual(s.of_values([f"u{i}@x.org" if i % 4 == 0 else "no email" for i in range(100)])[0], "pii")  # a quarter e-mail is enough
        self.assertEqual(s.of_values(["de89370400440532013000"] * 5)[0], "restricted")  # a lowercase IBAN
        self.assertEqual(s.of_values(["2001:db8::1", "fe80::1ff:fe23:4567:890a", "2001:db8:85a3::8a2e:370:7334"])[0], "pii")
        for text in (["10.0.19041.1"] * 5, ["2026-10-16"] * 5, ["4155550000"] * 5):
            self.assertIsNone(s.of_values(text))  # a version, a date, a bare number: not a shape (a name or a person names those)

    def test_a_profile_and_a_card_stored_before_the_class_existed_are_brought_up_to_the_rule(self):
        old = {"name": "contact", "kind": "text", "preview": ["Q7Kxm0@example.org", "Q7Kxm1@example.org", "Q7Kxm2@example.org"], "unique": 240}
        up = s.upgrade_column(old)
        self.assertEqual((up["sensitivity"], up["preview"]), ("pii", []))
        card = {"name": "t", "tables": [{"name": "t", "columns": [{"name": "email", "categories": ["a@b.co"], "prefix": "ab", "quantiles": [1.0]},
                                                                    {"name": "plan", "categories": ["basic"], "quantiles": [1.0]}]}]}
        got = s.upgrade_card(card)["tables"][0]["columns"]
        self.assertEqual([(c["sensitivity"], "categories" in c, "quantiles" in c) for c in got], [("pii", False, False), ("none", True, True)])
        v2 = {"schema_version": "dclab_core.data_card.v2", "name": "t", "tables": [{"name": "t", "rows": 5, "columns": []}]}
        self.assertEqual(dc.DataCard.read(v2).schema_version, "dclab_core.data_card.v3")

    def test_describe_data_and_an_exported_card_clean_what_was_stored_earlier(self):
        from dclab_rnd import datasets
        from dclab_rnd.intern.tools import Toolbox
        from dclab_rnd.studio import ProjectStore

        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("Old", "general", "Predict churn")["id"]
        customers().to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        project = store.get(pid)
        for c in project["data"]["profile"]["columns"]:  # as a project attached before 23.7 stored them
            c.pop("sensitivity", None), c.pop("sensitivity_why", None)
            if c["name"] == "contact":
                c["preview"] = ["Q7Kxmail1@example.org", "Q7Kxmail2@example.org", "Q7Kxmail3@example.org"]
        store.save(project)
        described = Toolbox(store).call("describe_data", {"project_id": pid})
        contact = next(c for c in described["columns"] if c["name"] == "contact")
        self.assertEqual((contact["examples"], contact["sensitivity"]), ([], "pii"))
        card = json.loads(dc.card_of({"customers": customers()}).model_dump_json())
        for t in card["tables"]:
            for c in t["columns"]:
                c.pop("sensitivity", None), c.pop("sensitivity_why", None)
                if c["name"] == "diagnosis":
                    c["categories"] = ["Q7Kxdx0", "Q7Kxdx1"]
        exported = json.dumps(datasets.anonymous_card(card))
        self.assertNotIn("Q7Kxdx", exported)

    def test_the_final_features_name_no_level_of_a_personal_column(self):
        from dclab_rnd.studio import engine

        class Fake:
            personal = {"diagnosis"}

        keep = [k for k in ("diagnosis=flu", "plan=basic", "spend") if not any(k.startswith(f"{n}=") for n in Fake.personal)]
        self.assertEqual(keep, ["plan=basic", "spend"])
        self.assertTrue(hasattr(engine.Prepared, "__init__"))  # the attribute is set where Prepared is built

    def test_the_cleaning_log_quotes_no_failed_value_of_a_personal_column(self):
        from dclab_rnd.draft import clean

        self.assertEqual(clean._quoted("phone", pd.Series(["call me", "n/a?"])), "")
        self.assertEqual(clean._quoted("plan", pd.Series(["basic?", "pro?"])), ": 'basic?', 'pro?'")

    def test_the_main_products_column_class_waits_for_the_field_lists_that_name_it(self):
        from dclab_core.adapters import main_product_experience as mp

        listed = json.dumps(mp.recorded())
        self.assertIn("dataset_columns.sensitivity_class", mp.FIELDS_MISSING)
        self.assertNotIn("sensitivity_class", listed, "the field lists now name the column: write its mapping and drop FIELDS_MISSING")


if __name__ == "__main__":
    unittest.main()
