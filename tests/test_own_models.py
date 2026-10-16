"""Package 26.7: a company's own decision models, exported, fine-tuned on DCLab's base, gated and served in its workspace."""

import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import pandas  # noqa: F401
    import sklearn  # noqa: F401

    HAVE_ML = True
except ImportError:  # .venv-agent
    HAVE_ML = False

POINT = "column.is_identifier"
NAMES = [("order_id", ["order", "id"]), ("product_code", ["product", "code"]), ("sku_code", ["sku", "code"]), ("customer_id", ["customer", "id"]),
         ("amount", ["amount"]), ("city", ["city"]), ("ticket_no", ["ticket", "no"]), ("zone", ["zone"])]


def _state(name, tokens, unique):
    return {"column": name, "name_tokens": tokens, "dtype": "string" if "code" in tokens else "integer", "uniqueness": unique,
            "nulls": "none", "cardinality": "very_high" if unique == "all" else "low"}


def company_says(tokens, unique):
    """This company's convention: a unique code is a product's code, never an identifier (DCLab's base says it is one)."""
    return unique == "all" and "code" not in tokens


def base_model():
    from dclab_core.answerers import ClassifierAnswerer

    rows = [(_state(n, t, u), u == "all") for n, t in NAMES for u in ("all", "under_1pct", "from_1_to_10pct")]
    model = ClassifierAnswerer(POINT, name="classifier", version="base1").fit(rows * 4)
    return model


@unittest.skipUnless(HAVE_ML, "needs the ML environment (.venv)")
class OwnModelTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd import ledger

        self.home = Path(tempfile.mkdtemp())
        self.book = ledger.FileLedger(self.home / "ledger", "wtest")
        rng = random.Random(0)
        for p in range(60):
            for name, tokens in NAMES:
                unique = rng.choice(["all", "all", "under_1pct"])
                rule = unique == "all"  # the rules (and the base) call every unique column an identifier
                truth = company_says(tokens, unique)
                corrected = truth != rule and p % 2 == 0  # half the corrections were made by a person in the studio: value_used is theirs
                record = ledger.decide(POINT, ("column", name), answers=[ledger.answer("rule", "rule:id", rule)],
                                       value_used=truth if corrected else rule, used="person" if corrected else "rule",
                                       state=_state(name, tokens, unique), options=["yes", "no"], project=f"p{p:03d}", ledger=self.book)
                if truth != rule:
                    ledger.label(record, truth, "user_decision_exposed", "analyst", ledger=self.book)
                else:
                    ledger.label(record, truth, "blind", "reviewer", weight=1.0, ledger=self.book)

    def test_export_keeps_whole_projects_in_one_split_and_no_sealed_row_in_the_chat_rows(self):
        from dclab_rnd import own_models

        manifest = own_models.export(self.home, self.book)
        info = manifest["points"][POINT]
        self.assertGreater(info["rows"], 400)  # a person's own choice is not also its label: those decisions carry none
        self.assertLess(info["rows"], 60 * len(NAMES))
        rows = own_models.rows(self.home, POINT)
        self.assertEqual(len({(r["group"], r["split"]) for r in rows}), len({r["group"] for r in rows}))
        self.assertGreater(info["by_split"].get("sealed", 0), 0)
        chat = (own_models.root(self.home) / "sets" / POINT / "chat.jsonl").read_text().splitlines()
        self.assertEqual(len(chat), info["rows"] - info["by_split"]["sealed"])
        self.assertTrue(all(json.loads(c)["metadata"]["split"] != "sealed" for c in chat))

    def test_fine_tuned_on_the_base_it_learns_the_companys_convention_and_the_base_is_untouched(self):
        from dclab_rnd import own_models

        base = base_model()
        before = base.model.predict_proba([__import__("dclab_core.answerers", fromlist=["features"]).features(_state("sku_code", ["sku", "code"], "all"))])[0].tolist()
        own_models.export(self.home, self.book)
        with mock.patch.object(own_models, "base_model", return_value=base):
            got = own_models.fit(self.home)
        card = got["cards"][POINT]
        self.assertEqual(card.name, "own")
        self.assertIn("stacked on DCLab's base model classifier@base1", card.chosen_by)
        self.assertTrue(any(f.startswith("base:") for f in card.fields))
        model = own_models.load(self.home)[0]
        code = model.answer(POINT, _state("sku_code", ["sku", "code"], "all"), ["yes", "no"])
        ident = model.answer(POINT, _state("order_id", ["order", "id"], "all"), ["yes", "no"])
        self.assertEqual((code.value, ident.value), (False, True))
        self.assertEqual(base.version, "base1")
        after = base.model.predict_proba([__import__("dclab_core.answerers", fromlist=["features"]).features(_state("sku_code", ["sku", "code"], "all"))])[0].tolist()
        self.assertEqual(before, after)

    def test_gated_once_on_its_own_sealed_decisions_and_then_a_candidate_in_its_workspace_only(self):
        from dclab_rnd import own_models

        own_models.export(self.home, self.book)
        with mock.patch.object(own_models, "base_model", return_value=base_model()):
            own_models.fit(self.home)
        got = own_models.gate(self.home)
        doc = got["judged"][POINT]
        self.assertEqual(doc["gate"]["version"], "decision_gate_v1")
        self.assertGreater(doc["sealed"]["decisions"], 0)
        rows = own_models.rows(self.home, POINT)
        sealed = [r for r in rows if r["split"] == "sealed"]
        self.assertEqual(doc["verdict"]["rule"]["accuracy"], round(sum(own_models.rule_answer(r) == r["label"]["value"] for r in sealed) / len(sealed), 4))
        self.assertLess(doc["verdict"]["rule"]["accuracy"], 1.0)  # the rule's own answers, not the person's corrections
        self.assertEqual(doc["result"], "pass")  # the company's convention, learned, beats the rule on its sealed projects
        again = own_models.gate(self.home)
        self.assertIn("read once", again["refused"][POINT])
        passed = own_models.passed(self.home)
        if doc["result"] == "pass":
            self.assertEqual([info["point"] for info in passed.values()], [POINT])
        else:
            self.assertEqual(passed, {})
        self.assertEqual(own_models.passed(Path(tempfile.mkdtemp())), {})  # another workspace sees none of it

    def test_a_passed_own_model_is_routed_in_its_workspace_only_and_answers_there(self):
        from dclab_rnd import own_models
        from dclab_rnd.ledger import routing

        class Store:
            doc = {}

            def get(self):
                return dict(self.doc)

            def save(self, doc):
                self.doc = doc
                return doc

        own_models.export(self.home, self.book)
        own_models.fit(self.home)
        model = own_models.load(self.home)[0]
        name = f"{model.name}@{model.version}"
        results = own_models.root(self.home) / "results"
        results.mkdir(parents=True)
        (results / f"{POINT}.json").write_text(json.dumps({"point": POINT, "result": "pass", "model": {"name": "own", "version": model.version},
                                                          "gate": {"version": "decision_gate_v1"}, "verdict": {}}))
        store = Store()
        with self.assertRaises(routing.DecisionRoutingError):  # another workspace (none given): not a candidate
            routing.switch(store, None, POINT, name, 1, "owner", "our own model passed")
        routing.switch(store, None, POINT, name, 1, "owner", "our own model passed", home=self.home)
        with self.assertRaises(routing.DecisionRoutingError):  # acting needs its shadow measurement first
            routing.switch(store, None, POINT, name, 3, "owner", "act", home=self.home)
        doc = {"decisions": {POINT: {"answerer": name, "level": 3}}}
        with mock.patch.object(own_models, "current", return_value=[model]):
            value, used, level, _, _ = routing.answer(doc, POINT, _state("sku_code", ["sku", "code"], "all"), ["yes", "no"], None)
        self.assertEqual((value, used, level), (False, "classifier", 3))

    def test_a_model_file_swapped_in_the_folder_is_never_loaded(self):
        from dclab_rnd import own_models

        own_models.export(self.home, self.book)
        own_models.fit(self.home)
        (own_models.root(self.home) / "models" / f"{POINT}.joblib").write_bytes(b"swapped")
        self.assertEqual(own_models.load(self.home), [])

    def test_too_few_projects_get_no_model_and_say_why(self):
        from dclab_rnd import own_models

        own_models.export(self.home, self.book)
        path = own_models.root(self.home) / "sets" / POINT / "rows.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        keep = [r for r in rows if r["group"] == rows[0]["group"]]
        path.write_text("".join(json.dumps(r) + "\n" for r in keep))
        got = own_models.fit(self.home)
        self.assertIn(POINT, got["skipped"])


if __name__ == "__main__":
    unittest.main()
