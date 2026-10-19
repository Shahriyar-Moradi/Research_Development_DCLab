"""Package 30.16: the matching kind.

The library keeps a record and every record it matches on one side of the split, forms candidate pairs by blocking
(and says how many true pairs blocking kept), compares fields by kind, chooses a model on the validation records and
reads the holdout once, beside the exact rule and the single best field; the kind refuses a solution that compares a
record's id or forms every pair; the engine runs it as its own machine on the copied bills, G4 passes, its code
reproduces the holdout; and the router chooses it only where a table of known pairs is shown.
"""

import sys
import unittest.mock
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
    import pydantic  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""


def people(n=600, seed=0):
    """A table of people with planted duplicates (a typo in the name, the same city, a birth year one off or the same)."""
    rng = np.random.default_rng(seed)
    first = ["anna", "omid", "sara", "leila", "reza", "maria", "john", "nina", "pedro", "yuki", "amir", "zara"]
    last = ["moradi", "smith", "garcia", "tanaka", "rossi", "karimi", "novak", "silva", "ahmadi", "kim"]
    rows = [{"id": f"p{i}", "name": f"{rng.choice(first)} {rng.choice(last)} {i % 97}", "city": str(rng.choice(["tehran", "paris", "lima", "osaka"])),
             "born": int(rng.integers(1950, 2005))} for i in range(n)]
    pairs = []
    for i in rng.choice(n, n // 5, replace=False):
        src = rows[i]
        name = list(src["name"])
        j = int(rng.integers(0, len(name)))
        name[j] = "x" if name[j] != "x" else "y"
        dup = {"id": f"d{i}", "name": "".join(name), "city": src["city"], "born": src["born"] + int(rng.integers(0, 2))}
        rows.append(dup)
        pairs.append((dup["id"], src["id"]))
    return pd.DataFrame(rows), pd.DataFrame(pairs, columns=["duplicate", "original"])


FIELDS = [{"column": "name", "kind": "text"}, {"column": "city", "kind": "category", "key": False}, {"column": "born", "kind": "number", "key": False}]


@unittest.skipIf(MISSING, MISSING)
class LibraryTests(unittest.TestCase):
    def test_entities_stay_whole_blocking_is_measured_and_the_model_beats_the_exact_rule(self):
        from dclab_core import matching

        table, pairs = people()
        ctx = matching.setup(table, id_left="id", fields=FIELDS, matches=pairs, block_text="name", top_k=5)
        part = dict(zip(ctx["left"]["__key"], ctx["left"]["__part"]))
        self.assertTrue(all(part[a] == part[b] for a, b in ctx["matches"]))  # a record and its match never on two sides
        self.assertEqual(sum(ctx["split"][p]["matches"] for p in matching.PARTS), len(pairs))
        cand = matching.candidates(ctx, "validation")
        self.assertGreaterEqual(cand["blocking_recall"], 0.9)  # the true pairs blocking keeps, measured
        picked = matching.validate(ctx)
        self.assertIn(picked["chosen"], matching.MODELS)
        held = matching.score(ctx, picked["chosen"], picked["scores"][picked["chosen"]].get("threshold"))
        self.assertGreater(held["pair_f1"], held["scores"]["exact"]["f1"])  # a typo defeats the exact rule, not the model
        self.assertGreaterEqual(held["pair_f1"], 0.7)

    def test_two_catalogues_link_and_a_repeated_id_is_refused(self):
        from dclab_core import matching

        table, pairs = people(400, seed=1)
        left = table[table["id"].str.startswith("p")].reset_index(drop=True)
        right = table[table["id"].str.startswith("d")].reset_index(drop=True)
        out = matching.evaluate(left, right, id_left="id", id_right="id", fields=FIELDS, matches=pairs[["original", "duplicate"]], block_text="name", top_k=5)
        self.assertGreaterEqual(out["holdout"]["pair_f1"], 0.7)
        self.assertGreaterEqual(out["holdout"]["blocking_recall"], 0.9)
        with self.assertRaisesRegex(ValueError, "repeated"):
            matching.setup(pd.concat([left, left.head(1)]), right, id_left="id", id_right="id", fields=FIELDS, matches=pairs, block_text="name")

    def test_a_block_too_large_is_said_never_cut(self):
        from dclab_core import matching

        table, pairs = people(300)
        ctx = matching.setup(table, id_left="id", fields=FIELDS, matches=pairs, block_equal=["city"])
        with unittest.mock.patch.object(matching, "MAX_PAIRS", 50), self.assertRaisesRegex(ValueError, "block on more columns"):
            matching.candidates(ctx, "train")

    def test_the_holdout_is_compared_with_every_record_and_its_pairs_never_cross_a_match(self):
        from dclab_core import matching

        table, pairs = people()
        ctx = matching.setup(table, id_left="id", fields=FIELDS, matches=pairs, block_text="name", top_k=5)
        held = matching.candidates(ctx, "holdout")
        self.assertEqual(set(held["R"]["__part"]), set(matching.PARTS))  # a new record is checked against all the others (the review)
        self.assertEqual(set(held["L"]["__part"]), {"holdout"})
        parts = dict(zip(ctx["left"]["__key"], ctx["left"]["__part"]))
        crossing = [(a, b) for a, b, t in zip(held["L"]["__key"].to_numpy()[held["li"]], held["R"]["__key"].to_numpy()[held["ri"]], held["y"])
                    if parts[a] != parts[b]]
        self.assertTrue(crossing)  # pairs with earlier records exist, and are never a known match: an entity is whole on one side
        self.assertFalse(any(t for a, b, t in zip(held["L"]["__key"].to_numpy()[held["li"]], held["R"]["__key"].to_numpy()[held["ri"]], held["y"])
                             if parts[a] != parts[b]))

    def test_the_threshold_is_the_best_of_every_score_and_rare_pairs_are_reached(self):
        from dclab_core import matching

        rng = np.random.default_rng(0)
        y = np.zeros(20_000, dtype=int)
        y[rng.choice(20_000, 20, replace=False)] = 1  # twenty true pairs in twenty thousand
        s = rng.uniform(0, 0.9, 20_000) + 0.2 * y
        t, m = matching._best_threshold(y, s, int(y.sum()))
        brute = max(matching._metrics(y, s >= v, int(y.sum()))["f1"] for v in np.unique(s))
        self.assertEqual(m["f1"], round(brute, 4))

    def test_swapped_or_unknown_matches_are_refused_and_float_ids_are_ids(self):
        from dclab_core import matching

        table, pairs = people(400, seed=1)
        left = table[table["id"].str.startswith("p")].reset_index(drop=True)
        right = table[table["id"].str.startswith("d")].reset_index(drop=True)
        with self.assertRaisesRegex(ValueError, "in that order"):  # the review: the mapping's columns swapped gave F1 0 silently
            matching.setup(left, right, id_left="id", id_right="id", fields=FIELDS, matches=pairs, block_text="name")
        numbered = table.assign(id=range(len(table)))
        mapping = dict(zip(table["id"], numbered["id"]))
        floats = pd.DataFrame({"a": pairs["duplicate"].map(mapping).astype(float), "b": pairs["original"].map(mapping).astype(float)})
        ctx = matching.setup(numbered, id_left="id", fields=FIELDS, matches=floats, block_text="name")
        self.assertEqual(ctx["matches_missing"], 0)  # "12.0" is the id 12

    def test_two_copies_of_one_record_match_each_other_and_text_neighbours_go_both_ways(self):
        from dclab_core import matching

        table = pd.DataFrame({"id": [f"r{i}" for i in range(10)], "name": ["alpha beta", "alpha betx", "alpha betz", "gamma delta", "gamma deltx",
                                                                           "omega sigma", "omega sigmx", "kappa", "zeta", "theta"]})
        pairs = pd.DataFrame({"copy": ["r1", "r2", "r4", "r6"], "original": ["r0", "r0", "r3", "r5"]})  # three groups: one a part
        for seed in range(200):  # the first seed whose three parts each hold a match (a part without one is refused)
            try:
                ctx = matching.setup(table, id_left="id", fields=[{"column": "name"}], matches=pairs, block_text="name", top_k=1, seed=seed)
                break
            except ValueError:
                continue
        self.assertIn(("L:r1", "L:r2"), ctx["matches"])  # copies of one original are the same record too
        everything = {p: matching.candidates(ctx, p) for p in matching.PARTS}
        found = {tuple(sorted((a, b))) for c in everything.values() for a, b, t in zip(c["L"]["__key"].to_numpy()[c["li"]], c["R"]["__key"].to_numpy()[c["ri"]], c["y"]) if t}
        originals = {tuple(sorted(("L:" + a, "L:" + b))) for a, b in zip(pairs["copy"], pairs["original"])}
        self.assertTrue(originals <= found)  # each copy is its original's nearest or the other way round: kept with one neighbour (both ways)
        dropped = len(ctx["matches"]) - len(found)
        self.assertEqual(sum(c["true_pairs"] - c["true_in_blocks"] for c in everything.values()), dropped)  # what blocking dropped is counted


@unittest.skipIf(MISSING, MISSING)
class KindTests(unittest.TestCase):
    def test_the_kind_refuses_a_compared_id_and_every_pair(self):
        from dclab_rnd.studio.solution import Solution

        base = {"kind": "matching", "task": "binary", "prediction_moment": "When a bill is checked, after both were sold."}
        good = {"id_column": "bill_id", "fields": [{"column": "amount", "kind": "number"}], "block_equal": ["items"]}
        Solution(**base, kind_fields=good)
        with self.assertRaisesRegex(ValueError, "never a field"):
            Solution(**base, kind_fields={**good, "fields": [{"column": "bill_id"}]})
        with self.assertRaisesRegex(ValueError, "too many"):
            Solution(**base, kind_fields={**good, "block_equal": []})
        with self.assertRaisesRegex(ValueError, "judged by"):
            Solution(**base, kind_fields=good, metric="roc_auc")
        with self.assertRaisesRegex(ValueError, "forbidden or an identifier"):  # rule 1: a forbidden column never reaches a pair
            Solution(**base, kind_fields=good, forbidden=[{"column": "amount", "reason": "written after the review"}])
        from dclab_rnd import kinds

        kind = kinds.get("matching")
        self.assertIn("title", kind.columns_of({"kind_fields": {**good, "block_text": "title"}}))  # every block column checked up front

    def test_cleaning_words_never_detect_matching(self):
        from dclab_rnd import kinds

        for words in ("Predict which customers will churn next month; the export has some duplicate rows.",
                      "Rank products for each customer; deduplicate the catalogue first.",
                      "Segment our shoppers; remove duplicate customers before you do.",
                      "How long until a machine fails? Some duplicate records are in the log."):
            self.assertNotEqual(kinds.detect(words)["key"], "matching", words)
        self.assertEqual(kinds.detect("Find the copied bills our cashiers return for cash.")["key"], "matching")

    def test_the_engine_runs_it_on_the_copied_bills_g4_passes_and_its_code_reproduces_the_holdout(self):
        from unittest import mock

        from dclab_rnd import kinds
        from dclab_rnd.expansion import runner
        from dclab_rnd.kinds import acceptance
        from dclab_rnd.models.gateway import private_scope
        from dclab_rnd.studio import code, engine

        kind = kinds.get("matching")
        case = kind.case(0)
        store, pid = acceptance.build(case, Path(tempfile.mkdtemp()))
        with runner.fast_profile(), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
            engine.run_all(store, pid)
        records = store.records(pid)
        for c in kind.accept_engine(acceptance.Run(case, store.get(pid)["solution"], records)):
            self.assertTrue(c["ok"], c)
        final = records["final"]["evidence"]
        self.assertTrue(final["gate_g4"]["passed"], final["gate_g4"])
        self.assertIsNotNone(final["task_assessment"]["threshold"]["value"])  # G4's rule: the threshold chosen on validation
        exported = code.export(store, pid, Path(tempfile.mkdtemp()) / "code", check=True)["checked"]
        self.assertTrue(exported["same"], exported)

    def test_the_router_chooses_matching_only_where_a_table_of_known_pairs_is_shown(self):
        import warnings

        from dclab_core import datacard as dc, problem_map as pm, solutions as sol

        warnings.filterwarnings("ignore")
        table, pairs = people(300)
        sub = pm.SubProblem(id="sp1", part="part1", title="The problem", family="rare_event", labels="file")
        m = pm.ProblemMap(size="single", parts=[pm.Part(id="part1", title="The problem")], sub_problems=[sub])
        with_pairs = dc.card_of({"table": table, "pairs": pairs}, name="people", roles={"pairs": "related"})
        self.assertEqual(sol.route(sub, with_pairs, m).chosen, "matching")
        alone = dc.card_of({"table": table}, name="people")
        self.assertNotEqual(sol.route(sub, alone, m).chosen, "matching")  # no table of known pairs: a rare event as before
        customers = pd.DataFrame({"customer_id": range(30), "age": range(30)})
        orders = pd.DataFrame({"order_id": range(100), "customer_id": [i % 30 for i in range(100)],
                               "ordered_at": pd.date_range("2024-01-01", periods=100, freq="D"), "amount": range(100)})
        cases = pd.DataFrame({"case_id": [f"c{i}" for i in range(20)], "customer_id": range(20)})
        for family in ("rare_event", "anomaly", "entity_tables"):  # the review: an orders table, a file of cases, routed to matching
            sub2 = pm.SubProblem(id="sp1", part="part1", title="The problem", family=family, labels="none" if family == "anomaly" else "file")
            m2 = pm.ProblemMap(size="single", parts=[pm.Part(id="part1", title="The problem")], sub_problems=[sub2])
            for name, frame in (("orders", orders), ("cases", cases)):
                card = dc.card_of({"table": customers, name: frame}, name="shop", roles={name: "related"})
                self.assertNotEqual(sol.route(sub2, card, m2).chosen, "matching", (family, name))


if __name__ == "__main__":
    unittest.main()
