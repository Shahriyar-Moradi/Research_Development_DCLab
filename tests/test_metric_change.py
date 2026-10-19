"""Package 30.20: why did this number change (the owner's pattern 6).

On the mall sales replica every planted cause is found and ranked first (the factor that moved, the segment that moved it, mix against rate),
the store whose door counter failed is refused with what is missing (while summing its visitors would say footfall fell), questions the data
cannot answer are refused with what they would need, a question that names its template gets the same answer whatever a model would say,
a model only chooses and fills (its choice validated, or the rules' stands), and the query under every answer is the one that made it.
"""

import sqlite3
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
    import pandas as pd
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""

SEEDS = (0, 1, 2, 3, 4)


def found(truth, a):
    """Whether an answer reaches what the planted cause says a careful analyst finds."""
    if truth.get("refused"):
        return a.status == "refused" and truth["missing"] in a.missing
    if a.status != "answered":
        return False
    if truth["first_factor"] != "any" and a.first_factor != truth["first_factor"]:
        return False
    if truth.get("first_segment") and (a.first_segment or {}).get("value") != truth["first_segment"]["value"]:
        return False
    if truth.get("effect") and (a.first_segment or {}).get("effect") != truth["effect"]:
        return False
    return not (truth["direction"] == "none" and (a.moved or a.first_segment))


class Scripted:
    """An advisor that answers from a list and keeps what it was asked."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def ask(self, purpose, evidence):
        self.asked.append((purpose, evidence))
        return self.answers.pop(0) if self.answers else None


@unittest.skipIf(MISSING, MISSING)
class MetricChangeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dclab_core import metric_change as mc
        from dclab_core.replicas import mall_sales

        cls.mc, cls.layer = mc, mall_sales.layer()
        cls.replicas = {s: mall_sales.generate(s) for s in SEEDS}
        cls.data = {s: mc.Data(cls.layer, {"footfall": r.footfall, "sales": r.sales}) for s, r in cls.replicas.items()}

    def ask(self, question, seed=0, advisor=None):
        return self.mc.ask(question, self.layer, {}, advisor, data=self.data[seed], bootstrap=300)

    def test_every_planted_cause_is_found_and_ranked_first_on_every_seed(self):
        for seed, r in self.replicas.items():
            for truth in r.truth:
                with self.subTest(seed=seed, case=truth["id"]):
                    a, said = self.ask(truth["question"], seed)
                    self.assertTrue(found(truth, a), a.words)
                    self.assertEqual(said["rules"]["template"], truth["template"])
        a, _ = self.ask(self.replicas[0].truth[0]["question"])  # the footfall drop: the factor first, its share of the change, an action
        self.assertEqual([f.name for f in a.factors][0], "footfall")
        self.assertAlmostEqual(sum(f.log_change for f in a.factors), np.log(a.current / a.previous), places=4)  # the factors add up, in logs
        self.assertIn("visitors", a.action)
        self.assertTrue(a.follow_up)

    def test_a_data_gap_is_refused_with_what_is_missing_where_summing_says_footfall_fell(self):
        r = self.replicas[0]
        gap = r.info["planted"]["data_gap"]
        f = r.footfall[(r.footfall["brand"] == gap["brand"]) & (r.footfall["mall"] == gap["mall"])]
        now = f[(f["date"] >= r.period[0]) & (f["date"] <= r.period[1])]["visitors"].sum()
        then = f[(f["date"] >= r.compare[0]) & (f["date"] <= r.compare[1])]["visitors"].sum()
        self.assertLess(now / then, 0.75)  # the trap bites: summed as they are, the store's visitors fell by more than a quarter
        a, _ = self.ask(next(t for t in r.truth if t["id"] == "data_gap")["question"])
        self.assertEqual((a.status, a.missing), ("refused", ["footfall"]))
        lost = [c for c in a.coverage if c.missing or c.zero]
        self.assertEqual(sum(len(c.missing) + len(c.zero) for c in lost), gap["days"])  # every failed day named
        self.assertIsNotNone(a.change_pct)  # the sales change itself is still said: the sales table is whole
        for template in ("metric_change", "metric_value"):  # footfall on its own is refused too: the tills sold while the counter read nothing
            q = f"{template}: metric=footfall; brand={gap['brand']}; mall={gap['mall']}; period={r.period[0][:7]}" + ("; compare=last_year" if template == "metric_change" else "")
            a, _ = self.ask(q)
            self.assertEqual(a.status, "refused", template)
            self.assertIsNone(a.change_pct)

    def test_what_the_data_does_not_hold_is_refused_with_what_it_would_need(self):
        for question, missing in (("Did the rival's new store cause our sales to fall in September 2026?", "competitor"),
                                  ("Why did Arlo's profit fall in September 2026?", "margin"),
                                  ("Why did the weather hurt footfall in Harbour?", "weather"),
                                  ("Why is the tea rate high?", "metric")):
            with self.subTest(question):
                a, _ = self.ask(question)
                self.assertEqual(a.status, "refused")
                self.assertIn(missing, a.missing)
                self.assertTrue(a.refused_because)
                self.assertEqual(a.queries, [])  # nothing was run

    def test_a_question_naming_its_template_gets_the_same_answer_whatever_a_model_says(self):
        truth = self.replicas[0].truth[2]  # the conversion drop
        p = truth["params"]
        question = (f"why_changed: metric={p['metric']}; brand={p['filters']['brand']}; period={p['period'][0]}..{p['period'][1]}; compare={p['compare']}")
        wrong = Scripted({"template": "metric_value", "params": {"metric": "footfall", "period": p["period"]}})
        off, said_off = self.ask(question)
        on, said_on = self.ask(question, advisor=wrong)
        self.assertEqual(wrong.asked, [])  # no model is asked when the question names its template
        self.assertEqual(on.model_dump(), off.model_dump())
        self.assertEqual((on.choice.source, on.first_factor), ("named", "conversion"))
        self.assertTrue(found(truth, on))

    def test_a_model_chooses_and_fills_and_is_validated(self):
        truth = self.replicas[0].truth[0]
        p = truth["params"]
        paraphrase = f"How come {p['filters']['brand']} earned less at {p['filters']['mall']} this September than the September before?"
        good = Scripted({"template": "why_changed", "params": {"metric": "sales", "filters": p["filters"], "period": p["period"], "compare": "last_year"}})
        a, said = self.ask(paraphrase, advisor=good)
        self.assertEqual((a.choice.source, a.first_factor), ("model", "footfall"))
        purpose, evidence = good.asked[0]
        self.assertEqual(purpose, self.mc.PURPOSE)
        self.assertEqual(set(evidence["menu"]["dimensions"]), {"mall", "brand", "category"})  # names and definitions, never a row or a number
        self.assertNotIn("visitors", str(evidence["menu"]["dimensions"]))
        self.assertIn("rules", said)  # what the rules read is kept beside the model's choice
        for bad in ({"template": "free_sql", "params": {"sql": "SELECT * FROM sales"}},  # never free SQL
                    {"template": "why_changed", "params": {"metric": "profit", "period": p["period"], "compare": "last_year"}},  # not approved
                    {"template": "why_changed", "params": {"metric": "sales", "filters": {"mall": "Atlantis"}, "period": p["period"], "compare": "last_year"}},
                    {"template": "why_changed", "params": {"metric": "sales", "period": ["2031-01-01", "2031-01-31"], "compare": "last_year"}}):
            with self.subTest(bad=bad):
                a, said = self.ask(truth["question"], advisor=Scripted(bad))
                self.assertEqual(a.choice.source, "rules")  # the model's choice failed its check: the rules' stands
                self.assertTrue(said["model_error"])
                self.assertTrue(found(truth, a))

    def test_the_query_under_an_answer_is_the_one_that_made_its_numbers(self):
        r, mc = self.replicas[1], self.mc
        a, _ = self.ask(r.truth[2]["question"], seed=1)
        db = sqlite3.connect(":memory:")
        for name, frame in (("footfall", r.footfall), ("sales", r.sales)):
            frame.to_sql(name, db, index=False)
        sums = {}
        for q in [q for q in a.queries if q.purpose == "numbers"]:  # rerun as shown, parameters bound (the drill and coverage queries beside them)
            out = pd.read_sql_query(q.sql, db, params=q.params)
            period = (q.params[0], q.params[1])
            for col in out.columns:
                if col != "day":
                    sums[(period, col)] = sums.get((period, col), 0) + out[col].sum()
        self.assertAlmostEqual(a.current, sums[(tuple(a.period), "sales")], places=2)
        self.assertAlmostEqual(a.previous, sums[(tuple(a.compare), "sales")], places=2)
        self.assertTrue(all("?" in q.sql and r.truth[2]["params"]["filters"]["brand"] not in q.sql for q in a.queries))  # values bound, never written in
        self.assertEqual(a.sources, ["footfall", "sales"])
        back = mc.MetricAnswer.read(a.model_dump_json())  # a versioned contract, read back whole
        self.assertEqual(back, a)

    def test_the_studios_sqlite_runner_and_the_cores_computation_give_the_same_answer(self):
        from dclab_rnd.metric_change import sqlite_runner

        r = self.replicas[2]
        on_sql = self.mc.Data(self.layer, None, runner=sqlite_runner(self.layer, {"footfall": r.footfall, "sales": r.sales}))
        self.assertEqual(on_sql.values, self.data[2].values)
        self.assertEqual((on_sql.first, on_sql.last), (self.data[2].first, self.data[2].last))
        for truth in r.truth:
            a = self.mc.ask(truth["question"], self.layer, None, data=on_sql, bootstrap=300)[0]
            b = self.ask(truth["question"], seed=2)[0]
            self.assertEqual(a.model_dump(), b.model_dump(), truth["id"])

    def test_the_campaign_keeps_every_answer_whole_and_reads_it_back(self):
        import json
        import tempfile

        from dclab_rnd import metric_change as campaign

        doc = campaign.run((0,), log=lambda *_: None)
        self.assertEqual(doc["by_arm"]["words_rules"], {"reached": 8, "asked": 8})
        self.assertEqual(doc["by_arm"]["named"]["reached"], doc["by_arm"]["named"]["asked"])
        with tempfile.TemporaryDirectory() as tmp:
            path = campaign.write(doc, campaign=Path(tmp) / "metric_change_v1")
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotIn("detail", {k for k, v in record.items() if isinstance(v, list)})
            kept = campaign.detail(path)
            self.assertEqual(len(kept), len(doc["rows"]))
            first = kept[0]
            self.assertEqual(self.mc.MetricAnswer.read(first["answer"]).question, first["question"])  # the answer whole, readable as its contract
            self.assertIn("rules", first["said"])  # and what each reader said about the choice
            self.assertTrue(first["answer"]["queries"] or first["answer"]["status"] == "refused")

    def test_the_reviews_gaps_are_refused_wherever_they_hide(self):
        r = self.replicas[0]
        gap, month = r.info["planted"]["data_gap"], pd.Timestamp(r.period[0]).strftime("%B %Y")
        for question in (f"Why is footfall down for {gap['brand']} in {month} against last year?",  # the store's gap under the brand's other stores
                         f"Why is conversion up for {gap['brand']} in {month} against last year?",
                         f"Which mall had the highest conversion for {gap['brand']} in {month}?"):  # top_segments checks coverage too
            with self.subTest(question):
                a, _ = self.ask(question)
                self.assertEqual(a.status, "refused", a.words)
                self.assertIn("footfall", a.missing)
        sales = r.sales.copy()  # a gap in the numerator's table: ten days of one store's tills gone
        store = sales[(sales["brand"] == r.info["planted"]["steady"])]["mall"].iloc[0]
        lost = sales["date"].isin(sorted(sales.loc[sales["date"] >= r.period[0], "date"].unique())[:10])
        sales = sales[~(lost & (sales["brand"] == r.info["planted"]["steady"]) & (sales["mall"] == store))]
        data = self.mc.Data(self.layer, {"footfall": r.footfall, "sales": sales})
        a = self.mc.ask(f"Why are {r.info['planted']['steady']}'s sales down in {store} in {month} against last year?", self.layer, None, data=data, bootstrap=200)[0]
        self.assertEqual(a.status, "refused")
        self.assertIn("transactions", a.missing)

    def test_periods_outside_the_data_and_unknown_keys_are_refused_never_crashing(self):
        steady = self.replicas[0].info["planted"]["steady"]
        for question, missing in ((f"why_changed: metric=sales; brand={steady}; period=2026-09-15..2026-10-14; compare=last_year", "period"),
                                  (f"Why did {steady}'s sales drop in October 2026 against last year?", "period"),
                                  ("metric_value: what were sales?", "parameter"),
                                  (f"why_changed: metric=sales; brnad={steady}; period=2026-09; compare=last_year", "parameter"),
                                  ("why_changed: metric=sales; period=2024-10; compare=last_year", "period")):  # its comparison before the data
            with self.subTest(question):
                a, _ = self.ask(question)
                self.assertEqual(a.status, "refused", a.words)
                self.assertIn(missing, a.missing)
        a, _ = self.ask("x" * 3000 + " Why did sales fall in September 2026?")  # a long question is kept whole, never a crash
        self.assertEqual(len(a.question), 3000 + len(" Why did sales fall in September 2026?"))

    def test_the_words_are_read_as_a_careful_analyst_would(self):
        r = self.replicas[0]
        steady, conv = r.info["planted"]["steady"], r.info["planted"]["conversion_drop"]["brand"]
        a, _ = self.ask(f"Why did {steady}'s sales change in March against last year?")
        self.assertEqual(a.period, ("2026-03-01", "2026-03-31"))  # a month alone: its latest whole one, said
        self.assertTrue(any("2026" in n for n in a.notes))
        a, _ = self.ask(f"why_changed: metric=sales; brand={steady}; period=2026-03-01..2026-03-31; compare=previous_period")
        self.assertEqual(a.days, (31, 28))
        self.assertTrue(any("per day" in n for n in a.notes))  # thirty-one days against twenty-eight are compared per day
        a, _ = self.ask(f"Why did {conv}'s sales drop in {pd.Timestamp(r.period[0]).strftime('%B %Y')} against last year, even though prices did not change?")
        self.assertEqual((a.status, a.first_factor), ("answered", "conversion"))  # a side remark about prices: said, never a refusal
        self.assertTrue(any("price" in n for n in a.notes))
        a, _ = self.ask(f"What were Arlo's sales in {next(m for m in self.data[0].values['mall'] if m not in self.malls_of(0, 'Arlo'))} in September 2026?")
        self.assertEqual(a.status, "refused")  # a store that does not exist has no number, not zero
        short = {k: v[v["date"] <= "2026-09-01"] for k, v in (("footfall", r.footfall), ("sales", r.sales))}
        data = self.mc.Data(self.layer, short)
        self.assertEqual(self.mc.rules("What were sales last month?", data).params.period, ("2026-08-01", "2026-08-31"))  # the data ends on the 1st

    def malls_of(self, seed, brand):
        f = self.replicas[seed].footfall
        return set(f.loc[f["brand"] == brand, "mall"])

    def test_the_models_reply_is_kept_whole_and_a_big_menu_never_crosses_its_limit(self):
        truth = self.replicas[0].truth[0]
        reply = {"template": "why_changed", "params": {"metric": "sales", "filters": truth["params"]["filters"], "period": truth["params"]["period"],
                                                       "compare": "last_year"}, "reason": "the visitors"}
        a, said = self.ask(truth["paraphrase"], advisor=Scripted(dict(reply)))
        self.assertEqual(said["model_raw"], reply)
        a, said = self.ask(self.replicas[0].truth[-1]["paraphrase"], advisor=Scripted(
            {"refused": True, "missing": ["the cost of the goods sold by brand and category"], "reason": "no cost table"}))
        self.assertEqual((a.status, a.missing), ("refused", ["margin"]))  # the live run: a refusal naming the concept's needs is the concept
        r = self.replicas[0]
        wide = r.sales.assign(**{f"d{i}": [f"v{i}_{j % 60}" for j in range(len(r.sales))] for i in range(4)})
        layer = self.mc.SemanticLayer(**{**self.layer.model_dump(), "dimensions": self.layer.model_dump()["dimensions"]
                                         + [{"name": f"d{i}", "label": f"D{i}", "columns": {"sales": f"d{i}"}} for i in range(4)]})
        data = self.mc.Data(layer, {"footfall": r.footfall, "sales": wide})
        self.assertLessEqual(sum(len(v) for v in self.mc.menu(data)["dimensions"].values()), self.mc.MENU_VALUES)

    def test_an_empty_segment_is_never_a_segment_on_either_path(self):
        from dclab_rnd.metric_change import sqlite_runner

        r = self.replicas[1]
        sales = r.sales.copy()
        sales.loc[sales.sample(frac=0.05, random_state=0).index, "category"] = None
        tables = {"footfall": r.footfall, "sales": sales}
        q = "Which category had the most sales in September 2026?"
        on_sql = self.mc.ask(q, self.layer, None, data=self.mc.Data(self.layer, None, runner=sqlite_runner(self.layer, tables)))[0]
        on_pandas = self.mc.ask(q, self.layer, tables)[0]
        self.assertEqual(on_sql.segments, on_pandas.segments)
        self.assertNotIn("None", [s["value"] for s in on_sql.segments])

    def test_a_why_inside_another_problem_keeps_that_problems_kind(self):
        from dclab_rnd import kinds

        self.assertEqual(kinds.detect("Forecast next week's sales for each store; why are sales down in winter?")["key"], "forecasting")
        self.assertEqual(kinds.detect("Which customers will churn next month? We also wonder why is revenue down.")["key"], "binary")
        self.assertEqual(kinds.detect(self.replicas[0].truth[0]["question"])["key"], "metric_change")

    def test_mix_and_rate_add_up_to_the_change(self):
        mc = self.mc
        days = [f"2026-0{m}-{d:02d}" for m in (1, 2) for d in range(1, 29)]
        rows = []
        for day in days:
            late = day.startswith("2026-02")
            for c, n, b in (("cheap", 30 if late else 10, 10.0), ("dear", 10, 50.0 if not late else 45.0)):
                rows.append({"date": day, "mall": "M", "brand": "B", "category": c, "transactions": n, "sales": n * b})
        sales = pd.DataFrame(rows)
        foot = pd.DataFrame({"date": days, "mall": "M", "brand": "B", "visitors": 400})
        data = mc.Data(self.layer, {"footfall": foot, "sales": sales})
        a, _ = mc.ask("why_changed: metric=basket; period=2026-02-01..2026-02-28; compare=previous_period", self.layer, {}, data=data, bootstrap=50)
        d = next(d for d in a.drills if d.dimension == "category")
        change = 100 * (a.current / a.previous - 1)
        self.assertAlmostEqual(sum(s.contribution_pct for s in d.segments), change, places=2)  # mix plus rate is the whole change
        self.assertAlmostEqual(sum(s.mix_pct + s.rate_pct for s in d.segments), change, places=2)
        cheap = next(s for s in d.segments if s.value == "cheap")
        dear = next(s for s in d.segments if s.value == "dear")
        self.assertLess(cheap.mix_pct, 0)        # more of the cheap category: the mix pulls the basket down
        self.assertAlmostEqual(cheap.rate_pct, 0, places=6)  # its own basket did not move
        self.assertLess(dear.rate_pct, 0)        # the dear category's basket fell: rate

    def test_a_layer_that_does_not_telescope_or_divides_ratios_is_refused(self):
        mc = self.mc
        good = self.layer.model_dump()
        bad = {**good, "factorizations": [{"metric": "sales", "factors": ["footfall", "conversion"]}]}  # sales is not footfall x conversion
        with self.assertRaisesRegex(ValueError, "telescope"):
            mc.SemanticLayer(**bad)
        ratio = {**good, "metrics": good["metrics"] + [{"name": "odd", "label": "Odd", "kind": "ratio", "numerator": "conversion", "denominator": "basket"}]}
        with self.assertRaisesRegex(ValueError, "two sums"):
            mc.SemanticLayer(**ratio)
        self.assertEqual(mc.SemanticLayer.read(self.layer.model_dump_json()), self.layer)


if __name__ == "__main__":
    unittest.main()
