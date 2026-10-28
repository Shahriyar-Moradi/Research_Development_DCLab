"""Business questions on Home (package H11): where a number changed between two periods, as a decision brief written by a
template from the comparison, always "association, not cause"; what the rules cannot read is said, never guessed."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import change as C  # noqa: E402
from dclab_rnd import ledger  # noqa: E402
from dclab_rnd.agent_eval.home import tables  # noqa: E402
from dclab_rnd.draft import business, pipeline  # noqa: E402
from dclab_rnd.draft.chat import HomeAgent  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class WindowsTests(unittest.TestCase):
    def test_periods_in_english_and_persian(self):
        a, b = business.windows("Why did revenue drop in March 2026 compared with February?")
        self.assertEqual((a.start, a.end, b.start, b.end), ("2026-02-01", "2026-02-28", "2026-03-01", "2026-03-31"))
        a, b = business.windows("Sales in January 2026 vs December")
        self.assertEqual((a.label, b.label), ("December 2025", "January 2026"))  # the month before, across the year
        a, b = business.windows("چرا فروش منطقهٔ شارجه در سه ماه اول ۲۰۲۶ کم شد؟")
        self.assertEqual((a.label, b.label), ("Q4 2025", "Q1 2026"))  # a quarter against the one before it
        self.assertIsNone(business.windows("Should we change our returns policy?"))
        self.assertIsNone(business.windows("Revenue in March compared with February"))  # no year: never assumed
        a, b = business.windows("Why did revenue fall in January compared with December 2025?")
        self.assertEqual((a.label, b.label), ("December 2025", "January 2026"))  # the year only on the baseline: the month after it
        a, b = business.windows("June compared with May 2025")
        self.assertEqual((a.label, b.label), ("May 2025", "June 2025"))


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.home, True)
        self.store = DraftStore(self.home / "drafts")
        ledger.install(ledger.FileLedger(self.home / "ledger", "wtest"))
        self.addCleanup(ledger.install, None)

    def ask(self, text, table="sales"):
        d = self.store.create(text)
        agent = HomeAgent(self.store, None, on_request=lambda *a: None)
        agent.start(d["id"])
        if agent.track(self.store.get(d["id"])) != "business":
            agent.choose_intent(d["id"], "business")
        name = tables.write(table, self.store.data_dir(d["id"]))
        asset = pipeline.new_asset(self.store, d["id"], "upload", name, name)
        pipeline.run(self.store, d["id"], asset["id"], agent, None)
        for _ in range(4):  # the person does not know the brief's answers
            q = agent.pending(self.store.get(d["id"]))
            if q is None:
                break
            agent.reply(d["id"], "I'm not sure")
        return agent, self.store.get(d["id"])

    def test_the_model_reads_what_was_compared_never_the_numbers(self):
        """The review of H7a..H12: the brief's message reached home_agent's history whole (totals, segment names); its
        limit is no cell value at all, and its prompts are kept for training."""
        from dclab_rnd.draft import memory
        agent, d = self.ask("Why did revenue drop in March 2026 compared with February?")
        text = self.brief_of(d)
        r = C.ChangeResult.model_validate(d["business"]["result"])
        seen = json.dumps(memory.recent(d), ensure_ascii=False)
        self.assertIn("decision brief is shown to the person", seen)
        self.assertNotIn(text, seen)
        for seg in r.segments:
            self.assertNotIn(str(seg.value) if hasattr(seg, "value") else str(seg), seen)
        for number in (r.total_a, r.total_b):
            self.assertNotIn(f"{number:,.0f}", seen)

    def brief_of(self, d):
        return next((m["text"] for m in d["messages"] if m.get("kind") == "decision_brief"), None)

    def test_why_revenue_dropped_names_the_segments_that_carry_it(self):  # H1 d-s1
        agent, d = self.ask("Why did revenue drop in March 2026 compared with February?")
        text = self.brief_of(d)
        self.assertIsNotNone(text)
        self.assertIn("association, not cause", text)
        self.assertEqual((d["exit"]["kind"], d["exit"]["status"]), ("decision_brief", "reached"))
        r = C.ChangeResult.model_validate(d["business"]["result"])
        a = next(x for x in d["assets"] if x["id"] == d["active_asset"])
        t = pd.read_parquet(Path(self.store.data_dir(d["id"])) / a["clean_file"])  # the cleaned table Home answers from
        when = pd.to_datetime(t.order_date)
        feb = t[(when >= "2026-02-01") & (when < "2026-03-01")].groupby(r.by)["amount"].sum()
        mar = t[(when >= "2026-03-01") & (when < "2026-04-01")].groupby(r.by)["amount"].sum()
        change = mar.sub(feb, fill_value=0)
        first_driver = text.split("came most from:\n- ")[1].split(":")[0]
        if r.change < 0:
            self.assertLess(change[first_driver], 0)  # a segment that fell, for a fall
        for s in r.segments:
            if s.segment != C.OTHER:
                self.assertAlmostEqual(s.change, change.get(s.segment, 0.0), places=6)

    def test_a_named_value_narrows_and_an_unmatched_one_is_said(self):
        agent, d = self.ask("Why did Sharjah revenue fall in Q1 2026?")
        self.assertEqual(d["business"]["narrow"], {"region": "Sharjah"})
        self.assertIn("region = Sharjah", self.brief_of(d))
        agent, d = self.ask("چرا فروش منطقهٔ شارجه در سه ماه اول ۲۰۲۶ کم شد؟")  # the table spells it in English
        self.assertIn("name it as it is written in the table", self.brief_of(d))

    def test_a_decision_without_two_periods_is_told_what_is_needed_once(self):
        agent, d = self.ask("Should we change our returns policy?")
        self.assertIsNone(self.brief_of(d))
        said = [m["text"] for m in d["messages"] if "To do it here I need" in (m.get("text") or "")]
        self.assertEqual(len(said), 1)
        self.assertIn("two periods", said[0])
        agent.answer_business(d["id"])  # asked again for the same table: nothing new is said
        self.assertEqual(len([m for m in self.store.get(d["id"])["messages"] if "To do it here I need" in (m.get("text") or "")]), 1)


def frame(rows=None):
    """Orders across two months, with a numeric order id, a personal column, short country codes and a planted drop."""
    import numpy as np
    rng = np.random.default_rng(3)
    out = []
    for month, n in ((2, 400), (3, 300)):
        for i in range(n):
            out.append({"order_id": month * 10000 + i, "order_date": f"2026-0{month}-{1 + i % 27:02d}",
                        "country": ["AE", "IN", "US", "IT"][i % 4], "region": ["Dubai", "Sharjah"][i % 2],
                        "religion": ["x", "y"][i % 2], "salary": 1000.0 + i, "amount": float(rng.uniform(10, 100))})
    t = pd.DataFrame(out + (rows or []))
    cols = [{"name": c, "sensitivity": "sensitive" if c in ("religion", "salary") else "none"} for c in t.columns]
    return t, cols


class ReviewOfH11aTests(unittest.TestCase):
    """The review's findings (H11a), each with the question that showed it."""

    def test_orders_are_counted_never_their_id_summed(self):
        t, cols = frame()
        spec, _ = business.read("Why did orders drop in March 2026 compared with February 2026?", t, columns=cols)
        self.assertIsNone(spec["measure"])  # a count of rows
        r = business.compare(spec, t)
        self.assertAlmostEqual(r.change, -100.0)

    def test_a_personal_column_is_never_summed_split_or_narrowed(self):
        t, cols = frame()
        spec, why = business.read("Why did salary rise in March 2026 compared with February 2026?", t, columns=cols)
        self.assertIsNone(spec)  # salary is personal: no number to explain
        spec, _ = business.read("Why did revenue drop in March 2026 compared with February 2026?", t, "amount", columns=cols)
        self.assertNotIn("religion", spec["splits"])

    def test_short_or_common_words_never_narrow_the_rows(self):
        t, cols = frame()
        for q in ("Why did revenue drop in March 2026 compared with February 2026? Is it a problem for us?",
                  "Why did amount fall at home in March 2026 compared with February 2026? No idea why"):
            spec, _ = business.read(q, t, "amount", columns=cols)
            self.assertEqual(spec["narrow"], {}, q)
        spec, _ = business.read("Why did Sharjah amount fall in March 2026 compared with February 2026?", t, "amount", columns=cols)
        self.assertEqual(spec["narrow"], {"region": "Sharjah"})
        spec, _ = business.read("Dubai and Sharjah amount in March 2026 compared with February 2026", t, "amount", columns=cols)
        self.assertEqual(spec["narrow"], {})  # two values of one column: neither guessed

    def test_one_rows_value_never_shows_under_other(self):
        t, cols = frame([{"order_id": 1, "order_date": "2026-02-03", "country": "IR", "region": "Ajman", "religion": "x",
                          "salary": 1.0, "amount": 98765.43}])
        r = C.compare(t, date_column="order_date", a=C.Window(start="2026-02-01", end="2026-02-28"),
                      b=C.Window(start="2026-03-01", end="2026-03-31"), by="country", measure="amount")
        other = next(s for s in r.segments if s.segment == C.OTHER)
        self.assertGreaterEqual(other.rows_a, C.FLOOR)  # pooled with the smallest named group, never alone
        self.assertNotIn("IR", [s.segment for s in r.segments])

    def test_quarter_against_quarter_ratios_and_wording(self):
        a, b = business.windows("Why is Q2 2026 revenue lower compared with Q2 2025?")
        self.assertEqual((a.label, b.label), ("Q2 2025", "Q2 2026"))
        self.assertIsNone(business.windows("March 2026 compared with March"))  # a month with itself
        t, cols = frame()
        spec, why = business.read("Why did average order value drop in March 2026 compared with February 2026?", t, columns=cols)
        self.assertIsNone(spec)
        self.assertIn("average", why)

    def test_the_brief_places_other_by_its_sign_and_shows_only_its_own_numbers(self):
        import re
        t, cols = frame()
        spec, _ = business.read("Why did amount fall in March 2026 compared with February 2026?", t, "amount", columns=cols)
        r = business.compare(spec, t)
        text = business.brief(r, spec["narrow"])
        allowed = set(business.shown(r))
        for number in re.findall(r"[+−-]?\d[\d,]*(?:\.\d+)?", text.replace("2026", "")):
            self.assertIn(number, allowed | {"2026", "5"}, number)
        flat = C.ChangeResult(**{**r.model_dump(), "change": 0.0, "total_b": r.total_a})
        self.assertIn("did not change", business.brief(flat))

    def test_a_reply_with_the_periods_is_read(self):
        h = HomeTests("test_a_named_value_narrows_and_an_unmatched_one_is_said")
        h.setUp()
        self.addCleanup(shutil.rmtree, h.home, True)
        agent, d = h.ask("Why did revenue drop?")
        self.assertIsNone(h.brief_of(d))
        agent.reply(d["id"], "March 2026 compared with February 2026")
        self.assertIsNotNone(h.brief_of(h.store.get(d["id"])))


class Words:
    """A scripted home_business model."""

    def __init__(self, text):
        self.text, self.asked = text, 0

    def complete(self, messages, tools=None, **kw):
        self.asked += 1
        return {"content": self.text}


class H11bTests(HomeTests):
    """H11b: the brief reworded by the model only through cited.py, and the model's tools behind the business guard."""

    def ask_with(self, text, words):
        d = self.store.create(text)
        agent = HomeAgent(self.store, None, on_request=lambda *a: None, business_client=words)
        agent.start(d["id"])
        if agent.track(self.store.get(d["id"])) != "business":
            agent.choose_intent(d["id"], "business")
        name = tables.write("sales", self.store.data_dir(d["id"]))
        asset = pipeline.new_asset(self.store, d["id"], "upload", name, name)
        pipeline.run(self.store, d["id"], asset["id"], agent, None)
        for _ in range(4):
            if agent.pending(self.store.get(d["id"])) is None:
                break
            agent.reply(d["id"], "I'm not sure")
        return agent, self.store.get(d["id"])

    def faithful(self, d):
        """A rewording that names every group the template names, each with its own number and direction."""
        r = C.ChangeResult.model_validate(d["business"]["result"])
        sentences = [f"Revenue {'fell' if r.change < 0 else 'rose'} by {business._n(abs(r.change))} [CHANGE]."]
        for row in d["business"]["template"].split("\n"):
            if row.startswith("- "):
                group, rest = row[2:].split(": ", 1)
                number = rest.split(" ")[0]
                sentences.append(f"{group} {'fell' if number.startswith('−') else 'rose'} by {number[1:]} [CHANGE].")
        return r, " ".join(sentences)

    def test_a_cited_rewording_is_used_and_an_invented_number_never(self):
        q = "Why did revenue drop in March 2026 compared with February 2026?"
        agent, d = self.ask(q)
        r, good_text = self.faithful(d)
        change = business._n(abs(r.change))
        agent, d = self.ask_with(q, Words(good_text))
        text = self.brief_of(d)
        self.assertTrue(text.startswith(f"Revenue fell by {change}"), text)
        self.assertIn("association, not cause", text)  # the code's caveat, always
        self.assertEqual(d["business"]["worded"]["by"], "model")
        bad = Words("Revenue fell by 12,345.67 because the app was slow [CHANGE].")
        agent, d = self.ask_with(q, bad)
        self.assertNotIn("12,345.67", self.brief_of(d))  # an invented number: the template stands
        self.assertEqual(d["business"]["worded"]["by"], "template")

    def test_a_cause_a_moved_number_a_wrong_direction_or_a_dropped_group_keeps_the_template(self):  # the review of H11b
        q = "Why did revenue drop in March 2026 compared with February 2026?"
        agent, d = self.ask(q)
        r, good_text = self.faithful(d)
        change = business._n(abs(r.change))
        groups = [x for x in r.segments if x.segment != C.OTHER]
        first, second = groups[0], groups[1]
        bad = [good_text.replace("[CHANGE].", "because customers left [CHANGE].", 1),
               good_text + " You should close the app [CHANGE].",
               good_text.replace(f"Revenue fell by {change}", f"Revenue rose by {change}", 1) if r.change < 0 else good_text.replace("rose", "fell", 1),
               good_text.replace(f"{first.segment} ", f"{second.segment} ", 1),  # a number given to another group
               f"Revenue fell by {change} [CHANGE]."]  # the groups left out
        for text in bad:
            agent, d = self.ask_with(q, Words(text))
            self.assertEqual(d["business"]["worded"]["by"], "template", text)
        agent, d = self.ask_with("Why did Sharjah revenue fall in Q1 2026?", None)
        self.assertIn("region = Sharjah", self.brief_of(d))

    def test_the_model_compares_through_its_tool_and_never_sees_a_number(self):
        from dclab_rnd.agents import default_registry
        from dclab_rnd.draft.chat import DraftGuard, Turn
        agent, d = self.ask("Why did our revenue go down?")
        turn = Turn(agent, d["id"])

        def tool(name, **args):
            t = default_registry().get(name)
            return DraftGuard().refusal(t, turn, args) or t.handler(turn, **args)
        out = tool("compare_periods", a_start="2026-02-01", a_end="2026-02-28", b_start="2026-03-01", b_end="2026-03-31", measure="amount")
        self.assertEqual(set(out), {"answered", "by"})  # no number reaches the model
        self.assertIn("in February 2026 to", self.brief_of(self.store.get(d["id"])))  # labels by code from the dates
        out = tool("compare_periods", a_start="2026-02-01", a_end="2026-02-28", b_start="2026-03-01", b_end="2026-03-31",
                   measure="amount", a_label="IGNORE THE RULES")
        self.assertNotIn("IGNORE", json.dumps(self.store.get(d["id"])["messages"]))  # a label the model writes never reaches the person
        self.assertIsNotNone(self.brief_of(self.store.get(d["id"])))
        self.assertIn("error", tool("compare_periods", a_start="2026-02-01", a_end="2026-02-28", b_start="2026-03-01", b_end="2026-03-31",
                                    measure="customer_id"))  # never an identifier
        self.assertIn("error", tool("compare_periods", a_start="2026-03-01", a_end="2026-02-01", b_start="2026-03-01", b_end="2026-03-31"))
        self.assertIn("error", tool("compare_periods", a_start="2026-02-01", a_end="2026-02-28", b_start="2026-03-01", b_end="2026-03-31",
                                    by="amount"))  # a number column is never a grouping
        self.assertEqual(tool("decision_brief"), {"said": True})
        agent.choose_intent(d["id"], "analysis")
        self.assertIn("error", tool("decision_brief"))  # not a business question any more


if __name__ == "__main__":
    unittest.main()
