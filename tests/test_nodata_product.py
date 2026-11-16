"""The rebuilt no-data design in the product (stage N2, 2026-11-16): behind DCLAB_NO_DATA_V2, off by default. The lead
agent's tool, the rules-first Home and the API route answer a person with no table through dclab_rnd.nodata; nothing
creates data; Build project is the one of the commit before this work. Scripted models only."""
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd.agents import case_agent as A  # noqa: E402
from dclab_rnd.agents.registry import Registry  # noqa: E402
from dclab_rnd.draft import chat as draft_chat  # noqa: E402
from dclab_rnd.draft import store as draft_store  # noqa: E402
from dclab_rnd.nodata import product  # noqa: E402

WAREHOUSE = ("I run operations for a furniture retailer with one warehouse. The owner wants to know whether we should open a second "
             "warehouse in the south next year. Nothing is in a system I could export for you. Deliveries to the south take about 6 days, "
             "to the north 2.")
WHY = "Why did complaints rise in March? We have no data export to share."
CHURN = "We want to predict churn but do not have a dataset yet. We have 5,000 customers and 12% of them leave each year."
BUSINESS = """Need: business

## The problem
Whether to open a second warehouse. The gap is 6 - 2 = 4 days.

## What needs care
The delay may not come from where the stock is.

## Options
Keep one warehouse, rent space from a partner for a season, or open your own. Renting tests the idea at the least cost.

## Business solution
Rent space in the south for one season and compare.

## Plan
1. Write down what must improve. Needs: the owner. Checked by: one sentence they accept.
2. Find where the days go. Needs: your knowledge of the route. Checked by: a list of the delays.
3. Trial with a partner. Needs: a partner. Checked by: delivery days before and after.
Stop when the trial shows no gain. If a step fails, keep one warehouse.

## What only real data can settle
How many sales the delay costs.

## Questions for you
- How many orders a week go south?
"""
BUILD_PINS = {
    "route": "fe96a025f033f1edddbd24029e4b3eb46075460cbb4355dee2ee81288f7d4ad9",
    "button": "ee0f00728e724931a99411eeecd32132e17890c3f75280ef31e64f3d8caa50b9",
    "handler": "0f20813693d19bf2733959085b1b6eed3d52b157773134539771a11711b28355",
    "function": "15eec7e07e13133d0f64e00bb374a03c02d60977a9d0a2b4405b4288b2df40d9"
}


class Client:
    def __init__(self, *replies):
        self.replies, self.seen, self.spent_eur = list(replies), [], 0.0

    def complete(self, messages, tools=None, **kwargs):
        self.seen.append(messages)
        self.spent_eur += 0.004
        return {"content": self.replies.pop(0) if self.replies else ""}


class Gateway:
    def __init__(self, **clients):
        self.clients = clients

    def client(self, purpose, **kwargs):
        return self.clients.get(purpose)


def env(on: bool, live: bool = True):
    return mock.patch.dict(os.environ, {product.SWITCH: "1" if on else "0", "DCLAB_NO_LIVE_MODELS": "0" if live else "1"})


def state(words: str, gateway=None) -> A.CaseState:
    s = A.CaseState(case=C.hear(C.new("draft-nd"), words))
    s.models, s.draft_id = gateway, "draft-nd"
    return s


def tool_says() -> str:
    registry = Registry()
    A.register(registry)
    return registry.tools["case_design_solution_without_data"].description


class SwitchTests(unittest.TestCase):
    def test_the_rebuilt_design_is_the_default_and_zero_puts_the_first_design_back(self):
        """The owner's decision of 2026-11-16, made with nodata_gate_v1 failed on time alone."""
        with mock.patch.dict(os.environ, {"DCLAB_NO_LIVE_MODELS": "1"}):
            for value in (None, "", "1"):
                os.environ.pop(product.SWITCH, None)
                if value is not None:
                    os.environ[product.SWITCH] = value
                self.assertTrue(product.enabled(), value)
            for value in ("0", "off", "false", "no"):
                os.environ[product.SWITCH] = value
                self.assertFalse(product.enabled(), value)
            out = A._design_solution_without_data(state(CHURN))  # switched off: the first design answers, as before
        self.assertTrue(out["blueprint_ready"])
        self.assertIn(product.SWITCH, (ROOT / ".env.example").read_text(encoding="utf-8"))
        self.assertIn("374 seconds", product.enabled.__doc__)  # the decision is recorded with the check that failed

    def test_the_tool_says_what_it_does_under_each_design(self):
        with env(False):
            self.assertIn("3-layer", tool_says())
        with env(True):
            self.assertIn("a business answer", tool_says())
            self.assertIn("Never ask for an upload again", tool_says())


class LeadAgentTests(unittest.TestCase):
    def test_a_business_question_with_no_data_is_answered_not_refused(self):
        with env(False, live=False):
            first = A._design_solution_without_data(state(WAREHOUSE))
        self.assertTrue(first["blueprint_ready"] and first["ml_architecture"])  # the first design draws a model blueprint for a decision
        with env(False, live=False):
            self.assertIn("no model blueprint fits", A._design_solution_without_data(state(WHY))["error"])  # or refuses, with nothing shown
        s = state(WAREHOUSE, Gateway(no_data_architect=Client(BUSINESS), no_data_critic=Client("No serious weakness found.")))
        with env(True):
            out = A._design_solution_without_data(s)
        self.assertEqual((out["solution_ready"], out["need"], out["written_by"]), (True, "business", "architect"))
        self.assertNotIn("Model", out["parts"])
        card = s.cards[-1]
        self.assertEqual((card["kind"], card["solution"]["version"]), ("no_data_solution", "nodata_v2"))
        self.assertIn("6 - 2 = 4", s.computed)  # a sum code checked is a source of this turn's reply
        self.assertGreater(s.spent_eur, 0)
        with env(True):
            self.assertIn("already built", A._design_solution_without_data(s)["error"])

    def test_a_shared_table_or_an_empty_budget_is_refused_as_before(self):
        with env(True):
            with_table = A.CaseState(case=C.hear(C.new("d"), CHURN), table=A.table_of("c.csv", [{"name": "id", "kind": "text", "noticed": []}], 10))
            self.assertIn("a table is shared", A._design_solution_without_data(with_table)["error"])
            broke = state(CHURN, Gateway(no_data_architect=Client(BUSINESS)))
            broke.cap_eur, broke.spent_eur = 0.1, 0.1
            self.assertIn("budget", A._design_solution_without_data(broke)["error"])

    def test_a_person_with_no_table_is_sent_to_the_solution_never_asked_for_a_file(self):
        with env(True):
            out = A._start(state(WAREHOUSE), "analysis", "query")
        self.assertEqual(out["error"], A.NO_TABLE_GO)
        with env(False):
            self.assertIn("share or upload the table", A._start(state(WAREHOUSE), "analysis", "query")["error"])  # the first design's words, unchanged


class RulesFirstHomeTests(unittest.TestCase):
    def test_home_shows_the_rebuilt_solution_and_makes_no_data(self):
        store = draft_store.DraftStore(Path(tempfile.mkdtemp()))
        ha = draft_chat.HomeAgent(store, None)
        ha.models = Gateway(no_data_architect=Client(BUSINESS), no_data_critic=Client("- It never says who decides."))
        d = store.create(WAREHOUSE)
        with env(True):
            sol = ha.provide_no_data_solution(d["id"])
        saved = store.get(d["id"])
        self.assertEqual((sol["version"], saved["no_data_solution"]["need"]), ("nodata_v2", "business"))
        self.assertEqual(saved["assets"], [])  # nothing made up: no table exists after the answer
        message = next(m for m in saved["messages"] if m.get("kind") == "no_data_solution")
        self.assertEqual(message["actions"], ["ask_simulate", "copy_spec"])
        self.assertNotIn("A second reader's findings", [n["title"] for n in message["solution"]["card"]["notes"]])  # off by default: it did not earn its time
        self.assertTrue(message["solution"]["simulation"]["questions"])  # made-up data begins with questions, never with a table

    def test_with_no_model_a_prediction_gets_the_first_designs_rules_and_a_decision_gets_a_plain_note(self):
        with env(True, live=False):
            rules = product.solve_for_product(CHURN)
            note = product.solve_for_product(WHY)
        self.assertEqual((rules["harness_report"]["mode"], rules["rules_note"]), ("deterministic_floor", "No model was available."))
        # where the first design's rules see nothing to predict, a plain note: never a model blueprint. (Where they do see
        # something, as for the warehouse, no model means the first design's answer, as today: the rules cannot tell.)
        self.assertEqual((note["mode"], note["card"]["parts"], note["markdown"]), ("rules", [], product.NO_MODEL_NOTE))


class ReviewOfTheWiringTests(unittest.TestCase):
    """The review of stage N2 (2026-11-16): six findings, each kept from coming back."""

    def home(self, problem, gateway=None):
        store = draft_store.DraftStore(Path(tempfile.mkdtemp()))
        ha = draft_chat.HomeAgent(store, None)
        if gateway is not None:
            ha.models = gateway
        return store, ha, store.create(problem)

    def test_made_up_data_begins_with_questions_never_with_the_one_click_way(self):
        asks = "We want to predict churn but have no data yet. Can you simulate some data for a demo? We have 5,000 customers."
        answer = BUSINESS.replace("Need: business", "Need: simulate")
        store, ha, d = self.home(asks, Gateway(no_data_architect=Client(answer)))
        store.update(d["id"], lambda x: x.update(intent={"track": "prediction"}))
        with env(True):
            ha.no_data_blueprint(d["id"])
        kinds = [m.get("kind") for m in store.get(d["id"])["messages"]]
        self.assertIn("synthetic_questions", kinds)
        self.assertNotIn("synthetic_way", kinds)  # the offer that carries the one-click AI build is never reached from the rebuilt design
        asked = next(m for m in store.get(d["id"])["messages"] if m.get("kind") == "synthetic_questions")
        self.assertEqual(asked["actions"], ["simulate_describe"])
        self.assertEqual(store.get(d["id"])["assets"], [])

    def test_with_no_questions_to_ask_nothing_about_made_up_data_is_offered(self):
        store, ha, d = self.home(WHY + " Can you simulate some data for a demo?")
        with env(True, live=False):
            self.assertIsNone(ha.ask_synthetic_first(d["id"], product.solve_for_product(WHY)))

    def test_the_plain_note_is_a_note_and_is_never_saved_or_drawn_as_a_solution(self):
        store, ha, d = self.home(WHY)
        with env(True, live=False):
            sol = ha.provide_no_data_solution(d["id"])
        saved = store.get(d["id"])
        self.assertTrue(product.is_note(sol))
        self.assertNotIn("no_data_solution", saved)
        self.assertFalse((saved.get("understanding") or {}).get("data_plan"))  # Home's question about data stays open, as with the switch off
        self.assertEqual([m.get("kind") for m in saved["messages"] if m.get("kind", "").startswith("no_data")], ["no_data_note"])

    def test_a_model_that_fails_on_a_decision_leaves_no_empty_solution_and_no_dead_end(self):
        for gateway in (Gateway(no_data_architect=Client("")), Gateway(), None):
            s = state(WHY, gateway)
            with env(True):
                out = A._design_solution_without_data(s)
                self.assertEqual((out["solution_ready"], out["next"]), (False, A.NO_SOLUTION_NEXT))
                self.assertEqual(s.cards, [])  # nothing is shown as a solution
                self.assertIsNone(s.no_data_solution)
                self.assertEqual(A._start(s, "analysis", "query")["error"], A.NO_TABLE_DONE)  # told to write, not to call the tool again

    def test_a_paid_request_that_came_back_empty_is_still_counted(self):
        s = state(CHURN, Gateway(no_data_architect=Client("")))
        with env(True):
            out = A._design_solution_without_data(s)
        self.assertEqual((out["solution_ready"], out["written_by"]), (True, "rules"))
        self.assertGreater(s.spent_eur, 0)
        self.assertIn("placeholders", out["next"])  # the first design's card: its assumed inputs are said to be placeholders
        self.assertIn("assumed_inputs", out)

    def test_a_person_who_has_the_table_or_only_used_the_words_is_still_asked_for_it(self):
        with env(True):
            for words in ("I want a churn model without data leakage. I will upload the customers table in a minute.",
                          "We have no data scientist, so I need help. I have the sales table ready to upload.",
                          "I had no data last week. I have the export now, shall I upload it?"):
                self.assertIn("share or upload the table", A._start(state(words), "analysis", "query")["error"], words)
            later = state("We have not collected any data yet.")
            later.case = C.hear(later.case, "Go ahead with what you suggested.")
            self.assertEqual(A._start(later, "analysis", "query")["error"], A.NO_TABLE_GO)  # what they said earlier still holds

    def test_the_first_designs_blueprint_is_still_a_card_when_no_model_writes_one(self):
        store, ha, d = self.home(CHURN)
        with env(True, live=False):
            sol = ha.provide_no_data_solution(d["id"])
        self.assertFalse(product.is_note(sol))
        self.assertIn("no_data_solution", store.get(d["id"]))


class BuildProjectUntouchedTests(unittest.TestCase):
    """The owner's rule: Build project stays exactly as it is. Its route, its button, its click handler and its function
    are the ones of the commit before the no-data work (919cdd98); a change to any of them fails here and is the owner's
    decision, never a side effect."""

    def parts(self):
        api = (ROOT / "dclab_rnd" / "draft" / "api.py").read_text(encoding="utf-8")
        home = (ROOT / "dclab_rnd" / "agentic" / "web" / "src" / "views" / "home.html").read_text(encoding="utf-8")
        route = api[api.index('    @router.post("/api/drafts/{draft_id}/build", status_code=201'):]
        route = route[:route.index("\n    @router.", 10)]
        function = home[home.index("    async function build("):]
        function = function[:function.index("\n    }\n") + 7]
        return {"route": route, "button": next(line.strip() for line in home.splitlines() if "const buildBtn" in line),
                "handler": next(line.strip() for line in home.splitlines() if "e.target.closest('[data-build]')" in line), "function": function}

    def test_build_project_is_the_one_before_this_work(self):
        for name, text in self.parts().items():
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), BUILD_PINS[name], f"Build project's {name} changed")

    def test_no_function_of_the_rebuilt_design_reaches_build(self):
        source = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "dclab_rnd" / "nodata").glob("*.py"))
        for word in ("build_project", "/build", "build_through_team", "projects.create"):
            self.assertNotIn(word, source)


if __name__ == "__main__":
    unittest.main()
