"""Package 30.18: the knowledge the roles read.

One read tool, ``find_examples``, gives the intern and /mcp the closest business examples a model may read; the decomposer, the
router and the critic read the closest examples' lines only when they are given, and they are given only where the knowledge arm
passed its sealed journeys and the switch is on, or a measurement asks; no private example ever reaches a model's prompt (the
gateway's recorded prompts are read); the search is measured on an independent writer's wording.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pandas  # noqa: F401
except ImportError as error:  # .venv-agent has no pandas
    MISSING = str(error)
else:
    MISSING = ""

FRAUD = "Our bikers and customers fake deliveries for refunds; the team can review 300 cases a day and must catch most of the fraud."


class Keep:
    """An advisor that abstains and keeps every evidence it was shown."""

    def __init__(self):
        self.seen = []

    def ask(self, purpose, evidence):
        self.seen.append((purpose, evidence))
        return None


@unittest.skipIf(MISSING, MISSING)
class KnowledgeRolesTest(unittest.TestCase):
    def test_find_examples_gives_only_what_a_model_may_read(self):
        from dclab_rnd import business_examples as kb
        from dclab_rnd.agents import default_registry

        private = {e.id for e in kb.load() if e.visibility == "rules_only"}
        self.assertTrue(private)
        registry = default_registry()
        self.assertEqual(registry.get("find_examples").effect, "read")
        out = registry.call("find_examples", {"words": FRAUD, "top": 5}, scope="project")
        ids = [e["id"] for e in out["examples"]]
        self.assertTrue(ids)
        self.assertFalse(set(ids) & private)  # the owner's own solution is found by the rules, never handed to a model
        self.assertIn("owner:delivery_fraud", [e["id"] for e in kb.find(FRAUD, top=5)])

    def test_the_roles_read_examples_only_when_given(self):
        from dclab_core import mapping, review, solutions
        from dclab_core import problem_map as pm

        lines = ["Two-stage screening (recipe: rare_event); how: a queue of the team's capacity", "Copied bills; what goes wrong: copies across the split"]
        for examples, present in ((lines, True), ([], False)):
            keep = Keep()
            mapping.draw(FRAUD, advisor=keep, examples=examples)
            sub = pm.SubProblem(id="sp1", part="part1", title="Find the fraud", family="rare_event", labels="file")
            m = pm.ProblemMap(size="complex", parts=[pm.Part(id="part1", title="Fraud"), pm.Part(id="part2", title="Queue")],
                              sub_problems=[sub, pm.SubProblem(id="sp2", part="part2", title="Rank for review", family="ranking", labels="none")])
            solutions.route_with(sub, None, m, None, keep, examples)
            seen = {p: e for p, e in keep.seen}
            self.assertEqual("similar_problems" in seen["decomposer"], present)
            self.assertEqual("similar_problems" in seen.get("solution_router", {}), present)
            if present:
                self.assertEqual(seen["decomposer"]["similar_problems"], lines)
        from dclab_core.contracts import Review

        for examples, present in ((lines, True), ([], False)):  # the critic: what goes wrong in similar problems, only when given
            keep = Keep()
            review.critique(keep, Review(), "binary", metric="roc_auc", examples=examples)
            self.assertEqual("pitfalls_of_similar_problems" in keep.seen[0][1], present)

    def test_the_switch_and_the_measurement_decide_what_a_role_reads(self):
        from dclab_rnd.agent_eval.journeys import programme
        from dclab_rnd.draft.map import examples_for

        class Advisor:
            pass

        a = Advisor()
        self.assertEqual(examples_for(FRAUD, None), [])  # no model, nothing to read
        with mock.patch.dict(os.environ, {"DCLAB_EXAMPLES": "on"}), mock.patch.object(programme, "examples_passed", return_value=False):
            self.assertEqual(examples_for(FRAUD, a), [])  # switched on, but the arm has not passed: the rules serve
        with mock.patch.dict(os.environ, {"DCLAB_EXAMPLES": "on"}), mock.patch.object(programme, "examples_passed", return_value=True):
            self.assertTrue(examples_for(FRAUD, a))
        a.examples = True  # a measurement asks for them
        with mock.patch.dict(os.environ, {"DCLAB_EXAMPLES": ""}):
            got = examples_for(FRAUD, a)
        self.assertTrue(got and all(len(x) <= 480 for x in got))
        self.assertIn("examples", programme.GATE)  # the gate was fixed before the arm ran
        with mock.patch.object(programme, "sealed_runs", return_value=[]):
            self.assertFalse(programme.examples_passed())

    def test_no_private_example_reaches_a_models_prompt(self):
        """30.7's open check: the gateway's recorded prompts, with examples on, hold no rules_only example."""
        from dclab_core import mapping
        from dclab_rnd import business_examples as kb
        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.draft.map import examples_for
        from dclab_rnd.models import settings
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage

        sent = []

        class Capture:
            def __init__(self, purpose):
                self.purpose = purpose

            def complete(self, messages, tools=None, max_tokens=1800, **options):
                sent.append(json.dumps(messages, default=str))
                return {"content": "{}", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}}

        gw = Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Capture(purpose))
        clean = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LLM_PROVIDER", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL",
                                 "DCLAB_NO_LIVE_MODELS")} | {f"DCLAB_TIER_{t.upper()}_{n}": "" for t in settings.TIERS for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")} \
            | {"OPENAI_API_KEY": "k"}  # a scripted transport behind a key nobody can use: as the data-limit tests make a tier available
        with mock.patch.dict(os.environ, clean):
            advisor = GatewayAdvisor(gw)
            advisor.examples = True
            self.assertTrue(gw.available("decomposer"))
            mapping.draw(FRAUD, advisor=advisor, examples=examples_for(FRAUD, advisor))
        self.assertTrue(sent)
        prompts = " ".join(sent)
        self.assertIn("similar_problems", prompts)
        for e in kb.load():
            if e.visibility == "rules_only":
                self.assertNotIn(e.id, prompts)
                self.assertNotIn(e.title, prompts)
        self.assertTrue(settings.PURPOSES["decomposer"])

    def test_the_search_is_measured_on_an_independent_writers_wording(self):
        from dclab_rnd import business_examples as kb

        doc = json.loads(kb.PARAPHRASES.read_text(encoding="utf-8"))
        self.assertEqual(len(doc["versions"]), 30)
        self.assertIn("saw only the ten questions", doc["writer"])
        measured = kb.paraphrase_recall()
        manifest = json.loads((kb.OUT / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["practice"]["other_words"]["found"], measured["found"])  # the committed knowledge base says what the search does
        self.assertLess(measured["found"], measured["questions"])  # an honest weak spot: other words are harder than the author's own


if __name__ == "__main__":
    unittest.main()
