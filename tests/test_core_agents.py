"""Package 18.1: roles, the agent contract of dclab_core (Part 7 of the playbook).

A role must give the same output as its deterministic fallback whenever the advisor is absent, abstains, fails or
answers something its validator refuses; its prompt is a pinned file the host can send as is.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import pydantic  # noqa: F401
except ImportError as error:
    PYDANTIC_MISSING = str(error)
else:
    PYDANTIC_MISSING = ""


def toy_role():
    from dclab_core.agents import Role

    def parse(answer, inputs):
        value = answer.get("unit")
        if value not in inputs["allowed"]:
            raise ValueError("not an allowed unit")
        return {"unit": value}

    return Role(name="problem_analyst", description="a toy", evidence=lambda i: {"columns": len(i["allowed"])},
                parse=parse, fallback=lambda i: {"unit": None}, merge=lambda base, new: {**base, **new})


@unittest.skipIf(PYDANTIC_MISSING, PYDANTIC_MISSING)
class RoleTest(unittest.TestCase):
    INPUTS = {"allowed": ["ride", "order"]}

    def test_an_accepted_answer_is_used_and_said_so(self):
        from dclab_core.agents import ScriptedAdvisor, run_role

        advisor = ScriptedAdvisor({"problem_analyst": {"unit": "ride"}})
        result = run_role(toy_role(), advisor, self.INPUTS)
        self.assertEqual((result.output, result.source), ({"unit": "ride"}, "advisor"))
        self.assertEqual(advisor.requests, [("problem_analyst", {"columns": 2})])

    def test_no_advisor_abstaining_failing_or_refused_all_give_the_fallback(self):
        from dclab_core.agents import ScriptedAdvisor, run_role

        class Broken:
            def ask(self, purpose, evidence):
                raise TimeoutError("down")

        fallback = {"unit": None}
        cases = {"no advisor": None, "the advisor abstained": ScriptedAdvisor({}), "failed": Broken(),
                 "invalid": ScriptedAdvisor({"problem_analyst": {"unit": "galaxy"}})}
        for reason, advisor in cases.items():
            with self.subTest(case=reason):
                result = run_role(toy_role(), advisor, self.INPUTS)
                self.assertEqual((result.output, result.source, result.reason), (fallback, "fallback", reason))

    def test_rows_in_the_evidence_are_a_bug_not_a_fallback(self):
        import pandas as pd

        from dclab_core.advisor import EvidenceError
        from dclab_core.agents import Role, ScriptedAdvisor, run_role

        leaky = Role(name="problem_analyst", description="leaks", evidence=lambda i: {"rows": pd.DataFrame({"a": [1]})},
                     parse=lambda a, i: a, fallback=lambda i: {})
        with self.assertRaises(EvidenceError):
            run_role(leaky, ScriptedAdvisor({"problem_analyst": {}}), {})


    def test_every_role_has_its_prompt_and_the_analyst_runs_through_its_role(self):
        from dclab_core import prompts
        from dclab_core.agents import ScriptedAdvisor, roles
        from dclab_core.understand import understand

        for name in roles():
            self.assertIn(name, prompts.names())
        advisor = ScriptedAdvisor({"problem_analyst": {"action": {"value": "flag rides for review", "quote": "flag rides for review"}}})
        brief = understand("Detect fraudulent rides. The team will flag rides for review.", advisor=advisor)
        self.assertEqual([purpose for purpose, _ in advisor.requests], ["problem_analyst"])
        self.assertEqual((brief.action.source, brief.action.value), ("advisor", "flag rides for review"))


class PromptTest(unittest.TestCase):
    def test_every_prompt_is_pinned(self):
        from dclab_core import prompts

        self.assertTrue(prompts.names())
        self.assertEqual(prompts.fingerprints(), prompts.recorded(),
                         "a prompt changed: measure its role again, then run python -m dclab_core.prompts --record")

    def test_the_analysts_prompt_names_the_rules_that_keep_it_safe(self):
        from dclab_core import prompts

        text = " ".join(prompts.text("problem_analyst").split())
        for words in ("copied exactly", "Leave a field out rather than guess", "{}", "never values"):
            self.assertIn(words, text)

    def test_the_prompts_ship_in_the_wheel(self):
        text = (ROOT / "dclab_core" / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"dclab_core.prompts" = ["*.md", "HASHES.json"]', text)
        self.assertIn('"dclab_core.agents"', text)


if __name__ == "__main__":
    unittest.main()
