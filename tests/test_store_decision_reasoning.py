"""Every decision row and every step of a process carries its reasoning, written by code from what was known when it was made
(the owner, 2026-10-28: every example carries its reasoning, labelled by its source); never the label or a person's answer."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core.ledger import Answer  # noqa: E402
from dclab_core.store import DecisionExample, Episode, EpisodeStep  # noqa: E402
from dclab_rnd.store.sets import reasoning_counts  # noqa: E402
from dclab_rnd.store.sets.decisions import decision_reasoning  # noqa: E402
from dclab_rnd.store.sets.others import _move_why  # noqa: E402


class DecisionReasoningTests(unittest.TestCase):
    def test_code_never_writes_a_decision_rows_reasoning(self):  # the review of C2: it would carry the label or argue against it
        answers = [Answer(by="rule", answerer="rule:id_v1", value=True, reason="unique on every row"),
                   Answer(by="person", answerer="owner", value=False)]
        self.assertIsNone(decision_reasoning("column.is_identifier", {"uniqueness": "all"}, ["yes", "no"], answers))
        row = DecisionExample.model_validate({"id": "m1:b", "model_type": "m1_students", "split": "train", "point": "p", "group": "g",
                                              "provenance": {"workspace": "w", "source": "ledger/x.jsonl", "entry": "e", "sha256": "0" * 64,
                                                             "at": "2026-10-28T00:00:00+00:00"}})
        self.assertEqual(reasoning_counts([row]), {"missing": 1})  # counted as missing, never as a stated reason

    def test_old_rows_read_and_a_reasoning_comes_with_its_source(self):
        old = {"schema_version": "dclab_core.decision_example.v2", "id": "m1:a", "model_type": "m1_students", "split": "train",
               "provenance": {"workspace": "w", "source": "ledger/x.jsonl", "entry": "e", "sha256": "0" * 64, "at": "2026-10-28T00:00:00+00:00"},
               "point": "column.is_identifier", "group": "general"}
        ex = DecisionExample.model_validate(old)
        self.assertEqual(ex.schema_version, "dclab_core.decision_example.v4")  # 0.145.0
        self.assertIsNone(ex.reasoning)
        with self.assertRaises(ValueError):
            DecisionExample.model_validate({**old, "reasoning": "because"})  # a reasoning never comes without its source
        ex = DecisionExample.model_validate({**old, "reasoning": "Known then: uniqueness all.", "reasoning_source": "from_record"})
        self.assertEqual(reasoning_counts([ex]), {"from_record": 1})

    def test_a_step_says_what_was_known_before_it_never_its_own_outcome(self):
        why = _move_why({"move": "run_stage", "from": "WF-03", "to": "WF-04", "status": "blocked", "failed_checks": ["leakage_audit"],
                         "rules": ["DCLAB-R01"]})  # the PREVIOUS move: why the agent tried again
        self.assertTrue(why.startswith("Before this move:"))
        self.assertIn("blocked by the workflow validator, failing leakage_audit (rules DCLAB-R01)", why)
        step = EpisodeStep(n=0, action="run_stage:WF-03>WF-04", reasoning=why, reasoning_source="from_record")
        ep = Episode(id="m6:x", model_type="m6_stepwise_decisions", split="train",
                     provenance={"workspace": "w", "source": "projects/p/transitions.jsonl", "entry": "e", "sha256": "0" * 64, "at": "2026-10-28T00:00:00+00:00"},
                     process="graph", steps=[step])
        self.assertEqual(reasoning_counts([ep]), {"from_record": 1})
        old = {**ep.model_dump(mode="json"), "schema_version": "dclab_core.episode.v1"}
        self.assertEqual(Episode.model_validate(old).schema_version, "dclab_core.episode.v3")  # 0.145.0


class NoModelTests(unittest.TestCase):
    """2026-10-30: the gap list counted ~460,000 rule and classifier rows as "no reasoning and no note"; a row no model made
    needs only the note saying so, and a row a model made stays a gap."""

    PROV = {"workspace": "w", "source": "ledger/x.jsonl", "entry": "e", "sha256": "0" * 64, "at": "2026-10-30T00:00:00+00:00"}

    def row(self, *by):
        return DecisionExample.model_validate({"id": "m1:" + "-".join(by), "model_type": "m1_students", "split": "train", "point": "p",
                                               "group": "g", "provenance": self.PROV,
                                               "answers": [{"by": b, "value": True} for b in by]})

    def test_rule_classifier_and_person_rows_need_no_thinking_but_a_models_row_does(self):
        from dclab_rnd.store.sets import why_no_reasoning

        self.assertEqual(reasoning_counts([self.row("rule"), self.row("classifier", "person", "rule")]), {"no_model": 2})
        self.assertEqual(reasoning_counts([self.row("model_role", "rule")]), {"missing": 1})  # a model answered: its thinking is the gap
        self.assertIn("rule", why_no_reasoning(self.row("rule").model_dump(mode="json")))

    def test_codes_steps_need_none_but_an_agents_moves_do(self):
        def ep(process):
            return Episode(id=f"m6:{process}", model_type="m6_stepwise_decisions", split="train", provenance=self.PROV, process=process,
                           steps=[EpisodeStep(n=0, action="queued:pipeline"), EpisodeStep(n=1, action="done:pipeline")])

        self.assertEqual(reasoning_counts([ep("job")]), {"no_model": 2})
        self.assertEqual(reasoning_counts([ep("graph")]), {"missing": 2})  # who made the move is not on the record
        self.assertEqual(reasoning_counts([ep("dclab_code")]), {"missing": 2})  # an agent's steps without its thinking

    def test_the_gap_list_names_only_rows_a_model_made(self):
        from dclab_rnd.store import coverage as cov

        def doc(**m1):
            base = {"examples": 10, "recorded": 0, "none": 0, "note": 0, "no_model": 0, "by_agent": {}, "by_track": {}}
            return {"sources": {"by_agent": {}, "by_track": {}}, "model_types": {"m1_students": {**base, **m1}}, "routing": {}}

        rules_only = [g for g in cov.gaps(doc(no_model=10)) if g.startswith("m1_students")]
        self.assertEqual(rules_only, [])
        mixed = [g for g in cov.gaps(doc(no_model=7, none=3)) if g.startswith("m1_students")]
        self.assertEqual(len(mixed), 2)
        self.assertIn("3 of 10 examples were made by a model", mixed[1])


if __name__ == "__main__":
    unittest.main()


class LinkedThinkingTests(unittest.TestCase):
    """2026-10-30: a model's answer in the ledger named no request, so its kept thinking could never reach its decision row;
    from now on it names it (``source``), the sets link the used answer's thinking, and older rows say why they cannot."""

    def test_a_models_answer_names_the_request_that_made_it(self):
        """Only inside the tool call that request made (the review: the last call in a context is often another's)."""
        from dclab_rnd import experience, ledger

        experience.new_model_request()
        self.assertIsNone(ledger.answer("model_role", "intern", True).source)  # outside a tool call: nothing borrowed
        token = experience._ACTING_REQUEST.set("mracting")
        try:
            self.assertEqual(ledger.answer("model_role", "intern", True).source, "mracting")
        finally:
            experience._ACTING_REQUEST.reset(token)
        self.assertIsNone(ledger.answer("rule", "rule:id_v1", True).source)  # a rule's answer names none
        self.assertEqual(ledger.answer("model_role", "intern", True, source="mrgiven").source, "mrgiven")

    def test_the_used_answers_thinking_is_linked_from_its_own_agents_request_only(self):
        from dclab_rnd.store.sets import decisions as D

        thoughts = {"mr1": ("intern", "Unique per row and named id: an identifier.", {}), "mr2": ("home_agent", "Someone else's call.", {})}
        used = {"used": "model_role", "value_used": True, "answers": [{"by": "rule", "value": False}, {"by": "model_role", "answerer": "intern", "value": True, "source": "mr1"}]}
        self.assertEqual(D._linked_thinking(used, thoughts)[0], "Unique per row and named id: an identifier.")
        stale = {**used, "answers": [{"by": "model_role", "answerer": "intern", "value": True, "source": "mr2"}]}
        self.assertIsNone(D._linked_thinking(stale, thoughts))  # another agent's request: never borrowed
        beside = {"used": "rule", "value_used": False, "answers": [{"by": "rule", "value": False}, {"by": "model_role", "answerer": "intern", "value": True, "source": "mr1"}]}
        self.assertIsNone(D._linked_thinking(beside, thoughts))  # the rule decided: the model's thinking is not its reason
        self.assertTrue(D._why_none(beside).startswith("the rule answer was used"))  # names what was used
        unnamed = {**used, "answers": [{"by": "model_role", "answerer": "intern", "value": True}]}
        self.assertEqual(D._why_none({**unnamed, "at": "2026-10-25T10:00:00+00:00"}), D.LINK_NOTE)  # before answers named requests
        self.assertEqual(D._why_none({**unnamed, "at": "2026-11-02T10:00:00+00:00"}), D.UNNAMED_NOTE)  # after: never "lost before"
        self.assertEqual(D._why_none(stale, thoughts), D.OTHER_NOTE)
        shadow = {**used, "answers": [{"by": "model_role", "answerer": "shadow:m", "value": True, "source": "shadow"}]}
        self.assertIsNone(D._why_none(shadow))  # a shadow's answer is no answer of the record
        self.assertIsNone(D._why_none({"used": "rule", "answers": [{"by": "rule", "value": 1}]}))  # no model: nothing owed

    def test_an_older_models_row_counts_as_lost_never_as_missing(self):
        import json
        import tempfile

        from dclab_rnd.store import coverage as cov
        from dclab_rnd.store.layout import Layout
        from dclab_rnd.store.sets import decisions as D

        layout = Layout(Path(tempfile.mkdtemp()))
        folder = layout.sets("m1_students", "column.is_identifier", "v1")
        folder.mkdir(parents=True)
        (folder / "MANIFEST.json").write_text("{}")
        row = DecisionExample.model_validate({"id": "m1:z", "model_type": "m1_students", "split": "train", "point": "p", "group": "g",
                                              "provenance": NoModelTests.PROV, "answers": [{"by": "model_role", "value": True}],
                                              "no_reasoning": D.LINK_NOTE})
        (folder / "train.jsonl").write_text(row.model_dump_json() + "\n")
        doc = cov.measure(layout, home=Path(tempfile.mkdtemp()))
        m = doc["model_types"]["m1_students"]
        self.assertEqual((m["lost"], m["none"]), (1, 0))
        self.assertFalse([g for g in cov.gaps(doc) if g.startswith("m1_students:") and "examples were made by a model" in g])
