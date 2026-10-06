"""Stage explanations (package A3.3): a model's sentences from the stage record, cited, number-checked, never "production-ready"."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from dclab_rnd import cited, models  # noqa: E402
from dclab_rnd.studio import agent, engine, explain  # noqa: E402

EID = "PRJ-ab12cd-final"
RECORD = {
    "stage": "final", "title": "Final holdout", "question": "How does the chosen model score on rows it never saw?", "experiment_id": EID,
    "claims": [
        {"claim_id": f"{EID}-C1", "kind": "fact", "statement": "extra_trees scored ROC-AUC 0.9248 on the once-consumed holdout (95% interval 0.8889 to 0.9807).",
         "limitations": ["A single holdout is one draw."]},
        {"claim_id": f"{EID}-C3", "kind": "fact", "statement": "Trivial baselines on the same holdout: majority_class: accuracy 0.5410, roc_auc 0.5000.", "limitations": []}],
    "notes": [{"severity": "info", "title": "LLM critique (advisory)", "text": "Looks fine, 0.77.", "proof": [], "source": "llm"},
              {"severity": "warning", "title": "Not production-approved", "text": "Benchmark evidence is not deployment readiness.", "proof": ["DCLAB-R22"], "source": "evidence"}],
}


class Scripted:
    def __init__(self, content):
        self.content, self.seen, self.verdicts = content, [], []

    def complete(self, messages, tools=None, max_tokens=None, **_):
        self.seen.append(messages[-1]["content"])
        if isinstance(self.content, Exception):
            raise self.content
        return {"content": self.content, "tool_calls": [], "usage": {}, "assistant_message": {"role": "assistant", "content": self.content}}

    def output(self, ok, reason=""):
        self.verdicts.append((ok, reason))


class CheckTests(unittest.TestCase):
    SOURCES = {"A": "The model scored 0.9248 on 61 rows; the baseline scored 0.50.", "B": "It used 5 folds and an interval of 0.8889 to 0.9807."}

    def kept(self, text, **kw):
        return cited.check(text, self.SOURCES, **kw)

    def test_a_number_that_is_not_in_the_cited_source_removes_the_sentence(self):
        out = self.kept("It scored 0.9248 on 61 rows [A]. It scored 0.93 [A]. It used 5 folds [A]. It used 5 folds [B]. The interval is 0.8889 to 0.9807 [B].")
        self.assertEqual([k["text"] for k in out["kept"]], ["It scored 0.9248 on 61 rows.", "It used 5 folds.", "The interval is 0.8889 to 0.9807."])
        self.assertEqual(out["dropped"], [cited.REASONS[2]] * 2)  # a rounded number, and a number from another source

    def test_numbers_compare_as_numbers_and_identifiers_are_not_numbers(self):
        self.assertEqual(cited.numbers("0.50 and +0.5 and 7,043 and 54.1% and .99 [PRJ-ab12cd-final-C1] WF-03 DCLAB-R22"), {"0.5", "+0.5", "7043", "54.1%", "0.99"})
        self.assertEqual([k["text"] for k in self.kept("The baseline scored 0.5 [A].")["kept"]], ["The baseline scored 0.5."])

    def test_a_number_cannot_be_hidden_by_how_it_is_written(self):
        sources = {"A": "extra_trees scored ROC-AUC 0.9248; the change was -0.0034 against 54.1% accuracy for 61 rows; tuning added 0.0129."}
        dropped = ["It scored .99 [A].", "The AUC0.99 is high [A].", "It used 61k rows [A].", "Accuracy was 61% [A].", "Accuracy was 0.541 [A].",
                   "The change was +0.0034 [A].", "It improved by -0.0129 [A].", "The score doubled [A].", "Ninety-nine percent of it [A].", "It used five folds [A]."]
        kept = ["It scored .9248 on the holdout [A].", "It used 61 rows [A].", "Accuracy was 54.1% [A].", "The change was -0.0034 [A].", "The change was 0.0034 [A]."]
        for sentence in dropped:
            self.assertEqual(cited.check(sentence, sources)["dropped"], [cited.REASONS[2]], sentence)
        for sentence in kept:
            self.assertEqual(len(cited.check(sentence, sources)["kept"]), 1, sentence)

    def test_a_sentence_ends_at_a_full_stop_or_a_line_and_a_bullet_is_a_sentence(self):
        out = self.kept("The model is safe, fair and ready to go. extra_trees scored 0.9248 on the holdout [A].")
        self.assertEqual(([k["text"] for k in out["kept"]], out["dropped"]), (["extra_trees scored 0.9248 on the holdout."], [cited.REASONS[0]]))
        out = self.kept("- It scored 0.9248 [A]\n- It used 5 folds [A]\n* e.g. the baseline scored 0.50 [A]")
        self.assertEqual([k["text"] for k in out["kept"]], ["It scored 0.9248", "e.g. the baseline scored 0.50"])
        self.assertEqual(out["dropped"], [cited.REASONS[2]])  # 5 folds is source B's
        self.assertEqual(len(self.kept("It scored 0.9248 vs. a baseline of 0.50 [A].")["kept"]), 1)  # "vs." does not end a sentence

    def test_a_sentence_must_cite_and_cite_only_what_was_given(self):
        out = self.kept("It scored well. It scored 0.9248 [Z]. It scored 0.9248 [A][Z]. Cited after the stop. [A]")
        self.assertEqual([k["text"] for k in out["kept"]], ["Cited after the stop."])
        self.assertEqual(out["dropped"], [cited.REASONS[0], cited.REASONS[1], cited.REASONS[1]])

    def test_a_denial_must_sit_right_before_the_phrase(self):
        said = ["There is no sign of leakage, so the model is production-ready [A].", "It is not only accurate but production ready [A].", "No one doubts the model is production-ready [A].",
                "The model is production  ready [A].", "The model is production\u00a0ready [A].", "The model is production\u2011ready [A].", "Production-readiness is met [A].",
                "It is deployment-ready [A].", "It is fit for production [A].", "It is ready to ship [A].", "It is ready to be deployed [A].", "It is ready for use in production [A].",
                "It can go to production [A].", "- No leakage was found [A]\n- The model is production-ready [A]"]
        for sentence in said:
            self.assertFalse([k for k in self.kept(sentence)["kept"] if "ready" in k["text"].lower() or "Production" in k["text"] or "production" in k["text"]], sentence)
        for sentence in ("It is not yet production-ready [A].", "This is not production-approved [A].", "It isn't ready for production [A].", "It is never ready to ship [A]."):
            self.assertEqual(len(self.kept(sentence)["kept"]), 1, sentence)
        self.assertEqual(cited.strip_overclaims("Findings:\n- No leakage was found.\n- The model is production-ready.\n- Small sample."), ("Findings:\nNo leakage was found.\nSmall sample.", 1))

    def test_disclaimers_survive_and_idioms_that_mean_the_opposite_do_not(self):
        for sentence in ("This does not mean the model is ready for production [A].", "Benchmark evidence should not be treated as production-ready [A].",
                         "It does not show the model is production-ready [A]."):
            self.assertEqual(len(self.kept(sentence)["kept"]), 1, sentence)  # a denial up to four words before the phrase
        for sentence in ("It is not just production-ready [A].", "It is not merely production-ready [A].", "It is without doubt production-ready [A].",
                         "It is not a surprise that it is production-ready [A]."):
            self.assertEqual(self.kept(sentence)["kept"], [], sentence)

    def test_a_stop_without_a_space_still_ends_a_sentence(self):
        out = self.kept("The model is safe and fair.It scored 0.9248 [A].The baseline scored 0.50 [A].")
        self.assertEqual([k["text"] for k in out["kept"]], ["It scored 0.9248.", "The baseline scored 0.50."])
        self.assertEqual(out["dropped"], [cited.REASONS[0]])

    def test_a_number_word_is_the_digit_it_names_and_a_name_with_digits_is_not_a_number(self):
        sources = {"A": "Used 10 folds and 5 candidates. Rule C1 and R22 apply."}
        for sentence in ("It used ten folds [A].", "It used five candidates [A].", "Claim C1 and rule R22 apply [A]."):
            self.assertEqual(len(cited.check(sentence, sources)["kept"]), 1, sentence)
        for sentence in ("It used twenty folds [A].", "It used the top two candidates [A].", "It used half the folds [A]."):
            self.assertEqual(cited.check(sentence, sources)["dropped"], [cited.REASONS[2]], sentence)

    def test_advisory_text_with_nothing_to_remove_comes_back_exactly_as_it_was(self):
        text = "Risks:\n- Leakage is possible\n- Small sample\n1. Check the time split"
        self.assertEqual(cited.strip_overclaims(text), (text, 0))
        self.assertEqual(cited.strip_overclaims("Risks:\n- Leakage is possible\n- The model is production-ready\n- Small sample"), ("Risks:\nLeakage is possible\nSmall sample", 1))

    def test_production_readiness_is_removed_unless_it_is_denied(self):
        out = self.kept("The model is production-ready [A]. This proves the model works [A]. It is not production-approved [A]. "
                        "It is ready for production [A]. It is not ready for production [A].")
        self.assertEqual([k["text"] for k in out["kept"]], ["It is not production-approved.", "It is not ready for production."])
        self.assertEqual(out["dropped"], [cited.REASONS[3]] * 3)
        self.assertEqual(cited.strip_overclaims("Scores are stable. The model is production-ready. Check the folds."), ("Scores are stable. Check the folds.", 1))

    def test_the_word_limit_keeps_the_first_sentences(self):
        out = self.kept("Alpha beta gamma delta epsilon zeta [A]. Eta theta iota kappa lambda mu [A].", limit_words=8)
        self.assertEqual((len(out["kept"]), out["words"], out["dropped"]), (1, 6, [cited.REASONS[4]]))


class PageTests(unittest.TestCase):
    def test_the_page_labels_the_text_as_written_by_a_model_from_these_records(self):
        core = (ROOT / "dclab_rnd" / "agentic" / "static" / "app" / "js" / "core.js").read_text(encoding="utf-8")
        self.assertIn("written by a model from these records", core)  # the pill DC.explanation puts on every explanation
        self.assertIn("function explanation(record, title)", core)
        for page in ("notebook", "reliability", "brief"):
            self.assertIn("DC.explanation(", (ROOT / "dclab_rnd" / "agentic" / "static" / "app" / "js" / "views" / f"{page}.js").read_text(encoding="utf-8"), page)


class ExplainTests(unittest.TestCase):
    GOOD = (f"The chosen model scored ROC-AUC 0.9248 on the holdout, inside an interval of 0.8889 to 0.9807 [{EID}-C1]. "
            f"Always predicting the majority class scores 0.5000 [{EID}-C3]. "
            f"A single holdout is one draw [{EID}-C1]. It is not production-approved [DCLAB-R22].")

    def test_sources_are_the_claims_the_deterministic_notes_and_the_records_they_cite(self):
        sources = explain.sources_of(RECORD)
        self.assertEqual(list(sources)[:4], [f"{EID}-C1", f"{EID}-C3", f"{EID}-N2", "DCLAB-R22"])  # the model's own note is not a source
        self.assertIn("A single holdout is one draw.", sources[f"{EID}-C1"])  # a claim's limits come with it

    def test_a_good_explanation_is_kept_with_its_label_and_a_bad_sentence_is_removed(self):
        model = Scripted(self.GOOD + f" It improved by 0.4 points [{EID}-C1]. The model is production-ready [{EID}-C1].")
        out = explain.explain(RECORD, model)
        self.assertEqual([s["cites"] for s in out["sentences"]], [[f"{EID}-C1"], [f"{EID}-C3"], [f"{EID}-C1"], ["DCLAB-R22"]])
        self.assertEqual((out["dropped"], out["written_by"]), (2, "model"))
        self.assertTrue(out["label"].startswith("Written by a model from the stage record"))
        self.assertLessEqual(out["words"], explain.WORDS)
        self.assertEqual(model.verdicts, [(False, "a number is not in the cited source; claims production readiness, a guarantee or a cause")])
        self.assertIn(f"[{EID}-C1] (fact)", model.seen[0])

    def test_nothing_is_written_without_a_model_or_when_nothing_passes(self):
        self.assertIsNone(explain.explain(RECORD, None))
        self.assertIsNone(explain.explain(RECORD, Scripted("It scored 0.93 [PRJ-ab12cd-final-C1].")))
        self.assertIsNone(explain.explain(RECORD, Scripted(RuntimeError("BudgetExceeded: the cap is reached"))))
        clean = Scripted(self.GOOD)
        self.assertEqual(explain.explain(RECORD, clean)["dropped"], 0)
        self.assertEqual(clean.verdicts, [(True, "")])


CLEAN = {k: "" for k in ("OPENAI_BASE_URL", "OPENAI_MODEL", "DCLAB_LLM_BASE_URL", "DCLAB_LOG_PROMPTS", "DCLAB_NO_LIVE_MODELS")} | {"OPENAI_API_KEY": "k"}


class Router:
    """A scripted transport that answers by what is asked: the critique and the explanation are separate requests, and
    the gateway builds a new client for each, so the answer cannot depend on the order."""
    prompts: list[str] = []

    def __init__(self, explanation):
        self.explanation = explanation

    def complete(self, messages, tools=None, max_tokens=1800, **options):
        Router.prompts.append(messages[-1]["content"])
        asked = "You explain one stage result" in messages[-1]["content"]
        text = self.explanation if asked else "Advisory words. The model is production-ready."
        return {"content": text, "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}, "assistant_message": {"role": "assistant", "content": text}}


class StageTests(unittest.TestCase):
    def test_a_stage_run_stores_the_checked_explanation_beside_the_notes(self):
        import test_model_data_limits as limits
        from dclab_rnd.models.gateway import Gateway
        from dclab_rnd.models.usage import FileUsage

        env = mock.patch.dict(os.environ, {**CLEAN, **{f"DCLAB_TIER_{t}_{n}": "" for t in ("STRONG", "STANDARD", "CHEAP") for n in ("BASE_URL", "MODEL", "API_KEY", "LOCAL")}})
        env.start()
        self.addCleanup(env.stop)
        from dclab_rnd.studio import ProjectStore, data
        store = ProjectStore(Path(tempfile.mkdtemp()) / "projects")
        pid = store.create("Explained", "general", "Predict churn")["id"]
        limits.marked_table().to_parquet(store.data_dir(pid) / "t.parquet", index=False)
        data.attach_data(store, pid, "t.parquet")
        p = store.get(pid)
        p["solution"] = {"target": "churned", "task": "binary", "positive_label": "1", "prediction_moment": "At the monthly snapshot of each customer.",
                         "forbidden": [], "identifiers": ["note"], "time_column": None, "group_column": None, "text_columns": [], "metric": "roc_auc", "notes": ""}
        p["settings"]["quick"] = True
        store.save(p)
        eid = f"PRJ-{pid[:6]}-data"
        Router.prompts = []
        said = (f"The data stage read 300 source rows [{eid}-C1]. It used 999 rows [{eid}-C1]. This is production-ready [{eid}-C1].")
        models.install(Gateway(FileUsage(Path(tempfile.mkdtemp()) / "u.jsonl"), transport=lambda tier, purpose: Router(said)))
        self.addCleanup(models.install, None)
        record = engine.execute(store, pid, "data", "human")
        self.assertEqual([s["text"] for s in record["explanation"]["sentences"]], ["The data stage read 300 source rows."])
        self.assertEqual((record["explanation"]["dropped"], record["explanation"]["written_by"]), (2, "model"))
        critique = next(n for n in record["notes"] if n["source"] == "llm")
        self.assertEqual(critique["text"], "Advisory words.")  # the same filter on the advisory note
        self.assertIn("explanation", store.read_stage(pid, "data"))
        self.assertTrue(all(n["source"] != "llm" for n in record["notes"][1:]))  # the deterministic notes are untouched
        self.assertEqual(len(Router.prompts), 2)
        # approving the rule's own choice keeps the explanation; overriding it removes the words about the other one
        store_record = store.read_stage(pid, "data")
        self.assertIn("explanation", store_record)
        store_record["decision"] = {"kind": "recipe", "selected": "a", "chosen": "a", "options": [{"id": "a"}, {"id": "b"}], "rule": "r"}
        store.write_stage(pid, "data", store_record)
        project = store.get(pid)
        engine.approve(store, pid, "data", "a")
        self.assertIn("explanation", store.read_stage(pid, "data"))
        project = store.get(pid)
        project["stages"]["data"]["status"] = "completed"
        store.save(project)
        engine.approve(store, pid, "data", "b")
        self.assertNotIn("explanation", store.read_stage(pid, "data"))
        self.assertFalse([m for m in Router.prompts if "Q7Kx" in m])  # the table's cells never reach the model


if __name__ == "__main__":
    unittest.main()
