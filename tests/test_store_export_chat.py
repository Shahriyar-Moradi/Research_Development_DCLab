"""D2 / P5: a set exported as chat JSONL in the model's own shape, and checked against its chat template.

Every assistant turn with recorded reasoning carries ``reasoning_content`` (a visible or a rule's thought never does), tool
calls have object arguments and tool turns their ``tool_call_id``; a sealed row is never exported; silver is refused; the
template check passes when the template keeps every earlier turn's thinking and fails, naming the turn, when it drops it.
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'assistant' %}"
    "{% if m.reasoning_content and (preserve_thinking or loop.last) %}<think>{{ m.reasoning_content }}</think>{% endif %}"
    "{% for tc in m.tool_calls or [] %}<tool_call>{{ tc.function.arguments | tojson }}</tool_call>{% endfor %}{{ m.content }}"
    "{% else %}<{{ m.role }}>{{ m.content }}{% endif %}\n"
    "{% endfor %}")


def provenance():
    from dclab_core.store import Provenance

    return Provenance(entry="e", sha256="0" * 64, source="s", workspace="w", at="2026-10-20T10:00:00+00:00", consent="train")


def chat(id_, split, sources=("recorded", "recorded")):
    from dclab_core.store import ChatExample, Message

    tools = [{"type": "function", "function": {"name": "note", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}}}}]
    messages = [Message(role="system", content="Be careful."), Message(role="user", content="Start.")]
    for n, source in enumerate(sources):
        labelled = {"reasoning": f"Thought {n}: call note.", "reasoning_source": source} if source else {"no_reasoning": "none kept"}
        messages.append(Message(role="assistant", content="", tool_calls=[{"id": f"c{n}", "name": "note", "arguments": {"text": f"step {n}"}}], **labelled))
        messages.append(Message(role="tool", content='{"ok": true}', tool_call_id=f"c{n}"))
    messages.append(Message(role="assistant", content="Done.", no_reasoning="an answer in words"))
    return ChatExample(id=id_, model_type="m4_workflow_llm", split=split, provenance=provenance(), task="t", stage="plan", messages=messages, tools=tools)


class ExportTests(unittest.TestCase):
    def setUp(self):
        from dclab_rnd.store import Layout
        from dclab_rnd.store.sets import write_set

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        self.layout = Layout(folder / "store")
        self.out = folder / "export"
        write_set(self.layout, "m4_workflow_llm", "chat/reasoning", [chat("a", "train"), chat("b", "dev"), chat("c", "sealed")], version="v1",
                  about="the reasoning-only set")
        write_set(self.layout, "m4_workflow_llm", "chat/silver", [chat("s", "train")], version="v1", consent="improve", about="silver")

    def test_the_export_has_the_models_shape_and_never_a_sealed_row(self):
        from dclab_rnd.store import export_chat

        report = export_chat.export(self.layout, "m4_workflow_llm", "chat/reasoning", "v1", self.out)
        self.assertEqual(report["counts"], {"train": 1, "dev": 1})
        self.assertFalse((self.out / "sealed.jsonl").exists())
        line = json.loads((self.out / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(line["tools"][0]["function"]["name"], "note")
        assistants = [m for m in line["messages"] if m["role"] == "assistant"]
        self.assertEqual([m.get("reasoning_content") for m in assistants], ["Thought 0: call note.", "Thought 1: call note.", None])
        self.assertEqual(assistants[0]["tool_calls"], [{"id": "c0", "type": "function", "function": {"name": "note", "arguments": {"text": "step 0"}}}])
        self.assertEqual([m["tool_call_id"] for m in line["messages"] if m["role"] == "tool"], ["c0", "c1"])
        manifest = json.loads((self.out / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["export"]["sealed"], "never exported")
        self.assertEqual(manifest["set"]["consent"], "train")

    def test_a_visible_thought_is_not_exported_as_reasoning(self):
        from dclab_rnd.store import export_chat

        line = export_chat.chat_line(chat("v", "train", sources=("visible", "recorded")))
        assistants = [m for m in line["messages"] if m["role"] == "assistant"]
        self.assertEqual([m.get("reasoning_content") for m in assistants], [None, "Thought 1: call note.", None])

    def test_silver_is_refused(self):
        from dclab_rnd.store import export_chat

        with self.assertRaises(export_chat.NotForTraining):
            export_chat.export(self.layout, "m4_workflow_llm", "chat/silver", "v1", self.out)

    def test_a_trajectory_becomes_turns_with_calls_and_whole_results(self):
        from dclab_core.store import SolverTrajectory, TrajectoryStep
        from dclab_rnd.store import export_chat

        t = SolverTrajectory(id="t", model_type="m5_solver_llm", split="train", provenance=provenance(), problem_kind="binary",
                             problem={"request": "Predict churn.", "data_sha256": "x", "answerer": "rules"},
                             steps=[TrajectoryStep(n=1, thought="Look first.", thought_source="recorded", tool="describe_data", arguments={"project_id": "p"},
                                                   result={"rows": 10}),
                                    TrajectoryStep(n=2, thought="the plan's line", thought_source="from_record", tool="run_all", arguments={"project_id": "p"},
                                                   result={"stages": 5})],
                             tools=[{"type": "function", "function": {"name": "describe_data"}}])
        line = export_chat.trajectory_line(t)
        self.assertEqual(json.loads(line["messages"][0]["content"]), {"problem_kind": "binary", "request": "Predict churn."})  # never the sha or the answerer
        self.assertEqual(line["messages"][1]["reasoning_content"], "Look first.")
        self.assertNotIn("reasoning_content", line["messages"][3])  # a rule's line is not the model's reasoning
        self.assertEqual(line["messages"][2], {"role": "tool", "tool_call_id": "call_1", "content": '{"rows": 10}'})


class TemplateCheckTests(unittest.TestCase):
    def line(self):
        from dclab_rnd.store import export_chat

        return export_chat.chat_line(chat("a", "train"))

    def test_a_template_that_keeps_every_turns_thinking_passes(self):
        from dclab_rnd.store import template_check

        rendered = template_check.render(TEMPLATE, self.line(), {"preserve_thinking": True})
        self.assertIn("<think>Thought 0: call note.</think>", rendered)
        self.assertIn('<tool_call>{"text": "step 0"}</tool_call>', rendered)
        self.assertEqual(template_check.check_line(rendered, self.line()), [])

    def test_a_template_that_drops_earlier_thinking_fails_naming_the_turn(self):
        from dclab_rnd.store import template_check

        rendered = template_check.render(TEMPLATE, self.line(), {"preserve_thinking": False})
        problems = template_check.check_line(rendered, self.line())
        self.assertEqual(len(problems), 2)
        self.assertTrue(problems[0].startswith("turn 2:"))  # the first assistant turn (after system and user)

    def test_arguments_rendered_as_a_python_repr_fail_even_when_the_schema_names_the_keys(self):
        """The tools' schema in the prompt quotes every argument key, so the check must look for the arguments' JSON itself."""
        from dclab_rnd.store import template_check

        repr_template = TEMPLATE.replace("{{ tc.function.arguments | tojson }}", "{{ tc.function.arguments }}") + "{{ tools | tojson }}"
        rendered = template_check.render(repr_template, self.line(), {"preserve_thinking": True})
        self.assertIn("{'text': 'step 0'}", rendered)
        problems = template_check.check_line(rendered, self.line())
        self.assertEqual(len(problems), 2)
        self.assertIn("did not render as their JSON object", problems[0])

    def test_a_thought_with_blank_lines_around_it_passes_and_sorted_keys_pass(self):
        from dclab_rnd.store import template_check

        line = self.line()
        line["messages"][2]["reasoning_content"] = "\n\nThought 0: call note.\n\n"  # templates strip the blank lines
        line["messages"][2]["tool_calls"][0]["function"]["arguments"] = {"text": "step 0", "again": False}
        sorted_template = TEMPLATE.replace("{{ tc.function.arguments | tojson }}", "{{ tc.function.arguments | tojson(sort_keys=True) }}")
        rendered = template_check.render(sorted_template, line, {"preserve_thinking": True})
        self.assertIn('{"again": false, "text": "step 0"}', rendered)
        self.assertEqual(template_check.check_line(rendered, line), [])

    def test_a_wire_shaped_call_with_string_arguments_and_a_tool_turn_without_its_id(self):
        from dclab_core.store import ChatExample, Message
        from dclab_rnd.store import export_chat

        ex = ChatExample(id="w", model_type="m4_workflow_llm", split="train", provenance=provenance(), task="t", stage="plan", messages=[
            Message(role="user", content="go"),
            Message(role="assistant", content="", reasoning="r", reasoning_source="recorded",
                    tool_calls=[{"id": "x1", "type": "function", "function": {"name": "note", "arguments": "{\"text\": \"a\"}"}}]),
            Message(role="tool", content="ok"),  # recorded without its call id
            Message(role="assistant", content="Done.", no_reasoning="words")])
        line = export_chat.chat_line(ex)
        self.assertEqual(line["messages"][1]["tool_calls"][0]["function"]["arguments"], {"text": "a"})
        self.assertEqual(line["messages"][2]["tool_call_id"], "x1")  # the preceding turn's call, in order

    def test_the_command_reads_an_export_and_a_template_file(self):
        from dclab_rnd.store import template_check

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        (folder / "train.jsonl").write_text(json.dumps(self.line()) + "\n", encoding="utf-8")
        (folder / "chat_template.jinja").write_text(TEMPLATE, encoding="utf-8")
        import io
        from contextlib import redirect_stdout

        out = io.StringIO()
        with redirect_stdout(out):
            code = template_check.main([str(folder / "train.jsonl"), "--tokenizer", str(folder)])
        self.assertEqual((code, json.loads(out.getvalue())["failed"]), (0, 0))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(template_check.main([str(folder / "train.jsonl"), "--tokenizer", str(folder), "--kwargs", "{}"]), 1)
        (folder / "chat_template.jinja").unlink()  # the template inside tokenizer_config.json, as many tokenizer folders carry it
        (folder / "tokenizer_config.json").write_text(json.dumps({"chat_template": TEMPLATE}), encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(template_check.main([str(folder / "train.jsonl"), "--tokenizer", str(folder)]), 0)

    def test_the_export_report_says_what_the_set_lacks(self):
        from dclab_rnd.store import Layout, export_chat
        from dclab_rnd.store.sets import write_set

        folder = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, folder, True)
        layout = Layout(folder / "store")
        bare = chat("n", "train")
        bare = bare.model_copy(update={"tools": []})  # calls, but the set kept no schemas
        write_set(layout, "m4_workflow_llm", "chat/bare", [bare], version="v1", about="bare")
        base = layout.sets("m4_workflow_llm", "chat", "bare", "v1")
        manifest = json.loads((base / "MANIFEST.json").read_text(encoding="utf-8"))
        manifest["counts"]["train"] = 4  # the files disagree with the manifest
        (base / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = export_chat.export(layout, "m4_workflow_llm", "chat/bare", "v1", folder / "out")
        self.assertEqual(report["lines_with_calls_but_no_tools"], 1)
        self.assertEqual(report["mismatch"], {"train": {"manifest": 4, "read": 1}})
        self.assertGreater(report["longest_line_chars"], 100)
        self.assertTrue(report["commit"] is None or len(report["commit"]) == 40)


if __name__ == "__main__":
    unittest.main()
