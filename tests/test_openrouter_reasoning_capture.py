"""2026-11-11 (DCLab moved to OpenRouter Qwen): the gateway keeps Qwen's reasoning the way it keeps Claude's thinking.

- The label says what the text is: Claude's API returns a summary ("API thinking summary"); OpenRouter returns the model's
  reasoning raw.
- ``reasoning`` and ``reasoning_details`` are words: kept only on a whole line (and a sealed exam's vault), never on a
  shape-only line, which keeps the redacted copy in ``thinking_redacted``.
- Every exchange line and usage row names its ``provider`` and ``source_class``, read from the endpoint's host.
- The request asks OpenRouter for an effort (DCLAB_THINKING_EFFORT, xhigh → high) and falls back once to ``enabled``.
Scripted transports only: no request leaves the machine.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.models import client as cl  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import settings  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402

OPENROUTER = {"name": "standard", "model": "qwen/qwen3.8-flash", "base_url": "https://openrouter.ai/api/v1", "api_key": "scripted"}
ANTHROPIC = {"name": "standard", "model": "claude-sonnet-5-5", "base_url": "https://api.anthropic.com", "api_key": "scripted"}
DETAILS = [{"type": "reasoning.text", "text": "The mail ann@example.com sits in a row; ", "format": "unknown", "index": 0},
           {"type": "reasoning.text", "text": "the target is churn.", "format": "unknown", "index": 1}]
QWEN_REPLY = {"content": "churn", "tool_calls": [], "usage": {"input_tokens": 10, "output_tokens": 5},
              "reasoning": "The mail ann@example.com sits in a row; the target is churn.",
              "thinking": [{"text": "The mail ann@example.com sits in a row; the target is churn."}], "reasoning_details": DETAILS,
              "reasoning_request": {"effort": "high"}}
CLAUDE_REPLY = {"content": "churn", "tool_calls": [], "usage": {"input_tokens": 10, "output_tokens": 5},
                "thinking": [{"text": "Summary: the target is churn.", "signature": "sig"}]}


class Scripted:
    thinking = None

    def __init__(self, reply):
        self.reply = reply

    def complete(self, messages, tools=None, max_tokens=None, **options):
        return json.loads(json.dumps(self.reply))


def _lines(home: Path, folder: str = "model_exchanges") -> list[dict]:
    return [json.loads(x) for f in sorted((home / folder).rglob("*.jsonl")) for x in f.read_text().splitlines()]


class SourceTest(unittest.TestCase):
    def test_provider_and_source_class_come_from_the_host(self):
        self.assertEqual(gw.source_of("api.anthropic.com", False), {"provider": "anthropic", "source_class": "commercial_model"})
        self.assertEqual(gw.source_of("openrouter.ai", False, "qwen/qwen3.8-flash"),
                         {"provider": "openrouter", "source_class": "open_weights_model"})
        self.assertEqual(gw.source_of("127.0.0.1:11434", True, "qwen3.5:4b"), {"provider": "local", "source_class": "open_weights_model"})
        self.assertEqual(gw.source_of("api.openai.com", False), {"provider": "openai", "source_class": "commercial_model"})
        self.assertEqual(gw.source_of("router.huggingface.co", False)["source_class"], "unknown")
        self.assertEqual(gw.source_of("not set", False)["provider"], "unknown")
        # a closed vendor's model served through OpenRouter is still a commercial model's words; gpt-oss is open weights
        self.assertEqual(gw.source_of("openrouter.ai", False, "anthropic/claude-sonnet-5-5")["source_class"], "commercial_model")
        self.assertEqual(gw.source_of("openrouter.ai", False, "openai/gpt-oss-120b")["source_class"], "open_weights_model")
        # an OpenRouter model of a family not named is unknown, never open weights by default
        self.assertEqual(gw.source_of("openrouter.ai", False, "mistralai/mistral-medium-3")["source_class"], "unknown")

    def test_a_summary_relayed_by_openrouter_is_called_a_summary(self):
        claude = gw.source_of("openrouter.ai", False, "anthropic/claude-sonnet-5-5")
        self.assertEqual(gw.thinking_label(claude, [{"type": "reasoning.text", "text": "x"}]), "API thinking summary (via OpenRouter)")
        qwen = gw.source_of("openrouter.ai", False, "qwen/qwen3.8-flash")
        self.assertEqual(gw.thinking_label(qwen, DETAILS), "model reasoning (raw, as returned by OpenRouter)")
        self.assertEqual(gw.thinking_label(qwen, [{"type": "reasoning.summary", "summary": "x"}]), "API thinking summary (via OpenRouter)")
        self.assertEqual(gw.thinking_label(gw.source_of("api.anthropic.com", False)), "API thinking summary")


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        self.text = mock.patch("dclab_rnd.experience.model_text_allowed", return_value=True)
        self.text.start()
        self.addCleanup(self.text.stop)

    def _bound(self, tier, reply, purpose="route_check"):
        g = gw.Gateway(FileUsage(self.home / "usage.jsonl"), transport=lambda t, p: Scripted(reply))
        return gw.Bound(g, purpose, settings.purpose(purpose), dict(tier, name=settings.purpose(purpose).tier), None, None)

    def test_an_openrouter_reply_is_kept_whole_with_its_reasoning_label_and_provider(self):
        bound = self._bound(OPENROUTER, QWEN_REPLY)
        bound.request_cells = False
        bound.complete([{"role": "user", "content": "which column?"}])
        line = _lines(self.home)[0]
        self.assertTrue(line["whole"])
        self.assertEqual(line["reasoning"], QWEN_REPLY["reasoning"])
        self.assertEqual(line["reasoning_details"], DETAILS)  # unmodified, as OpenRouter returned it
        self.assertEqual(line["thinking"][0]["text"], QWEN_REPLY["reasoning"])
        self.assertEqual(line["thinking_label"], "model reasoning (raw, as returned by OpenRouter)")
        self.assertEqual((line["provider"], line["source_class"]), ("openrouter", "open_weights_model"))
        self.assertEqual(line["reasoning_asked"], {"effort": "high"})
        row = json.loads((self.home / "usage.jsonl").read_text().splitlines()[0])
        self.assertEqual((row["provider"], row["source_class"]), ("openrouter", "open_weights_model"))

    def test_a_shape_only_line_keeps_no_raw_reasoning(self):
        bound = self._bound(OPENROUTER, QWEN_REPLY)
        bound.request_cells = True  # the prompt held cell values
        bound.complete([{"role": "user", "content": "rows"}])
        line = _lines(self.home)[0]
        self.assertFalse(line["whole"])
        self.assertNotIn("reasoning", line)
        self.assertNotIn("reasoning_details", line)
        self.assertNotIn("ann@example.com", json.dumps(line))
        self.assertIn("the target is churn", line["thinking_redacted"][0]["text"])  # kept redacted, as Claude's thinking is
        self.assertEqual(line["provider"], "openrouter")

    def test_a_sealed_exam_keeps_its_shape_here_and_the_reasoning_whole_in_its_vault(self):
        vault = self.home / "vault"
        self.text.stop()
        with mock.patch("dclab_rnd.experience.model_text_allowed", return_value=False), \
                mock.patch.dict(os.environ, {"DCLAB_SEALED_RUN": "CEX-T", "DCLAB_EXAM_VAULT": str(vault)}):
            bound = self._bound(OPENROUTER, QWEN_REPLY)
            bound.request_cells = False
            bound.complete([{"role": "user", "content": "hello"}])
        self.text.start()
        line = _lines(self.home)[0]
        self.assertFalse(line["whole"])
        self.assertNotIn("reasoning", line)
        self.assertNotIn("reasoning_details", line)
        self.assertNotIn("thinking_redacted", line)
        kept = _lines(vault)[0]
        self.assertEqual((kept["reasoning_details"], kept["reasoning"]), (DETAILS, QWEN_REPLY["reasoning"]))

    def test_claudes_exchange_keeps_its_summary_label(self):
        bound = self._bound(ANTHROPIC, CLAUDE_REPLY)
        bound.request_cells = False
        bound.complete([{"role": "user", "content": "which column?"}])
        line = _lines(self.home)[0]
        self.assertEqual(line["thinking_label"], "API thinking summary")
        self.assertEqual(line["thinking"], CLAUDE_REPLY["thinking"])
        self.assertEqual((line["provider"], line["source_class"]), ("anthropic", "commercial_model"))
        self.assertNotIn("reasoning", line)
        self.assertNotIn("reasoning_asked", line)

    def test_every_purpose_keeps_openrouters_reasoning(self):
        """Rule 8: a purpose shown no cells keeps the reasoning whole; one shown cells keeps it redacted. A new purpose is covered."""
        for name, p in settings.PURPOSES.items():
            if name in settings.NO_THINKING:
                continue
            bound = self._bound(OPENROUTER, QWEN_REPLY, purpose=name)
            bound._keep_exchange("r-" + name, [{"role": "user", "content": "x"}], json.loads(json.dumps(QWEN_REPLY)), "ok")
        lines = {x["purpose"]: x for x in _lines(self.home)}
        for name, p in settings.PURPOSES.items():
            if name in settings.NO_THINKING:
                continue
            line = lines[name]
            if int(p.cell_values or 0) > 0:
                self.assertTrue(line.get("thinking_redacted"), name)
                self.assertNotIn("reasoning_details", line, name)
            else:
                self.assertEqual(line.get("reasoning_details"), DETAILS, name)


def _response(reasoning="thought", details=None):
    message = SimpleNamespace(content="ok", tool_calls=None, reasoning=reasoning, reasoning_content=None, reasoning_details=details)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")],
                           usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2))


class Refusal(Exception):
    status_code = 400
    body = {"error": {"message": "reasoning.effort is not supported for this model", "code": 400}}


class EffortTest(unittest.TestCase):
    def setUp(self):
        cl.EFFORT_REFUSED.clear()
        self.addCleanup(cl.EFFORT_REFUSED.clear)
        self.sent: list[dict] = []

    def _client(self, answers):
        c = cl.ChatClient(model="qwen/qwen3.8-flash", base_url="https://openrouter.ai/api/v1", api_key="scripted")
        c.thinking = "adaptive"

        def create(**kwargs):
            self.sent.append(kwargs)
            answer = answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return answer

        c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return c

    def test_the_request_carries_the_effort(self):
        env = {"DCLAB_THINKING_EFFORT": "xhigh", "OPENROUTER_REASONING": "", "DCLAB_THINKING": ""}
        with mock.patch.dict(os.environ, env):
            out = self._client([_response(details=DETAILS)]).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(self.sent[0]["extra_body"]["reasoning"], {"effort": "high"})  # xhigh is Claude's word; OpenRouter's top is high
        self.assertEqual(out["reasoning_request"], {"effort": "high"})
        self.assertEqual(out["reasoning_details"], DETAILS)
        with mock.patch.dict(os.environ, {**env, "DCLAB_THINKING_EFFORT": "medium"}):
            self._client([_response()]).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(self.sent[1]["extra_body"]["reasoning"], {"effort": "medium"})
        with mock.patch.dict(os.environ, {**env, "DCLAB_THINKING_EFFORT": ""}):
            self._client([_response()]).complete([{"role": "user", "content": "hi"}])
        self.assertEqual(self.sent[2]["extra_body"]["reasoning"], {"enabled": True})

    def test_the_off_switches_still_send_no_reasoning(self):
        for env in ({"OPENROUTER_REASONING": "0", "DCLAB_THINKING": ""}, {"OPENROUTER_REASONING": "", "DCLAB_THINKING": "off"}):
            self.sent.clear()
            with mock.patch.dict(os.environ, {"DCLAB_THINKING_EFFORT": "high", **env}):
                self._client([_response()]).complete([{"role": "user", "content": "hi"}])
            self.assertNotIn("reasoning", self.sent[0].get("extra_body") or {}, env)

    def test_a_refused_effort_is_retried_once_with_enabled(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING_EFFORT": "high", "OPENROUTER_REASONING": "", "DCLAB_THINKING": ""}):
            client = self._client([Refusal("400 reasoning.effort is not supported"), _response(), _response()])
            out = client.complete([{"role": "user", "content": "hi"}])
            self.assertEqual([s["extra_body"]["reasoning"] for s in self.sent], [{"effort": "high"}, {"enabled": True}])
            self.assertEqual((out["refused_attempts"], out["reasoning_effort_refused"]), (1, "high"))
            self.assertEqual(out["reasoning_request"], {"enabled": True})
            client.complete([{"role": "user", "content": "again"}])  # learned: the next request asks for the default from the start
            self.assertEqual(self.sent[2]["extra_body"]["reasoning"], {"enabled": True})

    def test_an_unrelated_refusal_is_not_retried(self):
        for message in ("messages: field required", "Invalid assistant message: reasoning_details must be an array"):
            class Other(Exception):
                status_code = 400
                body = {"error": {"message": message}}

            self.sent.clear()
            with mock.patch.dict(os.environ, {"DCLAB_THINKING_EFFORT": "high", "OPENROUTER_REASONING": "", "DCLAB_THINKING": ""}):
                with self.assertRaises(RuntimeError):
                    self._client([Other("400 " + message), _response()]).complete([{"role": "user", "content": "hi"}])
            self.assertEqual(len(self.sent), 1, message)
            self.assertEqual(cl.EFFORT_REFUSED, set(), message)

    def test_a_single_attempt_learns_a_refused_effort_without_retrying(self):
        with mock.patch.dict(os.environ, {"DCLAB_THINKING_EFFORT": "high", "OPENROUTER_REASONING": "", "DCLAB_THINKING": ""}):
            client = self._client([Refusal("400 reasoning.effort is not supported"), _response()])
            with self.assertRaises(RuntimeError):
                client.complete([{"role": "user", "content": "hi"}], _single_attempt=True)
            self.assertEqual(len(self.sent), 1)  # one attempt: never a second request
            client.complete([{"role": "user", "content": "again"}], _single_attempt=True)
        self.assertEqual(self.sent[1]["extra_body"]["reasoning"], {"enabled": True})



class ReasoningAllowanceTests(unittest.TestCase):
    def test_the_reasoning_gets_room_beside_the_answer_on_openrouter(self):
        """SEX-008: 604 of 1,800 replies were empty because 300 tokens of reasoning used the whole limit."""
        from dclab_rnd.models import client as C

        c = C.ChatClient.__new__(C.ChatClient)
        c.base_url, c.model, c.chat_template_kwargs, c.thinking = "https://openrouter.ai/api/v1", "qwen/qwen3.8-flash", None, "adaptive"
        with mock.patch.dict(os.environ, {"DCLAB_REASONING_TOKENS": "", "OPENROUTER_REASONING": "1", "DCLAB_THINKING": ""}):
            sent = c._extra({"max_tokens": 300, "messages": []})
        self.assertEqual(sent["max_tokens"], 300 + 2048)
        with mock.patch.dict(os.environ, {"OPENROUTER_REASONING": "off"}):
            sent = c._extra({"max_tokens": 300, "messages": []})
        self.assertEqual(sent["max_tokens"], 300)  # no reasoning asked for: the limit is the caller's


if __name__ == "__main__":
    unittest.main()
