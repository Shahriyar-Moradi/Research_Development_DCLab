"""Stage 1 of the plan of 2026-11-15 (more room for the model): the model plumbing no longer breaks an answer before any
rule of the lead agent does. Each test is one row of the stage's table, found in the day's live runs:

1.1 an answer that is only a spent limit is asked again with room (and kept as what it was);
1.2 a reply cut inside a tool call is told so; the last step of a turn is announced;
1.3 a provider reply with nothing in it is retried, never a crash without a record;
1.4 a tool call written as text is run as a call, or taken out of the words;
1.5 arguments one step from the schema are put right, a tool is found without its prefix, a refusal names what is valid;
1.6 the provider's own cost is counted; the price rows of the models in use are today's;
1.7 the usage log is read once and then only what was appended; the second reading starts with the turn;
1.8 a client built for one model stays on it;
1.9 a reasoning summary is read as thinking.
Scripted transports only: no request leaves the machine.
"""

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_rnd.agents import Policy, Registry, Tool, run  # noqa: E402
from dclab_rnd.agents import registry as reg  # noqa: E402
from dclab_rnd.agents import runtime  # noqa: E402
from dclab_rnd.models import client as cl  # noqa: E402
from dclab_rnd.models import gateway as gw  # noqa: E402
from dclab_rnd.models import prices, settings  # noqa: E402
from dclab_rnd.models.usage import FileUsage  # noqa: E402

OPENROUTER = {"name": "standard", "model": "qwen/qwen3.8-flash", "base_url": "https://openrouter.ai/api/v1", "api_key": "scripted"}
TEXT_CALL = ('<tool_call>\n<function=query_table>\n<parameter=measures>\n[{"name": "n", "op": "count"}]\n</parameter>\n'
             '<parameter=group_by>\n["region"]\n</parameter>\n</function>\n</tool_call>')
TOOLS = [{"type": "function", "function": {"name": "query_table", "description": "q", "parameters": {"type": "object", "properties": {}}}}]


def message(content="", tool_calls=None, **more):
    return SimpleNamespace(content=content, tool_calls=tool_calls, **more)


def response(content="", finish="stop", usage=None, **more):
    usage = usage or SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    return SimpleNamespace(choices=[SimpleNamespace(message=message(content, **more), finish_reason=finish)], usage=usage)


def chat(answers):
    c = cl.ChatClient(model="qwen/qwen3.8-flash", base_url="https://openrouter.ai/api/v1", api_key="scripted")
    sent = []

    def create(**kwargs):
        sent.append(kwargs)
        answer = answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    c._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return c, sent


class ClientTests(unittest.TestCase):
    def test_a_reply_with_nothing_in_it_is_an_error_worth_one_more_try(self):
        for empty in (SimpleNamespace(choices=None, usage=None), SimpleNamespace(choices=[], usage=None),
                      SimpleNamespace(choices=[SimpleNamespace(message=None, finish_reason=None)], usage=None)):
            client, _ = chat([empty])
            with self.assertRaises(RuntimeError) as raised:
                client.complete([{"role": "user", "content": "hello"}])
            self.assertIn("EmptyReply", str(raised.exception))
            self.assertTrue(raised.exception.transient)  # the gateway tries again and records the failure

    def test_a_tool_call_written_as_text_runs_as_a_call_when_the_tool_was_offered(self):
        client, _ = chat([response("I will look.\n" + TEXT_CALL)])
        out = client.complete([{"role": "user", "content": "q"}], TOOLS)
        self.assertEqual(out["content"], "I will look.")
        self.assertEqual([(c["name"], c["arguments"]) for c in out["tool_calls"]],
                         [("query_table", {"measures": [{"name": "n", "op": "count"}], "group_by": ["region"]})])
        self.assertEqual(out["tool_calls_from_text"], 1)
        self.assertEqual(out["assistant_message"]["tool_calls"][0]["function"]["name"], "query_table")  # the next request sends a real call back
        as_json = '<tool_call>{"name": "query_table", "arguments": {"group_by": ["region"]}}</tool_call>'
        client, _ = chat([response(as_json)])
        self.assertEqual(client.complete([{"role": "user", "content": "q"}], TOOLS)["tool_calls"][0]["arguments"], {"group_by": ["region"]})

    def test_a_tool_call_written_as_text_never_reaches_the_person_as_the_answer(self):
        client, _ = chat([response(TEXT_CALL)])
        out = client.complete([{"role": "user", "content": "q"}], None)  # the last request of a turn: no tool offered
        self.assertEqual((out["content"], out["tool_calls"], out["tool_calls_stripped"]), ("", [], 1))
        client, _ = chat([response("My answer.\n" + TEXT_CALL.replace("query_table", "another_tool"))])
        out = client.complete([{"role": "user", "content": "q"}], TOOLS)  # a tool this request did not offer
        self.assertEqual((out["content"], out["tool_calls"]), ("My answer.", []))
        kept = "The tag <tool_call> opens a call; this is prose about it."
        client, _ = chat([response(kept)])
        self.assertEqual(client.complete([{"role": "user", "content": "q"}], TOOLS)["content"], kept)  # no readable call: the words stay

    def test_a_text_call_cut_at_the_limit_is_never_run_with_half_an_argument(self):
        """The review of stage 1: the reply ended inside the block; the call ran with "Churn is highest in the Premi"."""
        tools = [{"type": "function", "function": {"name": "case_answer", "description": "a", "parameters": {"type": "object", "properties": {}}}}]
        cut = "Here it is.\n<tool_call>\n<function=case_answer>\n<parameter=text>\nChurn is highest in the Premi"
        client, _ = chat([response(cut, finish="length")])
        out = client.complete([{"role": "user", "content": "q"}], tools)
        self.assertEqual(out["content"], "Here it is.")
        self.assertEqual([(c["name"], list(c["arguments"])) for c in out["tool_calls"]], [("case_answer", ["_raw"])])  # the loop says "cut", nothing runs
        self.assertTrue(out["tool_call_cut"])
        as_json = 'Here it is.\n<tool_call>{"name": "case_answer", "arguments": {"text": "Churn is highest in the Premi'
        client, _ = chat([response(as_json, finish="length")])
        out = client.complete([{"role": "user", "content": "q"}], tools)
        self.assertEqual((out["content"], list(out["tool_calls"][0]["arguments"])), ("Here it is.", ["_raw"]))
        client, _ = chat([response(cut, finish="length")])
        out = client.complete([{"role": "user", "content": "q"}], None)  # no tool offered: taken out of the words, nothing run
        self.assertEqual((out["content"], out["tool_calls"], out["tool_calls_stripped"]), ("Here it is.", [], 1))
        whole_then_cut = TEXT_CALL + "\n<tool_call>\n<function=query_table>\n<parameter=measures>\n[{\"name\": \"n\""
        client, _ = chat([response(whole_then_cut, finish="length")])
        out = client.complete([{"role": "user", "content": "q"}], TOOLS)
        self.assertEqual([("_raw" in c["arguments"]) for c in out["tool_calls"]], [False, True])  # the whole one runs, the cut one does not
        self.assertEqual(len({c["id"] for c in out["tool_calls"]}), 2)

    def test_two_replies_never_share_a_text_calls_id(self):
        client, _ = chat([response(TEXT_CALL), response(TEXT_CALL)])
        first = client.complete([{"role": "user", "content": "q"}], TOOLS)["tool_calls"][0]["id"]
        second = client.complete([{"role": "user", "content": "q"}], TOOLS)["tool_calls"][0]["id"]
        self.assertNotEqual(first, second)

    def test_the_providers_cost_and_the_thinking_tokens_are_read(self):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, cost=0.00042,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=40))
        client, _ = chat([response("an answer", usage=usage)])
        self.assertEqual(client.complete([{"role": "user", "content": "q"}])["usage"],
                         {"input_tokens": 100, "output_tokens": 50, "reasoning_tokens": 40, "provider_cost": 0.00042})
        client, _ = chat([response("an answer")])
        self.assertEqual(client.complete([{"role": "user", "content": "q"}])["usage"], {"input_tokens": 10, "output_tokens": 5})

    def test_a_reasoning_summary_is_read_as_thinking(self):
        details = [SimpleNamespace(model_dump=lambda: {"type": "reasoning.summary", "summary": "The outcome is churn."})]
        client, _ = chat([response("churn", reasoning_details=details)])
        out = client.complete([{"role": "user", "content": "q"}])
        self.assertEqual(out["thinking"], [{"text": "The outcome is churn."}])

    def test_a_rate_limit_that_names_its_wait_says_so(self):
        limited = Exception("rate limited")
        limited.response = SimpleNamespace(headers={"retry-after": "7"})
        self.assertEqual(cl.ChatClient(model="m", base_url="https://openrouter.ai/api/v1", api_key="k")._failed(limited).retry_after, 7.0)
        limited.response = SimpleNamespace(headers={"retry-after": "900"})
        self.assertEqual(cl.ChatClient(model="m", base_url="https://openrouter.ai/api/v1", api_key="k")._failed(limited).retry_after, cl.RETRY_AFTER_MAX)


class Plan:
    """A transport that answers from a list and keeps what it was asked."""

    thinking = None

    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    def complete(self, messages, tools=None, max_tokens=None, **options):
        self.asked.append(max_tokens)
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return json.loads(json.dumps(answer))


def lines(home: Path) -> list[dict]:
    return [json.loads(x) for f in sorted((home / "model_exchanges").rglob("*.jsonl")) for x in f.read_text().splitlines()]


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp())
        text = mock.patch("dclab_rnd.experience.model_text_allowed", return_value=True)
        text.start()
        self.addCleanup(text.stop)
        self.sleeps: list[float] = []

    def bound(self, answers, purpose="route_check", **more):
        self.plan = Plan(answers)
        g = gw.Gateway(FileUsage(self.home / "usage.jsonl"), transport=lambda t, p: self.plan, sleep=self.sleeps.append)
        b = gw.Bound(g, purpose, settings.purpose(purpose), dict(OPENROUTER, name=settings.purpose(purpose).tier), None, None, **more)
        b.request_cells = False
        return b

    def test_an_answer_that_is_only_a_spent_limit_is_asked_again_with_room(self):
        spent = {"content": "", "tool_calls": [], "finish_reason": "length", "usage": {"input_tokens": 50, "output_tokens": 3848},
                 "thinking": [{"text": "a long thought"}], "reasoning": "a long thought"}
        good = {"content": "the plan", "tool_calls": [], "finish_reason": "stop", "usage": {"input_tokens": 50, "output_tokens": 900}}
        out = self.bound([spent, good]).complete([{"role": "user", "content": "plan it"}], None, 1800)
        self.assertEqual(out["content"], "the plan")
        self.assertEqual(self.plan.asked, [1800, 9800])  # four times, or 8,000 more: the same request, more room
        self.assertEqual(out["retried_after_empty"], {"limit": 1800, "again_with": 9800, "first_output_tokens": 3848})
        first, second = lines(self.home)
        self.assertEqual((first["finish_reason"], first["empty_cut"], first["thinking"]), ("length", True, [{"text": "a long thought"}]))  # kept as what it was
        self.assertEqual((second["finish_reason"], second.get("empty_cut")), ("stop", None))
        self.assertEqual(second["retried_after_empty"], out["retried_after_empty"])  # the kept exchange says it was a second asking
        self.assertEqual(len((self.home / "usage.jsonl").read_text().splitlines()), 2)  # both requests are counted

    def test_an_empty_answer_is_asked_again_once_and_a_strict_call_never(self):
        spent = {"content": "", "tool_calls": [], "finish_reason": "length", "usage": {"input_tokens": 1, "output_tokens": 1}}
        out = self.bound([spent, spent]).complete([{"role": "user", "content": "q"}], None, 1800)
        self.assertEqual((out["content"], len(self.plan.asked)), ("", 2))  # twice empty: said as it is, not tried a third time
        self.bound([spent]).complete([{"role": "user", "content": "q"}], None, 1800, single_attempt=True)
        self.assertEqual(self.plan.asked, [1800])
        cut_call = {**spent, "tool_calls": [{"id": "c", "name": "t", "arguments": {}}]}
        self.bound([cut_call]).complete([{"role": "user", "content": "q"}], None, 1800)
        self.assertEqual(self.plan.asked, [1800])  # a reply that holds a call is not empty
        self.assertIsNone(gw.more_room(gw.MORE_ROOM_MAX))
        self.assertIsNone(gw.more_room(None))

    def test_a_retry_a_cap_refuses_leaves_the_caller_with_the_empty_answer_it_had(self):
        spent = {"content": "", "tool_calls": [], "finish_reason": "length", "usage": {"input_tokens": 50, "output_tokens": 3848}}
        bound = self.bound([spent, {"content": "never sent", "tool_calls": [], "usage": {}}])
        bound.run_limit_eur = 0.004  # room for the first request, none for one with four times the limit
        out = bound.complete([{"role": "user", "content": "plan it"}], None, 1800)
        self.assertEqual((out["content"], self.plan.asked), ("", [1800]))
        self.assertIn("BudgetExceeded", out["retry_refused"])

    def test_a_client_built_for_one_model_stays_on_it(self):
        judge = {"name": "strong", "model": "nvidia/nemotron-3.5-super-vl:free", "base_url": "https://openrouter.ai/api/v1", "api_key": "scripted"}
        g = gw.Gateway(FileUsage(self.home / "usage.jsonl"), transport=lambda t, p: Plan([{"content": "ok", "tool_calls": [], "usage": {}}] * 2))
        fixed = gw.Bound(g, "route_check", settings.purpose("route_check"), judge, None, None, fixed=True)
        fixed.complete([{"role": "user", "content": "q"}])
        self.assertEqual((fixed.model, fixed.tier["name"]), ("nvidia/nemotron-3.5-super-vl:free", "strong"))
        moved = gw.Bound(g, "route_check", settings.purpose("route_check"), judge, None, None)  # as before: the routing may move it
        moved.complete([{"role": "user", "content": "q"}])
        self.assertNotEqual(moved.tier["name"], "strong")

    def test_the_providers_own_cost_is_what_is_counted(self):
        reply = {"content": "ok", "tool_calls": [], "usage": {"input_tokens": 1000, "output_tokens": 1000, "provider_cost": 0.0123}}
        self.bound([reply]).complete([{"role": "user", "content": "q"}])
        row = json.loads((self.home / "usage.jsonl").read_text().splitlines()[0])
        self.assertEqual((row["cost_eur"], row["cost_basis"]), (0.0123, "provider"))
        self.assertEqual(prices.cost("a/model-with-no-price", False, 10, 10, provider_cost=0.5), (0.5, "provider"))  # counted though the table lacks it
        self.assertEqual(prices.cost("a/model-with-no-price", False, 10, 10), (None, "no price"))  # no price is ever guessed
        from dclab_rnd.models.usage import FileUsage as Usage

        totals = Usage(self.home / "usage.jsonl").totals()
        self.assertEqual((totals["priced"], totals["unpriced"]), (1, 0))  # the review: a request the provider costed read as "no configured price"

    def test_a_provider_cost_of_zero_on_a_priced_model_is_not_believed(self):
        # a bring-your-own-key account is told the provider's fee only: the table's price counts, so a cap still moves
        self.assertEqual(prices.cost("qwen/qwen3.8-flash", False, 1_000_000, 1_000_000, provider_cost=0.0), (0.62, "price"))
        self.assertEqual(prices.cost("nvidia/nemotron-3.5-super-vl:free", False, 1_000_000, 1_000_000, provider_cost=0.0), (0.0, "provider"))
        self.assertEqual(prices.cost("a/model-with-no-price", False, 10, 10, provider_cost=0.0), (0.0, "provider"))

    def test_a_failure_that_is_no_runtime_error_is_recorded_before_it_is_raised(self):
        with self.assertRaises(TypeError):
            self.bound([TypeError("'NoneType' object is not subscriptable")]).complete([{"role": "user", "content": "q"}])
        row = json.loads((self.home / "usage.jsonl").read_text().splitlines()[0])
        self.assertEqual(row["outcome"], "TypeError: the request failed")

    def test_a_rate_limit_is_waited_for_as_long_as_it_asks(self):
        limited = RuntimeError("RateLimitError: rate limited")
        limited.transient, limited.retry_after = True, 12.0
        out = self.bound([limited, {"content": "ok", "tool_calls": [], "usage": {}}]).complete([{"role": "user", "content": "q"}])
        self.assertEqual((out["content"], self.sleeps), ("ok", [12.0]))


class PriceTests(unittest.TestCase):
    def test_the_models_in_use_have_todays_prices(self):
        flash = prices.price_of("qwen/qwen3.8-flash")
        self.assertEqual((flash.input, flash.output, flash.as_of), (0.15, 0.47, "2026-11-15"))
        self.assertEqual(prices.price_of("qwen3.8-flash"), prices.price_of("qwen/qwen3.8-flash").__class__(0.15, 0.47, "2026-11-15", flash.source))
        self.assertEqual(prices.price_of("qwen/qwen3.8-27b").output, 1.875)
        for free in ("nvidia/nemotron-3.5-super-vl:free", "nvidia/nemotron-3-ultra-550b-a55b:free"):
            self.assertEqual(prices.cost(free, False, 10_000, 10_000), (0.0, "price"))  # a checked price of zero: a cap can count it


class UsageTests(unittest.TestCase):
    def test_the_log_is_parsed_once_and_then_only_what_was_appended(self):
        path = Path(tempfile.mkdtemp()) / "usage.jsonl"
        usage = FileUsage(path)
        for n in range(3):
            usage.record({"at": "2026-11-15T10:00:00+00:00", "purpose": "route_check", "cost_eur": 0.01})
        self.assertEqual(usage.spent("2026-11-01"), 0.03)
        offset = usage._offset
        self.assertEqual(usage.spent("2026-11-01"), 0.03)  # nothing new: nothing read again
        self.assertEqual(usage._offset, offset)
        usage.record({"at": "2026-11-15T10:00:01+00:00", "purpose": "route_check", "cost_eur": 0.02})
        self.assertEqual((usage.spent("2026-11-01"), len(usage._rows)), (0.05, 4))
        other = FileUsage(path)  # another process appends: seen by this one
        other.record({"at": "2026-11-15T10:00:02+00:00", "purpose": "route_check", "cost_eur": 0.05})
        self.assertEqual(usage.spent("2026-11-01"), 0.1)

    def test_the_kept_rows_hold_no_prompt_text(self):
        path = Path(tempfile.mkdtemp()) / "usage.jsonl"
        usage = FileUsage(path)
        usage.record({"at": "2026-11-15T10:00:00+00:00", "project_id": "p1", "prompt": "a long prompt " * 500, "cost_eur": 0.01})
        usage.record({"at": "2026-11-15T10:00:00+00:00", "project_id": "p1", "cost_eur": 0.01})
        self.assertEqual(usage.count_prompts(project_id="p1"), 1)
        self.assertEqual([r.get("prompt") for r in usage._rows], [True, None])  # whether it has one; the text stays in the file
        self.assertIn("a long prompt", usage.recent()[1]["prompt"])  # the latest rows are read whole from the file (newest first)
        self.assertEqual([r.get("cost_eur") for r in usage.recent(1)], [0.01])
        many = FileUsage(path.with_name("many.jsonl"))
        for n in range(3000):
            many.record({"at": "2026-11-15T10:00:00+00:00", "seconds": n, "prompt": "p" * 200})
        self.assertEqual([r["seconds"] for r in many.recent(3)], [2999, 2998, 2997])  # read from the end, in blocks
        self.assertIn("a long prompt", path.read_text(encoding="utf-8"))
        self.assertEqual(usage.forget_prompts(project_id="p1"), 1)
        self.assertEqual(usage.count_prompts(project_id="p1"), 0)

    def test_a_replaced_or_shortened_log_is_read_again_from_its_start(self):
        path = Path(tempfile.mkdtemp()) / "usage.jsonl"
        usage = FileUsage(path)
        for n in range(3):
            usage.record({"at": "2026-11-15T10:00:00+00:00", "cost_eur": 0.01})
        self.assertEqual(usage.spent("2026-11-01"), 0.03)
        fresh = path.with_suffix(".tmp")
        fresh.write_text(json.dumps({"at": "2026-11-15T10:00:00+00:00", "cost_eur": 0.5}) + "\n", encoding="utf-8")
        fresh.replace(path)
        self.assertEqual(usage.spent("2026-11-01"), 0.5)
        path.write_text(json.dumps({"at": "2026-11-15T10:00:00+00:00", "cost_eur": 0.25}), encoding="utf-8")  # no newline at its end
        self.assertEqual(usage.spent("2026-11-01"), 0.25)
        path.unlink()
        self.assertEqual(usage.spent("2026-11-01"), 0.0)


def obj(properties=None, required=()):
    return {"type": "object", "properties": properties or {}, "required": list(required)}


def said(content=None, calls=(), finish="stop"):
    calls = [{"id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(calls)]
    return {"content": content, "tool_calls": calls, "finish_reason": finish, "usage": {"input_tokens": 1, "output_tokens": 1},
            "assistant_message": {"role": "assistant", "content": content,
                                  "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]}}


class Model:
    def __init__(self, *replies):
        self.replies, self.seen = list(replies), []

    def complete(self, messages, tools=None, **options):
        self.seen.append({"messages": [dict(m) for m in messages], "options": options})
        return self.replies.pop(0) if self.replies else said("done")


class ArgumentTests(unittest.TestCase):
    PROPS = {"filters": {"type": "array"}, "measures": {"type": "array"}, "limit": {"type": "integer"}, "note": {"type": "string"},
             "options": {"type": "object"}, "strict": {"type": "boolean"}}

    def test_a_list_or_an_object_sent_as_its_text_is_read(self):
        got = reg.repaired({"filters": '[{"column": "a", "op": "eq", "value": 1}]', "options": '{"a": 1}', "limit": "5", "strict": "true",
                            "note": "[not a list, a note]"}, self.PROPS)
        self.assertEqual(got, {"filters": [{"column": "a", "op": "eq", "value": 1}], "options": {"a": 1}, "limit": "5", "strict": "true",
                               "note": "[not a list, a note]"})  # a number or a yes/no sent as text is still a wrong type to read (H1)
        self.assertEqual(reg.repaired({"filters": "all of them", "limit": "five", "filters2": "[1]"}, self.PROPS),
                         {"filters": "all of them", "limit": "five", "filters2": "[1]"})  # what does not parse to the wanted type goes on as it came

    def test_a_wrong_type_is_still_a_wrong_type(self):
        self.assertEqual(reg.repaired({"note": 2024, "limit": "5"}, self.PROPS), {"note": 2024, "limit": "5"})  # refused by the check, with words to read (H1)
        self.assertEqual(reg.repaired({"either": "[1]"}, {"either": {"type": ["array", "string"]}}), {"either": "[1]"})  # several types: left alone
        calls, _ = cl.text_tool_calls("<tool_call><function=t><parameter=year>2024</parameter><parameter=ids>[1, 2]</parameter></function></tool_call>")
        self.assertEqual(calls[0]["arguments"], {"year": "2024", "ids": [1, 2]})  # a tagged value is text unless it is written as a list or an object

    def test_one_arguments_text_holding_the_next_ones_is_split(self):
        merged = {"filters": '\n[{"column": "destination", "op": "ne", "value": "TEST"}]\n<parameter=measures>\n[{"name": "n", "op": "count"}]\n'}
        self.assertEqual(reg.repaired(merged, self.PROPS), {"filters": [{"column": "destination", "op": "ne", "value": "TEST"}],
                                                            "measures": [{"name": "n", "op": "count"}]})

    def test_an_unknown_argument_is_refused_with_the_valid_names(self):
        r = Registry()
        r.register(Tool("start", "Start.", obj({"track": {"type": "string"}, "tool": {"type": "string"}}, ["track"]), lambda track, tool=None: {"ok": track}))
        error = r.call("start", {"track": "prediction", "outcome": "churned"})["error"]
        self.assertIn("Unknown argument(s) for start: outcome", error)
        self.assertIn("Its arguments are: tool, track", error)
        self.assertEqual(r.call("start", {"track": "prediction"}), {"ok": "prediction"})


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.r = Registry()
        self.r.register(Tool("case_request_build", "Build.", obj({"what": {"type": "string"}}), lambda what=None: {"built": what}))
        self.r.register(Tool("case_note", "Note.", obj({"value": {"type": "string"}}), lambda value=None: {"noted": value}))

    def policy(self, **more):
        return Policy(name="t", system="", tools=("case_request_build", "case_note"), **more)

    def test_a_tool_named_without_its_prefix_is_found(self):
        result = run(self.policy(), Model(said(None, [("request_build", {"what": "a model"})]), said("Done.")), self.r, [{"role": "user", "content": "go"}])
        self.assertEqual([(s.tool, s.ok, s.result) for s in result.steps], [("case_request_build", True, {"built": "a model"})])
        missing = run(self.policy(), Model(said(None, [("build", {})]), said("Done.")), self.r, [{"role": "user", "content": "go"}])
        self.assertIn("is not available here", missing.steps[0].result["error"])  # no guess: only the exact name after the prefix

    def test_a_reply_cut_inside_a_call_is_told_it_was_cut(self):
        cut = said(None, [("case_note", {"_raw": '{"value": "a long no'})], finish="length")
        result = run(self.policy(), Model(cut, said("Done.")), self.r, [{"role": "user", "content": "go"}])
        self.assertIn("cut at its length limit", result.steps[0].result["error"])
        self.assertNotIn("not valid JSON", result.steps[0].result["error"])
        whole = said(None, [("case_note", {"_raw": "{broken"})])  # not cut: the old message stands
        self.assertIn("not valid JSON", run(self.policy(), Model(whole, said("Done.")), self.r, [{"role": "user", "content": "go"}]).steps[0].result["error"])

    def test_the_reply_limit_of_the_policy_is_sent_and_the_last_step_is_announced(self):
        model = Model(said(None, [("case_note", {"value": "a"})]), said(None, [("case_note", {"value": "b"})]), said("My message."))
        result = run(self.policy(max_tokens=3000, warn_last=True, max_replies=3), model, self.r, [{"role": "user", "content": "go"}])
        self.assertEqual([c["options"] for c in model.seen], [{"max_tokens": 3000}] * 3)
        tools = [m["content"] for m in result.messages if m.get("role") == "tool"]
        self.assertNotIn(runtime.LAST_STEP, tools[0])
        self.assertTrue(tools[1].endswith(runtime.LAST_STEP))  # before the third and last reply
        quiet = Model(said(None, [("case_note", {"value": "a"})]), said("My message."))
        result = run(self.policy(max_replies=2), quiet, self.r, [{"role": "user", "content": "go"}])
        self.assertEqual(quiet.seen[0]["options"], {})  # a policy that sets neither is sent as before
        self.assertNotIn(runtime.LAST_STEP, json.dumps(result.messages))


class RouteAheadTests(unittest.TestCase):
    def test_the_second_reading_starts_with_the_turn_and_is_asked_once(self):
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent

        asked, release = [], threading.Event()

        class Second:
            def complete(self, messages, tools=None, **options):
                asked.append(messages[-1]["content"])
                release.wait(5)
                return {"content": "analysis", "tool_calls": [], "usage": {}}

        state = case_agent.CaseState(case=C.hear(C.CaseFile(case_id="t"), "Why did our numbers move last quarter?"))
        state.route_client, state.go_ahead = Second(), True  # a turn where a propose or a start can reach it
        case_agent.route_ahead(state)
        self.assertIsNotNone(state.route_thread)  # under way beside the lead model's own work
        case_agent.route_ahead(state)  # a second start in the same turn starts nothing
        release.set()
        self.assertEqual(case_agent.route_now(state), "analysis")
        self.assertEqual(case_agent.route_now(state), "analysis")
        self.assertEqual(len(asked), 1)
        self.assertIsNone(state.route_thread)

    def test_a_turn_that_can_only_ask_is_not_charged_and_a_finished_turn_does_not_wait(self):
        import time

        from dclab_core import case as C
        from dclab_rnd.agents import case_agent

        class Never:
            def complete(self, *args, **kwargs):
                raise AssertionError("the second reading was asked for")

        asking = case_agent.CaseState(case=C.hear(C.CaseFile(case_id="t"), "Hello, I have a question about our numbers."))
        asking.route_client = Never()
        self.assertFalse(C.readiness(asking.case).may_propose)
        case_agent.route_ahead(asking)  # neither a propose nor a start can reach the second reading this turn
        self.assertIsNone(asking.route_thread)
        self.assertLessEqual(case_agent.ROUTE_WAIT, 5)

        seen = {}

        class Second:
            def complete(self, messages, tools=None, **options):
                from dclab_rnd import experience

                seen.update(experience._IDS.get())
                return {"content": "analysis", "tool_calls": [], "usage": {}}

        state = case_agent.CaseState(case=C.hear(C.CaseFile(case_id="t"), "Why did our numbers move last quarter?"))
        state.route_client, state.go_ahead = Second(), True
        state.trace = SimpleNamespace(run_id="run-123")
        case_agent.route_ahead(state)
        state.route_thread.join(5)
        self.assertEqual(seen.get("run"), "run-123")  # the review: the thread's request had lost the run it belongs to
        self.assertFalse(state.route_thread.is_alive())
        case_agent.route_ahead(state)  # one left over by a failed turn is cleared, and the reading it made is not asked again
        self.assertIsNone(state.route_thread)

    def test_it_is_not_started_where_it_would_not_be_read(self):
        from dclab_core import case as C
        from dclab_rnd.agents import case_agent

        class Never:
            def complete(self, *args, **kwargs):
                raise AssertionError("the second reading was asked for")

        named = case_agent.CaseState(case=C.hear(C.CaseFile(case_id="t"), "Predict which customers will churn next month."))
        named.route_client = Never()
        case_agent.route_ahead(named)  # their own words name the help: they outrank the second reading
        self.assertIsNone(named.route_thread)
        chip = case_agent.CaseState(case=C.hear(C.CaseFile(case_id="t"), "Help me with this."))
        chip.route_client, chip.chip = Never(), "analysis"
        case_agent.route_ahead(chip)
        self.assertIsNone(chip.route_thread)
        with mock.patch.dict("os.environ", {"DCLAB_ROUTE_AHEAD": "0"}):
            off = case_agent.CaseState(case=C.hear(C.CaseFile(case_id="t"), "Help me with this."))
            off.route_client, off.go_ahead = Never(), True
            case_agent.route_ahead(off)
            self.assertIsNone(off.route_thread)


if __name__ == "__main__":
    unittest.main()
