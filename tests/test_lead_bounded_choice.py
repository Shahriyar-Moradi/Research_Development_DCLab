"""Packages A1 and A2 (Agentic_Plan_and_Context_Plan_2026-11-20): bounded Lead choice menu, observation-driven
next action, Advisor consultation, training-fold comparison with idempotency, and workspace switch."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dclab_core import case as C  # noqa: E402
from dclab_rnd.agents import case_agent, decision_loop, default_registry  # noqa: E402
from dclab_rnd.draft import analyze  # noqa: E402
from dclab_rnd.draft.case_home import CaseHome  # noqa: E402
from dclab_rnd.draft.store import DraftStore  # noqa: E402


class _ScriptedClient:
    def __init__(self, replies, *, cost_per_call=0.0):
        self.replies = list(replies)
        self.sent = []
        self.spent_eur = 0.0
        self.cost_per_call = cost_per_call

    def complete(self, messages, tools=None, max_tokens=None):
        self.sent.append({"messages": list(messages), "tools": tools})
        self.spent_eur += self.cost_per_call
        if not self.replies:
            return {"content": "", "tool_calls": [], "usage": {}}
        nxt = self.replies.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class _Gateway:
    def __init__(self, clients):
        self.clients = dict(clients)

    def client(self, purpose, draft_id=None, model=None):
        return self.clients.get(purpose)

    def available(self, purpose):
        return purpose in self.clients


def _delivery_frame(n=60):
    from dclab_core.replicas import delivery_time

    return delivery_time.generate(seed=11, n=n).orders


def _home_with_table(home_dir: Path, client, specialist=None, *, enable_bounded=True):
    store = DraftStore(home_dir / "drafts")
    if enable_bounded:
        decision_loop.set_settings(home_dir, enabled=True)
    draft = store.create("Predict actual_minutes for new food orders.")
    frame = _delivery_frame(60)
    asset_dir = store.data_dir(draft["id"])
    parquet_path = asset_dir / "clean.parquet"
    frame.to_parquet(parquet_path, index=False)
    analysis = analyze.analyze(frame)
    store.update(draft["id"], lambda d: d.update(
        analysis=analysis,
        assets=[{"id": "a1", "name": "delivery_time.csv", "filename": "delivery_time.csv", "clean_file": "clean.parquet", "status": "ready"}],
        active_asset="a1",
    ))
    gw = _Gateway({"case_agent": client, **({"case_specialist": specialist} if specialist is not None else {})})
    return store, draft["id"], CaseHome(store, gw)


class BoundedChoiceA1Tests(unittest.TestCase):
    def test_switch_defaults_off_and_preserves_team_settings(self):
        from dclab_rnd.orchestra import settings as team_settings

        home = Path(tempfile.mkdtemp())
        self.assertFalse(decision_loop.is_enabled(home))
        self.assertFalse(decision_loop.advisory_roles_enabled(home))
        decision_loop.set_settings(home, enabled=True, advisory_gated_roles=True)
        self.assertTrue(decision_loop.is_enabled(home))
        self.assertTrue(decision_loop.advisory_roles_enabled(home))
        # Orchestra task-planning switches stay completely intact.
        ts = team_settings.settings(home)
        self.assertEqual((ts["task_plan_preview"], ts["task_execution"], ts["admit_proposals"]), (True, True, True))

    def test_home_entry_asks_when_moment_missing_inspects_when_confirmed_and_stops_on_stop(self):
        home = Path(tempfile.mkdtemp())
        ask_reply = {"content": "", "tool_calls": [{"id": "c1", "name": "choose_action", "arguments": {
            "choice": "ask", "reason": "Prediction time is unknown and determines which columns exist.",
            "evidence_refs": ["case:goal", "table:delivery_time.csv"],
            "arguments": {"question": "When is the delivery time predicted: at order creation or after dispatch?"},
        }}]}
        inspect_reply = {"content": "", "tool_calls": [{"id": "c2", "name": "choose_action", "arguments": {
            "choice": "inspect", "reason": "Prediction moment is confirmed at order creation; inspect columns for post-order leakage.",
            "evidence_refs": ["table:delivery_time.csv"],
            "arguments": {"outcome": "actual_minutes"},
        }}]}
        stop_after_inspect = {"content": "", "tool_calls": [{"id": "c3", "name": "choose_action", "arguments": {
            "choice": "stop", "reason": "Inspection finished cleanly.",
            "arguments": {"message": "Checked leakage candidates; ready for your review."},
        }}]}
        stop_reply = {"content": "", "tool_calls": [{"id": "c4", "name": "choose_action", "arguments": {
            "choice": "stop", "reason": "User asked to stop without running tools.",
            "arguments": {"message": "Stopped as requested without running any tool."},
        }}]}
        client = _ScriptedClient([ask_reply, inspect_reply, stop_after_inspect, stop_reply])
        store, draft_id, ch = _home_with_table(home, client, enable_bounded=True)

        # Turn 1: prediction moment missing -> Lead asks once and waits for person.
        ch.reply(draft_id, "Help me predict actual_minutes for new orders.")
        d1 = store.get(draft_id)
        decisions1 = d1["case_state"]["decisions"]
        self.assertEqual(decisions1[-1]["choice"], "ask")
        self.assertEqual(decisions1[-1]["status"], "waiting_for_person")
        self.assertEqual(decisions1[-1]["committed_effects"], 1)
        self.assertIn("When is the delivery time predicted", d1["messages"][-1]["text"])

        # Turn 2: moment confirmed -> Lead runs read-only inspect, then stops cleanly.
        ch.reply(draft_id, "The prediction is made when the order is placed, before a courier is assigned.")
        d2 = store.get(draft_id)
        choices2 = [x["choice"] for x in d2["case_state"]["decisions"][-2:]]
        self.assertEqual(choices2, ["inspect", "stop"])
        self.assertEqual(d2["case_state"]["decisions"][-2]["status"], "succeeded")

        # Turn 3: user asks to stop -> Lead stops with zero tool calls and zero effects.
        ch.reply(draft_id, "Stop here and do not run anything else.")
        d3 = store.get(draft_id)
        last = d3["case_state"]["decisions"][-1]
        self.assertEqual((last["choice"], last["status"], last["committed_effects"], last["tool"]),
                         ("stop", "stopped", 0, None))

    def test_forbidden_stale_and_invalid_choices_have_no_effect_and_are_traced(self):
        reg = default_registry()
        state = case_agent.CaseState(case=C.new("case-guard"), table={"table": "t.csv", "rows": 50, "columns": [{"name": "a"}, {"name": "y"}]},
                                     frame=_delivery_frame(40))
        menu = decision_loop.build_menu(state, reg, allow_a2=False)
        self.assertNotIn("consult", [o.id for o in menu.options])
        self.assertNotIn("compare", [o.id for o in menu.options])

        # Forbidden choice not in menu -> blocked, zero effect.
        blocked = decision_loop.dispatch_choice(state, menu, {"choice": "delete_everything", "reason": "bad", "menu_revision": menu.revision}, reg)
        self.assertEqual((blocked.status, blocked.blocked_reason, blocked.committed_effects), ("blocked", "forbidden_choice", 0))

        # Stale menu_revision -> blocked, zero effect.
        stale = decision_loop.dispatch_choice(state, menu, {"choice": "ask", "reason": "old", "menu_revision": "stale_rev",
                                                            "arguments": {"question": "What?"}}, reg)
        self.assertEqual((stale.status, stale.blocked_reason, stale.committed_effects), ("blocked", "stale_menu_revision", 0))
        self.assertEqual(state.said, [])

        # Invalid reply through run_bounded_turn -> safe bounded fallback with traceable failed decision.
        bad_client = _ScriptedClient([{"content": "just rambling without json or tool call", "tool_calls": []}])
        out = decision_loop.run_bounded_turn(state, bad_client, reg, [{"role": "user", "content": "hi"}])
        self.assertEqual(out["observations"][-1]["status"], "failed")
        self.assertEqual(out["observations"][-1]["blocked_reason"], "invalid_selection")
        self.assertIn("Could not validate", out["text"])


class ObservationLoopA2Tests(unittest.TestCase):
    def test_observation_changes_next_action_revises_on_leak_and_stops_on_clean(self):
        reg = default_registry()
        frame = _delivery_frame(60)
        table_view = {"table": "delivery_time.csv", "rows": len(frame),
                      "columns": [{"name": c, "kind": "numeric"} for c in frame.columns]}
        state = case_agent.CaseState(case=C.new("case-rev"), table=table_view, frame=frame)
        state.turns.append({"role": "user", "content": "Predict actual_minutes when the order is placed."})

        # Branch 1: inspect finds delivered_at/courier_rating -> Lead revises plan to exclude post-dispatch columns.
        replies_leak = [
            {"content": "", "tool_calls": [{"id": "c1", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Check columns for post-order leakage.", "arguments": {"outcome": "actual_minutes"}}}]},
            {"content": "", "tool_calls": [{"id": "c2", "name": "choose_action", "arguments": {
                "choice": "revise", "reason": "Post-order columns delivered_at and courier_rating leak future info at order creation.",
                "arguments": {"exclude_columns": ["delivered_at", "courier_rating"],
                              "steps": ["Exclude delivered_at and courier_rating", "Use order_time chronological split"]}}}]},
            {"content": "", "tool_calls": [{"id": "c3", "name": "choose_action", "arguments": {
                "choice": "stop", "reason": "Provisional revision ready for approval."}}]},
        ]
        out = decision_loop.run_bounded_turn(state, _ScriptedClient(replies_leak), reg, [{"role": "user", "content": "start"}])
        self.assertEqual([o["choice"] for o in out["observations"]], ["inspect", "revise", "stop"])
        self.assertEqual(state.decision_revisions, 1)
        self.assertIn("delivered_at", state.plan)
        self.assertTrue(out["observations"][1]["result"]["requires_approval"])

        # Branch 2: alternative clean observation stops without revision.
        clean_state = case_agent.CaseState(case=C.new("case-clean"), table=table_view, frame=frame)
        replies_clean = [
            {"content": "", "tool_calls": [{"id": "c1", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Check columns.", "arguments": {"outcome": "actual_minutes"}}}]},
            {"content": "", "tool_calls": [{"id": "c2", "name": "choose_action", "arguments": {
                "choice": "stop", "reason": "No revision required."}}]},
        ]
        out_clean = decision_loop.run_bounded_turn(clean_state, _ScriptedClient(replies_clean), reg, [{"role": "user", "content": "start"}])
        self.assertEqual([o["choice"] for o in out_clean["observations"]], ["inspect", "stop"])
        self.assertEqual(clean_state.decision_revisions, 0)

    def test_compare_runs_on_training_folds_only_and_is_idempotent_on_retry(self):
        reg = default_registry()
        frame = _delivery_frame(80)
        table_view = {"table": "delivery_time.csv", "rows": len(frame),
                      "columns": [{"name": c, "kind": "numeric"} for c in frame.columns]}
        state = case_agent.CaseState(case=C.new("case-cmp"), table=table_view, frame=frame)
        menu = decision_loop.build_menu(state, reg, allow_a2=True)
        sel = {
            "choice": "compare",
            "reason": "Compare distance-only against full pre-order features on training folds.",
            "menu_revision": menu.revision,
            "idempotency_key": "exp-delivery-001",
            "arguments": {"outcome": "actual_minutes", "features_a": ["distance_km"],
                          "features_b": ["distance_km", "couriers_online", "basket_items"]},
        }
        first = decision_loop.dispatch_choice(state, menu, sel, reg)
        self.assertEqual(first.status, "succeeded")
        self.assertFalse(first.result["holdout_touched"])
        self.assertEqual(first.result["scope"], "training_folds_only")
        self.assertEqual(state.decision_experiments, 1)

        # Repeated delivery with the same idempotency_key replays without running a second experiment.
        menu2 = decision_loop.build_menu(state, reg, allow_a2=True)
        sel_retry = {**sel, "menu_revision": menu2.revision}
        second = decision_loop.dispatch_choice(state, menu2, sel_retry, reg)
        self.assertTrue(second.replayed)
        self.assertEqual(second.committed_effects, 0)
        self.assertEqual(state.decision_experiments, 1)
        self.assertEqual(second.result, first.result)

    def test_consult_charges_shared_budget_and_handles_approval_budget_and_repeat_stops(self):
        reg = default_registry()
        frame = _delivery_frame(60)
        table_view = {
            "table": "delivery_time.csv",
            "rows": len(frame),
            "columns": [
                {"name": c, "kind": "numeric", "noticed": (["outcome-like name"] if c == "actual_minutes" else [])}
                for c in frame.columns
            ],
        }
        # Budget exhaustion stops cleanly after paid consult.
        spec_client = _ScriptedClient([{"content": json.dumps({"findings": []}), "tool_calls": []}], cost_per_call=0.6)
        main_client = _ScriptedClient([
            {"content": "", "tool_calls": [{"id": "c1", "name": "choose_action", "arguments": {
                "choice": "consult", "reason": "Ask leakage auditor about courier_rating.",
                "arguments": {"specialist": "leakage_auditor", "question": "Is courier_rating post-order?"}}}]},
            {"content": "", "tool_calls": [{"id": "c2", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Should never run after budget hit."}}]},
        ], cost_per_call=0.05)
        state = case_agent.CaseState(case=C.new("case-bud"), table=table_view, frame=frame,
                                     specialist_client=spec_client, cap_eur=0.5)
        out = decision_loop.run_bounded_turn(state, main_client, reg, [{"role": "user", "content": "check"}])
        self.assertEqual(out["observations"][0]["choice"], "consult")
        self.assertTrue(out["observations"][0]["result"]["provisional"])
        self.assertFalse(out["observations"][0]["result"]["carries_approval"])
        self.assertTrue(out["observations"][0]["result"]["insufficient_evidence"])
        self.assertEqual(out["observations"][-1]["blocked_reason"], "budget_exhausted")
        self.assertTrue(state.waiting_on_budget)
        self.assertEqual(len(main_client.sent), 1)

        # Pending approval omits non-stop actions and blocks any attempted action.
        pending_state = case_agent.CaseState(case=C.new("case-app"), table=table_view, frame=frame, approvals_pending=True)
        pending_menu = decision_loop.build_menu(pending_state, reg)
        self.assertEqual([o.id for o in pending_menu.options], ["stop"])

        # Repeated identical choice with unchanged evidence stops cleanly without a second tool execution.
        rep_state = case_agent.CaseState(case=C.new("case-rep"), table=table_view, frame=frame)
        rep_client = _ScriptedClient([
            {"content": "", "tool_calls": [{"id": "r1", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Inspect once.", "arguments": {"outcome": "actual_minutes"}}}]},
            {"content": "", "tool_calls": [{"id": "r2", "name": "choose_action", "arguments": {
                "choice": "inspect", "reason": "Inspect identically again.", "arguments": {"outcome": "actual_minutes"}}}]},
        ])
        rep_out = decision_loop.run_bounded_turn(rep_state, rep_client, reg, [{"role": "user", "content": "hi"}])
        self.assertEqual([o["status"] for o in rep_out["observations"]], ["succeeded", "stopped"])
        self.assertEqual(rep_out["observations"][-1]["blocked_reason"], "repeated_unchanged_choice")
        self.assertEqual(rep_out["observations"][-1]["committed_effects"], 0)

        # Unavailable model yields explicit typed outcome with zero committed effects.
        no_model_state = case_agent.CaseState(case=C.new("case-nm"), table=table_view, frame=frame)
        nm_out = decision_loop.run_bounded_turn(no_model_state, None, reg, [{"role": "user", "content": "hi"}])
        self.assertEqual(nm_out["observations"][0]["blocked_reason"], "model_unavailable")
        self.assertEqual(nm_out["observations"][0]["committed_effects"], 0)


if __name__ == "__main__":
    unittest.main()
