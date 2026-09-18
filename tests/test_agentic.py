"""Agent runtime tests: .venv-agent/bin/python -m pytest tests/test_agentic.py."""
import asyncio
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

if importlib.util.find_spec("nooa") is None:
    raise unittest.SkipTest("NOOA tests run in the isolated .venv-agent environment")

import pytest
from fastapi.testclient import TestClient
from nooa.unifiedllm.fake import FakeLLMClient
from nooa.unifiedllm import LLMResponse
from langgraph.checkpoint.memory import InMemorySaver

from dclab_rnd.agentic.agents import ResearchPlanner, ask
from dclab_rnd.agentic.schemas import RunRequest, Experiment, Agenda, Feature
from dclab_rnd.agentic.store import Store
from dclab_rnd.agentic.engine import build_graph, check_citations, worker
from dclab_rnd.agentic.server import create_app

def experiment(model="logistic_regression"):
    return Experiment(dataset="bank_marketing", title="Transparent baseline", hypothesis="A transparent baseline establishes model-family comparison evidence.", model=model, parameters={}, features=[], drop_columns=[], stress_columns=["balance"], evidence_ids=[], reference_evidence_id=None, expected_learning="A useful development baseline").model_dump()

def test_nooa_real_typed_strategy():
    value = Agenda(goal_interpretation="Learn with evidence", research_questions=["Which baseline?"], sequence=["profile", "fit"], success_criteria=["paired evidence"], limitations=["development CV"])
    llm = FakeLLMClient([LLMResponse(raw_response=None, content=value.model_dump_json(), tool_calls=[], finish_reason="stop", assistant_message={"role":"assistant", "content":value.model_dump_json()})])
    result = asyncio.run(ask(ResearchPlanner, "plan", llm, {"goal":"Learn with evidence"}))
    assert result == value.model_dump()
    assert llm.call_count == 1
    assert not llm.last_tools  # Predict strategy never requests code execution.

def test_schema_and_citations():
    with pytest.raises(ValueError): RunRequest(datasets=["../../secret"])
    with pytest.raises(ValueError): RunRequest(max_experiments=51)
    with pytest.raises(ValueError): Feature(name="fe_x", operation="ratio", inputs=["one"], rationale="test")
    with pytest.raises(ValueError): Experiment(**{**experiment(), "shell": "anything"})
    with pytest.raises(ValueError): check_citations(["invented"], [])
    with pytest.raises(ValueError): check_citations([], [], required=True)

def test_graph_adapts_after_critique_and_persists():
    with tempfile.TemporaryDirectory() as directory:
        store=Store(Path(directory)); request=RunRequest(max_experiments=2).model_dump(); store.create("test",request)
        calls=[]
        async def agent(cls, method, client, context):
            calls.append(method)
            if method=="plan": return {"goal_interpretation":"test"}
            if method=="review": return {"leakage_risks":["duration"]}
            if method=="propose":
                if context["attempted"]:
                    assert context["critique"][0]["next_question"] == "Compare trees"
                    assert context["trials"][0]["metrics"]["roc_auc"] == .7
                value=experiment("extra_trees" if context["attempted"] else "logistic_regression")
                if context["attempted"]:
                    value["evidence_ids"]=["test:trial-001"]
                    value["reference_evidence_id"]="test:trial-001"
                return value
            if method=="critique": return {"evidence_ids":[t["id"] for t in context["trials"]], "next_question":"Compare trees", "continue_research":True}
            return {"summary":"Provisional evidence", "lessons":[{"claim":"A narrow result", "evidence_ids":["test:trial-001"], "scope":"test", "counterevidence":"none measured", "follow_up":"confirm"}], "theoretical_principles":["Use paired evidence"], "workflow_blocks":["profile → fit → critique"], "unanswered_questions":[]}
        async def tool(req):
            if req["action"]=="profile": return [{"dataset":"bank_marketing"}]
            score=.8 if req["plan"]["model"]=="extra_trees" else .7
            return {"dataset":"bank_marketing", "plan":req["plan"], "metrics":{"roc_auc":score},"fold_standard_deviation":{},"input_sensitivity":[],"stress_tests":[],"output_calibration":[],"wall_seconds":1,"limitations":["test"],"split_hash":"same", "data_hashes":{}, "folds":[{"roc_auc":score}]*3}
        async def check():
            saver=InMemorySaver(); graph=build_graph(store,"test",None,saver,tool,agent)
            cfg={"configurable":{"thread_id":"test"}}
            result=await graph.ainvoke({"config":request,"trials":[],"critiques":[]},cfg)
            assert len(result["trials"])==2
            assert result["trials"][1]["paired_comparison"]["mean_deltas"]["roc_auc"]==pytest.approx(.1)
            assert (await graph.aget_state(cfg)).next==()
        asyncio.run(check())
        assert calls==["plan","review","propose","critique","propose","critique","synthesize"]
        assert len(store.knowledge())==1
        assert store.export("test")["training_ready"] is False

def test_local_api_security_and_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as directory, TestClient(create_app(Path(directory))) as client:
        assert client.get("/").status_code==200
        assert client.get("/api/config").json()["api_key_configured"] is False
        assert client.post("/api/runs",json={}).status_code==403
        token=client.get("/api/config").json()["csrf"]
        assert client.post("/api/runs",json={},headers={"X-DCLab-Token":token}).status_code==409
        assert client.get("/api/config",headers={"Host":"evil.example"}).status_code==400
        assert client.get("/api/runs/missing").status_code==404
        assert client.get("/").headers["Content-Security-Policy"].startswith("default-src 'self'")

def test_real_worker_profile_and_trial():
    with tempfile.TemporaryDirectory() as directory:
        profile=asyncio.run(worker({"action":"profile","datasets":["bank_marketing"],"max_rows":300}))
        assert profile[0]["rows"]==300
        result=asyncio.run(worker({"action":"experiment","plan":experiment(),"max_rows":300,"repeats":1,"output":directory}))
        assert "duration" not in result["retained_columns"]
        assert len(result["folds"])==3
        assert result["production_approved"] is False
        assert len(result["stress_tests"])==3
        assert len(result["input_sensitivity"])==15
        assert len(result["repeat_summary"])==1
        assert set(result["repeat_summary"][0])=={"repeat","roc_auc","average_precision","log_loss","brier"}
        assert Path(directory,"recipe.json").exists()
