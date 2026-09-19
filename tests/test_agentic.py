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
    with pytest.raises(ValueError): RunRequest(project="hyperack", datasets=["telco_churn"])

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

def test_archive_materializes_process_files_and_index():
    from dclab_rnd.agentic.archive import archive_run, write_studio_index, run_label

    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        store = Store(home)
        request = RunRequest(max_experiments=1).model_dump()
        store.create("runabc123", request)
        store.event("runabc123", "profiles", [{"dataset": "bank_marketing", "rows": 100}])
        store.event(
            "runabc123",
            "agenda",
            {
                "goal_interpretation": "test",
                "research_questions": ["q"],
                "sequence": ["s"],
                "success_criteria": ["c"],
                "limitations": ["l"],
            },
        )
        store.update("runabc123", status="failed", phase="Research planner", error="rate limit")
        store.event("runabc123", "failure", {"type": "RateLimitError"})
        manifest = archive_run(store, "runabc123", reset_snapshots=True)
        root = home / "runabc123"
        assert (root / "run_config.json").exists()
        assert (root / "LABEL.txt").exists()
        assert (root / "process" / "01_profiles.json").exists()
        assert (root / "process" / "02_agenda.json").exists()
        assert (root / "process" / "99_failures.jsonl").exists()
        assert (root / "events").is_dir()
        assert (root / "trajectory.json").exists()
        assert (root / "manifest.json").exists()
        assert (root / "README.md").exists()
        assert manifest["label"] == run_label(store.get("runabc123"))
        write_studio_index(store)
        assert (home / "STUDIO_ARCHIVE_INDEX.json").exists()
        assert (home / "catalog" / "by_label" / manifest["label"]).exists()
        assert (home / "README.md").exists()
        assert (root / "clean" / "run_card.json").exists()
        assert (home / "clean_exports" / "sft_chat.jsonl").exists()
        assert (home / "clean_exports" / "MANIFEST.json").exists()

def test_clean_export_builds_compact_trials_and_sft():
    from dclab_rnd.agentic.clean_export import write_run_clean, write_studio_clean_exports, compact_trial_from_disk

    with tempfile.TemporaryDirectory() as directory:
        home = Path(directory)
        store = Store(home)
        store.create("runclean1", RunRequest(max_experiments=1, datasets=["bank_marketing"]).model_dump())
        store.event(
            "runclean1",
            "agenda",
            {
                "goal_interpretation": "learn",
                "research_questions": ["baseline?"],
                "sequence": ["profile", "fit"],
                "success_criteria": ["paired"],
                "limitations": ["dev cv"],
            },
        )
        store.event(
            "runclean1",
            "proposal",
            experiment("logistic_regression"),
        )
        store.event(
            "runclean1",
            "critique",
            {
                "observation": "AUC 0.7",
                "interpretation": "useful baseline",
                "limitations": ["dev cv"],
                "next_question": "try trees",
                "evidence_ids": ["runclean1:trial-001"],
                "continue_research": False,
            },
        )
        trial = home / "runclean1" / "trial-001"
        trial.mkdir(parents=True)
        plan = experiment("logistic_regression")
        (trial / "recipe.json").write_text(
            json.dumps({"schema_version": 1, "plan": plan, "max_rows": 300, "repeats": 1}),
            encoding="utf-8",
        )
        (trial / "result.json").write_text(
            json.dumps(
                {
                    "dataset": "bank_marketing",
                    "plan": plan,
                    "rows": 300,
                    "feature_count": 5,
                    "metrics": {"roc_auc": 0.71, "average_precision": 0.4, "log_loss": 0.5, "brier": 0.2},
                    "fold_standard_deviation": {"roc_auc": 0.01},
                    "input_sensitivity": [{"column": "balance", "auc_drop": 0.02}],
                    "stress_tests": [{"columns": ["balance"], "roc_auc": 0.68, "auc_drop": 0.03}],
                    "limitations": ["test"],
                    "production_approved": False,
                    "wall_seconds": 1.2,
                }
            ),
            encoding="utf-8",
        )
        (trial / "oof_predictions.jsonl").write_text(
            '{"row_id":0,"repeat":0,"fold":0,"target":0,"probability":0.1}\n',
            encoding="utf-8",
        )
        write_run_clean(store, "runclean1")
        card = json.loads((home / "runclean1" / "clean" / "run_card.json").read_text(encoding="utf-8"))
        assert card["trial_count"] == 1
        assert card["best_cv_auc"] == 0.71
        trial_line = (home / "runclean1" / "clean" / "trials.jsonl").read_text(encoding="utf-8")
        assert "roc_auc" in trial_line
        assert "probability" not in trial_line  # OOF not embedded in clean trials
        compact = compact_trial_from_disk("runclean1", trial)
        assert compact["metrics"]["roc_auc"] == 0.71
        assert "oof_predictions" not in compact
        export_root = write_studio_clean_exports(store)
        sft_lines = (export_root / "sft_chat.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(sft_lines) >= 3
        tasks = {json.loads(line)["task"] for line in sft_lines}
        assert "plan_agenda" in tasks
        assert "propose_experiment" in tasks
        assert "critique_experiment" in tasks
        assert (export_root / "leaderboard.csv").exists()
        # Raw trial artifacts untouched.
        assert (trial / "oof_predictions.jsonl").exists()
        assert (trial / "result.json").exists()

def test_local_api_security_and_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with tempfile.TemporaryDirectory() as directory, TestClient(create_app(Path(directory))) as client:
        assert client.get("/").status_code==200
        assert client.get("/api/config").json()["api_key_configured"] is False
        cfg=client.get("/api/config").json()
        assert {p["key"] for p in cfg["projects"]}=={"general","hyperack","telco_churn"}
        assert cfg["default_model"]=="gpt-5.6-terra"
        assert client.post("/api/runs",json={}).status_code==403
        token=client.get("/api/config").json()["csrf"]
        assert client.post("/api/runs",json={},headers={"X-DCLab-Token":token}).status_code==409
        assert client.get("/api/config",headers={"Host":"evil.example"}).status_code==400
        assert client.get("/api/runs/missing").status_code==404
        assert client.get("/guide").status_code==200
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
