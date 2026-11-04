"""Handwritten TP-01C metadata fixtures, not a workflow adapter or an execution run."""
import copy

from dclab_core.solution_workflow import solution_workflow
from dclab_core.task_graph import TaskGraph
from dclab_core.task_execution_plan import TaskExecutionPlan

D = "d" * 64
PROVENANCE = {"method": "deterministic", "reference_id": "fixture", "producer_version": "v1"}


def case():
    from dclab_core.task_fingerprints import capability_fingerprint, graph_fingerprint, metadata_fingerprint, workflow_fingerprint
    workflow = solution_workflow({"family": "classification"})
    caps = {
        "run_stage": {"name": "run_stage", "scope": "project", "effect": "write", "move": "run_stage", "allowed": True,
                      "replay_safe": False, "parameters": {"type": "object", "properties": {"project_id": {"type": "string"}, "stage": {"type": "string", "enum": ["data", "leakage", "features", "models", "final"]}}, "required": ["project_id", "stage"]}},
        "export_notebook": {"name": "export_notebook", "scope": "project", "effect": "write", "move": "capture", "allowed": True,
                            "replay_safe": False, "parameters": {"type": "object", "properties": {"project_id": {"type": "string"}}, "required": ["project_id"]}},
    }
    specs = [
        ("data", "action", ["lock_holdout", "train_only_analysis", "class_balance"], [], ["table"], ["data_record"]),
        ("leakage", "action", ["leakage_audit"], ["data"], ["data_record"], ["leakage_record"]),
        ("features", "action", [], ["leakage"], ["leakage_record"], ["features_record"]),
        ("models", "action", ["choose_on_validation"], ["features"], ["features_record"], ["models_record"]),
        ("holdout", "gate", ["holdout_gate"], ["models"], ["models_record"], ["holdout_permission"]),
        ("final", "action", ["threshold", "final_once"], ["holdout"], ["models_record", "holdout_permission"], ["final_record"]),
        ("report", "action", ["report_limits"], ["final"], ["final_record"], ["report_file"]),
    ]
    nodes = [{"id": "root", "kind": "group", "title": "Churn", "objective": "Understand churn before retention action"}]
    edges = []
    for ident, kind, steps, dependencies, needs, produces in specs:
        check = {"id": ident + "_proof", "kind": "approval" if kind == "gate" else "lineage", "description": "Current owner evidence exists", "evidence_refs": produces}
        node = {"id": ident, "parent_id": "root", "title": ident, "objective": "Preserve current approved inputs", "kind": kind,
                "source_step_ids": steps, "dependencies": dependencies, "needs": needs, "produces": produces, "acceptance_checks": [check]}
        if kind == "action":
            name = "export_notebook" if ident == "report" else "run_stage"
            node.update(scope="project", capability={"name": name, "scope": "project", "schema_fingerprint": capability_fingerprint(caps[name])})
        nodes.append(node)
        edges.append({"source": "root", "target": ident, "kind": "grouping"})
        edges.extend({"source": dep, "target": ident, "kind": "dependency"} for dep in dependencies)
    data = {"schema_version": "dclab_core.task_graph.v1", "graph_id": "churn", "revision": 1, "objective": "Understand churn before retention action",
            "source_refs": [{"kind": "solution", "id": "solution", "fingerprint": D},
                            {"kind": "solution_workflow", "id": "workflow", "fingerprint": workflow_fingerprint(workflow)},
                            {"kind": "dataset", "id": "table", "fingerprint": D},
                            {"kind": "approval", "id": "source_approval", "fingerprint": D}],
            "solution_fingerprint": D, "workflow_fingerprint": workflow_fingerprint(workflow), "nodes": nodes, "edges": edges, "provenance": PROVENANCE}
    data["fingerprint"] = graph_fingerprint(data)
    graph = TaskGraph.read(data)
    artifacts = {"table": {"fingerprint": D, "value_type": "artifact"},
                 "project": {"fingerprint": metadata_fingerprint("project1"), "value_type": "string", "value": "project1"}}
    for stage in ("data", "leakage", "features", "models", "final"):
        artifacts[stage + "_stage"] = {"fingerprint": metadata_fingerprint(stage), "value_type": "string", "value": stage}
    approval = {"project_id": "project1", "workflow_fingerprint": graph.workflow_fingerprint,
                "sources": [ref.model_dump(mode="json") for ref in graph.source_refs if ref.kind != "approval"], "approved": True, "reference_id": "source_approval", "fingerprint": D, "solution_fingerprint": D,
                "inputs": {"table": D, "project": artifacts["project"]["fingerprint"]}}
    state = {"graph_id": graph.graph_id, "graph_revision": graph.revision, "graph_fingerprint": graph.fingerprint,
             "project_id": "project1", "solution_fingerprint": D, "workflow_fingerprint": graph.workflow_fingerprint,
             "artifacts": artifacts, "approval": approval, "tasks": {}, "holdout_uses": 0, "capabilities": caps}
    return graph, workflow, caps, state


def reseal(graph, **updates):
    from dclab_core.task_fingerprints import graph_fingerprint
    data = graph.model_dump(mode="json"); data.update(updates); data["fingerprint"] = graph_fingerprint(data)
    return TaskGraph.read(data)


def state_for(graph, state):
    state = copy.deepcopy(state)
    state.update(graph_id=graph.graph_id, graph_revision=graph.revision, graph_fingerprint=graph.fingerprint)
    return state


def verified(graph, state, task_id):
    state = copy.deepcopy(state)
    node = next(n for n in graph.nodes if n.id == task_id)
    for artifact in node.produces:
        state["artifacts"][artifact] = {"fingerprint": D, "value_type": "artifact", "producer_task": task_id}
    receipt = {"status": "verified", "verification_ref": task_id + "_verification", "project_id": state["project_id"],
               "graph_revision": graph.revision, "graph_fingerprint": graph.fingerprint,
               "inputs": {a: state["artifacts"][a]["fingerprint"] for a in node.needs},
               "outputs": {a: state["artifacts"][a]["fingerprint"] for a in node.produces}}
    if node.kind == "gate":
        state["holdout_approval"] = {**state["approval"], "reference_id": "holdout_approval", "used": False,
            "inputs": {**state["approval"]["inputs"], **{ident: state["artifacts"][ident]["fingerprint"] for ident in node.needs}}}
        receipt["approval_ref"] = {"id": "holdout_approval", "fingerprint": D}
    state["tasks"][task_id] = receipt
    if task_id == "final":
        state["holdout_uses"] = 1; state["holdout_approval"]["used"] = True
    return state


def execution_plan(graph, state, task_id="data"):
    from dclab_core.task_fingerprints import plan_fingerprint
    node = next(n for n in graph.nodes if n.id == task_id)
    names = ["project"] + ([task_id + "_stage"] if node.capability.name == "run_stage" else [])
    refs = list(dict.fromkeys([*node.needs, *names]))
    arguments = [{"name": "project_id", "value": {"kind": "input", "ref_id": "project", "value_type": "string"}}]
    if node.capability.name == "run_stage": arguments.append({"name": "stage", "value": {"kind": "input", "ref_id": task_id + "_stage", "value_type": "string"}})
    data = {"schema_version": "dclab_core.task_execution_plan.v1", "graph_id": graph.graph_id, "graph_revision": graph.revision,
            "graph_fingerprint": graph.fingerprint, "task_id": task_id, "plan_id": task_id + "_plan", "revision": 1,
            "expected_inputs": [{"id": ident, "fingerprint": state["artifacts"][ident]["fingerprint"], "value_type": state["artifacts"][ident]["value_type"]} for ident in refs],
            "selected_capabilities": [node.capability.model_dump(mode="json")],
            "actions": [{"id": "call", "capability_name": node.capability.name, "scope": node.scope, "arguments": arguments}],
            "declared_outputs": [{"id": ident, "action_id": "call", "value_type": "artifact"} for ident in node.produces],
            "acceptance_checks": [c.model_dump(mode="json") for c in node.acceptance_checks],
            "budget": {"max_tool_calls": 1, "max_seconds": 120}, "provenance": PROVENANCE, "status": "ready"}
    data["fingerprint"] = plan_fingerprint(data)
    return TaskExecutionPlan.read(data)


def programme_case():
    from dclab_core.solution_map import SolutionMap, SplitDesign, Block, HoldoutDesign, SolutionStep
    from dclab_core.task_fingerprints import capability_fingerprint, workflow_fingerprint
    graph, wf, caps, state = case()
    split = SplitDesign(strategy="stratified")
    source = SolutionMap(map="a"*16, blocks=[Block(id=f"blk{i}",part=f"part{i}",sub_problem=f"sp{i}",title="Churn component",family="classification",
        approach="classification",workflow=wf,split=split,final="deferred",defers=["final_once"]) for i in (1,2)],
        steps=[SolutionStep(kind="fusion",inputs=["sp1","sp2"],output="combined"),SolutionStep(kind="score",inputs=["combined"],output="holdout_score",reads_holdout=True)],
        holdout=HoldoutDesign(split=split,read_by="composition",step="holdout_score"))
    caps["run_programme"] = {"name":"run_programme","scope":"project","effect":"write","move":"run_programme","allowed":True,"replay_safe":False,
        "parameters":{"type":"object","properties":{"project_id":{"type":"string"}},"required":["project_id"]}}
    data = graph.model_dump(mode="json")
    data["source_refs"] = [ref for ref in data["source_refs"] if ref["kind"]!="solution_workflow"] + [{"kind":"solution_map","id":"map","fingerprint":source.fingerprint}]
    data["workflow_fingerprint"] = workflow_fingerprint(source)
    data["nodes"] = [data["nodes"][0],dict(id="holdout",parent_id="root",kind="gate",title="Human gate",objective="Approve the shared final read",
        produces=["holdout_permission"],acceptance_checks=[dict(id="permission",kind="approval",description="Current owner permission exists",evidence_refs=["holdout_permission"])]),
        dict(id="programme",parent_id="root",kind="action",title="Atomic programme",objective="One shared split and final score",dependencies=["holdout"],needs=["table","holdout_permission"],produces=["programme_record"],scope="project",
        capability=dict(name="run_programme",scope="project",schema_fingerprint=capability_fingerprint(caps["run_programme"])),
        acceptance_checks=[dict(id="programme_proof",kind="lineage",description="Existing programme record has one holdout read",evidence_refs=["programme_record"])])]
    data["edges"] = [dict(source="root",target=ident,kind="grouping") for ident in ("holdout","programme")] + [dict(source="holdout",target="programme",kind="dependency")]
    graph = reseal(graph, **{k:v for k,v in data.items() if k!="fingerprint"})
    state=state_for(graph,state);state["workflow_fingerprint"]=graph.workflow_fingerprint;state["capabilities"]=caps
    state["approval"]["workflow_fingerprint"]=graph.workflow_fingerprint
    state["approval"]["sources"]=[ref.model_dump(mode="json") for ref in graph.source_refs if ref.kind!="approval"]
    return graph,source,caps,state
