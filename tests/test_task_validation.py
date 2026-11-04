"""TP-01C: pure planning safety, stale-input refusal and canonical identities."""
import copy
import json
import subprocess
import sys
import unittest

from dclab_core.task_fingerprints import graph_fingerprint, plan_fingerprint, workflow_fingerprint, capability_fingerprint
from dclab_core.task_validation import validate_graph, validate_execution_plan, ready_tasks
from dclab_core.task_execution_plan import TaskExecutionPlan
from tests.task_validation_cases import D, case, reseal, state_for, verified, execution_plan


class GraphValidationTests(unittest.TestCase):
    def setUp(self): self.graph, self.workflow, self.caps, self.state = case()

    def issues(self, graph): return validate_graph(graph, self.workflow, known_capabilities=self.caps)

    def test_valid_binary_graph_and_sequential_deterministic_readiness(self):
        self.assertEqual(self.issues(self.graph), [])
        state = self.state
        for ident in ("data", "leakage", "features", "models"):
            self.assertEqual(ready_tasks(self.graph, state, source_workflow=self.workflow), [ident])
            state = verified(self.graph, state, ident)
        self.assertEqual(ready_tasks(self.graph, state, source_workflow=self.workflow), [])
        state = verified(self.graph, state, "holdout")
        self.assertEqual(ready_tasks(self.graph, state, source_workflow=self.workflow), ["final"])
        state = verified(self.graph, state, "final")
        self.assertEqual(ready_tasks(self.graph, state, source_workflow=self.workflow), ["report"])
        state = verified(self.graph, state, "report")
        self.assertEqual(ready_tasks(self.graph, state, source_workflow=self.workflow), [])

    def test_cycles_missing_parents_and_dangling_dependencies(self):
        for length in (2, 3):
            data = self.graph.model_dump(mode="json")
            ids = [n["id"] for n in data["nodes"][1:1+length]]
            for index, ident in enumerate(ids): next(n for n in data["nodes"] if n["id"] == ident)["dependencies"] = [ids[(index+1)%length]]
            data["edges"] = [e for e in data["edges"] if e["kind"] == "grouping"] + [dict(source=dep,target=n["id"],kind="dependency") for n in data["nodes"] for dep in n["dependencies"]]
            self.assertTrue(any("cycle" in p for p in self.issues(reseal(self.graph, nodes=data["nodes"],edges=data["edges"]))))
        for field, value, word in (("parent_id", "absent", "parent"), ("dependencies", ["absent"], "dependency"), ("dependencies", ["data"], "self")):
            nodes = self.graph.model_dump(mode="json")["nodes"];nodes[1][field] = value
            self.assertTrue(any(word in p for p in self.issues(reseal(self.graph,nodes=nodes))))

    def test_parent_cycle_and_group_dependencies_are_rejected(self):
        data = self.graph.model_dump(mode="json");data["nodes"][0]["parent_id"] = "other"
        data["nodes"].append(dict(id="other",parent_id="root",kind="group",title="Other",objective="Organize"))
        data["edges"] += [dict(source="other",target="root",kind="grouping"),dict(source="root",target="other",kind="grouping")]
        self.assertTrue(any("parent cycle" in p for p in self.issues(reseal(self.graph,nodes=data["nodes"],edges=data["edges"]))))
        nodes=self.graph.model_dump(mode="json")["nodes"];nodes[1]["dependencies"]=["root"]
        self.assertTrue(any("group" in p for p in self.issues(reseal(self.graph,nodes=nodes))))

    def test_group_cannot_claim_effects_or_source_safety_coverage(self):
        nodes=self.graph.model_dump(mode="json")["nodes"];nodes[0]["produces"]=["fake"]
        self.assertTrue(any("group" in p for p in self.issues(reseal(self.graph,nodes=nodes))))
        nodes=self.graph.model_dump(mode="json")["nodes"];nodes[0]["source_step_ids"]=["leakage_audit"];nodes[2]["source_step_ids"]=[]
        self.assertTrue(any("leakage_audit" in p for p in self.issues(reseal(self.graph,nodes=nodes))))

    def test_missing_leakage_repeated_final_and_weakened_gate_are_rejected(self):
        for mutate, word in ((lambda n:n[2].update(source_step_ids=[]),"leakage_audit"),
                             (lambda n:n[3].update(source_step_ids=["final_once"]),"final_once"),
                             (lambda n:n[5].update(kind="verification"),"human gate")):
            nodes=self.graph.model_dump(mode="json")["nodes"]; mutate(nodes)
            self.assertTrue(any(word in p for p in self.issues(reseal(self.graph,nodes=nodes))))

    def test_omitted_dependency_cannot_bypass_gate_or_leakage(self):
        for target, word in (("final","holdout_gate"),("models","leakage_audit"),("report","final_once")):
            data=self.graph.model_dump(mode="json");node=next(n for n in data["nodes"] if n["id"]==target);node["dependencies"]=[]
            data["edges"]=[e for e in data["edges"] if not(e["kind"]=="dependency" and e["target"]==target)]
            self.assertTrue(any(word in p for p in self.issues(reseal(self.graph,nodes=data["nodes"],edges=data["edges"]))))

    def test_edge_representation_artifact_producer_and_dependency_must_agree(self):
        self.assertTrue(any("edge" in p for p in self.issues(reseal(self.graph,edges=[]))))
        nodes=self.graph.model_dump(mode="json")["nodes"];nodes[2]["needs"]=["unmade"]
        self.assertTrue(any("artifact" in p for p in self.issues(reseal(self.graph,nodes=nodes))))
        nodes=self.graph.model_dump(mode="json")["nodes"];nodes[3]["produces"]=["data_record"]
        self.assertTrue(any("producer" in p for p in self.issues(reseal(self.graph,nodes=nodes))))

    def test_source_mutation_missing_source_and_stale_hash_are_refused_without_rewrite(self):
        self.assertTrue(validate_graph(self.graph))
        wf=self.workflow.model_copy(update={"steps":[s for s in self.workflow.steps if s.id!="leakage_audit"]},deep=True)
        self.assertTrue(validate_graph(self.graph,wf))
        stale=self.graph.model_copy(update={"objective":"Another objective"})
        before=stale.model_dump_json()
        self.assertTrue(any("fingerprint" in p for p in self.issues(stale)))
        self.assertEqual(stale.model_dump_json(),before)

    def test_changed_revision_input_and_unverified_claim_are_distinct(self):
        state=copy.deepcopy(self.state);state["graph_revision"]=2
        with self.assertRaisesRegex(ValueError,"revision"): ready_tasks(self.graph,state,source_workflow=self.workflow)
        state=copy.deepcopy(self.state);state["artifacts"]["table"]["fingerprint"]="e"*64
        with self.assertRaisesRegex(ValueError,"input artifact"): ready_tasks(self.graph,state,source_workflow=self.workflow)
        for status in ("completed","verified","unknown"):
            state=copy.deepcopy(self.state);state["tasks"]["data"]={"status":status}
            with self.assertRaises(ValueError): ready_tasks(self.graph,state,source_workflow=self.workflow)

    def test_missing_approval_and_inflight_or_changed_evidence_do_not_advance(self):
        state=copy.deepcopy(self.state);state["approval"]["approved"]=False
        self.assertEqual(ready_tasks(self.graph,state,source_workflow=self.workflow),[])
        state=copy.deepcopy(self.state);state["tasks"]["data"]={"status":"running"}
        self.assertEqual(ready_tasks(self.graph,state,source_workflow=self.workflow),[])
        state=verified(self.graph,self.state,"data");state["artifacts"]["data_record"]["fingerprint"]="f"*64
        with self.assertRaisesRegex(ValueError,"output artifact"): ready_tasks(self.graph,state,source_workflow=self.workflow)


class FingerprintTests(unittest.TestCase):
    def test_benign_reordering_and_serialization_are_stable(self):
        graph,wf,caps,state=case(); data=graph.model_dump(mode="json")
        data["nodes"].reverse();data["edges"].reverse();data["source_refs"].reverse()
        for node in data["nodes"]:
            for key in ("source_step_ids","needs","produces","dependencies","acceptance_checks"): node[key].reverse()
        self.assertEqual(graph_fingerprint(data),graph.fingerprint)
        self.assertEqual(graph_fingerprint(json.loads(graph.model_dump_json())),graph.fingerprint)
        plan=execution_plan(graph,state);data=plan.model_dump(mode="json");data["expected_inputs"].reverse();data["actions"][0]["arguments"].reverse()
        self.assertEqual(plan_fingerprint(data),plan.fingerprint)
        code="import json,sys; from dclab_core.task_fingerprints import graph_fingerprint; print(graph_fingerprint(json.load(sys.stdin)))"
        result=subprocess.run([sys.executable,"-c",code],input=graph.model_dump_json(),capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),graph.fingerprint)

    def test_semantic_change_changes_hash_provenance_and_status_do_not(self):
        graph,wf,caps,state=case()
        self.assertNotEqual(graph_fingerprint(reseal(graph,revision=2)),graph.fingerprint)
        refs=graph.model_dump(mode="json")["source_refs"];refs[2]["fingerprint"]="e"*64
        self.assertNotEqual(graph_fingerprint(reseal(graph,source_refs=refs)),graph.fingerprint)
        data=graph.model_dump(mode="json");data["provenance"]["reference_id"]="another"
        self.assertEqual(graph_fingerprint(data),graph.fingerprint)
        plan=execution_plan(graph,state);data=plan.model_dump(mode="json");data["status"]="draft"
        self.assertEqual(plan_fingerprint(data),plan.fingerprint)
        data["budget"]["max_seconds"]=121
        self.assertNotEqual(plan_fingerprint(data),plan.fingerprint)
        changed=copy.deepcopy(caps["run_stage"]);changed["move"]="other"
        self.assertNotEqual(capability_fingerprint(changed),capability_fingerprint(caps["run_stage"]))


class ExecutionPlanValidationTests(unittest.TestCase):
    def setUp(self): self.graph,self.wf,self.caps,self.state=case();self.plan=execution_plan(self.graph,self.state)
    def issues(self,plan=None,graph=None,caps=None,state=None):
        return validate_execution_plan(plan or self.plan,graph or self.graph,self.caps if caps is None else caps,source_workflow=self.wf,observed_state=self.state if state is None else state)
    def test_valid_binary_plan_and_pure_no_mutation(self):
        before=copy.deepcopy(self.state);self.assertEqual(self.issues(),[]);self.assertEqual(before,self.state)
    def test_cross_scope_missing_capability_permission_and_stale_schema_are_refused(self):
        for mutate in (lambda c:c.clear(),lambda c:c["run_stage"].update(scope="draft"),lambda c:c["run_stage"].update(allowed=False),lambda c:c["run_stage"].update(move="other")):
            caps=copy.deepcopy(self.caps);mutate(caps);self.assertTrue(self.issues(caps=caps))
    def test_missing_approval_unknown_outcome_and_stale_input_are_refused(self):
        state=copy.deepcopy(self.state);state["approval"]["approved"]=False;self.assertTrue(any("approval" in p for p in self.issues(state=state)))
        state=copy.deepcopy(self.state);state["tasks"]["data"]={"status":"unknown"};self.assertTrue(self.issues(state=state))
        state=copy.deepcopy(self.state);state["artifacts"]["project"]["fingerprint"]="f"*64
        self.assertTrue(any("input" in p for p in self.issues(state=state)))
    def test_graph_revision_task_binding_and_plan_hash_are_checked(self):
        for updates in ({"graph_revision":2},{"task_id":"models"},{"fingerprint":"f"*64}):
            plan=self.plan.model_copy(update=updates);self.assertTrue(self.issues(plan))
    def test_hidden_final_wrong_stage_and_duplicate_non_idempotent_action_are_refused(self):
        from dclab_core.task_fingerprints import metadata_fingerprint
        data=self.plan.model_dump(mode="json");data["actions"].append(copy.deepcopy(data["actions"][0]));data["actions"][1]["id"]="second";data["budget"]["max_tool_calls"]=2
        data["fingerprint"]=plan_fingerprint(data)
        self.assertTrue(any("effect" in p for p in self.issues(TaskExecutionPlan.read(data))))
        state=copy.deepcopy(self.state);state["artifacts"]["data_stage"]={"fingerprint":metadata_fingerprint("final"),"value_type":"string","value":"final"}
        plan=execution_plan(self.graph,state)
        self.assertTrue(any("final" in p for p in self.issues(plan,state=state)))
    def test_final_requires_current_unused_human_permission(self):
        state=self.state
        for ident in ("data","leakage","features","models","holdout"): state=verified(self.graph,state,ident)
        plan=execution_plan(self.graph,state,"final")
        self.assertEqual(self.issues(plan,state=state),[])
        for field,value in (("holdout_uses",1),("holdout_approval",None)):
            bad=copy.deepcopy(state);bad[field]=value
            self.assertTrue(any("holdout" in p for p in self.issues(plan,state=bad)))
    def test_weakened_checks_and_missing_outputs_are_refused(self):
        for field,value in (("declared_outputs",[]),("acceptance_checks",[dict(id="other",kind="artifact",description="Trust agent")])):
            data=self.plan.model_dump(mode="json");data[field]=value;data["fingerprint"]=plan_fingerprint(data)
            self.assertTrue(self.issues(TaskExecutionPlan.read(data)))


class AdditionalSafetyTests(unittest.TestCase):
    def test_failed_task_is_not_automatically_ready_again(self):
        graph,wf,caps,state=case();state["tasks"]["data"]={"status":"failed"}
        self.assertEqual(ready_tasks(graph,state,source_workflow=wf),[])

    def test_approval_is_bound_to_project_and_current_input_metadata(self):
        graph,wf,caps,state=case();plan=execution_plan(graph,state)
        for mutate in (lambda s:s["approval"].update(project_id="another"),lambda s:s["approval"]["inputs"].update(project="f"*64)):
            bad=copy.deepcopy(state);mutate(bad)
            self.assertTrue(validate_execution_plan(plan,graph,caps,source_workflow=wf,observed_state=bad))

    def test_graph_rejects_duplicate_non_idempotent_export_and_unclassified_stage_effects(self):
        graph,wf,caps,state=case()
        for ident in ("report","features"):
            data=graph.model_dump(mode="json");node=copy.deepcopy(next(n for n in data["nodes"] if n["id"]==ident))
            node.update(id="repeat",source_step_ids=[],produces=["other_record"],needs=[],dependencies=[])
            data["nodes"].append(node);data["edges"].append(dict(source="root",target="repeat",kind="grouping"))
            bad=reseal(graph,nodes=data["nodes"],edges=data["edges"])
            self.assertTrue(any("effect" in p for p in validate_graph(bad,wf,known_capabilities=caps)))

    def test_invalid_snapshot_returns_diagnostics_instead_of_crashing(self):
        graph,wf,caps,state=case();plan=execution_plan(graph,state)
        self.assertTrue(validate_execution_plan(plan,graph,None,source_workflow=wf,observed_state=state))
        for artifacts in ([],None,"invalid"):
            bad=copy.deepcopy(state);bad["artifacts"]=artifacts
            with self.assertRaises(ValueError): ready_tasks(graph,bad,source_workflow=wf)

    def test_missing_or_changed_gate_inputs_cannot_be_verified_by_a_reference_alone(self):
        graph,wf,caps,state=case()
        for ident in ("data","leakage","features","models","holdout"): state=verified(graph,state,ident)
        state["holdout_approval"]["inputs"]["table"]="f"*64
        with self.assertRaisesRegex(ValueError,"approval"): ready_tasks(graph,state,source_workflow=wf)

    def test_changed_approved_workflow_or_source_map_cannot_keep_old_permission(self):
        graph,wf,caps,state=case();plan=execution_plan(graph,state)
        for field,value in (("workflow_fingerprint", "f"*64), ("sources", [])):
            bad=copy.deepcopy(state);bad["approval"][field]=value
            self.assertTrue(validate_execution_plan(plan,graph,caps,source_workflow=wf,observed_state=bad))

    def test_missing_gate_model_binding_and_false_consumption_are_rejected(self):
        graph,wf,caps,state=case()
        for ident in ("data","leakage","features","models","holdout"): state=verified(graph,state,ident)
        bad=copy.deepcopy(state);del bad["holdout_approval"]["inputs"]["models_record"]
        with self.assertRaises(ValueError): ready_tasks(graph,bad,source_workflow=wf)
        bad=copy.deepcopy(state);bad["holdout_uses"]=False
        self.assertEqual(ready_tasks(graph,bad,source_workflow=wf),[])

    def test_both_branches_are_ready_in_lexical_order_but_only_one_is_returned(self):
        graph,wf,caps,state=case()
        # Add an independent supported metadata read with measured output requirements.
        cap={"name":"get_record","scope":"project","effect":"read","move":None,"allowed":True,"replay_safe":True,
             "parameters":{"type":"object","properties":{"project_id":{"type":"string"}},"required":["project_id"]}}
        caps["get_record"]=cap;state["capabilities"]=caps
        data=graph.model_dump(mode="json");data["nodes"].append(dict(id="aaa",parent_id="root",kind="action",title="Read evidence",objective="Read existing metadata",scope="project",
             capability=dict(name="get_record",scope="project",schema_fingerprint=capability_fingerprint(cap)),
             acceptance_checks=[dict(id="read",kind="artifact",description="Existing record reference verified")]))
        data["edges"].append(dict(source="root",target="aaa",kind="grouping"));graph=reseal(graph,nodes=data["nodes"],edges=data["edges"]);state=state_for(graph,state)
        self.assertEqual(ready_tasks(graph,state,source_workflow=wf),["aaa"])


class ProgrammeBoundaryTests(unittest.TestCase):
    def test_one_atomic_programme_with_separate_human_gate_passes(self):
        from tests.task_validation_cases import programme_case
        graph,source,caps,state=programme_case()
        self.assertEqual(validate_graph(graph,source_solution_map=source,known_capabilities=caps),[])
        self.assertEqual(ready_tasks(graph,state,source_solution_map=source),[])
        state=verified(graph,state,"holdout")
        self.assertEqual(ready_tasks(graph,state,source_solution_map=source),["programme"])
        plan=execution_plan(graph,state,"programme")
        self.assertEqual(validate_execution_plan(plan,graph,caps,source_solution_map=source,observed_state=state),[])

    def test_complex_map_cannot_gain_flat_leaf_final_or_lose_its_gate(self):
        from tests.task_validation_cases import programme_case
        graph,source,caps,state=programme_case();data=graph.model_dump(mode="json")
        data["nodes"][-1]["source_step_ids"]=["final_once"]
        bad=reseal(graph,nodes=data["nodes"])
        self.assertTrue(any("ambiguous" in p for p in validate_graph(bad,source_solution_map=source,known_capabilities=caps)))
        data=graph.model_dump(mode="json");data["nodes"][-1]["dependencies"]=[]
        data["edges"]=[e for e in data["edges"] if e["kind"]=="grouping"]
        self.assertTrue(any("human gate" in p for p in validate_graph(reseal(graph,nodes=data["nodes"],edges=data["edges"]),source_solution_map=source,known_capabilities=caps)))


class ReviewRegressionTests(unittest.TestCase):
    def test_argument_named_description_remains_part_of_capability_identity(self):
        graph,wf,caps,state=case();cap=copy.deepcopy(caps["run_stage"])
        cap["parameters"]["properties"]["description"]={"type":"string"};cap["parameters"]["required"].append("description")
        first=capability_fingerprint(cap);cap["parameters"]["properties"]["description"]={"type":"integer"}
        self.assertNotEqual(first,capability_fingerprint(cap))

    def test_training_analysis_cannot_be_independent_of_lock_or_after_audit(self):
        graph,wf,caps,state=case();cap={"name":"describe_data","scope":"project","effect":"read","move":None,"allowed":True,"replay_safe":True,
            "parameters":{"type":"object","properties":{"project_id":{"type":"string"}},"required":["project_id"]}}
        caps["describe_data"]=cap;state["capabilities"]=caps
        data=graph.model_dump(mode="json");data["nodes"][1]["source_step_ids"].remove("train_only_analysis")
        data["nodes"].append(dict(id="aaa",parent_id="root",kind="action",title="Training analysis",objective="Inspect training evidence",source_step_ids=["train_only_analysis"],scope="project",
            capability=dict(name="describe_data",scope="project",schema_fingerprint=capability_fingerprint(cap)),
            acceptance_checks=[dict(id="training_proof",kind="lineage",description="Training data only")]))
        data["edges"].append(dict(source="root",target="aaa",kind="grouping"))
        bad=reseal(graph,nodes=data["nodes"],edges=data["edges"])
        self.assertTrue(any("train_only_analysis" in p for p in validate_graph(bad,wf,known_capabilities=caps)))


class EvidenceShapeTests(unittest.TestCase):
    def test_existing_outputs_without_verified_receipt_cannot_be_scheduled_again(self):
        graph,wf,caps,state=case();state=verified(graph,state,"data")
        state["tasks"].pop("data")
        with self.assertRaisesRegex(ValueError,"unverified outputs"):
            ready_tasks(graph,state,source_workflow=wf)

    def test_boolean_flags_are_not_verification_or_approval_references(self):
        graph,wf,caps,state=case();done=verified(graph,state,"data")
        done["tasks"]["data"]["verification_ref"]=True
        with self.assertRaises(ValueError): ready_tasks(graph,done,source_workflow=wf)
        for ident in ("data","leakage","features","models","holdout"): state=verified(graph,state,ident)
        state["holdout_approval"]["fingerprint"]=True
        state["tasks"]["holdout"]["approval_ref"]["fingerprint"]=True
        with self.assertRaises(ValueError): ready_tasks(graph,state,source_workflow=wf)

    def test_artifact_hashes_are_real_digests_not_equal_truthy_flags(self):
        graph,wf,caps,state=case();state=verified(graph,state,"data")
        state["artifacts"]["data_record"]["fingerprint"]=True
        state["tasks"]["data"]["outputs"]["data_record"]=True
        with self.assertRaises(ValueError): ready_tasks(graph,state,source_workflow=wf)

    def test_host_revision_boolean_is_not_revision_one(self):
        graph,wf,caps,state=case();state["graph_revision"]=True
        with self.assertRaises(ValueError): ready_tasks(graph,state,source_workflow=wf)


class PurityAndIdentityTests(unittest.TestCase):
    def test_runtime_callbacks_are_not_accepted_as_a_metadata_snapshot(self):
        graph,wf,caps,state=case();plan=execution_plan(graph,state)
        def forbidden(): raise AssertionError("validator invoked a runtime handler")
        caps["run_stage"]["handler"]=forbidden
        self.assertTrue(validate_execution_plan(plan,graph,caps,source_workflow=wf,observed_state=state))

    def test_action_order_is_semantic_even_when_dictionary_order_is_not(self):
        graph,wf,caps,state=case();plan=execution_plan(graph,state);data=plan.model_dump(mode="json")
        data["actions"].append({**copy.deepcopy(data["actions"][0]),"id":"second"});data["budget"]["max_tool_calls"]=2
        first=plan_fingerprint(data);data["actions"].reverse()
        self.assertNotEqual(first,plan_fingerprint(data))

    def test_noop_and_blocked_plans_are_not_completion_or_execution(self):
        graph,wf,caps,state=case();data=execution_plan(graph,state).model_dump(mode="json")
        for status in ("noop","blocked"):
            data.update(status=status,reason="No executable operation is authorized",actions=[],declared_outputs=[],selected_capabilities=[])
            data["fingerprint"]=plan_fingerprint(data)
            self.assertEqual(validate_execution_plan(TaskExecutionPlan.read(data),graph,caps,source_workflow=wf),[])


class PostFinalSourceTests(unittest.TestCase):
    def test_every_wf10_source_step_waits_for_final_not_only_report_limits(self):
        from dclab_core.solution_workflow import solution_workflow
        graph,wf,caps,state=case();source=solution_workflow({"family":"uplift"})
        cap={"name":"get_results","scope":"project","effect":"read","move":None,"allowed":True,"replay_safe":True,
             "parameters":{"type":"object","properties":{"project_id":{"type":"string"}},"required":["project_id"]}}
        caps["get_results"]=cap
        data=graph.model_dump(mode="json");data["nodes"][1]["source_step_ids"]=["lock_holdout","train_only_analysis","arms_check"]
        next(n for n in data["nodes"] if n["id"]=="final")["source_step_ids"]=["final_once"]
        data["nodes"].append(dict(id="aaa",parent_id="root",kind="action",title="Read Qini",objective="Read final evidence",source_step_ids=["qini"],dependencies=["data"],needs=["data_record"],produces=["qini_record"],scope="project",
            capability=dict(name="get_results",scope="project",schema_fingerprint=capability_fingerprint(cap)),acceptance_checks=[dict(id="qini",kind="lineage",description="Final evidence exists")]))
        data["edges"] += [dict(source="root",target="aaa",kind="grouping"),dict(source="data",target="aaa",kind="dependency")]
        data["workflow_fingerprint"]=workflow_fingerprint(source)
        for ref in data["source_refs"]:
            if ref["kind"]=="solution_workflow": ref["fingerprint"]=data["workflow_fingerprint"]
        bad=reseal(graph,**{k:v for k,v in data.items() if k!="fingerprint"})
        self.assertTrue(any("qini" in p and "final_once" in p for p in validate_graph(bad,source,known_capabilities=caps)))
        # The same final stage may produce this WF-10 evidence without a duplicate operation.
        data["nodes"]=[n for n in data["nodes"] if n["id"]!="aaa"]
        next(n for n in data["nodes"] if n["id"]=="final")["source_step_ids"].append("qini")
        data["edges"]=[e for e in data["edges"] if e["target"]!="aaa"]
        good=reseal(graph,**{k:v for k,v in data.items() if k!="fingerprint"})
        self.assertEqual(validate_graph(good,source,known_capabilities=caps),[])

    def test_scientific_capability_must_retain_its_guarded_operation(self):
        graph,wf,caps,state=case()
        for field,value in (("move","capture"),("effect","read")):
            changed=copy.deepcopy(caps);changed["run_stage"][field]=value
            nodes=graph.model_dump(mode="json")["nodes"]
            for node in nodes:
                if node.get("capability") and node["capability"]["name"]=="run_stage": node["capability"]["schema_fingerprint"]=capability_fingerprint(changed["run_stage"])
            rebuilt=reseal(graph,nodes=nodes)
            self.assertTrue(validate_graph(rebuilt,wf,known_capabilities=changed))
