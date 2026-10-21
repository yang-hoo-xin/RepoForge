"""Read-only native-tool delivery admission, not a real compiler or model run."""
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.mod_agent.artifacts import ArtifactRef, ArtifactStore
from src.mod_agent.evaluation.isolated_delivery_adapter import _task_agent_verifier_binding
from src.mod_agent.proposal.builder import PatchProposalBuilder
from tests.unit import test_fresh_planning_continuation as source_fixture
from tests.unit import test_task_agent_execution as task_fixture


@pytest.fixture
def delivery(short_workspace, monkeypatch, request):
    # Scope: production-like run/a layout with a real Store and scripted tool responses.
    # 为什么要有它：不能为了复用测试夹具放宽交付原件必须位于本次run内的限制。
    monkeypatch.setattr(source_fixture, "ArtifactStore", lambda _path: ArtifactStore(short_workspace / "workspace/a"))
    def inputs_with_run_local_store(path):
        inputs = source_fixture._inputs(path)
        return replace(inputs, request=replace(inputs.request, artifact_store_root=str(inputs.store.root)))
    monkeypatch.setattr(task_fixture, "_inputs", inputs_with_run_local_store)
    def plan_inside_product_layout(inputs):
        from src.mod_agent.workflow.graph import _run_create_implementation_plan
        state = _run_create_implementation_plan(source_fixture._direct_graph_state(inputs), inputs.store,
            formal_requirement_coverage_required=True,
            planning_target_selection_factory=task_fixture._selector(inputs, task_fixture._Client()))
        assert not state["blockers"], state
        return SimpleNamespace(implementation_plan_ref=ArtifactRef(feature_id=inputs.feature_ref.feature_id,
            artifact_type="implementation_plan", path=state["artifacts"]["implementation_plan"],
            artifact_hash=state["artifact_hashes"]["implementation_plan"]))
    monkeypatch.setattr(task_fixture, "_patch_request", plan_inside_product_layout)
    monkeypatch.setattr("socket.socket.connect", lambda *_: pytest.fail("No network"))
    case = task_fixture.case.__wrapped__(short_workspace)
    candidate_mode = getattr(request, "param", "baseline")
    if candidate_mode == "current":
        owner, outcome, _ = task_fixture.execute_current_revision(case)
    else:
        owner, outcome = task_fixture.execute(case)
    assert outcome.status == "candidate_ready", outcome.termination
    handoff = owner.validate_handoff(outcome, case.context)
    root = case.inputs.workspace
    proposal = PatchProposalBuilder().build({**case.plan, "proposed_changes": list(handoff.changes)},
        evidence_pack=case.evidence, artifact_hashes=case.state["artifact_hashes"], require_git_binding=False,
        source_snippet_char_limit=65536).to_artifact_payload()
    proposal["context_pack"]["task_agent_lineage"] = {"execution_ref": outcome.execution_ref,
        "candidate_ref": outcome.candidate_ref, "check_ref": outcome.check_ref,
        "approval_inherited": False, "whole_task_delivered": False}
    declaration = {"technical_executor": "task_agent", "constraint_policy": "source_priority_v1",
                   "allowed_edit_paths_relative_to_project": ["Behavior.cs"]}
    if candidate_mode == "current":
        declaration["candidate_mode"] = candidate_mode
    argv = ["run-request", "--target-project", str(root / "generated_mods/SyntheticMod"),
        "--workspace-root", str(root), "--settings", "config/settings.yaml", "--local-ollama",
        "--technical-executor", "task_agent", "--constraint-policy", "source_priority_v1",
        "--task-agent-edit-paths", '["Behavior.cs"]', "--task-agent-candidate-mode", candidate_mode,
        "--defer-approval", "--", "controlled test"]
    worker = json.loads(Path(owner.worker_configuration_ref["path"]).read_bytes())
    worker["policy"] = asdict(owner.policy)
    reference = handoff.check["report"]["report_ref"]
    values = dict(root=root, initial={"run_id": case.state["run_id"], "model_call_budget": outcome.budget_after,
        "task_agent_execution_ref": outcome.execution_ref, "task_agent_outcome": outcome.to_dict(),
        "patch_execution_receipts": []}, manifest={"request_id": case.state["request_id"],
        "technical_execution": declaration, "nuget_isolated_build_profile": {
            "working_directory": str(root / "generated_mods/SyntheticMod")}},
        snapshot={"business_argv": argv}, worker_ref=owner.worker_configuration_ref, worker=worker,
        verification=reference, report=json.loads(Path(reference["path"]).read_bytes()),
        proposal=proposal, execution_check_ref={"path": "test-only-execution-check", "sha256": "b" * 64})
    return case, values


@pytest.mark.parametrize("delivery", ["baseline", "current"], indirect=True)
def test_native_tool_delivery_binding_preserves_real_origin_without_old_worker_receipt(delivery):
    case, values = delivery
    before = case.inputs.behavior.read_bytes()
    result = _task_agent_verifier_binding(**values)
    assert result["source"] == "caller_bound_factory_and_native_task_tool_execution"
    assert "worker_receipt_refs" not in result
    assert case.inputs.behavior.read_bytes() == before


def test_omitted_legal_policy_default_has_same_execution_semantics(delivery):
    _, values = delivery
    assert values["worker"]["policy"].pop("diagnose_behavior_failures") is False
    assert _task_agent_verifier_binding(**values)["task_agent_execution_ref"]


@pytest.mark.parametrize("change", ["scope", "mode", "candidate_mode", "budget", "lineage", "proposal", "plan", "old_worker", "policy"])
def test_native_tool_delivery_rejects_wrong_scope_or_current_candidate(delivery, change):
    _, original = delivery
    values = deepcopy(original)
    if change == "scope":
        values["manifest"]["technical_execution"]["allowed_edit_paths_relative_to_project"] = ["Plugin.cs"]
    elif change == "mode":
        argv = values["snapshot"]["business_argv"]
        argv[argv.index("--technical-executor") + 1] = "workflow"
    elif change == "candidate_mode":
        argv = values["snapshot"]["business_argv"]
        argv[argv.index("--task-agent-candidate-mode") + 1] = "current"
    elif change == "budget":
        values["initial"]["model_call_budget"]["consumed"] = 0
    elif change == "lineage":
        values["proposal"]["context_pack"]["task_agent_lineage"]["approval_inherited"] = True
    elif change == "proposal":
        values["proposal"]["diff_intent"][0]["rationale"] = "a different candidate"
    elif change == "plan":
        values["proposal"]["context_pack"]["implementation_plan_hash"] = "a" * 64
    elif change == "old_worker":
        values["initial"]["patch_execution_receipts"] = [{"fake": "old worker"}]
    else:
        values["worker"]["policy"]["diagnose_behavior_failures"] = True
    with pytest.raises(ValueError):
        _task_agent_verifier_binding(**values)
