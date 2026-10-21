"""New-trial admission over controlled byte-bound history; no provider or Build."""
import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.libs.llm.agent_turn import AgentMessage, AgentToolResult, AgentTurnRequest
from src.mod_agent import comparison_checkpoint
from src.mod_agent.artifacts import ArtifactRef
from src.mod_agent.comparison_checkpoint import (
    ComparisonCheckpointError,
    admit_checkpoint_verifier,
    admit_comparison_checkpoint,
)
from src.mod_agent.proposal.failure_state import failure_signature, source_fact_hashes


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    return file_ref(path)


def file_ref(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def case(tmp_path, monkeypatch):
    old, workspace = tmp_path / "historical", tmp_path / "trial"
    files = ["SignalController.cs", "Plugin.cs", "Contracts.cs", "NetworkIntegration.cs",
             "SquadSignal.csproj", "NuGet.Config", "Dependencies.props"]
    snapshots, baseline = {}, []
    for name in files:
        content = "// retained " + name
        for root in (old, workspace):
            path = root / "generated_mods/SquadSignal" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        reference = file_ref(old / "generated_mods/SquadSignal" / name)
        snapshots[name] = reference["sha256"]
        baseline.append({"location": "executed_workspace", "relative_path": name, "ref": reference})
    factory = tmp_path / "factory.py"
    factory.write_text("raise AssertionError('Evaluator must never be imported during admission')\n", encoding="utf-8")
    probe_files = []
    for name in ("Probe.dll", "Probe.deps.json", "Probe.runtimeconfig.json", "dotnet.exe"):
        path = tmp_path / name
        path.write_text("controlled-not-executable", encoding="utf-8")
        probe_files.append(file_ref(path))
    probe = save(tmp_path / "probe.json", {"files": probe_files[:3], "dotnet": probe_files[3]})
    nuget = file_ref(old / "generated_mods/SquadSignal/NuGet.Config")
    policy = {"max_total_model_turns": 18, "max_tool_rounds": 6, "max_observation_bytes": 32768,
        "model_call_budget_policy": "task_total_only", "require_source_inspection": False,
        "finalize_after_first_observation": False, "allow_batched_reads": True,
        "allow_upstream_request": True, "enable_supporting_reads": True}
    store = SimpleNamespace(root=workspace / "a", stale=False, envelopes={})

    def worker(root):
        return {"schema_version": "patch_worker_configuration_v1", "policy": dict(policy), "enable_recovery_loop": True,
            "behavior_evaluator": {"factory_ref": file_ref(factory), "factory_name": "create", "kwargs": {
                "output_root": str(root / "v"), "probe_configuration_ref": probe,
                "compilation_kwargs": {"source_root": str(root / "generated_mods/SquadSignal"), "workspace_root": str(root),
                    "artifact_store_root": str(root / "a"), "output_root": str(root / "c"),
                    "nuget_config_ref": nuget, "project_relative_path": "SquadSignal.csproj", "timeout_seconds": 120,
                    "dependency_binding": {"source_root": str(tmp_path / "dependencies"), "file_count": 165,
                        "snapshot_sha256": "d" * 64}}}}}

    historical_worker = save(old / "worker.json", worker(old))
    current_worker = worker(workspace)
    current_worker["policy"]["max_total_model_turns"] = 15
    current_worker_ref = save(workspace / "worker.json", current_worker)
    profile = {"model": "qwen3-coder:30b", "context_window": 65536, "output_budget": 4096,
        "timeout_seconds": 180.0, "base_url": "http://127.0.0.1:11434"}
    identity = {"digest": "f" * 64, "model": profile["model"]}
    selected, artifacts, refs = {}, {}, {}
    for index, kind in enumerate(("feature_spec", "evidence_pack", "implementation_plan")):
        digest = str(index + 1) * 64
        path = old / "a/features/feature_controlled/r" / kind / (digest + ".json")
        reference = ArtifactRef(feature_id="feature_controlled", artifact_type=kind, artifact_hash=digest, path=str(path))
        parents = [] if index == 0 else ["feature_spec"] if index == 1 else ["feature_spec", "evidence_pack"]
        envelope = {"artifact_hash": digest, "data": {"control": kind}, "status": "success",
            "schema_version": "test", "feature_id": "feature_controlled", "artifact_type": kind,
            "inputs": {"upstream_artifacts": [{key: selected[parent][key] for key in ("artifact_type", "artifact_hash", "path")}
                for parent in parents]}}
        byte_ref = save(path, envelope)
        refs[kind] = reference
        selected[kind] = asdict(reference)
        artifacts[kind] = {"artifact_ref": asdict(reference), "file_ref": byte_ref}
        store.envelopes[digest] = envelope
    source_store = SimpleNamespace(root=old / "a")
    source_store.semantic_read_immutable = lambda ref: json.loads(Path(ref.path).read_bytes())
    monkeypatch.setattr(comparison_checkpoint, "ArtifactStore", lambda root:
        store if Path(root).resolve() == store.root.resolve() else source_store)
    mapping_rows = []
    for index, kind in enumerate(("feature_spec", "evidence_pack", "implementation_plan")):
        original = refs[kind]
        digest = str(index + 4) * 64
        target_path = workspace / "a/features/feature_controlled/r" / kind / (digest + ".json")
        target = ArtifactRef(feature_id="feature_controlled", artifact_type=kind, artifact_hash=digest, path=str(target_path))
        document = deepcopy(store.envelopes[original.artifact_hash])
        document["artifact_hash"] = digest
        for upstream in document["inputs"]["upstream_artifacts"]:
            replacement = refs[upstream["artifact_type"]]
            upstream.update(path=replacement.path, artifact_hash=replacement.artifact_hash)
        save(target_path, document)
        store.envelopes[digest] = document
        mapping_rows.append({"source_ref": asdict(original), "derived_ref": asdict(target)})
        refs[kind] = target
    store.semantic_read_immutable = lambda ref: deepcopy(store.envelopes[ref.artifact_hash])
    store.check_freshness = lambda _ref: SimpleNamespace(is_fresh=not store.stale)
    outputs = [{"binding_id": "edit_binding_001" if index == 0 else "support_" + str(index),
        "source_hash": snapshots[name], "source_snapshot_sha256": "a" * 64,
        "source_ref": "binding://" + name, "truncated": False, "excerpt": "// retained " + name}
        for index, name in enumerate(files[:4])]
    tools = tuple(AgentToolResult("call-" + str(index), "inspect_plan_binding_source", output)
                  for index, output in enumerate(outputs))
    provider = {"message": {"role": "assistant", "tool_calls": ["four native calls"]}}
    response1 = {"provider_state": provider, "tool_calls": [{"call_id": item.call_id} for item in tools]}
    code = {"baseline_sha256": "sha256:" + "a" * 64, "code_sha256": "sha256:" + "b" * 64,
            "files": ["generated_mods/SquadSignal/SignalController.cs"]}
    after = "// candidate"
    response2 = {"content": json.dumps({"changes": [{"edit_binding_id": "edit_binding_001", "after_text": after}]})}
    failure = {"status": "behavior_failed", "failure_owner": "patch_implementation", "owner_binding_ids": ["edit_binding_001"],
        "observations": [{"check_id": "check" + str(i), "observed": {"count": 3}} for i in range(17)],
        "generated_changes_sha256": "c" * 64, "source_snapshot_sha256": "a" * 64,
        "materialized_code_identity": code, "verification_identity": {"controlled": "same"}}
    failure_ref = save(old / "first-failure.json", failure)
    admitted = {**failure, "report_ref": failure_ref, "admission": "accepted", "execution_origin": "executed"}
    checks = {"checks": [{"passed": i < 9} for i in range(26)], "materialized_code_identity": code}
    checks_ref = save(old / "checks.json", checks)
    state = {"message_type": "patch_agent_loop_state", "budget_state": {"model_turns_used": 2,
        "remaining_model_turns": 13, "tool_rounds_used": 4, "observation_bytes_used": 16214}}
    request = AgentTurnRequest(messages=(AgentMessage("system", "controlled contract"), AgentMessage("user", "controlled task"),
        AgentMessage("assistant", response2["content"]), AgentMessage("user", {"message_type": "patch_behavior_correction",
            "validator_feedback": {"report_sha256": failure_ref["sha256"], "observations": failure["observations"]}}),
        AgentMessage("user", state)), tool_results=tools, provider_state=provider, tool_result_insert_index=2)
    pending = asdict(request)
    branch = deepcopy(pending)
    branch["messages"] = branch["messages"][:-1]
    pending_ref = save(old / "request-003.json", pending)
    response1_ref, response2_ref = save(old / "response-001.json", response1), save(old / "response-002.json", response2)
    shared = {"maximum": 18, "consumed": 5, "remaining": 13, "by_stage": {name: {"consumed": n}
        for name, n in {"initial_patch": 2, "planning": 1, "requirement_findings": 1, "requirement_projection": 1, "repair": 0}.items()}}
    event = {"event_index": 20, "phase": "local_patch_post_response_model_admitted", "model_call_count": 2,
        "shared_model_budget": shared, "factory_receipt": {"profile": profile, "model_identity": identity,
        "worker_turns": [{"turn_index": 1, "response_ref": response1_ref}, {"turn_index": 2, "response_ref": response2_ref}],
        "patch_agent_loop": {"policy": {"max_total_model_turns": 15}, "tool_rounds_used": 4,
            "total_observation_bytes": 6182, "behavior_feedback_budget": {"combined_utf8_bytes": 16214, "feedback_utf8_bytes": 10032},
            "observation_trace": [{**row, "admitted_to_model": True} for row in outputs], "behavior_evaluations": [admitted]}}}
    journal = {"run_id": "historical-source-run", "request_id": "source-request", "worker_configuration_ref": historical_worker,
        "selected_artifacts": selected, "events": [event, {"event_index": 21, "phase": "local_patch_worker_turn_requested",
            "model_call_count": 3, "shared_model_budget": {"consumed": 6}}],
        "execution_observations": {"a333697e623343de": {"started_at": "2026-10-14T11:43:23.923222+00:00"}}}
    journal_ref = save(old / "receipt.json", journal)
    prep_ref = save(old / "preparation.json", {"source_snapshot": snapshots, "wall_seconds": 600, "product_calls": 18})
    process_ref = save(old / "process.json", {"started_at": "2026-10-14T11:37:39.133439+00:00"})
    scope_ref = save(old / "c5-scope-regression/result.json", {"schema": "c5-scope-regression-retention.v1",
        "preparation_ref": prep_ref, "process_ref": process_ref, "receipt_ref": journal_ref,
        "behavior_evaluations": [{"evaluation_index": 1, "report_ref": failure_ref, "checks_ref": checks_ref}],
        "feedback_consumption": [{"revision_index": 1, "patch_turn_index": 3, "request_ref": pending_ref}],
        "historical_usage": {"routing_transport_calls": 1}})
    fixture = {"status": "prefix_extracted_runner_adapter_required", "model_input_boundary": "Only branch-request is model-facing",
        "source_refs": [scope_ref], "checkpoint_origin": {"receipt_ref": journal_ref, "event_index": 20,
            "next_event_index": 21, "future_model_responses_or_failures_in_model_input": False},
        "files": {"pending-request.json": pending_ref, "branch-request.json": save(old / "branch.json", branch),
            "prefix-event-20.json": save(old / "event20.json", event), "first-failure.json": failure_ref,
            "first-checks.json": checks_ref, "candidate-response.json": response2_ref},
        "budget": {"pending_state": state["budget_state"], "routing_calls_outside_product_limit": 1},
        "candidate": {"after_text_sha256": hashlib.sha256(after.encode()).hexdigest(), "identity": code},
        "baseline_refs": baseline, "artifacts": artifacts,
        "model": {"profile": profile, "identity": identity,
            "admission_ref": save(old / "model-admission.json", {"actual_model_identity": identity})},
        "wall": {"original_process_started_at": "2026-10-14T11:37:39.133439+00:00",
            "original_limit_seconds": 600, "admitted_branch_seconds": 255}}
    fixture_ref = save(old / "fixture.json", fixture)
    mapping = {"schema_version": "comparison-derived-inputs.v1", "strategy": "identity_v1",
        "fixture_ref": fixture_ref, "artifacts": mapping_rows}
    kwargs = {"trial_id": "comparison-A-1", "workspace_root": workspace, "artifact_store": store,
        **{name + "_ref": ref for name, ref in refs.items()}, "worker_configuration_ref": current_worker_ref, "model_profile": profile,
        "derived_inputs_ref": save(workspace / "derived.json", mapping)}
    return SimpleNamespace(fixture=fixture, ref=fixture_ref, kwargs=kwargs, worker=current_worker,
        root=tmp_path, workspace=workspace, store=store, mapping=mapping)


def test_bound_native_history_budgets_and_failure_seed_have_no_execution_or_reservation(case):
    before = {str(path): path.read_bytes() for path in case.root.rglob("*") if path.is_file()}
    admitted = admit_comparison_checkpoint(case.ref, **case.kwargs)
    assert admitted.product_consumed == 5 and admitted.product_remaining == 13 and admitted.routing_calls == 1
    assert (admitted.consumed_model_turns, admitted.consumed_tool_rounds, admitted.consumed_observation_bytes) == (2, 4, 16214)
    assert admitted.wall_seconds == 255 and admitted.worker_policy.max_total_model_turns == 15
    assert len(admitted.request.tool_results) == 4 and all(isinstance(row.output, dict) for row in admitted.request.tool_results)
    assert len(admitted.source_facts) == 4 and list(admitted.no_progress_seed.values()) == [1]
    assert admitted.provenance["reservation_performed"] is False
    assert admit_comparison_checkpoint(case.ref, **case.kwargs).trial_id == admitted.trial_id
    assert before == {str(path): path.read_bytes() for path in case.root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("changed", ["original_trial", "model", "baseline", "stale", "plan", "policy", "evaluator", "pending", "routing", "wall"])
def test_wrong_identity_or_reset_cannot_be_admitted(case, changed):
    if changed == "original_trial":
        case.kwargs["trial_id"] = "historical-source-run"
    elif changed == "model":
        case.kwargs["model_profile"] = {**case.kwargs["model_profile"], "context_window": 32768}
    elif changed == "baseline":
        (case.workspace / "generated_mods/SquadSignal/SignalController.cs").write_text("drift", encoding="utf-8")
    elif changed == "stale":
        case.store.stale = True
    elif changed == "plan":
        case.kwargs["implementation_plan_ref"] = ArtifactRef(feature_id="feature_controlled", artifact_type="implementation_plan",
            artifact_hash="9" * 64, path=case.kwargs["implementation_plan_ref"].path)
    elif changed in {"policy", "evaluator"}:
        if changed == "policy":
            case.worker["policy"]["max_total_model_turns"] = 19
        else:
            case.worker["behavior_evaluator"]["factory_name"] = "other"
        case.kwargs["worker_configuration_ref"] = save(Path(case.kwargs["worker_configuration_ref"]["path"]), case.worker)
    else:
        if changed == "pending":
            branch = json.loads(Path(case.fixture["files"]["branch-request.json"]["path"]).read_bytes())
            branch["messages"][1]["content"] = "new unauthorized instruction"
            case.fixture["files"]["branch-request.json"] = save(case.root / "changed-branch.json", branch)
        elif changed == "routing":
            case.fixture["budget"]["routing_calls_outside_product_limit"] = 0
        else:
            case.fixture["wall"]["admitted_branch_seconds"] = 600
        case.ref = save(case.root / "changed-fixture.json", case.fixture)
        case.mapping["fixture_ref"] = case.ref
        case.kwargs["derived_inputs_ref"] = save(case.workspace / "changed-derived.json", case.mapping)
    with pytest.raises(ComparisonCheckpointError):
        admit_comparison_checkpoint(case.ref, **case.kwargs)


def test_signature_matches_original_projection_and_does_not_own_retry_count():
    observations = [{"binding_id": "b", "source_hash": "s", "source_ref": "binding://b", "admitted_to_model": True},
                    {"binding_id": "not-visible", "source_hash": "other"}]
    facts = source_fact_hashes(observations)
    report = {"materialized_code_identity": {"code_sha256": "c"}, "status": "behavior_failed", "failure_owner": "patch_implementation",
        "owner_binding_ids": ["b"], "observations": [{"observed": 2, "result_ref": {"path": "diagnostic"}}]}
    from src.mod_agent.model_qualification import canonical_hash
    expected = canonical_hash({"code": report["materialized_code_identity"], "status": report["status"],
        "failure_owner": report["failure_owner"], "owner_binding_ids": ["b"], "observations": [{"observed": 2}],
        "verification_identity": None, "source_facts": facts})
    assert failure_signature(report, facts + facts) == expected and len(facts) == 1


@pytest.mark.parametrize("change", ["data", "upstream", "strategy", "duplicate", "historical_store"])
def test_mapping_declaration_cannot_override_exact_data_or_upstream_edges(case, change):
    if change == "historical_store":
        case.store.root = Path(case.fixture["artifacts"]["implementation_plan"]["artifact_ref"]["path"]).parents[4]
    elif change == "data":
        case.store.envelopes[case.kwargs["implementation_plan_ref"].artifact_hash]["data"]["extra"] = "unauthorized"
    elif change == "upstream":
        case.store.envelopes[case.kwargs["implementation_plan_ref"].artifact_hash]["inputs"]["upstream_artifacts"][0]["path"] = "other"
    elif change == "strategy":
        case.mapping["strategy"] = "trust_me"
    else:
        case.mapping["artifacts"].append(deepcopy(case.mapping["artifacts"][0]))
    case.kwargs["derived_inputs_ref"] = save(case.workspace / "changed-derived.json", case.mapping)
    with pytest.raises(ComparisonCheckpointError):
        admit_comparison_checkpoint(case.ref, **case.kwargs)


def relocated_sources(case):
    reference = case.fixture["artifacts"]["evidence_pack"]
    document = json.loads(Path(reference["file_ref"]["path"]).read_bytes())
    old_root = case.root / "historical"
    rows = []
    for item in case.fixture["baseline_refs"][:4]:
        relative = "generated_mods/SquadSignal/" + item["relative_path"]
        digest = item["ref"]["sha256"]
        rows.append({"source_type": "target_source", "source_path": relative,
            "source_ref": item["ref"]["path"], "source_hash": digest, "file_hash": digest, "doc_hash": digest,
            "evidence_id": "evidence:" + relative, "span": {"start_line": 1, "end_line": 1},
            "excerpt": "// retained " + item["relative_path"]})
    document["data"] = {"items": rows, "original_text": rows[0]["source_ref"],
        "historical_trace": {"path": rows[0]["source_ref"]}}
    reference["file_ref"] = save(Path(reference["file_ref"]["path"]), document)
    new_data, changes = comparison_checkpoint.relocate_checkpoint_source_items(document["data"],
        fixture=case.fixture, historical_workspace=old_root, workspace=case.workspace)
    case.store.envelopes[case.kwargs["evidence_pack_ref"].artifact_hash]["data"] = new_data
    case.ref = save(case.root / "relocated-fixture.json", case.fixture)
    case.mapping.update(fixture_ref=case.ref, source_relocation_policy="target_source_baseline_v1")
    case.kwargs["derived_inputs_ref"] = save(case.workspace / "relocated-derived.json", case.mapping)
    return document["data"], new_data, changes


def test_source_address_relocation_passes_original_gate_without_changing_history(case):
    from src.mod_agent.fresh_planning_continuation import (
        PlanningContinuationError,
        _validate_source_items,
    )

    original, relocated, changes = relocated_sources(case)
    with pytest.raises(PlanningContinuationError, match="SOURCE_REF_MISMATCH"):
        _validate_source_items(original, case.workspace)
    assert _validate_source_items(relocated, case.workspace)
    admitted = admit_comparison_checkpoint(case.ref, **case.kwargs)
    assert admitted.provenance["derived_inputs"]["source_address_changes"] == changes
    assert [row["json_pointer"] for row in changes] == [f"/data/items/{i}/source_ref" for i in range(4)]
    restored = deepcopy(relocated)
    for index, row in enumerate(original["items"]):
        restored["items"][index]["source_ref"] = row["source_ref"]
    assert restored == original
    assert admitted.request.messages[1].content == "controlled task"
    assert admitted.known_failure["report_ref"] == case.fixture["files"]["first-failure.json"]
    assert list(admitted.no_progress_seed.values()) == [1]


@pytest.mark.parametrize("changed", ["no_policy", "unknown_policy", "new_ref", "old_ref", "source_path",
    "source_hash", "file_hash", "doc_hash", "source_type", "excerpt", "trace", "old_bytes", "new_bytes"])
def test_source_relocation_cannot_rewrite_content_or_authorize_other_files(case, changed):
    relocated_sources(case)
    data = case.store.envelopes[case.kwargs["evidence_pack_ref"].artifact_hash]["data"]
    if changed in {"no_policy", "unknown_policy"}:
        if changed == "no_policy":
            case.mapping.pop("source_relocation_policy")
        else:
            case.mapping["source_relocation_policy"] = "trust_all_paths"
        case.kwargs["derived_inputs_ref"] = save(case.workspace / "bad-policy.json", case.mapping)
    elif changed == "old_ref":
        data["items"][0]["source_ref"] = case.fixture["baseline_refs"][0]["ref"]["path"]
    elif changed == "new_ref":
        data["items"][0]["source_ref"] = str(case.workspace / "unrelated.cs")
    elif changed in {"source_path", "source_hash", "file_hash", "doc_hash", "source_type", "excerpt"}:
        data["items"][0][changed] = "unauthorized change"
    elif changed == "trace":
        data["historical_trace"]["path"] = data["items"][0]["source_ref"]
    else:
        root = case.root / "historical" if changed == "old_bytes" else case.workspace
        (root / "generated_mods/SquadSignal/SignalController.cs").write_text("changed", encoding="utf-8")
    with pytest.raises(ComparisonCheckpointError):
        admit_comparison_checkpoint(case.ref, **case.kwargs)


@pytest.mark.parametrize("changed", ["reference", "relative_path", "source_hash", "file_hash", "doc_hash", "manifest"])
def test_source_relocation_derivation_checks_original_address_and_all_hash_aliases(case, changed):
    original, _relocated, _changes = relocated_sources(case)
    if changed == "reference":
        original["items"][0]["source_ref"] = str(case.workspace / original["items"][0]["source_path"])
    elif changed == "relative_path":
        original["items"][0]["source_path"] = "generated_mods/SquadSignal/../SquadSignal/SignalController.cs"
    elif changed == "manifest":
        case.fixture["baseline_refs"][0]["ref"]["path"] = str(case.workspace / original["items"][0]["source_path"])
    else:
        original["items"][0][changed] = "0" * 64
    with pytest.raises(ComparisonCheckpointError):
        comparison_checkpoint.relocate_checkpoint_source_items(original, fixture=case.fixture,
            historical_workspace=case.root / "historical", workspace=case.workspace)


def source_priority(case):
    from tests.unit.test_fresh_planning_continuation import _inputs

    inputs = _inputs(case.root / "coverage")
    feature = deepcopy(inputs.feature_payload)
    evidence = inputs.store.read(inputs.evidence_ref)["data"]
    hint = next(row["text"] for row in evidence["requirement_coverage"]["requirements"] if row["role"] == "preservation")
    source_data = {"feature_spec": feature, "evidence_pack": evidence,
        "implementation_plan": {"control": "original plan", "side_effect_boundaries": [hint]}}
    context = comparison_checkpoint.build_constraint_context(feature_spec=feature, evidence_pack=evidence, plan_boundaries=[hint])
    for kind, data in source_data.items():
        old_ref = case.fixture["artifacts"][kind]["artifact_ref"]
        original = json.loads(Path(old_ref["path"]).read_bytes())
        original["data"] = data
        case.fixture["artifacts"][kind]["file_ref"] = save(Path(old_ref["path"]), original)
        current = case.store.envelopes[case.kwargs[kind + "_ref"].artifact_hash]
        current["data"] = deepcopy(data)
    current_plan = case.store.envelopes[case.kwargs["implementation_plan_ref"].artifact_hash]
    current_plan["data"].update(side_effect_boundaries=context["hard_boundaries"], constraint_context=context)
    case.ref = save(case.root / "source-priority-fixture.json", case.fixture)
    case.mapping.update(strategy="source_priority_v1", fixture_ref=case.ref)
    case.kwargs["derived_inputs_ref"] = save(case.workspace / "source-priority-derived.json", case.mapping)
    return context


def test_source_priority_recomputes_full_context_from_original_sources(case):
    context = source_priority(case)
    checked = admit_comparison_checkpoint(case.ref, **case.kwargs)
    assert checked.provenance["derived_inputs"]["constraint_context"] == context
    assert checked.provenance["derived_inputs"]["selected_strategy_recomputed"] is True
    assert checked.request.messages[1].content == "controlled task"  # No helper-injected repair instruction.
    assert len(checked.request.tool_results) == 4 and len(checked.source_facts) == 4


@pytest.mark.parametrize("change", ["hard_boundary", "hint", "extra_plan_field", "feature", "claim_identity"])
def test_source_priority_cannot_smuggle_additional_plan_or_feature_changes(case, change):
    source_priority(case)
    plan = case.store.envelopes[case.kwargs["implementation_plan_ref"].artifact_hash]["data"]
    if change == "hard_boundary":
        plan["side_effect_boundaries"] = []
    elif change == "hint":
        plan["constraint_context"]["model_preservation_hints"][0]["text"] = "invented"
    elif change == "extra_plan_field":
        plan["target_files"] = ["outside.cs"]
    elif change == "feature":
        case.store.envelopes[case.kwargs["feature_spec_ref"].artifact_hash]["data"]["user_request"] = "new scope"
    else:
        case.mapping["strategy"] = "identity_v1"
        case.kwargs["derived_inputs_ref"] = save(case.workspace / "wrong-policy-derived.json", case.mapping)
    with pytest.raises(ComparisonCheckpointError):
        admit_comparison_checkpoint(case.ref, **case.kwargs)


@pytest.fixture
def verifier_case(case, monkeypatch):
    original_platform, current_platform = case.root / "old-platform", case.root / "current-platform"
    repo = Path(comparison_checkpoint.__file__).resolve().parents[2]
    components = {"old": {}, "new": {}}
    for name, relative in comparison_checkpoint._VERIFIER_COMPONENTS.items():
        for label, root in (("old", original_platform), ("new", current_platform)):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / relative, destination)
            components[label][name] = file_ref(destination)
    monkeypatch.setattr(comparison_checkpoint, "__file__", str(current_platform / "src/mod_agent/comparison_checkpoint.py"))
    current_worker = case.worker
    selected = current_worker["behavior_evaluator"]
    probe_ref = selected["kwargs"]["probe_configuration_ref"]
    probe = json.loads(Path(probe_ref["path"]).read_bytes())
    checker = {"validator_sha256": "0" * 64, "inputs_sha256": "1" * 64, "environment_sha256": "2" * 64,
        "environment_identity_scope": "controlled", "deterministic_reuse_allowed": False}
    final = {"validator_sha256": "0" * 64, "inputs_sha256": "3" * 64, "environment_sha256": "4" * 64,
             "deterministic_reuse_allowed": False}
    old_layers = comparison_checkpoint._validator_layers(components["old"], checker, probe_ref,
        selected["factory_ref"], probe["files"], final)
    new_layers = comparison_checkpoint._validator_layers(components["new"], old_layers["checker"], probe_ref,
        selected["factory_ref"], probe["files"], old_layers["probe"])
    original_failure = json.loads(Path(case.fixture["files"]["first-failure.json"]["path"]).read_bytes())
    original_failure["verification_identity"] = old_layers["probe"]
    failure_ref = save(case.root / "identity-failure.json", original_failure)
    case.fixture["files"]["first-failure.json"] = failure_ref
    case.fixture["candidate"]["compilation_ref"] = save(case.root / "compilation.json", {"verification_identity": old_layers["checker"]})
    feedback_ref = save(case.root / "compilation-feedback.json", {"verification_identity": old_layers["compilation_feedback"]})
    journal = json.loads(Path(case.fixture["checkpoint_origin"]["receipt_ref"]["path"]).read_bytes())
    failed = journal["events"][0]["factory_receipt"]["patch_agent_loop"]["behavior_evaluations"][0]
    failed.update(verification_identity=old_layers["probe"], report_ref=failure_ref, compilation_feedback_ref=feedback_ref)
    case.fixture["files"]["prefix-event-20.json"] = save(case.root / "identity-event20.json", journal["events"][0])
    case.fixture["checkpoint_origin"]["receipt_ref"] = save(case.root / "identity-journal.json", journal)
    pending = json.loads(Path(case.fixture["files"]["pending-request.json"]["path"]).read_bytes())
    pending["messages"][-2]["content"]["validator_feedback"]["report_sha256"] = failure_ref["sha256"]
    case.fixture["files"]["pending-request.json"] = save(case.root / "identity-request.json", pending)
    branch = deepcopy(pending)
    branch["messages"].pop()
    case.fixture["files"]["branch-request.json"] = save(case.root / "identity-branch.json", branch)
    scope = json.loads(Path(case.fixture["source_refs"][0]["path"]).read_bytes())
    scope.update(receipt_ref=case.fixture["checkpoint_origin"]["receipt_ref"],
        actual_source_evidence={"snapshot_ref": save(original_platform / "execution_snapshot.json", {"controlled": True})})
    scope["behavior_evaluations"][0]["report_ref"] = failure_ref
    scope["feedback_consumption"][0]["request_ref"] = case.fixture["files"]["pending-request.json"]
    case.fixture["source_refs"] = [save(Path(case.fixture["source_refs"][0]["path"]), scope)]
    case.ref = save(case.root / "identity-fixture.json", case.fixture)
    case.mapping["fixture_ref"] = case.ref
    case.kwargs["derived_inputs_ref"] = save(case.root / "identity-derived.json", case.mapping)
    checked = admit_comparison_checkpoint(case.ref, **case.kwargs)

    class Evaluator:
        def __init__(self):
            self.identity = deepcopy(new_layers["probe"])
            self.reads = 0
            self.configuration = {"configuration_ref": case.kwargs["worker_configuration_ref"],
                "factory_ref": selected["factory_ref"], "factory_name": selected["factory_name"],
                "kwargs_sha256": comparison_checkpoint.canonical_hash(selected["kwargs"])}

        def configuration_receipt(self):
            return deepcopy(self.configuration)

        def verification_identity(self):
            self.reads += 1
            return deepcopy(self.identity)

        def __call__(self, _context):
            pytest.fail("Address equivalence must not run a verifier")

    return SimpleNamespace(case=case, checked=checked, evaluator=Evaluator(), components=components, current_platform=current_platform)


def test_address_equivalence_is_recomputed_without_rewriting_old_failure_or_authorizing_reuse(verifier_case):
    value = verifier_case
    before = deepcopy(asdict(value.checked))
    first = admit_checkpoint_verifier(value.checked, value.evaluator)
    second = admit_checkpoint_verifier(value.checked, value.evaluator)
    assert first == second and value.evaluator.reads == 2 and asdict(value.checked) == before
    assert first["verification_identity"] == value.evaluator.identity
    assert first["no_progress_seed"] != value.checked.no_progress_seed
    assert list(first["no_progress_seed"].values()) == [1]
    assert first["known_failure_projection"]["new_execution"] is False
    assert "execution_origin" not in first["known_failure_projection"]
    assert first["verification_identity"]["deterministic_reuse_allowed"] is False


@pytest.mark.parametrize("changed", ["bytes", "inputs", "environment", "flags", "validator", "extra", "configuration", "seed", "report"])
def test_address_equivalence_rejects_anything_beyond_the_eight_path_changes(verifier_case, changed):
    value = verifier_case
    if changed == "bytes":
        Path(value.components["new"]["checker"]["path"]).write_text("changed bytes", encoding="utf-8")
    elif changed in {"inputs", "environment", "validator"}:
        value.evaluator.identity[changed + "_sha256"] = "0" * 64
    elif changed == "flags":
        value.evaluator.identity["deterministic_reuse_allowed"] = True
    elif changed == "extra":
        value.evaluator.identity["unreviewed_capability"] = True
    elif changed == "configuration":
        value.evaluator.configuration["kwargs_sha256"] = "0" * 64
    elif changed == "seed":
        value.checked.no_progress_seed[next(iter(value.checked.no_progress_seed))] = 0
    else:
        Path(value.checked.known_failure["report_ref"]["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        admit_checkpoint_verifier(value.checked, value.evaluator)
